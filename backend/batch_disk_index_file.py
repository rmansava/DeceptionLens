"""
Batch DISK feature indexer - FILE-BASED storage version.
Pre-computes DISK keypoints and descriptors and saves as .npz files.

Storage: D:\disk-features\books\{BookName}\{filename}.npz
Auto-moves to NAS T:\disk-features\books after each book completes.
Moves are queued and run one at a time in background thread.

Path remapping: Reads from D:\books\pdf-images but stores paths as T:\archiverelated\books
"""
import os
import sys
import time
import shutil
import threading
import queue
from datetime import datetime
from pathlib import Path

BOOKS_ROOT = r"D:\books\pdf-images"
FEATURES_ROOT = r"D:\disk-features"  # Local storage for speed
NAS_FEATURES_ROOT = r"T:\disk-features"  # NAS destination
CATEGORY = "books"
LOG_FILE = "batch_disk_index_file.log"
PROGRESS_FILE = "batch_disk_progress_file.txt"

# Path remapping: read from D:, store paths as T: (for NAS)
PATH_REMAP = (r"D:\books", r"T:\archiverelated\books")

# Move queue for background transfers
move_queue = queue.Queue()
move_thread = None
move_stats = {"moved": 0, "failed": 0, "pending": 0, "current": None}


# ANSI color codes
CYAN = "\033[96m"
RESET = "\033[0m"


def log(msg, color=None):
    """Print to console and append to log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if color:
        print(f"{color}{line}{RESET}", flush=True)
    else:
        print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def get_all_books():
    """Get all book directories."""
    books = []
    for entry in os.listdir(BOOKS_ROOT):
        full_path = os.path.join(BOOKS_ROOT, entry)
        if os.path.isdir(full_path):
            books.append(entry)
    return sorted(books)


def count_images(book_path):
    """Count images in a book directory."""
    files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']:
        for f in Path(book_path).glob(ext):
            files.add(str(f).lower())
        for f in Path(book_path).glob(ext.upper()):
            files.add(str(f).lower())
    return len(files)


def move_worker():
    """Background worker that processes move queue one at a time."""
    global move_stats
    while True:
        book_name = move_queue.get()
        if book_name is None:  # Poison pill to stop
            move_stats["current"] = None
            move_queue.task_done()
            break

        move_stats["current"] = book_name
        src_dir = Path(FEATURES_ROOT) / CATEGORY / book_name
        dst_dir = Path(NAS_FEATURES_ROOT) / CATEGORY / book_name

        if not src_dir.exists():
            log(f"[MOVE] Skipped (not found): {book_name[:40]}...", color=CYAN)
            move_stats["failed"] += 1
            move_stats["pending"] -= 1
            move_stats["current"] = None
            move_queue.task_done()
            continue

        try:
            # Create destination parent if needed
            dst_dir.parent.mkdir(parents=True, exist_ok=True)

            # If destination exists, remove it first (re-indexing case)
            if dst_dir.exists():
                shutil.rmtree(dst_dir)

            # Move the entire folder
            move_start = time.time()
            shutil.move(str(src_dir), str(dst_dir))
            move_time = time.time() - move_start

            # Count files moved
            npz_count = len(list(dst_dir.glob("*.npz")))
            move_stats["moved"] += 1
            move_stats["pending"] -= 1
            log(f"[MOVE] {book_name[:50]} ({npz_count} files, {move_time:.1f}s)", color=CYAN)

        except Exception as e:
            log(f"[MOVE ERROR] {book_name[:40]} - {e}", color=CYAN)
            move_stats["failed"] += 1
            move_stats["pending"] -= 1

        move_stats["current"] = None
        move_queue.task_done()


def queue_move(book_name):
    """Queue a book for background move to NAS."""
    global move_stats
    move_stats["pending"] += 1
    move_queue.put(book_name)


def print_status(current_idx, total_books, current_book, eta_hours):
    """Print a compact status line showing all activity."""
    # Build status parts
    progress = f"[{current_idx}/{total_books}]"

    # Truncate book names for display
    proc_name = current_book[:35] + "..." if len(current_book) > 35 else current_book

    move_info = ""
    if move_stats["pending"] > 0 or move_stats["current"]:
        move_name = move_stats["current"][:25] + "..." if move_stats["current"] and len(move_stats["current"]) > 25 else (move_stats["current"] or "waiting")
        move_info = f" | Moving: {move_name} ({move_stats['pending']} queued)"

    eta_info = f" | ETA: {eta_hours:.1f}h" if eta_hours > 0 else ""

    status = f"{progress} Processing: {proc_name}{move_info}{eta_info}"

    # Print with carriage return to overwrite (but also log normally)
    print(f"\r{status:<120}", end="", flush=True)


def start_move_thread():
    """Start the background move worker thread."""
    global move_thread
    move_thread = threading.Thread(target=move_worker, daemon=True)
    move_thread.start()
    log("Move worker thread started (background NAS transfers)")


def stop_move_thread():
    """Stop the move worker thread and wait for queue to drain."""
    if move_stats["pending"] > 0:
        log(f"Waiting for {move_stats['pending']} pending moves to complete...")
    move_queue.put(None)  # Poison pill
    move_queue.join()
    log(f"All moves complete: {move_stats['moved']} moved, {move_stats['failed']} failed")


def main():
    log("=" * 60)
    log("BATCH DISK FEATURE INDEXER - FILE STORAGE")
    log("Extracting DISK keypoints + descriptors for LightGlue")
    log(f"Local storage: {FEATURES_ROOT}\\{CATEGORY}")
    log(f"NAS destination: {NAS_FEATURES_ROOT}\\{CATEGORY}")
    log("Auto-move: ON (queued, 1 at a time in background)")
    log("=" * 60)

    # Start background move worker
    start_move_thread()

    # Import here to avoid loading models until needed
    from disk_indexer_file import DiskIndexerFile

    books = get_all_books()
    total = len(books)
    log(f"Total books: {total}")

    # Load progress
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            completed = set(line.strip() for line in f if line.strip())
        log(f"Resuming: {len(completed)} books already completed")

    # Filter out completed
    to_index = [b for b in books if b not in completed]
    log(f"Remaining to index: {len(to_index)}")

    if not to_index:
        log("All books already indexed!")
        return

    log("=" * 60)

    # Initialize indexer (loads DISK model)
    log("Loading DISK model...")
    log(f"Path remapping: {PATH_REMAP[0]} -> {PATH_REMAP[1]}")
    indexer = DiskIndexerFile(
        category=CATEGORY,
        features_root=FEATURES_ROOT,
        batch_size=20,
        path_remap=PATH_REMAP,
        show_progress=False,  # Disable tqdm to avoid interleaved output with background moves
        device="cpu"  # Force CPU so GPU is free for other tasks (DINOv2 indexer)
    )
    log("DISK model loaded.")

    # Show initial stats
    stats = indexer.get_stats()
    log(f"Current storage stats: {stats['total_images']:,} images, {stats['total_storage_mb']:.1f} MB")
    log("=" * 60)

    start_time = time.time()

    print()  # Start fresh line for status updates

    for i, book in enumerate(to_index):
        book_path = os.path.join(BOOKS_ROOT, book)
        img_count = count_images(book_path)

        # Calculate ETA
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 0
        remaining = len(to_index) - i
        eta_hours = (avg_time * remaining) / 3600 if avg_time > 0 else 0

        if img_count == 0:
            print()  # New line before log
            log(f"[{i+1}/{len(to_index)}] SKIP (no images): {book[:50]}")
            with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
                f.write(book + '\n')
            continue

        # Show live status
        print_status(i + 1, len(to_index), book, eta_hours)

        # Index book
        idx_start = time.time()
        try:
            result = indexer.index_directory(book_path, book_name=book, skip_existing=True)
            idx_time = time.time() - idx_start

            # Print result on new line, then continue status
            print()  # New line
            log(f"[{i+1}/{len(to_index)}] {book[:45]} | {img_count} imgs | +{result['indexed']} new, {result['skipped']} skip | {idx_time:.1f}s")

            # Queue for NAS move (background, one at a time)
            if result['indexed'] > 0 or result['skipped'] > 0:
                queue_move(book)

        except Exception as e:
            idx_time = time.time() - idx_start
            print()  # New line
            log(f"[{i+1}/{len(to_index)}] FAILED: {book[:40]} - {e}")

        # Mark completed
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(book + '\n')

    # Wait for pending moves to complete
    print()  # Ensure we're on a new line
    log("=" * 60)
    log("Indexing complete! Waiting for pending NAS moves...")
    stop_move_thread()

    # Final stats
    log("=" * 60)
    log("Batch DISK indexing complete!")
    total_time = time.time() - start_time
    log(f"Total time: {total_time/3600:.1f} hours")
    log(f"Move stats: {move_stats['moved']} moved, {move_stats['failed']} failed")

    indexer.close()


if __name__ == "__main__":
    main()
