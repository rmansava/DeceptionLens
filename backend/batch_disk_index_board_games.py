r"""
Batch DISK feature indexer for Board Games.
Extracts DISK keypoints + descriptors for LightGlue geometric verification.

Source: T:\archiverelated\board games (NAS - read directly)
Local features: D:\disk-features\board_games (fast SSD writes)
NAS features: T:\disk-features\board_games (final destination)

Each folder is indexed, saved to local SSD, then moved to NAS in background.
"""
import os
import sys
import time
import shutil
import threading
import queue
from datetime import datetime
from pathlib import Path

BOARD_GAMES_ROOT = r"T:\archiverelated\board games"
FEATURES_ROOT = r"D:\disk-features"  # Local SSD for fast writes
NAS_FEATURES_ROOT = r"T:\disk-features"  # NAS destination
CATEGORY = "board_games"
LOG_FILE = "batch_disk_board_games.log"
PROGRESS_FILE = "batch_disk_board_games_progress.txt"

# Move queue for background transfers
move_queue = queue.Queue()
move_thread = None
move_stats = {"moved": 0, "failed": 0, "pending": 0, "current": None}

# ANSI color codes
CYAN = "\033[96m"
PINK = "\033[95m"
GREEN = "\033[92m"
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


def get_all_folders():
    """Get all board game directories."""
    folders = []
    for entry in os.listdir(BOARD_GAMES_ROOT):
        full_path = os.path.join(BOARD_GAMES_ROOT, entry)
        if os.path.isdir(full_path):
            folders.append(entry)
    return sorted(folders)


def count_images(folder_path):
    """Count images in a folder (recursive)."""
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                count += 1
    return count


def move_worker():
    """Background worker that moves folders from local SSD to NAS."""
    global move_stats
    while True:
        folder_name = move_queue.get()
        if folder_name is None:  # Poison pill
            move_stats["current"] = None
            move_queue.task_done()
            break

        move_stats["current"] = folder_name
        src_dir = Path(FEATURES_ROOT) / CATEGORY / folder_name
        dst_dir = Path(NAS_FEATURES_ROOT) / CATEGORY / folder_name

        if not src_dir.exists():
            log(f"[MOVE] Skipped (not found): {folder_name[:40]}...", color=CYAN)
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
            log(f"[MOVE] {folder_name[:50]} ({npz_count} files, {move_time:.1f}s)", color=CYAN)

        except Exception as e:
            log(f"[MOVE ERROR] {folder_name[:40]} - {e}", color=CYAN)
            move_stats["failed"] += 1
            move_stats["pending"] -= 1

        move_stats["current"] = None
        move_queue.task_done()


def queue_move(folder_name):
    """Queue a folder for background move to NAS."""
    global move_stats
    move_stats["pending"] += 1
    move_queue.put(folder_name)


def print_status(current_idx, total_folders, current_folder, eta_hours):
    """Print a compact status line showing all activity."""
    progress = f"[{current_idx}/{total_folders}]"

    # Truncate folder names for display
    proc_name = current_folder[:35] + "..." if len(current_folder) > 35 else current_folder

    move_info = ""
    if move_stats["pending"] > 0 or move_stats["current"]:
        move_name = move_stats["current"][:25] + "..." if move_stats["current"] and len(move_stats["current"]) > 25 else (move_stats["current"] or "waiting")
        move_info = f" | Moving: {move_name} ({move_stats['pending']} queued)"

    eta_info = f" | ETA: {eta_hours:.1f}h" if eta_hours > 0 else ""

    status = f"{progress} Processing: {proc_name}{move_info}{eta_info}"
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
    log("BATCH DISK FEATURE INDEXER - BOARD GAMES")
    log("Extracting DISK keypoints + descriptors for LightGlue")
    log(f"Source: {BOARD_GAMES_ROOT}")
    log(f"Local storage: {FEATURES_ROOT}\\{CATEGORY}")
    log(f"NAS destination: {NAS_FEATURES_ROOT}\\{CATEGORY}")
    log("Auto-move: ON (queued, 1 at a time in background)")
    log("=" * 60)

    # Create directories
    os.makedirs(os.path.join(FEATURES_ROOT, CATEGORY), exist_ok=True)
    os.makedirs(os.path.join(NAS_FEATURES_ROOT, CATEGORY), exist_ok=True)

    # Start background move worker
    start_move_thread()

    # Import here to avoid loading models until needed
    from disk_indexer_file import DiskIndexerFile

    folders = get_all_folders()
    total = len(folders)
    log(f"Total folders: {total}")

    # Load progress
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            completed = set(line.strip() for line in f if line.strip())
        log(f"Resuming: {len(completed)} folders already completed")

    # Filter out completed and those already on NAS
    to_index = []
    skipped_nas = 0
    for folder in folders:
        if folder in completed:
            continue
        # Check if already exists on NAS
        nas_path = os.path.join(NAS_FEATURES_ROOT, CATEGORY, folder)
        if os.path.exists(nas_path):
            # Check if it has .npz files
            npz_count = len(list(Path(nas_path).glob("*.npz")))
            if npz_count > 0:
                skipped_nas += 1
                continue
        to_index.append(folder)

    log(f"Remaining to index: {len(to_index)}")
    if skipped_nas > 0:
        log(f"Skipped {skipped_nas} folders (already on NAS)", color=PINK)

    if not to_index:
        log("All folders already indexed!")
        stop_move_thread()
        return

    log("=" * 60)

    # Initialize indexer (loads DISK model)
    log("Loading DISK model...")
    indexer = DiskIndexerFile(
        category=CATEGORY,
        features_root=FEATURES_ROOT,
        batch_size=20,
        path_remap=None,  # No remapping needed - paths are already NAS paths
        show_progress=False,  # Disable tqdm to avoid interleaved output
        device="cuda"  # Use GPU for faster indexing
    )
    log("DISK model loaded.")

    # Show initial stats
    stats = indexer.get_stats()
    log(f"Current storage stats: {stats['total_images']:,} images, {stats['total_storage_mb']:.1f} MB")
    log("=" * 60)

    start_time = time.time()

    print()  # Start fresh line for status updates

    for i, folder in enumerate(to_index):
        folder_path = os.path.join(BOARD_GAMES_ROOT, folder)
        img_count = count_images(folder_path)

        # Calculate ETA
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 0
        remaining = len(to_index) - i
        eta_hours = (avg_time * remaining) / 3600 if avg_time > 0 else 0

        if img_count == 0:
            print()  # New line before log
            log(f"[{i+1}/{len(to_index)}] SKIP (no images): {folder[:50]}")
            with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
                f.write(folder + '\n')
            continue

        # Show live status
        print_status(i + 1, len(to_index), folder, eta_hours)

        # Index folder
        idx_start = time.time()
        try:
            result = indexer.index_directory(folder_path, book_name=folder, skip_existing=True)
            idx_time = time.time() - idx_start

            # Print result on new line
            print()  # New line
            log(f"[{i+1}/{len(to_index)}] {folder[:45]} | {img_count} imgs | +{result['indexed']} new, {result['skipped']} skip | {idx_time:.1f}s", color=GREEN)

            # Queue for NAS move (background)
            if result['indexed'] > 0 or result['skipped'] > 0:
                queue_move(folder)

        except Exception as e:
            idx_time = time.time() - idx_start
            print()  # New line
            log(f"[{i+1}/{len(to_index)}] FAILED: {folder[:40]} - {e}")

        # Mark completed
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(folder + '\n')

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
