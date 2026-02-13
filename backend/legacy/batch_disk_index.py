"""
Batch DISK feature indexer for all books in D:\books\pdf-images
Pre-computes DISK keypoints and descriptors for fast LightGlue verification.
Stores features in SQL Server (trivia.dbo.DiskFeatures table).

Path remapping: Reads from D:\books but stores paths as T:\archiverelated\books
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BOOKS_ROOT = r"D:\books\pdf-images"
NAS_BACKUP = r"T:\archiverelated\books\pdf-images"
LOG_FILE = "batch_disk_index.log"
PROGRESS_FILE = "batch_disk_progress.txt"

# Path remapping: read from D:, store as T: (for NAS)
PATH_REMAP = (r"D:\books", r"T:\archiverelated\books")


class Colors:
    PINK = '\033[95m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


def log(msg, color=None):
    """Print to console and append to log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    # Colored console output
    if color:
        print(f"{color}{line}{Colors.RESET}", flush=True)
    else:
        print(line, flush=True)
    # Plain text to file
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


def main():
    log("=" * 60)
    log("BATCH DISK FEATURE INDEXER STARTED")
    log("Extracting DISK keypoints + descriptors for LightGlue")
    log("Storage: SQL Server trivia.dbo.DiskFeatures")
    log("=" * 60)

    # Import here to avoid loading models until needed
    from disk_indexer import DiskIndexer

    books = get_all_books()
    total = len(books)
    log(f"Total books: {total}")

    # Load progress
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            completed = set(line.strip() for line in f if line.strip())
        log(f"Resuming: {len(completed)} books already completed")

    # Filter out completed and those already on NAS
    to_index = []
    skipped_nas = 0
    for b in books:
        if b in completed:
            continue
        # Check if already exists on NAS backup
        nas_path = os.path.join(NAS_BACKUP, b)
        if os.path.exists(nas_path):
            skipped_nas += 1
            continue
        to_index.append(b)

    log(f"Remaining to index: {len(to_index)}")
    if skipped_nas > 0:
        log(f"Skipped {skipped_nas} books (already on NAS)", Colors.PINK)

    if not to_index:
        log("All books already indexed!")
        return

    log("=" * 60)

    # Initialize indexer (loads DISK model)
    log("Loading DISK model...")
    log(f"Path remapping: {PATH_REMAP[0]} -> {PATH_REMAP[1]}")
    indexer = DiskIndexer(batch_size=20, path_remap=PATH_REMAP)
    log("DISK model loaded.")

    # Show initial stats
    stats = indexer.get_stats()
    log(f"Current DB stats: {stats['total_images']:,} images, {stats['total_storage_mb']:.1f} MB")
    log("=" * 60)

    start_time = time.time()

    for i, book in enumerate(to_index):
        book_path = os.path.join(BOOKS_ROOT, book)
        img_count = count_images(book_path)

        if img_count == 0:
            log(f"[{i+1}/{len(to_index)}] SKIP (no images): {book[:50]}...")
            with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
                f.write(book + '\n')
            continue

        log(f"[{i+1}/{len(to_index)}] Indexing DISK: {book[:50]}...")
        log(f"    Images: {img_count}")

        # Index book
        idx_start = time.time()
        try:
            result = indexer.index_directory(book_path, book_name=book, skip_existing=True)
            idx_time = time.time() - idx_start
            log(f"    Result: indexed={result['indexed']}, skipped={result['skipped']}, failed={result['failed']} ({idx_time:.1f}s)")
        except Exception as e:
            idx_time = time.time() - idx_start
            log(f"    FAILED: {e} ({idx_time:.1f}s)")

        # Mark completed
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(book + '\n')

        # Progress estimate
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = len(to_index) - (i + 1)
        eta_seconds = avg_time * remaining
        eta_hours = eta_seconds / 3600

        log(f"    Progress: {i+1}/{len(to_index)} | ETA: {eta_hours:.1f} hours")
        log("-" * 40)

    # Final stats
    log("=" * 60)
    log("Batch DISK indexing complete!")
    total_time = time.time() - start_time
    log(f"Total time: {total_time/3600:.1f} hours")

    stats = indexer.get_stats()
    log(f"Final DB stats:")
    log(f"  Total images: {stats['total_images']:,}")
    log(f"  Total books: {stats['total_books']:,}")
    log(f"  Total keypoints: {stats['total_keypoints']:,}")
    log(f"  Storage used: {stats['total_storage_mb']:.1f} MB")
    log(f"  Avg per image: {stats['avg_storage_per_image_kb']:.1f} KB")

    indexer.close()


if __name__ == "__main__":
    main()
