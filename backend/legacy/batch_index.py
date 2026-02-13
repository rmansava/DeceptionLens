"""
Batch indexer for all books in D:\books\pdf-images
Uses OpenSearch for both visual and face embeddings (more robust than ChromaDB)
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BOOKS_ROOT = r"D:\books\pdf-images"
LOG_FILE = "batch_index.log"

# Books already indexed (skip these)
ALREADY_INDEXED = {
    "encyclopedia of monsters",
    "Television cartoon shows an illustrated encyclopedia",
}


def log(msg):
    """Print to console and append to log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
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


def is_already_indexed(book_name):
    """Check if book is already indexed."""
    return book_name.lower() in [b.lower() for b in ALREADY_INDEXED]


def index_book_opensearch(book_name):
    """Index book using OpenSearch (visual + faces in one pass)."""
    book_path = os.path.join(BOOKS_ROOT, book_name)

    cmd = [
        sys.executable, "opensearch_indexer.py", book_path
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def count_images(book_path):
    """Count images in a book directory (deduplicated for case-insensitive filesystems)."""
    files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']:
        for f in Path(book_path).glob(ext):
            files.add(str(f).lower())
        for f in Path(book_path).glob(ext.upper()):
            files.add(str(f).lower())
    return len(files)


def main():
    log("=" * 60)
    log("BATCH INDEXER STARTED (OpenSearch)")
    log("Visual: dinov2-books index")
    log("Faces: faces-books index")
    log("=" * 60)

    books = get_all_books()
    total = len(books)

    # Filter out already indexed
    to_index = [b for b in books if not is_already_indexed(b)]
    skipped = total - len(to_index)

    log(f"Total books: {total}")
    log(f"Already indexed (skipping): {skipped}")
    log(f"To index: {len(to_index)}")

    # Progress tracking
    progress_file = "batch_progress_opensearch.txt"
    completed = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            completed = set(line.strip() for line in f if line.strip())
        log(f"Resuming from previous run: {len(completed)} books already completed")

    # Filter out completed
    to_index = [b for b in to_index if b not in completed]
    log(f"Remaining to index: {len(to_index)}")
    log("=" * 60)

    start_time = time.time()

    for i, book in enumerate(to_index):
        book_path = os.path.join(BOOKS_ROOT, book)
        img_count = count_images(book_path)

        if img_count == 0:
            log(f"[{i+1}/{len(to_index)}] SKIP (no images): {book[:60]}...")
            with open(progress_file, 'a', encoding='utf-8') as f:
                f.write(book + '\n')
            continue

        log(f"[{i+1}/{len(to_index)}] Indexing: {book[:60]}...")
        log(f"    Images: {img_count}")

        # Index visual + faces in one pass
        idx_start = time.time()
        success = index_book_opensearch(book)
        idx_time = time.time() - idx_start
        log(f"    Result: {'OK' if success else 'FAILED'} ({idx_time:.1f}s)")

        # Mark as completed
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(book + '\n')

        # Progress estimate
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = len(to_index) - (i + 1)
        eta_seconds = avg_time * remaining
        eta_hours = eta_seconds / 3600

        log(f"    Progress: {i+1}/{len(to_index)} | ETA: {eta_hours:.1f} hours")
        log("-" * 40)

    log("=" * 60)
    log("Batch indexing complete!")
    total_time = time.time() - start_time
    log(f"Total time: {total_time/3600:.1f} hours")


if __name__ == "__main__":
    main()
