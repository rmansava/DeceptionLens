"""
Batch indexer for all books in D:\books\pdf-images
Indexes into the 'books' collection, one book at a time.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

BOOKS_ROOT = r"D:\books\pdf-images"
COLLECTION = "books"
DB_PATH = "./chroma_db"

# Books already indexed (skip these)
ALREADY_INDEXED = {
    "encyclopedia of monsters",
    "Television Cartoon Shows An Illustrated Encyclopedia, 1949 2003 Volume 1 (2nd Ed)",
}

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

def index_book(book_name, mode):
    """Index a single book with the given mode."""
    book_path = os.path.join(BOOKS_ROOT, book_name)

    cmd = [
        sys.executable, "main.py", "index",
        "--dir", book_path,
        "--collection", COLLECTION,
        "--db-path", DB_PATH,
        "--mode", mode,
        "--batch-size", "20"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def count_images(book_path):
    """Count images in a book directory."""
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']:
        count += len(list(Path(book_path).glob(ext)))
        count += len(list(Path(book_path).glob(ext.upper())))
    return count

def main():
    books = get_all_books()
    total = len(books)

    # Filter out already indexed
    to_index = [b for b in books if not is_already_indexed(b)]
    skipped = total - len(to_index)

    print(f"Total books: {total}")
    print(f"Already indexed (skipping): {skipped}")
    print(f"To index: {len(to_index)}")
    print("=" * 60)

    # Progress tracking
    progress_file = "batch_progress.txt"
    completed = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            completed = set(line.strip() for line in f if line.strip())
        print(f"Resuming from previous run: {len(completed)} books already completed")

    # Filter out completed
    to_index = [b for b in to_index if b not in completed]
    print(f"Remaining to index: {len(to_index)}")
    print("=" * 60)

    start_time = time.time()

    for i, book in enumerate(to_index):
        book_path = os.path.join(BOOKS_ROOT, book)
        img_count = count_images(book_path)

        if img_count == 0:
            print(f"[{i+1}/{len(to_index)}] SKIP (no images): {book[:60]}...")
            with open(progress_file, 'a', encoding='utf-8') as f:
                f.write(book + '\n')
            continue

        print(f"\n[{i+1}/{len(to_index)}] Indexing: {book[:60]}...")
        print(f"    Images: {img_count}")

        # Visual pass
        print("    Visual pass...", end=" ", flush=True)
        success, stdout, stderr = index_book(book, "visual_only")
        if success:
            print("OK")
        else:
            print("FAILED")
            print(f"    Error: {stderr[:200]}")

        # Faces pass
        print("    Faces pass...", end=" ", flush=True)
        success, stdout, stderr = index_book(book, "faces_only")
        if success:
            print("OK")
        else:
            print("FAILED")
            print(f"    Error: {stderr[:200]}")

        # Mark as completed
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(book + '\n')

        # Progress estimate
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = len(to_index) - (i + 1)
        eta_seconds = avg_time * remaining
        eta_hours = eta_seconds / 3600

        print(f"    Progress: {i+1}/{len(to_index)} | ETA: {eta_hours:.1f} hours")

    print("\n" + "=" * 60)
    print("Batch indexing complete!")
    total_time = time.time() - start_time
    print(f"Total time: {total_time/3600:.1f} hours")

if __name__ == "__main__":
    main()
