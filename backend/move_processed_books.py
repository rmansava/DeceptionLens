"""
Move the 4,743 confirmed processed books to backup NAS.
Frees up ~9TB on T: drive while keeping unprocessed books.
"""

import os
import subprocess
import time

BOOKS_LIST = "D:/processed_books_list.txt"
SOURCE_BASE = "T:/faiss/disk_retrieval/books"
DEST_BASE = "\\\\disk80\\backup\\ds923 backup\\faiss\\disk_retrieval\\books"

def main():
    print()
    print("=" * 70)
    print("  MOVE PROCESSED BOOKS")
    print("=" * 70)
    print()

    # Load book list
    print(f"  Loading book list from: {BOOKS_LIST}")
    with open(BOOKS_LIST, 'r', encoding='utf-8') as f:
        books = [line.strip() for line in f if line.strip()]

    print(f"  Found {len(books)} books to move")
    print()
    print(f"  Source: {SOURCE_BASE}")
    print(f"  Destination: {DEST_BASE}")
    print()

    # Estimate
    avg_size_gb = 13000 / 6895  # 13TB / total books
    total_gb = len(books) * avg_size_gb
    hours = (total_gb / (100/1024)) / 3600  # At 100MB/s

    print(f"  Estimated size: ~{total_gb:.0f}GB")
    print(f"  Estimated time: ~{hours:.0f} hours at 100MB/s")
    print()

    response = input("  Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("  Cancelled.")
        return

    print()
    print("-" * 70)
    print()

    start_time = time.time()
    success_count = 0
    error_count = 0
    skipped_count = 0
    errors = []

    for i, book in enumerate(books):
        source = os.path.join(SOURCE_BASE, book)
        dest = os.path.join(DEST_BASE, book)

        # Check if already moved (resume support)
        source_exists = os.path.exists(source)
        dest_exists = os.path.exists(dest)

        if not source_exists and dest_exists:
            # Already moved successfully
            skipped_count += 1
            if i % 100 == 0:  # Show progress every 100 books
                print(f"  [{i+1}/{len(books)}] Skipping (already moved): {book[:50]}...")
            continue

        if not source_exists:
            print(f"  [{i+1}/{len(books)}] WARNING: Source not found: {book[:50]}")
            error_count += 1
            errors.append((book, "Source not found"))
            continue

        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = len(books) - (i + 1)
        eta = remaining / rate / 60 if rate > 0 else 0

        print(f"  [{i+1}/{len(books)}] Moving: {book[:50]}...")
        print(f"    Success: {success_count} | Errors: {error_count} | ETA: {eta:.0f}m")

        # robocopy with /MOVE - show progress
        cmd = [
            'robocopy',
            source,
            dest,
            '/MOVE',
            '/E',
            '/R:3',
            '/W:5',
            '/COPY:DAT',
            '/ETA'  # Show estimated time for each file
        ]

        try:
            # Run robocopy with live output
            result = subprocess.run(cmd, timeout=600)
            # robocopy returns 0-7 for success, 8+ for errors
            if result.returncode < 8:
                success_count += 1
                print(f"    ✓ Moved successfully")
            else:
                error_count += 1
                errors.append((book, result.returncode))
                print(f"    ERROR: Exit code {result.returncode}")
        except Exception as e:
            error_count += 1
            errors.append((book, str(e)))
            print(f"    ERROR: {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Books moved:          {success_count}")
    print(f"  Already at dest:      {skipped_count}")
    print(f"  Errors:               {error_count}")
    print(f"  Total processed:      {success_count + skipped_count + error_count}/{len(books)}")
    print(f"  Time:                 {elapsed/3600:.1f} hours")
    print()

    if errors:
        print("  Errors encountered:")
        for book, err in errors[:10]:
            print(f"    {book}: {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")
        print()


if __name__ == "__main__":
    main()
