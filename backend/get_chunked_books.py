"""
Quick scan to get list of books actually in chunks.
Faster than build_chunk_index.bat - only extracts unique books.
"""

import json
import os
from glob import glob
import time

NAS_CHUNKS_DIR = "T:/faiss/disk_retrieval/chunks"
OUTPUT_FILE = "D:/faiss/disk_retrieval/chunked_books.txt"

def main():
    print()
    print("=" * 70)
    print("  GET CHUNKED BOOKS LIST")
    print("=" * 70)
    print()

    chunk_files = sorted(glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*_paths.json")))
    print(f"  Found {len(chunk_files)} chunks to scan")
    print()

    all_books = set()
    start = time.time()

    for i, paths_file in enumerate(chunk_files):
        chunk_name = os.path.basename(paths_file).replace('_paths.json', '')
        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (len(chunk_files) - i - 1) if i > 0 else 0

        print(f"\r  [{i+1}/{len(chunk_files)}] {chunk_name} | Books found: {len(all_books)} | ETA: {eta/60:.1f}m    ", end="", flush=True)

        try:
            with open(paths_file, 'r') as f:
                paths = json.load(f)

            for p in paths:
                sep = '/' if '/' in p else '\\'
                book = p.split(sep)[0]
                all_books.add(book)

        except Exception as e:
            print(f"\n  Warning: Failed to read {chunk_name}: {e}")

    print()
    print()

    # Sort and save
    sorted_books = sorted(all_books)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for book in sorted_books:
            f.write(f'{book}\n')

    elapsed = time.time() - start

    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Unique books in chunks: {len(all_books)}")
    print(f"  Total books on NAS:     6,895")
    print(f"  Remaining to process:   {6895 - len(all_books)}")
    print(f"  Time:                   {elapsed/60:.1f} min")
    print()
    print(f"  Output: {OUTPUT_FILE}")
    print()

    # Show sample
    print("  First 10 books:")
    for book in sorted_books[:10]:
        print(f"    {book}")
    print()
    print("  Last 10 books:")
    for book in sorted_books[-10:]:
        print(f"    {book}")
    print()


if __name__ == "__main__":
    main()
