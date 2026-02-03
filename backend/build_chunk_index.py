"""
Build an index mapping books to chunks for fast lookups.

Scans all chunk paths.json files and creates:
- book_to_chunks.json: {book_name: [chunk_numbers]}
- chunk_to_books.json: {chunk_number: [book_names]}

Run this once after consolidation completes. Takes ~30-60 min due to NAS reads.
"""

import json
import os
from glob import glob
from collections import defaultdict
import time

NAS_CHUNKS_DIR = "T:/faiss/disk_retrieval/chunks"
OUTPUT_DIR = "D:/faiss/disk_retrieval"

def main():
    print()
    print("=" * 70)
    print("  BUILD CHUNK INDEX")
    print("=" * 70)
    print()
    print(f"  Source: {NAS_CHUNKS_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    chunk_files = sorted(glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*_paths.json")), reverse=True)
    print(f"  Found {len(chunk_files)} chunks (scanning newest first)")
    print()

    book_to_chunks = defaultdict(set)
    chunk_to_books = {}

    start = time.time()
    import gc

    for i, paths_file in enumerate(chunk_files):
        chunk_name = os.path.basename(paths_file).replace('_paths.json', '')
        chunk_num = chunk_name.replace('chunk_', '')

        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (len(chunk_files) - i - 1) if i > 0 else 0

        try:
            with open(paths_file, 'r') as f:
                paths = json.load(f)

            books = set()
            for p in paths:
                # Path format: D:/books/pdf-images/BOOKNAME/page.jpg
                # Book name is at index 3
                parts = p.replace('\\', '/').split('/')
                if len(parts) >= 4:
                    book = parts[3]
                    books.add(book)
                    book_to_chunks[book].add(chunk_num)

            chunk_to_books[chunk_num] = sorted(books)

            # Print progress with book count
            print(f"\r  [{i+1}/{len(chunk_files)}] {chunk_name} - {len(books)} books (ETA: {eta/60:.1f}m)    ", end="", flush=True)

            # Save progress after every chunk
            b2c_path = os.path.join(OUTPUT_DIR, "book_to_chunks.json")
            c2b_path = os.path.join(OUTPUT_DIR, "chunk_to_books.json")

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Convert to list format and save without indentation (saves memory)
            book_to_chunks_list = {k: sorted(v) for k, v in book_to_chunks.items()}

            with open(b2c_path, 'w') as f:
                json.dump(book_to_chunks_list, f)
            with open(c2b_path, 'w') as f:
                json.dump(chunk_to_books, f)

            # Force garbage collection to free memory immediately
            del book_to_chunks_list
            gc.collect()

        except Exception as e:
            print(f"\n  Warning: Failed to read {chunk_name}: {e}")

    print()
    print()

    # Convert sets to sorted lists for JSON
    book_to_chunks_list = {k: sorted(v) for k, v in book_to_chunks.items()}

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    b2c_path = os.path.join(OUTPUT_DIR, "book_to_chunks.json")
    c2b_path = os.path.join(OUTPUT_DIR, "chunk_to_books.json")

    print(f"  Saving book_to_chunks.json ({len(book_to_chunks_list)} books)...")
    with open(b2c_path, 'w') as f:
        json.dump(book_to_chunks_list, f)

    print(f"  Saving chunk_to_books.json ({len(chunk_to_books)} chunks)...")
    with open(c2b_path, 'w') as f:
        json.dump(chunk_to_books, f)

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Books indexed:  {len(book_to_chunks_list)}")
    print(f"  Chunks scanned: {len(chunk_to_books)}")
    print(f"  Time:           {elapsed/60:.1f} min")
    print()
    print(f"  Output files:")
    print(f"    {b2c_path}")
    print(f"    {c2b_path}")
    print()


if __name__ == "__main__":
    main()
