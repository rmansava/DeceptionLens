"""
Find books that haven't been indexed yet.

Compares books in source directory against books in chunk index.
"""

import json
import os
from pathlib import Path

BOOKS_SOURCE_DIR = "D:/books/pdf-images"
BOOK_TO_CHUNKS_INDEX = "D:/faiss/disk_retrieval/book_to_chunks.json"
OUTPUT_FILE = "D:/faiss/disk_retrieval/unprocessed_books.txt"

def main():
    print()
    print("=" * 70)
    print("  FIND UNPROCESSED BOOKS")
    print("=" * 70)
    print()

    # Load processed books from index
    print(f"Loading processed books from: {BOOK_TO_CHUNKS_INDEX}")
    with open(BOOK_TO_CHUNKS_INDEX, 'r', encoding='utf-8') as f:
        book_to_chunks = json.load(f)

    processed_books = set(book_to_chunks.keys())
    print(f"  Processed books in index: {len(processed_books)}")
    print()

    # Get all books from source directory
    print(f"Scanning books in: {BOOKS_SOURCE_DIR}")
    all_books = set()
    for entry in os.listdir(BOOKS_SOURCE_DIR):
        book_path = os.path.join(BOOKS_SOURCE_DIR, entry)
        if os.path.isdir(book_path):
            all_books.add(entry)

    print(f"  Total books in source: {len(all_books)}")
    print()

    # Find unprocessed books
    unprocessed = sorted(all_books - processed_books)

    print(f"  Unprocessed books: {len(unprocessed)}")
    print()

    # Save to file
    print(f"Saving unprocessed book list to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for book in unprocessed:
            f.write(f"{book}\n")

    print()
    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Total books:       {len(all_books)}")
    print(f"  Processed:         {len(processed_books)}")
    print(f"  Unprocessed:       {len(unprocessed)}")
    print(f"  Coverage:          {len(processed_books)/len(all_books)*100:.1f}%")
    print()
    print(f"  Output: {OUTPUT_FILE}")
    print()

    # Show first 10 unprocessed books as sample
    if unprocessed:
        print("  First 10 unprocessed books:")
        for book in unprocessed[:10]:
            print(f"    - {book}")
        if len(unprocessed) > 10:
            print(f"    ... and {len(unprocessed)-10} more")
        print()


if __name__ == "__main__":
    main()
