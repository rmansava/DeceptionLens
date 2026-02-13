"""
Re-index missing books that were skipped due to glob bracket bug.

Usage:
    python reindex_missing_books.py
    python reindex_missing_books.py --dry-run
    python reindex_missing_books.py --book "Specific Book Name"
"""

import os
import sys
import argparse
from opensearch_indexer import OpenSearchIndexer

MISSING_BOOKS_FILE = "C:/Users/rmans/missing_books_paths.txt"


def main():
    parser = argparse.ArgumentParser(description="Re-index missing books")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be indexed")
    parser.add_argument("--book", type=str, help="Index a specific book by name (partial match)")
    parser.add_argument("--visual-only", action="store_true", help="Only index visual embeddings")
    parser.add_argument("--faces-only", action="store_true", help="Only index face embeddings")

    args = parser.parse_args()

    # Load missing books
    if not os.path.exists(MISSING_BOOKS_FILE):
        print(f"Missing books file not found: {MISSING_BOOKS_FILE}")
        print("Run the verification script first to generate the list.")
        sys.exit(1)

    with open(MISSING_BOOKS_FILE, 'r', encoding='utf-8') as f:
        missing_paths = [line.strip() for line in f if line.strip()]

    # Filter to specific book if requested
    if args.book:
        missing_paths = [p for p in missing_paths if args.book.lower() in p.lower()]
        if not missing_paths:
            print(f"No missing books match: {args.book}")
            sys.exit(1)

    # Filter out non-existent or empty folders
    valid_paths = []
    for path in missing_paths:
        if os.path.exists(path):
            # Check if has images
            image_count = len([f for f in os.listdir(path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'))])
            if image_count > 0:
                valid_paths.append((path, image_count))

    print(f"Missing books to re-index: {len(valid_paths)}")
    total_pages = sum(count for _, count in valid_paths)
    print(f"Total pages: {total_pages:,}")
    print()

    if args.dry_run:
        print("DRY RUN - Would index:")
        for path, count in valid_paths[:20]:
            book_name = os.path.basename(path)
            print(f"  {book_name[:60]}... ({count} pages)")
        if len(valid_paths) > 20:
            print(f"  ... and {len(valid_paths) - 20} more")
        sys.exit(0)

    # Initialize indexer
    indexer = OpenSearchIndexer(
        visual_index="dinov2-books",
        faces_index="faces-books"
    )

    print(f"Starting re-indexing of {len(valid_paths)} books...")
    print()

    success_count = 0
    fail_count = 0
    total_indexed = {"visual": 0, "faces": 0}

    for i, (path, expected_count) in enumerate(valid_paths):
        book_name = os.path.basename(path)
        print(f"[{i+1}/{len(valid_paths)}] {book_name[:50]}... ({expected_count} pages)")

        try:
            result = indexer.index_directory(path, book_name=book_name)
            total_indexed["visual"] += result.get("visual", 0)
            total_indexed["faces"] += result.get("faces", 0)
            print(f"  -> Indexed: {result.get('visual', 0)} visual, {result.get('faces', 0)} faces")
            success_count += 1
        except Exception as e:
            print(f"  -> FAILED: {e}")
            fail_count += 1

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Books processed: {success_count + fail_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total indexed - Visual: {total_indexed['visual']:,}, Faces: {total_indexed['faces']:,}")


if __name__ == "__main__":
    main()
