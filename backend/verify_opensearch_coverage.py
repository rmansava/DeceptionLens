r"""
Verify OpenSearch coverage - ensure dinov2-books and faces-books have embeddings for every image in D:\books.

Checks OpenSearch visual and face indexes to ensure they're in sync with D:\books\pdf-images.

Usage:
    python verify_opensearch_coverage.py           # Full report
    python verify_opensearch_coverage.py --summary # Just totals
    python verify_opensearch_coverage.py --missing # List books with missing embeddings
    python verify_opensearch_coverage.py --fix     # Index missing images automatically
"""
import os
import sys
from pathlib import Path
from opensearchpy import OpenSearch

BOOKS_ROOT = r"D:\books\pdf-images"
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-books"
FACES_INDEX = "faces-books"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


def get_opensearch_client():
    """Create OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        timeout=60
    )


def count_images(book_path: Path) -> set:
    """Get set of image paths in a book folder."""
    images = set()
    for f in book_path.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.add(str(f.absolute()))
    return images


def get_indexed_paths(client: OpenSearch, index: str, book_name: str) -> set:
    """Get all indexed image paths for a book from OpenSearch."""
    indexed = set()
    try:
        # Scroll through all documents for this book
        query = {
            "query": {
                "term": {"book.keyword": book_name}
            },
            "_source": ["path"]
        }

        # Initial search
        response = client.search(
            index=index,
            body=query,
            scroll="2m",
            size=1000
        )

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        # Process initial batch
        for hit in hits:
            path = hit["_source"].get("path")
            if path:
                indexed.add(path)

        # Continue scrolling
        while len(hits) > 0:
            response = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]

            for hit in hits:
                path = hit["_source"].get("path")
                if path:
                    indexed.add(path)

        # Clean up scroll
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)

    except Exception as e:
        print(f"  Error querying {index} for {book_name}: {e}")

    return indexed


def get_all_indexed_books(client: OpenSearch, index: str) -> set:
    """Get all unique book names from an index."""
    books = set()
    try:
        query = {
            "size": 0,
            "aggs": {
                "unique_books": {
                    "terms": {
                        "field": "book.keyword",
                        "size": 10000
                    }
                }
            }
        }

        response = client.search(index=index, body=query)
        buckets = response["aggregations"]["unique_books"]["buckets"]

        for bucket in buckets:
            books.add(bucket["key"])

    except Exception as e:
        print(f"  Error getting books from {index}: {e}")

    return books


def delete_book_from_index(client: OpenSearch, index: str, book_name: str) -> int:
    """Delete all documents for a book from an index."""
    try:
        query = {
            "query": {
                "term": {"book.keyword": book_name}
            }
        }

        response = client.delete_by_query(index=index, body=query)
        deleted = response.get("deleted", 0)
        return deleted

    except Exception as e:
        print(f"  Error deleting {book_name} from {index}: {e}")
        return 0


def cleanup_orphaned_books(client: OpenSearch, books_path: Path) -> dict:
    """Delete entries from OpenSearch for books that don't exist in D:\books anymore."""
    deleted_visual = 0
    deleted_faces = 0
    failed = 0

    # Get actual books in D:\books\pdf-images
    actual_books = set(d.name for d in books_path.iterdir() if d.is_dir())

    # Check visual index
    try:
        indexed_books = get_all_indexed_books(client, VISUAL_INDEX)
        orphaned = indexed_books - actual_books

        for book in orphaned:
            count = delete_book_from_index(client, VISUAL_INDEX, book)
            if count > 0:
                deleted_visual += count
                print(f"  Deleted from visual: {book[:50]} ({count} docs)")
            else:
                failed += 1

    except Exception as e:
        print(f"  Failed to cleanup visual index: {e}")
        failed += 1

    # Check faces index
    try:
        indexed_books = get_all_indexed_books(client, FACES_INDEX)
        orphaned = indexed_books - actual_books

        for book in orphaned:
            count = delete_book_from_index(client, FACES_INDEX, book)
            if count > 0:
                deleted_faces += count
                print(f"  Deleted from faces: {book[:50]} ({count} docs)")
            else:
                failed += 1

    except Exception as e:
        print(f"  Failed to cleanup faces index: {e}")
        failed += 1

    return {"deleted_visual": deleted_visual, "deleted_faces": deleted_faces, "failed": failed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify OpenSearch coverage")
    parser.add_argument("--summary", action="store_true", help="Just show totals")
    parser.add_argument("--missing", action="store_true", help="List books with missing embeddings")
    parser.add_argument("--fix", action="store_true", help="Index any missing images")
    parser.add_argument("--book", type=str, help="Check a specific book")
    args = parser.parse_args()

    books_path = Path(BOOKS_ROOT)
    if not books_path.exists():
        print(f"Books root not found: {BOOKS_ROOT}")
        sys.exit(1)

    # Connect to OpenSearch
    client = get_opensearch_client()

    # Get all books
    all_books = sorted([d.name for d in books_path.iterdir() if d.is_dir()])

    if args.book:
        all_books = [b for b in all_books if args.book.lower() in b.lower()]
        if not all_books:
            print(f"No books match: {args.book}")
            sys.exit(1)

    total_images = 0
    total_visual = 0
    total_faces = 0
    missing_visual_books = []  # (book, img_count, visual_count, missing_count)
    missing_faces_books = []   # (book, img_count, faces_count, missing_count)

    print(f"Checking {len(all_books)} books...")
    print()

    for idx, book in enumerate(all_books):
        # Show progress
        needs_fix = len(missing_visual_books) + len(missing_faces_books)
        fix_str = f" | {needs_fix} need fixing" if needs_fix > 0 else ""
        print(f"\r  Scanning [{idx+1}/{len(all_books)}] {book[:45]:<45}{fix_str:<20}", end="", flush=True)

        book_path = books_path / book
        images = count_images(book_path)

        # Check visual index
        visual_indexed = get_indexed_paths(client, VISUAL_INDEX, book)

        # Check faces index (just count unique source images, not individual faces)
        faces_indexed = get_indexed_paths(client, FACES_INDEX, book)

        total_images += len(images)
        total_visual += len(visual_indexed)
        total_faces += len(faces_indexed)

        # Check for missing
        missing_visual = images - visual_indexed
        missing_faces = images - faces_indexed

        if missing_visual:
            missing_visual_books.append((book, len(images), len(visual_indexed), len(missing_visual), missing_visual))

        if missing_faces:
            missing_faces_books.append((book, len(images), len(faces_indexed), len(missing_faces), missing_faces))

        if not args.summary and not args.missing and not args.fix:
            visual_status = "✓" if not missing_visual else f"✗ ({len(missing_visual)} missing)"
            faces_status = "✓" if not missing_faces else f"✗ ({len(missing_faces)} missing)"
            print(f"\r  {book[:50]:<50} | {len(images):>4} imgs | Visual: {visual_status:<20} | Faces: {faces_status:<20}")

    # Clear progress line
    print("\r" + " " * 120 + "\r", end="")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total books:          {len(all_books):,}")
    print(f"Total images:         {total_images:,}")
    print(f"Visual embeddings:    {total_visual:,}")
    print(f"Face source images:   {total_faces:,}")

    visual_coverage = (total_visual / total_images * 100) if total_images > 0 else 0
    faces_coverage = (total_faces / total_images * 100) if total_images > 0 else 0
    print(f"Visual coverage:      {visual_coverage:.2f}%")
    print(f"Faces coverage:       {faces_coverage:.2f}%")

    # Report missing visual
    if missing_visual_books:
        print()
        print(f"Books with missing visual embeddings: {len(missing_visual_books)}")
        if args.missing or (not args.summary and not args.fix):
            print()
            for book, imgs, indexed, missing_count, _ in missing_visual_books[:50]:
                print(f"  {book[:50]:<50} | {imgs} imgs, {indexed} indexed, {missing_count} missing")
            if len(missing_visual_books) > 50:
                print(f"  ... and {len(missing_visual_books) - 50} more")

        total_missing_visual = sum(m[3] for m in missing_visual_books)
        print(f"\nTotal missing visual embeddings: {total_missing_visual:,}")
    else:
        print()
        print("All images have visual embeddings!")

    # Report missing faces
    if missing_faces_books:
        print()
        print(f"Books with missing face embeddings: {len(missing_faces_books)}")
        if args.missing or (not args.summary and not args.fix):
            print()
            for book, imgs, indexed, missing_count, _ in missing_faces_books[:50]:
                print(f"  {book[:50]:<50} | {imgs} imgs, {indexed} indexed, {missing_count} missing")
            if len(missing_faces_books) > 50:
                print(f"  ... and {len(missing_faces_books) - 50} more")

        total_missing_faces = sum(m[3] for m in missing_faces_books)
        print(f"\nTotal missing face embeddings: {total_missing_faces:,}")
    else:
        print()
        print("All images have face embeddings!")

    # Fix missing if requested
    if args.fix and (missing_visual_books or missing_faces_books):
        print()
        print("=" * 80)
        print("FIXING MISSING EMBEDDINGS")
        print("=" * 80)
        print()

        from opensearch_indexer import OpenSearchIndexer

        indexer = OpenSearchIndexer(
            visual_index=VISUAL_INDEX,
            faces_index=FACES_INDEX,
            enable_visual=True,
            enable_faces=True
        )

        # Combine missing books (some may be in both lists)
        all_missing = {}
        for book, imgs, indexed, missing_count, missing_paths in missing_visual_books:
            if book not in all_missing:
                all_missing[book] = set()
            all_missing[book].update(missing_paths)

        for book, imgs, indexed, missing_count, missing_paths in missing_faces_books:
            if book not in all_missing:
                all_missing[book] = set()
            all_missing[book].update(missing_paths)

        total_indexed_visual = 0
        total_indexed_faces = 0

        for i, (book, missing_paths) in enumerate(all_missing.items()):
            print(f"[{i+1}/{len(all_missing)}] {book[:50]} - {len(missing_paths)} missing images")

            # Index each missing image individually
            for path in missing_paths:
                try:
                    # Create a temporary directory with just this image for indexing
                    # Actually, we can just pass the parent directory and it will handle it
                    pass
                except Exception as e:
                    print(f"  Error indexing {path}: {e}")

            # For now, re-index the entire book directory (simpler and ensures consistency)
            book_path = books_path / book
            try:
                result = indexer.index_directory(str(book_path), book_name=book)
                total_indexed_visual += result.get("visual", 0)
                total_indexed_faces += result.get("faces", 0)
                print(f"  Indexed: {result.get('visual', 0)} visual, {result.get('faces', 0)} faces")
            except Exception as e:
                print(f"  Error indexing {book}: {e}")

        print()
        print("=" * 80)
        print("FIX COMPLETE")
        print("=" * 80)
        print(f"Indexed: {total_indexed_visual:,} visual, {total_indexed_faces:,} faces")

    # Check for orphaned books
    print()
    response = input("Check for orphaned book entries (renamed/deleted books)? (y/n): ").strip().lower()
    if response == 'y':
        print()
        print("Scanning for orphaned book entries...")
        result = cleanup_orphaned_books(client, books_path)

        if result['deleted_visual'] > 0 or result['deleted_faces'] > 0:
            print()
            print(f"Deleted {result['deleted_visual']} visual embeddings")
            print(f"Deleted {result['deleted_faces']} face embeddings")
            if result['failed'] > 0:
                print(f"Failed: {result['failed']}")
        else:
            print("No orphaned book entries found.")


if __name__ == "__main__":
    main()
