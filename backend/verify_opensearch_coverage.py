r"""
Verify OpenSearch coverage - ensure dinov2-books has embeddings for every image in D:\books.

Checks OpenSearch visual index to ensure it's in sync with D:\books\pdf-images.
Note: Face index is not checked since it only contains images with detected faces (partial by design).

Usage:
    python verify_opensearch_coverage.py           # Full report
    python verify_opensearch_coverage.py --summary # Just totals
    python verify_opensearch_coverage.py --fix     # Index missing images automatically

Log file: verify_opensearch_coverage.log (same directory)
"""
import sys
import logging
from pathlib import Path
from datetime import datetime
from opensearchpy import OpenSearch

# Set up logging to file and console
LOG_FILE = Path(__file__).parent / "verify_opensearch_coverage.log"

def setup_logging():
    """Configure logging to both file and console."""
    logger = logging.getLogger("verify_opensearch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

BOOKS_ROOT = r"D:\books\pdf-images"
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-books"

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
        query = {
            "query": {
                "term": {"book": book_name}
            },
            "_source": ["path"]
        }

        response = client.search(
            index=index,
            body=query,
            scroll="2m",
            size=1000
        )

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        for hit in hits:
            path = hit["_source"].get("path")
            if path:
                indexed.add(path)

        while len(hits) > 0:
            response = client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]

            for hit in hits:
                path = hit["_source"].get("path")
                if path:
                    indexed.add(path)

        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)

    except Exception as e:
        logger.info(f"  Error querying {index} for {book_name}: {e}")

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
                        "field": "book",
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
        logger.info(f"  Error getting books from {index}: {e}")

    return books


def delete_book_from_index(client: OpenSearch, index: str, book_name: str) -> int:
    """Delete all documents for a book from an index."""
    try:
        query = {
            "query": {
                "term": {"book": book_name}
            }
        }

        response = client.delete_by_query(index=index, body=query)
        deleted = response.get("deleted", 0)
        return deleted

    except Exception as e:
        logger.info(f"  Error deleting {book_name} from {index}: {e}")
        return 0


def cleanup_orphaned_books(client: OpenSearch, books_path: Path) -> dict:
    """Delete entries from OpenSearch for books that don't exist in D:\books anymore."""
    deleted = 0
    failed = 0

    actual_books = set(d.name for d in books_path.iterdir() if d.is_dir())

    try:
        indexed_books = get_all_indexed_books(client, VISUAL_INDEX)
        orphaned = indexed_books - actual_books

        for book in orphaned:
            count = delete_book_from_index(client, VISUAL_INDEX, book)
            if count > 0:
                deleted += count
                logger.info(f"  Deleted: {book} ({count} docs)")
            else:
                failed += 1

    except Exception as e:
        logger.info(f"  Failed to cleanup: {e}")
        failed += 1

    return {"deleted": deleted, "failed": failed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify OpenSearch visual coverage")
    parser.add_argument("--summary", action="store_true", help="Just show totals")
    parser.add_argument("--fix", action="store_true", help="Index any missing images")
    parser.add_argument("--book", type=str, help="Check a specific book")
    args = parser.parse_args()

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"OpenSearch Visual Coverage - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    books_path = Path(BOOKS_ROOT)
    if not books_path.exists():
        logger.info(f"Books root not found: {BOOKS_ROOT}")
        sys.exit(1)

    client = get_opensearch_client()

    all_books = sorted([d.name for d in books_path.iterdir() if d.is_dir()])

    if args.book:
        all_books = [b for b in all_books if args.book.lower() in b.lower()]
        if not all_books:
            logger.info(f"No books match: {args.book}")
            sys.exit(1)

    total_images = 0
    total_indexed = 0
    missing_books = []  # (book, img_count, indexed_count, missing_count, missing_paths)

    logger.info(f"Checking {len(all_books)} books...")
    logger.info("")

    for idx, book in enumerate(all_books):
        needs_fix = len(missing_books)
        fix_str = f" | {needs_fix} need fixing" if needs_fix > 0 else ""
        print(f"\r  Scanning [{idx+1}/{len(all_books)}] {book[:45]:<45}{fix_str:<20}", end="", flush=True)

        book_path = books_path / book
        images = count_images(book_path)
        indexed = get_indexed_paths(client, VISUAL_INDEX, book)

        total_images += len(images)
        total_indexed += len(indexed)

        missing = images - indexed

        if missing:
            missing_books.append((book, len(images), len(indexed), len(missing), missing))
            logger.info(f"MISSING: {book} | {len(images)} imgs, {len(indexed)} indexed, {len(missing)} missing")

    print("\r" + " " * 120 + "\r", end="")
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total books:          {len(all_books):,}")
    logger.info(f"Total images:         {total_images:,}")
    logger.info(f"Visual embeddings:    {total_indexed:,}")

    coverage = (total_indexed / total_images * 100) if total_images > 0 else 0
    logger.info(f"Coverage:             {coverage:.2f}%")

    if missing_books:
        logger.info("")
        logger.info(f"Books with missing embeddings: {len(missing_books)}")
        if not args.summary:
            logger.info("")
            for book, imgs, indexed, missing_count, _ in missing_books:
                logger.info(f"  {book} | {imgs} imgs, {indexed} indexed, {missing_count} missing")

        total_missing = sum(m[3] for m in missing_books)
        logger.info(f"\nTotal missing: {total_missing:,}")
    else:
        logger.info("")
        logger.info("All images have visual embeddings!")

    # Fix missing if requested
    if args.fix and missing_books:
        logger.info("")
        logger.info("=" * 80)
        logger.info("FIXING MISSING EMBEDDINGS")
        logger.info("=" * 80)
        logger.info("")

        from opensearch_indexer import OpenSearchIndexer

        indexer = OpenSearchIndexer(
            visual_index=VISUAL_INDEX,
            enable_visual=True,
            enable_faces=False
        )

        total_fixed = 0

        for i, (book, imgs, indexed, missing_count, missing_paths) in enumerate(missing_books):
            logger.info(f"[{i+1}/{len(missing_books)}] {book} - {missing_count} missing")

            # Delete existing stale entries first (handles renamed files)
            if indexed > 0:
                deleted = delete_book_from_index(client, VISUAL_INDEX, book)
                logger.info(f"  Deleted {deleted} stale entries")

            book_path = books_path / book
            try:
                result = indexer.index_directory(str(book_path), book_name=book)
                total_fixed += result.get("visual", 0)
                logger.info(f"  Indexed: {result.get('visual', 0)}")
            except Exception as e:
                logger.info(f"  Error: {e}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("FIX COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Indexed: {total_fixed:,} visual embeddings")

    # Check for orphaned books
    print()
    response = input("Check for orphaned book entries (renamed/deleted books)? (y/n): ").strip().lower()
    if response == 'y':
        logger.info("")
        logger.info("Scanning for orphaned book entries...")
        result = cleanup_orphaned_books(client, books_path)

        if result['deleted'] > 0:
            logger.info("")
            logger.info(f"Deleted {result['deleted']} orphaned embeddings")
            if result['failed'] > 0:
                logger.info(f"Failed: {result['failed']}")
        else:
            logger.info("No orphaned book entries found.")

    logger.info("")
    logger.info(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
