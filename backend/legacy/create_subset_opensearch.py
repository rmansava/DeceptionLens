"""
Create a subset OpenSearch collection from the main collection.

Usage:
    python create_subset_opensearch.py --list books.txt --output best_books --source books_visual
    python create_subset_opensearch.py --list albums.txt --output albums_visual --source all_visual

The list file should contain one book/folder name per line (partial matches work):
    encyclopedia of monsters
    world atlas
    life magazine 1985

This copies matching documents to a new collection - no GPU or model needed.
"""

import argparse
import json
import os
import sys
import time
from opensearchpy import OpenSearch


# OpenSearch connection settings
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))


def get_client() -> OpenSearch:
    """Create OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        timeout=60
    )


def load_filter_list(list_path: str) -> list:
    """Load list of book/folder names to include."""
    with open(list_path, 'r', encoding='utf-8') as f:
        items = [line.strip().lower() for line in f if line.strip()]
    return items


def get_index_settings(client: OpenSearch, source_index: str) -> dict:
    """Get settings and mappings from source index."""
    settings = client.indices.get_settings(index=source_index)
    mappings = client.indices.get_mapping(index=source_index)

    # Extract just what we need
    source_settings = settings[source_index]['settings']['index']
    source_mappings = mappings[source_index]['mappings']

    return {
        "settings": {
            "index": {
                "number_of_shards": source_settings.get('number_of_shards', 1),
                "number_of_replicas": source_settings.get('number_of_replicas', 0),
                "knn": source_settings.get('knn', True)
            }
        },
        "mappings": source_mappings
    }


def create_subset_collection(
    source_collection: str,
    filter_list: list,
    output_collection: str,
    batch_size: int = 500,
    verbose: bool = True
) -> dict:
    """
    Create a subset OpenSearch collection.

    Args:
        source_collection: Source collection name (e.g., 'books_visual')
        filter_list: List of book/folder names to include (partial match)
        output_collection: Name for the new collection
        batch_size: Documents per batch for bulk insert
        verbose: Print progress

    Returns:
        Stats dict with counts
    """
    start_time = time.time()
    client = get_client()

    # Check source exists
    if not client.indices.exists(index=source_collection):
        raise ValueError(f"Source collection not found: {source_collection}")

    # Get source count
    source_count = client.count(index=source_collection)['count']
    if verbose:
        print(f"Source collection: {source_collection}")
        print(f"  Total documents: {source_count:,}")
        print(f"  Filter items: {len(filter_list)}")

    # Delete target if exists
    if client.indices.exists(index=output_collection):
        if verbose:
            print(f"\nDeleting existing collection: {output_collection}")
        client.indices.delete(index=output_collection)

    # Create target with same settings/mappings
    if verbose:
        print(f"\nCreating collection: {output_collection}")

    index_config = get_index_settings(client, source_collection)
    client.indices.create(index=output_collection, body=index_config)

    # Scroll through source and filter
    if verbose:
        print("\nScanning and filtering documents...")

    matched_count = 0
    scanned_count = 0
    matched_items = set()
    batch = []

    # Use scroll API for large collections
    scroll_response = client.search(
        index=source_collection,
        body={"query": {"match_all": {}}, "size": 1000},
        scroll="5m"
    )

    scroll_id = scroll_response['_scroll_id']
    hits = scroll_response['hits']['hits']

    while hits:
        for hit in hits:
            scanned_count += 1
            path = hit['_source'].get('path', '')
            path_lower = path.lower()

            # Check if matches any filter
            for item in filter_list:
                if item in path_lower:
                    # Add to batch
                    batch.append({
                        "_index": output_collection,
                        "_id": hit['_id'],
                        "_source": hit['_source']
                    })
                    matched_count += 1
                    matched_items.add(item)
                    break

            # Bulk insert when batch is full
            if len(batch) >= batch_size:
                bulk_body = []
                for doc in batch:
                    bulk_body.append({"index": {"_index": doc["_index"], "_id": doc["_id"]}})
                    bulk_body.append(doc["_source"])
                client.bulk(body=bulk_body, refresh=False)
                batch = []

                if verbose:
                    print(f"  Scanned: {scanned_count:,} | Matched: {matched_count:,}")

        # Get next batch
        scroll_response = client.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = scroll_response['_scroll_id']
        hits = scroll_response['hits']['hits']

    # Insert remaining batch
    if batch:
        bulk_body = []
        for doc in batch:
            bulk_body.append({"index": {"_index": doc["_index"], "_id": doc["_id"]}})
            bulk_body.append(doc["_source"])
        client.bulk(body=bulk_body, refresh=True)

    # Clear scroll
    client.clear_scroll(scroll_id=scroll_id)

    # Refresh index
    client.indices.refresh(index=output_collection)

    elapsed = time.time() - start_time

    # Get final count
    final_count = client.count(index=output_collection)['count']

    if verbose:
        print(f"\nCompleted in {elapsed:.1f} seconds")
        print(f"  Matched {len(matched_items)}/{len(filter_list)} filter items")

        # Show missing items
        missing = set(filter_list) - matched_items
        if missing:
            print(f"\n  Not found in collection:")
            for item in sorted(missing)[:10]:
                print(f"    - {item}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

    return {
        "source_collection": source_collection,
        "output_collection": output_collection,
        "source_total": source_count,
        "subset_total": final_count,
        "matched_filters": len(matched_items),
        "total_filters": len(filter_list),
        "missing_filters": list(set(filter_list) - matched_items),
        "elapsed_seconds": elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create a subset OpenSearch collection from the main collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_subset_opensearch.py --list best_books.txt --output best_books_visual --source books_visual
    python create_subset_opensearch.py --list albums.txt --output albums_visual

List file format (one per line, partial match):
    encyclopedia of monsters
    world atlas
    life magazine
        """
    )

    parser.add_argument(
        "--list", "-l",
        required=True,
        help="Path to text file with book/folder names (one per line)"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Name for the new collection"
    )

    parser.add_argument(
        "--source", "-s",
        default="books_visual",
        help="Source collection name (default: books_visual)"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=500,
        help="Documents per batch for bulk insert (default: 500)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Load filter list
    if not os.path.exists(args.list):
        print(f"Error: List file not found: {args.list}")
        sys.exit(1)

    filter_list = load_filter_list(args.list)

    if not filter_list:
        print("Error: List file is empty")
        sys.exit(1)

    # Create subset
    try:
        stats = create_subset_collection(
            source_collection=args.source,
            filter_list=filter_list,
            output_collection=args.output,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )

        print(f"\n{'='*60}")
        print(f"SUBSET COLLECTION CREATED")
        print(f"{'='*60}")
        print(f"Source: {stats['source_collection']} ({stats['source_total']:,} documents)")
        print(f"Subset: {stats['output_collection']} ({stats['subset_total']:,} documents)")
        print(f"Reduction: {100*(1 - stats['subset_total']/stats['source_total']):.1f}%")
        print(f"Filters matched: {stats['matched_filters']}/{stats['total_filters']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
