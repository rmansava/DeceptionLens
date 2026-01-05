"""
Create subset indexes from the main indexes (both CLIP/FAISS and DINOv2/OpenSearch).

Usage:
    python create_subset_index.py --list books.txt --name best_books
    python create_subset_index.py --list albums.txt --name albums --clip-only
    python create_subset_index.py --list magazines.txt --name mags --opensearch-only

The list file should contain one book/folder name per line (partial matches work):
    encyclopedia of monsters
    world atlas
    life magazine 1985

This extracts from existing indexes - no GPU or model needed.
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Optional imports
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from opensearchpy import OpenSearch
    HAS_OPENSEARCH = True
except ImportError:
    HAS_OPENSEARCH = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# CLIP/FAISS defaults
DEFAULT_CLIP_SOURCE = r"D:\faiss\books"
DEFAULT_CLIP_OUTPUT = r"D:\faiss"

# OpenSearch defaults
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))
DEFAULT_OPENSEARCH_SOURCE = "books_visual"


# ============================================================================
# UTILITIES
# ============================================================================

def load_filter_list(list_path: str) -> list:
    """Load list of book/folder names to include."""
    with open(list_path, 'r', encoding='utf-8') as f:
        items = [line.strip().lower() for line in f if line.strip()]
    return items


def matches_filter(path: str, filter_list: list) -> bool:
    """Check if path matches any item in filter list."""
    path_lower = path.lower()
    return any(item in path_lower for item in filter_list)


# ============================================================================
# CLIP/FAISS SUBSET
# ============================================================================

def create_clip_subset(
    source_dir: str,
    filter_list: list,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """Create a subset FAISS index from the main index."""
    if not HAS_FAISS:
        raise ImportError("faiss not installed. Run: pip install faiss-cpu")

    start_time = time.time()

    index_path = os.path.join(source_dir, "index.faiss")
    paths_path = os.path.join(source_dir, "paths.json")

    if verbose:
        print(f"\n[CLIP/FAISS]")
        print(f"  Source: {source_dir}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    # Load
    index = faiss.read_index(index_path)
    with open(paths_path, 'r', encoding='utf-8') as f:
        all_paths = json.load(f)

    if verbose:
        print(f"  Total: {index.ntotal:,} images")

    # Filter
    matching_indices = []
    matching_paths = []
    matched_items = set()

    for i, path in enumerate(all_paths):
        for item in filter_list:
            if item in path.lower():
                matching_indices.append(i)
                matching_paths.append(path)
                matched_items.add(item)
                break

    if verbose:
        print(f"  Matched: {len(matching_paths):,} images")

    if not matching_indices:
        print(f"  WARNING: No matching images found in CLIP index!")
        return {"subset_total": 0, "matched_filters": 0}

    # Extract embeddings
    if verbose:
        print(f"  Extracting embeddings...")

    dimension = index.d
    subset_embeddings = np.zeros((len(matching_indices), dimension), dtype=np.float32)

    for i, idx in enumerate(matching_indices):
        subset_embeddings[i] = index.reconstruct(idx)
        if verbose and (i + 1) % 50000 == 0:
            print(f"    {i + 1:,}/{len(matching_indices):,}")

    # Create new index
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(subset_embeddings)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    new_index_path = os.path.join(output_dir, "index.faiss")
    new_paths_path = os.path.join(output_dir, "paths.json")

    faiss.write_index(new_index, new_index_path)
    with open(new_paths_path, 'w', encoding='utf-8') as f:
        json.dump(matching_paths, f, indent=2)

    elapsed = time.time() - start_time

    if verbose:
        print(f"  Saved: {output_dir}")
        print(f"  Time: {elapsed:.1f}s")

    return {
        "source_total": index.ntotal,
        "subset_total": len(matching_paths),
        "matched_filters": len(matched_items),
        "elapsed_seconds": elapsed,
        "output_dir": output_dir
    }


# ============================================================================
# OPENSEARCH SUBSET
# ============================================================================

def get_opensearch_client() -> 'OpenSearch':
    """Create OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        timeout=60
    )


def get_index_settings(client, source_index: str) -> dict:
    """Get settings and mappings from source index."""
    settings = client.indices.get_settings(index=source_index)
    mappings = client.indices.get_mapping(index=source_index)

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


def create_opensearch_subset(
    source_collection: str,
    filter_list: list,
    output_collection: str,
    batch_size: int = 500,
    verbose: bool = True
) -> dict:
    """Create a subset OpenSearch collection."""
    if not HAS_OPENSEARCH:
        raise ImportError("opensearch-py not installed. Run: pip install opensearch-py")

    start_time = time.time()
    client = get_opensearch_client()

    if verbose:
        print(f"\n[DINOv2/OpenSearch]")
        print(f"  Source: {source_collection}")

    if not client.indices.exists(index=source_collection):
        raise ValueError(f"Source collection not found: {source_collection}")

    source_count = client.count(index=source_collection)['count']
    if verbose:
        print(f"  Total: {source_count:,} documents")

    # Delete target if exists
    if client.indices.exists(index=output_collection):
        client.indices.delete(index=output_collection)

    # Create target
    index_config = get_index_settings(client, source_collection)
    client.indices.create(index=output_collection, body=index_config)

    # Scroll and filter
    matched_count = 0
    matched_items = set()
    batch = []

    scroll_response = client.search(
        index=source_collection,
        body={"query": {"match_all": {}}, "size": 1000},
        scroll="5m"
    )

    scroll_id = scroll_response['_scroll_id']
    hits = scroll_response['hits']['hits']
    scanned = 0

    while hits:
        for hit in hits:
            scanned += 1
            path = hit['_source'].get('path', '')

            for item in filter_list:
                if item in path.lower():
                    batch.append({
                        "_index": output_collection,
                        "_id": hit['_id'],
                        "_source": hit['_source']
                    })
                    matched_count += 1
                    matched_items.add(item)
                    break

            if len(batch) >= batch_size:
                bulk_body = []
                for doc in batch:
                    bulk_body.append({"index": {"_index": doc["_index"], "_id": doc["_id"]}})
                    bulk_body.append(doc["_source"])
                client.bulk(body=bulk_body, refresh=False)
                batch = []

                if verbose and matched_count % 10000 == 0:
                    print(f"    Scanned: {scanned:,} | Matched: {matched_count:,}")

        scroll_response = client.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = scroll_response['_scroll_id']
        hits = scroll_response['hits']['hits']

    # Insert remaining
    if batch:
        bulk_body = []
        for doc in batch:
            bulk_body.append({"index": {"_index": doc["_index"], "_id": doc["_id"]}})
            bulk_body.append(doc["_source"])
        client.bulk(body=bulk_body, refresh=True)

    client.clear_scroll(scroll_id=scroll_id)
    client.indices.refresh(index=output_collection)

    final_count = client.count(index=output_collection)['count']
    elapsed = time.time() - start_time

    if verbose:
        print(f"  Matched: {final_count:,} documents")
        print(f"  Time: {elapsed:.1f}s")

    return {
        "source_total": source_count,
        "subset_total": final_count,
        "matched_filters": len(matched_items),
        "elapsed_seconds": elapsed,
        "output_collection": output_collection
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create subset indexes from main indexes (CLIP and OpenSearch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create both CLIP and OpenSearch subsets
    python create_subset_index.py --list best_books.txt --name best_books

    # CLIP only
    python create_subset_index.py --list albums.txt --name albums --clip-only

    # OpenSearch only
    python create_subset_index.py --list magazines.txt --name mags --opensearch-only

List file format (one per line, partial match):
    encyclopedia of monsters
    world atlas
    life magazine
        """
    )

    parser.add_argument(
        "--list", "-l",
        required=True,
        help="Path to text file with book/folder names"
    )

    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name for the subset (used for both CLIP dir and OpenSearch collection)"
    )

    parser.add_argument(
        "--clip-source",
        default=DEFAULT_CLIP_SOURCE,
        help=f"Source CLIP index directory (default: {DEFAULT_CLIP_SOURCE})"
    )

    parser.add_argument(
        "--clip-output",
        default=DEFAULT_CLIP_OUTPUT,
        help=f"Output directory for CLIP subset (default: {DEFAULT_CLIP_OUTPUT})"
    )

    parser.add_argument(
        "--opensearch-source",
        default=DEFAULT_OPENSEARCH_SOURCE,
        help=f"Source OpenSearch collection (default: {DEFAULT_OPENSEARCH_SOURCE})"
    )

    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Only create CLIP/FAISS subset"
    )

    parser.add_argument(
        "--opensearch-only",
        action="store_true",
        help="Only create OpenSearch subset"
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

    verbose = not args.quiet
    do_clip = not args.opensearch_only
    do_opensearch = not args.clip_only

    print("=" * 60)
    print(f"CREATING SUBSET: {args.name}")
    print("=" * 60)
    print(f"Filter items: {len(filter_list)}")

    results = {}

    # CLIP/FAISS
    if do_clip:
        try:
            clip_output = os.path.join(args.clip_output, args.name)
            results['clip'] = create_clip_subset(
                source_dir=args.clip_source,
                filter_list=filter_list,
                output_dir=clip_output,
                verbose=verbose
            )
        except Exception as e:
            print(f"\n[CLIP] Error: {e}")
            results['clip'] = {"error": str(e)}

    # OpenSearch
    if do_opensearch:
        try:
            opensearch_name = f"{args.name}_visual"
            results['opensearch'] = create_opensearch_subset(
                source_collection=args.opensearch_source,
                filter_list=filter_list,
                output_collection=opensearch_name,
                verbose=verbose
            )
        except Exception as e:
            print(f"\n[OpenSearch] Error: {e}")
            results['opensearch'] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'clip' in results and 'error' not in results['clip']:
        r = results['clip']
        print(f"\nCLIP/FAISS:")
        print(f"  Source: {r['source_total']:,} → Subset: {r['subset_total']:,}")
        print(f"  Output: {r['output_dir']}")

    if 'opensearch' in results and 'error' not in results['opensearch']:
        r = results['opensearch']
        print(f"\nDINOv2/OpenSearch:")
        print(f"  Source: {r['source_total']:,} → Subset: {r['subset_total']:,}")
        print(f"  Collection: {r['output_collection']}")

    # Show missing items
    all_matched = set()
    for key in ['clip', 'opensearch']:
        if key in results and 'matched_filters' in results[key]:
            # We don't track individual items in summary, just count
            pass

    print(f"\nFilters matched: {results.get('clip', {}).get('matched_filters', '?')}/{len(filter_list)}")


if __name__ == "__main__":
    main()
