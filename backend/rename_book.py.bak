"""
Rename a book across all indexes (OpenSearch + FAISS).

Usage:
    python rename_book.py --old "Old Book Name" --new "New Book Name"
    python rename_book.py --old "Old Book Name" --new "New Book Name" --dry-run
    python rename_book.py --old "Old Book Name" --new "New Book Name" --rename-files

Options:
    --dry-run       Show what would be changed without making changes
    --rename-files  Also rename the actual folder on disk
"""

import argparse
import json
import os
import shutil
from opensearchpy import OpenSearch

# Configuration
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200

FAISS_PATHS = [
    "D:/faiss/books/paths.json",
]

OPENSEARCH_INDEXES = [
    "dinov2-books",
    "faces-books",
]

BOOKS_ROOT = "D:/books/pdf-images"


def get_opensearch_client():
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        timeout=300
    )


def find_keypoint_indexes(client, old_name: str) -> list:
    """Find any keypoint indexes that might match the book name."""
    # Keypoint indexes are named like: keypoint-index-books-{book-slug}
    slug = old_name.lower().replace(" ", "-").replace("_", "-")

    # Get all indexes
    indexes = client.cat.indices(format="json")
    matching = []

    for idx in indexes:
        idx_name = idx.get("index", "")
        if idx_name.startswith("keypoint-index-books-") and slug in idx_name.lower():
            matching.append(idx_name)

    return matching


def update_opensearch_index(client, index_name: str, old_name: str, new_name: str, dry_run: bool) -> dict:
    """Update book name in an OpenSearch index."""

    # Count matching documents
    count_query = {
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"book": old_name}},
                    {"match_phrase": {"metadata.book": old_name}},
                    {"wildcard": {"path": f"*{old_name}*"}}
                ],
                "minimum_should_match": 1
            }
        }
    }

    try:
        count_result = client.count(index=index_name, body=count_query)
        doc_count = count_result.get("count", 0)
    except Exception as e:
        return {"index": index_name, "status": "error", "message": str(e)}

    if doc_count == 0:
        return {"index": index_name, "status": "skipped", "message": "No matching documents"}

    if dry_run:
        return {"index": index_name, "status": "dry-run", "would_update": doc_count}

    # Update documents
    update_script = """
        if (ctx._source.book != null && ctx._source.book.contains(params.old_name)) {
            ctx._source.book = ctx._source.book.replace(params.old_name, params.new_name);
        }
        if (ctx._source.path != null && ctx._source.path.contains(params.old_name)) {
            ctx._source.path = ctx._source.path.replace(params.old_name, params.new_name);
        }
        if (ctx._source.metadata != null) {
            if (ctx._source.metadata.book != null && ctx._source.metadata.book.contains(params.old_name)) {
                ctx._source.metadata.book = ctx._source.metadata.book.replace(params.old_name, params.new_name);
            }
            if (ctx._source.metadata.path != null && ctx._source.metadata.path.contains(params.old_name)) {
                ctx._source.metadata.path = ctx._source.metadata.path.replace(params.old_name, params.new_name);
            }
            if (ctx._source.metadata.filename != null && ctx._source.metadata.filename.contains(params.old_name)) {
                ctx._source.metadata.filename = ctx._source.metadata.filename.replace(params.old_name, params.new_name);
            }
        }
        if (ctx._source.filename != null && ctx._source.filename.contains(params.old_name)) {
            ctx._source.filename = ctx._source.filename.replace(params.old_name, params.new_name);
        }
    """

    update_body = {
        "query": count_query["query"],
        "script": {
            "source": update_script,
            "params": {
                "old_name": old_name,
                "new_name": new_name
            }
        }
    }

    try:
        result = client.update_by_query(
            index=index_name,
            body=update_body,
            wait_for_completion=True,
            refresh=True
        )

        updated = result.get("updated", 0)
        failures = result.get("failures", [])

        if failures:
            return {"index": index_name, "status": "partial", "updated": updated, "failures": len(failures)}

        return {"index": index_name, "status": "success", "updated": updated}

    except Exception as e:
        return {"index": index_name, "status": "error", "message": str(e)}


def update_faiss_paths(paths_file: str, old_name: str, new_name: str, dry_run: bool) -> dict:
    """Update book name in FAISS paths.json file."""

    if not os.path.exists(paths_file):
        return {"file": paths_file, "status": "skipped", "message": "File not found"}

    try:
        with open(paths_file, "r") as f:
            paths = json.load(f)
    except Exception as e:
        return {"file": paths_file, "status": "error", "message": f"Failed to read: {e}"}

    # Count and update paths
    updated_count = 0
    new_paths = []

    for path in paths:
        if old_name in path:
            new_path = path.replace(old_name, new_name)
            new_paths.append(new_path)
            updated_count += 1
        else:
            new_paths.append(path)

    if updated_count == 0:
        return {"file": paths_file, "status": "skipped", "message": "No matching paths"}

    if dry_run:
        return {"file": paths_file, "status": "dry-run", "would_update": updated_count}

    # Backup original
    backup_file = paths_file + ".backup"
    try:
        shutil.copy2(paths_file, backup_file)
    except Exception as e:
        return {"file": paths_file, "status": "error", "message": f"Failed to backup: {e}"}

    # Write updated paths
    try:
        with open(paths_file, "w") as f:
            json.dump(new_paths, f)

        return {"file": paths_file, "status": "success", "updated": updated_count, "backup": backup_file}
    except Exception as e:
        # Restore backup
        shutil.copy2(backup_file, paths_file)
        return {"file": paths_file, "status": "error", "message": f"Failed to write: {e}"}


def rename_folder(old_name: str, new_name: str, dry_run: bool) -> dict:
    """Rename the actual book folder on disk."""

    old_path = os.path.join(BOOKS_ROOT, old_name)
    new_path = os.path.join(BOOKS_ROOT, new_name)

    if not os.path.exists(old_path):
        return {"folder": old_path, "status": "skipped", "message": "Folder not found"}

    if os.path.exists(new_path):
        return {"folder": new_path, "status": "error", "message": "Target folder already exists"}

    if dry_run:
        return {"folder": old_path, "status": "dry-run", "would_rename_to": new_path}

    try:
        os.rename(old_path, new_path)
        return {"folder": old_path, "status": "success", "renamed_to": new_path}
    except Exception as e:
        return {"folder": old_path, "status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Rename a book across all indexes")
    parser.add_argument("--old", required=True, help="Old book name")
    parser.add_argument("--new", required=True, help="New book name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    parser.add_argument("--rename-files", action="store_true", help="Also rename folder on disk")

    args = parser.parse_args()

    old_name = args.old
    new_name = args.new
    dry_run = args.dry_run
    rename_files = args.rename_files

    print("=" * 60)
    print("Book Rename Tool")
    print("=" * 60)
    print(f"Old name: {old_name}")
    print(f"New name: {new_name}")
    print(f"Dry run:  {dry_run}")
    print(f"Rename files: {rename_files}")
    print()

    results = []

    # 1. Rename folder first (if requested)
    if rename_files:
        print("-" * 60)
        print("STEP 1: Renaming folder on disk")
        print("-" * 60)
        result = rename_folder(old_name, new_name, dry_run)
        results.append(("Folder", result))
        print(f"  {result}")
        print()

    # 2. Update OpenSearch indexes
    print("-" * 60)
    print("STEP 2: Updating OpenSearch indexes")
    print("-" * 60)

    client = get_opensearch_client()

    # Main indexes
    for index_name in OPENSEARCH_INDEXES:
        print(f"  Processing {index_name}...")
        result = update_opensearch_index(client, index_name, old_name, new_name, dry_run)
        results.append((index_name, result))
        print(f"    {result}")

    # Keypoint indexes
    keypoint_indexes = find_keypoint_indexes(client, old_name)
    if keypoint_indexes:
        print(f"\n  Found {len(keypoint_indexes)} keypoint index(es):")
        for idx in keypoint_indexes:
            print(f"    - {idx}")
            # Note: Keypoint indexes are per-book, so renaming would require
            # creating a new index with the new name and reindexing
            if not dry_run:
                print(f"      WARNING: Keypoint index renaming not implemented.")
                print(f"      You may need to re-index this book for keypoints.")

    print()

    # 3. Update FAISS paths
    print("-" * 60)
    print("STEP 3: Updating FAISS paths.json")
    print("-" * 60)

    for paths_file in FAISS_PATHS:
        print(f"  Processing {paths_file}...")
        result = update_faiss_paths(paths_file, old_name, new_name, dry_run)
        results.append((paths_file, result))
        print(f"    {result}")

    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = 0
    error_count = 0
    skip_count = 0

    for name, result in results:
        status = result.get("status", "unknown")
        if status == "success":
            success_count += 1
            print(f"  ✓ {name}: {result.get('updated', 'done')}")
        elif status == "dry-run":
            print(f"  ~ {name}: would update {result.get('would_update', 'N/A')}")
        elif status == "skipped":
            skip_count += 1
            print(f"  - {name}: skipped ({result.get('message', '')})")
        else:
            error_count += 1
            print(f"  ✗ {name}: {result.get('message', 'failed')}")

    print()
    if dry_run:
        print("DRY RUN - No changes were made. Run without --dry-run to apply.")
    else:
        print(f"Done! Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
