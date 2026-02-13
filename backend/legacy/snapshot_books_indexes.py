"""
Snapshot script for dinov2-books and faces-books OpenSearch indexes.

Usage:
    python snapshot_books_indexes.py

Creates snapshots in the existing /snapshots repo (D:\opensearch-books),
then copies them to T: drive destinations.
"""

import requests
import time
import sys
import shutil
import os
from datetime import datetime

OPENSEARCH_HOST = "http://localhost:9200"

# Use existing repo that's already mounted in Docker
REPO_NAME = "books"
REPO_PATH = "/snapshots"  # Maps to D:\opensearch-books in Docker
SOURCE_PATH = "D:\\opensearch-books"  # Windows path to same location

# Snapshot configurations
SNAPSHOTS = [
    {
        "index": "dinov2-books",
        "snapshot_prefix": "dinov2-books",
        "dest_path": "T:\\opensearch-dino-books",
    },
    {
        "index": "faces-books",
        "snapshot_prefix": "faces-books",
        "dest_path": "T:\\opensearch-faces-books",
    },
]


def check_index_exists(index_name: str) -> bool:
    """Check if an index exists."""
    resp = requests.head(f"{OPENSEARCH_HOST}/{index_name}")
    return resp.status_code == 200


def get_index_doc_count(index_name: str) -> int:
    """Get document count for an index."""
    resp = requests.get(f"{OPENSEARCH_HOST}/{index_name}/_count")
    if resp.status_code == 200:
        return resp.json().get("count", 0)
    return 0


def create_snapshot(repo_name: str, snapshot_name: str, index_name: str) -> bool:
    """Create a snapshot and wait for completion."""
    print(f"Creating snapshot '{snapshot_name}' of index '{index_name}'...")

    start_time = time.time()

    resp = requests.put(
        f"{OPENSEARCH_HOST}/_snapshot/{repo_name}/{snapshot_name}?wait_for_completion=true",
        json={
            "indices": index_name,
            "ignore_unavailable": True,
            "include_global_state": False
        },
        headers={"Content-Type": "application/json"},
        timeout=7200  # 2 hour timeout for large indexes
    )

    elapsed = time.time() - start_time

    if resp.status_code == 200:
        result = resp.json()
        snapshot_info = result.get("snapshot", {})
        state = snapshot_info.get("state", "UNKNOWN")
        shards = snapshot_info.get("shards", {})

        if state == "SUCCESS":
            print(f"  Snapshot completed successfully in {elapsed:.1f} seconds")
            print(f"  Shards: {shards.get('successful', 0)}/{shards.get('total', 0)} successful")
            return True
        else:
            print(f"  WARNING: Snapshot state is '{state}'")
            failures = snapshot_info.get("failures", [])
            if failures:
                print(f"  Failures: {failures}")
            return False
    else:
        print(f"  ERROR: Failed to create snapshot: {resp.text}")
        return False


def copy_snapshot_to_dest(repo_name: str, snapshot_name: str, source_base: str, dest_path: str) -> bool:
    """Copy snapshot files to destination."""
    print(f"Copying snapshot to {dest_path}...")

    # Snapshot files are in: source_base/indices/<index_uuid>/
    # But we need to copy the whole snapshot structure
    snapshot_src = source_base

    if not os.path.exists(snapshot_src):
        print(f"  ERROR: Source path not found: {snapshot_src}")
        return False

    # Create dest if needed
    os.makedirs(dest_path, exist_ok=True)

    # Copy relevant snapshot files
    # The snapshot data is organized by snapshot name and index
    try:
        # Get snapshot info to find the index UUID
        resp = requests.get(f"{OPENSEARCH_HOST}/_snapshot/{repo_name}/{snapshot_name}")
        if resp.status_code != 200:
            print(f"  ERROR: Could not get snapshot info")
            return False

        snapshot_info = resp.json().get("snapshots", [{}])[0]
        indices = snapshot_info.get("indices", [])

        # Copy the snapshot metadata files
        files_to_copy = []
        for f in os.listdir(snapshot_src):
            if f.startswith("snap-") or f.startswith("index-") or f == "index.latest":
                files_to_copy.append(f)

        for f in files_to_copy:
            src = os.path.join(snapshot_src, f)
            dst = os.path.join(dest_path, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied: {f}")

        # Copy the indices directory
        indices_src = os.path.join(snapshot_src, "indices")
        indices_dst = os.path.join(dest_path, "indices")
        if os.path.exists(indices_src):
            if os.path.exists(indices_dst):
                shutil.rmtree(indices_dst)
            shutil.copytree(indices_src, indices_dst)
            print(f"  Copied: indices/")

        return True

    except Exception as e:
        print(f"  ERROR copying files: {e}")
        return False


def main():
    print("=" * 60)
    print("OpenSearch Books Index Snapshot Script")
    print("=" * 60)
    print()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Check OpenSearch connectivity
    try:
        resp = requests.get(f"{OPENSEARCH_HOST}/_cluster/health", timeout=5)
        if resp.status_code != 200:
            print("ERROR: Cannot connect to OpenSearch")
            sys.exit(1)
        health = resp.json()
        print(f"OpenSearch cluster: {health.get('cluster_name')} ({health.get('status')})")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Cannot connect to OpenSearch: {e}")
        sys.exit(1)

    # Verify repo exists
    resp = requests.get(f"{OPENSEARCH_HOST}/_snapshot/{REPO_NAME}")
    if resp.status_code != 200:
        print(f"ERROR: Repository '{REPO_NAME}' not found")
        sys.exit(1)

    print(f"Using repository: {REPO_NAME}")
    print()

    results = []

    for config in SNAPSHOTS:
        index_name = config["index"]
        snapshot_name = f"{config['snapshot_prefix']}-{timestamp}"
        dest_path = config["dest_path"]

        print("-" * 60)
        print(f"Processing: {index_name}")
        print("-" * 60)

        # Check if index exists
        if not check_index_exists(index_name):
            print(f"  SKIPPING: Index '{index_name}' does not exist")
            results.append((index_name, "SKIPPED", "Index not found"))
            continue

        doc_count = get_index_doc_count(index_name)
        print(f"  Index has {doc_count:,} documents")

        # Create snapshot
        if not create_snapshot(REPO_NAME, snapshot_name, index_name):
            results.append((index_name, "FAILED", "Snapshot creation failed"))
            continue

        # Copy to destination
        if copy_snapshot_to_dest(REPO_NAME, snapshot_name, SOURCE_PATH, dest_path):
            results.append((index_name, "SUCCESS", f"Copied to {dest_path}"))
        else:
            results.append((index_name, "PARTIAL", "Snapshot created but copy failed"))

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for index_name, status, message in results:
        print(f"  {index_name}: {status} - {message}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
