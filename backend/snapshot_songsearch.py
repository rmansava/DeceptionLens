"""
Snapshot script for songsearch OpenSearch index.

Usage:
    python snapshot_songsearch.py

Creates snapshots in the /songsearch-snapshots repo (D:\opensearch-songsearch),
then copies them to T: drive destination.

Prerequisites:
    Docker volume mount: -v D:\opensearch-songsearch:/songsearch-snapshots
"""

import requests
import time
import sys
import shutil
import os
from datetime import datetime

OPENSEARCH_HOST = "http://localhost:9200"

# Use repo mounted in Docker
REPO_NAME = "songsearch-repo"
REPO_PATH = "/songsearch-snapshots"  # Maps to D:\opensearch-songsearch in Docker
SOURCE_PATH = r"D:\opensearch-songsearch"  # Windows path to same location

# Snapshot configuration
INDEX_NAME = "songsearch"
SNAPSHOT_PREFIX = "songsearch"
DEST_PATH = r"T:\opensearch-songsearch-backup"


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


def ensure_repository() -> bool:
    """Ensure the snapshot repository exists."""
    # Check if repo exists
    resp = requests.get(f"{OPENSEARCH_HOST}/_snapshot/{REPO_NAME}")
    if resp.status_code == 200:
        print(f"Repository '{REPO_NAME}' exists")
        return True

    # Create repo
    print(f"Creating repository '{REPO_NAME}'...")
    resp = requests.put(
        f"{OPENSEARCH_HOST}/_snapshot/{REPO_NAME}",
        json={
            "type": "fs",
            "settings": {
                "location": REPO_PATH
            }
        },
        headers={"Content-Type": "application/json"}
    )

    if resp.status_code == 200:
        print(f"Repository '{REPO_NAME}' created")
        return True
    else:
        print(f"ERROR creating repository: {resp.text}")
        print(f"\nMake sure Docker has volume mount:")
        print(f"  -v {SOURCE_PATH}:{REPO_PATH}")
        return False


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

    snapshot_src = source_base

    if not os.path.exists(snapshot_src):
        print(f"  ERROR: Source path not found: {snapshot_src}")
        return False

    # Create dest if needed
    os.makedirs(dest_path, exist_ok=True)

    try:
        # Get snapshot info to find the index UUID
        resp = requests.get(f"{OPENSEARCH_HOST}/_snapshot/{repo_name}/{snapshot_name}")
        if resp.status_code != 200:
            print(f"  ERROR: Could not get snapshot info")
            return False

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
    print("OpenSearch Songsearch Index Snapshot Script")
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

    # Ensure repo exists
    if not ensure_repository():
        sys.exit(1)

    print()
    print("-" * 60)
    print(f"Processing: {INDEX_NAME}")
    print("-" * 60)

    # Check if index exists
    if not check_index_exists(INDEX_NAME):
        print(f"  ERROR: Index '{INDEX_NAME}' does not exist")
        sys.exit(1)

    doc_count = get_index_doc_count(INDEX_NAME)
    print(f"  Index has {doc_count:,} documents")

    snapshot_name = f"{SNAPSHOT_PREFIX}-{timestamp}"

    # Create snapshot
    if not create_snapshot(REPO_NAME, snapshot_name, INDEX_NAME):
        print("FAILED: Snapshot creation failed")
        sys.exit(1)

    # Copy to destination
    if copy_snapshot_to_dest(REPO_NAME, snapshot_name, SOURCE_PATH, DEST_PATH):
        print()
        print("=" * 60)
        print("SUCCESS!")
        print(f"  Snapshot: {snapshot_name}")
        print(f"  Documents: {doc_count:,}")
        print(f"  Backed up to: {DEST_PATH}")
        print("=" * 60)
    else:
        print("PARTIAL: Snapshot created but copy failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
