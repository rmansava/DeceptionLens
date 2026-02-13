"""
Restore board games indexes from NAS snapshots.

Restores dinov2-board_games and faces-board_games from T: drive NAS backups.

Usage:
    python restore_boardgames_indexes.py              # Restore both indexes
    python restore_boardgames_indexes.py --visual    # Visual only
    python restore_boardgames_indexes.py --faces     # Faces only
    python restore_boardgames_indexes.py --list      # List available snapshots

Prerequisites:
    1. Copy NAS snapshots to local staging area:
       - Copy T:\\opensearch-dino-boardgames\\* to D:\\opensearch-boardgames\\
       - Or add Docker volume mount for NAS paths

    2. Register snapshot repository (done automatically by this script)
"""

import requests
import sys
import os
import shutil
from datetime import datetime

OPENSEARCH_HOST = "http://localhost:9200"

# NAS snapshot locations
NAS_VISUAL = r"T:\opensearch-dino-boardgames"
NAS_FACES = r"T:\opensearch-faces-boardgames"

# Local staging (must be mounted in Docker)
LOCAL_STAGING = r"D:\opensearch-boardgames"
DOCKER_PATH = "/boardgames-snapshots"

# Note: Add this volume mount to Docker:
#   -v D:\opensearch-boardgames:/boardgames-snapshots

REPO_NAME = "boardgames"

# Index names
VISUAL_INDEX = "dinov2-board_games"
FACES_INDEX = "faces-board_games"


def check_opensearch():
    """Check OpenSearch connectivity."""
    try:
        resp = requests.get(f"{OPENSEARCH_HOST}/_cluster/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            print(f"OpenSearch: {health.get('cluster_name')} ({health.get('status')})")
            return True
    except Exception as e:
        print(f"ERROR: Cannot connect to OpenSearch: {e}")
    return False


def copy_from_nas(nas_path: str, local_path: str) -> bool:
    """Copy snapshot files from NAS to local staging."""
    if not os.path.exists(nas_path):
        print(f"  NAS path not found: {nas_path}")
        return False

    print(f"  Copying from {nas_path}...")
    print(f"  To: {local_path}")

    os.makedirs(local_path, exist_ok=True)

    try:
        # Copy all contents
        for item in os.listdir(nas_path):
            src = os.path.join(nas_path, item)
            dst = os.path.join(local_path, item)

            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"    Copied: {item}/")
            else:
                shutil.copy2(src, dst)
                print(f"    Copied: {item}")

        return True
    except Exception as e:
        print(f"  ERROR copying: {e}")
        return False


def register_repository() -> bool:
    """Register the snapshot repository."""
    print(f"\nRegistering repository '{REPO_NAME}'...")

    # Check if repo exists
    resp = requests.get(f"{OPENSEARCH_HOST}/_snapshot/{REPO_NAME}")
    if resp.status_code == 200:
        print(f"  Repository '{REPO_NAME}' already exists")
        return True

    # Create repo
    resp = requests.put(
        f"{OPENSEARCH_HOST}/_snapshot/{REPO_NAME}",
        json={
            "type": "fs",
            "settings": {
                "location": DOCKER_PATH
            }
        },
        headers={"Content-Type": "application/json"}
    )

    if resp.status_code == 200:
        print(f"  Repository '{REPO_NAME}' created")
        return True
    else:
        print(f"  ERROR creating repository: {resp.text}")
        print(f"\n  Make sure Docker has a volume mount:")
        print(f"    -v {LOCAL_STAGING}:{DOCKER_PATH}")
        return False


def list_snapshots() -> list:
    """List available snapshots."""
    resp = requests.get(f"{OPENSEARCH_HOST}/_snapshot/{REPO_NAME}/_all")
    if resp.status_code != 200:
        print(f"  ERROR listing snapshots: {resp.text}")
        return []

    snapshots = resp.json().get("snapshots", [])
    return snapshots


def find_latest_snapshot(prefix: str, snapshots: list) -> dict:
    """Find the latest snapshot with given prefix."""
    matching = [s for s in snapshots if s["snapshot"].startswith(prefix)]
    if not matching:
        return None
    # Sort by end_time descending
    matching.sort(key=lambda s: s.get("end_time", ""), reverse=True)
    return matching[0]


def restore_index(snapshot_name: str, index_name: str) -> bool:
    """Restore an index from a snapshot."""
    print(f"\nRestoring '{index_name}' from snapshot '{snapshot_name}'...")

    # Check if index already exists
    resp = requests.head(f"{OPENSEARCH_HOST}/{index_name}")
    if resp.status_code == 200:
        print(f"  Index '{index_name}' already exists")
        response = input(f"  Delete and restore? (y/n): ").strip().lower()
        if response != 'y':
            print("  Skipping...")
            return False

        # Delete existing index
        print(f"  Deleting existing index...")
        resp = requests.delete(f"{OPENSEARCH_HOST}/{index_name}")
        if resp.status_code != 200:
            print(f"  ERROR deleting index: {resp.text}")
            return False

    # Restore from snapshot
    resp = requests.post(
        f"{OPENSEARCH_HOST}/_snapshot/{REPO_NAME}/{snapshot_name}/_restore?wait_for_completion=true",
        json={
            "indices": index_name,
            "ignore_unavailable": True,
            "include_global_state": False
        },
        headers={"Content-Type": "application/json"},
        timeout=7200
    )

    if resp.status_code == 200:
        result = resp.json()
        shards = result.get("snapshot", {}).get("shards", {})
        print(f"  Restored successfully")
        print(f"  Shards: {shards.get('successful', 0)}/{shards.get('total', 0)}")

        # Get doc count
        count_resp = requests.get(f"{OPENSEARCH_HOST}/{index_name}/_count")
        if count_resp.status_code == 200:
            count = count_resp.json().get("count", 0)
            print(f"  Documents: {count:,}")

        return True
    else:
        print(f"  ERROR restoring: {resp.text}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Restore board games indexes from NAS snapshots")
    parser.add_argument("--visual", action="store_true", help="Restore visual index only")
    parser.add_argument("--faces", action="store_true", help="Restore faces index only")
    parser.add_argument("--list", action="store_true", help="List available snapshots")
    parser.add_argument("--skip-copy", action="store_true", help="Skip NAS copy (use existing local files)")
    args = parser.parse_args()

    print("=" * 60)
    print("Board Games Index Restore")
    print("=" * 60)

    if not check_opensearch():
        sys.exit(1)

    # Determine what to restore
    do_visual = not args.faces or args.visual
    do_faces = not args.visual or args.faces

    if not args.visual and not args.faces:
        do_visual = True
        do_faces = True

    # Step 1: Copy from NAS (unless skipped)
    if not args.skip_copy and not args.list:
        print("\n" + "-" * 60)
        print("Step 1: Copy snapshots from NAS")
        print("-" * 60)

        if do_visual and os.path.exists(NAS_VISUAL):
            copy_from_nas(NAS_VISUAL, LOCAL_STAGING)
        elif do_visual:
            print(f"  Visual NAS path not found: {NAS_VISUAL}")

        # Note: faces might be in separate snapshot or same location
        # For now, assume visual and faces are in same snapshot repo

    # Step 2: Register repository
    print("\n" + "-" * 60)
    print("Step 2: Register snapshot repository")
    print("-" * 60)

    if not os.path.exists(LOCAL_STAGING):
        print(f"  ERROR: Local staging not found: {LOCAL_STAGING}")
        print(f"  Copy snapshots from NAS first or create the directory")
        sys.exit(1)

    if not register_repository():
        print("\nTo fix this, add to your Docker OpenSearch container:")
        print(f"  -v {LOCAL_STAGING}:{DOCKER_PATH}")
        sys.exit(1)

    # Step 3: List snapshots
    print("\n" + "-" * 60)
    print("Step 3: Available snapshots")
    print("-" * 60)

    snapshots = list_snapshots()
    if not snapshots:
        print("  No snapshots found in repository")
        sys.exit(1)

    for snap in snapshots:
        name = snap["snapshot"]
        state = snap.get("state", "?")
        indices = ", ".join(snap.get("indices", []))
        end_time = snap.get("end_time", "")[:19] if snap.get("end_time") else "?"
        print(f"  {name} ({state}) - {indices} - {end_time}")

    if args.list:
        return

    # Step 4: Restore indexes
    print("\n" + "-" * 60)
    print("Step 4: Restore indexes")
    print("-" * 60)

    results = []

    if do_visual:
        # Find visual snapshot
        visual_snap = find_latest_snapshot("dinov2-board", snapshots)
        if not visual_snap:
            # Try without prefix - might be named differently
            visual_snap = next((s for s in snapshots if VISUAL_INDEX in s.get("indices", [])), None)

        if visual_snap:
            success = restore_index(visual_snap["snapshot"], VISUAL_INDEX)
            results.append((VISUAL_INDEX, "SUCCESS" if success else "FAILED"))
        else:
            print(f"  No snapshot found containing {VISUAL_INDEX}")
            results.append((VISUAL_INDEX, "NOT FOUND"))

    if do_faces:
        # Find faces snapshot
        faces_snap = find_latest_snapshot("faces-board", snapshots)
        if not faces_snap:
            faces_snap = next((s for s in snapshots if FACES_INDEX in s.get("indices", [])), None)

        if faces_snap:
            success = restore_index(faces_snap["snapshot"], FACES_INDEX)
            results.append((FACES_INDEX, "SUCCESS" if success else "FAILED"))
        else:
            print(f"  No snapshot found containing {FACES_INDEX}")
            results.append((FACES_INDEX, "NOT FOUND"))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for index_name, status in results:
        print(f"  {index_name}: {status}")

    print("\nDone!")


if __name__ == "__main__":
    main()
