#!/usr/bin/env python3
"""
Verify all images on NAS are indexed in OpenSearch.
Scans source directory and reports any missing images.
Can identify and delete duplicate images (same content, different path).
"""

import os
import hashlib
import argparse
import pyodbc
from opensearchpy import OpenSearch

VISUAL_INDEX = "board_games_visual"
FACES_INDEX = "board_games_faces"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

# SQL Server connection
SQL_CONNECTION = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=DeceptionLens;"
    "Trusted_Connection=yes;"
)


def get_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_original_path_for_hash(cursor, file_hash: str, collection: str) -> str:
    """Get the original file path that has this hash."""
    cursor.execute(
        "SELECT FilePath FROM ImageHashes WHERE FileHash = ? AND Collection = ?",
        (file_hash, collection)
    )
    row = cursor.fetchone()
    return row[0] if row else None


def scan_for_images(source_path: str) -> list:
    """Scan directory for all image files."""
    image_paths = []
    for root, _, files in os.walk(source_path):
        for file in files:
            if os.path.splitext(file.lower())[1] in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, file))
    return image_paths


def load_opensearch_paths(os_client, index_name: str) -> set:
    """Load all existing document IDs (file paths) from OpenSearch index."""
    paths = set()

    if not os_client.indices.exists(index=index_name):
        print(f"  Index {index_name} does not exist!")
        return paths

    query = {"query": {"match_all": {}}, "_source": False}

    try:
        response = os_client.search(
            index=index_name,
            body=query,
            scroll="2m",
            size=10000
        )

        scroll_id = response.get("_scroll_id")
        hits = response["hits"]["hits"]

        while hits:
            for hit in hits:
                paths.add(hit["_id"])

            response = os_client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]

        if scroll_id:
            os_client.clear_scroll(scroll_id=scroll_id)

    except Exception as e:
        print(f"Error loading OpenSearch paths: {e}")

    return paths


def main():
    parser = argparse.ArgumentParser(description="Verify indexing completeness and find duplicates")
    parser.add_argument("--source", required=True, help="Source directory to scan")
    parser.add_argument("--show-missing", action="store_true", help="Show list of missing files")
    parser.add_argument("--find-duplicates", action="store_true", help="Check if missing files are content duplicates")
    parser.add_argument("--delete-duplicates", action="store_true", help="DELETE duplicate files (requires --find-duplicates)")
    parser.add_argument("--limit", type=int, default=100, help="Limit files shown (default 100)")
    parser.add_argument("--collection", default="board_games_dino", help="Collection name for SQL lookup")
    args = parser.parse_args()

    if args.delete_duplicates and not args.find_duplicates:
        print("Error: --delete-duplicates requires --find-duplicates")
        return

    # Connect to OpenSearch
    os_client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_compress=True,
        use_ssl=False
    )

    # Connect to SQL Server (for duplicate detection)
    conn = None
    cursor = None
    if args.find_duplicates:
        try:
            conn = pyodbc.connect(SQL_CONNECTION)
            cursor = conn.cursor()
        except Exception as e:
            print(f"Warning: Could not connect to SQL Server: {e}")
            print("Duplicate detection will be skipped.")

    print(f"\n{'='*60}")
    print("INDEXING VERIFICATION")
    print(f"{'='*60}")
    print(f"Source: {args.source}")

    # Scan NAS for images
    print(f"\nScanning {args.source}...")
    all_images = scan_for_images(args.source)
    total_on_nas = len(all_images)
    print(f"  Found {total_on_nas:,} images on NAS")

    # Load OpenSearch paths
    print(f"\nLoading paths from OpenSearch...")
    visual_paths = load_opensearch_paths(os_client, VISUAL_INDEX)
    print(f"  Found {len(visual_paths):,} in {VISUAL_INDEX}")

    face_paths = load_opensearch_paths(os_client, FACES_INDEX)
    # Extract base image paths from face IDs (remove _face_N suffix)
    face_image_paths = set()
    for p in face_paths:
        if "_face_" in p:
            base_path = p.rsplit("_face_", 1)[0]
            face_image_paths.add(base_path)
        else:
            face_image_paths.add(p)
    print(f"  Found {len(face_paths):,} face embeddings from {len(face_image_paths):,} images in {FACES_INDEX}")

    # Find missing images
    nas_set = set(all_images)
    missing_visual = nas_set - visual_paths
    missing_faces = nas_set - face_image_paths

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total images on NAS:        {total_on_nas:,}")
    print(f"Total in visual index:      {len(visual_paths):,}")
    print(f"Missing from visual index:  {len(missing_visual):,}")
    print(f"Images with faces indexed:  {len(face_image_paths):,}")

    coverage = (len(visual_paths) / total_on_nas * 100) if total_on_nas > 0 else 0
    print(f"\nVisual coverage: {coverage:.2f}%")

    if len(missing_visual) == 0:
        print("\n*** ALL IMAGES ARE INDEXED! ***")
    else:
        print(f"\n*** {len(missing_visual):,} IMAGES NOT IN OPENSEARCH ***")

        if args.show_missing and not args.find_duplicates:
            print(f"\nMissing files (showing first {args.limit}):")
            for i, path in enumerate(sorted(missing_visual)[:args.limit]):
                print(f"  {path}")
            if len(missing_visual) > args.limit:
                print(f"  ... and {len(missing_visual) - args.limit:,} more")

    # Find duplicates
    if args.find_duplicates and cursor and len(missing_visual) > 0:
        print(f"\n{'='*60}")
        print("DUPLICATE DETECTION")
        print(f"{'='*60}")
        print(f"Checking {len(missing_visual):,} missing files for content duplicates...")
        print("(This will read and hash each file)")

        duplicates = []  # (duplicate_path, original_path, file_size)
        truly_missing = []
        errors = []

        from tqdm import tqdm
        for path in tqdm(sorted(missing_visual), desc="Checking duplicates"):
            try:
                file_hash = get_file_hash(path)
                original_path = get_original_path_for_hash(cursor, file_hash, args.collection)

                if original_path and original_path != path:
                    # This is a duplicate - same content exists at another path
                    file_size = os.path.getsize(path)
                    duplicates.append((path, original_path, file_size))
                else:
                    truly_missing.append(path)
            except Exception as e:
                errors.append((path, str(e)))

        total_dup_size = sum(d[2] for d in duplicates)

        print(f"\n{'='*60}")
        print("DUPLICATE ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Content duplicates found:   {len(duplicates):,}")
        print(f"Space used by duplicates:   {total_dup_size / (1024**3):.2f} GB")
        print(f"Truly missing (not dupes):  {len(truly_missing):,}")
        print(f"Errors (couldn't check):    {len(errors):,}")

        if duplicates:
            print(f"\nDuplicate files (showing first {args.limit}):")
            for i, (dup_path, orig_path, size) in enumerate(duplicates[:args.limit]):
                print(f"  DUP:  {dup_path}")
                print(f"  ORIG: {orig_path}")
                print(f"  SIZE: {size / 1024:.1f} KB")
                print()
            if len(duplicates) > args.limit:
                print(f"  ... and {len(duplicates) - args.limit:,} more duplicates")

        if truly_missing:
            print(f"\nTruly missing files (not duplicates, showing first {min(20, len(truly_missing))}):")
            for path in truly_missing[:20]:
                print(f"  {path}")
            if len(truly_missing) > 20:
                print(f"  ... and {len(truly_missing) - 20:,} more")

        # Delete duplicates if requested
        if args.delete_duplicates and duplicates:
            print(f"\n{'='*60}")
            print("DELETING DUPLICATES")
            print(f"{'='*60}")

            confirm = input(f"Are you sure you want to DELETE {len(duplicates):,} duplicate files? (yes/no): ")
            if confirm.lower() == "yes":
                deleted = 0
                delete_errors = 0
                for dup_path, orig_path, size in tqdm(duplicates, desc="Deleting"):
                    try:
                        os.remove(dup_path)
                        deleted += 1
                    except Exception as e:
                        delete_errors += 1
                        print(f"  Error deleting {dup_path}: {e}")

                print(f"\nDeleted: {deleted:,} files")
                print(f"Errors:  {delete_errors:,} files")
                print(f"Space freed: {total_dup_size / (1024**3):.2f} GB")
            else:
                print("Deletion cancelled.")

    # Cleanup
    if conn:
        conn.close()


if __name__ == "__main__":
    main()
