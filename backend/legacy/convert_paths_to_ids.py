"""
Convert chunk paths.json files (3.3 TB) to compact ID arrays (~95 GB).

Each paths.json stores a file path string per keypoint (~11,000x redundancy).
This script converts them to:
  - chunk_XXX_ids.npy: int32 array mapping keypoint index -> path ID
  - path_lookup.json: global list where index = path ID, value = path string

Input:  T:/faiss/disk_retrieval/chunks/chunk_XXX_paths.json  (2-6 GB each, 606 files)
Output: D:/faiss/disk_retrieval/chunk_ids/chunk_XXX_ids.npy  (~100-200 MB each)
        D:/faiss/disk_retrieval/chunk_ids/path_lookup.json    (~500 MB, one file)

Resumable: tracks progress in conversion_progress.txt
"""

import json
import numpy as np
import os
import sys
import time
from glob import glob

NAS_CHUNKS_DIR = "T:/faiss/disk_retrieval/chunks"
OUTPUT_DIR = "D:/faiss/disk_retrieval/chunk_ids"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "conversion_progress.txt")
LOOKUP_FILE = os.path.join(OUTPUT_DIR, "path_lookup.json")
PATH_TO_ID_FILE = os.path.join(OUTPUT_DIR, "path_to_id.json")


def load_progress():
    """Load set of completed chunk names."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_lookup(path_to_id):
    """Save lookup tables (both directions)."""
    # Save path_to_id for conversion
    with open(PATH_TO_ID_FILE, 'w') as f:
        json.dump(path_to_id, f)

    # Save id_to_path for search (list indexed by ID)
    id_to_path = [''] * len(path_to_id)
    for path, pid in path_to_id.items():
        id_to_path[pid] = path

    with open(LOOKUP_FILE, 'w') as f:
        json.dump(id_to_path, f)

    print(f"  Saved lookup: {len(path_to_id):,} unique paths "
          f"({os.path.getsize(LOOKUP_FILE) / 1e6:.1f} MB)")


def convert_chunk(paths_file, output_file, path_to_id, next_id):
    """Convert one chunk's paths.json to an ids.npy file."""
    # Load paths.json from NAS
    load_start = time.time()
    with open(paths_file, 'r') as f:
        paths = json.load(f)
    load_time = time.time() - load_start

    # Convert to IDs
    ids = np.empty(len(paths), dtype=np.int32)
    new_paths = 0
    for i, path in enumerate(paths):
        if path not in path_to_id:
            path_to_id[path] = next_id
            next_id += 1
            new_paths += 1
        ids[i] = path_to_id[path]

    # Save compact ID array
    np.save(output_file, ids)

    # Free memory
    unique_in_chunk = len(set(paths))
    entry_count = len(paths)
    del paths

    return entry_count, unique_in_chunk, new_paths, next_id, load_time


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all paths files
    paths_files = sorted(glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*_paths.json")))
    print(f"Found {len(paths_files)} chunk paths files")

    if not paths_files:
        print("No paths files found!")
        return

    # Load existing progress
    completed = load_progress()
    print(f"Already completed: {len(completed)} chunks")

    # Load or create path_to_id mapping
    if os.path.exists(PATH_TO_ID_FILE):
        print("Loading existing path_to_id mapping...")
        with open(PATH_TO_ID_FILE, 'r') as f:
            path_to_id = json.load(f)
        next_id = max(path_to_id.values()) + 1 if path_to_id else 0
        print(f"  Loaded {len(path_to_id):,} existing paths (next_id={next_id})")
    else:
        path_to_id = {}
        next_id = 0

    # Process each chunk
    total_entries = 0
    total_saved_bytes = 0
    start_time = time.time()
    remaining = [f for f in paths_files
                 if os.path.basename(f).replace('_paths.json', '') not in completed]

    print(f"Chunks to process: {len(remaining)}")
    print(f"{'='*70}")

    for i, paths_file in enumerate(remaining):
        chunk_name = os.path.basename(paths_file).replace('_paths.json', '')
        ids_file = os.path.join(OUTPUT_DIR, f"{chunk_name}_ids.npy")
        source_size = os.path.getsize(paths_file)

        print(f"\n[{i+1}/{len(remaining)}] {chunk_name} "
              f"({source_size / 1e9:.1f} GB from NAS)...")

        try:
            entry_count, unique_in_chunk, new_paths, next_id, load_time = \
                convert_chunk(paths_file, ids_file, path_to_id, next_id)

            ids_size = os.path.getsize(ids_file)
            total_entries += entry_count
            total_saved_bytes += (source_size - ids_size)
            ratio = source_size / ids_size if ids_size > 0 else 0

            print(f"  Loaded in {load_time:.1f}s | "
                  f"{entry_count:,} keypoints, {unique_in_chunk:,} unique pages")
            print(f"  {source_size/1e9:.1f} GB -> {ids_size/1e6:.0f} MB "
                  f"({ratio:.0f}x smaller) | {new_paths:,} new paths")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Mark as complete
        with open(PROGRESS_FILE, 'a') as f:
            f.write(f"{chunk_name}\n")
        completed.add(chunk_name)

        # Save lookup every 10 chunks
        if (i + 1) % 10 == 0:
            save_lookup(path_to_id)

        # ETA
        elapsed = time.time() - start_time
        chunks_done = i + 1
        avg_per_chunk = elapsed / chunks_done
        remaining_count = len(remaining) - chunks_done
        eta_seconds = remaining_count * avg_per_chunk
        eta_hours = eta_seconds / 3600

        print(f"  Total unique paths: {len(path_to_id):,} | "
              f"ETA: {eta_hours:.1f}h ({remaining_count} chunks left)")

    # Final save
    print(f"\n{'='*70}")
    print(f"Saving final path lookup...")
    save_lookup(path_to_id)

    elapsed = time.time() - start_time
    print(f"\nConversion complete!")
    print(f"  Chunks converted: {len(completed)}")
    print(f"  Unique paths: {len(path_to_id):,}")
    print(f"  Total keypoints: {total_entries:,}")
    print(f"  Space saved: {total_saved_bytes / 1e12:.2f} TB")
    print(f"  Time: {elapsed / 3600:.1f} hours")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
