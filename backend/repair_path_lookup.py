"""
Repair path_lookup.json after consolidation resume bug.

The second consolidation run started path_to_id from 0, overwriting the original
path_lookup.json. This script reconstructs the correct mapping by replaying the
ID assignment in the same order the consolidation used (sorted book names).

Per-book shards may be on T: or S: (some were moved to free space).

Steps:
1. Load consolidation state to get all processed books
2. Find each book's paths.json on T: or S:
3. Replay ID assignment in sorted order (matching original consolidation)
4. Rewrite all chunk_*_ids.npy files with corrected IDs
5. Save merged path_lookup.json
"""

import os
import json
import numpy as np
import time
import sys

# Per-book shard locations (check both)
SHARD_DIRS = [
    r"T:\faiss\disk_retrieval\books",
    r"S:\faiss\disk_retrieval\books",
]

CHUNK_IDS_DIR = r"D:\faiss\disk_retrieval\chunk_ids"
STATE_FILE = r"D:\faiss\disk_retrieval\consolidation_state.json"


def find_book_paths_json(book_name):
    """Find a book's paths.json on T: or S:."""
    for shard_dir in SHARD_DIRS:
        paths_file = os.path.join(shard_dir, book_name, "paths.json")
        if os.path.exists(paths_file):
            return paths_file
    return None


def main():
    print("=" * 70)
    print("  REPAIR PATH_LOOKUP.JSON")
    print("=" * 70)
    print()

    # Load consolidation state
    if not os.path.exists(STATE_FILE):
        print(f"ERROR: State file not found: {STATE_FILE}")
        sys.exit(1)

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    processed_books = sorted(state.get('processed_books', []))
    print(f"  Books in consolidation state: {len(processed_books):,}")

    # Find all chunk ID files
    id_files = sorted([f for f in os.listdir(CHUNK_IDS_DIR)
                       if f.startswith("chunk_") and f.endswith("_ids.npy")])
    print(f"  Chunk ID files: {len(id_files)}")

    # Backup current path_lookup
    lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    if os.path.exists(lookup_file):
        backup = lookup_file + ".backup"
        if not os.path.exists(backup):
            with open(lookup_file, 'r') as f:
                data = f.read()
            with open(backup, 'w') as f:
                f.write(data)
            print(f"  Backed up existing path_lookup to .backup")

    # Step 1: Replay ID assignment for ALL books in sorted order
    # The consolidation processes books in sorted order and assigns IDs
    # as it encounters new paths
    print()
    print("  Step 1: Replaying ID assignment from per-book shards...")
    print()

    path_to_id = {}
    next_id = 0
    books_found = 0
    books_missing = 0
    missing_books = []

    start = time.time()
    for i, book in enumerate(processed_books):
        elapsed = time.time() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 1
        eta = (len(processed_books) - i - 1) / rate / 60
        pct = (i + 1) / len(processed_books) * 100
        print(f"\r    [{i+1:,}/{len(processed_books):,}] {pct:5.1f}% | "
              f"Paths: {len(path_to_id):,} | "
              f"ETA: {eta:.1f}m   ", end="", flush=True)

        paths_file = find_book_paths_json(book)
        if paths_file is None:
            books_missing += 1
            missing_books.append(book)
            continue

        try:
            with open(paths_file, 'r') as f:
                paths = json.load(f)

            for path in paths:
                if path not in path_to_id:
                    path_to_id[path] = next_id
                    next_id += 1

            books_found += 1
        except Exception as e:
            print(f"\n    ERROR reading {book}: {e}")
            books_missing += 1
            missing_books.append(book)

    print(f"\r    Done! {books_found:,} books, {len(path_to_id):,} unique paths"
          f"                    ")
    if books_missing:
        print(f"    WARNING: {books_missing:,} books not found on T: or S:")
        for b in missing_books[:10]:
            print(f"      {b}")
        if len(missing_books) > 10:
            print(f"      ... and {len(missing_books) - 10} more")

    # Step 2: Build the new path_lookup (list indexed by ID)
    print()
    print("  Step 2: Building new path_lookup.json...")

    max_pid = max(path_to_id.values(), default=-1)
    id_to_path = [''] * (max_pid + 1)
    for path, pid in path_to_id.items():
        if pid < 0:
            raise ValueError(f"Negative path id for {path}: {pid}")
        id_to_path[pid] = path

    with open(lookup_file, 'w') as f:
        json.dump(id_to_path, f)

    lookup_size = os.path.getsize(lookup_file) / (1024**2)
    print(f"    Saved: {len(id_to_path):,} paths ({lookup_size:.1f} MB)")

    # Step 3: Rewrite all chunk ID files
    # Each chunk's _ids.npy contains compact IDs that map into path_lookup.
    # We need to rebuild these from the per-book paths in processing order.
    print()
    print("  Step 3: Rebuilding chunk ID files...")
    print()

    # We need to know which books went into which chunk and in what order.
    # The consolidation processes books in sorted order and fills chunks sequentially.
    # We replay this to regenerate correct IDs per chunk.

    # First, figure out how many vectors each chunk has (from existing FAISS chunks)
    chunk_sizes = {}
    for id_file in id_files:
        chunk_name = id_file.replace("_ids.npy", "")
        ids = np.load(os.path.join(CHUNK_IDS_DIR, id_file))
        chunk_sizes[chunk_name] = len(ids)

    print(f"    Total vectors across {len(chunk_sizes)} chunks: "
          f"{sum(chunk_sizes.values()):,}")

    # Now replay consolidation: process books in sorted order, assign IDs,
    # and fill chunks with the same vector counts
    MAX_VECTORS_PER_CHUNK = 21_000_000  # Must match consolidation config

    chunk_num = 1
    current_ids = []
    chunks_written = 0

    for i, book in enumerate(processed_books):
        pct3 = (i + 1) / len(processed_books) * 100
        print(f"\r    Replaying book {i+1:,}/{len(processed_books):,} {pct3:5.1f}% "
              f"(chunk {chunk_num})   ", end="", flush=True)

        paths_file = find_book_paths_json(book)
        if paths_file is None:
            continue

        try:
            with open(paths_file, 'r') as f:
                paths = json.load(f)
        except Exception:
            continue

        # Build IDs for this book's vectors (same as consolidation does)
        book_ids = []
        for path in paths:
            book_ids.append(path_to_id[path])

        # Split across chunk boundaries (same logic as consolidation)
        offset = 0
        while offset < len(book_ids):
            space = MAX_VECTORS_PER_CHUNK - len(current_ids)
            take = min(space, len(book_ids) - offset)

            current_ids.extend(book_ids[offset:offset + take])
            offset += take

            if len(current_ids) >= MAX_VECTORS_PER_CHUNK:
                # Save this chunk's IDs
                ids_file = os.path.join(CHUNK_IDS_DIR,
                                        f"chunk_{chunk_num:03d}_ids.npy")
                np.save(ids_file, np.array(current_ids, dtype=np.int32))
                chunks_written += 1
                chunk_num += 1
                current_ids = []

    # Save final partial chunk
    if current_ids:
        ids_file = os.path.join(CHUNK_IDS_DIR,
                                f"chunk_{chunk_num:03d}_ids.npy")
        np.save(ids_file, np.array(current_ids, dtype=np.int32))
        chunks_written += 1

    print(f"\r    Rewrote {chunks_written} chunk ID files"
          f"                              ")

    # Verify
    print()
    print("  Verifying...")
    total_vectors = 0
    for id_file in sorted(os.listdir(CHUNK_IDS_DIR)):
        if id_file.startswith("chunk_") and id_file.endswith("_ids.npy"):
            ids = np.load(os.path.join(CHUNK_IDS_DIR, id_file))
            max_id = ids.max() if len(ids) > 0 else 0
            if max_id >= len(id_to_path):
                print(f"    ERROR: {id_file} has ID {max_id} but path_lookup "
                      f"only has {len(id_to_path)} entries!")
            total_vectors += len(ids)

    print(f"    Total vectors in ID files: {total_vectors:,}")
    print(f"    Path lookup entries: {len(id_to_path):,}")

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print("  REPAIR COMPLETE!")
    print("=" * 70)
    print(f"  Books processed: {books_found:,}")
    print(f"  Unique paths: {len(path_to_id):,}")
    print(f"  Chunks rewritten: {chunks_written}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print()


if __name__ == '__main__':
    main()
