"""
Split oversized FAISS chunks into smaller pieces that fit in 32GB RAM.

Run this after consolidation to split any chunks > 30GB into ~20GB pieces.
"""

import faiss
import numpy as np
import json
import os
import shutil
from glob import glob
import time

# Config
NAS_CHUNKS_DIR = "T:/faiss/disk_retrieval/chunks"
LOCAL_BUFFER = "D:/faiss/disk_retrieval/chunk_buffer"
MAX_CHUNK_SIZE_GB = 30  # Split chunks larger than this
TARGET_CHUNK_SIZE_GB = 20  # Target size for split chunks

# 128 dims * 4 bytes = 512 bytes per vector
BYTES_PER_VECTOR = 128 * 4
TARGET_VECTORS = int(TARGET_CHUNK_SIZE_GB * (1024**3) / BYTES_PER_VECTOR)


def get_chunk_info(chunk_path):
    """Get chunk size and vector count."""
    size_gb = os.path.getsize(chunk_path) / (1024**3)
    # Estimate vectors from file size
    vectors = int(size_gb * (1024**3) / BYTES_PER_VECTOR)
    return size_gb, vectors


def split_chunk(chunk_num, dry_run=False):
    """Split a single oversized chunk into smaller pieces."""
    chunk_path = os.path.join(NAS_CHUNKS_DIR, f"chunk_{chunk_num:03d}.faiss")
    paths_path = os.path.join(NAS_CHUNKS_DIR, f"chunk_{chunk_num:03d}_paths.json")

    if not os.path.exists(chunk_path):
        print(f"  Chunk {chunk_num} not found")
        return []

    size_gb, est_vectors = get_chunk_info(chunk_path)

    if size_gb <= MAX_CHUNK_SIZE_GB:
        print(f"  Chunk {chunk_num}: {size_gb:.1f}GB - OK (under {MAX_CHUNK_SIZE_GB}GB)")
        return []

    print(f"  Chunk {chunk_num}: {size_gb:.1f}GB - OVERSIZED, splitting...")

    if dry_run:
        num_splits = int(np.ceil(size_gb / TARGET_CHUNK_SIZE_GB))
        print(f"    Would split into {num_splits} chunks of ~{TARGET_CHUNK_SIZE_GB}GB each")
        return []

    # Copy to local for fast processing
    os.makedirs(LOCAL_BUFFER, exist_ok=True)
    local_chunk = os.path.join(LOCAL_BUFFER, f"chunk_{chunk_num:03d}.faiss")
    local_paths = os.path.join(LOCAL_BUFFER, f"chunk_{chunk_num:03d}_paths.json")

    print(f"    Copying to local SSD...")
    t0 = time.time()
    shutil.copy2(chunk_path, local_chunk)
    shutil.copy2(paths_path, local_paths)
    print(f"    Copied in {time.time() - t0:.0f}s")

    # Load index and paths
    print(f"    Loading index...")
    t0 = time.time()
    index = faiss.read_index(local_chunk)
    with open(local_paths, 'r') as f:
        paths = json.load(f)
    print(f"    Loaded {index.ntotal:,} vectors in {time.time() - t0:.0f}s")

    # Extract all vectors
    print(f"    Extracting vectors...")
    t0 = time.time()
    vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * 128)
    vectors = vectors.reshape(-1, 128).copy()
    print(f"    Extracted in {time.time() - t0:.0f}s")

    del index

    # Calculate split points
    num_vectors = len(vectors)
    num_splits = int(np.ceil(num_vectors / TARGET_VECTORS))
    vectors_per_split = num_vectors // num_splits

    print(f"    Splitting into {num_splits} chunks of ~{vectors_per_split:,} vectors each")

    new_chunk_nums = []

    for i in range(num_splits):
        start_idx = i * vectors_per_split
        end_idx = start_idx + vectors_per_split if i < num_splits - 1 else num_vectors

        split_vectors = vectors[start_idx:end_idx]
        split_paths = paths[start_idx:end_idx]

        # Create new chunk number (e.g., 029 becomes 029a, 029b, etc.)
        # Actually, let's use decimal: 029.1, 029.2 -> chunk_029_001, chunk_029_002
        new_chunk_name = f"chunk_{chunk_num:03d}_{i+1:03d}"
        new_chunk_nums.append(new_chunk_name)

        print(f"    Creating {new_chunk_name} ({len(split_vectors):,} vectors)...")
        t0 = time.time()

        # Create new index
        new_index = faiss.IndexFlatIP(128)
        new_index.add(split_vectors)

        # Save locally first
        local_new_chunk = os.path.join(LOCAL_BUFFER, f"{new_chunk_name}.faiss")
        local_new_paths = os.path.join(LOCAL_BUFFER, f"{new_chunk_name}_paths.json")

        faiss.write_index(new_index, local_new_chunk)
        with open(local_new_paths, 'w') as f:
            json.dump(split_paths, f)

        # Move to NAS
        nas_new_chunk = os.path.join(NAS_CHUNKS_DIR, f"{new_chunk_name}.faiss")
        nas_new_paths = os.path.join(NAS_CHUNKS_DIR, f"{new_chunk_name}_paths.json")

        shutil.move(local_new_chunk, nas_new_chunk)
        shutil.move(local_new_paths, nas_new_paths)

        new_size = os.path.getsize(nas_new_chunk) / (1024**3)
        print(f"      Saved {new_size:.1f}GB in {time.time() - t0:.0f}s")

        del new_index

    # Remove original oversized chunk
    print(f"    Removing original oversized chunk...")
    os.remove(chunk_path)
    os.remove(paths_path)

    # Cleanup local buffer
    if os.path.exists(local_chunk):
        os.remove(local_chunk)
    if os.path.exists(local_paths):
        os.remove(local_paths)

    print(f"    Done! Split chunk {chunk_num} into {num_splits} pieces")
    return new_chunk_nums


def main():
    print()
    print("=" * 70)
    print("  SPLIT LARGE CHUNKS")
    print("=" * 70)
    print()
    print(f"  Source:     {NAS_CHUNKS_DIR}")
    print(f"  Max size:   {MAX_CHUNK_SIZE_GB}GB (chunks larger will be split)")
    print(f"  Target:     {TARGET_CHUNK_SIZE_GB}GB per split chunk")
    print()

    # Find all chunks
    chunk_files = sorted(glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*.faiss")))
    # Filter to only original chunks (not already split ones like chunk_029_001)
    original_chunks = [f for f in chunk_files if len(os.path.basename(f).replace('.faiss', '').split('_')) == 2]

    print(f"  Found {len(original_chunks)} original chunks")
    print()

    # Check which chunks are oversized
    print("-" * 70)
    print("  Scanning for oversized chunks...")
    print()

    oversized = []
    for chunk_file in original_chunks:
        chunk_num = int(os.path.basename(chunk_file).replace('chunk_', '').replace('.faiss', ''))
        size_gb, _ = get_chunk_info(chunk_file)

        if size_gb > MAX_CHUNK_SIZE_GB:
            oversized.append((chunk_num, size_gb))
            print(f"  chunk_{chunk_num:03d}: {size_gb:.1f}GB - OVERSIZED")

    if not oversized:
        print("  No oversized chunks found! All chunks are under 30GB.")
        return

    print()
    print(f"  Found {len(oversized)} oversized chunks to split")
    print()
    print("-" * 70)
    print("  Splitting oversized chunks...")
    print()

    all_new_chunks = []
    for chunk_num, size_gb in oversized:
        new_chunks = split_chunk(chunk_num)
        all_new_chunks.extend(new_chunks)
        print()

    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Split {len(oversized)} oversized chunks into {len(all_new_chunks)} smaller chunks")
    print()


if __name__ == "__main__":
    main()
