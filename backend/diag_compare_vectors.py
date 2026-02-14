"""Compare vectors between per-book shard and consolidated chunk for page 206."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
import json
import time
from collections import Counter
from disk_searcher import extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

# Load query
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: {len(descriptors)} keypoints\n")

# =====================================================
# Source 1: Per-book shard
# =====================================================
print("=" * 80)
print("PER-BOOK SHARD")
print("=" * 80)

shard_dir = "T:/faiss/disk_retrieval/books/Encyclopedia Of Monsters, The (ISBN 0816023034)"
shard_index = faiss.read_index(os.path.join(shard_dir, "index.faiss"), faiss.IO_FLAG_MMAP)
with open(os.path.join(shard_dir, "paths.json"), 'r') as f:
    shard_paths = json.load(f)

print(f"Shard: {shard_index.ntotal:,} vectors, {len(set(shard_paths)):,} unique pages")

# Count page 206 vectors
p206_indices_shard = [i for i, p in enumerate(shard_paths) if 'page206' in os.path.basename(p).lower()]
print(f"Page 206 vectors in shard: {len(p206_indices_shard):,}")

# Get actual vectors for page 206
shard_all_vecs = faiss.vector_to_array(shard_index.codes).view("float32").reshape(shard_index.ntotal, 128)
p206_vecs_shard = shard_all_vecs[p206_indices_shard]

# Check norms
norms = np.linalg.norm(p206_vecs_shard, axis=1)
print(f"Page 206 vector norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

# Compute scores against query
scores_shard = descriptors @ p206_vecs_shard.T  # (618, N_p206)
max_per_kp_shard = scores_shard.max(axis=1)  # best score per keypoint
kp_above_07_shard = (max_per_kp_shard >= 0.7).sum()
print(f"Keypoints with page206 match >= 0.7: {kp_above_07_shard}")
print(f"Max score: {max_per_kp_shard.max():.4f}")

# Search the shard and check page 206 votes
distances, indices = shard_index.search(descriptors, 5)
shard_votes = Counter()
p206_shard_votes = 0
for i in range(len(descriptors)):
    for j in range(5):
        idx = indices[i][j]
        if idx >= 0 and distances[i][j] >= 0.7:
            path = shard_paths[idx]
            shard_votes[path] += 1
            if 'page206' in os.path.basename(path).lower():
                p206_shard_votes += 1

print(f"\nk=5 search: page 206 = {p206_shard_votes} votes (rank ", end="")
for rank, (path, v) in enumerate(shard_votes.most_common(), 1):
    if 'page206' in os.path.basename(path).lower():
        print(f"#{rank})")
        break

# Show a few example shard paths
print(f"\nSample shard paths:")
for p in list(set(shard_paths))[:3]:
    print(f"  {p}")

del shard_index

# =====================================================
# Source 2: Consolidated chunk 184
# =====================================================
print(f"\n{'=' * 80}")
print("CONSOLIDATED CHUNK 184")
print("=" * 80)

chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
chunk_index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)

print(f"Chunk: {chunk_index.ntotal:,} vectors")

# Find page 206 vectors in chunk
p206_indices_chunk = []
chunk_all_vecs = faiss.vector_to_array(chunk_index.codes).view("float32").reshape(chunk_index.ntotal, 128)

# Check how many vectors belong to page 206
p206_count = 0
p206_sample_path = None
for i in range(chunk_index.ntotal):
    path = resolve_path(paths_or_ids, id_to_path, i)
    if path and 'page206' in os.path.basename(path).lower() and 'onster' in path.lower():
        p206_indices_chunk.append(i)
        p206_count += 1
        if p206_sample_path is None:
            p206_sample_path = path
    if i % 5_000_000 == 0 and i > 0:
        print(f"  Scanned {i:,}/{chunk_index.ntotal:,} vectors, found {p206_count} page206...")

print(f"Page 206 vectors in chunk: {p206_count:,}")
if p206_sample_path:
    print(f"Sample path: {p206_sample_path}")

if p206_count > 0:
    p206_vecs_chunk = chunk_all_vecs[p206_indices_chunk]

    # Check norms
    norms_chunk = np.linalg.norm(p206_vecs_chunk, axis=1)
    print(f"Page 206 vector norms: min={norms_chunk.min():.4f}, max={norms_chunk.max():.4f}, mean={norms_chunk.mean():.4f}")

    # Compute scores against query
    scores_chunk = descriptors @ p206_vecs_chunk.T
    max_per_kp_chunk = scores_chunk.max(axis=1)
    kp_above_07_chunk = (max_per_kp_chunk >= 0.7).sum()
    print(f"Keypoints with page206 match >= 0.7: {kp_above_07_chunk}")
    print(f"Max score: {max_per_kp_chunk.max():.4f}")

    # Compare vectors between shard and chunk
    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print("=" * 80)
    print(f"  Shard page206 vectors: {len(p206_indices_shard):,}")
    print(f"  Chunk page206 vectors: {p206_count:,}")
    print(f"  Shard kp >= 0.7: {kp_above_07_shard}")
    print(f"  Chunk kp >= 0.7: {kp_above_07_chunk}")
    print(f"  Shard max score: {max_per_kp_shard.max():.4f}")
    print(f"  Chunk max score: {max_per_kp_chunk.max():.4f}")

    if len(p206_indices_shard) == p206_count:
        # Check if vectors are identical
        # Sort both sets by first few values for comparison
        shard_sorted = p206_vecs_shard[np.lexsort(p206_vecs_shard[:, :3].T)]
        chunk_sorted = p206_vecs_chunk[np.lexsort(p206_vecs_chunk[:, :3].T)]
        if np.allclose(shard_sorted, chunk_sorted, atol=1e-6):
            print(f"  Vectors are IDENTICAL between shard and chunk")
        else:
            diff = np.abs(shard_sorted - chunk_sorted).max()
            print(f"  Vectors DIFFER! Max abs difference: {diff:.6f}")
    else:
        print(f"  DIFFERENT VECTOR COUNTS - can't directly compare")

    # Also check: how many total book vectors are in each?
    enc_monster_count_shard = shard_index.ntotal if hasattr(shard_index, 'ntotal') else len(p206_indices_shard)
    enc_monster_count_chunk = 0
    for i in range(min(100000, chunk_index.ntotal)):
        path = resolve_path(paths_or_ids, id_to_path, i)
        if path and 'onster' in path.lower() and 'ncyclopedia' in path.lower() and '0816023034' in path:
            enc_monster_count_chunk += 1
    if chunk_index.ntotal > 100000:
        # Extrapolate
        est = enc_monster_count_chunk * (chunk_index.ntotal / 100000)
        print(f"  Estimated Enc of Monsters vectors in chunk: ~{int(est):,} (sampled first 100K)")
    else:
        print(f"  Enc of Monsters vectors in chunk: {enc_monster_count_chunk:,}")
else:
    print("  NO PAGE 206 VECTORS FOUND IN CHUNK!")

del chunk_index
print("\nDone.")
