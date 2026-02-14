"""
Diagnostic: Compare FP16 vs FP32 search on a single chunk.
Tests whether FP16 precision is dropping borderline matches below the 0.7 threshold.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import faiss
import time
from collections import Counter
from db_helper import get_connection
from disk_searcher import extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

k = 5
threshold = 0.7
BATCH_SIZE = 4_000_000  # Same as production


def gpu_search_single(all_vectors, descriptors, k, dtype):
    """Search all DB vectors against query descriptors in batches. Returns (distances, indices)."""
    n_vectors = all_vectors.shape[0]
    query = torch.from_numpy(np.ascontiguousarray(descriptors)).to(device='cuda', dtype=dtype)
    n_kp = len(descriptors)

    floor = -2.0 if dtype == torch.float16 else -1e9
    running_dist = torch.full((n_kp, k), floor, dtype=dtype, device='cuda')
    running_idx = torch.full((n_kp, k), -1, dtype=torch.long, device='cuda')

    for start in range(0, n_vectors, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_vectors)
        db_slice = np.ascontiguousarray(all_vectors[start:end])
        db_tensor = torch.from_numpy(db_slice).to(device='cuda', dtype=dtype)
        db_t = db_tensor.t()

        scores = torch.mm(query, db_t)
        batch_k = min(k, end - start)
        batch_scores, batch_idx = scores.topk(batch_k, dim=1)
        batch_idx += start

        combined_scores = torch.cat([running_dist, batch_scores], dim=1)
        combined_idx = torch.cat([running_idx, batch_idx], dim=1)
        topk_scores, topk_pos = combined_scores.topk(k, dim=1)
        running_dist = topk_scores
        running_idx = combined_idx.gather(1, topk_pos)

        del db_tensor, db_t, scores, batch_scores, batch_idx, combined_scores, combined_idx

    dist_np = running_dist.cpu().numpy()
    idx_np = running_idx.cpu().numpy()
    del query, running_dist, running_idx
    torch.cuda.empty_cache()
    return dist_np, idx_np


def count_votes(distances, indices, descriptors, paths_or_ids, id_to_path, thresh):
    """Count votes from search results."""
    votes = Counter()
    total_above = 0
    for i in range(len(descriptors)):
        for j in range(k):
            idx = indices[i][j]
            if idx >= 0 and float(distances[i][j]) >= thresh:
                total_above += 1
                path = resolve_path(paths_or_ids, id_to_path, idx)
                if path:
                    votes[path] += 1
    return votes, total_above


# -- Step 1: Find Encyclopedia of Monsters page 210 paths --
print("=" * 70)
print("  FP16 vs FP32 DIAGNOSTIC")
print("=" * 70)

conn = get_connection()
cursor = conn.cursor()
cursor.execute(
    "SELECT CompactId, ImagePath FROM DiskPathLookup "
    "WHERE Collection = 'books' AND ImagePath LIKE '%Encyclopedia of Monsters%page210.jpg'"
)
target_paths = []
for row in cursor.fetchall():
    print(f"  Target: ID {row[0]} -> {row[1]}")
    target_paths.append(row[1])
cursor.close()
conn.close()

if not target_paths:
    print("  ERROR: No Encyclopedia of Monsters page 210 found in DB")
    sys.exit(1)

# -- Step 2: Load query image --
query_path = "D:/trivpics/2023-5.jpg"
print(f"\n  Query image: {query_path}")
with open(query_path, 'rb') as f:
    image_bytes = f.read()

print("  Extracting DISK features...")
descriptors = extract_disk_features(image_bytes)
print(f"  Extracted {len(descriptors)} keypoints")

# -- Step 3: Load chunk 184 --
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]

local_chunk = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
if os.path.exists(local_chunk):
    load_file = local_chunk
    print(f"\n  Loading chunk from local buffer: {local_chunk}")
else:
    load_file = chunk_file
    print(f"\n  Loading chunk from NAS: {chunk_file}")

load_start = time.time()
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
print(f"  Loaded: {index.ntotal:,} vectors in {time.time() - load_start:.1f}s")

n_vectors = index.ntotal
dim = index.d
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(n_vectors, dim)

# -- Step 4: Search with FP32 --
print(f"\n{'-' * 70}")
print("  TEST 1: FP32 (original precision)")
print(f"{'-' * 70}")

t0 = time.time()
dist_fp32, idx_fp32 = gpu_search_single(all_vectors, descriptors, k, torch.float32)
fp32_time = time.time() - t0
print(f"  Search time: {fp32_time:.1f}s")

votes_fp32, total_above_fp32 = count_votes(dist_fp32, idx_fp32, descriptors, paths_or_ids, id_to_path, threshold)

print(f"  Total matches above {threshold}: {total_above_fp32:,}")
print(f"  Unique paths with votes: {len(votes_fp32):,}")

for tp in target_paths:
    v = votes_fp32.get(tp, 0)
    short = os.path.basename(tp)
    print(f"  Enc. of Monsters: {v} votes  ({short})")

print(f"\n  Top 10 results (FP32):")
for path, v in votes_fp32.most_common(10):
    short = os.path.basename(path)
    print(f"    {v:4d} votes: {short}")

# Score distribution near threshold
near = ((dist_fp32 >= 0.68) & (dist_fp32 < 0.72)).sum()
just_above = ((dist_fp32 >= 0.70) & (dist_fp32 < 0.71)).sum()
just_below = ((dist_fp32 >= 0.69) & (dist_fp32 < 0.70)).sum()
print(f"\n  Score distribution near threshold (top-{k} only):")
print(f"    0.68-0.72: {near:,} scores")
print(f"    0.69-0.70 (just below): {just_below:,} scores")
print(f"    0.70-0.71 (just above): {just_above:,} scores")

# -- Step 5: Search with FP16 --
print(f"\n{'-' * 70}")
print("  TEST 2: FP16 (current production setting)")
print(f"{'-' * 70}")

t0 = time.time()
dist_fp16, idx_fp16 = gpu_search_single(all_vectors, descriptors, k, torch.float16)
fp16_time = time.time() - t0
print(f"  Search time: {fp16_time:.1f}s")

votes_fp16, total_above_fp16 = count_votes(dist_fp16, idx_fp16, descriptors, paths_or_ids, id_to_path, threshold)

print(f"  Total matches above {threshold}: {total_above_fp16:,}")
print(f"  Unique paths with votes: {len(votes_fp16):,}")

for tp in target_paths:
    v = votes_fp16.get(tp, 0)
    short = os.path.basename(tp)
    print(f"  Enc. of Monsters: {v} votes  ({short})")

print(f"\n  Top 10 results (FP16):")
for path, v in votes_fp16.most_common(10):
    short = os.path.basename(path)
    print(f"    {v:4d} votes: {short}")

# -- Step 6: FP16 with lowered thresholds --
for lowered in [0.69, 0.68, 0.65]:
    print(f"\n{'-' * 70}")
    print(f"  TEST: FP16 with threshold={lowered}")
    print(f"{'-' * 70}")

    votes_low, total_low = count_votes(dist_fp16, idx_fp16, descriptors, paths_or_ids, id_to_path, lowered)

    print(f"  Total matches above {lowered}: {total_low:,}")
    enc_low = sum(votes_low.get(tp, 0) for tp in target_paths)
    print(f"  Enc. of Monsters total: {enc_low} votes")

    print(f"  Top 5:")
    for path, v in votes_low.most_common(5):
        short = os.path.basename(path)
        print(f"    {v:4d} votes: {short}")

# -- Summary --
print(f"\n{'=' * 70}")
print("  SUMMARY")
print(f"{'=' * 70}")
print(f"  FP32 speed: {fp32_time:.1f}s  |  FP16 speed: {fp16_time:.1f}s  |  Speedup: {fp32_time/fp16_time:.1f}x")
print(f"  FP32 total matches (>={threshold}): {total_above_fp32:,}")
print(f"  FP16 total matches (>={threshold}): {total_above_fp16:,}  ({total_above_fp16/max(1,total_above_fp32)*100:.0f}%)")

enc_fp32 = sum(votes_fp32.get(tp, 0) for tp in target_paths)
enc_fp16 = sum(votes_fp16.get(tp, 0) for tp in target_paths)
print(f"\n  Encyclopedia of Monsters votes (all 4 copies combined):")
print(f"    FP32 (threshold=0.7):  {enc_fp32}")
print(f"    FP16 (threshold=0.7):  {enc_fp16}")
print(f"{'=' * 70}")
