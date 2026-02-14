"""Compare FAISS CPU search vs PyTorch GPU search on chunk 184 for the dino image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import faiss
import time
from collections import Counter
from disk_searcher import extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

k = 5
threshold = 0.7

# Load query
print("Loading query image...")
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    image_bytes = f.read()
descriptors = extract_disk_features(image_bytes)
print(f"  {len(descriptors)} keypoints")

# Load chunk
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local_chunk = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local_chunk if os.path.exists(local_chunk) else chunk_file

print(f"Loading chunk 184...")
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
print(f"  {index.ntotal:,} vectors")

# --- FAISS CPU search ---
print(f"\n{'='*60}")
print("FAISS CPU index.search()")
print(f"{'='*60}")
t0 = time.time()
distances_faiss, indices_faiss = index.search(descriptors, k)
faiss_time = time.time() - t0
print(f"  Time: {faiss_time:.1f}s")

votes_faiss = Counter()
total_faiss = 0
target_votes_faiss = 0
for i in range(len(descriptors)):
    for j in range(k):
        idx = indices_faiss[i][j]
        if idx >= 0 and distances_faiss[i][j] >= threshold:
            total_faiss += 1
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path and 'ncyclopedia' in path.lower() and 'onster' in path.lower():
                target_votes_faiss += 1
            if path:
                votes_faiss[path] += 1

print(f"  Total matches >= {threshold}: {total_faiss:,}")
print(f"  Encyclopedia of Monsters votes: {target_votes_faiss}")
print(f"  Top 5:")
for path, v in votes_faiss.most_common(5):
    print(f"    {v:4d}: {os.path.basename(path)}")

# Score stats
print(f"  Score range: {distances_faiss.min():.4f} to {distances_faiss.max():.4f}")
print(f"  Scores >= 0.7: {(distances_faiss >= 0.7).sum()}")
print(f"  Scores >= 0.6: {(distances_faiss >= 0.6).sum()}")

# --- PyTorch GPU FP32 ---
print(f"\n{'='*60}")
print("PyTorch GPU FP32 (batched matmul)")
print(f"{'='*60}")
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(index.ntotal, index.d)
query_t = torch.from_numpy(np.ascontiguousarray(descriptors)).cuda().float()

BATCH = 4_000_000
running_dist = torch.full((len(descriptors), k), -1e9, device='cuda')
running_idx = torch.full((len(descriptors), k), -1, dtype=torch.long, device='cuda')

t0 = time.time()
for start in range(0, index.ntotal, BATCH):
    end = min(start + BATCH, index.ntotal)
    db = torch.from_numpy(np.ascontiguousarray(all_vectors[start:end])).cuda().float()
    scores = torch.mm(query_t, db.t())
    bk = min(k, end - start)
    bs, bi = scores.topk(bk, dim=1)
    bi += start
    cs = torch.cat([running_dist, bs], dim=1)
    ci = torch.cat([running_idx, bi], dim=1)
    tk, tp = cs.topk(k, dim=1)
    running_dist = tk
    running_idx = ci.gather(1, tp)
    del db, scores, bs, bi, cs, ci
gpu_time = time.time() - t0

dist_gpu = running_dist.cpu().numpy()
idx_gpu = running_idx.cpu().numpy()
print(f"  Time: {gpu_time:.1f}s")

votes_gpu = Counter()
total_gpu = 0
for i in range(len(descriptors)):
    for j in range(k):
        idx = idx_gpu[i][j]
        if idx >= 0 and dist_gpu[i][j] >= threshold:
            total_gpu += 1
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path:
                votes_gpu[path] += 1

print(f"  Total matches >= {threshold}: {total_gpu:,}")
print(f"  Top 5:")
for path, v in votes_gpu.most_common(5):
    print(f"    {v:4d}: {os.path.basename(path)}")

# Compare
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"  FAISS matches: {total_faiss}  |  GPU matches: {total_gpu}")
print(f"  Same top-k indices: ", end="")
same = np.sort(indices_faiss, axis=1) == np.sort(idx_gpu, axis=1)
print(f"{same.all(axis=1).sum()}/{len(descriptors)} keypoints identical")

# Check if indices differ
diffs = 0
for i in range(len(descriptors)):
    faiss_set = set(indices_faiss[i])
    gpu_set = set(idx_gpu[i])
    if faiss_set != gpu_set:
        diffs += 1
        if diffs <= 5:
            print(f"  KP {i}: FAISS={sorted(faiss_set)}, GPU={sorted(gpu_set)}")
            print(f"    FAISS scores: {distances_faiss[i]}")
            print(f"    GPU scores:   {dist_gpu[i]}")
print(f"  Total keypoints with different top-{k}: {diffs}/{len(descriptors)}")

# Now try with higher k
for test_k in [20, 50, 100]:
    print(f"\n{'='*60}")
    print(f"FAISS CPU with k={test_k}")
    print(f"{'='*60}")
    t0 = time.time()
    d_big, i_big = index.search(descriptors, test_k)
    print(f"  Time: {time.time()-t0:.1f}s")

    votes_big = Counter()
    total_big = 0
    target_big = 0
    for i in range(len(descriptors)):
        for j in range(test_k):
            idx = i_big[i][j]
            if idx >= 0 and d_big[i][j] >= threshold:
                total_big += 1
                path = resolve_path(paths_or_ids, id_to_path, idx)
                if path and 'ncyclopedia' in path.lower() and 'onster' in path.lower():
                    target_big += 1
                if path:
                    votes_big[path] += 1

    print(f"  Total matches >= {threshold}: {total_big:,}")
    print(f"  Encyclopedia of Monsters votes: {target_big}")
    print(f"  Top 5:")
    for path, v in votes_big.most_common(5):
        print(f"    {v:4d}: {os.path.basename(path)}")

del query_t, running_dist, running_idx
torch.cuda.empty_cache()
print("\nDone.")
