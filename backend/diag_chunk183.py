"""Search chunk 183 with the dino image to verify Encyclopedia of Monsters results."""
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

CHUNK = 183
k = 5
threshold = 0.7
BATCH = 4_000_000

# Load query
print("Loading query image (D:/trivpics/2023-5.jpg)...")
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    image_bytes = f.read()
descriptors = extract_disk_features(image_bytes)
print(f"  {len(descriptors)} keypoints extracted")

# Load chunk
chunk_file = f"T:/faiss/disk_retrieval/chunks/chunk_{CHUNK:03d}.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local_chunk = os.path.join(LOCAL_CHUNK_BUFFER, f"chunk_{CHUNK:03d}.faiss")
load_file = local_chunk if os.path.exists(local_chunk) else chunk_file

print(f"\nLoading chunk {CHUNK} from {load_file}...")
t0 = time.time()
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
print(f"  {index.ntotal:,} vectors (loaded in {time.time()-t0:.1f}s)")

# Load path mapping
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)

# GPU search
print(f"\nSearching on GPU (k={k}, threshold={threshold})...")
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(index.ntotal, index.d)
query_t = torch.from_numpy(np.ascontiguousarray(descriptors)).cuda().float()

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
search_time = time.time() - t0

distances = running_dist.cpu().numpy()
indices = running_idx.cpu().numpy()
print(f"  Search time: {search_time:.1f}s")

# Tally votes
votes = Counter()
total_matches = 0
for i in range(len(descriptors)):
    for j in range(k):
        idx = indices[i][j]
        if idx >= 0 and distances[i][j] >= threshold:
            total_matches += 1
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path:
                votes[path] += 1

print(f"\n  Total matches >= {threshold}: {total_matches:,}")
print(f"\n  Top 20 results:")
print(f"  {'Votes':>6}  Path")
print(f"  {'-----':>6}  ----")
for path, v in votes.most_common(20):
    marker = " <<<" if 'onster' in path.lower() and 'ncyclopedia' in path.lower() else ""
    print(f"  {v:6d}  {os.path.basename(path)}{marker}")

# Score stats
print(f"\n  Score range: {distances.min():.4f} to {distances.max():.4f}")
print(f"  Scores >= 0.8: {(distances >= 0.8).sum():,}")
print(f"  Scores >= 0.7: {(distances >= 0.7).sum():,}")
print(f"  Scores >= 0.6: {(distances >= 0.6).sum():,}")

del query_t, running_dist, running_idx
torch.cuda.empty_cache()
print("\nDone.")
