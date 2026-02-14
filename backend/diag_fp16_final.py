"""Final FP16 diagnostic: compare FP32 vs FP16 Encyclopedia of Monsters votes using fuzzy matching."""
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
BATCH = 4_000_000

# Load query
print("Loading query...")
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"  {len(descriptors)} keypoints")

# Load chunk 184
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file

print(f"Loading chunk 184 from {'buffer' if os.path.exists(local) else 'NAS'}...")
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(index.ntotal, index.d)
print(f"  {index.ntotal:,} vectors")

# Show what path ID 495272 resolves to
ids = np.load(os.path.join(ids_dir, "chunk_184_ids.npy"))
# Find first position of ID 495272
pos = np.where(ids == 495272)[0][0]
resolved = resolve_path(ids, id_to_path, pos)
print(f"\n  ID 495272 resolves to: {resolved}")
print(f"  (Using id_to_path[495272] = {id_to_path[495272]})")


def batched_search(all_vectors, descriptors, k, dtype):
    """Batched GPU search, returns (distances, indices) as numpy."""
    n = all_vectors.shape[0]
    q = torch.from_numpy(np.ascontiguousarray(descriptors)).to(device='cuda', dtype=dtype)
    floor = -2.0 if dtype == torch.float16 else -1e9
    rd = torch.full((len(descriptors), k), floor, dtype=dtype, device='cuda')
    ri = torch.full((len(descriptors), k), -1, dtype=torch.long, device='cuda')

    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        db = torch.from_numpy(np.ascontiguousarray(all_vectors[s:e])).to(device='cuda', dtype=dtype)
        scores = torch.mm(q, db.t())
        bk = min(k, e - s)
        bs, bi = scores.topk(bk, dim=1)
        bi += s
        cs = torch.cat([rd, bs], dim=1)
        ci = torch.cat([ri, bi], dim=1)
        tk, tp = cs.topk(k, dim=1)
        rd = tk
        ri = ci.gather(1, tp)
        del db, scores, bs, bi, cs, ci

    d = rd.cpu().numpy()
    i = ri.cpu().numpy()
    del q, rd, ri
    torch.cuda.empty_cache()
    return d, i


def count_and_report(label, distances, indices):
    votes = Counter()
    total = 0
    enc_monster = 0
    for i in range(len(descriptors)):
        for j in range(k):
            idx = indices[i][j]
            if idx >= 0 and float(distances[i][j]) >= threshold:
                total += 1
                path = resolve_path(paths_or_ids, id_to_path, idx)
                if path:
                    votes[path] += 1
                    if 'onster' in path.lower() and 'ncyclopedia' in path.lower():
                        enc_monster += 1

    print(f"\n  {label}:")
    print(f"    Total >= {threshold}: {total}")
    print(f"    Enc of Monsters: {enc_monster} votes")
    print(f"    Top 5:")
    for path, v in votes.most_common(5):
        short = os.path.basename(path)
        is_target = " <-- TARGET" if 'onster' in path.lower() and 'ncyclopedia' in path.lower() else ""
        print(f"      {v:4d}: {short}{is_target}")
    return votes, total, enc_monster


# FAISS CPU
print("\n" + "=" * 60)
t0 = time.time()
d_faiss, i_faiss = index.search(descriptors, k)
print(f"FAISS CPU: {time.time()-t0:.1f}s")
count_and_report("FAISS CPU (k=5)", d_faiss, i_faiss)

# GPU FP32
print("\n" + "=" * 60)
t0 = time.time()
d32, i32 = batched_search(all_vectors, descriptors, k, torch.float32)
print(f"GPU FP32: {time.time()-t0:.1f}s")
v32, _, enc32 = count_and_report("GPU FP32 (k=5)", d32, i32)

# GPU FP16
print("\n" + "=" * 60)
t0 = time.time()
d16, i16 = batched_search(all_vectors, descriptors, k, torch.float16)
print(f"GPU FP16: {time.time()-t0:.1f}s")
v16, _, enc16 = count_and_report("GPU FP16 (k=5)", d16, i16)

# Compare indices
print("\n" + "=" * 60)
print("INDEX COMPARISON")
print("=" * 60)

# FP32 vs FP16
diff_count = 0
lost_monster_votes = 0
gained_monster_votes = 0
for i in range(len(descriptors)):
    set32 = set(i32[i].tolist())
    set16 = set(i16[i].tolist())
    if set32 != set16:
        diff_count += 1
        # Check if any lost indices were Encyclopedia of Monsters
        lost = set32 - set16
        gained = set16 - set32
        for idx in lost:
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path and 'onster' in path.lower():
                lost_monster_votes += 1
        for idx in gained:
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path and 'onster' in path.lower():
                gained_monster_votes += 1

print(f"  FP32 vs FP16: {diff_count}/{len(descriptors)} keypoints have different top-{k}")
print(f"  Enc of Monsters votes lost by FP16:   {lost_monster_votes}")
print(f"  Enc of Monsters votes gained by FP16: {gained_monster_votes}")
print(f"  Net change: {gained_monster_votes - lost_monster_votes}")

# FAISS vs FP32
faiss_diff = 0
for i in range(len(descriptors)):
    if set(i_faiss[i].tolist()) != set(i32[i].tolist()):
        faiss_diff += 1
print(f"  FAISS vs FP32: {faiss_diff}/{len(descriptors)} keypoints differ")

print(f"\n  SUMMARY:")
print(f"    FAISS CPU: {count_and_report.__code__.co_varnames} -- see above")
print(f"    Enc of Monsters: FAISS=? FP32={enc32} FP16={enc16}")
print("\nDone.")
