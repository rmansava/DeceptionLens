"""Test threshold-only search on chunk 184 with the dino image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
import torch
import time
from collections import Counter
from disk_searcher import (
    extract_disk_features, load_chunk_paths, resolve_path,
    _gpu_threshold_vote_batch, _gpu_search_batch, LOCAL_CHUNK_BUFFER
)
from collections_config import COLLECTIONS

print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load query
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: 2023-5.jpg ({len(descriptors)} keypoints)\n")

# Load chunk 184
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
print(f"Chunk 184: {index.ntotal:,} vectors\n")

# =====================================================
# OLD WAY: top-k=5 with threshold filter
# =====================================================
print("=" * 80)
print("OLD: top-k=5 search + threshold=0.7 filter")
print("=" * 80)

t0 = time.time()
topk_results = _gpu_search_batch(index, [("dino", descriptors)], k=5)
distances, indices = topk_results["dino"]
topk_time = time.time() - t0

topk_votes = Counter()
for i in range(len(descriptors)):
    for j in range(5):
        idx = indices[i][j]
        if idx >= 0 and distances[i][j] >= 0.7:
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path:
                topk_votes[path] += 1

topk_matched = sum(1 for i in range(len(descriptors)) for j in range(5)
                   if indices[i][j] >= 0 and distances[i][j] >= 0.7)

print(f"  Time: {topk_time:.1f}s")
print(f"  Total matches: {topk_matched}")
print(f"  Unique pages: {len(topk_votes)}")

enc_votes_topk = 0
page206_rank_topk = "-"
page206_votes_topk = 0
for rank, (path, v) in enumerate(topk_votes.most_common(), 1):
    if 'onster' in path.lower() and 'ncyclopedia' in path.lower():
        enc_votes_topk += v
    if 'page206' in os.path.basename(path).lower() and 'onster' in path.lower():
        page206_rank_topk = f"#{rank}"
        page206_votes_topk = v

print(f"  Encyclopedia of Monsters total: {enc_votes_topk} votes")
print(f"  Page 206: rank {page206_rank_topk} ({page206_votes_topk} votes)")
print(f"\n  Top 10:")
for rank, (path, v) in enumerate(topk_votes.most_common(10), 1):
    print(f"    #{rank} ({v}v): {os.path.basename(path)[:70]}")

del topk_results
torch.cuda.synchronize()
torch.cuda.empty_cache()
del index

# =====================================================
# NEW WAY: threshold-only search
# =====================================================
print(f"\n{'=' * 80}")
print("NEW: threshold=0.7 (ALL above-threshold matches vote)")
print("=" * 80)

# Reload fresh
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)

t0 = time.time()
threshold_votes, match_counts = _gpu_threshold_vote_batch(
    index, [("dino", descriptors)], 0.7, paths_or_ids, id_to_path
)
threshold_time = time.time() - t0
votes = threshold_votes["dino"]
matched = match_counts["dino"]

print(f"  Time: {threshold_time:.1f}s")
print(f"  Total matches: {matched:,}")
print(f"  Unique pages: {len(votes):,}")

enc_votes_thresh = 0
page206_rank_thresh = "-"
page206_votes_thresh = 0
for rank, (path, v) in enumerate(votes.most_common(), 1):
    if 'onster' in path.lower() and 'ncyclopedia' in path.lower():
        enc_votes_thresh += v
    if 'page206' in os.path.basename(path).lower() and 'onster' in path.lower():
        page206_rank_thresh = f"#{rank}"
        page206_votes_thresh = v

print(f"  Encyclopedia of Monsters total: {enc_votes_thresh} votes")
print(f"  Page 206: rank {page206_rank_thresh} ({page206_votes_thresh} votes)")
print(f"\n  Top 20:")
for rank, (path, v) in enumerate(votes.most_common(20), 1):
    bn = os.path.basename(path)
    if len(bn) > 70: bn = bn[:67] + "..."
    print(f"    #{rank} ({v}v): {bn}")

# Find other Encyclopedia of Monsters pages in top results
print(f"\n  All Encyclopedia of Monsters pages with votes:")
monster_pages = [(path, v) for path, v in votes.most_common()
                 if 'onster' in path.lower() and 'ncyclopedia' in path.lower()]
for path, v in monster_pages[:15]:
    for rank, (p, _) in enumerate(votes.most_common(), 1):
        if p == path:
            break
    print(f"    #{rank} ({v}v): {os.path.basename(path)[:70]}")

# =====================================================
# Comparison
# =====================================================
print(f"\n{'=' * 80}")
print("COMPARISON")
print("=" * 80)
print(f"  {'':>30} | {'Top-k=5':>12} | {'Threshold':>12}")
print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}")
print(f"  {'Search time':>30} | {topk_time:>10.1f}s | {threshold_time:>10.1f}s")
print(f"  {'Total matches':>30} | {topk_matched:>12,} | {matched:>12,}")
print(f"  {'Unique pages':>30} | {len(topk_votes):>12,} | {len(votes):>12,}")
print(f"  {'Enc. of Monsters votes':>30} | {enc_votes_topk:>12,} | {enc_votes_thresh:>12,}")
print(f"  {'Page 206 votes':>30} | {page206_votes_topk:>12} | {page206_votes_thresh:>12}")
print(f"  {'Page 206 rank':>30} | {page206_rank_topk:>12} | {page206_rank_thresh:>12}")
