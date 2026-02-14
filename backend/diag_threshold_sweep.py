"""Sweep threshold values and test score-weighted voting."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
import torch
import time
from collections import Counter
from disk_searcher import (
    extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER,
    GPU_SEARCH_USE_FP16, GPU_SEARCH_BATCH_SIZE, GPU_SEARCH_MAX_SCORES_BYTES
)
from collections_config import COLLECTIONS

print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load query
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: {len(descriptors)} keypoints")

# Load chunk
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
print(f"Chunk 184: {index.ntotal:,} vectors\n")

n_vectors = index.ntotal
dim = index.d
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(n_vectors, dim)

dtype = torch.float16 if GPU_SEARCH_USE_FP16 else torch.float32
score_bytes = 2 if GPU_SEARCH_USE_FP16 else 4
batch_size = GPU_SEARCH_BATCH_SIZE
use_compact_ids = id_to_path is not None and len(id_to_path) > 0

q_tensor = torch.from_numpy(np.ascontiguousarray(descriptors)).to(device='cuda', dtype=dtype)

def resolve_votes(db_indices, vote_counts):
    """Convert DB indices + vote counts to a page Counter."""
    votes = Counter()
    if use_compact_ids:
        compact_ids = paths_or_ids[db_indices]
        valid_mask = (compact_ids >= 0) & (compact_ids < len(id_to_path))
        valid_cids = compact_ids[valid_mask]
        valid_counts = vote_counts[valid_mask]
        if len(valid_cids) > 0:
            unique_cids, inv = np.unique(valid_cids, return_inverse=True)
            grouped = np.zeros(len(unique_cids), dtype=np.float64)
            np.add.at(grouped, inv, valid_counts)
            for cid, count in zip(unique_cids, grouped):
                votes[id_to_path[int(cid)]] += count
    return votes

# Pre-compute all scores in batches, collecting per-vector max, sum, etc.
print("Computing scores against all DB vectors...\n")

# We'll collect votes for different strategies in one pass
thresholds = [0.7, 0.75, 0.8, 0.85]

# Strategy results: {strategy_name: Counter()}
all_results = {}
for t in thresholds:
    all_results[f"count_t{t}"] = Counter()      # binary vote count
    all_results[f"weighted_t{t}"] = Counter()    # score-weighted votes

current = 0
while current < n_vectors:
    end = min(current + batch_size, n_vectors)
    db_count = end - current

    db_slice = np.ascontiguousarray(all_vectors[current:end])
    db_tensor = torch.from_numpy(db_slice).to(device='cuda', dtype=dtype)
    db_t = db_tensor.t()

    max_qb = max(1, GPU_SEARCH_MAX_SCORES_BYTES // (db_count * score_bytes))

    # Accumulators per strategy
    count_accum = {t: torch.zeros(db_count, dtype=torch.int32, device='cuda') for t in thresholds}
    weighted_accum = {t: torch.zeros(db_count, dtype=torch.float32, device='cuda') for t in thresholds}

    for q_start in range(0, len(descriptors), max_qb):
        q_end = min(q_start + max_qb, len(descriptors))
        q_batch = q_tensor[q_start:q_end]

        scores = torch.mm(q_batch, db_t)

        for t in thresholds:
            mask = scores >= t
            count_accum[t] += mask.sum(dim=0).int()
            # Score-weighted: multiply scores by mask, sum across keypoints
            weighted_accum[t] += (scores * mask.float()).sum(dim=0)

        del scores

    # Resolve votes for each strategy
    for t in thresholds:
        # Binary count
        nonzero = count_accum[t].nonzero(as_tuple=True)[0]
        if len(nonzero) > 0:
            db_idx = (nonzero + current).cpu().numpy()
            counts = count_accum[t][nonzero].cpu().numpy().astype(np.float64)
            chunk_votes = resolve_votes(db_idx, counts)
            all_results[f"count_t{t}"].update(chunk_votes)

        # Weighted
        nonzero = (weighted_accum[t] > 0).nonzero(as_tuple=True)[0]
        if len(nonzero) > 0:
            db_idx = (nonzero + current).cpu().numpy()
            weights = weighted_accum[t][nonzero].cpu().numpy().astype(np.float64)
            chunk_votes = resolve_votes(db_idx, weights)
            all_results[f"weighted_t{t}"].update(chunk_votes)

    del db_tensor, db_t, count_accum, weighted_accum
    current = end
    print(f"  Processed {current:,}/{n_vectors:,}")

torch.cuda.empty_cache()

# Report
TARGET = 'page206'
TARGET_BOOK = 'onster'

print(f"\n{'Strategy':>20} | {'Total':>10} | {'Pages':>6} | {'p206 votes':>12} | {'p206 rank':>9} | {'#1':>50} | {'#1 v':>10}")
print("-" * 130)

for name, votes in sorted(all_results.items()):
    total = sum(votes.values())
    n_pages = len(votes)

    p206_v = 0
    p206_rank = "-"
    for rank, (path, v) in enumerate(votes.most_common(), 1):
        if TARGET in os.path.basename(path).lower() and TARGET_BOOK in path.lower():
            p206_v = v
            p206_rank = f"#{rank}"
            break

    top_path, top_v = votes.most_common(1)[0] if votes else ("", 0)
    top_short = os.path.basename(top_path)
    if len(top_short) > 50: top_short = top_short[:47] + "..."

    print(f"{name:>20} | {total:>10.0f} | {n_pages:>6} | {p206_v:>12.1f} | {p206_rank:>9} | {top_short:>50} | {top_v:>10.1f}")

# Show detailed results for the best strategy
print(f"\n\nBest strategies detail:")
for name in ["count_t0.85", "weighted_t0.85", "weighted_t0.8"]:
    votes = all_results[name]
    print(f"\n{'='*80}")
    print(f"Strategy: {name}")
    print(f"{'='*80}")
    print(f"Top 15:")
    for rank, (path, v) in enumerate(votes.most_common(15), 1):
        marker = " <== CORRECT PAGE" if TARGET in os.path.basename(path).lower() and TARGET_BOOK in path.lower() else ""
        print(f"  #{rank} ({v:.1f}): {os.path.basename(path)[:65]}{marker}")
