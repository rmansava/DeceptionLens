"""Sweep large k values on chunk 184 for the dino image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
import time
from collections import Counter
from disk_searcher import extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

threshold = 0.7
TARGET_PAGE = "page210.jpg"

# Load query
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: 2023-5.jpg ({len(descriptors)} keypoints)\n")

# Load chunk
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
print(f"Chunk 184: {index.ntotal:,} vectors\n")

print(f"{'k':>6} | {'Search':>7} | {'Votes':>7} | {'Total':>7} | {'Page210':>8} | {'Rank':>5} | {'#1 Result':>50} | {'#1 Votes':>8}")
print("-" * 120)

for k in [1000, 5000, 7500, 10000, 15000, 20000]:
    t0 = time.time()
    distances, indices = index.search(descriptors, k)
    search_time = time.time() - t0

    t1 = time.time()
    votes = Counter()
    for i in range(len(descriptors)):
        for j in range(k):
            idx = indices[i][j]
            if idx >= 0 and distances[i][j] >= threshold:
                path = resolve_path(paths_or_ids, id_to_path, idx)
                if path:
                    votes[path] += 1
    vote_time = time.time() - t1

    # Find page 210
    page210_votes = 0
    page210_rank = "-"
    for rank, (path, v) in enumerate(votes.most_common(), 1):
        if TARGET_PAGE in path and 'onster' in path.lower():
            page210_votes = v
            page210_rank = f"#{rank}"
            break

    # Top result
    top_path, top_votes = votes.most_common(1)[0] if votes else ("none", 0)
    top_short = os.path.basename(top_path)
    if len(top_short) > 50:
        top_short = top_short[:47] + "..."

    total_matches = sum(1 for i in range(len(descriptors)) for j in range(k)
                        if indices[i][j] >= 0 and distances[i][j] >= threshold)

    print(f"{k:>6} | {search_time:>5.1f}s | {vote_time:>5.1f}s | {total_matches:>7,} | {page210_votes:>8} | {page210_rank:>5} | {top_short:>50} | {top_votes:>8}")

print("\nDone.")
