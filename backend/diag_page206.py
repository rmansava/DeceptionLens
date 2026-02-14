"""Check page206 votes across k values."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
from collections import Counter
from disk_searcher import extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: {len(descriptors)} keypoints\n")

chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)

# Search at max k once, then filter down
k = 20000
distances, indices = index.search(descriptors, k)
print(f"Searched with k={k}\n")

print(f"{'k':>6} | {'p206 votes':>10} | {'p206 rank':>9} | {'p210 votes':>10} | {'p210 rank':>9} | {'#1':>50} | {'#1 v':>5}")
print("-" * 120)

for test_k in [5, 20, 50, 100, 250, 500, 1000, 5000, 10000, 20000]:
    votes = Counter()
    for i in range(len(descriptors)):
        for j in range(test_k):
            idx = indices[i][j]
            if idx >= 0 and distances[i][j] >= 0.7:
                path = resolve_path(paths_or_ids, id_to_path, idx)
                if path:
                    votes[path] += 1

    p206_votes = p206_rank = p210_votes = p210_rank = 0
    for rank, (path, v) in enumerate(votes.most_common(), 1):
        bn = os.path.basename(path).lower()
        if 'page206' in bn and 'monster' in path.lower():
            p206_votes = v
            p206_rank = rank
        if 'page210' in bn and 'monster' in path.lower():
            p210_votes = v
            p210_rank = rank

    top_path, top_v = votes.most_common(1)[0] if votes else ("", 0)
    top_short = os.path.basename(top_path)
    if len(top_short) > 50: top_short = top_short[:47] + "..."

    print(f"{test_k:>6} | {p206_votes:>10} | {('#'+str(p206_rank)) if p206_rank else '-':>9} | {p210_votes:>10} | {('#'+str(p210_rank)) if p210_rank else '-':>9} | {top_short:>50} | {top_v:>5}")
