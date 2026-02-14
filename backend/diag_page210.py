"""How many votes does page 210 SPECIFICALLY get?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
from collections import Counter
from disk_searcher import extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

k = 5
threshold = 0.7

# Load query + chunk
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())

chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)

# FAISS search
distances, indices = index.search(descriptors, k)

# Count ALL votes
votes = Counter()
for i in range(len(descriptors)):
    for j in range(k):
        idx = indices[i][j]
        if idx >= 0 and distances[i][j] >= threshold:
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path:
                votes[path] += 1

# Show top 20 results
print("Top 20 results from chunk 184 (k=5, threshold=0.7):")
print("-" * 80)
for rank, (path, v) in enumerate(votes.most_common(20), 1):
    name = os.path.basename(path)
    marker = " <<<" if "page210" in name and "onster" in path.lower() else ""
    print(f"  #{rank:2d}  {v:4d} votes  {name}{marker}")

# Specifically check page 210
print(f"\nAll Encyclopedia of Monsters pages in results:")
for path, v in votes.most_common():
    if 'onster' in path.lower() and 'ncyclopedia' in path.lower():
        print(f"  {v:4d} votes  {os.path.basename(path)}")
