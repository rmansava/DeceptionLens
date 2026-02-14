"""Show the #1 result path at k=10000."""
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

chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)

distances, indices = index.search(descriptors, 10000)

votes = Counter()
for i in range(len(descriptors)):
    for j in range(10000):
        idx = indices[i][j]
        if idx >= 0 and distances[i][j] >= 0.7:
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path:
                votes[path] += 1

for rank, (path, v) in enumerate(votes.most_common(5), 1):
    print(f"#{rank} ({v} votes): {path}")
