"""Test chunks 824 and 969 (other Encyclopedia of Monsters copies)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
import time
from collections import Counter
from disk_searcher import extract_disk_features, load_chunk_paths, resolve_path, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: {len(descriptors)} keypoints\n")

ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]

for chunk_num in [824, 969]:
    chunk_file = f"T:/faiss/disk_retrieval/chunks/chunk_{chunk_num:03d}.faiss"
    local = os.path.join(LOCAL_CHUNK_BUFFER, f"chunk_{chunk_num:03d}.faiss")
    load_file = local if os.path.exists(local) else chunk_file

    print(f"{'='*80}")
    print(f"CHUNK {chunk_num}")
    print(f"{'='*80}")

    t0 = time.time()
    index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
    paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
    print(f"Loaded: {index.ntotal:,} vectors in {time.time()-t0:.1f}s\n")

    for k in [5, 100, 1000, 10000]:
        distances, indices = index.search(descriptors, k)

        votes = Counter()
        for i in range(len(descriptors)):
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0 and distances[i][j] >= 0.7:
                    path = resolve_path(paths_or_ids, id_to_path, idx)
                    if path:
                        votes[path] += 1

        # Find any monster encyclopedia pages
        monster_pages = [(path, v) for path, v in votes.most_common()
                         if 'onster' in path.lower() and 'ncyclopedia' in path.lower()]

        top_path, top_v = votes.most_common(1)[0] if votes else ("", 0)
        top_short = os.path.basename(top_path)
        if len(top_short) > 50: top_short = top_short[:47] + "..."

        monster_total = sum(v for _, v in monster_pages)
        best_monster = monster_pages[0] if monster_pages else (None, 0)
        best_m_name = os.path.basename(best_monster[0]) if best_monster[0] else "-"

        print(f"  k={k:>5}: #1={top_short} ({top_v}v) | Monsters total={monster_total}v, best={best_m_name} ({best_monster[1]}v)")

        if k == 10000 and monster_pages:
            print(f"    Top 5 Monster pages:")
            for path, v in monster_pages[:5]:
                # find rank
                for rank, (p, _) in enumerate(votes.most_common(), 1):
                    if p == path:
                        break
                print(f"      #{rank} ({v}v): {os.path.basename(path)}")
            print(f"    Top 5 overall:")
            for rank, (path, v) in enumerate(votes.most_common(5), 1):
                print(f"      #{rank} ({v}v): {os.path.basename(path)}")

    del index
    print()
