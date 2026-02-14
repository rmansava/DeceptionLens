"""Test per-book shard search: 1 book, 10 books, 50 books."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
import json
import torch
import time
import random
from collections import Counter
from disk_searcher import extract_disk_features

SHARDS_DIR = "T:/faiss/disk_retrieval/books"
TARGET_BOOK = "Encyclopedia Of Monsters, The (ISBN 0816023034)"
K = 5
THRESHOLD = 0.7

# Load query
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: {len(descriptors)} keypoints")
print(f"Shards dir: {SHARDS_DIR}")

# Get all available shards
all_shards = sorted([d for d in os.listdir(SHARDS_DIR)
                     if os.path.isdir(os.path.join(SHARDS_DIR, d))
                     and os.path.exists(os.path.join(SHARDS_DIR, d, "index.faiss"))
                     and os.path.exists(os.path.join(SHARDS_DIR, d, "paths.json"))])
print(f"Total shards with index.faiss: {len(all_shards)}")
print(f"Target book: {TARGET_BOOK}")
assert TARGET_BOOK in all_shards, f"Target book not found in shards!"

def search_shards(shard_names, descriptors, k, threshold, label):
    """Search a list of per-book shards and return aggregated page votes."""
    print(f"\n{'=' * 80}")
    print(f"{label}: Searching {len(shard_names)} book shard(s)")
    print(f"{'=' * 80}")

    all_votes = Counter()
    total_vectors = 0
    total_time = 0

    for i, shard_name in enumerate(shard_names):
        shard_dir = os.path.join(SHARDS_DIR, shard_name)
        index_file = os.path.join(shard_dir, "index.faiss")
        paths_file = os.path.join(shard_dir, "paths.json")

        t0 = time.time()
        index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
        with open(paths_file, 'r') as f:
            paths = json.load(f)
        load_time = time.time() - t0

        t1 = time.time()
        distances, indices = index.search(descriptors, k)
        search_time = time.time() - t1

        # Accumulate votes
        matched = 0
        for qi in range(len(descriptors)):
            for j in range(k):
                idx = indices[qi][j]
                if idx >= 0 and distances[qi][j] >= threshold:
                    if idx < len(paths):
                        all_votes[paths[idx]] += 1
                        matched += 1

        n_vecs = index.ntotal
        total_vectors += n_vecs
        total_time += load_time + search_time

        if len(shard_names) <= 10 or shard_name == TARGET_BOOK:
            top_v = all_votes.most_common(1)[0][1] if all_votes else 0
            print(f"  [{i+1}/{len(shard_names)}] {shard_name[:60]}: "
                  f"{n_vecs:,} vecs, {matched} matches, {load_time:.1f}s load, {search_time:.1f}s search")

        del index, paths

    print(f"\n  Total: {total_vectors:,} vectors across {len(shard_names)} books in {total_time:.1f}s")
    print(f"  Unique pages with votes: {len(all_votes)}")

    # Find page 206
    p206_rank = "-"
    p206_votes = 0
    for rank, (path, v) in enumerate(all_votes.most_common(), 1):
        if 'page206' in os.path.basename(path).lower() and 'onster' in path.lower():
            p206_rank = f"#{rank}"
            p206_votes = v
            break

    print(f"  Page 206: rank {p206_rank} ({p206_votes} votes)")

    print(f"\n  Top 15 results:")
    for rank, (path, v) in enumerate(all_votes.most_common(15), 1):
        bn = os.path.basename(path)
        book = os.path.basename(os.path.dirname(path))
        correct = " <== CORRECT" if 'page206' in bn.lower() and 'onster' in path.lower() else ""
        bk = book[:40] if len(book) <= 40 else book[:37] + "..."
        print(f"    #{rank} ({v}v): [{bk}] {bn[:40]}{correct}")

    return all_votes

# =====================================================
# Test 1: Single book (just Encyclopedia of Monsters)
# =====================================================
search_shards([TARGET_BOOK], descriptors, K, THRESHOLD, "TEST 1: SINGLE BOOK")

# =====================================================
# Test 2: 10 books (target + 9 random)
# =====================================================
random.seed(42)
other_books = [b for b in all_shards if b != TARGET_BOOK]
test_10 = [TARGET_BOOK] + random.sample(other_books, 9)
search_shards(test_10, descriptors, K, THRESHOLD, "TEST 2: 10 BOOKS")

# =====================================================
# Test 3: 50 books (target + 49 random)
# =====================================================
test_50 = [TARGET_BOOK] + random.sample(other_books, 49)
search_shards(test_50, descriptors, K, THRESHOLD, "TEST 3: 50 BOOKS")

print("\nDone.")
