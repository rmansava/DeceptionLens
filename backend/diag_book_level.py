"""Test book-level (directory) aggregation with k=5 and threshold approaches."""
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

def group_by_book(votes):
    """Group page-level votes by book (directory name)."""
    book_votes = Counter()
    book_pages = {}  # book -> Counter of pages
    for path, v in votes.items():
        book = os.path.dirname(path)
        book_name = os.path.basename(book) if book else path
        book_votes[book_name] += v
        if book_name not in book_pages:
            book_pages[book_name] = Counter()
        book_pages[book_name][path] += v
    return book_votes, book_pages

print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load query
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: {len(descriptors)} keypoints\n")

# Load chunk 184
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
local = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local if os.path.exists(local) else chunk_file
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
paths_or_ids, id_to_path = load_chunk_paths(chunk_file, ids_dir)
print(f"Chunk 184: {index.ntotal:,} vectors\n")

# =====================================================
# Test 1: k=5 with book-level aggregation
# =====================================================
print("=" * 80)
print("k=5 search -> book-level aggregation")
print("=" * 80)

topk_results = _gpu_search_batch(index, [("dino", descriptors)], k=5)
distances, indices = topk_results["dino"]

page_votes = Counter()
for i in range(len(descriptors)):
    for j in range(5):
        idx = indices[i][j]
        if idx >= 0 and distances[i][j] >= 0.7:
            path = resolve_path(paths_or_ids, id_to_path, idx)
            if path:
                page_votes[path] += 1

book_votes, book_pages = group_by_book(page_votes)

print(f"\nTop 10 BOOKS:")
for rank, (book, v) in enumerate(book_votes.most_common(10), 1):
    n_pages = len(book_pages[book])
    correct = " <== CORRECT BOOK" if 'onster' in book.lower() else ""
    bk = book[:65] if len(book) <= 65 else book[:62] + "..."
    print(f"  #{rank} ({v} votes, {n_pages} pages): {bk}{correct}")

# Show pages within the winning book
if book_votes:
    winner_book = book_votes.most_common(1)[0][0]
    print(f"\nTop pages in winning book ({winner_book[:50]}...):")
    winner_pages = book_pages[winner_book]
    for rank, (path, v) in enumerate(winner_pages.most_common(10), 1):
        bn = os.path.basename(path)
        correct = " <== CORRECT" if 'page206' in bn.lower() else ""
        print(f"  #{rank} ({v}v): {bn[:60]}{correct}")

del topk_results
torch.cuda.synchronize()
torch.cuda.empty_cache()
del index

# =====================================================
# Test 2: Threshold search -> book-level aggregation
# =====================================================
print(f"\n{'=' * 80}")
print("Threshold=0.7 search -> book-level aggregation")
print("=" * 80)

index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
threshold_votes, match_counts = _gpu_threshold_vote_batch(
    index, [("dino", descriptors)], 0.7, paths_or_ids, id_to_path
)
page_votes_t = threshold_votes["dino"]
book_votes_t, book_pages_t = group_by_book(page_votes_t)

print(f"\nTop 10 BOOKS:")
for rank, (book, v) in enumerate(book_votes_t.most_common(10), 1):
    n_pages = len(book_pages_t[book])
    correct = " <== CORRECT BOOK" if 'onster' in book.lower() else ""
    bk = book[:65] if len(book) <= 65 else book[:62] + "..."
    print(f"  #{rank} ({v:,} votes, {n_pages} pages): {bk}{correct}")

# Show pages within correct book
correct_book = [b for b in book_votes_t if 'onster' in b.lower()]
if correct_book:
    cbook = correct_book[0]
    print(f"\nPages in correct book ({cbook[:50]}...):")
    cpages = book_pages_t[cbook]
    p206_rank = "-"
    for rank, (path, v) in enumerate(cpages.most_common(), 1):
        if 'page206' in os.path.basename(path).lower():
            p206_rank = f"#{rank}"
            break
    print(f"  Total pages: {len(cpages)}")
    print(f"  Page 206 rank within book: {p206_rank}")
    print(f"\n  Top 10 pages:")
    for rank, (path, v) in enumerate(cpages.most_common(10), 1):
        bn = os.path.basename(path)
        correct = " <== CORRECT" if 'page206' in bn.lower() else ""
        print(f"    #{rank} ({v:,}v): {bn[:60]}{correct}")

del index
torch.cuda.empty_cache()

# =====================================================
# Test 3: Threshold -> book -> best page per keypoint
# =====================================================
print(f"\n{'=' * 80}")
print("Unique keypoint voting per book (each kp votes once per book)")
print("=" * 80)

# For this we need to do the search differently
# Load index again
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
n_vectors = index.ntotal
dim = index.d
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(n_vectors, dim)

dtype = torch.float16
batch_size = 4_000_000
score_bytes = 2
use_compact_ids = id_to_path is not None and len(id_to_path) > 0

q_tensor = torch.from_numpy(np.ascontiguousarray(descriptors)).to(device='cuda', dtype=dtype)

# For each keypoint, find the best-scoring page (highest single match)
# Track: keypoint -> page -> best_score
kp_best = [{} for _ in range(len(descriptors))]  # keypoint_idx -> {page_path: best_score}

current = 0
while current < n_vectors:
    end = min(current + batch_size, n_vectors)
    db_count = end - current

    db_slice = np.ascontiguousarray(all_vectors[current:end])
    db_tensor = torch.from_numpy(db_slice).to(device='cuda', dtype=dtype)
    db_t = db_tensor.t()

    max_qb = max(1, int(4 * (1024**3)) // (db_count * score_bytes))

    for q_start in range(0, len(descriptors), max_qb):
        q_end = min(q_start + max_qb, len(descriptors))
        q_batch = q_tensor[q_start:q_end]
        scores = torch.mm(q_batch, db_t)

        # For each keypoint, find all above-threshold columns and their scores
        # Use top-k=200 per keypoint to limit work
        batch_k = min(200, db_count)
        top_scores, top_cols = scores.topk(batch_k, dim=1)

        # Transfer to CPU
        top_scores_cpu = top_scores.cpu().numpy()
        top_cols_cpu = (top_cols + current).cpu().numpy()
        del scores, top_scores, top_cols

        # For each keypoint in this sub-batch
        for qi in range(q_end - q_start):
            kp_idx = q_start + qi
            for j in range(batch_k):
                score = float(top_scores_cpu[qi, j])
                if score < 0.7:
                    break  # sorted descending, rest will be lower
                db_idx = int(top_cols_cpu[qi, j])
                # Resolve to path
                if use_compact_ids:
                    if 0 <= db_idx < len(paths_or_ids):
                        cid = paths_or_ids[db_idx]
                        if 0 <= cid < len(id_to_path):
                            path = id_to_path[cid]
                        else:
                            continue
                    else:
                        continue
                else:
                    if 0 <= db_idx < len(paths_or_ids):
                        path = paths_or_ids[db_idx]
                    else:
                        continue

                # Track best score per page per keypoint
                if path not in kp_best[kp_idx] or score > kp_best[kp_idx][path]:
                    kp_best[kp_idx][path] = score

    del db_tensor, db_t
    current = end
    print(f"  Processed {current:,}/{n_vectors:,}")

torch.cuda.empty_cache()

# Now count unique keypoint votes:
# Method A: Each keypoint's single best page gets 1 vote
# Method B: Each keypoint's best page per book gets 1 vote per book
page_votes_best = Counter()  # method A
book_page_votes = Counter()  # method B: (book, page) -> votes

for kp_idx in range(len(descriptors)):
    pages = kp_best[kp_idx]
    if not pages:
        continue

    # Method A: overall best page for this keypoint
    best_page = max(pages, key=pages.get)
    page_votes_best[best_page] += 1

    # Method B: best page per book for this keypoint
    book_bests = {}  # book -> (page, score)
    for path, score in pages.items():
        book = os.path.dirname(path)
        if book not in book_bests or score > book_bests[book][1]:
            book_bests[book] = (path, score)
    for book, (page, score) in book_bests.items():
        book_page_votes[page] += 1

print(f"\nMethod A: Each keypoint -> 1 vote for overall best page")
print(f"  Total voting keypoints: {sum(1 for p in kp_best if p)}")
p206_rank_a = "-"
for rank, (path, v) in enumerate(page_votes_best.most_common(), 1):
    if 'page206' in os.path.basename(path).lower() and 'onster' in path.lower():
        p206_rank_a = f"#{rank} ({v}v)"
        break
print(f"  Page 206: {p206_rank_a}")
print(f"  Top 10:")
for rank, (path, v) in enumerate(page_votes_best.most_common(10), 1):
    correct = " <== CORRECT" if 'page206' in os.path.basename(path).lower() and 'onster' in path.lower() else ""
    print(f"    #{rank} ({v}v): {os.path.basename(path)[:60]}{correct}")

print(f"\nMethod B: Each keypoint -> 1 vote per book (best page in that book)")
book_agg_b = Counter()
book_pages_b = {}
for path, v in book_page_votes.items():
    book = os.path.basename(os.path.dirname(path))
    book_agg_b[book] += v
    if book not in book_pages_b:
        book_pages_b[book] = Counter()
    book_pages_b[book][path] += v

print(f"  Top 5 BOOKS:")
for rank, (book, v) in enumerate(book_agg_b.most_common(5), 1):
    correct = " <== CORRECT" if 'onster' in book.lower() else ""
    bk = book[:60] if len(book) <= 60 else book[:57] + "..."
    print(f"    #{rank} ({v} votes): {bk}{correct}")

correct_book_b = [b for b in book_pages_b if 'onster' in b.lower()]
if correct_book_b:
    cpages_b = book_pages_b[correct_book_b[0]]
    p206_rank_b = "-"
    for rank, (path, v) in enumerate(cpages_b.most_common(), 1):
        if 'page206' in os.path.basename(path).lower():
            p206_rank_b = f"#{rank} ({v}v)"
            break
    print(f"\n  Within Encyclopedia of Monsters:")
    print(f"    Page 206: {p206_rank_b}")
    print(f"    Top 10 pages:")
    for rank, (path, v) in enumerate(cpages_b.most_common(10), 1):
        correct = " <== CORRECT" if 'page206' in os.path.basename(path).lower() else ""
        print(f"      #{rank} ({v}v): {os.path.basename(path)[:55]}{correct}")
