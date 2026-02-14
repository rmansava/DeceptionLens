"""Build a temporary per-book FAISS index for Encyclopedia of Monsters and test search."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import faiss
import torch
import time
from collections import Counter
from disk_searcher import extract_disk_features

BOOK_DIR = "T:/disk-features/books/Encyclopedia Of Monsters, The (ISBN 0816023034)"
IMAGE_DIR = "D:/books/pdf-images/Encyclopedia Of Monsters, The (ISBN 0816023034)"

# Load query
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    descriptors = extract_disk_features(f.read())
print(f"Query: {len(descriptors)} keypoints\n")

# Build per-book index from .npz feature files
print("Building per-book FAISS index from .npz files...")
npz_files = sorted([f for f in os.listdir(BOOK_DIR) if f.endswith('.npz')])
print(f"  Found {len(npz_files)} page feature files")

all_vectors = []
paths = []  # one path per vector, pointing to the source page image

t0 = time.time()
for npz_file in npz_files:
    npz_path = os.path.join(BOOK_DIR, npz_file)
    data = np.load(npz_path)
    descs = data['descriptors']  # (N, 128) normalized descriptors

    # Derive the page image path from the npz filename
    page_name = npz_file.replace('.npz', '.jpg')
    page_path = os.path.join(IMAGE_DIR, page_name)

    for _ in range(len(descs)):
        paths.append(page_path)
    all_vectors.append(descs)

all_vectors = np.vstack(all_vectors).astype('float32')
build_time = time.time() - t0
print(f"  Total vectors: {len(all_vectors):,} from {len(npz_files)} pages")
print(f"  Built in {build_time:.1f}s\n")

# Create FAISS index
index = faiss.IndexFlatIP(128)
index.add(all_vectors)
print(f"FAISS index: {index.ntotal:,} vectors\n")

# Search with k=5 (simulating old per-book search)
print("=" * 80)
print("Per-book search with k=5, threshold=0.7")
print("=" * 80)

t0 = time.time()
distances, indices = index.search(descriptors, 5)
search_time = time.time() - t0

votes = Counter()
matched = 0
for i in range(len(descriptors)):
    for j in range(5):
        idx = indices[i][j]
        if idx >= 0 and distances[i][j] >= 0.7:
            votes[paths[idx]] += 1
            matched += 1

print(f"  Search time: {search_time:.3f}s")
print(f"  Total matches: {matched}")
print(f"  Unique pages with votes: {len(votes)}")

p206_rank = "-"
p206_votes = 0
for rank, (path, v) in enumerate(votes.most_common(), 1):
    if 'page206' in os.path.basename(path).lower():
        p206_rank = f"#{rank}"
        p206_votes = v
        break

print(f"  Page 206: rank {p206_rank} ({p206_votes} votes)")
print(f"\n  Top 15:")
for rank, (path, v) in enumerate(votes.most_common(15), 1):
    bn = os.path.basename(path)
    correct = " <== CORRECT" if 'page206' in bn.lower() else ""
    print(f"    #{rank} ({v}v): {bn[:65]}{correct}")

# Also test GPU search for speed comparison
if torch.cuda.is_available():
    print(f"\n{'=' * 80}")
    print("GPU search (same per-book index)")
    print("=" * 80)

    q_tensor = torch.from_numpy(descriptors).to(device='cuda', dtype=torch.float16)
    db_tensor = torch.from_numpy(all_vectors).to(device='cuda', dtype=torch.float16)
    db_t = db_tensor.t()

    t0 = time.time()
    scores = torch.mm(q_tensor, db_t)
    top_scores, top_idx = scores.topk(5, dim=1)
    torch.cuda.synchronize()
    gpu_time = time.time() - t0

    # Accumulate votes
    top_scores_cpu = top_scores.cpu().numpy()
    top_idx_cpu = top_idx.cpu().numpy()

    gpu_votes = Counter()
    for i in range(len(descriptors)):
        for j in range(5):
            if top_scores_cpu[i][j] >= 0.7:
                gpu_votes[paths[top_idx_cpu[i][j]]] += 1

    p206_rank_gpu = "-"
    p206_votes_gpu = 0
    for rank, (path, v) in enumerate(gpu_votes.most_common(), 1):
        if 'page206' in os.path.basename(path).lower():
            p206_rank_gpu = f"#{rank}"
            p206_votes_gpu = v
            break

    print(f"  GPU search time: {gpu_time:.3f}s")
    print(f"  Page 206: rank {p206_rank_gpu} ({p206_votes_gpu} votes)")
    print(f"\n  Top 15:")
    for rank, (path, v) in enumerate(gpu_votes.most_common(15), 1):
        bn = os.path.basename(path)
        correct = " <== CORRECT" if 'page206' in bn.lower() else ""
        print(f"    #{rank} ({v}v): {bn[:65]}{correct}")

    del q_tensor, db_tensor, db_t, scores
    torch.cuda.empty_cache()

del index
print("\nDone.")
