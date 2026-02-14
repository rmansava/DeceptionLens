"""Direct score comparison: dino crop keypoints vs Encyclopedia of Monsters vectors in chunk 184."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import faiss
import time
from disk_searcher import extract_disk_features, LOCAL_CHUNK_BUFFER
from collections_config import COLLECTIONS

# Load query
print("Loading query image...")
with open("D:/trivpics/2023-5.jpg", 'rb') as f:
    image_bytes = f.read()
descriptors = extract_disk_features(image_bytes)
print(f"  618 keypoints extracted: {descriptors.shape}")

# Load chunk 184
chunk_file = "T:/faiss/disk_retrieval/chunks/chunk_184.faiss"
local_chunk = os.path.join(LOCAL_CHUNK_BUFFER, "chunk_184.faiss")
load_file = local_chunk if os.path.exists(local_chunk) else chunk_file

print(f"\nLoading chunk 184...")
index = faiss.read_index(load_file, faiss.IO_FLAG_MMAP)
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(index.ntotal, index.d)
print(f"  {index.ntotal:,} vectors, dim={index.d}")

# Load IDs
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
ids = np.load(os.path.join(ids_dir, "chunk_184_ids.npy"))
print(f"  IDs loaded: {len(ids):,}")

# Find positions of Encyclopedia of Monsters (ID 495272)
target_id = 495272
positions = np.where(ids == target_id)[0]
print(f"\n  ID {target_id} found at {len(positions):,} positions")
print(f"  Position range: {positions[0]:,} - {positions[-1]:,}")

# Extract just the target vectors
target_vectors = all_vectors[positions]
print(f"  Target vectors shape: {target_vectors.shape}")

# Compute scores: (618 keypoints) x (18177 target vectors)
print(f"\nComputing direct inner products...")
q = torch.from_numpy(np.ascontiguousarray(descriptors)).cuda().float()
t = torch.from_numpy(np.ascontiguousarray(target_vectors)).cuda().float()

scores = torch.mm(q, t.t())  # (618, 18177)
print(f"  Scores matrix: {scores.shape}")

# Per-keypoint: best match against target vectors
best_per_kp, _ = scores.max(dim=1)  # (618,)
best_np = best_per_kp.cpu().numpy()

print(f"\n  Per-keypoint best score against page 210 vectors:")
print(f"    Max:    {best_np.max():.4f}")
print(f"    Mean:   {best_np.mean():.4f}")
print(f"    Median: {np.median(best_np):.4f}")
print(f"    Min:    {best_np.min():.4f}")
print(f"    Std:    {best_np.std():.4f}")

# How many keypoints have good matches?
for t_val in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]:
    count = (best_np >= t_val).sum()
    print(f"    >= {t_val}: {count} keypoints")

# Now check: for the matching keypoints, what rank does page 210 get among ALL 21M vectors?
print(f"\n  Checking ranks for top-matching keypoints...")
# Take keypoints with best scores >= 0.6 against target
good_kps = np.where(best_np >= 0.6)[0]
if len(good_kps) == 0:
    good_kps = np.argsort(best_np)[-10:]  # take best 10 anyway
    print(f"  (No keypoints >= 0.6, taking top {len(good_kps)} by score)")

print(f"  Checking {len(good_kps)} keypoints...")

# For each good keypoint, compute its score against ALL vectors and find rank of the target
for kp_idx in good_kps[:10]:  # limit to 10 for speed
    kp_score_vs_target = best_np[kp_idx]

    # Score against all 21M vectors in batches
    kp_vec = q[kp_idx:kp_idx+1]  # (1, 128)
    best_global = -999.0
    count_better = 0

    batch_size = 4_000_000
    for start in range(0, index.ntotal, batch_size):
        end = min(start + batch_size, index.ntotal)
        db_batch = torch.from_numpy(np.ascontiguousarray(all_vectors[start:end])).cuda().float()
        batch_scores = torch.mm(kp_vec, db_batch.t()).squeeze()
        n_better = (batch_scores > kp_score_vs_target).sum().item()
        count_better += n_better
        batch_max = batch_scores.max().item()
        if batch_max > best_global:
            best_global = batch_max
        del db_batch, batch_scores

    print(f"    KP {kp_idx}: score vs target = {kp_score_vs_target:.4f}, "
          f"global best = {best_global:.4f}, "
          f"vectors with better score = {count_better:,}")

del q, t, scores
torch.cuda.empty_cache()

print("\nDone.")
