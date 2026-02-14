"""Debug threshold search - pinpoint CUDA error."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import faiss
import torch
import time
from disk_searcher import (
    extract_disk_features, load_chunk_paths, LOCAL_CHUNK_BUFFER,
    GPU_SEARCH_USE_FP16, GPU_SEARCH_BATCH_SIZE, GPU_SEARCH_MAX_SCORES_BYTES
)
from collections_config import COLLECTIONS

print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

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
print(f"Chunk: {index.ntotal:,} vectors, dim={index.d}\n")

n_vectors = index.ntotal
dim = index.d
all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(n_vectors, dim)

dtype = torch.float16 if GPU_SEARCH_USE_FP16 else torch.float32
threshold = 0.7
batch_size = GPU_SEARCH_BATCH_SIZE
score_bytes = 2 if GPU_SEARCH_USE_FP16 else 4

q_tensor = torch.from_numpy(np.ascontiguousarray(descriptors)).to(device='cuda', dtype=dtype)

# Process first DB batch
current = 0
end = min(batch_size, n_vectors)
db_count = end - current

print(f"DB batch: {db_count:,} vectors")
db_slice = np.ascontiguousarray(all_vectors[current:end])
db_tensor = torch.from_numpy(db_slice).to(device='cuda', dtype=dtype)
db_t = db_tensor.t()
torch.cuda.synchronize()
print(f"  db_tensor loaded: {db_tensor.shape}")

max_qb = max(1, GPU_SEARCH_MAX_SCORES_BYTES // (db_count * score_bytes))
print(f"  max_qb: {max_qb}")
print(f"  Using sub-batch of {min(max_qb, len(descriptors))} keypoints\n")

q_batch = q_tensor[:min(max_qb, len(descriptors))]

# Step 1: matmul
print("Step 1: torch.mm...")
free_before = torch.cuda.mem_get_info()[0] / (1024**3)
scores = torch.mm(q_batch, db_t)
torch.cuda.synchronize()
free_after = torch.cuda.mem_get_info()[0] / (1024**3)
print(f"  OK: {scores.shape}, {scores.dtype}, VRAM used: {free_before - free_after:.2f}GB")
print(f"  VRAM free: {free_after:.2f}GB")
print(f"  Score range: {scores.min().item():.3f} to {scores.max().item():.3f}")

# Step 2: threshold comparison
print("\nStep 2: scores >= threshold...")
mask = scores >= threshold
torch.cuda.synchronize()
free_after2 = torch.cuda.mem_get_info()[0] / (1024**3)
print(f"  OK: {mask.shape}, {mask.dtype}, VRAM used: {free_after - free_after2:.2f}GB")
print(f"  VRAM free: {free_after2:.2f}GB")
print(f"  Above threshold: {mask.sum().item():,}")

# Step 3a: sum per column
print("\nStep 3a: mask.sum(dim=0)...")
votes_per_vec = mask.sum(dim=0)
torch.cuda.synchronize()
print(f"  OK: {votes_per_vec.shape}")
nonzero = (votes_per_vec > 0).sum().item()
print(f"  Non-zero vectors: {nonzero:,}")
del votes_per_vec

# Step 3b: torch.where
print("\nStep 3b: torch.where(mask)...")
try:
    rows, cols = torch.where(mask)
    torch.cuda.synchronize()
    print(f"  OK: {len(rows):,} entries")
except RuntimeError as e:
    print(f"  FAILED: {e}")

# Step 3c: nonzero
print("\nStep 3c: mask.nonzero(as_tuple=True)...")
try:
    rows2, cols2 = mask.nonzero(as_tuple=True)
    torch.cuda.synchronize()
    print(f"  OK: {len(rows2):,} entries")
except RuntimeError as e:
    print(f"  FAILED: {e}")

del mask, scores, db_tensor, db_t
torch.cuda.empty_cache()
print("\nDone.")
