r"""
Build DISK keypoint chunks for books - direct from images to 10GB chunks.

Same approach as print ads/cereal/board games: goes straight from images to
search-ready chunks. Use this for new books instead of the old two-step
(per-book shards -> consolidation) pipeline.

  1. List all images across all book subfolders
  2. Extract DISK features per image on GPU
  3. Accumulate vectors, flush to chunk when hitting ~10 GB (19.5M vectors)
  4. Compact IDs from the start (no paths.json bloat)

Input:  Local copy of book page images (D:\books\pdf-images)
Output: chunk_XXX.faiss  -> NAS (T:/faiss/disk_retrieval/chunks/)
        chunk_XXX_ids.npy -> local SSD (D:/faiss/disk_retrieval/chunk_ids/)
        path_lookup.json  -> local SSD (same dir as IDs)

Paths stored match the existing convention (D:/books/pdf-images/...).
Resumable via progress file. Loads existing path_lookup.json to continue IDs.
"""

import os
import sys
import gc
import json
import time
import shutil
import numpy as np
import faiss
import torch
import cv2
from glob import glob
from datetime import datetime
from threading import Thread

try:
    import kornia.feature as KF
    import kornia as K
except ImportError:
    print("ERROR: Kornia not installed. Run: pip install kornia")
    sys.exit(1)


# ============================================================================
# CONFIG - Edit these paths as needed
# ============================================================================

# Source: book page images (already on local SSD)
LOCAL_IMAGES_DIR = r"D:\books\pdf-images"

# No path remapping needed - stored paths use D:/books/pdf-images/... to match
# existing chunks built by consolidation
NAS_IMAGES_DIR = r"D:\books\pdf-images"

# Output: FAISS chunks go to NAS (same dir as existing book chunks)
NAS_CHUNKS_DIR = r"T:\faiss\disk_retrieval\chunks"
LOCAL_CHUNKS_BUFFER = r"D:\faiss\disk_retrieval\chunks"  # Write here first, then copy to NAS

# Output: Compact IDs stay on local SSD (same dir as existing book chunk IDs)
CHUNK_IDS_DIR = r"D:\faiss\disk_retrieval\chunk_ids"

# Progress tracking
PROGRESS_DIR = CHUNK_IDS_DIR
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "build_progress_newbooks.json")
LOG_FILE = os.path.join(PROGRESS_DIR, "build_newbooks_log.txt")

# Chunk sizing: target ~10GB per chunk
# 10GB = ~19.5M vectors at 128 dims * 4 bytes = 512 bytes/vector
MAX_VECTORS_PER_CHUNK = 19_500_000  # ~10 GB

# DISK extraction settings
MAX_IMAGE_DIM = 1600  # Resize images larger than this
GPU_BATCH_SIZE = 1    # Images per GPU batch (1 is safest for varied sizes)

# ============================================================================


def log(msg):
    """Print and log to file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def remap_path(local_path):
    """Convert local read path to stored path."""
    if local_path.startswith(LOCAL_IMAGES_DIR):
        return NAS_IMAGES_DIR + local_path[len(LOCAL_IMAGES_DIR):]
    local_norm = local_path.replace('\\', '/')
    local_dir_norm = LOCAL_IMAGES_DIR.replace('\\', '/')
    if local_norm.startswith(local_dir_norm):
        return NAS_IMAGES_DIR + local_norm[len(local_dir_norm):]
    return local_path


def find_all_images():
    """Find all images across all subfolders."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif', '*.bmp')
    files = set()
    for ext in extensions:
        for f in glob(os.path.join(LOCAL_IMAGES_DIR, '**', ext), recursive=True):
            files.add(os.path.normpath(f))
        for f in glob(os.path.join(LOCAL_IMAGES_DIR, '**', ext.upper()), recursive=True):
            files.add(os.path.normpath(f))
    return sorted(files)


def load_progress():
    """Load build progress. Returns (next_chunk_num, set of processed image paths, path_to_id dict, next_id)."""
    # Always load existing path_lookup if it exists (shared with consolidation chunks)
    path_to_id = {}
    next_id = 0
    lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    if os.path.exists(lookup_file):
        log(f"  Loading existing path_lookup.json...")
        with open(lookup_file, 'r') as f:
            id_to_path = json.load(f)
        path_to_id = {p: i for i, p in enumerate(id_to_path)}
        next_id = len(id_to_path)
        log(f"  Loaded {len(path_to_id):,} existing paths (next_id={next_id})")
        del id_to_path

    if not os.path.exists(PROGRESS_FILE):
        # Find the highest existing chunk number to continue from
        existing_chunks = glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*.faiss"))
        if existing_chunks:
            max_chunk = max(
                int(os.path.basename(f).replace('chunk_', '').replace('.faiss', ''))
                for f in existing_chunks
            )
            log(f"  Found existing chunks up to {max_chunk}, will start at {max_chunk + 1}")
            return max_chunk + 1, set(), path_to_id, next_id
        return 1, set(), path_to_id, next_id

    try:
        with open(PROGRESS_FILE, 'r') as f:
            state = json.load(f)
        processed = set(state.get('processed_images', []))
        next_chunk = state.get('next_chunk', 1)

        return next_chunk, processed, path_to_id, next_id
    except Exception as e:
        log(f"Warning: Could not load progress: {e}")
        return 1, set(), path_to_id, next_id


def save_progress(next_chunk, processed_images, next_id):
    """Save build progress."""
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    state = {
        'next_chunk': next_chunk,
        'next_id': next_id,
        'processed_count': len(processed_images),
        'last_updated': datetime.now().isoformat(),
        'processed_images': list(processed_images)
    }
    temp = PROGRESS_FILE + '.tmp'
    with open(temp, 'w') as f:
        json.dump(state, f)
    shutil.move(temp, PROGRESS_FILE)


def save_path_lookup(path_to_id):
    """Save the global path lookup (list indexed by ID) - used during search."""
    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)

    id_to_path = [''] * len(path_to_id)
    for path, pid in path_to_id.items():
        id_to_path[pid] = path

    lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    with open(lookup_file, 'w') as f:
        json.dump(id_to_path, f)

    log(f"  Saved path_lookup.json: {len(path_to_id):,} unique paths ({os.path.getsize(lookup_file) / 1e6:.1f} MB)")


def preprocess_image(image_path, max_dim=MAX_IMAGE_DIM):
    """Load and preprocess image for DISK extraction. Returns tensor or None."""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        h, w = img.shape[:2]

        # Resize if too large
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        # Pad to multiples of 16 (required by DISK)
        new_h = ((h + 15) // 16) * 16
        new_w = ((w + 15) // 16) * 16
        if new_h != h or new_w != w:
            img = cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = K.image_to_tensor(img, False).float() / 255.0
        return tensor
    except Exception:
        return None


# Background NAS copy state
_nas_copy_thread = None
_nas_copy_error = None


def _nas_copy_worker(local_faiss, nas_faiss, chunk_num):
    """Background worker to copy a chunk from local SSD to NAS."""
    global _nas_copy_error
    try:
        copy_start = time.time()
        shutil.copy2(local_faiss, nas_faiss)
        copy_time = time.time() - copy_start
        faiss_size = os.path.getsize(nas_faiss) / (1024**3)
        log(f"  NAS copy done: chunk {chunk_num:03d} ({faiss_size:.1f} GB in {copy_time:.0f}s)")
        try:
            os.remove(local_faiss)
        except Exception:
            pass
    except Exception as e:
        _nas_copy_error = str(e)
        log(f"  ERROR: NAS copy failed for chunk {chunk_num:03d}: {e}")


def wait_for_nas_copy():
    """Wait for any pending NAS copy to finish."""
    global _nas_copy_thread, _nas_copy_error
    if _nas_copy_thread is not None:
        _nas_copy_thread.join()
        _nas_copy_thread = None
        if _nas_copy_error:
            log(f"  WARNING: Previous NAS copy had error: {_nas_copy_error}")
            _nas_copy_error = None


def save_chunk(chunk_num, all_descriptors, all_ids, num_images):
    """Build FAISS index and save chunk + IDs. NAS copy runs in background."""
    global _nas_copy_thread

    if not all_descriptors:
        return 0

    wait_for_nas_copy()

    log(f"  Building FAISS index for chunk {chunk_num}...")
    all_desc = np.vstack(all_descriptors)
    num_vectors = len(all_desc)

    index = faiss.IndexFlatIP(128)
    index.add(all_desc)

    # Save chunk .faiss to local buffer first
    os.makedirs(LOCAL_CHUNKS_BUFFER, exist_ok=True)
    local_faiss = os.path.join(LOCAL_CHUNKS_BUFFER, f"chunk_{chunk_num:03d}.faiss")
    faiss.write_index(index, local_faiss)
    faiss_size = os.path.getsize(local_faiss) / (1024**3)

    # Save compact IDs (stays on local SSD)
    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)
    ids_array = np.array(all_ids, dtype=np.int32)
    ids_file = os.path.join(CHUNK_IDS_DIR, f"chunk_{chunk_num:03d}_ids.npy")
    np.save(ids_file, ids_array)
    ids_size = os.path.getsize(ids_file) / (1024**2)

    log(f"  Chunk {chunk_num:03d}: {num_vectors:,} vectors from {num_images} images "
        f"({faiss_size:.1f} GB index, {ids_size:.0f} MB IDs)")

    # Queue NAS copy in background
    os.makedirs(NAS_CHUNKS_DIR, exist_ok=True)
    nas_faiss = os.path.join(NAS_CHUNKS_DIR, f"chunk_{chunk_num:03d}.faiss")
    _nas_copy_thread = Thread(target=_nas_copy_worker, args=(local_faiss, nas_faiss, chunk_num))
    _nas_copy_thread.daemon = True
    _nas_copy_thread.start()
    log(f"  NAS copy queued (background): {local_faiss} -> {nas_faiss}")

    # Cleanup
    del all_desc, all_descriptors, all_ids, ids_array, index
    gc.collect()

    return num_vectors


def main():
    log("=" * 70)
    log("BOOKS DISK CHUNK BUILDER (Direct to Chunks)")
    log(f"Source:          {LOCAL_IMAGES_DIR}")
    log(f"Paths stored as: {NAS_IMAGES_DIR}")
    log(f"Chunks output:   {NAS_CHUNKS_DIR}")
    log(f"Compact IDs:     {CHUNK_IDS_DIR}")
    log(f"Target/chunk:    {MAX_VECTORS_PER_CHUNK:,} vectors (~10 GB)")
    log("=" * 70)

    # Check source exists
    if not os.path.exists(LOCAL_IMAGES_DIR):
        log(f"ERROR: Source directory not found: {LOCAL_IMAGES_DIR}")
        sys.exit(1)

    # Find all images
    log("Scanning for images...")
    all_images = find_all_images()
    log(f"Found {len(all_images):,} images")

    if not all_images:
        log("No images found!")
        sys.exit(1)

    # Load progress
    next_chunk, processed, path_to_id, next_id = load_progress()
    log(f"Resume: chunk {next_chunk}, {len(processed):,} images already done, {len(path_to_id):,} unique paths")

    # Filter out already processed
    remaining = [f for f in all_images if f not in processed]
    log(f"Remaining: {len(remaining):,} images")

    if not remaining:
        log("All images already processed!")
        return

    # Load DISK model
    log("Loading DISK model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = KF.DISK.from_pretrained('depth').to(device).eval()
    log(f"DISK model loaded on {device}")

    # Estimate chunks (rough, actual count depends on keypoints per image)
    est_chunks = max(1, int(len(remaining) * 15000 / MAX_VECTORS_PER_CHUNK))
    log(f"~{est_chunks} chunks estimated (10 GB target, actual depends on keypoints)")
    log("=" * 70)

    start_time = time.time()
    total_vectors = 0
    total_ok = 0
    total_failed = 0
    chunk_num = next_chunk
    images_processed = 0

    # Accumulator for current chunk
    chunk_descriptors = []
    chunk_ids = []
    chunk_vector_count = 0
    chunk_image_count = 0
    chunk_start = time.time()

    for i, image_path in enumerate(remaining):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"    Image {i+1}/{len(remaining)} (chunk {chunk_num:03d}, "
                  f"{chunk_vector_count:,} vectors)...", end='\r', flush=True)

        # Preprocess
        tensor = preprocess_image(image_path)
        if tensor is None:
            total_failed += 1
            processed.add(image_path)
            continue

        # Extract DISK features
        try:
            tensor = tensor.to(device)
            with torch.no_grad():
                feats = extractor(tensor)[0]
                descriptors = feats.descriptors.cpu().numpy()  # (N, 128)

            if len(descriptors) == 0:
                total_failed += 1
                processed.add(image_path)
                del tensor
                continue

            # Get or assign path ID
            stored_path = remap_path(image_path)
            if stored_path not in path_to_id:
                path_to_id[stored_path] = next_id
                next_id += 1
            pid = path_to_id[stored_path]

            # Accumulate
            chunk_descriptors.append(descriptors)
            chunk_ids.extend([pid] * len(descriptors))
            chunk_vector_count += len(descriptors)
            chunk_image_count += 1
            total_ok += 1

        except Exception:
            total_failed += 1

        processed.add(image_path)
        del tensor
        if (i + 1) % 200 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Flush chunk when we hit the vector cap
        if chunk_vector_count >= MAX_VECTORS_PER_CHUNK:
            print(f"    Image {i+1}/{len(remaining)} done.    ")
            chunk_time = time.time() - chunk_start

            log(f"\n[Chunk {chunk_num:03d}] Flushing {chunk_vector_count:,} vectors "
                f"from {chunk_image_count} images")

            num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
            total_vectors += num_vectors
            images_processed += chunk_image_count

            # Save progress
            chunk_num += 1
            save_progress(chunk_num, processed, next_id)
            save_path_lookup(path_to_id)

            # ETA
            elapsed = time.time() - start_time
            images_left = len(remaining) - (i + 1)
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            eta_hours = (images_left / rate) / 3600

            log(f"  Time: {chunk_time:.0f}s | Total vectors: {total_vectors:,} | "
                f"Unique paths: {len(path_to_id):,} | ETA: {eta_hours:.1f}h")

            # Reset accumulator
            chunk_descriptors = []
            chunk_ids = []
            chunk_vector_count = 0
            chunk_image_count = 0
            chunk_start = time.time()

    # Flush remaining vectors as final chunk
    if chunk_descriptors:
        log(f"\n[Chunk {chunk_num:03d}] Final chunk: {chunk_vector_count:,} vectors "
            f"from {chunk_image_count} images")
        num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
        total_vectors += num_vectors
        images_processed += chunk_image_count
        chunk_num += 1
        save_progress(chunk_num, processed, next_id)
        save_path_lookup(path_to_id)

    # Wait for any pending NAS copy before reporting final stats
    wait_for_nas_copy()

    # Final summary
    log("\n" + "=" * 70)
    log("BUILD COMPLETE!")
    chunks_created = chunk_num - next_chunk
    log(f"  Chunks created: {chunks_created}")
    log(f"  Total vectors:  {total_vectors:,}")
    log(f"  Images OK:      {total_ok:,}")
    log(f"  Images failed:  {total_failed:,}")
    log(f"  Unique paths:   {len(path_to_id):,}")
    log(f"  Avg chunk size: {total_vectors / max(chunks_created, 1):,.0f} vectors "
        f"(~{total_vectors * 512 / max(chunks_created, 1) / (1024**3):.1f} GB)")
    log(f"  Total time:     {(time.time() - start_time) / 3600:.1f} hours")
    log(f"  Chunks at:      {NAS_CHUNKS_DIR}")
    log(f"  Compact IDs at: {CHUNK_IDS_DIR}")
    log("=" * 70)


if __name__ == '__main__':
    main()
