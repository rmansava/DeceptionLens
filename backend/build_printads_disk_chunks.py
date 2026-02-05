r"""
Build DISK keypoint chunks for print ads - direct to chunks with compact IDs.

Unlike books (which go through per-book shards then consolidation), print ads
are individual images with no natural grouping. This script goes straight from
images to search-ready chunks:

  1. List all images across all subfolders
  2. Process in batches of IMAGES_PER_CHUNK (~5000)
  3. Each batch: extract DISK features -> build FAISS index -> save chunk + compact IDs
  4. Compact IDs from the start (no paths.json bloat)

Input:  Local copy of print ads (fast reads from SSD)
Output: chunk_XXX.faiss  -> NAS (T:/faiss/disk_retrieval/printads_chunks/)
        chunk_XXX_ids.npy -> local SSD (D:/faiss/disk_retrieval/printads_chunk_ids/)
        path_lookup.json  -> local SSD (same dir as IDs)

Paths stored point to the NAS originals (T:/archiverelated/print ads/...).
Resumable via progress file.
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

try:
    import kornia.feature as KF
    import kornia as K
except ImportError:
    print("ERROR: Kornia not installed. Run: pip install kornia")
    sys.exit(1)


# ============================================================================
# CONFIG - Edit these paths as needed
# ============================================================================

# Source: local copy of print ads for fast reading (scans all subfolders)
LOCAL_IMAGES_DIR = r"C:\printads"

# Path remapping: stored paths point to NAS originals
NAS_IMAGES_DIR = r"T:\archiverelated\print ads"

# Output: FAISS chunks go to NAS (searched via rolling buffer copy)
NAS_CHUNKS_DIR = r"T:\faiss\disk_retrieval\printads_chunks"
LOCAL_CHUNKS_BUFFER = r"D:\faiss\disk_retrieval\printads_chunks"  # Write here first, then copy to NAS

# Output: Compact IDs stay on local SSD (fast reads during search)
CHUNK_IDS_DIR = r"D:\faiss\disk_retrieval\printads_chunk_ids"

# Progress tracking
PROGRESS_DIR = CHUNK_IDS_DIR
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "build_progress.json")
LOG_FILE = os.path.join(PROGRESS_DIR, "build_log.txt")

# Chunk sizing: ~5000 images * ~4000 keypoints = ~20M vectors * 128 dims * 4 bytes = ~10GB per chunk
# Adjust based on your RAM. 5000 is conservative for 32GB RAM.
IMAGES_PER_CHUNK = 5000

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
    """Convert local read path to NAS storage path."""
    # Replace local prefix with NAS prefix
    if local_path.startswith(LOCAL_IMAGES_DIR):
        return NAS_IMAGES_DIR + local_path[len(LOCAL_IMAGES_DIR):]
    # Try with normalized separators
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
    if not os.path.exists(PROGRESS_FILE):
        return 1, set(), {}, 0

    try:
        with open(PROGRESS_FILE, 'r') as f:
            state = json.load(f)
        processed = set(state.get('processed_images', []))
        next_chunk = state.get('next_chunk', 1)
        next_id = state.get('next_id', 0)

        # Load path_to_id from existing path_lookup
        path_to_id = {}
        lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
        if os.path.exists(lookup_file):
            with open(lookup_file, 'r') as f:
                id_to_path = json.load(f)
            path_to_id = {p: i for i, p in enumerate(id_to_path)}
            next_id = len(id_to_path)

        return next_chunk, processed, path_to_id, next_id
    except Exception as e:
        log(f"Warning: Could not load progress: {e}")
        return 1, set(), {}, 0


def save_progress(next_chunk, processed_images, next_id):
    """Save build progress (image list stored separately to keep this small)."""
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
    """Save the global path lookup (both directions)."""
    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)

    # Save id_to_path (list indexed by ID) - used during search
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


def build_chunk(chunk_num, image_batch, extractor, device, path_to_id, next_id):
    """
    Extract DISK features for a batch of images and build a FAISS chunk.

    Returns:
        (num_vectors, num_images_ok, num_failed, next_id)
    """
    all_descriptors = []
    all_ids = []  # int32 IDs corresponding to each descriptor's source image
    ok = 0
    failed = 0

    for i, image_path in enumerate(image_batch):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"    Image {i+1}/{len(image_batch)}...", end='\r', flush=True)

        # Preprocess
        tensor = preprocess_image(image_path)
        if tensor is None:
            failed += 1
            continue

        # Extract DISK features
        try:
            tensor = tensor.to(device)
            with torch.no_grad():
                feats = extractor(tensor)[0]
                descriptors = feats.descriptors.cpu().numpy()  # (N, 128)

            if len(descriptors) == 0:
                failed += 1
                del tensor
                continue

            # Get or assign path ID (using NAS path)
            nas_path = remap_path(image_path)
            if nas_path not in path_to_id:
                path_to_id[nas_path] = next_id
                next_id += 1
            pid = path_to_id[nas_path]

            # Accumulate
            all_descriptors.append(descriptors)
            all_ids.extend([pid] * len(descriptors))
            ok += 1

        except Exception as e:
            failed += 1

        # Cleanup
        del tensor
        if (i + 1) % 200 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"    Image {len(image_batch)}/{len(image_batch)} done.    ")

    if not all_descriptors:
        return 0, ok, failed, next_id

    # Build FAISS index
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

    # Copy to NAS
    os.makedirs(NAS_CHUNKS_DIR, exist_ok=True)
    nas_faiss = os.path.join(NAS_CHUNKS_DIR, f"chunk_{chunk_num:03d}.faiss")
    shutil.copy2(local_faiss, nas_faiss)

    # Delete local buffer copy
    try:
        os.remove(local_faiss)
    except Exception:
        pass

    # Save compact IDs (stays on local SSD)
    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)
    ids_array = np.array(all_ids, dtype=np.int32)
    ids_file = os.path.join(CHUNK_IDS_DIR, f"chunk_{chunk_num:03d}_ids.npy")
    np.save(ids_file, ids_array)
    ids_size = os.path.getsize(ids_file) / (1024**2)

    log(f"  Chunk {chunk_num:03d}: {num_vectors:,} vectors from {ok} images "
        f"({faiss_size:.1f} GB index, {ids_size:.0f} MB IDs)")

    # Cleanup
    del all_desc, all_descriptors, all_ids, ids_array, index
    gc.collect()

    return num_vectors, ok, failed, next_id


def main():
    log("=" * 70)
    log("PRINT ADS DISK CHUNK BUILDER")
    log(f"Source (local):  {LOCAL_IMAGES_DIR}")
    log(f"Paths stored as: {NAS_IMAGES_DIR}")
    log(f"Chunks output:   {NAS_CHUNKS_DIR}")
    log(f"Compact IDs:     {CHUNK_IDS_DIR}")
    log(f"Images/chunk:    {IMAGES_PER_CHUNK}")
    log("=" * 70)

    # Check source exists
    if not os.path.exists(LOCAL_IMAGES_DIR):
        log(f"ERROR: Source directory not found: {LOCAL_IMAGES_DIR}")
        log(f"Copy print ads from NAS to local drive first.")
        log(f"  robocopy \"{NAS_IMAGES_DIR}\" \"{LOCAL_IMAGES_DIR}\" /E /R:2 /W:5")
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

    # Process in chunks
    total_chunks = (len(remaining) + IMAGES_PER_CHUNK - 1) // IMAGES_PER_CHUNK
    log(f"Will produce ~{total_chunks} chunks")
    log("=" * 70)

    start_time = time.time()
    total_vectors = 0
    total_ok = 0
    total_failed = 0
    chunk_num = next_chunk

    for batch_start in range(0, len(remaining), IMAGES_PER_CHUNK):
        batch_end = min(batch_start + IMAGES_PER_CHUNK, len(remaining))
        batch = remaining[batch_start:batch_end]
        batch_idx = (batch_start // IMAGES_PER_CHUNK) + 1

        log(f"\n[Chunk {chunk_num:03d}] ({batch_idx}/{total_chunks}) "
            f"Images {batch_start+1}-{batch_end} of {len(remaining)}")

        chunk_start = time.time()
        num_vectors, ok, failed, next_id = build_chunk(
            chunk_num, batch, extractor, device, path_to_id, next_id
        )
        chunk_time = time.time() - chunk_start

        total_vectors += num_vectors
        total_ok += ok
        total_failed += failed

        # Update processed set
        for img in batch:
            processed.add(img)

        # Save progress + path lookup
        chunk_num += 1
        save_progress(chunk_num, processed, next_id)
        save_path_lookup(path_to_id)

        # ETA
        elapsed = time.time() - start_time
        chunks_done = batch_idx
        avg_per_chunk = elapsed / chunks_done
        remaining_chunks = total_chunks - chunks_done
        eta_hours = (avg_per_chunk * remaining_chunks) / 3600

        log(f"  Time: {chunk_time:.0f}s | Total vectors: {total_vectors:,} | "
            f"Unique paths: {len(path_to_id):,} | ETA: {eta_hours:.1f}h")

    # Final summary
    log("\n" + "=" * 70)
    log("BUILD COMPLETE!")
    log(f"  Chunks created: {chunk_num - next_chunk}")
    log(f"  Total vectors:  {total_vectors:,}")
    log(f"  Images OK:      {total_ok:,}")
    log(f"  Images failed:  {total_failed:,}")
    log(f"  Unique paths:   {len(path_to_id):,}")
    log(f"  Total time:     {(time.time() - start_time) / 3600:.1f} hours")
    log(f"  Chunks at:      {NAS_CHUNKS_DIR}")
    log(f"  Compact IDs at: {CHUNK_IDS_DIR}")
    log("=" * 70)


if __name__ == '__main__':
    main()
