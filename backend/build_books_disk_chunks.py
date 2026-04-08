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
import traceback
import numpy as np
import faiss
import torch
import cv2
from glob import glob
from datetime import datetime, timedelta
import queue
from threading import Thread, Lock
from disk_chunk_db import sync_chunk_to_db, sync_paths_to_db, create_tables

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
CUDA_BAD_IMAGES_FILE = os.path.join(PROGRESS_DIR, "cuda_bad_images.txt")
PREPROCESS_BAD_IMAGES_FILE = os.path.join(PROGRESS_DIR, "preprocess_bad_images.txt")

# Chunk sizing: target ~10GB per chunk
# 10GB = ~19.5M vectors at 128 dims * 4 bytes = 512 bytes/vector
MAX_VECTORS_PER_CHUNK = int(os.environ.get("DISK_MAX_VECTORS_PER_CHUNK", "19500000"))  # ~10 GB

# Collection name for DB sync
COLLECTION_NAME = "books"

# DISK extraction settings
_max_dim_raw = os.environ.get("DISK_MAX_IMAGE_DIM", "none").strip().lower()
if _max_dim_raw in ("0", "none", ""):
    MAX_IMAGE_DIM = None
else:
    MAX_IMAGE_DIM = int(_max_dim_raw)
GPU_BATCH_SIZE = 1    # Images per GPU batch (1 is safest for varied sizes)

# Prefetch pipeline: load/preprocess images on CPU threads while GPU works
PREFETCH_WORKERS = 4   # Background threads for image loading
PREFETCH_QUEUE_SIZE = 16  # Max preprocessed tensors held in memory
PROGRESS_HEARTBEAT_SEC = 120  # Persist periodic progress snapshots in log
CUDA_ERROR_STREAK_FOR_RECOVERY = max(5, int(os.environ.get("DISK_CUDA_ERROR_STREAK_FOR_RECOVERY", "32")))
CUDA_MAX_RECOVERY_ATTEMPTS = max(0, int(os.environ.get("DISK_CUDA_MAX_RECOVERY_ATTEMPTS", "1")))
CUDA_POISON_EXIT_CODE = int(os.environ.get("DISK_CUDA_POISON_EXIT_CODE", "86"))
PATH_DB_SYNC_EVERY_CHUNKS = max(1, int(os.environ.get("DISK_PATH_DB_SYNC_EVERY_CHUNKS", "5")))
PROGRESS_SAVE_EVERY_IMAGES = max(100, int(os.environ.get("DISK_PROGRESS_SAVE_EVERY_IMAGES", "1000")))

# ============================================================================


_console_lock = Lock()
_progress_line_len = 0


class CudaPoisonedError(RuntimeError):
    """Raised when CUDA context is poisoned and the process should restart."""


def _clear_progress_line_locked():
    """Clear one-line in-place status output (call under _console_lock)."""
    global _progress_line_len
    if _progress_line_len > 0:
        print('\r' + (' ' * _progress_line_len) + '\r', end='', flush=True)
        _progress_line_len = 0


def update_progress_line(message: str):
    """Update a stable one-line status display."""
    global _progress_line_len
    with _console_lock:
        padded = message
        if len(padded) < _progress_line_len:
            padded += ' ' * (_progress_line_len - len(padded))
        print('\r' + padded, end='', flush=True)
        _progress_line_len = max(_progress_line_len, len(message))


def format_progress_line(loop_index: int, total_images: int, chunk_num: int, chunk_vectors: int,
                         chunk_images: int, keypoints_found: int, status_label: str,
                         image_name: str) -> str:
    """Compose one-line progress message with chunk fill and per-image status."""
    chunk_gb = (chunk_vectors * 512) / (1024 ** 3)
    return (
        f"Image {loop_index+1:,}/{total_images:,} | chunk {chunk_num:03d} | "
        f"chunk_img {chunk_images:,} | vec {chunk_vectors:,} ({chunk_gb:0.2f}GB) | "
        f"kp {keypoints_found:,} | {status_label} | {image_name}"
    )


def format_eta(seconds: float) -> str:
    """Human-readable ETA string in days/hours/minutes."""
    if not np.isfinite(seconds) or seconds <= 0:
        return "unknown"
    secs = int(round(seconds))
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def log(msg):
    """Print and log to file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _console_lock:
        _clear_progress_line_locked()
        print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def is_cuda_runtime_error(exc: Exception) -> bool:
    """Detect CUDA runtime failures that can poison the context."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return ("cuda error" in msg) or ("cudnn" in msg) or ("illegal memory access" in msg)


def reload_disk_model(device):
    """Best-effort CUDA recovery by rebuilding DISK model."""
    if device.type == 'cuda':
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    model = KF.DISK.from_pretrained('depth').to(device).eval()
    return model


def normalize_to_nas_path(path):
    """Normalize local/stored path variants to the canonical stored path."""
    p = os.path.normpath(path)
    local_base = os.path.normpath(LOCAL_IMAGES_DIR)
    nas_base = os.path.normpath(NAS_IMAGES_DIR)
    if p.startswith(local_base):
        return nas_base + p[len(local_base):]
    p_fwd = p.replace('\\', '/')
    local_fwd = local_base.replace('\\', '/')
    nas_fwd = nas_base.replace('\\', '/')
    if p_fwd.startswith(local_fwd):
        return nas_fwd + p_fwd[len(local_fwd):]
    return p


def remap_path(local_path):
    """Convert local read path to stored path."""
    return normalize_to_nas_path(local_path)


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
        path_to_id = {normalize_to_nas_path(p): i for i, p in enumerate(id_to_path)}
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
        processed = {normalize_to_nas_path(p) for p in state.get('processed_images', [])}
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


def save_path_lookup(path_to_id, sync_db=True):
    """Save the global path lookup (list indexed by ID) - used during search."""
    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)

    max_pid = max(path_to_id.values(), default=-1)
    id_to_path = [''] * (max_pid + 1)
    for path, pid in path_to_id.items():
        if pid < 0:
            raise ValueError(f"Negative path id for {path}: {pid}")
        id_to_path[pid] = path

    lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    with open(lookup_file, 'w') as f:
        json.dump(id_to_path, f)

    log(f"  Saved path_lookup.json: {len(path_to_id):,} unique paths ({os.path.getsize(lookup_file) / 1e6:.1f} MB)")

    # Sync new paths to DB (throttled at chunk boundaries)
    if sync_db:
        sync_paths_to_db(COLLECTION_NAME, path_to_id)


def load_bad_images_list(list_file, label):
    """Load a persisted quarantine file into a normalized path set."""
    if not os.path.exists(list_file):
        return set()
    bad = set()
    try:
        with open(list_file, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                path = line.split('\t', 1)[0]
                bad.add(normalize_to_nas_path(path))
    except Exception as e:
        log(f"Warning: Could not load {label} image list: {e}")
    return bad


def load_cuda_bad_images():
    """Load image paths that previously poisoned CUDA and should be skipped."""
    return load_bad_images_list(CUDA_BAD_IMAGES_FILE, "CUDA bad")


def load_preprocess_bad_images():
    """Load image paths that consistently fail CPU-side preprocessing."""
    return load_bad_images_list(PREPROCESS_BAD_IMAGES_FILE, "preprocess bad")


def record_bad_image(image_path, reason, bad_images, list_file, label):
    """Append an image path to a quarantine list once."""
    norm_path = normalize_to_nas_path(image_path)
    if norm_path in bad_images:
        return
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    ts = datetime.now().isoformat(timespec='seconds')
    with open(list_file, 'a', encoding='utf-8') as f:
        f.write(f"{norm_path}\t{ts}\t{reason}\n")
    bad_images.add(norm_path)
    log(f"  Quarantined {label} image: {norm_path}")


def record_cuda_bad_image(image_path, reason, bad_images):
    """Append an image path to CUDA quarantine list once."""
    record_bad_image(image_path, reason, bad_images, CUDA_BAD_IMAGES_FILE, "CUDA-poison")


def record_preprocess_bad_image(image_path, reason, bad_images):
    """Append an image path to preprocess quarantine list once."""
    record_bad_image(image_path, reason, bad_images, PREPROCESS_BAD_IMAGES_FILE, "preprocess-fail")


def preprocess_image(image_path, max_dim=MAX_IMAGE_DIM):
    """Load and preprocess image for DISK extraction."""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "imdecode-none"

        h, w = img.shape[:2]

        # Resize only if max_dim is set (used for OOM retry)
        if max_dim is not None and max(h, w) > max_dim:
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
        return tensor, "ok"
    except Exception as e:
        return None, f"preprocess-exception:{type(e).__name__}"


def prefetch_worker(path_queue, result_queue):
    """Background worker: load and preprocess images so GPU never waits for I/O."""
    while True:
        image_path = path_queue.get()
        if image_path is None:  # poison pill
            path_queue.task_done()
            break
        tensor, prep_status = preprocess_image(image_path)
        result_queue.put((image_path, tensor, prep_status))
        path_queue.task_done()


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
    num_vectors = int(sum(desc.shape[0] for desc in all_descriptors))
    if num_vectors != len(all_ids):
        raise RuntimeError(
            f"Chunk {chunk_num:03d} vector/id mismatch: vectors={num_vectors:,}, ids={len(all_ids):,}"
        )

    # Add descriptors incrementally to avoid a second full-size contiguous np.vstack()
    # allocation that can trigger std::bad_alloc on long runs.
    index = faiss.IndexFlatIP(128)
    try:
        for i, desc in enumerate(all_descriptors):
            if desc is None or len(desc) == 0:
                continue
            if desc.dtype != np.float32 or not desc.flags['C_CONTIGUOUS']:
                desc = np.ascontiguousarray(desc, dtype=np.float32)
            index.add(desc)
            all_descriptors[i] = None
            if (i + 1) % 128 == 0:
                gc.collect()
    except MemoryError as e:
        raise MemoryError(
            "FAISS add failed with std::bad_alloc. "
            "Restart with a lower DISK_MAX_VECTORS_PER_CHUNK (e.g. 18000000)."
        ) from e
    finally:
        all_descriptors.clear()
        gc.collect()

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

    # Sync to DB
    sync_chunk_to_db(COLLECTION_NAME, chunk_num, ids_array)

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
    del ids_array, index
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
    log(f"Image resize:    max_dim={MAX_IMAGE_DIM if MAX_IMAGE_DIM is not None else 'none'}")
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
    cuda_bad_images = load_cuda_bad_images()
    preprocess_bad_images = load_preprocess_bad_images()
    known_bad_images = set(cuda_bad_images)
    known_bad_images.update(preprocess_bad_images)
    log(f"Resume: chunk {next_chunk}, {len(processed):,} images already done, {len(path_to_id):,} unique paths")
    if cuda_bad_images:
        log(f"CUDA quarantine list: {len(cuda_bad_images):,} images")
    if preprocess_bad_images:
        log(f"Preprocess quarantine list: {len(preprocess_bad_images):,} images")

    # Filter out already processed
    remaining = []
    for image_path in all_images:
        stored_path = remap_path(image_path)
        if stored_path in processed or stored_path in path_to_id or stored_path in known_bad_images:
            continue
        remaining.append(image_path)
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

    # Start prefetch pipeline: background threads load images while GPU works
    path_queue = queue.Queue()
    result_queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    workers = []
    for _ in range(PREFETCH_WORKERS):
        t = Thread(target=prefetch_worker, args=(path_queue, result_queue), daemon=True)
        t.start()
        workers.append(t)
    for image_path in remaining:
        path_queue.put(image_path)
    for _ in range(PREFETCH_WORKERS):
        path_queue.put(None)  # poison pills
    log(f"Prefetch pipeline started: {PREFETCH_WORKERS} workers, queue depth {PREFETCH_QUEUE_SIZE}")

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
    chunk_pending_processed = set()
    chunk_start = time.time()
    last_heartbeat = time.time()
    cuda_error_streak = 0
    cuda_recovery_attempts = 0
    path_db_unsynced_chunks = 0

    def save_restart_checkpoint():
        committed_processed = processed.difference(chunk_pending_processed)
        save_progress(chunk_num, committed_processed, next_id)

    for i in range(len(remaining)):
        image_path, tensor, prep_status = result_queue.get()
        image_name = os.path.basename(image_path)
        if len(image_name) > 64:
            image_name = image_name[:61] + "..."
        keypoints_found = 0
        status_label = "ok"
        mark_processed = True
        stored_path = remap_path(image_path)

        if tensor is None:
            total_failed += 1
            record_preprocess_bad_image(image_path, prep_status, preprocess_bad_images)
            known_bad_images.add(stored_path)
            if total_failed <= 20:
                log(f"  PREPROCESS FAIL: {image_path} ({prep_status})")
            elif total_failed == 21:
                log(f"  (suppressing further preprocess-fail logs)")
            processed.add(stored_path)
            status_label = "preprocess-fail"
            update_progress_line(format_progress_line(
                i, len(remaining), chunk_num, chunk_vector_count, chunk_image_count,
                0, status_label, image_name
            ))
            continue

        # Extract DISK features (with OOM retry at reduced resolution)
        try:
            t_shape = tensor.shape
            tensor = tensor.to(device)
            with torch.no_grad():
                try:
                    feats = extractor(tensor)[0]
                except torch.cuda.OutOfMemoryError:
                    del tensor
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                    gc.collect()
                    tensor, retry_status = preprocess_image(image_path, max_dim=2048)
                    if tensor is None:
                        total_failed += 1
                        record_preprocess_bad_image(image_path, retry_status, preprocess_bad_images)
                        known_bad_images.add(stored_path)
                        processed.add(stored_path)
                        status_label = "oom-retry-fail"
                        if total_failed <= 20:
                            log(f"  OOM RETRY FAIL: {image_path} ({retry_status})")
                        update_progress_line(format_progress_line(
                            i, len(remaining), chunk_num, chunk_vector_count, chunk_image_count,
                            0, status_label, image_name
                        ))
                        continue
                    tensor = tensor.to(device)
                    feats = extractor(tensor)[0]
                descriptors = feats.descriptors.cpu().numpy()  # (N, 128)
                keypoints_found = int(len(descriptors))

            if len(descriptors) == 0:
                total_failed += 1
                if total_failed <= 20:
                    log(f"  ZERO KEYPOINTS: {image_path} (tensor shape {t_shape})")
                processed.add(stored_path)
                status_label = "no-keypoints"
                del tensor
                update_progress_line(format_progress_line(
                    i, len(remaining), chunk_num, chunk_vector_count, chunk_image_count,
                    0, status_label, image_name
                ))
                continue

            if total_ok < 10 or total_ok % 5000 == 0:
                log(f"  SAMPLE OK: {os.path.basename(image_path)} "
                    f"shape={t_shape} keypoints={len(descriptors)}")

            # Get or assign path ID
            if stored_path not in path_to_id:
                path_to_id[stored_path] = next_id
                next_id += 1
            pid = path_to_id[stored_path]

            # Accumulate
            chunk_descriptors.append(descriptors)
            chunk_ids.extend([pid] * len(descriptors))
            chunk_vector_count += len(descriptors)
            chunk_image_count += 1
            chunk_pending_processed.add(stored_path)
            total_ok += 1
            cuda_error_streak = 0

        except Exception as e:
            total_failed += 1
            status_label = f"extract-{type(e).__name__}"
            if total_failed <= 20:
                log(f"  EXTRACT FAIL: {image_path} - {type(e).__name__}: {e}")

            if is_cuda_runtime_error(e):
                mark_processed = False
                cuda_error_streak += 1
                if "illegal memory access" in str(e).lower():
                    cuda_error_streak = CUDA_ERROR_STREAK_FOR_RECOVERY
                if cuda_error_streak == 1 or (cuda_error_streak % 10 == 0):
                    log(
                        f"  CUDA error streak {cuda_error_streak} "
                        f"(recoveries used: {cuda_recovery_attempts}/{CUDA_MAX_RECOVERY_ATTEMPTS})"
                    )

                if cuda_error_streak >= CUDA_ERROR_STREAK_FOR_RECOVERY:
                    if cuda_recovery_attempts < CUDA_MAX_RECOVERY_ATTEMPTS:
                        cuda_recovery_attempts += 1
                        log(
                            f"  CUDA recovery attempt {cuda_recovery_attempts}/"
                            f"{CUDA_MAX_RECOVERY_ATTEMPTS} after streak={cuda_error_streak}"
                        )
                        try:
                            extractor = reload_disk_model(device)
                            cuda_error_streak = 0
                            log("  CUDA recovery succeeded (DISK model reloaded)")
                        except Exception as reload_err:
                            reason = f"recovery-failed: {type(reload_err).__name__}: {reload_err}"
                            record_cuda_bad_image(stored_path, reason, cuda_bad_images)
                            save_restart_checkpoint()
                            raise CudaPoisonedError(
                                f"CUDA recovery failed after streak={cuda_error_streak}: "
                                f"{type(reload_err).__name__}: {reload_err}"
                            ) from reload_err
                    else:
                        reason = f"streak-exhausted: {type(e).__name__}: {e}"
                        record_cuda_bad_image(stored_path, reason, cuda_bad_images)
                        save_restart_checkpoint()
                        raise CudaPoisonedError(
                            f"Aborting build: CUDA error streak={cuda_error_streak} "
                            f"after {cuda_recovery_attempts} recovery attempts. "
                            f"Last error: {type(e).__name__}: {e}"
                        ) from e

        if mark_processed:
            processed.add(stored_path)
        del tensor
        update_progress_line(format_progress_line(
            i, len(remaining), chunk_num, chunk_vector_count, chunk_image_count,
            keypoints_found, status_label, image_name
        ))

        if (time.time() - last_heartbeat) >= PROGRESS_HEARTBEAT_SEC:
            elapsed = time.time() - start_time
            images_done = i + 1
            images_left = len(remaining) - images_done
            rate = images_done / elapsed if elapsed > 0 else 0.0
            eta_seconds = (images_left / rate) if rate > 0 else float("inf")
            eta_str = format_eta(eta_seconds)
            finish_str = (
                (datetime.now() + timedelta(seconds=eta_seconds)).strftime("%m-%d %H:%M")
                if np.isfinite(eta_seconds) and eta_seconds > 0 else "unknown"
            )
            log(
                f"Heartbeat: image {images_done:,}/{len(remaining):,}, "
                f"chunk {chunk_num:03d}, chunk_vectors={chunk_vector_count:,}, "
                f"chunk_images={chunk_image_count:,}, last_kp={keypoints_found:,}, "
                f"ok={total_ok:,}, failed={total_failed:,}, "
                f"cuda_streak={cuda_error_streak}, cuda_recoveries={cuda_recovery_attempts}, "
                f"eta={eta_str}, finish={finish_str}"
            )
            last_heartbeat = time.time()

        if (i + 1) % PROGRESS_SAVE_EVERY_IMAGES == 0:
            save_restart_checkpoint()

        if (i + 1) % 200 == 0:
            gc.collect()
            if device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass

        # Flush chunk when we hit the vector cap
        if chunk_vector_count >= MAX_VECTORS_PER_CHUNK:
            chunk_time = time.time() - chunk_start

            log(f"\n[Chunk {chunk_num:03d}] Flushing {chunk_vector_count:,} vectors "
                f"from {chunk_image_count} images")

            num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
            total_vectors += num_vectors
            images_processed += chunk_image_count

            # Save progress
            chunk_num += 1
            save_progress(chunk_num, processed, next_id)
            path_db_unsynced_chunks += 1
            sync_db_now = path_db_unsynced_chunks >= PATH_DB_SYNC_EVERY_CHUNKS
            save_path_lookup(path_to_id, sync_db=sync_db_now)
            if sync_db_now:
                path_db_unsynced_chunks = 0
            chunk_pending_processed.clear()

            # ETA
            elapsed = time.time() - start_time
            images_left = len(remaining) - (i + 1)
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta_seconds = (images_left / rate) if rate > 0 else float("inf")

            log(f"  Time: {chunk_time:.0f}s | Total vectors: {total_vectors:,} | "
                f"Unique paths: {len(path_to_id):,} | ETA: {format_eta(eta_seconds)}")

            # Reset accumulator
            chunk_descriptors = []
            chunk_ids = []
            chunk_vector_count = 0
            chunk_image_count = 0
            chunk_start = time.time()

    save_restart_checkpoint()

    # Flush remaining vectors as final chunk
    if chunk_descriptors:
        log(f"\n[Chunk {chunk_num:03d}] Final chunk: {chunk_vector_count:,} vectors "
            f"from {chunk_image_count} images")
        num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
        total_vectors += num_vectors
        images_processed += chunk_image_count
        chunk_num += 1
        save_progress(chunk_num, processed, next_id)
        save_path_lookup(path_to_id, sync_db=True)
        path_db_unsynced_chunks = 0
        chunk_pending_processed.clear()

    # Wait for prefetch workers to finish
    for t in workers:
        t.join()

    # Wait for any pending NAS copy before reporting final stats
    wait_for_nas_copy()
    with _console_lock:
        _clear_progress_line_locked()

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
    try:
        main()
    except CudaPoisonedError as e:
        log(f"CUDA context poisoned: {e}")
        log(f"Exiting with code {CUDA_POISON_EXIT_CODE} for supervisor restart.")
        sys.exit(CUDA_POISON_EXIT_CODE)
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc().rstrip())
        raise
