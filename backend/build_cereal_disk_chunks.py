r"""
Build DISK keypoint chunks for cereal boxes - direct to chunks with compact IDs.

Pipelined: copies batches of folders from NAS to local buffer, processes on GPU,
deletes processed files. Keeps 3 batches buffered so GPU never waits for NAS I/O.
Within each batch, a prefetch queue loads/preprocesses images on CPU threads so
the GPU never waits for disk reads either.

  1. List all subfolders on NAS
  2. Copy batch of folders to local SSD buffer (sized by GB, not folder count)
  3. Extract DISK features per image on GPU (prefetched by background threads)
  4. Delete processed local files, copy next batch
  5. Flush to chunk when hitting ~10 GB (19.5M vectors)

Progress tracked per-image (via path_to_id). Adding new images to existing
folders will be picked up on re-run.

Input:  NAS cereal images (T:/archiverelated/cereal)
Output: chunk_XXX.faiss  -> NAS (T:/faiss/disk_retrieval/cereal_chunks/)
        chunk_XXX_ids.npy -> local SSD (D:/faiss/disk_retrieval/cereal_chunk_ids/)
        path_lookup.json  -> local SSD (same dir as IDs)

Paths stored point to the NAS originals (T:/archiverelated/cereal/...).
Resumable via progress file.
"""

import os
import sys
import gc
import json
import time
import shutil
import subprocess
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import torch
import cv2
import queue
from collections import Counter
from datetime import datetime, timedelta
from threading import Thread, Lock, Event
from collections import deque
from disk_chunk_db import sync_chunk_to_db, sync_paths_to_db, create_tables

try:
    import kornia.feature as KF
    import kornia as K
except ImportError:
    print("ERROR: Kornia not installed. Run: pip install kornia")
    sys.exit(1)


# ============================================================================
# CONFIG
# ============================================================================

# Source: NAS (read from here in batches)
NAS_SOURCE_DIR = r"T:\archiverelated\cereal"

# Local source candidates (if present, process directly without NAS staging)
LOCAL_SOURCE_CANDIDATES = (
    r"C:\cereal",
)

# Local buffer: copy batches here for fast GPU reads
LOCAL_BUFFER_DIR = r"C:\cereal_buffer"

# Path prefix stored in path_lookup (matches NAS source)
NAS_IMAGES_DIR = r"T:\archiverelated\cereal"

# Output: FAISS chunks go to NAS
NAS_CHUNKS_DIR = r"T:\faiss\disk_retrieval\cereal_chunks"
LOCAL_CHUNKS_BUFFER = r"D:\faiss\disk_retrieval\cereal_chunks"

# Output: Compact IDs stay on local SSD
CHUNK_IDS_DIR = r"D:\faiss\disk_retrieval\cereal_chunk_ids"

# Progress tracking
PROGRESS_DIR = CHUNK_IDS_DIR
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "build_progress.json")
LOG_FILE = os.path.join(PROGRESS_DIR, "build_log.txt")
CUDA_BAD_IMAGES_FILE = os.path.join(PROGRESS_DIR, "cuda_bad_images.txt")
PREPROCESS_BAD_IMAGES_FILE = os.path.join(PROGRESS_DIR, "preprocess_bad_images.txt")
BUFFER_MANIFEST_FILE = os.path.join(PROGRESS_DIR, "buffer_manifest.json")

# Chunk sizing
MAX_VECTORS_PER_CHUNK = int(os.environ.get("DISK_MAX_VECTORS_PER_CHUNK", "19500000"))  # ~10 GB

# Collection name for DB sync
COLLECTION_NAME = "cereal"

# Buffer settings - sized by total file size, not folder count
BATCH_SIZE_GB = 20          # Copy ~20 GB of folders per batch
MAX_BUFFERED_BATCHES = 5    # Keep up to 5 batches on local SSD (~100 GB max)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}

# DISK extraction settings
_max_dim_raw = os.environ.get("DISK_MAX_IMAGE_DIM", "4096").strip().lower()
if _max_dim_raw in ("0", "none", ""):
    MAX_IMAGE_DIM = None
else:
    MAX_IMAGE_DIM = int(_max_dim_raw)
GPU_BATCH_SIZE = 1

# Prefetch pipeline: load/preprocess images on CPU threads while GPU works
PREFETCH_WORKERS = 4   # Background threads for image loading
PREFETCH_QUEUE_SIZE = 16  # Max preprocessed tensors held in memory
PROGRESS_HEARTBEAT_SEC = 120  # Periodic heartbeat log cadence
CUDA_ERROR_STREAK_FOR_RECOVERY = max(5, int(os.environ.get("DISK_CUDA_ERROR_STREAK_FOR_RECOVERY", "32")))
CUDA_MAX_RECOVERY_ATTEMPTS = max(0, int(os.environ.get("DISK_CUDA_MAX_RECOVERY_ATTEMPTS", "1")))
CUDA_POISON_EXIT_CODE = int(os.environ.get("DISK_CUDA_POISON_EXIT_CODE", "86"))
PATH_DB_SYNC_EVERY_CHUNKS = max(1, int(os.environ.get("DISK_PATH_DB_SYNC_EVERY_CHUNKS", "5")))

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


def format_progress_line(folder_idx: int, folder_total: int, image_idx: int, image_total: int,
                         chunk_num: int, chunk_vectors: int, chunk_images: int,
                         keypoints_found: int, status_label: str, image_name: str) -> str:
    """Compose single-line per-image status for console visibility."""
    chunk_gb = (chunk_vectors * 512) / (1024 ** 3)
    return (
        f"Folder {folder_idx:,}/{folder_total:,} | image {image_idx:,}/{image_total:,} "
        f"| chunk {chunk_num:03d} | chunk_img {chunk_images:,} "
        f"| vec {chunk_vectors:,} ({chunk_gb:0.2f}GB) "
        f"| kp {keypoints_found:,} | {status_label} | {image_name}"
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


def get_local_source_dir():
    """Return local source directory if one exists, else None."""
    for candidate in LOCAL_SOURCE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def normalize_to_nas_path(path, local_source_dir=None):
    """Normalize local/buffered paths to canonical NAS paths."""
    p = os.path.normpath(path)
    nas_base = os.path.normpath(NAS_IMAGES_DIR)

    def remap_prefix(old_root, new_root):
        old = os.path.normpath(old_root)
        new = os.path.normpath(new_root)
        p_cmp = os.path.normcase(p)
        old_cmp = os.path.normcase(old)
        if p_cmp == old_cmp:
            return new
        if p_cmp.startswith(old_cmp + os.sep):
            return new + p[len(old):]
        return None

    remapped = remap_prefix(NAS_IMAGES_DIR + "_buffer", nas_base)
    if remapped is not None:
        rel = remapped[len(nas_base):].replace('\\', '/')
        rel = re.sub(r'^/\._batch_\d+/', '/', rel)
        return os.path.normpath(nas_base + rel)

    candidates = []
    if local_source_dir:
        candidates.append(os.path.normpath(local_source_dir))
    candidates.extend(os.path.normpath(candidate) for candidate in LOCAL_SOURCE_CANDIDATES)

    seen = set()
    for local_root in candidates:
        key = local_root.lower()
        if key in seen:
            continue
        seen.add(key)
        remapped = remap_prefix(local_root, nas_base)
        if remapped is not None:
            return os.path.normpath(remapped)

    buffer_norm = os.path.normpath(LOCAL_BUFFER_DIR)
    remapped = remap_prefix(buffer_norm, nas_base)
    if remapped is not None:
        relative = remapped[len(nas_base):].replace('\\', '/')
        relative = re.sub(r'^/\._batch_\d+/', '/', relative)
        return os.path.normpath(nas_base + relative)

    return p


def make_nas_path(local_path, local_source_dir=None):
    """Convert local buffer path to NAS storage path."""
    return normalize_to_nas_path(local_path, local_source_dir=local_source_dir)


def find_images_recursive(directory):
    """Find all images recursively in a directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(root, f))
    return sorted(files)


def count_images_in_source(folder_name, source_dir):
    """Count image files in a source folder (recursive)."""
    count = 0
    folder_path = os.path.join(source_dir, folder_name)
    try:
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    count += 1
    except Exception:
        pass
    return count


def get_folder_size_bytes(folder_path):
    """Get total size of a folder in bytes."""
    total = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except Exception:
        pass
    return total


class FolderBuffer:
    """Manages pipelined copying of folders from NAS to local SSD, sized by GB."""

    def __init__(self, all_folders, source_dir=NAS_SOURCE_DIR, root_batch_map=None, manifest_state=None):
        self.source_dir = source_dir
        self.root_batch_map = dict(root_batch_map or {})
        self.pending = deque()
        self.ready = deque()        # (folder_name, size_bytes) copied and ready
        self.processing = {}        # folder_name -> size_bytes (0 while copy is in-flight)
        self.lock = Lock()
        self.stop = Event()
        self.thread = None
        self.copied_count = 0
        self.copied_bytes = 0
        self.ready_bytes = 0
        self.processing_bytes = 0
        self.errors = []
        self._restore_or_seed(all_folders, manifest_state)

    def _restore_or_seed(self, all_folders, manifest_state):
        restored = False
        if manifest_state and manifest_has_work(manifest_state):
            restored = self._restore_from_manifest(manifest_state)
        if not restored:
            self.pending = deque(all_folders)
            self._persist_manifest_locked()

    def _restore_from_manifest(self, manifest_state):
        manifest_root_batch_map = manifest_state.get('root_batch_map', {})
        if manifest_root_batch_map:
            self.root_batch_map = manifest_root_batch_map

        seen = set()

        def restore_ready(folder):
            if folder in seen:
                return
            local_dir = os.path.join(LOCAL_BUFFER_DIR, folder)
            if os.path.exists(local_dir):
                actual_size = get_folder_size_bytes(local_dir)
                if actual_size > 0:
                    self.ready.append((folder, actual_size))
                    self.ready_bytes += actual_size
                    self.copied_count += 1
                    self.copied_bytes += actual_size
                    seen.add(folder)
                    return
            self.pending.append(folder)
            seen.add(folder)

        for folder in manifest_state.get('pending_units', []):
            if folder not in seen:
                self.pending.append(folder)
                seen.add(folder)

        for entry in manifest_state.get('ready_units', []):
            folder = entry.get('folder')
            if folder:
                restore_ready(folder)

        for entry in manifest_state.get('processing_units', []):
            folder = entry.get('folder')
            if folder:
                restore_ready(folder)

        self._persist_manifest_locked()
        return bool(self.pending or self.ready or self.processing)

    def _persist_manifest_locked(self):
        save_buffer_manifest({
            'source_dir': self.source_dir,
            'root_batch_map': self.root_batch_map,
            'pending_units': list(self.pending),
            'ready_units': [
                {'folder': folder, 'size_bytes': int(size)}
                for folder, size in self.ready
            ],
            'processing_units': [
                {'folder': folder, 'size_bytes': int(size)}
                for folder, size in self.processing.items()
            ],
        })

    def start(self):
        self.thread = Thread(target=self._copy_worker, daemon=True)
        self.thread.start()

    def stop_copying(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=10)

    def _copy_worker(self):
        """Background thread: copy folders from NAS to local buffer."""
        batch_size_bytes = int(BATCH_SIZE_GB * 1024**3)
        while not self.stop.is_set():
            with self.lock:
                buffered = self.ready_bytes + self.processing_bytes
                need_more = buffered < MAX_BUFFERED_BATCHES * batch_size_bytes

            if not need_more:
                time.sleep(0.5)
                continue

            if not self.pending:
                time.sleep(1)
                with self.lock:
                    if not self.pending:
                        break
                continue

            # Copy folders until we hit the batch size target
            batch_bytes = 0
            while batch_bytes < batch_size_bytes:
                with self.lock:
                    if not self.pending:
                        break
                    folder = self.pending.popleft()
                    self.processing[folder] = 0
                    self._persist_manifest_locked()

                if self.stop.is_set():
                    break

                is_root_batch = folder in self.root_batch_map
                if is_root_batch:
                    src = self.source_dir
                    dst = os.path.join(LOCAL_BUFFER_DIR, folder)
                else:
                    src = os.path.join(self.source_dir, folder)
                    dst = os.path.join(LOCAL_BUFFER_DIR, folder)

                try:
                    if os.path.exists(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    if is_root_batch:
                        # Copy specific files from root into virtual batch folder
                        os.makedirs(dst, exist_ok=True)
                        batch_files = self.root_batch_map[folder]
                        def _copy_one(fn):
                            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                        with ThreadPoolExecutor(max_workers=16) as pool:
                            list(pool.map(_copy_one, batch_files))
                    else:
                        shutil.copytree(src, dst)
                    folder_bytes = get_folder_size_bytes(dst)
                    batch_bytes += folder_bytes
                    with self.lock:
                        self.processing.pop(folder, None)
                        self.ready.append((folder, folder_bytes))
                        self.ready_bytes += folder_bytes
                        self.copied_count += 1
                        self.copied_bytes += folder_bytes
                        self._persist_manifest_locked()
                except Exception as e:
                    with self.lock:
                        self.processing.pop(folder, None)
                        self._persist_manifest_locked()
                    self.errors.append((folder, str(e)))

    def get_next_batch(self, timeout=120):
        """Get folders ready for processing (up to one batch worth)."""
        batch_size_bytes = int(BATCH_SIZE_GB * 1024**3)
        start = time.time()
        batch = []
        batch_bytes = 0
        while time.time() - start < timeout:
            with self.lock:
                while self.ready and batch_bytes < batch_size_bytes:
                    folder, size = self.ready.popleft()
                    self.ready_bytes -= size
                    self.processing[folder] = size
                    self.processing_bytes += size
                    batch.append((folder, size))
                    batch_bytes += size
                if batch:
                    self._persist_manifest_locked()
            if batch:
                return batch
            if not self.has_more():
                return batch
            time.sleep(0.5)
        return batch

    def mark_done(self, folder, size):
        """Delete a processed folder from local buffer."""
        with self.lock:
            size = self.processing.pop(folder, size)
            self.processing_bytes -= size
            self._persist_manifest_locked()
        local_dir = os.path.join(LOCAL_BUFFER_DIR, folder)
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir, ignore_errors=True)
        if not self.has_more():
            clear_buffer_manifest()

    def has_more(self):
        with self.lock:
            return bool(self.pending or self.ready or self.processing)

    def status(self):
        with self.lock:
            return {
                'pending': len(self.pending),
                'ready': len(self.ready),
                'ready_gb': self.ready_bytes / (1024**3),
                'processing': len(self.processing),
                'copied': self.copied_count,
                'copied_gb': self.copied_bytes / (1024**3),
                'errors': len(self.errors),
            }


class LocalFolderBuffer:
    """Local-mode folder iterator without copy/delete staging."""

    def __init__(self, all_folders):
        self.pending = deque(all_folders)
        self.processing = set()
        self.done = 0

    def start(self):
        return

    def stop_copying(self):
        return

    def get_next_batch(self, timeout=120):
        if not self.pending:
            return []
        folder = self.pending.popleft()
        self.processing.add(folder)
        return [(folder, 0)]

    def mark_done(self, folder, size):
        self.processing.discard(folder)
        self.done += 1

    def has_more(self):
        return bool(self.pending or self.processing)

    def status(self):
        return {
            'pending': len(self.pending),
            'ready': 0,
            'ready_gb': 0.0,
            'processing': len(self.processing),
            'copied': self.done,
            'copied_gb': 0.0,
            'errors': 0,
        }


def load_progress():
    """Load build progress. Image-level tracking via path_to_id."""
    default = (1, {}, {}, 0)
    if not os.path.exists(PROGRESS_FILE):
        return default

    try:
        with open(PROGRESS_FILE, 'r') as f:
            state = json.load(f)
        next_chunk = state.get('next_chunk', 1)
        next_id = state.get('next_id', 0)
        scanned_counts = state.get('scanned_folder_counts', {})

        path_to_id = {}
        lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
        if os.path.exists(lookup_file):
            with open(lookup_file, 'r') as f:
                id_to_path = json.load(f)
            path_to_id = {
                normalize_to_nas_path(p): i
                for i, p in enumerate(id_to_path)
                if p
            }
            next_id = max(next_id, max(path_to_id.values(), default=-1) + 1)

        return next_chunk, scanned_counts, path_to_id, next_id
    except Exception as e:
        log(f"Warning: Could not load progress: {e}")
        return default


def save_progress(next_chunk, scanned_counts, next_id):
    """Save build progress."""
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    state = {
        'next_chunk': next_chunk,
        'next_id': next_id,
        'scanned_folder_count': len(scanned_counts),
        'last_updated': datetime.now().isoformat(),
        'scanned_folder_counts': scanned_counts,
    }
    temp = PROGRESS_FILE + '.tmp'
    with open(temp, 'w') as f:
        json.dump(state, f)
    shutil.move(temp, PROGRESS_FILE)


def load_buffer_manifest():
    if not os.path.exists(BUFFER_MANIFEST_FILE):
        return None
    try:
        with open(BUFFER_MANIFEST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log(f"Warning: Could not load buffer manifest: {e}")
        return None


def manifest_has_work(state):
    if not state:
        return False
    return bool(
        state.get('pending_units') or
        state.get('ready_units') or
        state.get('processing_units')
    )


def save_buffer_manifest(state):
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    temp = BUFFER_MANIFEST_FILE + '.tmp'
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(state, f)
    shutil.move(temp, BUFFER_MANIFEST_FILE)


def clear_buffer_manifest():
    try:
        if os.path.exists(BUFFER_MANIFEST_FILE):
            os.remove(BUFFER_MANIFEST_FILE)
    except OSError:
        pass


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
    log(f"  Saved path_lookup.json: {len(path_to_id):,} unique paths "
        f"({os.path.getsize(lookup_file) / 1e6:.1f} MB)")

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
        if max_dim is not None and max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
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


# Background NAS chunk copy
_nas_copy_thread = None
_nas_copy_error = None


def _nas_copy_worker(local_faiss, nas_faiss, chunk_num):
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
    global _nas_copy_thread, _nas_copy_error
    if _nas_copy_thread is not None:
        _nas_copy_thread.join()
        _nas_copy_thread = None
        if _nas_copy_error:
            log(f"  WARNING: Previous NAS copy had error: {_nas_copy_error}")
            _nas_copy_error = None


def save_chunk(chunk_num, all_descriptors, all_ids, num_images):
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

    os.makedirs(LOCAL_CHUNKS_BUFFER, exist_ok=True)
    local_faiss = os.path.join(LOCAL_CHUNKS_BUFFER, f"chunk_{chunk_num:03d}.faiss")
    faiss.write_index(index, local_faiss)
    faiss_size = os.path.getsize(local_faiss) / (1024**3)

    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)
    ids_array = np.array(all_ids, dtype=np.int32)
    ids_file = os.path.join(CHUNK_IDS_DIR, f"chunk_{chunk_num:03d}_ids.npy")
    np.save(ids_file, ids_array)
    ids_size = os.path.getsize(ids_file) / (1024**2)

    # Sync to DB
    sync_chunk_to_db(COLLECTION_NAME, chunk_num, ids_array)

    log(f"  Chunk {chunk_num:03d}: {num_vectors:,} vectors from {num_images} images "
        f"({faiss_size:.1f} GB index, {ids_size:.0f} MB IDs)")

    os.makedirs(NAS_CHUNKS_DIR, exist_ok=True)
    nas_faiss = os.path.join(NAS_CHUNKS_DIR, f"chunk_{chunk_num:03d}.faiss")
    _nas_copy_thread = Thread(target=_nas_copy_worker, args=(local_faiss, nas_faiss, chunk_num))
    _nas_copy_thread.daemon = True
    _nas_copy_thread.start()

    del ids_array, index
    gc.collect()
    return num_vectors


def find_folders_with_new_images(all_folders, scanned_counts, path_to_id, source_dir, bad_images=None):
    """Find folders that have unprocessed images."""
    remaining = []
    skipped = 0
    bad_images = bad_images or set()

    known_per_folder = Counter()
    prefix = NAS_IMAGES_DIR.replace('\\', '/') + '/'
    for nas_path in path_to_id:
        normalized = normalize_to_nas_path(nas_path).replace('\\', '/')
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):]
            parts = rel.split('/')
            folder = parts[0] if len(parts) > 1 else "."
            known_per_folder[folder] += 1
    for nas_path in bad_images:
        normalized = normalize_to_nas_path(nas_path).replace('\\', '/')
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):]
            parts = rel.split('/')
            folder = parts[0] if len(parts) > 1 else "."
            known_per_folder[folder] += 1

    for i, folder in enumerate(all_folders):
        if (i + 1) % 5000 == 0:
            print(f"\r  Checking folders: {i+1:,}/{len(all_folders):,} "
                  f"({skipped:,} skipped, {len(remaining):,} to process)...",
                  end="", flush=True)

        # Virtual batch folders (._batch_XX) are never skipped; dedup is per-image
        if folder.startswith("._batch_"):
            remaining.append(folder)
            continue

        if folder in scanned_counts:
            current_count = count_images_in_source(folder, source_dir)
            known_count = known_per_folder.get(folder, 0)
            if current_count == scanned_counts[folder] and current_count == known_count:
                skipped += 1
                continue
        elif known_per_folder.get(folder, 0) > 0:
            current_count = count_images_in_source(folder, source_dir)
            known_count = known_per_folder.get(folder, 0)
            if current_count > 0 and known_count >= current_count:
                skipped += 1
                continue

        remaining.append(folder)

    if len(all_folders) > 5000:
        print()
    return remaining, skipped


def main():
    local_source_dir = get_local_source_dir()
    local_mode = local_source_dir is not None
    source_dir = local_source_dir if local_mode else NAS_SOURCE_DIR

    log("=" * 70)
    log("CEREAL DISK CHUNK BUILDER (Pipelined NAS -> Local -> GPU)")
    if local_mode:
        log(f"Local source:    {local_source_dir}")
    else:
        log(f"NAS source:      {NAS_SOURCE_DIR}")
        log(f"Local buffer:    {LOCAL_BUFFER_DIR}")
    log(f"Paths stored as: {NAS_IMAGES_DIR}")
    log(f"Chunks output:   {NAS_CHUNKS_DIR}")
    log(f"Compact IDs:     {CHUNK_IDS_DIR}")
    log(f"Target/chunk:    {MAX_VECTORS_PER_CHUNK:,} vectors (~10 GB)")
    log(f"Image resize:    max_dim={MAX_IMAGE_DIM if MAX_IMAGE_DIM is not None else 'none'}")
    if local_mode:
        log("Source mode:     local_direct")
    else:
        log(f"Source mode:     nas_buffer ({BATCH_SIZE_GB} GB/batch, {MAX_BUFFERED_BATCHES} batches max)")
    log("=" * 70)

    # Load progress
    next_chunk, scanned_counts, path_to_id, next_id = load_progress()
    cuda_bad_images = load_cuda_bad_images()
    preprocess_bad_images = load_preprocess_bad_images()
    known_bad_images = set(cuda_bad_images)
    known_bad_images.update(preprocess_bad_images)
    log(f"Resume: chunk {next_chunk}, {len(path_to_id):,} images already processed, "
        f"{len(scanned_counts):,} folders fully scanned")
    if cuda_bad_images:
        log(f"CUDA quarantine list: {len(cuda_bad_images):,} images")
    if preprocess_bad_images:
        log(f"Preprocess quarantine list: {len(preprocess_bad_images):,} images")

    root_batch_map = {}
    manifest_state = None
    if not local_mode:
        manifest_state = load_buffer_manifest()

    if not local_mode and manifest_has_work(manifest_state):
        remaining = []
        seen = set()
        for folder in manifest_state.get('pending_units', []):
            if folder not in seen:
                remaining.append(folder)
                seen.add(folder)
        for entry in manifest_state.get('ready_units', []):
            folder = entry.get('folder')
            if folder and folder not in seen:
                remaining.append(folder)
                seen.add(folder)
        for entry in manifest_state.get('processing_units', []):
            folder = entry.get('folder')
            if folder and folder not in seen:
                remaining.append(folder)
                seen.add(folder)
        root_batch_map = manifest_state.get('root_batch_map', {})
        log(f"Resuming from buffer manifest: {len(remaining):,} queued/staged units")
    else:
        # List all folders in selected source
        if not os.path.exists(source_dir):
            log(f"ERROR: Source directory not found: {source_dir}")
            return
        log(f"Scanning source for folders: {source_dir}")
        all_folders = sorted([d for d in os.listdir(source_dir)
                             if os.path.isdir(os.path.join(source_dir, d))])
        root_image_files = []
        try:
            with os.scandir(source_dir) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False) and \
                       os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                        root_image_files.append(entry.name)
        except OSError:
            pass
        if root_image_files:
            root_image_files.sort()
            FLAT_BATCH_SIZE = 10000
            batch_names = []
            for i in range(0, len(root_image_files), FLAT_BATCH_SIZE):
                batch_name = f"._batch_{i // FLAT_BATCH_SIZE:02d}"
                batch_files = root_image_files[i:i + FLAT_BATCH_SIZE]
                root_batch_map[batch_name] = batch_files
                batch_names.append(batch_name)
            all_folders = batch_names + all_folders
            log(f"Root contains {len(root_image_files):,} images; split into {len(root_batch_map)} "
                f"virtual batches ({FLAT_BATCH_SIZE:,} files each)")
        log(f"Found {len(all_folders):,} folders in source")

        # Find folders with new/unprocessed images
        log("Checking for folders with new images...")
        remaining, skipped = find_folders_with_new_images(
            all_folders,
            scanned_counts,
            path_to_id,
            source_dir,
            bad_images=known_bad_images,
        )
        log(f"Skipped {skipped:,} unchanged folders, {len(remaining):,} folders to process")

    if not remaining:
        log("All images already processed!")
        if not local_mode:
            clear_buffer_manifest()
        return

    # Clean local buffer (NAS-buffer mode only)
    if not local_mode:
        if manifest_has_work(manifest_state):
            os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)
            log("Reusing staged local buffer from manifest")
        elif os.path.exists(LOCAL_BUFFER_DIR):
            log("Cleaning local buffer...")
            shutil.rmtree(LOCAL_BUFFER_DIR, ignore_errors=True)
            os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)
        else:
            os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)

    # Load DISK model
    log("Loading DISK model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = KF.DISK.from_pretrained('depth').to(device).eval()
    log(f"DISK model loaded on {device}")
    log("=" * 70)

    # Start source buffer
    if local_mode:
        buffer = LocalFolderBuffer(remaining)
    else:
        buffer = FolderBuffer(
            remaining,
            source_dir=source_dir,
            root_batch_map=root_batch_map,
            manifest_state=manifest_state,
        )
    buffer.start()

    # Start prefetch pipeline (local SSD -> preprocessed tensors)
    path_queue = queue.Queue()
    result_queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    prefetch_threads = []
    for _ in range(PREFETCH_WORKERS):
        t = Thread(target=prefetch_worker, args=(path_queue, result_queue), daemon=True)
        t.start()
        prefetch_threads.append(t)
    log(f"Prefetch pipeline started: {PREFETCH_WORKERS} workers, queue depth {PREFETCH_QUEUE_SIZE}")

    # Wait for first staged batch (NAS mode only)
    if not local_mode:
        log("Waiting for first batch to copy...")
        while True:
            s = buffer.status()
            if s['ready_gb'] >= BATCH_SIZE_GB * 0.5 or s['ready'] >= len(remaining):
                break
            print(f"\r    Copied {s['copied']:,} folders ({s['copied_gb']:.1f} GB)...",
                  end="", flush=True)
            time.sleep(1)
        print()

    start_time = time.time()
    total_vectors = 0
    total_ok = 0
    total_skipped = 0
    total_failed = 0
    total_images = 0
    chunk_num = next_chunk
    folders_done = 0

    # Chunk accumulator
    chunk_descriptors = []
    chunk_ids = []
    chunk_vector_count = 0
    chunk_image_count = 0
    chunk_start = time.time()
    last_heartbeat = time.time()
    cuda_error_streak = 0
    cuda_recovery_attempts = 0
    path_db_unsynced_chunks = 0

    def save_restart_checkpoint():
        save_progress(chunk_num, scanned_counts, next_id)

    while buffer.has_more():
        batch = buffer.get_next_batch()
        if not batch:
            if buffer.has_more():
                s = buffer.status()
                # A single large folder/unit can take longer than get_next_batch timeout.
                if s['pending'] == 0 and s['ready'] == 0 and s['processing'] == 0 and s['errors'] > 0:
                    sample = buffer.errors[0] if getattr(buffer, "errors", None) else ("unknown", "unknown copy error")
                    raise RuntimeError(
                        f"Staging failed: no folders available and no in-flight work. "
                        f"errors={s['errors']}. First error: {sample[0]} -> {sample[1]}"
                    )
                log(
                    f"No staged batch ready yet (pending {s['pending']}, ready {s['ready']}, "
                    f"processing {s['processing']}, copied {s['copied']}, errors {s['errors']}); waiting..."
                )
                continue
            break

        for folder, folder_size in batch:
            folder_dir = os.path.join(local_source_dir if local_mode else LOCAL_BUFFER_DIR, folder)
            images = find_images_recursive(folder_dir)

            # Filter to only unprocessed images and feed into prefetch queue
            to_process = []
            for image_path in images:
                nas_path = make_nas_path(image_path, local_source_dir=local_source_dir)
                if nas_path in path_to_id or nas_path in known_bad_images:
                    total_skipped += 1
                    continue
                to_process.append(image_path)
                path_queue.put(image_path)

            folder_image_total = len(to_process)

            # Consume prefetched results from GPU
            for j in range(folder_image_total):
                image_path, tensor, prep_status = result_queue.get()
                total_images += 1
                folder_image_idx = j + 1
                nas_path = make_nas_path(image_path, local_source_dir=local_source_dir)
                image_name = os.path.basename(image_path)
                if len(image_name) > 64:
                    image_name = image_name[:61] + "..."
                keypoints_found = 0
                status_label = "ok"

                if tensor is None:
                    total_failed += 1
                    record_preprocess_bad_image(image_path, prep_status, preprocess_bad_images)
                    known_bad_images.add(nas_path)
                    if total_failed <= 20:
                        log(f"  PREPROCESS FAIL: {image_path} ({prep_status})")
                    elif total_failed == 21:
                        log(f"  (suppressing further preprocess-fail logs)")
                    status_label = "preprocess-fail"
                    update_progress_line(format_progress_line(
                        folders_done + 1, len(remaining), folder_image_idx, folder_image_total,
                        chunk_num, chunk_vector_count, chunk_image_count,
                        0, status_label, image_name
                    ))
                    continue

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
                                known_bad_images.add(nas_path)
                                status_label = "oom-retry-fail"
                                if total_failed <= 20:
                                    log(f"  OOM RETRY FAIL: {image_path} ({retry_status})")
                                update_progress_line(format_progress_line(
                                    folders_done + 1, len(remaining), folder_image_idx, folder_image_total,
                                    chunk_num, chunk_vector_count, chunk_image_count,
                                    0, status_label, image_name
                                ))
                                continue
                            tensor = tensor.to(device)
                            feats = extractor(tensor)[0]
                        descriptors = feats.descriptors.cpu().numpy()
                        keypoints_found = int(len(descriptors))

                    if len(descriptors) == 0:
                        total_failed += 1
                        if total_failed <= 20:
                            log(f"  ZERO KEYPOINTS: {image_path} (tensor shape {t_shape})")
                        status_label = "no-keypoints"
                        del tensor
                        update_progress_line(format_progress_line(
                            folders_done + 1, len(remaining), folder_image_idx, folder_image_total,
                            chunk_num, chunk_vector_count, chunk_image_count,
                            0, status_label, image_name
                        ))
                        continue

                    if total_ok < 10 or total_ok % 5000 == 0:
                        log(f"  SAMPLE OK: {os.path.basename(image_path)} "
                            f"shape={t_shape} keypoints={len(descriptors)}")

                    path_to_id[nas_path] = next_id
                    next_id += 1
                    pid = path_to_id[nas_path]

                    chunk_descriptors.append(descriptors)
                    chunk_ids.extend([pid] * len(descriptors))
                    chunk_vector_count += len(descriptors)
                    chunk_image_count += 1
                    total_ok += 1
                    cuda_error_streak = 0

                except Exception as e:
                    total_failed += 1
                    status_label = f"extract-{type(e).__name__}"
                    if total_failed <= 20:
                        log(f"  EXTRACT FAIL: {image_path} - {type(e).__name__}: {e}")

                    if is_cuda_runtime_error(e):
                        if "illegal memory access" in str(e).lower():
                            cuda_error_streak = CUDA_ERROR_STREAK_FOR_RECOVERY
                        else:
                            cuda_error_streak += 1
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
                                    record_cuda_bad_image(nas_path, reason, cuda_bad_images)
                                    save_restart_checkpoint()
                                    raise CudaPoisonedError(
                                        f"CUDA recovery failed after streak={cuda_error_streak}: "
                                        f"{type(reload_err).__name__}: {reload_err}"
                                    ) from reload_err
                            else:
                                reason = f"streak-exhausted: {type(e).__name__}: {e}"
                                record_cuda_bad_image(nas_path, reason, cuda_bad_images)
                                save_restart_checkpoint()
                                raise CudaPoisonedError(
                                    f"Aborting build: CUDA error streak={cuda_error_streak} "
                                    f"after {cuda_recovery_attempts} recovery attempts. "
                                    f"Last error: {type(e).__name__}: {e}"
                                ) from e

                del tensor
                update_progress_line(format_progress_line(
                    folders_done + 1, len(remaining), folder_image_idx, folder_image_total,
                    chunk_num, chunk_vector_count, chunk_image_count,
                    keypoints_found, status_label, image_name
                ))

                if (time.time() - last_heartbeat) >= PROGRESS_HEARTBEAT_SEC:
                    elapsed = time.time() - start_time
                    partial_folder = (folder_image_idx / max(folder_image_total, 1))
                    approx_folders_done = folders_done + partial_folder
                    folders_left = max(0.0, len(remaining) - approx_folders_done)
                    rate = approx_folders_done / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (folders_left / rate) if rate > 0 else float("inf")
                    eta_str = format_eta(eta_seconds)
                    finish_str = (
                        (datetime.now() + timedelta(seconds=eta_seconds)).strftime("%m-%d %H:%M")
                        if np.isfinite(eta_seconds) and eta_seconds > 0 else "unknown"
                    )
                    s = buffer.status()
                    log(
                        f"Heartbeat: folder {folders_done+1:,}/{len(remaining):,} "
                        f"image {folder_image_idx:,}/{folder_image_total:,}, "
                        f"chunk {chunk_num:03d}, chunk_vectors={chunk_vector_count:,}, "
                        f"chunk_images={chunk_image_count:,}, last_kp={keypoints_found:,}, "
                        f"new={total_ok:,}, failed={total_failed:,}, skipped={total_skipped:,}, "
                        f"buffer_ready={s['ready_gb']:.1f}GB, "
                        f"cuda_streak={cuda_error_streak}, "
                        f"cuda_recoveries={cuda_recovery_attempts}, "
                        f"eta={eta_str}, finish={finish_str}"
                    )
                    last_heartbeat = time.time()

                if total_images % 200 == 0:
                    gc.collect()
                    if device.type == 'cuda':
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            pass

                # Flush chunk
                if chunk_vector_count >= MAX_VECTORS_PER_CHUNK:
                    chunk_time = time.time() - chunk_start
                    log(f"\n[Chunk {chunk_num:03d}] Flushing {chunk_vector_count:,} vectors "
                        f"from {chunk_image_count} images")

                    num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
                    total_vectors += num_vectors

                    chunk_num += 1
                    save_progress(chunk_num, scanned_counts, next_id)
                    path_db_unsynced_chunks += 1
                    sync_db_now = path_db_unsynced_chunks >= PATH_DB_SYNC_EVERY_CHUNKS
                    save_path_lookup(path_to_id, sync_db=sync_db_now)
                    if sync_db_now:
                        path_db_unsynced_chunks = 0

                    elapsed = time.time() - start_time
                    partial_folder = (folder_image_idx / max(folder_image_total, 1))
                    approx_folders_done = folders_done + partial_folder
                    folders_left = max(0.0, len(remaining) - approx_folders_done)
                    rate = approx_folders_done / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (folders_left / rate) if rate > 0 else float("inf")
                    s = buffer.status()

                    log(f"  Time: {chunk_time:.0f}s | Vectors: {total_vectors:,} | "
                        f"Folders: {folders_done:,}/{len(remaining):,} | "
                        f"Buffer: {s['ready_gb']:.1f} GB ready | ETA: {format_eta(eta_seconds)}")

                    chunk_descriptors = []
                    chunk_ids = []
                    chunk_vector_count = 0
                    chunk_image_count = 0
                    chunk_start = time.time()

            # Folder done - record image count for skip optimization on re-run
            scanned_counts[folder] = len(images)
            buffer.mark_done(folder, folder_size)
            folders_done += 1

            if folder_image_total == 0:
                folder_label = folder if len(folder) <= 64 else (folder[:61] + "...")
                update_progress_line(format_progress_line(
                    folders_done, len(remaining), 0, 0, chunk_num, chunk_vector_count,
                    chunk_image_count, 0, "folder-skip", folder_label
                ))

        # Save progress after each batch
        save_progress(chunk_num, scanned_counts, next_id)

    # Stop folder buffer
    buffer.stop_copying()

    # Stop prefetch workers
    for _ in range(PREFETCH_WORKERS):
        path_queue.put(None)
    for t in prefetch_threads:
        t.join()

    # Flush final chunk
    if chunk_descriptors:
        log(f"\n[Chunk {chunk_num:03d}] Final chunk: {chunk_vector_count:,} vectors "
            f"from {chunk_image_count} images")
        num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
        total_vectors += num_vectors
        chunk_num += 1
        save_progress(chunk_num, scanned_counts, next_id)
        save_path_lookup(path_to_id, sync_db=True)
        path_db_unsynced_chunks = 0

    wait_for_nas_copy()
    with _console_lock:
        _clear_progress_line_locked()

    # Cleanup buffer (NAS mode only)
    if (not local_mode) and os.path.exists(LOCAL_BUFFER_DIR):
        shutil.rmtree(LOCAL_BUFFER_DIR, ignore_errors=True)

    # Summary
    elapsed = time.time() - start_time
    chunks_created = chunk_num - next_chunk
    log("\n" + "=" * 70)
    log("BUILD COMPLETE!")
    log(f"  Chunks created: {chunks_created}")
    log(f"  Total vectors:  {total_vectors:,}")
    log(f"  Images new:     {total_ok:,}")
    log(f"  Images skipped: {total_skipped:,}")
    log(f"  Images failed:  {total_failed:,}")
    log(f"  Folders done:   {folders_done:,}")
    log(f"  Unique paths:   {len(path_to_id):,}")
    if chunks_created > 0:
        log(f"  Avg chunk size: {total_vectors / chunks_created:,.0f} vectors "
            f"(~{total_vectors * 512 / chunks_created / (1024**3):.1f} GB)")
    log(f"  Total time:     {elapsed / 3600:.1f} hours")
    log(f"  Chunks at:      {NAS_CHUNKS_DIR}")
    log(f"  Compact IDs at: {CHUNK_IDS_DIR}")
    if buffer.errors:
        log(f"  Copy errors:    {len(buffer.errors)}")
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
