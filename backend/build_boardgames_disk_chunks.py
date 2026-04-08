r"""
Build DISK keypoint chunks for board games with dual source modes.

Mode A (preferred if present): process local `C:/boardgames` directly.
Mode B (fallback): stage folders from NAS `T:/archiverelated/board games`
into `C:/boardgames-temp` with a bounded ~100GB temp buffer.

  1. Scan source folders for new/unprocessed images
  2. (NAS mode) copy folders into local temp buffer in batches
  3. Extract DISK features per image on GPU
  4. Flush to chunk when hitting ~10GB (19.5M vectors)
  5. Continue until source is exhausted

Input:  Local `C:/boardgames` OR NAS `T:/archiverelated/board games`
Output: chunk_XXX.faiss  -> NAS (T:/faiss/disk_retrieval/boardgames_chunks/)
        chunk_XXX_ids.npy -> local SSD (D:/faiss/disk_retrieval/boardgames_chunk_ids/)
        path_lookup.json  -> local SSD (same dir as IDs)

Paths stored point to the NAS originals (T:/archiverelated/board games/...).
Resumable via progress file.
"""

import os
import sys
import gc
import json
import re
import time
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import torch
import cv2
from datetime import datetime, timedelta
import queue
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
# CONFIG - Edit these paths as needed
# ============================================================================

# Source: NAS originals
NAS_SOURCE_DIR = r"T:\archiverelated\board games"

# Local temp buffer for staged processing
LOCAL_BUFFER_DIR = r"C:\boardgames-temp"

# Legacy local source path (older progress files may contain this prefix)
LEGACY_LOCAL_IMAGES_DIR = r"C:\boardgames"

# Path remapping: stored paths point to NAS originals
NAS_IMAGES_DIR = r"T:\archiverelated\board games"

# Output: FAISS chunks go to NAS (searched via rolling buffer copy)
NAS_CHUNKS_DIR = r"T:\faiss\disk_retrieval\boardgames_chunks"
LOCAL_CHUNKS_BUFFER = r"D:\faiss\disk_retrieval\boardgames_chunks"  # Write here first, then copy to NAS

# Output: Compact IDs stay on local SSD (fast reads during search)
CHUNK_IDS_DIR = r"D:\faiss\disk_retrieval\boardgames_chunk_ids"

# Progress tracking
PROGRESS_DIR = CHUNK_IDS_DIR
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "build_progress.json")
LOG_FILE = os.path.join(PROGRESS_DIR, "build_log.txt")
CUDA_BAD_IMAGES_FILE = os.path.join(PROGRESS_DIR, "cuda_bad_images.txt")
PREPROCESS_BAD_IMAGES_FILE = os.path.join(PROGRESS_DIR, "preprocess_bad_images.txt")
BUFFER_MANIFEST_FILE = os.path.join(PROGRESS_DIR, "buffer_manifest.json")

# Chunk sizing: target ~10GB per chunk for GPU FAISS (fits in 16GB VRAM with headroom)
# 10GB = ~19.5M vectors at 128 dims * 4 bytes = 512 bytes/vector.
# Can be tuned at runtime to reduce RAM pressure during flush.
MAX_VECTORS_PER_CHUNK = int(os.environ.get("DISK_MAX_VECTORS_PER_CHUNK", "19500000"))

# Collection name for DB sync
COLLECTION_NAME = "board_games"

# DISK extraction settings
_max_dim_raw = os.environ.get("DISK_MAX_IMAGE_DIM", "4096").strip().lower()
if _max_dim_raw in ("0", "none", ""):
    MAX_IMAGE_DIM = None
else:
    MAX_IMAGE_DIM = int(_max_dim_raw)
GPU_BATCH_SIZE = 1    # Images per GPU batch (1 is safest for varied sizes)

# Prefetch pipeline: load/preprocess images on CPU threads while GPU works
PREFETCH_WORKERS = 4   # Background threads for image loading
PREFETCH_QUEUE_SIZE = 16  # Max preprocessed tensors held in memory
PREFETCH_PATH_QUEUE_SIZE = 128  # Max pending file paths staged for workers
PREFETCH_RESULT_TIMEOUT_SEC = 120  # Detect stalled prefetch instead of hanging forever
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}

# NAS -> temp buffer pipeline (up to ~100GB buffered at a time)
BATCH_SIZE_GB = 20
MAX_BUFFERED_BATCHES = 5  # 20 * 5 = 100GB

PROGRESS_HEARTBEAT_SEC = 120  # Persist periodic progress snapshots in log
PROGRESS_SAVE_EVERY_IMAGES = max(100, int(os.environ.get("DISK_PROGRESS_SAVE_EVERY_IMAGES", "1000")))
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
    """Clear one-line in-place progress output (call under _console_lock)."""
    global _progress_line_len
    if _progress_line_len > 0:
        print('\r' + (' ' * _progress_line_len) + '\r', end='', flush=True)
        _progress_line_len = 0


def update_progress_line(message: str):
    """Update a single in-place status line to avoid noisy scrolling output."""
    global _progress_line_len
    with _console_lock:
        padded = message
        if len(padded) < _progress_line_len:
            padded += ' ' * (_progress_line_len - len(padded))
        print('\r' + padded, end='', flush=True)
        _progress_line_len = max(_progress_line_len, len(message))


def format_progress_line(loop_index: int, total_images: int, chunk_num: int, chunk_vectors: int,
                         chunk_images: int, keypoints_found: int, status_label: str, image_name: str) -> str:
    """Compose a stable one-line status message with chunk fill stats."""
    chunk_gb = (chunk_vectors * 512) / (1024 ** 3)
    return (
        f"Image {loop_index+1:,}/{total_images:,} | chunk {chunk_num:03d} "
        f"| chunk_img {chunk_images:,} | vec {chunk_vectors:,} ({chunk_gb:0.2f}GB) "
        f"| kp {keypoints_found:,} | {status_label} | {image_name}"
    )


def format_eta(seconds: float) -> str:
    """Human-readable ETA (days/hours/minutes)."""
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
    """Return True for CUDA runtime failures that usually poison the context."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return ("cuda error" in msg) or ("cudnn" in msg) or ("illegal memory access" in msg)


def reload_disk_model(device):
    """Best-effort CUDA recovery: clear cache and recreate DISK model."""
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
    """Normalize stored/processed path variants to canonical NAS path."""
    p = os.path.normpath(path)
    local_temp = os.path.normpath(LOCAL_BUFFER_DIR)
    local_legacy = os.path.normpath(LEGACY_LOCAL_IMAGES_DIR)
    nas_base = os.path.normpath(NAS_IMAGES_DIR)
    if p.startswith(local_temp):
        rel = (nas_base + p[len(local_temp):])[len(nas_base):].replace('\\', '/')
        rel = re.sub(r'/\._batch_\d+/', '/', rel)
        return os.path.normpath(nas_base + rel)
    if p.startswith(local_legacy):
        return os.path.normpath(nas_base + p[len(local_legacy):])
    return p


def make_nas_path(local_path):
    """Map a temp-buffer local file path to stored NAS path."""
    return normalize_to_nas_path(local_path)


def find_images_recursive(directory):
    """Find all image files recursively in a directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(root, name))
    return sorted(files)


def get_folder_size_bytes(folder_path):
    """Get total folder size in bytes."""
    total = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except Exception:
        pass
    return total


def collect_source_units(source_dir):
    """Collect smaller source units so NAS staging is bounded by copied bytes."""
    units = []
    root_batch_map = {}

    top_level_dirs = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    for top in top_level_dirs:
        top_path = os.path.join(source_dir, top)

        child_dirs = sorted([
            d for d in os.listdir(top_path)
            if os.path.isdir(os.path.join(top_path, d))
        ])
        for child in child_dirs:
            units.append(os.path.join(top, child))

        root_image_files = []
        try:
            with os.scandir(top_path) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False) and \
                       os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS:
                        root_image_files.append(entry.name)
        except OSError:
            pass

        if root_image_files:
            root_image_files.sort()
            flat_batch_size = 1000
            for i in range(0, len(root_image_files), flat_batch_size):
                batch_name = os.path.join(top, f"._batch_{i // flat_batch_size:03d}")
                root_batch_map[batch_name] = root_image_files[i:i + flat_batch_size]
                units.append(batch_name)

    return sorted(units), root_batch_map


class FolderBuffer:
    """Copies NAS units into local temp buffer and keeps a bounded queue ready."""

    def __init__(self, folders, root_batch_map=None, manifest_state=None):
        self.root_batch_map = dict(root_batch_map or {})
        self.pending = deque()
        self.ready = deque()  # (folder_name, size_bytes)
        self.processing = {}
        self.lock = Lock()
        self.stop = Event()
        self.thread = None
        self.copied_count = 0
        self.copied_bytes = 0
        self.ready_bytes = 0
        self.processing_bytes = 0
        self.errors = []
        self._restore_or_seed(folders, manifest_state)

    def _restore_or_seed(self, folders, manifest_state):
        restored = False
        if manifest_state and manifest_has_work(manifest_state):
            restored = self._restore_from_manifest(manifest_state)
        if not restored:
            self.pending = deque(folders)
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
                    src = NAS_SOURCE_DIR
                    dst = os.path.join(LOCAL_BUFFER_DIR, folder)
                else:
                    src = os.path.join(NAS_SOURCE_DIR, folder)
                    dst = os.path.join(LOCAL_BUFFER_DIR, folder)
                try:
                    if os.path.exists(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    if is_root_batch:
                        os.makedirs(dst, exist_ok=True)
                        batch_files = self.root_batch_map[folder]
                        src_root = os.path.join(NAS_SOURCE_DIR, os.path.dirname(folder))

                        def _copy_one(fn):
                            shutil.copy2(os.path.join(src_root, fn), os.path.join(dst, fn))

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
        """Return up to one batch worth of ready folders."""
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
    """Local-mode folder iterator: no copy, just stage folders from C:\\boardgames."""

    def __init__(self, folders):
        self.pending = deque(folders)
        self.processing = set()
        self.done = 0
        self.total = len(folders)
        self.errors = []

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
    """Load build progress. Returns (next_chunk_num, set of processed image paths, path_to_id dict, next_id)."""
    if not os.path.exists(PROGRESS_FILE):
        return 1, set(), {}, 0

    try:
        with open(PROGRESS_FILE, 'r') as f:
            state = json.load(f)
        processed = {normalize_to_nas_path(p) for p in state.get('processed_images', [])}
        next_chunk = state.get('next_chunk', 1)
        next_id = state.get('next_id', 0)

        # Load path_to_id from existing path_lookup
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

        return next_chunk, processed, path_to_id, next_id
    except Exception as e:
        log(f"Warning: Could not load progress: {e}")
        return 1, set(), {}, 0


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


def prefetch_worker(path_queue, result_queue, error_queue):
    """Background worker: load and preprocess images so GPU never waits for I/O."""
    while True:
        image_path = path_queue.get()
        try:
            if image_path is None:  # poison pill
                return
            tensor, prep_status = preprocess_image(image_path)
            result_queue.put((image_path, tensor, prep_status))
        except Exception as e:
            # Surface worker failure to the main loop so it can fail fast.
            error_queue.put(f"{type(e).__name__}: {e}")
            result_queue.put((None, None, None))
            return
        finally:
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
    # allocation (~10GB) that can trigger std::bad_alloc on long runs.
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
    local_mode = os.path.exists(LEGACY_LOCAL_IMAGES_DIR)
    source_mode = "local" if local_mode else "nas_buffer"
    root_batch_map = {}

    log("=" * 70)
    log("BOARD GAMES DISK CHUNK BUILDER")
    log(f"NAS source:      {NAS_SOURCE_DIR}")
    if local_mode:
        log(f"Local source:    {LEGACY_LOCAL_IMAGES_DIR}")
    else:
        log(f"Local buffer:    {LOCAL_BUFFER_DIR}")
    log(f"Paths stored as: {NAS_IMAGES_DIR}")
    log(f"Chunks output:   {NAS_CHUNKS_DIR}")
    log(f"Compact IDs:     {CHUNK_IDS_DIR}")
    log(f"Target/chunk:    {MAX_VECTORS_PER_CHUNK:,} vectors (~10 GB)")
    log(f"Image resize:    max_dim={MAX_IMAGE_DIM if MAX_IMAGE_DIM is not None else 'none'}")
    if not local_mode:
        log(f"Buffer target:   {BATCH_SIZE_GB * MAX_BUFFERED_BATCHES} GB "
            f"({BATCH_SIZE_GB} GB/batch x {MAX_BUFFERED_BATCHES})")
    log(f"Source mode:     {source_mode}")
    log("=" * 70)

    # Check source exists
    manifest_state = None
    if local_mode:
        all_folders = sorted([
            d for d in os.listdir(LEGACY_LOCAL_IMAGES_DIR)
            if os.path.isdir(os.path.join(LEGACY_LOCAL_IMAGES_DIR, d))
        ])
        log(f"Found {len(all_folders):,} folders in local source")
    else:
        manifest_state = load_buffer_manifest()
        if manifest_has_work(manifest_state):
            all_folders = []
            seen = set()
            for folder in manifest_state.get('pending_units', []):
                if folder not in seen:
                    all_folders.append(folder)
                    seen.add(folder)
            for entry in manifest_state.get('ready_units', []):
                folder = entry.get('folder')
                if folder and folder not in seen:
                    all_folders.append(folder)
                    seen.add(folder)
            for entry in manifest_state.get('processing_units', []):
                folder = entry.get('folder')
                if folder and folder not in seen:
                    all_folders.append(folder)
                    seen.add(folder)
            root_batch_map = manifest_state.get('root_batch_map', {})
            log(f"Resuming from buffer manifest: {len(all_folders):,} queued/staged units")
        else:
            if not os.path.exists(NAS_SOURCE_DIR):
                log(f"ERROR: NAS source directory not found: {NAS_SOURCE_DIR}")
                sys.exit(1)
            all_folders, root_batch_map = collect_source_units(NAS_SOURCE_DIR)
            log(f"Found {len(all_folders):,} source units on NAS")

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

    # Find folders that still contain unprocessed images
    if not local_mode and manifest_has_work(manifest_state):
        remaining_folders = all_folders
        total_remaining_images = len(remaining_folders)
        log(f"Using manifest queue without NAS rescan: {len(remaining_folders):,} units")
    elif local_mode:
        log("Scanning local folders for new images...")
    else:
        log("Scanning NAS folders for new images...")
        remaining_folders = []
        total_remaining_images = 0
        for i, folder in enumerate(all_folders, 1):
            if not local_mode and '/._batch_' in folder.replace('\\', '/'):
                remaining_folders.append(folder)
                total_remaining_images += len(root_batch_map.get(folder, []))
                continue
            if local_mode:
                folder_path = os.path.join(LEGACY_LOCAL_IMAGES_DIR, folder)
            else:
                folder_path = os.path.join(NAS_SOURCE_DIR, folder)
            folder_new = 0
            for root, dirs, files in os.walk(folder_path):
                for name in files:
                    if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                        continue
                    nas_path = normalize_to_nas_path(os.path.join(root, name))
                    if nas_path in processed or nas_path in known_bad_images:
                        continue
                    folder_new += 1
            if folder_new > 0:
                remaining_folders.append(folder)
                total_remaining_images += folder_new
            if i % 200 == 0:
                update_progress_line(
                    f"Folder scan {i:,}/{len(all_folders):,} | folders_with_new {len(remaining_folders):,} "
                    f"| new_images {total_remaining_images:,}"
                )
        with _console_lock:
            _clear_progress_line_locked()
    log(f"Folders with new images: {len(remaining_folders):,}")
    log(f"Remaining images: {total_remaining_images:,}")

    if total_remaining_images == 0:
        log("All images already processed!")
        if not local_mode:
            clear_buffer_manifest()
        return

    # Clean local buffer (NAS mode only)
    if not local_mode:
        if manifest_has_work(manifest_state):
            os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)
            log("Reusing staged local temp buffer from manifest")
        elif os.path.exists(LOCAL_BUFFER_DIR):
            log("Cleaning local temp buffer...")
            shutil.rmtree(LOCAL_BUFFER_DIR, ignore_errors=True)
            os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)
        else:
            os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)

    # Load DISK model
    log("Loading DISK model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = KF.DISK.from_pretrained('depth').to(device).eval()
    log(f"DISK model loaded on {device}")

    # Estimate chunks (rough, actual count depends on keypoints per image)
    est_chunks = max(1, int(total_remaining_images * 15000 / MAX_VECTORS_PER_CHUNK))
    log(f"~{est_chunks} chunks estimated (10 GB target, actual depends on keypoints)")
    log("=" * 70)

    # Start folder buffer
    if local_mode:
        buffer = LocalFolderBuffer(remaining_folders)
    else:
        buffer = FolderBuffer(remaining_folders, root_batch_map=root_batch_map, manifest_state=manifest_state)
    buffer.start()

    # Start prefetch pipeline (local temp -> preprocessed tensors)
    path_queue = queue.Queue()
    result_queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    prefetch_errors = queue.Queue()
    workers = []
    for _ in range(PREFETCH_WORKERS):
        t = Thread(target=prefetch_worker, args=(path_queue, result_queue, prefetch_errors), daemon=True)
        t.start()
        workers.append(t)

    log(
        f"Prefetch pipeline started: {PREFETCH_WORKERS} workers, "
        f"result queue depth {PREFETCH_QUEUE_SIZE}"
    )

    # Wait for first folders to stage (NAS mode only)
    if not local_mode:
        log("Waiting for first folders to stage...")
        while True:
            s = buffer.status()
            if s['ready_gb'] >= max(1, BATCH_SIZE_GB // 2) or s['ready'] >= len(remaining_folders):
                break
            update_progress_line(
                f"Staging... copied {s['copied']:,} folders ({s['copied_gb']:.1f}GB), "
                f"ready {s['ready_gb']:.1f}GB"
            )
            time.sleep(1)
        with _console_lock:
            _clear_progress_line_locked()

    start_time = time.time()
    total_vectors = 0
    total_ok = 0
    total_failed = 0
    total_skipped = 0
    chunk_num = next_chunk
    folders_done = 0

    # Accumulator for current chunk
    chunk_descriptors = []
    chunk_ids = []
    chunk_vector_count = 0
    chunk_image_count = 0
    chunk_pending_processed = set()  # Successful images in current unflushed chunk
    chunk_start = time.time()
    last_heartbeat = time.time()
    cuda_error_streak = 0
    cuda_recovery_attempts = 0
    path_db_unsynced_chunks = 0

    def save_restart_checkpoint():
        committed_processed = processed.difference(chunk_pending_processed)
        save_progress(chunk_num, committed_processed, next_id)

    loop_index = 0
    while buffer.has_more():
        batch = buffer.get_next_batch()
        if not batch:
            if buffer.has_more():
                s = buffer.status()
                # Large folders can exceed get_next_batch timeout; wait instead of exiting.
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
            if local_mode:
                folder_dir = os.path.join(LEGACY_LOCAL_IMAGES_DIR, folder)
            else:
                folder_dir = os.path.join(LOCAL_BUFFER_DIR, folder)
            images = find_images_recursive(folder_dir)

            to_process = []
            for image_path in images:
                nas_path = make_nas_path(image_path)
                if nas_path in processed or nas_path in cuda_bad_images:
                    total_skipped += 1
                    continue
                to_process.append(image_path)
                path_queue.put(image_path)

            folder_image_total = len(to_process)
            for j in range(folder_image_total):
                try:
                    image_path, tensor, prep_status = result_queue.get(timeout=PREFETCH_RESULT_TIMEOUT_SEC)
                except queue.Empty:
                    worker_error = None
                    if not prefetch_errors.empty():
                        worker_error = prefetch_errors.get()
                    dead_workers = sum(1 for t in workers if not t.is_alive())
                    if worker_error:
                        raise RuntimeError(f"Prefetch worker failed: {worker_error}")
                    if dead_workers > 0:
                        raise RuntimeError(f"Prefetch stalled: {dead_workers}/{PREFETCH_WORKERS} workers exited early")
                    log("Prefetch timeout: waiting for workers to produce next preprocessed image...")
                    continue

                if image_path is None:
                    worker_error = prefetch_errors.get() if not prefetch_errors.empty() else "unknown worker error"
                    raise RuntimeError(f"Prefetch worker failed: {worker_error}")

                idx = loop_index
                loop_index += 1
                image_name = os.path.basename(image_path)
                if len(image_name) > 64:
                    image_name = image_name[:61] + "..."
                keypoints_found = 0
                status_label = "ok"
                mark_processed = True
                nas_path = make_nas_path(image_path)

                if tensor is None:
                    total_failed += 1
                    record_preprocess_bad_image(image_path, prep_status, preprocess_bad_images)
                    known_bad_images.add(nas_path)
                    if total_failed <= 20:
                        log(f"  PREPROCESS FAIL: {image_path} ({prep_status})")
                    elif total_failed == 21:
                        log("  (suppressing further preprocess-fail logs)")
                    status_label = "preprocess-fail"
                    processed.add(nas_path)
                    update_progress_line(format_progress_line(
                        idx, total_remaining_images, chunk_num, chunk_vector_count, chunk_image_count,
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
                                processed.add(nas_path)
                                update_progress_line(format_progress_line(
                                    idx, total_remaining_images, chunk_num, chunk_vector_count, chunk_image_count,
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
                        processed.add(nas_path)
                        del tensor
                        update_progress_line(format_progress_line(
                            idx, total_remaining_images, chunk_num, chunk_vector_count, chunk_image_count,
                            0, status_label, image_name
                        ))
                        continue

                    if total_ok < 10 or total_ok % 5000 == 0:
                        log(f"  SAMPLE OK: {os.path.basename(image_path)} "
                            f"shape={t_shape} keypoints={len(descriptors)}")

                    if nas_path not in path_to_id:
                        path_to_id[nas_path] = next_id
                        next_id += 1
                    pid = path_to_id[nas_path]

                    chunk_descriptors.append(descriptors)
                    chunk_ids.extend([pid] * len(descriptors))
                    chunk_vector_count += len(descriptors)
                    chunk_image_count += 1
                    chunk_pending_processed.add(nas_path)
                    total_ok += 1
                    cuda_error_streak = 0

                except Exception as e:
                    total_failed += 1
                    status_label = f"extract-{type(e).__name__}"
                    if total_failed <= 20:
                        log(f"  EXTRACT FAIL: {image_path} - {type(e).__name__}: {e}")

                    if is_cuda_runtime_error(e):
                        mark_processed = False
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
                                    f"  CUDA recovery attempt {cuda_recovery_attempts}/{CUDA_MAX_RECOVERY_ATTEMPTS} "
                                    f"after streak={cuda_error_streak}"
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

                if mark_processed:
                    processed.add(nas_path)
                del tensor
                update_progress_line(format_progress_line(
                    idx, total_remaining_images, chunk_num, chunk_vector_count, chunk_image_count,
                    keypoints_found, status_label, image_name
                ))

                if (time.time() - last_heartbeat) >= PROGRESS_HEARTBEAT_SEC:
                    elapsed = time.time() - start_time
                    images_done = idx + 1
                    images_left = max(0, total_remaining_images - images_done)
                    rate = images_done / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (images_left / rate) if rate > 0 else float("inf")
                    eta_str = format_eta(eta_seconds)
                    finish_str = (
                        (datetime.now() + timedelta(seconds=eta_seconds)).strftime("%m-%d %H:%M")
                        if np.isfinite(eta_seconds) and eta_seconds > 0 else "unknown"
                    )
                    s = buffer.status()
                    log(
                        f"Heartbeat: image {images_done:,}/{total_remaining_images:,}, "
                        f"chunk {chunk_num:03d}, chunk_vectors={chunk_vector_count:,}, "
                        f"chunk_images={chunk_image_count:,}, last_kp={keypoints_found:,}, "
                        f"ok={total_ok:,}, failed={total_failed:,}, skipped={total_skipped:,}, "
                        f"buffer_ready={s['ready_gb']:.1f}GB, "
                        f"cuda_streak={cuda_error_streak}, cuda_recoveries={cuda_recovery_attempts}, "
                        f"eta={eta_str}, finish={finish_str}"
                    )
                    last_heartbeat = time.time()

                if (idx + 1) % PROGRESS_SAVE_EVERY_IMAGES == 0:
                    save_restart_checkpoint()

                if (idx + 1) % 200 == 0:
                    gc.collect()
                    if device.type == 'cuda':
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            pass

                if chunk_vector_count >= MAX_VECTORS_PER_CHUNK:
                    chunk_time = time.time() - chunk_start
                    log(f"\n[Chunk {chunk_num:03d}] Flushing {chunk_vector_count:,} vectors "
                        f"from {chunk_image_count} images")

                    num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
                    total_vectors += num_vectors

                    chunk_num += 1
                    save_progress(chunk_num, processed, next_id)
                    path_db_unsynced_chunks += 1
                    sync_db_now = path_db_unsynced_chunks >= PATH_DB_SYNC_EVERY_CHUNKS
                    save_path_lookup(path_to_id, sync_db=sync_db_now)
                    if sync_db_now:
                        path_db_unsynced_chunks = 0
                    chunk_pending_processed.clear()

                    elapsed = time.time() - start_time
                    images_done = idx + 1
                    images_left = max(0, total_remaining_images - images_done)
                    rate = images_done / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (images_left / rate) if rate > 0 else float("inf")
                    s = buffer.status()
                    log(f"  Time: {chunk_time:.0f}s | Total vectors: {total_vectors:,} | "
                        f"Unique paths: {len(path_to_id):,} | Buffer: {s['ready_gb']:.1f}GB | "
                        f"ETA: {format_eta(eta_seconds)}")

                    chunk_descriptors = []
                    chunk_ids = []
                    chunk_vector_count = 0
                    chunk_image_count = 0
                    chunk_start = time.time()

            buffer.mark_done(folder, folder_size)
            folders_done += 1

        save_restart_checkpoint()

    # Flush remaining vectors as final chunk
    if chunk_descriptors:
        log(f"\n[Chunk {chunk_num:03d}] Final chunk: {chunk_vector_count:,} vectors "
            f"from {chunk_image_count} images")
        num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
        total_vectors += num_vectors
        chunk_num += 1
        save_progress(chunk_num, processed, next_id)
        save_path_lookup(path_to_id, sync_db=True)
        path_db_unsynced_chunks = 0
        chunk_pending_processed.clear()

    # Stop folder buffer
    buffer.stop_copying()

    # Stop prefetch workers
    for _ in range(PREFETCH_WORKERS):
        path_queue.put(None)
    path_queue.join()

    # Wait for prefetch workers to finish
    for t in workers:
        t.join()

    if (not local_mode) and os.path.exists(LOCAL_BUFFER_DIR):
        shutil.rmtree(LOCAL_BUFFER_DIR, ignore_errors=True)

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
    log(f"  Images skipped: {total_skipped:,}")
    log(f"  Folders done:   {folders_done:,}/{len(remaining_folders):,}")
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
