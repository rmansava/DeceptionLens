r"""
Build DISK keypoint chunks for albums - direct to chunks with compact IDs.

Pipelined: copies batches of folders from NAS to local buffer, processes on GPU,
deletes processed files. Keeps 3 batches buffered so GPU never waits.

  1. List all subfolders on NAS
  2. Copy batch of folders to local SSD buffer (sized by GB, not folder count)
  3. Extract DISK features per image on GPU
  4. Delete processed local files, copy next batch
  5. Flush to chunk when hitting ~10 GB (19.5M vectors)

Progress tracked per-image (via path_to_id). Adding new images to existing
folders will be picked up on re-run.

Input:  NAS albums (T:/archiverelated/albums)
Output: chunk_XXX.faiss  -> NAS (T:/faiss/disk_retrieval/albums_chunks/)
        chunk_XXX_ids.npy -> local SSD (D:/faiss/disk_retrieval/albums_chunk_ids/)
        path_lookup.json  -> local SSD (same dir as IDs)

Paths stored point to the NAS originals (T:/archiverelated/albums/...).
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
from collections import Counter
from datetime import datetime
from threading import Thread, Lock, Event
from collections import deque

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
NAS_SOURCE_DIR = r"T:\archiverelated\albums"

# Local buffer: copy batches here for fast GPU reads
LOCAL_BUFFER_DIR = r"C:\albums_buffer"

# Path prefix stored in path_lookup (matches NAS source)
NAS_IMAGES_DIR = r"T:\archiverelated\albums"

# Output: FAISS chunks go to NAS
NAS_CHUNKS_DIR = r"T:\faiss\disk_retrieval\albums_chunks"
LOCAL_CHUNKS_BUFFER = r"D:\faiss\disk_retrieval\albums_chunks"

# Output: Compact IDs stay on local SSD
CHUNK_IDS_DIR = r"D:\faiss\disk_retrieval\albums_chunk_ids"

# Progress tracking
PROGRESS_DIR = CHUNK_IDS_DIR
PROGRESS_FILE = os.path.join(PROGRESS_DIR, "build_progress.json")
LOG_FILE = os.path.join(PROGRESS_DIR, "build_log.txt")

# Chunk sizing
MAX_VECTORS_PER_CHUNK = 19_500_000  # ~10 GB

# Buffer settings - sized by total file size, not folder count
BATCH_SIZE_GB = 8           # Copy ~8 GB of folders per batch
MAX_BUFFERED_BATCHES = 3    # Keep up to 3 batches on local SSD (~24 GB max)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}

# DISK extraction settings
MAX_IMAGE_DIM = 1600
GPU_BATCH_SIZE = 1

# ============================================================================


def log(msg):
    """Print and log to file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def make_nas_path(local_path):
    """Convert local buffer path to NAS storage path."""
    local_norm = os.path.normpath(local_path)
    buffer_norm = os.path.normpath(LOCAL_BUFFER_DIR)
    if local_norm.startswith(buffer_norm):
        relative = local_norm[len(buffer_norm):]
        return NAS_IMAGES_DIR + relative.replace('\\', '/')
    return local_path


def find_images_recursive(directory):
    """Find all images recursively in a directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(root, f))
    return sorted(files)


def count_images_on_nas(folder_name):
    """Count image files in a NAS folder (recursive)."""
    count = 0
    folder_path = os.path.join(NAS_SOURCE_DIR, folder_name)
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

    def __init__(self, all_folders):
        self.pending = deque(all_folders)
        self.ready = deque()        # (folder_name, size_bytes) copied and ready
        self.processing = set()     # Currently being processed
        self.lock = Lock()
        self.stop = Event()
        self.thread = None
        self.copied_count = 0
        self.copied_bytes = 0
        self.ready_bytes = 0
        self.processing_bytes = 0
        self.errors = []

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

                if self.stop.is_set():
                    break

                src = os.path.join(NAS_SOURCE_DIR, folder)
                dst = os.path.join(LOCAL_BUFFER_DIR, folder)

                try:
                    if os.path.exists(dst):
                        shutil.rmtree(dst, ignore_errors=True)
                    shutil.copytree(src, dst)
                    folder_bytes = get_folder_size_bytes(dst)
                    batch_bytes += folder_bytes
                    with self.lock:
                        self.ready.append((folder, folder_bytes))
                        self.ready_bytes += folder_bytes
                        self.copied_count += 1
                        self.copied_bytes += folder_bytes
                except Exception as e:
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
                    self.processing.add(folder)
                    self.processing_bytes += size
                    batch.append((folder, size))
                    batch_bytes += size
            if batch:
                return batch
            if not self.has_more():
                return batch
            time.sleep(0.5)
        return batch

    def mark_done(self, folder, size):
        """Delete a processed folder from local buffer."""
        with self.lock:
            self.processing.discard(folder)
            self.processing_bytes -= size
        local_dir = os.path.join(LOCAL_BUFFER_DIR, folder)
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir, ignore_errors=True)

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
        # folder -> image count when last fully processed
        scanned_counts = state.get('scanned_folder_counts', {})

        path_to_id = {}
        lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
        if os.path.exists(lookup_file):
            with open(lookup_file, 'r') as f:
                id_to_path = json.load(f)
            path_to_id = {p: i for i, p in enumerate(id_to_path)}
            next_id = len(id_to_path)

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


def save_path_lookup(path_to_id):
    """Save the global path lookup."""
    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)
    id_to_path = [''] * len(path_to_id)
    for path, pid in path_to_id.items():
        id_to_path[pid] = path

    lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    with open(lookup_file, 'w') as f:
        json.dump(id_to_path, f)
    log(f"  Saved path_lookup.json: {len(path_to_id):,} unique paths "
        f"({os.path.getsize(lookup_file) / 1e6:.1f} MB)")


def preprocess_image(image_path, max_dim=MAX_IMAGE_DIM):
    """Load and preprocess image for DISK extraction."""
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
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
        return tensor
    except Exception:
        return None


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
    all_desc = np.vstack(all_descriptors)
    num_vectors = len(all_desc)

    index = faiss.IndexFlatIP(128)
    index.add(all_desc)

    os.makedirs(LOCAL_CHUNKS_BUFFER, exist_ok=True)
    local_faiss = os.path.join(LOCAL_CHUNKS_BUFFER, f"chunk_{chunk_num:03d}.faiss")
    faiss.write_index(index, local_faiss)
    faiss_size = os.path.getsize(local_faiss) / (1024**3)

    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)
    ids_array = np.array(all_ids, dtype=np.int32)
    ids_file = os.path.join(CHUNK_IDS_DIR, f"chunk_{chunk_num:03d}_ids.npy")
    np.save(ids_file, ids_array)
    ids_size = os.path.getsize(ids_file) / (1024**2)

    log(f"  Chunk {chunk_num:03d}: {num_vectors:,} vectors from {num_images} images "
        f"({faiss_size:.1f} GB index, {ids_size:.0f} MB IDs)")

    os.makedirs(NAS_CHUNKS_DIR, exist_ok=True)
    nas_faiss = os.path.join(NAS_CHUNKS_DIR, f"chunk_{chunk_num:03d}.faiss")
    _nas_copy_thread = Thread(target=_nas_copy_worker, args=(local_faiss, nas_faiss, chunk_num))
    _nas_copy_thread.daemon = True
    _nas_copy_thread.start()

    del all_desc, all_descriptors, all_ids, ids_array, index
    gc.collect()
    return num_vectors


def find_folders_with_new_images(all_folders, scanned_counts, path_to_id):
    """Find folders that have unprocessed images.

    Checks each previously-scanned folder's image count against what was recorded.
    New folders are always included. Folders with unchanged counts are skipped.
    """
    remaining = []
    skipped = 0

    # Count processed images per folder from path_to_id
    processed_per_folder = Counter()
    prefix = NAS_IMAGES_DIR.replace('\\', '/') + '/'
    for nas_path in path_to_id:
        if nas_path.startswith(prefix):
            rel = nas_path[len(prefix):]
            folder = rel.split('/')[0]
            processed_per_folder[folder] += 1

    for i, folder in enumerate(all_folders):
        if (i + 1) % 5000 == 0:
            print(f"\r  Checking folders: {i+1:,}/{len(all_folders):,} "
                  f"({skipped:,} skipped, {len(remaining):,} to process)...",
                  end="", flush=True)

        if folder in scanned_counts:
            # Quick check: does the NAS folder still have the same image count?
            current_count = count_images_on_nas(folder)
            processed_count = processed_per_folder.get(folder, 0)
            if current_count == scanned_counts[folder] and current_count == processed_count:
                skipped += 1
                continue

        remaining.append(folder)

    if len(all_folders) > 5000:
        print()
    return remaining, skipped


def main():
    log("=" * 70)
    log("ALBUMS DISK CHUNK BUILDER (Pipelined NAS -> Local -> GPU)")
    log(f"NAS source:      {NAS_SOURCE_DIR}")
    log(f"Local buffer:    {LOCAL_BUFFER_DIR}")
    log(f"Paths stored as: {NAS_IMAGES_DIR}")
    log(f"Chunks output:   {NAS_CHUNKS_DIR}")
    log(f"Compact IDs:     {CHUNK_IDS_DIR}")
    log(f"Target/chunk:    {MAX_VECTORS_PER_CHUNK:,} vectors (~10 GB)")
    log(f"Buffer:          {BATCH_SIZE_GB} GB/batch, {MAX_BUFFERED_BATCHES} batches max")
    log("=" * 70)

    # List all folders on NAS
    log("Scanning NAS for folders...")
    all_folders = sorted([d for d in os.listdir(NAS_SOURCE_DIR)
                         if os.path.isdir(os.path.join(NAS_SOURCE_DIR, d))])
    log(f"Found {len(all_folders):,} folders on NAS")

    # Load progress
    next_chunk, scanned_counts, path_to_id, next_id = load_progress()
    log(f"Resume: chunk {next_chunk}, {len(path_to_id):,} images already processed, "
        f"{len(scanned_counts):,} folders fully scanned")

    # Find folders with new/unprocessed images
    log("Checking for folders with new images...")
    remaining, skipped = find_folders_with_new_images(all_folders, scanned_counts, path_to_id)
    log(f"Skipped {skipped:,} unchanged folders, {len(remaining):,} folders to process")

    if not remaining:
        log("All images already processed!")
        return

    # Clean local buffer
    if os.path.exists(LOCAL_BUFFER_DIR):
        log("Cleaning local buffer...")
        shutil.rmtree(LOCAL_BUFFER_DIR, ignore_errors=True)
    os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)

    # Load DISK model
    log("Loading DISK model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = KF.DISK.from_pretrained('depth').to(device).eval()
    log(f"DISK model loaded on {device}")
    log("=" * 70)

    # Start buffer
    buffer = FolderBuffer(remaining)
    buffer.start()

    # Wait for first batch
    log("Waiting for first batch to copy...")
    batch_target = BATCH_SIZE_GB * 1024**3
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

    while buffer.has_more():
        batch = buffer.get_next_batch()
        if not batch:
            break

        for folder, folder_size in batch:
            folder_dir = os.path.join(LOCAL_BUFFER_DIR, folder)
            images = find_images_recursive(folder_dir)
            folder_new = 0

            for image_path in images:
                # Image-level skip check
                nas_path = make_nas_path(image_path)
                if nas_path in path_to_id:
                    total_skipped += 1
                    continue

                tensor = preprocess_image(image_path)
                if tensor is None:
                    total_failed += 1
                    continue

                try:
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        feats = extractor(tensor)[0]
                        descriptors = feats.descriptors.cpu().numpy()

                    if len(descriptors) == 0:
                        total_failed += 1
                        del tensor
                        continue

                    path_to_id[nas_path] = next_id
                    next_id += 1
                    pid = path_to_id[nas_path]

                    chunk_descriptors.append(descriptors)
                    chunk_ids.extend([pid] * len(descriptors))
                    chunk_vector_count += len(descriptors)
                    chunk_image_count += 1
                    total_ok += 1
                    total_images += 1
                    folder_new += 1

                except Exception:
                    total_failed += 1

                del tensor
                if total_images % 200 == 0:
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                # Flush chunk
                if chunk_vector_count >= MAX_VECTORS_PER_CHUNK:
                    chunk_time = time.time() - chunk_start
                    log(f"\n[Chunk {chunk_num:03d}] Flushing {chunk_vector_count:,} vectors "
                        f"from {chunk_image_count} images")

                    num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
                    total_vectors += num_vectors

                    chunk_num += 1
                    save_progress(chunk_num, scanned_counts, next_id)
                    save_path_lookup(path_to_id)

                    elapsed = time.time() - start_time
                    folders_left = len(remaining) - folders_done
                    rate = folders_done / elapsed if elapsed > 0 else 1
                    eta_hours = (folders_left / rate) / 3600 if rate > 0 else 0
                    s = buffer.status()

                    log(f"  Time: {chunk_time:.0f}s | Vectors: {total_vectors:,} | "
                        f"Folders: {folders_done:,}/{len(remaining):,} | "
                        f"Buffer: {s['ready_gb']:.1f} GB ready | ETA: {eta_hours:.1f}h")

                    chunk_descriptors = []
                    chunk_ids = []
                    chunk_vector_count = 0
                    chunk_image_count = 0
                    chunk_start = time.time()

            # Folder done - record image count for skip optimization on re-run
            scanned_counts[folder] = len(images)
            buffer.mark_done(folder, folder_size)
            folders_done += 1

            # Progress line
            elapsed = time.time() - start_time
            rate = folders_done / elapsed if elapsed > 0 else 1
            eta = (len(remaining) - folders_done) / rate / 3600 if rate > 0 else 0
            pct = folders_done / len(remaining) * 100
            s = buffer.status()
            print(f"\r  Folder {folders_done:,}/{len(remaining):,} {pct:5.1f}% | "
                  f"New: {total_ok:,} Skip: {total_skipped:,} | "
                  f"Vectors: {chunk_vector_count:,} | "
                  f"Buffer: {s['ready_gb']:.1f} GB | ETA: {eta:.1f}h   ",
                  end="", flush=True)

        # Save progress after each batch
        save_progress(chunk_num, scanned_counts, next_id)

    # Stop buffer
    buffer.stop_copying()

    # Flush final chunk
    if chunk_descriptors:
        log(f"\n[Chunk {chunk_num:03d}] Final chunk: {chunk_vector_count:,} vectors "
            f"from {chunk_image_count} images")
        num_vectors = save_chunk(chunk_num, chunk_descriptors, chunk_ids, chunk_image_count)
        total_vectors += num_vectors
        chunk_num += 1
        save_progress(chunk_num, scanned_counts, next_id)
        save_path_lookup(path_to_id)

    wait_for_nas_copy()

    # Cleanup buffer
    if os.path.exists(LOCAL_BUFFER_DIR):
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
    main()
