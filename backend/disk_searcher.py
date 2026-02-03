"""
DISK Keypoint Search - finds source pages for cropped images.

Uses consolidated FAISS chunks for fast searching across ~7000 books.
Each query keypoint votes for the source image it matches.

Streaming search mode: Copies chunks from NAS to local SSD for fast searching,
then deletes local copy before moving to next chunk. This handles 13TB+ indexes
that don't fit on local storage.
"""

import faiss
import numpy as np
import json
import os
import shutil
from glob import glob
from collections import Counter
import logging
import torch
import time
import kornia.feature as KF
import kornia as K
from threading import Thread
from queue import Queue, Empty

logger = logging.getLogger(__name__)

# Paths - NAS storage with local SSD buffer for fast searching
NAS_CHUNKS_DIR = "T:/faiss/disk_retrieval/chunks"    # Source: chunks on NAS
LOCAL_CHUNK_BUFFER = "D:/faiss/disk_retrieval/chunk_buffer"  # Buffer: copy here for fast reads
BOOKS_DIR = "T:/faiss/disk_retrieval/books"  # Fallback if no chunks yet

# Legacy path for backwards compatibility
CHUNKS_DIR = NAS_CHUNKS_DIR

# DISK model (lazy loaded)
_disk_model = None
_device = None


def get_disk_model():
    """Lazy-load DISK feature extractor."""
    global _disk_model, _device
    if _disk_model is None:
        logger.info("Loading DISK model...")
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _disk_model = KF.DISK.from_pretrained('depth').to(_device).eval()
        logger.info(f"DISK model loaded on {_device}")
    return _disk_model, _device


def extract_disk_features(image_bytes: bytes) -> np.ndarray:
    """Extract DISK keypoint descriptors from image bytes."""
    import cv2
    from PIL import Image
    import io

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    h, w = image_np.shape[:2]

    # Pad to multiples of 16 (required by DISK)
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16

    if new_h != h or new_w != w:
        padded = np.zeros((new_h, new_w, 3), dtype=image_np.dtype)
        padded[:h, :w] = image_np
        image_np = padded

    # Convert to tensor (RGB, no grayscale conversion)
    image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    model, device = get_disk_model()
    image_tensor = image_tensor.to(device)

    # Extract features
    with torch.no_grad():
        feats = model(image_tensor)[0]  # Returns list, take first element
        keypoints = feats.keypoints.cpu().numpy()  # (N, 2)
        descriptors = feats.descriptors.cpu().numpy()  # (N, 128)

    if len(descriptors) == 0:
        return np.array([]).reshape(0, 128)

    # Normalize descriptors
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-8)

    return descriptors.astype('float32')


def search_chunks(query_descriptors: np.ndarray, k: int = 5, threshold: float = 0.7, top_n: int = 50, specific_chunks: list = None, search_id: int = None, progress_callback=None):
    """
    Search consolidated chunks for matching images using streaming mode.

    Streaming mode copies each chunk from NAS to local SSD before searching,
    then deletes the local copy. This handles indexes larger than local storage.

    Args:
        query_descriptors: Normalized DISK descriptors from query image
        k: Number of nearest neighbors per descriptor
        threshold: Minimum similarity score to count as vote
        top_n: Number of top results to return
        specific_chunks: Optional list of chunk numbers to search (e.g., [142, 200])
        search_id: Optional search session ID for live tracking
        progress_callback: Optional callback for progress updates

    Returns:
        List of (path, votes) tuples sorted by votes descending
    """
    if len(query_descriptors) == 0:
        logger.warning("No keypoints extracted from query image")
        return []

    # Check for chunks on NAS first
    chunk_files = sorted(glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*.faiss")))

    # Filter to specific chunks if requested
    if specific_chunks:
        filtered = []
        for chunk_file in chunk_files:
            chunk_name = os.path.basename(chunk_file)
            chunk_num = chunk_name.replace('chunk_', '').replace('.faiss', '')
            if chunk_num in specific_chunks or int(chunk_num) in specific_chunks:
                filtered.append(chunk_file)
        chunk_files = filtered
        logger.info(f"Searching only chunks: {specific_chunks}")

    if chunk_files:
        # Use rolling buffer for faster searching (parallel copy + search)
        return _search_chunks_rolling_buffer(query_descriptors, chunk_files, k, threshold, top_n, buffer_size=5, search_id=search_id, progress_callback=progress_callback)
    else:
        # Fallback to per-book search
        logger.warning("No chunks found, falling back to per-book search (slower)")
        return _search_books_mode(query_descriptors, k, threshold, top_n)


def _copy_chunk_worker(copy_queue: Queue, chunk_files: list, start_idx: int, buffer_size: int):
    """Background worker to copy chunks ahead of time."""
    for i in range(start_idx, min(start_idx + buffer_size, len(chunk_files))):
        nas_chunk_file = chunk_files[i]
        nas_paths_file = nas_chunk_file.replace('.faiss', '_paths.json')
        chunk_name = os.path.basename(nas_chunk_file)
        paths_name = os.path.basename(nas_paths_file)

        local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)
        local_paths_file = os.path.join(LOCAL_CHUNK_BUFFER, paths_name)

        # Check if already cached
        chunk_exists = os.path.exists(local_chunk_file) and os.path.exists(local_paths_file)
        same_size = chunk_exists and os.path.getsize(local_chunk_file) == os.path.getsize(nas_chunk_file)

        if not same_size:
            logger.info(f"  Background: Copying chunk {i+1} ({chunk_name})...")
            copy_start = time.time()
            shutil.copy2(nas_chunk_file, local_chunk_file)
            shutil.copy2(nas_paths_file, local_paths_file)
            copy_time = time.time() - copy_start
            chunk_size_gb = os.path.getsize(local_chunk_file) / (1024**3)
            logger.info(f"  Background: Copied {chunk_size_gb:.1f}GB in {copy_time:.1f}s")

        # Signal that this chunk is ready
        copy_queue.put((i, local_chunk_file, local_paths_file))


def _search_chunks_rolling_buffer(query_descriptors: np.ndarray, chunk_files: list, k: int, threshold: float, top_n: int, buffer_size: int = 5, search_id: int = None, progress_callback=None):
    """
    Search using rolling buffer: maintain 5 chunks in local buffer, copy next chunk while searching current.

    This parallelizes copy and search operations for much faster throughput.

    Args:
        search_id: Optional search session ID for live progress tracking
        progress_callback: Optional callback(chunk_idx, total_chunks, top_results, elapsed_ms) for progress updates
    """
    all_votes = Counter()
    total_chunks = len(chunk_files)

    os.makedirs(LOCAL_CHUNK_BUFFER, exist_ok=True)
    search_start = time.time()

    # Start copying first buffer_size chunks
    copy_queue = Queue()
    next_copy_idx = buffer_size

    logger.info(f"Starting rolling buffer search (buffer size: {buffer_size} chunks)")
    copy_thread = Thread(target=_copy_chunk_worker, args=(copy_queue, chunk_files, 0, buffer_size))
    copy_thread.daemon = True
    copy_thread.start()

    for chunk_idx in range(total_chunks):
        nas_chunk_file = chunk_files[chunk_idx]
        chunk_name = os.path.basename(nas_chunk_file)
        local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)
        local_paths_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name.replace('.faiss', '_paths.json'))

        # Wait for this chunk to be ready (if it's being copied)
        logger.info(f"[{chunk_idx + 1}/{total_chunks}] Waiting for {chunk_name}...")
        ready = False
        while not ready:
            try:
                ready_idx, _, _ = copy_queue.get(timeout=1)
                if ready_idx == chunk_idx:
                    ready = True
                else:
                    # Put it back for later
                    copy_queue.put((ready_idx, _, _))
            except Empty:
                # Check if file exists (might have been cached)
                if os.path.exists(local_chunk_file) and os.path.exists(local_paths_file):
                    ready = True

        # Load and search
        logger.info(f"  Searching {chunk_name}...")
        load_start = time.time()
        index = faiss.read_index(local_chunk_file, faiss.IO_FLAG_MMAP)
        with open(local_paths_file, 'r') as f:
            paths = json.load(f)
        load_time = time.time() - load_start
        logger.info(f"  Loaded in {load_time:.1f}s (mmap)")

        # Search
        search_start_chunk = time.time()
        distances, indices = index.search(query_descriptors, k)
        search_time = time.time() - search_start_chunk
        logger.info(f"  Searched in {search_time:.1f}s")

        # Accumulate votes
        for i in range(len(query_descriptors)):
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0 and distances[i][j] >= threshold:
                    all_votes[paths[idx]] += 1

        # Free memory
        del index, paths

        # Delete this chunk to make room
        try:
            os.remove(local_chunk_file)
            os.remove(local_paths_file)
            logger.info(f"  Deleted {chunk_name} from buffer")
        except Exception as e:
            logger.warning(f"  Failed to delete {chunk_name}: {e}")

        # Start copying next chunk if available
        if next_copy_idx < total_chunks:
            copy_thread = Thread(target=_copy_chunk_worker, args=(copy_queue, chunk_files, next_copy_idx, 1))
            copy_thread.daemon = True
            copy_thread.start()
            next_copy_idx += 1

        # Progress update
        elapsed = time.time() - search_start
        avg_per_chunk = elapsed / (chunk_idx + 1)
        remaining = (total_chunks - chunk_idx - 1) * avg_per_chunk
        logger.info(f"  Progress: {chunk_idx + 1}/{total_chunks} chunks | "
                   f"ETA: {remaining/60:.1f}m | Top vote: {all_votes.most_common(1)[0][1] if all_votes else 0}")

        # Call progress callback for live updates
        if progress_callback:
            top_results = [
                {'path': path, 'votes': votes, 'score': votes / all_votes.most_common(1)[0][1] if all_votes else 0}
                for path, votes in all_votes.most_common(100)
            ]
            progress_callback(chunk_idx + 1, total_chunks, top_results, int(elapsed * 1000))

    total_time = time.time() - search_start
    logger.info(f"Search complete: {total_chunks} chunks in {total_time/60:.1f}m")

    return all_votes.most_common(top_n)


def _search_chunks_streaming(query_descriptors: np.ndarray, chunk_files: list, k: int, threshold: float, top_n: int):
    """
    Search using streaming mode: copy chunk from NAS → local SSD → search → delete.

    This allows searching indexes larger than local storage by only keeping
    one chunk on the SSD at a time.
    """
    all_votes = Counter()
    total_chunks = len(chunk_files)

    # Ensure buffer directory exists
    os.makedirs(LOCAL_CHUNK_BUFFER, exist_ok=True)

    search_start = time.time()

    for chunk_idx, nas_chunk_file in enumerate(chunk_files):
        nas_paths_file = nas_chunk_file.replace('.faiss', '_paths.json')
        chunk_name = os.path.basename(nas_chunk_file)
        paths_name = os.path.basename(nas_paths_file)

        if not os.path.exists(nas_paths_file):
            logger.warning(f"Missing paths file for {chunk_name}")
            continue

        # Copy chunk from NAS to local SSD (skip if already cached)
        local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)
        local_paths_file = os.path.join(LOCAL_CHUNK_BUFFER, paths_name)

        # Check if files already exist and have same size (skip copy)
        chunk_exists = os.path.exists(local_chunk_file) and os.path.exists(local_paths_file)
        same_size = chunk_exists and os.path.getsize(local_chunk_file) == os.path.getsize(nas_chunk_file)

        if same_size:
            logger.info(f"[{chunk_idx + 1}/{total_chunks}] Using cached {chunk_name} from local SSD")
            copy_time = 0
        else:
            logger.info(f"[{chunk_idx + 1}/{total_chunks}] Copying {chunk_name} to local SSD...")
            copy_start = time.time()
            shutil.copy2(nas_chunk_file, local_chunk_file)
            shutil.copy2(nas_paths_file, local_paths_file)
            copy_time = time.time() - copy_start
            chunk_size_gb = os.path.getsize(local_chunk_file) / (1024**3)
            logger.info(f"  Copied {chunk_size_gb:.1f}GB in {copy_time:.1f}s ({chunk_size_gb/copy_time:.1f} GB/s)")

        # Load and search from local SSD (fast)
        logger.info(f"  Searching {chunk_name}...")
        load_start = time.time()

        # Use memory-mapped loading for fast access to large indexes
        index = faiss.read_index(local_chunk_file, faiss.IO_FLAG_MMAP)
        with open(local_paths_file, 'r') as f:
            paths = json.load(f)

        load_time = time.time() - load_start
        logger.info(f"  Loaded in {load_time:.1f}s (mmap)")

        # Search
        search_start_chunk = time.time()
        distances, indices = index.search(query_descriptors, k)
        search_time = time.time() - search_start_chunk
        logger.info(f"  Searched in {search_time:.1f}s")

        # Accumulate votes
        for i in range(len(query_descriptors)):
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0 and distances[i][j] >= threshold:
                    all_votes[paths[idx]] += 1

        # Free memory and delete local copy
        del index, paths

        try:
            os.remove(local_chunk_file)
            os.remove(local_paths_file)
        except Exception as e:
            logger.warning(f"  Failed to cleanup local files: {e}")

        # Progress update
        elapsed = time.time() - search_start
        avg_per_chunk = elapsed / (chunk_idx + 1)
        remaining = (total_chunks - chunk_idx - 1) * avg_per_chunk
        logger.info(f"  Progress: {chunk_idx + 1}/{total_chunks} chunks | "
                   f"ETA: {remaining/60:.1f}m | Top vote: {all_votes.most_common(1)[0][1] if all_votes else 0}")

    total_time = time.time() - search_start
    logger.info(f"Search complete: {total_chunks} chunks in {total_time/60:.1f}m")

    return all_votes.most_common(top_n)


def _search_chunks_mode(query_descriptors: np.ndarray, chunk_files: list, k: int, threshold: float, top_n: int):
    """Legacy: Search using consolidated chunks directly (requires chunks on local storage)."""
    all_votes = Counter()

    for chunk_file in chunk_files:
        paths_file = chunk_file.replace('.faiss', '_paths.json')

        if not os.path.exists(paths_file):
            logger.warning(f"Missing paths file for {chunk_file}")
            continue

        logger.info(f"Searching {os.path.basename(chunk_file)}...")

        # Load chunk with memory mapping
        index = faiss.read_index(chunk_file, faiss.IO_FLAG_MMAP)
        with open(paths_file, 'r') as f:
            paths = json.load(f)

        # Search
        distances, indices = index.search(query_descriptors, k)

        # Accumulate votes
        for i in range(len(query_descriptors)):
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0 and distances[i][j] >= threshold:
                    all_votes[paths[idx]] += 1

        # Free memory
        del index

    return all_votes.most_common(top_n)


def _search_books_mode(query_descriptors: np.ndarray, k: int, threshold: float, top_n: int):
    """Fallback: search per-book indexes (slower)."""
    all_votes = Counter()

    book_dirs = [d for d in os.listdir(BOOKS_DIR)
                 if os.path.isfile(os.path.join(BOOKS_DIR, d, "index.faiss"))]

    for i, book in enumerate(book_dirs):
        index_path = os.path.join(BOOKS_DIR, book, "index.faiss")
        paths_path = os.path.join(BOOKS_DIR, book, "paths.json")

        if not os.path.exists(paths_path):
            continue

        try:
            index = faiss.read_index(index_path)
            with open(paths_path, 'r') as f:
                paths = json.load(f)

            distances, indices = index.search(query_descriptors, k)

            for qi in range(len(query_descriptors)):
                for j in range(k):
                    idx = indices[qi][j]
                    if idx >= 0 and distances[qi][j] >= threshold:
                        all_votes[paths[idx]] += 1

            del index

        except Exception as e:
            logger.warning(f"Error searching {book}: {e}")
            continue

        if (i + 1) % 500 == 0:
            logger.info(f"Searched {i + 1}/{len(book_dirs)} books...")

    return all_votes.most_common(top_n)


def search_disk(image_bytes: bytes, top_k: int = 50, k: int = 5, threshold: float = 0.7, specific_chunks: list = None):
    """
    Main entry point for DISK search.

    Args:
        image_bytes: Query image as bytes
        top_k: Number of results to return
        k: Nearest neighbors per keypoint
        threshold: Minimum similarity for voting
        specific_chunks: Optional list of chunk numbers to search (e.g., [142])

    Returns:
        List of dicts with 'path', 'votes', 'score' keys
    """
    # Extract features
    logger.info("Extracting DISK features from query...")
    descriptors = extract_disk_features(image_bytes)
    logger.info(f"Extracted {len(descriptors)} keypoints")

    if len(descriptors) == 0:
        return []

    # Search
    results = search_chunks(descriptors, k=k, threshold=threshold, top_n=top_k, specific_chunks=specific_chunks)

    # Format results
    formatted = []
    max_votes = results[0][1] if results else 1

    for path, votes in results:
        formatted.append({
            'path': path,
            'votes': votes,
            'score': votes / max_votes,  # Normalize to 0-1
            'verified_matches': votes
        })

    return formatted


# For testing
if __name__ == "__main__":
    import sys
    # Configure logging to show output
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("DISK Search Starting...", flush=True)
    if len(sys.argv) > 1:
        print(f"Reading image: {sys.argv[1]}", flush=True)
        with open(sys.argv[1], 'rb') as f:
            image_bytes = f.read()
        print(f"Image size: {len(image_bytes)} bytes", flush=True)

        # Check for specific chunks argument
        specific_chunks = None
        if len(sys.argv) > 2:
            # Parse chunk numbers from command line (e.g., "142" or "142,200,300")
            specific_chunks = sys.argv[2].split(',')
            print(f"Searching only chunks: {specific_chunks}", flush=True)

        results = search_disk(image_bytes, top_k=10, specific_chunks=specific_chunks)
        print(f"\nResults ({len(results)} found):")
        for r in results:
            print(f"{r['votes']:4d} votes: {r['path']}")
