"""
DISK Keypoint Search - finds source pages for cropped images.

Uses consolidated FAISS chunks for fast searching across multiple collections
(books, print_ads, board_games, albums, comics). Each query keypoint votes
for the source image it matches.

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

from collections_config import get_disk_collections

logger = logging.getLogger(__name__)

# Local SSD buffer for streaming chunks during search (shared across all categories)
LOCAL_CHUNK_BUFFER = "D:/faiss/disk_retrieval/chunk_buffer"

# Cached path lookups per category IDs dir
_id_to_path_cache = {}

# DISK model (lazy loaded)
_disk_model = None
_device = None

# GPU search via PyTorch (works on Windows, no faiss-gpu needed)
_gpu_search_available = None  # None = not checked, True/False after check

def _check_gpu_search():
    """Check if GPU search via PyTorch is available."""
    global _gpu_search_available
    if _gpu_search_available is not None:
        return _gpu_search_available
    _gpu_search_available = torch.cuda.is_available()
    if _gpu_search_available:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU search enabled via PyTorch ({torch.cuda.get_device_name(0)}, {vram_gb:.0f}GB VRAM)")
    else:
        logger.info("GPU search not available, using CPU FAISS")
    return _gpu_search_available

def _gpu_search(index, query_descriptors, k):
    """
    GPU-accelerated brute-force inner product search using PyTorch.

    Replaces faiss IndexFlatIP.search() — same inputs/outputs.
    """
    results = _gpu_search_batch(index, [("single", query_descriptors)], k)
    return results["single"]


def _gpu_search_batch(index, query_list, k):
    """
    GPU-accelerated search for MULTIPLE query descriptor sets against ONE index.

    Loads each DB batch to GPU ONCE and searches ALL query sets against it,
    avoiding redundant CPU->GPU transfers.

    Args:
        index: FAISS IndexFlatIP
        query_list: List of (name, descriptors_ndarray) tuples
        k: Number of nearest neighbors

    Returns:
        Dict of {name: (distances, indices)} numpy arrays
    """
    n_vectors = index.ntotal
    dim = index.d

    # Get all vectors as a numpy view directly from FAISS internal storage (zero copy)
    all_vectors = faiss.vector_to_array(index.codes).view("float32").reshape(n_vectors, dim)

    batch_size = 2_000_000  # ~1GB per batch at dim=128

    # Prepare per-query state on GPU
    query_tensors = {}
    running_distances = {}
    running_indices = {}
    for name, descriptors in query_list:
        if len(descriptors) == 0:
            continue
        query_tensors[name] = torch.from_numpy(descriptors).cuda()
        running_distances[name] = torch.full((len(descriptors), k), -1e9, device='cuda')
        running_indices[name] = torch.full((len(descriptors), k), -1, dtype=torch.long, device='cuda')

    # Max scores matrix ~4GB to prevent OOM (scores = n_query * db_count * 4 bytes)
    MAX_SCORES_BYTES = 4 * 1024 ** 3

    # Load each DB batch to GPU ONCE, search ALL queries against it
    for start in range(0, n_vectors, batch_size):
        end = min(start + batch_size, n_vectors)
        db_count = end - start

        # Single CPU->GPU transfer per batch
        db_tensor = torch.from_numpy(all_vectors[start:end].copy()).cuda()
        db_t = db_tensor.t()  # Transpose once, reuse for all queries
        batch_k = min(k, db_count)

        # Max query keypoints per sub-batch to keep scores matrix under limit
        max_qb = max(1, MAX_SCORES_BYTES // (db_count * 4))

        for name, q_tensor in query_tensors.items():
            n_kp = q_tensor.shape[0]

            for q_start in range(0, n_kp, max_qb):
                q_end = min(q_start + max_qb, n_kp)
                q_batch = q_tensor[q_start:q_end]

                # Inner product: (n_query_sub, dim) @ (dim, db_count) = (n_query_sub, db_count)
                scores = torch.mm(q_batch, db_t)

                batch_scores, batch_idx = scores.topk(batch_k, dim=1)
                batch_idx += start

                # Merge with running top-k for this keypoint slice
                rd = running_distances[name][q_start:q_end]
                ri = running_indices[name][q_start:q_end]
                combined_scores = torch.cat([rd, batch_scores], dim=1)
                combined_indices = torch.cat([ri, batch_idx], dim=1)
                topk_scores, topk_pos = combined_scores.topk(k, dim=1)
                running_distances[name][q_start:q_end] = topk_scores
                running_indices[name][q_start:q_end] = combined_indices.gather(1, topk_pos)

                del scores, batch_scores, batch_idx, combined_scores, combined_indices, rd, ri

        del db_tensor, db_t

    # Collect results
    results = {}
    for name, descriptors in query_list:
        if len(descriptors) == 0:
            results[name] = (np.empty((0, k)), np.empty((0, k), dtype=np.int64))
        else:
            results[name] = (
                running_distances[name].cpu().numpy(),
                running_indices[name].cpu().numpy()
            )

    # Cleanup
    del query_tensors, running_distances, running_indices
    torch.cuda.empty_cache()

    return results


def get_disk_model():
    """Lazy-load DISK feature extractor."""
    global _disk_model, _device
    if _disk_model is None:
        logger.info("Loading DISK model...")
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _disk_model = KF.DISK.from_pretrained('depth').to(_device).eval()
        logger.info(f"DISK model loaded on {_device}")
    return _disk_model, _device


def get_id_to_path(chunk_ids_dir):
    """Load path lookup table for a category. Cached per IDs directory."""
    global _id_to_path_cache
    if chunk_ids_dir not in _id_to_path_cache:
        lookup_file = os.path.join(chunk_ids_dir, "path_lookup.json")
        if os.path.exists(lookup_file):
            logger.info(f"Loading path lookup from {lookup_file}...")
            load_start = time.time()
            with open(lookup_file, 'r') as f:
                _id_to_path_cache[chunk_ids_dir] = json.load(f)
            logger.info(f"Loaded {len(_id_to_path_cache[chunk_ids_dir]):,} path mappings in {time.time()-load_start:.1f}s")
        else:
            logger.info(f"No path_lookup.json found in {chunk_ids_dir}")
            _id_to_path_cache[chunk_ids_dir] = []
    return _id_to_path_cache[chunk_ids_dir]


def load_chunk_paths(chunk_file, chunk_ids_dir):
    """
    Load path data for a chunk. Uses compact IDs if available, falls back to NAS paths.json.

    Args:
        chunk_file: Path to the .faiss chunk file
        chunk_ids_dir: Directory containing compact IDs for this category

    Returns:
        (paths_or_ids, id_to_path_or_None)
        - If IDs available: (np.ndarray of int32 IDs, list of path strings)
        - If fallback: (list of path strings, None)
    """
    chunk_name = os.path.basename(chunk_file).replace('.faiss', '')
    ids_file = os.path.join(chunk_ids_dir, f"{chunk_name}_ids.npy")

    if os.path.exists(ids_file):
        # Fast path: load compact ID array from local SSD
        load_start = time.time()
        ids = np.load(ids_file)
        id_to_path = get_id_to_path(chunk_ids_dir)
        logger.info(f"  Loaded {chunk_name}_ids.npy ({len(ids):,} entries) in {time.time()-load_start:.1f}s")
        return ids, id_to_path
    else:
        # Slow path: read full paths.json from NAS (same dir as chunk)
        chunks_dir = os.path.dirname(chunk_file)
        nas_paths_file = os.path.join(chunks_dir, f"{chunk_name}_paths.json")
        if os.path.exists(nas_paths_file):
            load_start = time.time()
            with open(nas_paths_file, 'r') as f:
                paths = json.load(f)
            logger.info(f"  Loaded {chunk_name}_paths.json from NAS ({len(paths):,} entries) in {time.time()-load_start:.1f}s")
            return paths, None
        else:
            logger.warning(f"  No IDs or paths file found for {chunk_name}")
            return [], None


def resolve_path(paths_or_ids, id_to_path, idx):
    """Resolve a FAISS index to a file path string. Returns None if out of bounds."""
    if idx < 0 or idx >= len(paths_or_ids):
        return None
    if id_to_path is not None and len(id_to_path) > 0:
        compact_id = paths_or_ids[idx]
        if compact_id < 0 or compact_id >= len(id_to_path):
            return None
        return id_to_path[compact_id]
    else:
        return paths_or_ids[idx]


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


def _collect_chunks(categories=None):
    """
    Collect all chunk files across selected categories.

    Returns list of (chunk_file, chunk_ids_dir) tuples sorted by filename,
    and a dict of category -> chunk_count for progress tracking.
    """
    disk_collections = get_disk_collections(categories)

    all_chunks = []
    category_counts = {}

    for cat_name, cat_config in disk_collections.items():
        chunks_dir = cat_config["chunks_dir"]
        ids_dir = cat_config["ids_dir"]

        chunk_files = sorted(glob(os.path.join(chunks_dir, "chunk_*.faiss")),
                             key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
        category_counts[cat_name] = len(chunk_files)

        for cf in chunk_files:
            all_chunks.append((cf, ids_dir))

        if chunk_files:
            logger.info(f"  {cat_name}: {len(chunk_files)} chunks from {chunks_dir}")
        else:
            logger.info(f"  {cat_name}: no chunks found in {chunks_dir}")

    return all_chunks, category_counts


def search_chunks(query_descriptors: np.ndarray, k: int = 5, threshold: float = 0.7, top_n: int = 50, specific_chunks: list = None, categories: list = None, search_id: int = None, progress_callback=None):
    """
    Search consolidated chunks for matching images using streaming mode.

    Args:
        query_descriptors: Normalized DISK descriptors from query image
        k: Number of nearest neighbors per descriptor
        threshold: Minimum similarity score to count as vote
        top_n: Number of top results to return
        specific_chunks: Optional list of chunk numbers to search (e.g., [142, 200])
        categories: Optional list of categories to search (None = all)
        search_id: Optional search session ID for live tracking
        progress_callback: Optional callback for progress updates

    Returns:
        List of (path, votes) tuples sorted by votes descending
    """
    if len(query_descriptors) == 0:
        logger.warning("No keypoints extracted from query image")
        return []

    # Collect chunks from all selected categories
    cat_label = ",".join(categories) if categories else "all"
    logger.info(f"Collecting chunks for categories: {cat_label}")
    all_chunks, category_counts = _collect_chunks(categories)

    # Filter to specific chunks if requested
    if specific_chunks:
        filtered = []
        for chunk_file, ids_dir in all_chunks:
            chunk_name = os.path.basename(chunk_file)
            chunk_num = chunk_name.replace('chunk_', '').replace('.faiss', '')
            if chunk_num in specific_chunks or int(chunk_num) in specific_chunks:
                filtered.append((chunk_file, ids_dir))
        all_chunks = filtered
        logger.info(f"Filtered to specific chunks: {specific_chunks}")

    if all_chunks:
        logger.info(f"Searching {len(all_chunks)} total chunks across {len(category_counts)} categories")
        return _search_chunks_rolling_buffer(query_descriptors, all_chunks, k, threshold, top_n, buffer_size=5, search_id=search_id, progress_callback=progress_callback)
    else:
        logger.warning("No chunks found for any selected category")
        return []


def _copy_chunk_worker(copy_queue: Queue, chunk_list: list, start_idx: int, count: int, buffer_size: int = 5):
    """Background worker to copy chunks ahead of time.

    Throttles so at most buffer_size chunks are in the local buffer at once.
    Waits when the buffer is full before copying the next chunk.

    Args:
        chunk_list: List of (chunk_file, chunk_ids_dir) tuples
        buffer_size: Max chunks to keep ahead in the local buffer
    """
    for i in range(start_idx, min(start_idx + count, len(chunk_list))):
        # Throttle: wait if buffer is full (count .faiss files in buffer dir)
        while True:
            try:
                buffered = len([f for f in os.listdir(LOCAL_CHUNK_BUFFER) if f.endswith('.faiss')])
                if buffered < buffer_size:
                    break
                time.sleep(1)
            except OSError:
                break

        nas_chunk_file, _ = chunk_list[i]
        chunk_name = os.path.basename(nas_chunk_file)
        local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)

        # Check if already cached
        chunk_exists = os.path.exists(local_chunk_file)
        same_size = chunk_exists and os.path.getsize(local_chunk_file) == os.path.getsize(nas_chunk_file)

        if not same_size:
            logger.info(f"  Background: Copying chunk {i+1} ({chunk_name})...")
            copy_start = time.time()
            shutil.copy2(nas_chunk_file, local_chunk_file)
            copy_time = time.time() - copy_start
            chunk_size_gb = os.path.getsize(local_chunk_file) / (1024**3)
            logger.info(f"  Background: Copied {chunk_size_gb:.1f}GB in {copy_time:.1f}s")

        # Signal that this chunk is ready
        copy_queue.put((i, local_chunk_file))


def _search_chunks_rolling_buffer(query_descriptors: np.ndarray, chunk_list: list, k: int, threshold: float, top_n: int, buffer_size: int = 5, search_id: int = None, progress_callback=None):
    """
    Search using rolling buffer: maintain chunks in local buffer, copy next chunk while searching current.

    Args:
        chunk_list: List of (chunk_file, chunk_ids_dir) tuples
        search_id: Optional search session ID for live progress tracking
        progress_callback: Optional callback(chunk_idx, total_chunks, top_results, elapsed_ms)
    """
    all_votes = Counter()
    total_chunks = len(chunk_list)
    use_gpu = _check_gpu_search()

    # Clean buffer of any stale files from previous searches
    if os.path.exists(LOCAL_CHUNK_BUFFER):
        for f in os.listdir(LOCAL_CHUNK_BUFFER):
            try:
                os.remove(os.path.join(LOCAL_CHUNK_BUFFER, f))
            except OSError:
                pass
    os.makedirs(LOCAL_CHUNK_BUFFER, exist_ok=True)
    search_start = time.time()

    # Start continuous background copy of ALL chunks (keeps NAS pipe full)
    copy_queue = Queue()

    logger.info(f"Starting rolling buffer search (buffer size: {buffer_size} chunks, GPU: {use_gpu})")
    copy_thread = Thread(target=_copy_chunk_worker, args=(copy_queue, chunk_list, 0, total_chunks, buffer_size))
    copy_thread.daemon = True
    copy_thread.start()

    for chunk_idx in range(total_chunks):
        nas_chunk_file, chunk_ids_dir = chunk_list[chunk_idx]
        chunk_name = os.path.basename(nas_chunk_file)
        local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)

        # Wait for this chunk to be ready
        logger.info(f"[{chunk_idx + 1}/{total_chunks}] Waiting for {chunk_name}...")
        ready = False
        while not ready:
            try:
                ready_idx, _ = copy_queue.get(timeout=1)
                if ready_idx == chunk_idx:
                    ready = True
                else:
                    copy_queue.put((ready_idx, _))
            except Empty:
                pass  # keep waiting for queue signal

        # Load and search
        logger.info(f"  Searching {chunk_name}...")
        load_start = time.time()
        index = faiss.read_index(local_chunk_file, faiss.IO_FLAG_MMAP)

        # Load paths using this chunk's category-specific IDs dir
        paths_or_ids, id_to_path = load_chunk_paths(nas_chunk_file, chunk_ids_dir)

        load_time = time.time() - load_start
        logger.info(f"  Loaded in {load_time:.1f}s")

        # Search (GPU via PyTorch or CPU via FAISS)
        search_start_chunk = time.time()
        if use_gpu:
            distances, indices = _gpu_search(index, query_descriptors, k)
        else:
            distances, indices = index.search(query_descriptors, k)
        search_time = time.time() - search_start_chunk
        logger.info(f"  Searched in {search_time:.1f}s ({'GPU' if use_gpu else 'CPU'})")

        # Accumulate votes
        for i in range(len(query_descriptors)):
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0 and distances[i][j] >= threshold:
                    path = resolve_path(paths_or_ids, id_to_path, idx)
                    if path is not None:
                        all_votes[path] += 1

        # Free memory
        del index, paths_or_ids

        # Delete this chunk to make room for more in the buffer
        try:
            os.remove(local_chunk_file)
            logger.info(f"  Deleted {chunk_name} from buffer")
        except Exception as e:
            logger.warning(f"  Failed to delete {chunk_name}: {e}")

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


def search_disk(image_bytes: bytes, top_k: int = 50, k: int = 5, threshold: float = 0.7, specific_chunks: list = None, categories: list = None, progress_callback=None):
    """
    Main entry point for DISK search.

    Args:
        image_bytes: Query image as bytes
        top_k: Number of results to return
        k: Nearest neighbors per keypoint
        threshold: Minimum similarity for voting
        specific_chunks: Optional list of chunk numbers to search (e.g., [142])
        categories: Optional list of categories to search (None = all)
        progress_callback: Optional callback for live progress updates

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
    results = search_chunks(descriptors, k=k, threshold=threshold, top_n=top_k, specific_chunks=specific_chunks, categories=categories, progress_callback=progress_callback)

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


def get_total_chunks(categories=None):
    """Get total chunk count across selected categories (for progress tracking)."""
    _, category_counts = _collect_chunks(categories)
    return sum(category_counts.values())


def _search_chunks_rolling_buffer_batch(query_list: list, chunk_list: list, k: int, threshold: float, top_n: int, buffer_size: int = 5, progress_callback=None, check_stopped=None):
    """
    Search using rolling buffer with MULTIPLE query images at once.

    Loads each chunk once and searches ALL query images against it.
    Same rolling buffer pattern as single-image version.

    Args:
        query_list: List of (image_name, descriptors_ndarray) tuples
        chunk_list: List of (chunk_file, chunk_ids_dir) tuples
        progress_callback: Optional callback(chunk_idx, total_chunks, per_image_results, elapsed_ms)
        check_stopped: Optional callable returning set of image names to skip

    Returns:
        Dict of {image_name: Counter()} with vote counts per image
    """
    per_image_votes = {name: Counter() for name, _ in query_list}
    total_chunks = len(chunk_list)
    use_gpu = _check_gpu_search()

    # Clean buffer
    if os.path.exists(LOCAL_CHUNK_BUFFER):
        for f in os.listdir(LOCAL_CHUNK_BUFFER):
            try:
                os.remove(os.path.join(LOCAL_CHUNK_BUFFER, f))
            except OSError:
                pass
    os.makedirs(LOCAL_CHUNK_BUFFER, exist_ok=True)
    search_start = time.time()

    # Start continuous background copy of ALL chunks (keeps NAS pipe full)
    copy_queue = Queue()

    logger.info(f"Starting batch rolling buffer search ({len(query_list)} images, buffer: {buffer_size} chunks, GPU: {use_gpu})")
    copy_thread = Thread(target=_copy_chunk_worker, args=(copy_queue, chunk_list, 0, total_chunks, buffer_size))
    copy_thread.daemon = True
    copy_thread.start()

    for chunk_idx in range(total_chunks):
        nas_chunk_file, chunk_ids_dir = chunk_list[chunk_idx]
        chunk_name = os.path.basename(nas_chunk_file)
        local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)

        # Wait for this chunk to be ready
        logger.info(f"[{chunk_idx + 1}/{total_chunks}] Waiting for {chunk_name}...")
        ready = False
        while not ready:
            try:
                ready_idx, _ = copy_queue.get(timeout=1)
                if ready_idx == chunk_idx:
                    ready = True
                else:
                    copy_queue.put((ready_idx, _))
            except Empty:
                pass

        # Load chunk
        logger.info(f"  Loading {chunk_name}...")
        load_start = time.time()
        index = faiss.read_index(local_chunk_file, faiss.IO_FLAG_MMAP)
        paths_or_ids, id_to_path = load_chunk_paths(nas_chunk_file, chunk_ids_dir)
        load_time = time.time() - load_start
        logger.info(f"  Loaded in {load_time:.1f}s")

        # Check for stopped images every 10 chunks
        if check_stopped and chunk_idx % 10 == 0:
            stopped = check_stopped()
            if stopped:
                active_query_list = [(n, d) for n, d in query_list if n not in stopped]
                if not active_query_list:
                    logger.info("All images stopped by user. Ending search.")
                    break
                if len(active_query_list) < len(query_list):
                    logger.info(f"  Skipping {len(query_list) - len(active_query_list)} stopped images")
                    query_list = active_query_list

        # Search ALL query images against this chunk
        search_start_chunk = time.time()

        if use_gpu:
            # Batch GPU search: load DB vectors to GPU ONCE, search all images
            gpu_results = _gpu_search_batch(index, query_list, k)
            gpu_time = time.time() - search_start_chunk
            logger.info(f"  GPU search ({len(query_list)} images): {gpu_time:.1f}s")

            # Accumulate votes from GPU results
            vote_start = time.time()
            for name, descriptors in query_list:
                if len(descriptors) == 0:
                    continue
                img_start = time.time()
                distances, indices = gpu_results[name]
                votes = per_image_votes[name]
                matched = 0
                for i in range(len(descriptors)):
                    for j in range(k):
                        idx = indices[i][j]
                        if idx >= 0 and distances[i][j] >= threshold:
                            path = resolve_path(paths_or_ids, id_to_path, idx)
                            if path is not None:
                                votes[path] += 1
                                matched += 1
                top_votes = votes.most_common(1)[0][1] if votes else 0
                logger.info(f"    {name}: {time.time() - img_start:.1f}s ({len(descriptors)} kp, {matched} matches, top={top_votes} votes)")
            vote_time = time.time() - vote_start
            logger.info(f"  Vote accumulation: {vote_time:.1f}s")
            del gpu_results
        else:
            for name, descriptors in query_list:
                if len(descriptors) == 0:
                    continue
                img_start = time.time()
                distances, indices = index.search(descriptors, k)
                votes = per_image_votes[name]
                for i in range(len(descriptors)):
                    for j in range(k):
                        idx = indices[i][j]
                        if idx >= 0 and distances[i][j] >= threshold:
                            path = resolve_path(paths_or_ids, id_to_path, idx)
                            if path is not None:
                                votes[path] += 1
                logger.info(f"    {name}: {time.time() - img_start:.1f}s ({len(descriptors)} keypoints)")

        search_time = time.time() - search_start_chunk
        logger.info(f"  Searched {len(query_list)} images in {search_time:.1f}s ({'GPU' if use_gpu else 'CPU'})")

        # Free memory
        del index, paths_or_ids

        # Delete chunk to make room for more in the buffer
        try:
            os.remove(local_chunk_file)
        except Exception as e:
            logger.warning(f"  Failed to delete {chunk_name}: {e}")

        # Progress
        elapsed = time.time() - search_start
        avg_per_chunk = elapsed / (chunk_idx + 1)
        remaining = (total_chunks - chunk_idx - 1) * avg_per_chunk
        logger.info(f"  Progress: {chunk_idx + 1}/{total_chunks} chunks | ETA: {remaining/60:.1f}m")

        if progress_callback:
            per_image_results = {}
            for name, votes in per_image_votes.items():
                if votes:
                    max_v = votes.most_common(1)[0][1]
                    per_image_results[name] = [
                        {'path': p, 'votes': v, 'score': v / max_v}
                        for p, v in votes.most_common(100)
                    ]
                else:
                    per_image_results[name] = []
            progress_callback(chunk_idx + 1, total_chunks, per_image_results, int(elapsed * 1000))

    total_time = time.time() - search_start
    logger.info(f"Batch search complete: {total_chunks} chunks, {len(query_list)} images in {total_time/60:.1f}m")

    return per_image_votes


def search_disk_batch(image_list: list, top_k: int = 50, k: int = 5, threshold: float = 0.7, categories: list = None, progress_callback=None, check_stopped=None):
    """
    Batch DISK search: search multiple images in one pass through all chunks.

    Args:
        image_list: List of (image_bytes, image_name) tuples
        top_k: Number of results per image
        k: Nearest neighbors per keypoint
        threshold: Minimum similarity for voting
        categories: Optional list of categories (None = all)
        progress_callback: Optional callback(chunk_idx, total_chunks, per_image_results, elapsed_ms)
        check_stopped: Optional callable returning set of image names to skip

    Returns:
        Dict of {image_name: [{'path', 'votes', 'score'}, ...]}
    """
    # Extract features for all images
    query_list = []
    for image_bytes, image_name in image_list:
        logger.info(f"Extracting features: {image_name}...")
        descriptors = extract_disk_features(image_bytes)
        logger.info(f"  {image_name}: {len(descriptors)} keypoints")
        query_list.append((image_name, descriptors))

    # Collect chunks
    cat_label = ",".join(categories) if categories else "all"
    logger.info(f"Collecting chunks for categories: {cat_label}")
    all_chunks, category_counts = _collect_chunks(categories)

    if not all_chunks:
        logger.warning("No chunks found")
        return {name: [] for name, _ in query_list}

    logger.info(f"Searching {len(all_chunks)} chunks across {len(category_counts)} categories")

    # Search
    per_image_votes = _search_chunks_rolling_buffer_batch(
        query_list, all_chunks, k, threshold, top_k,
        buffer_size=5, progress_callback=progress_callback,
        check_stopped=check_stopped
    )

    # Format results
    results = {}
    for name, votes in per_image_votes.items():
        top = votes.most_common(top_k)
        max_v = top[0][1] if top else 1
        results[name] = [
            {'path': p, 'votes': v, 'score': v / max_v, 'verified_matches': v}
            for p, v in top
        ]

    return results


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
            specific_chunks = sys.argv[2].split(',')
            print(f"Searching only chunks: {specific_chunks}", flush=True)

        # Check for categories argument
        categories = None
        if len(sys.argv) > 3:
            categories = sys.argv[3].split(',')
            print(f"Searching categories: {categories}", flush=True)

        results = search_disk(image_bytes, top_k=10, specific_chunks=specific_chunks, categories=categories)
        print(f"\nResults ({len(results)} found):")
        for r in results:
            print(f"{r['votes']:4d} votes: {r['path']}")
