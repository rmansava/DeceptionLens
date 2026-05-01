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
import pickle
import re
from pathlib import Path
from glob import glob
from collections import Counter
import logging
import torch
import time
import kornia.feature as KF
import kornia as K
from threading import Thread
from queue import Queue, Empty
from heapq import nlargest

from collections_config import COLLECTIONS, get_disk_collections

logger = logging.getLogger(__name__)

# Local SSD buffer for streaming chunks during search (shared across all categories)
LOCAL_CHUNK_BUFFER = "D:/faiss/disk_retrieval/chunk_buffer"
# Direct NAS mode: mmap chunks directly from NAS instead of copying to local SSD.
# Faster on 10GbE (~62s/chunk vs ~113s with copy).
# Set to comma-separated drive letters that support fast direct reads (10GbE).
# Chunks on other drives use the rolling buffer (copy to local SSD first).
DIRECT_NAS_DRIVES = set(
    d.strip().upper().rstrip(":")
    for d in os.environ.get("DISK_DIRECT_NAS_DRIVES", "").split(",")
    if d.strip()
)
SEARCH_CHECKPOINT_DIR = os.environ.get(
    "DISK_SEARCH_CHECKPOINT_DIR",
    "D:/faiss/disk_retrieval/search_checkpoints"
)
SEARCH_CHECKPOINT_INTERVAL = max(1, int(os.environ.get("DISK_SEARCH_CHECKPOINT_INTERVAL", "5")))

# Cached path lookups per category IDs dir
_id_to_path_cache = {}

# DISK model (lazy loaded)
_disk_model = None
_device = None

# GPU search via PyTorch (works on Windows, no faiss-gpu needed)
_gpu_search_available = None  # None = not checked, True/False after check
GPU_SEARCH_BATCH_SIZE = int(os.environ.get("DISK_GPU_BATCH_SIZE", "4000000"))
GPU_SEARCH_USE_FP16 = os.environ.get("DISK_GPU_FP16", "1") != "0"
GPU_SEARCH_MAX_SCORES_BYTES = int(float(os.environ.get("DISK_GPU_MAX_SCORES_GB", "4")) * (1024 ** 3))
MAX_QUERY_KEYPOINTS = int(os.environ.get("DISK_MAX_QUERY_KEYPOINTS", "2000"))
DISK_PAGE0_DEBOOST = float(os.environ.get("DISK_PAGE0_DEBOOST", "0.5"))
DISK_PAGE0_DEBOOST = min(1.0, max(0.0, DISK_PAGE0_DEBOOST))
_PAGE0_BASENAME_RE = re.compile(r"(^|[-_])page0\.[a-z0-9]+$", re.IGNORECASE)
DISK_LOCALIZE_MIN_MATCHES = max(4, int(os.environ.get("DISK_LOCALIZE_MIN_MATCHES", "8")))
DISK_LOCALIZE_SIM_THRESHOLD = float(os.environ.get("DISK_LOCALIZE_SIM_THRESHOLD", "0.7"))
DISK_LOCALIZE_SIM_THRESHOLD = min(1.0, max(0.0, DISK_LOCALIZE_SIM_THRESHOLD))
DISK_LOCALIZE_RANSAC_REPROJ = float(os.environ.get("DISK_LOCALIZE_RANSAC_REPROJ", "8.0"))
DISK_LOCALIZE_TOP_N = max(1, int(os.environ.get("DISK_LOCALIZE_TOP_N", "1000")))
DISK_RERANK_ENABLED_DEFAULT = os.environ.get("DISK_RERANK_ENABLED_DEFAULT", "0") == "1"
DISK_RERANK_INLIER_WEIGHT = float(os.environ.get("DISK_RERANK_INLIER_WEIGHT", "1000"))
DISK_RERANK_MATCH_WEIGHT = float(os.environ.get("DISK_RERANK_MATCH_WEIGHT", "10"))
DISK_FEATURE_ROOTS = [
    root.strip().replace("\\", "/").rstrip("/")
    for root in os.environ.get("DISK_FEATURE_ROOTS", "T:/disk-features,S:/disk-features").split(",")
    if root.strip()
]


def _is_page0_result(path: str) -> bool:
    if not path:
        return False
    base = os.path.basename(path.replace("\\", "/"))
    return _PAGE0_BASENAME_RE.search(base) is not None


def _adjusted_vote(path: str, raw_votes: int) -> float:
    if DISK_PAGE0_DEBOOST < 1.0 and _is_page0_result(path):
        return float(raw_votes) * DISK_PAGE0_DEBOOST
    return float(raw_votes)


def _rank_vote_counter(votes: Counter, top_n: int, excluded_paths: set = None) -> list:
    """
    Rank vote results with optional page-0 deboosting.

    Returns list of (path, raw_votes, adjusted_votes), sorted descending by adjusted score.
    """
    if top_n <= 0 or not votes:
        return []
    excluded_paths = excluded_paths or set()
    candidate_items = votes.items()
    if excluded_paths:
        candidate_items = (
            (path, raw_votes)
            for path, raw_votes in votes.items()
            if _normalize_result_path(path) not in excluded_paths
        )

    top_items = nlargest(
        top_n,
        candidate_items,
        key=lambda item: (_adjusted_vote(item[0], item[1]), item[1])
    )
    return [(path, int(raw), _adjusted_vote(path, int(raw))) for path, raw in top_items]


def _normalize_result_path(path: str) -> str:
    return (path or "").replace("\\", "/").rstrip("/").lower()


def _build_disk_feature_roots() -> dict:
    """
    Build collection -> list of disk-features roots, including S:/ fallback.
    """
    roots = {}
    for collection, cfg in COLLECTIONS.items():
        base = cfg.get("disk_features")
        if not base:
            continue
        candidates = [Path(base)]
        base_name = Path(base).name
        for root in DISK_FEATURE_ROOTS:
            candidates.append(Path(root) / base_name)

        # Preserve order, drop duplicates.
        seen = set()
        uniq = []
        for p in candidates:
            norm = str(p).replace("\\", "/").lower()
            if norm in seen:
                continue
            seen.add(norm)
            uniq.append(p)
        roots[collection] = uniq
    return roots


_DISK_FEATURE_ROOTS_BY_COLLECTION = _build_disk_feature_roots()

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

    # Get a zero-copy numpy view of FAISS internal storage (avoids 10GB allocation)
    xb = faiss.rev_swig_ptr(index.get_xb(), n_vectors * dim)
    all_vectors = xb.reshape(n_vectors, dim)

    batch_size = GPU_SEARCH_BATCH_SIZE

    # Prepare per-query state on GPU
    query_tensors = {}
    running_distances = {}
    running_indices = {}
    tensor_dtype = torch.float16 if GPU_SEARCH_USE_FP16 else torch.float32
    distance_floor = -2.0 if GPU_SEARCH_USE_FP16 else -1e9
    for name, descriptors in query_list:
        if len(descriptors) == 0:
            continue
        query_tensors[name] = torch.from_numpy(np.ascontiguousarray(descriptors)).to(device='cuda', dtype=tensor_dtype)
        running_distances[name] = torch.full((len(descriptors), k), distance_floor, dtype=tensor_dtype, device='cuda')
        running_indices[name] = torch.full((len(descriptors), k), -1, dtype=torch.long, device='cuda')

    score_elem_bytes = 2 if GPU_SEARCH_USE_FP16 else 4
    min_batch_vectors = 250_000
    current = 0

    # Load each DB batch to GPU ONCE, search ALL queries against it
    while current < n_vectors:
        target_end = min(current + batch_size, n_vectors)
        end = target_end

        while True:
            db_count = end - current
            db_slice = None
            db_tensor = None
            db_t = None
            try:
                # If FAISS slice is already contiguous, this avoids an unnecessary copy.
                db_slice = np.ascontiguousarray(all_vectors[current:end])
                db_tensor = torch.from_numpy(db_slice).to(device='cuda', dtype=tensor_dtype)
                db_t = db_tensor.t()  # Transpose once, reuse for all queries
                batch_k = min(k, db_count)

                # Max query keypoints per sub-batch to keep scores matrix under the memory cap
                max_qb = max(1, GPU_SEARCH_MAX_SCORES_BYTES // (db_count * score_elem_bytes))

                for name, q_tensor in query_tensors.items():
                    n_kp = q_tensor.shape[0]

                    for q_start in range(0, n_kp, max_qb):
                        q_end = min(q_start + max_qb, n_kp)
                        q_batch = q_tensor[q_start:q_end]

                        # Inner product: (n_query_sub, dim) @ (dim, db_count) = (n_query_sub, db_count)
                        try:
                            scores = torch.mm(q_batch, db_t)
                        except RuntimeError:
                            # FP16 CUBLAS error fallback to FP32
                            scores = torch.mm(q_batch.float(), db_t.float()).half() if tensor_dtype == torch.float16 else torch.mm(q_batch, db_t)

                        batch_scores, batch_idx = scores.topk(batch_k, dim=1)
                        batch_idx += current

                        # Merge with running top-k for this keypoint slice
                        rd = running_distances[name][q_start:q_end]
                        ri = running_indices[name][q_start:q_end]
                        combined_scores = torch.cat([rd, batch_scores], dim=1)
                        combined_indices = torch.cat([ri, batch_idx], dim=1)
                        topk_scores, topk_pos = combined_scores.topk(k, dim=1)
                        running_distances[name][q_start:q_end] = topk_scores
                        running_indices[name][q_start:q_end] = combined_indices.gather(1, topk_pos)

                        del scores, batch_scores, batch_idx, combined_scores, combined_indices, rd, ri

                del db_tensor, db_t, db_slice
                current = end
                break

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                del db_tensor, db_t, db_slice
                torch.cuda.empty_cache()
                if db_count <= min_batch_vectors:
                    raise
                end = current + max(min_batch_vectors, db_count // 2)
                logger.warning(
                    f"GPU OOM in DISK search batch ({db_count:,} vectors). "
                    f"Retrying with {end - current:,} vectors."
                )

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


def _checkpoint_path(search_id: int) -> str:
    return os.path.join(SEARCH_CHECKPOINT_DIR, f"search_{search_id}.pkl")


def save_search_checkpoint(search_id: int, current_chunk: int, total_chunks: int, votes: Counter):
    """Persist DISK search state for resume support."""
    if not search_id:
        return

    try:
        os.makedirs(SEARCH_CHECKPOINT_DIR, exist_ok=True)
        payload = {
            "search_id": int(search_id),
            "current_chunk": int(current_chunk),
            "total_chunks": int(total_chunks),
            "votes": dict(votes),
            "saved_at": time.time()
        }
        path = _checkpoint_path(search_id)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"Failed to save DISK checkpoint for search #{search_id}: {e}")


def load_search_checkpoint(search_id: int):
    """Load persisted DISK search state."""
    if not search_id:
        return None

    path = _checkpoint_path(search_id)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load DISK checkpoint for search #{search_id}: {e}")
        return None


def clear_search_checkpoint(search_id: int):
    """Remove persisted DISK search state."""
    if not search_id:
        return
    path = _checkpoint_path(search_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to clear DISK checkpoint for search #{search_id}: {e}")


def _accumulate_votes_vectorized(votes: Counter, distances, indices, threshold, paths_or_ids, id_to_path):
    """
    Vectorized vote accumulation from kNN results.

    Returns the number of unique (query keypoint, page) votes that were counted.
    """
    if distances.size == 0 or indices.size == 0 or len(paths_or_ids) == 0:
        return 0

    mask = (indices >= 0) & (distances >= threshold)
    if not np.any(mask):
        return 0

    row_idx, col_idx = np.nonzero(mask)
    if row_idx.size == 0:
        return 0

    valid_indices = indices[row_idx, col_idx].astype(np.int64, copy=False)
    in_range = (valid_indices >= 0) & (valid_indices < len(paths_or_ids))
    if not np.any(in_range):
        return 0
    row_idx = row_idx[in_range]
    valid_indices = valid_indices[in_range]

    if id_to_path is not None and len(id_to_path) > 0:
        compact_ids = np.asarray(paths_or_ids[valid_indices], dtype=np.int64)
        compact_in_range = (compact_ids >= 0) & (compact_ids < len(id_to_path))
        if not np.any(compact_in_range):
            return 0
        row_idx = row_idx[compact_in_range]
        compact_ids = compact_ids[compact_in_range]

        # Deduplicate repeated neighbors from the same query keypoint that hit the same page.
        # Without this, one keypoint can cast up to k votes for one page in dense "hub" pages.
        keypoint_page_pairs = np.empty((compact_ids.size, 2), dtype=np.int64)
        keypoint_page_pairs[:, 0] = row_idx.astype(np.int64, copy=False)
        keypoint_page_pairs[:, 1] = compact_ids
        unique_pairs = np.unique(keypoint_page_pairs, axis=0)

        unique_ids, counts = np.unique(unique_pairs[:, 1], return_counts=True)
        for compact_id, count in zip(unique_ids.tolist(), counts.tolist()):
            votes[id_to_path[compact_id]] += int(count)
        return int(unique_pairs.shape[0])

    # Fallback mode (paths.json): dedupe in Python because paths are strings.
    seen = set()
    page_counts = Counter()
    for q_row, vector_idx in zip(row_idx.tolist(), valid_indices.tolist()):
        path = paths_or_ids[vector_idx]
        if path is None:
            continue
        key = (int(q_row), path)
        if key in seen:
            continue
        seen.add(key)
        page_counts[path] += 1

    for path, count in page_counts.items():
        votes[path] += int(count)
    return int(len(seen))


def extract_disk_features_bundle(image_bytes: bytes):
    """Extract DISK keypoints/descriptors and image size from image bytes."""
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
        return {
            "descriptors": np.array([]).reshape(0, 128).astype('float32'),
            "keypoints": np.array([]).reshape(0, 2).astype('float32'),
            "image_size": (w, h),
        }

    # Normalize descriptors
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / (norms + 1e-8)

    # Cap keypoints to limit GPU search time in batch mode
    if MAX_QUERY_KEYPOINTS > 0 and len(descriptors) > MAX_QUERY_KEYPOINTS:
        # Keep keypoints with highest pre-normalization norms (most distinctive)
        top_idx = np.argsort(norms.ravel())[-MAX_QUERY_KEYPOINTS:]
        descriptors = descriptors[top_idx]
        keypoints = keypoints[top_idx]

    return {
        "descriptors": descriptors.astype('float32'),
        "keypoints": keypoints.astype('float32'),
        "image_size": (w, h),
    }


def extract_disk_features(image_bytes: bytes) -> np.ndarray:
    """Extract DISK keypoint descriptors from image bytes."""
    return extract_disk_features_bundle(image_bytes)["descriptors"]


def _normalize_slashes(path: str) -> str:
    return (path or "").replace("\\", "/").rstrip("/")


def _collection_source_prefixes(collection: str, config: dict) -> list:
    prefixes = []
    source_path = _normalize_slashes(config.get("source_path", ""))
    if source_path:
        prefixes.append(source_path)
        if source_path.startswith("T:/"):
            prefixes.append("S:/" + source_path[3:])
        if source_path.startswith("S:/"):
            prefixes.append("T:/" + source_path[3:])

    if collection == "books":
        prefixes.extend([
            "D:/books",
            "D:/books/pdf-images",
            "T:/archiverelated/books/pdf-images",
            "S:/archiverelated/books/pdf-images",
        ])

    seen = set()
    uniq = []
    for p in prefixes:
        norm = p.lower()
        if norm in seen:
            continue
        seen.add(norm)
        uniq.append(p)
    return uniq


def _image_path_to_disk_npz_path(image_path: str):
    """
    Map a source image path to a DISK feature .npz path.

    Returns:
        (npz_path, collection) or (None, None)
    """
    normalized_path = _normalize_slashes(image_path)
    normalized_lower = normalized_path.lower()

    for collection, config in COLLECTIONS.items():
        roots = _DISK_FEATURE_ROOTS_BY_COLLECTION.get(collection)
        if not roots:
            continue

        rel = None
        for prefix in _collection_source_prefixes(collection, config):
            pfx = prefix.lower()
            if normalized_lower == pfx:
                rel = ""
                break
            if normalized_lower.startswith(pfx + "/"):
                rel = normalized_path[len(prefix):].lstrip("/\\")
                break

        if rel is None:
            continue

        rel = _normalize_slashes(rel)
        if collection == "books" and rel.lower().startswith("pdf-images/"):
            rel = rel[len("pdf-images/"):]
        if not rel:
            continue

        rel_npz = str(Path(rel).with_suffix(".npz")).replace("\\", "/")

        for root in roots:
            candidate = root / rel_npz
            if candidate.exists():
                return str(candidate), collection

    return None, None


def _load_disk_npz(npz_path: str):
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if "descriptors" not in npz or "keypoints" not in npz:
                return None

            descriptors = np.asarray(npz["descriptors"], dtype=np.float32)
            keypoints = np.asarray(npz["keypoints"], dtype=np.float32)

            if descriptors.ndim != 2 or descriptors.shape[1] != 128:
                return None
            if keypoints.ndim != 2 or keypoints.shape[1] != 2:
                return None

            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            descriptors = descriptors / (norms + 1e-8)

            width = 0
            height = 0
            if "image_size" in npz:
                size = np.asarray(npz["image_size"]).reshape(-1)
                if size.size >= 2:
                    # Stored as [height, width]
                    height = int(size[0])
                    width = int(size[1])
    except Exception:
        return None

    if width <= 0:
        width = int(np.ceil(np.max(keypoints[:, 0])) + 1) if len(keypoints) else 0
    if height <= 0:
        height = int(np.ceil(np.max(keypoints[:, 1])) + 1) if len(keypoints) else 0

    return {
        "descriptors": descriptors,
        "keypoints": keypoints,
        "image_size": (max(width, 1), max(height, 1)),
    }


def _estimate_match_box(query_features: dict, candidate_features: dict):
    """
    Estimate where query appears in candidate image.

    Returns dict with normalized box coordinates and match stats, or None.
    """
    try:
        import cv2
    except Exception:
        return None

    q_desc = query_features["descriptors"]
    q_kp = query_features["keypoints"]
    qw, qh = query_features["image_size"]

    c_desc = candidate_features["descriptors"]
    c_kp = candidate_features["keypoints"]
    cw, ch = candidate_features["image_size"]

    if len(q_desc) < 4 or len(c_desc) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    raw_matches = matcher.match(q_desc, c_desc)
    if not raw_matches:
        return None

    max_l2_distance = float(np.sqrt(max(0.0, 2.0 - (2.0 * DISK_LOCALIZE_SIM_THRESHOLD))))
    good = [m for m in raw_matches if m.distance <= max_l2_distance]
    if len(good) < DISK_LOCALIZE_MIN_MATCHES:
        return None

    src_pts = np.float32([q_kp[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([c_kp[m.trainIdx] for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, DISK_LOCALIZE_RANSAC_REPROJ)
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if H is not None and inlier_count >= 4:
        corners = np.float32([[0, 0], [qw - 1, 0], [qw - 1, qh - 1], [0, qh - 1]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    else:
        if inlier_mask is not None and inlier_count > 0:
            projected = dst_pts[inlier_mask.ravel().astype(bool)].reshape(-1, 2)
        else:
            projected = dst_pts.reshape(-1, 2)

    if projected.size == 0:
        return None

    x1 = float(np.min(projected[:, 0]))
    y1 = float(np.min(projected[:, 1]))
    x2 = float(np.max(projected[:, 0]))
    y2 = float(np.max(projected[:, 1]))

    x1 = max(0.0, min(float(cw - 1), x1))
    y1 = max(0.0, min(float(ch - 1), y1))
    x2 = max(0.0, min(float(cw - 1), x2))
    y2 = max(0.0, min(float(ch - 1), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "match_x1": x1 / float(cw),
        "match_y1": y1 / float(ch),
        "match_x2": x2 / float(cw),
        "match_y2": y2 / float(ch),
        "match_inliers": inlier_count,
        "match_total": int(len(good)),
    }


def localize_disk_results(image_bytes: bytes, results: list, top_n: int = None):
    """
    Secondary pass: estimate highlight regions for top DISK candidates.

    Args:
        image_bytes: Query image bytes.
        results: DISK result rows (mutated in place).
        top_n: Max number of candidates to localize (defaults to DISK_LOCALIZE_TOP_N).
    """
    if not results:
        return results

    limit = min(len(results), top_n or DISK_LOCALIZE_TOP_N)
    if limit <= 0:
        return results

    query_features = extract_disk_features_bundle(image_bytes)
    if len(query_features["descriptors"]) == 0:
        return results

    start = time.time()
    localized = 0
    for i in range(limit):
        path = results[i].get("path")
        if not path:
            continue

        npz_path, _ = _image_path_to_disk_npz_path(path)
        if not npz_path:
            continue

        candidate = _load_disk_npz(npz_path)
        if not candidate:
            continue

        match_box = _estimate_match_box(query_features, candidate)
        if not match_box:
            continue

        results[i].update(match_box)
        localized += 1

    elapsed = time.time() - start
    logger.info(
        f"Secondary localization: {localized}/{limit} candidates in {elapsed:.1f}s "
        f"({(elapsed / max(limit, 1)):.3f}s each)"
    )
    return results


def rerank_disk_results(results: list) -> list:
    """
    Re-rank DISK results using geometric verification signals when available.

    This is a post-search rerank. Votes remain as fallback for rows without
    localization fields.
    """
    if not results:
        return results

    ranked = []
    for idx, row in enumerate(results):
        inliers = int(row.get("match_inliers") or 0)
        total = int(row.get("match_total") or 0)
        votes = int(row.get("votes") or row.get("verified_matches") or 0)
        adjusted_votes = _adjusted_vote(row.get("path", ""), votes)

        if inliers >= 20:
            tier = 3
        elif inliers >= 10:
            tier = 2
        elif inliers >= 5:
            tier = 1
        else:
            tier = 0

        rerank_score = (
            (tier * 1_000_000_000.0) +
            (inliers * DISK_RERANK_INLIER_WEIGHT) +
            (total * DISK_RERANK_MATCH_WEIGHT) +
            adjusted_votes
        )
        item = dict(row)
        item["rerank_score"] = float(rerank_score)
        ranked.append((rerank_score, idx, item))

    ranked.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return [item for _, _, item in ranked]


def _collect_chunks(categories=None):
    """
    Collect all chunk files across selected categories.

    Returns list of (chunk_file, chunk_ids_dir) tuples in SEARCH_ORDER priority,
    and a dict of category -> chunk_count for progress tracking.
    For multi-dir collections, deduplicates by chunk name (first dir wins).
    """
    disk_collections = get_disk_collections(categories)

    all_chunks = []
    category_counts = {}

    for cat_name, cat_config in disk_collections.items():
        chunks_dirs = cat_config["chunks_dirs"]
        ids_dir = cat_config["ids_dir"]

        # Collect from all dirs, deduplicate by chunk name (first dir wins)
        seen_chunks = set()
        cat_chunks = []
        for chunks_dir in chunks_dirs:
            chunk_files = sorted(glob(os.path.join(chunks_dir, "chunk_*.faiss")),
                                 key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
            for cf in chunk_files:
                chunk_name = os.path.basename(cf)
                if chunk_name not in seen_chunks:
                    seen_chunks.add(chunk_name)
                    cat_chunks.append((cf, ids_dir))

        category_counts[cat_name] = len(cat_chunks)
        all_chunks.extend(cat_chunks)

        if cat_chunks:
            dir_summary = ", ".join(f"{d}" for d in chunks_dirs)
            logger.info(f"  {cat_name}: {len(cat_chunks)} chunks from {dir_summary}")
        else:
            logger.info(f"  {cat_name}: no chunks found")

    return all_chunks, category_counts


def search_chunks(
    query_descriptors: np.ndarray,
    k: int = 5,
    threshold: float = 0.7,
    top_n: int = 50,
    specific_chunks: list = None,
    categories: list = None,
    search_id: int = None,
    progress_callback=None,
    check_stopped=None,
    start_chunk: int = 1,
    initial_votes: dict = None,
    excluded_paths: set = None
):
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
        start_chunk: Optional global chunk position (1-based) to resume from
        initial_votes: Optional existing vote counts to seed resume

    Returns:
        List of (path, votes) tuples sorted by votes descending
    """
    if len(query_descriptors) == 0:
        logger.warning("No keypoints extracted from query image")
        return []

    excluded_paths = {_normalize_result_path(p) for p in (excluded_paths or set()) if p}

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
        return _search_chunks_rolling_buffer(
            query_descriptors,
            all_chunks,
            k,
            threshold,
            top_n,
            buffer_size=5,
            search_id=search_id,
            progress_callback=progress_callback,
            check_stopped=check_stopped,
            start_chunk=start_chunk,
            initial_votes=initial_votes,
            excluded_paths=excluded_paths
        )
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


def _search_chunks_rolling_buffer(
    query_descriptors: np.ndarray,
    chunk_list: list,
    k: int,
    threshold: float,
    top_n: int,
    buffer_size: int = 5,
    search_id: int = None,
    progress_callback=None,
    check_stopped=None,
    start_chunk: int = 1,
    initial_votes: dict = None,
    excluded_paths: set = None
):
    """
    Search using rolling buffer: maintain chunks in local buffer, copy next chunk while searching current.

    Args:
        chunk_list: List of (chunk_file, chunk_ids_dir) tuples
        search_id: Optional search session ID for live progress tracking
        progress_callback: Optional callback(chunk_idx, total_chunks, top_results, elapsed_ms)
    """
    all_votes = Counter(initial_votes or {})
    excluded_paths = excluded_paths or set()
    total_chunks = len(chunk_list)
    start_idx = max(0, min(total_chunks, int(start_chunk) - 1))
    use_gpu = _check_gpu_search()
    last_completed_chunk = start_idx
    stopped = False

    if start_idx > 0:
        logger.info(f"Resuming DISK search from chunk {start_idx + 1}/{total_chunks}")

    search_start = time.time()

    # Split chunks into direct-NAS (fast drives) and buffer-needed (slow drives)
    buffer_chunks = [
        (i, cf, ids) for i, (cf, ids) in enumerate(chunk_list)
        if i >= start_idx and os.path.splitdrive(cf)[0].rstrip(":").upper() not in DIRECT_NAS_DRIVES
    ]
    needs_buffer = len(buffer_chunks) > 0
    copy_queue = None

    logger.info(f"Starting search ({total_chunks} chunks, direct NAS drives: {DIRECT_NAS_DRIVES}, GPU: {use_gpu})")
    if needs_buffer:
        # Clean buffer of any stale files from previous searches
        if os.path.exists(LOCAL_CHUNK_BUFFER):
            for f in os.listdir(LOCAL_CHUNK_BUFFER):
                try:
                    os.remove(os.path.join(LOCAL_CHUNK_BUFFER, f))
                except OSError:
                    pass
        os.makedirs(LOCAL_CHUNK_BUFFER, exist_ok=True)

        # Build a chunk list of only the buffer-needed chunks for the copy worker
        buffer_chunk_list = [(cf, ids) for _, cf, ids in buffer_chunks]
        buffer_idx_set = {i for i, _, _ in buffer_chunks}
        copy_queue = Queue()
        logger.info(f"  Rolling buffer for {len(buffer_chunks)} slow-drive chunks (buffer size: {buffer_size})")
        copy_thread = Thread(
            target=_copy_chunk_worker,
            args=(copy_queue, buffer_chunk_list, 0, len(buffer_chunk_list), buffer_size)
        )
        copy_thread.daemon = True
        copy_thread.start()
        buffer_copy_idx = 0  # tracks position in buffer_chunk_list

    for chunk_idx in range(start_idx, total_chunks):
        # Check if search was stopped
        if check_stopped and check_stopped():
            logger.info(f"Search stopped by user at chunk {chunk_idx}/{total_chunks}")
            stopped = True
            break

        nas_chunk_file, chunk_ids_dir = chunk_list[chunk_idx]
        chunk_name = os.path.basename(nas_chunk_file)
        chunk_drive = os.path.splitdrive(nas_chunk_file)[0].rstrip(":").upper()
        use_direct = chunk_drive in DIRECT_NAS_DRIVES

        if use_direct:
            # Direct NAS mode: mmap straight from NAS path
            logger.info(f"[{chunk_idx + 1}/{total_chunks}] {chunk_name} (direct)...")
            load_start = time.time()
            index = faiss.read_index(nas_chunk_file, faiss.IO_FLAG_MMAP)
        else:
            # Rolling buffer mode: wait for local copy
            local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)

            logger.info(f"[{chunk_idx + 1}/{total_chunks}] Waiting for {chunk_name} (buffer)...")
            ready = False
            while not ready:
                if check_stopped and check_stopped():
                    stopped = True
                    break
                try:
                    ready_idx, _ = copy_queue.get(timeout=1)
                    if ready_idx == buffer_copy_idx:
                        ready = True
                        buffer_copy_idx += 1
                    else:
                        copy_queue.put((ready_idx, _))
                except Empty:
                    pass  # keep waiting for queue signal

            if check_stopped and check_stopped():
                logger.info(f"Search stopped by user while waiting for chunk")
                stopped = True
                break

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
            try:
                distances, indices = _gpu_search_batch(index, [("query", query_descriptors)], k)["query"]
                search_method = "GPU"
            except Exception as gpu_err:
                logger.error(f"  GPU search failed, falling back to CPU: {gpu_err}")
                distances, indices = index.search(query_descriptors, k)
                search_method = "CPU-fallback"
        else:
            distances, indices = index.search(query_descriptors, k)
            search_method = "CPU"
        search_time = time.time() - search_start_chunk
        logger.info(f"  Searched in {search_time:.1f}s ({search_method})")

        # Accumulate votes
        _accumulate_votes_vectorized(
            all_votes, distances, indices, threshold, paths_or_ids, id_to_path
        )
        last_completed_chunk = chunk_idx + 1

        # Free memory
        del index, paths_or_ids

        if not use_direct:
            # Delete this chunk to make room for more in the buffer
            try:
                os.remove(local_chunk_file)
                logger.info(f"  Deleted {chunk_name} from buffer")
            except Exception as e:
                logger.warning(f"  Failed to delete {chunk_name}: {e}")

        # Progress update
        elapsed = time.time() - search_start
        chunks_done_this_run = (chunk_idx - start_idx) + 1
        avg_per_chunk = elapsed / chunks_done_this_run
        remaining = (total_chunks - chunk_idx - 1) * avg_per_chunk
        top_ranked = _rank_vote_counter(all_votes, 1, excluded_paths=excluded_paths)
        top_raw = top_ranked[0][1] if top_ranked else 0
        top_adjusted = top_ranked[0][2] if top_ranked else 0.0
        logger.info(f"  Progress: {chunk_idx + 1}/{total_chunks} chunks | "
                    f"ETA: {remaining/60:.1f}m | Top vote: {top_raw} (adj {top_adjusted:.1f})")

        # Call progress callback for live updates
        if progress_callback:
            ranked_top = _rank_vote_counter(all_votes, 100, excluded_paths=excluded_paths)
            max_adjusted = ranked_top[0][2] if ranked_top else 1.0
            top_results = [
                {
                    'path': path,
                    'votes': votes,
                    'verified_matches': votes,
                    'combined_score': adjusted,
                    'score': adjusted / max_adjusted if max_adjusted > 0 else 0.0
                }
                for path, votes, adjusted in ranked_top
            ]
            progress_callback(chunk_idx + 1, total_chunks, top_results, int(elapsed * 1000))

        if search_id and (
            last_completed_chunk == total_chunks or
            last_completed_chunk % SEARCH_CHECKPOINT_INTERVAL == 0
        ):
            save_search_checkpoint(search_id, last_completed_chunk, total_chunks, all_votes)

    total_time = time.time() - search_start
    logger.info(f"Search complete: {total_chunks} chunks in {total_time/60:.1f}m")

    if search_id:
        if not stopped and last_completed_chunk >= total_chunks:
            clear_search_checkpoint(search_id)
        else:
            save_search_checkpoint(search_id, last_completed_chunk, total_chunks, all_votes)

    ranked_final = _rank_vote_counter(all_votes, top_n, excluded_paths=excluded_paths)
    return [(path, votes) for path, votes, _ in ranked_final]


def search_disk(
    image_bytes: bytes,
    top_k: int = 50,
    k: int = 5,
    threshold: float = 0.7,
    specific_chunks: list = None,
    categories: list = None,
    progress_callback=None,
    check_stopped=None,
    search_id: int = None,
    start_chunk: int = 1,
    initial_votes: dict = None,
    excluded_paths: set = None
):
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
        search_id: Optional search session ID for checkpointing
        start_chunk: Optional global chunk position (1-based) to resume from
        initial_votes: Optional existing vote counts to seed resume

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
    results = search_chunks(
        descriptors,
        k=k,
        threshold=threshold,
        top_n=top_k,
        specific_chunks=specific_chunks,
        categories=categories,
        search_id=search_id,
        progress_callback=progress_callback,
        check_stopped=check_stopped,
        start_chunk=start_chunk,
        initial_votes=initial_votes,
        excluded_paths=excluded_paths
    )

    # Format results
    formatted = []
    max_adjusted = _adjusted_vote(results[0][0], results[0][1]) if results else 1.0

    for path, votes in results:
        adjusted = _adjusted_vote(path, votes)
        formatted.append({
            'path': path,
            'votes': votes,
            'score': adjusted / max_adjusted if max_adjusted > 0 else 0.0,
            'verified_matches': votes,
            'combined_score': adjusted
        })

    return formatted


def get_total_chunks(categories=None):
    """Get total chunk count across selected categories (for progress tracking)."""
    _, category_counts = _collect_chunks(categories)
    return sum(category_counts.values())


BATCH_CHECKPOINT_DIR = os.environ.get("DISK_BATCH_CHECKPOINT_DIR", "D:/faiss/disk_retrieval/batch_checkpoints")
BATCH_CHECKPOINT_INTERVAL = max(1, int(os.environ.get("DISK_BATCH_CHECKPOINT_INTERVAL", "5")))


def _save_batch_checkpoint(checkpoint_file: str, chunk_idx: int, per_image_votes: dict):
    """Save batch search progress to a checkpoint file."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    data = {
        "chunk_idx": chunk_idx,
        "votes": {name: dict(votes) for name, votes in per_image_votes.items()},
    }
    tmp = checkpoint_file + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, checkpoint_file)
    logger.info(f"  Checkpoint saved at chunk {chunk_idx}")


def _load_batch_checkpoint(checkpoint_file: str):
    """Load batch search checkpoint. Returns (chunk_idx, per_image_votes) or None."""
    if not os.path.exists(checkpoint_file):
        return None
    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        chunk_idx = data["chunk_idx"]
        votes = {name: Counter(v) for name, v in data["votes"].items()}
        return chunk_idx, votes
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def _search_chunks_rolling_buffer_batch(query_list: list, chunk_list: list, k: int, threshold: float, top_n: int, buffer_size: int = 5, progress_callback=None, check_stopped=None, excluded_paths: set = None, checkpoint_file: str = None):
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
    excluded_paths = excluded_paths or set()
    total_chunks = len(chunk_list)
    use_gpu = _check_gpu_search()
    start_idx = 0

    # Resume from checkpoint if available
    if checkpoint_file:
        checkpoint = _load_batch_checkpoint(checkpoint_file)
        if checkpoint:
            start_idx, saved_votes = checkpoint
            # Restore votes for images that exist in current query_list
            for name, votes in saved_votes.items():
                if name in per_image_votes:
                    per_image_votes[name] = votes
            logger.info(f"Resumed from checkpoint: chunk {start_idx}/{total_chunks} ({len(saved_votes)} images)")
        else:
            logger.info("No checkpoint found, starting from chunk 1")

    search_start = time.time()

    # Split chunks into direct-NAS (fast drives) and buffer-needed (slow drives)
    buffer_chunks = [
        (i, cf, ids) for i, (cf, ids) in enumerate(chunk_list)
        if i >= start_idx and os.path.splitdrive(cf)[0].rstrip(":").upper() not in DIRECT_NAS_DRIVES
    ]
    needs_buffer = len(buffer_chunks) > 0
    copy_queue = None

    logger.info(f"Starting batch search ({len(query_list)} images, {total_chunks} chunks, start={start_idx+1}, direct NAS drives: {DIRECT_NAS_DRIVES}, GPU: {use_gpu})")
    if needs_buffer:
        if os.path.exists(LOCAL_CHUNK_BUFFER):
            for f in os.listdir(LOCAL_CHUNK_BUFFER):
                try:
                    os.remove(os.path.join(LOCAL_CHUNK_BUFFER, f))
                except OSError:
                    pass
        os.makedirs(LOCAL_CHUNK_BUFFER, exist_ok=True)

        buffer_chunk_list = [(cf, ids) for _, cf, ids in buffer_chunks]
        copy_queue = Queue()
        logger.info(f"  Rolling buffer for {len(buffer_chunks)} slow-drive chunks (buffer size: {buffer_size})")
        copy_thread = Thread(target=_copy_chunk_worker, args=(copy_queue, buffer_chunk_list, 0, len(buffer_chunk_list), buffer_size))
        copy_thread.daemon = True
        copy_thread.start()
        buffer_copy_idx = 0

    for chunk_idx in range(start_idx, total_chunks):
        nas_chunk_file, chunk_ids_dir = chunk_list[chunk_idx]
        chunk_name = os.path.basename(nas_chunk_file)
        chunk_drive = os.path.splitdrive(nas_chunk_file)[0].rstrip(":").upper()
        use_direct = chunk_drive in DIRECT_NAS_DRIVES

        if use_direct:
            logger.info(f"[{chunk_idx + 1}/{total_chunks}] {chunk_name} (direct)...")
            load_start = time.time()
            index = faiss.read_index(nas_chunk_file, faiss.IO_FLAG_MMAP)
        else:
            local_chunk_file = os.path.join(LOCAL_CHUNK_BUFFER, chunk_name)

            logger.info(f"[{chunk_idx + 1}/{total_chunks}] Waiting for {chunk_name} (buffer)...")
            ready = False
            while not ready:
                try:
                    ready_idx, _ = copy_queue.get(timeout=1)
                    if ready_idx == buffer_copy_idx:
                        ready = True
                        buffer_copy_idx += 1
                    else:
                        copy_queue.put((ready_idx, _))
                except Empty:
                    pass

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
                matched = _accumulate_votes_vectorized(
                    votes, distances, indices, threshold, paths_or_ids, id_to_path
                )
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
                _accumulate_votes_vectorized(
                    votes, distances, indices, threshold, paths_or_ids, id_to_path
                )
                logger.info(f"    {name}: {time.time() - img_start:.1f}s ({len(descriptors)} keypoints)")

        search_time = time.time() - search_start_chunk
        logger.info(f"  Searched {len(query_list)} images in {search_time:.1f}s ({'GPU' if use_gpu else 'CPU'})")

        # Free memory
        del index, paths_or_ids

        if not use_direct:
            # Delete chunk to make room for more in the buffer
            try:
                os.remove(local_chunk_file)
            except Exception as e:
                logger.warning(f"  Failed to delete {chunk_name}: {e}")

        # Save checkpoint periodically
        if checkpoint_file and (chunk_idx + 1) % BATCH_CHECKPOINT_INTERVAL == 0:
            _save_batch_checkpoint(checkpoint_file, chunk_idx + 1, per_image_votes)

        # Progress
        chunks_done = chunk_idx - start_idx + 1
        elapsed = time.time() - search_start
        avg_per_chunk = elapsed / chunks_done
        remaining = (total_chunks - chunk_idx - 1) * avg_per_chunk
        logger.info(f"  Progress: {chunk_idx + 1}/{total_chunks} chunks | ETA: {remaining/60:.1f}m")

        if progress_callback:
            per_image_results = {}
            for name, votes in per_image_votes.items():
                if votes:
                    ranked_top = _rank_vote_counter(votes, 100, excluded_paths=excluded_paths)
                    max_adjusted = ranked_top[0][2] if ranked_top else 1.0
                    per_image_results[name] = [
                        {
                            'path': p,
                            'votes': v,
                            'verified_matches': v,
                            'combined_score': adjusted,
                            'score': adjusted / max_adjusted if max_adjusted > 0 else 0.0
                        }
                        for p, v, adjusted in ranked_top
                    ]
                else:
                    per_image_results[name] = []
            progress_callback(chunk_idx + 1, total_chunks, per_image_results, int(elapsed * 1000))

    total_time = time.time() - search_start
    logger.info(f"Batch search complete: {total_chunks} chunks, {len(query_list)} images in {total_time/60:.1f}m")

    return per_image_votes


def search_disk_batch(image_list: list, top_k: int = 50, k: int = 5, threshold: float = 0.7, categories: list = None, progress_callback=None, check_stopped=None, excluded_paths: set = None, checkpoint_file: str = None):
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
    excluded_paths = {_normalize_result_path(p) for p in (excluded_paths or set()) if p}

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
        check_stopped=check_stopped,
        excluded_paths=excluded_paths,
        checkpoint_file=checkpoint_file
    )

    # Format results
    results = {}
    for name, votes in per_image_votes.items():
        ranked_top = _rank_vote_counter(votes, top_k, excluded_paths=excluded_paths)
        max_adjusted = ranked_top[0][2] if ranked_top else 1.0
        results[name] = [
            {
                'path': p,
                'votes': v,
                'score': adjusted / max_adjusted if max_adjusted > 0 else 0.0,
                'verified_matches': v,
                'combined_score': adjusted
            }
            for p, v, adjusted in ranked_top
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
