"""
CLIP re-ranking with DISK keypoints and template matching.

Pipeline:
1. CLIP semantic search (fast) -> get 20K candidates
2. DISK keypoint filtering (parallel) -> filter to top 1K
3. Template matching (parallel) -> precise re-ranking

If DISK descriptors are unavailable for a candidate, the code falls back to
ORB keypoints for that candidate only.
"""

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import faiss
import numpy as np

from collections_config import COLLECTIONS

DISK_MATCH_THRESHOLD = float(os.environ.get("CLIP_RERANK_DISK_THRESHOLD", "0.7"))
DISK_STRONG_MATCHES = int(os.environ.get("CLIP_RERANK_DISK_STRONG_KP", "30"))
DISK_MODERATE_MATCHES = int(os.environ.get("CLIP_RERANK_DISK_MODERATE_KP", "15"))
CLIP_RERANK_DISK_ROOTS = [
    root.strip().replace("\\", "/").rstrip("/")
    for root in os.environ.get("CLIP_RERANK_DISK_ROOTS", "S:/disk-features,T:/disk-features").split(",")
    if root.strip()
]
CLIP_RERANK_NPZ_CACHE_SIZE = max(1_000, int(os.environ.get("CLIP_RERANK_NPZ_CACHE_SIZE", "100000")))

_NPZ_PATH_CACHE: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
_NPZ_PATH_CACHE_LOCK = Lock()


def _cache_get_npz_path(cache_key: Tuple[str, str]) -> Optional[str]:
    with _NPZ_PATH_CACHE_LOCK:
        cached = _NPZ_PATH_CACHE.get(cache_key)
        if cached is None:
            return None
        _NPZ_PATH_CACHE.move_to_end(cache_key)
        return cached


def _cache_set_npz_path(cache_key: Tuple[str, str], npz_path: str) -> None:
    with _NPZ_PATH_CACHE_LOCK:
        _NPZ_PATH_CACHE[cache_key] = npz_path
        _NPZ_PATH_CACHE.move_to_end(cache_key)
        while len(_NPZ_PATH_CACHE) > CLIP_RERANK_NPZ_CACHE_SIZE:
            _NPZ_PATH_CACHE.popitem(last=False)


def _cache_delete_npz_path(cache_key: Tuple[str, str]) -> None:
    with _NPZ_PATH_CACHE_LOCK:
        _NPZ_PATH_CACHE.pop(cache_key, None)


def normalize_path(path: str) -> str:
    """
    Normalize file paths to handle encoding mismatches.
    Fixes common issues like straight apostrophe (') vs curly apostrophe (').
    """
    if os.path.exists(path):
        return path

    normalized = path.replace("'", "\u2019")
    if os.path.exists(normalized):
        return normalized

    normalized = path.replace("\u2019", "'")
    if os.path.exists(normalized):
        return normalized

    return path


def _path_norm(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def get_disk_features_dirs(collection: str) -> List[str]:
    """Return candidate DISK feature directories for this collection."""
    cfg = COLLECTIONS.get(collection) or {}
    disk_root = cfg.get("disk_features")
    if not disk_root:
        return []

    candidates: List[str] = []
    root_path = Path(disk_root)
    candidates.append(os.path.normpath(str(root_path)))

    # Add mirrored drive fallback (T: <-> S:) for NAS moves.
    norm_root = _path_norm(str(root_path))
    if norm_root.lower().startswith("t:/"):
        candidates.append(os.path.normpath("S:/" + norm_root[3:]))
    elif norm_root.lower().startswith("s:/"):
        candidates.append(os.path.normpath("T:/" + norm_root[3:]))

    # Add root-level overrides, preserving collection subfolder name.
    collection_leaf = root_path.name
    for root in CLIP_RERANK_DISK_ROOTS:
        candidates.append(os.path.normpath(str(Path(root) / collection_leaf)))

    deduped: List[str] = []
    seen = set()
    for cand in candidates:
        key = _path_norm(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def _candidate_source_prefixes(collection: str) -> List[str]:
    """
    Build source path prefixes used by CLIP result paths for this collection.
    Includes known alternate prefixes for local/temp paths.
    """
    prefixes: List[str] = []
    cfg = COLLECTIONS.get(collection) or {}
    src = cfg.get("source_path")
    if src:
        prefixes.append(_path_norm(src))

    if collection == "books":
        prefixes.extend([
            "T:/archiverelated/books/pdf-images",
            "T:/archiverelated/books",
            "D:/books/pdf-images",
            "D:/books",
        ])
    elif collection == "print_ads":
        prefixes.extend([
            "T:/archiverelated/print ads",
            "D:/archiverelated/print ads",
            "D:/print ads",
        ])
    elif collection == "board_games":
        prefixes.extend([
            "T:/archiverelated/board games",
            "D:/archiverelated/board games",
        ])

    deduped: List[str] = []
    seen = set()
    for pref in prefixes:
        key = pref.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pref)
    return deduped


def _extract_relative_candidates(image_path: str, collection: str) -> List[str]:
    """Extract candidate relative paths that may map to disk-features/*.npz."""
    path_norm = _path_norm(image_path)
    lower_path = path_norm.lower()

    candidates: List[str] = []
    for prefix in _candidate_source_prefixes(collection):
        prefix_with_sep = prefix.lower() + "/"
        if lower_path.startswith(prefix_with_sep):
            rel = path_norm[len(prefix) + 1 :]
            if rel:
                candidates.append(rel)

    # Fallback marker-based extraction if prefix matching failed.
    if not candidates:
        markers = {
            "books": "/pdf-images/",
            "print_ads": "/print ads/",
            "board_games": "/board games/",
        }
        marker = markers.get(collection)
        if marker and marker in lower_path:
            idx = lower_path.index(marker) + len(marker)
            rel = path_norm[idx:].lstrip("/")
            if rel:
                candidates.append(rel)

    if not candidates:
        candidates.append(os.path.basename(path_norm))

    normalized: List[str] = []
    for rel in candidates:
        parts = [p for p in rel.split("/") if p]
        if not parts:
            continue

        # Books paths often include "pdf-images/" in source; disk-features do not.
        if collection == "books" and parts[0].lower() == "pdf-images":
            parts = parts[1:]
            if not parts:
                continue

        # Print ads CLIP paths commonly include market folder like "ebay/SubfolderX/...".
        # disk-features/print_ads is rooted at "SubfolderX/...".
        if (
            collection == "print_ads"
            and len(parts) >= 2
            and parts[0].lower() in ("ebay", "etsy")
            and parts[1].lower().startswith("subfolder")
        ):
            parts = parts[1:]

        normalized.append("/".join(parts))

    deduped: List[str] = []
    seen = set()
    for rel in normalized:
        key = rel.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped


def image_path_to_npz(image_path: str, collection: str) -> Optional[str]:
    """
    Map indexed image path to precomputed DISK feature .npz path.

    Returns existing .npz file path if found, else None.
    """
    cache_key = (collection.lower(), _path_norm(image_path).lower())
    cached = _cache_get_npz_path(cache_key)
    if cached:
        if os.path.exists(cached):
            return cached
        _cache_delete_npz_path(cache_key)

    disk_roots = get_disk_features_dirs(collection)
    if not disk_roots:
        return None

    rel_candidates = _extract_relative_candidates(image_path, collection)
    for disk_root in disk_roots:
        for rel in rel_candidates:
            rel_npz = os.path.splitext(rel)[0] + ".npz"
            candidate = os.path.normpath(os.path.join(disk_root, rel_npz.replace("/", os.sep)))
            if os.path.exists(candidate):
                _cache_set_npz_path(cache_key, candidate)
                return candidate

    return None


def load_disk_descriptors(npz_path: str) -> Optional[np.ndarray]:
    """
    Load normalized float32 DISK descriptors (N, 128) from a .npz file.
    Returns None if load fails.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
        try:
            if "descriptors" in data.files:
                desc = data["descriptors"]
            else:
                desc = None
                for key in data.files:
                    arr = data[key]
                    if arr.ndim == 2 and arr.shape[1] == 128:
                        desc = arr
                        break
                if desc is None:
                    return None
        finally:
            data.close()

        if desc is None or desc.size == 0:
            return np.empty((0, 128), dtype=np.float32)
        if desc.ndim != 2 or desc.shape[1] != 128:
            return None

        desc = np.ascontiguousarray(desc.astype(np.float32, copy=False))
        norms = np.linalg.norm(desc, axis=1, keepdims=True)
        desc = desc / (norms + 1e-8)
        return desc
    except Exception:
        return None


def disk_descriptor_match(
    query_desc: np.ndarray,
    candidate_desc: np.ndarray,
    threshold: float = DISK_MATCH_THRESHOLD
) -> int:
    """
    Count mutual nearest-neighbor descriptor matches above threshold.

    Uses cosine similarity via inner product on normalized vectors.
    """
    if query_desc is None or candidate_desc is None:
        return 0
    if len(query_desc) == 0 or len(candidate_desc) == 0:
        return 0

    query_desc = np.ascontiguousarray(query_desc, dtype=np.float32)
    candidate_desc = np.ascontiguousarray(candidate_desc, dtype=np.float32)

    # Forward NN: query -> candidate.
    idx_c = faiss.IndexFlatIP(candidate_desc.shape[1])
    idx_c.add(candidate_desc)
    fwd_scores, fwd_idx = idx_c.search(query_desc, 1)

    # Backward NN: candidate -> query.
    idx_q = faiss.IndexFlatIP(query_desc.shape[1])
    idx_q.add(query_desc)
    _, bwd_idx = idx_q.search(candidate_desc, 1)

    fwd = fwd_idx[:, 0]
    scores = fwd_scores[:, 0]
    valid = fwd >= 0
    if not np.any(valid):
        return 0

    q_ids = np.arange(query_desc.shape[0], dtype=np.int64)
    mutual = np.zeros_like(valid)
    v_fwd = fwd[valid].astype(np.int64, copy=False)
    mutual[valid] = (bwd_idx[v_fwd, 0] == q_ids[valid]) & (scores[valid] >= threshold)
    return int(np.count_nonzero(mutual))


# ============================================================================
# TIER 2 FALLBACK: ORB KEYPOINT MATCHING
# ============================================================================

def orb_keypoint_match(page_gray: np.ndarray, query_gray: np.ndarray) -> int:
    """
    ORB keypoint matching fallback for candidates missing DISK descriptors.
    """
    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(query_gray, None)
    kp2, des2 = orb.detectAndCompute(page_gray, None)

    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 50]
    return len(good_matches)


# ============================================================================
# TIER 3: MULTI-SCALE TEMPLATE MATCHING
# ============================================================================

def multi_scale_template_match(
    page_gray: np.ndarray,
    query_gray: np.ndarray,
    scales: List[float] = None
) -> Tuple[float, Optional[Tuple[int, int]], float]:
    """
    Multi-scale template matching with histogram equalization for contrast invariance.
    """
    if scales is None:
        scales = [0.25, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    best_score = 0.0
    best_location = None
    best_scale = 1.0

    page_norm = cv2.equalizeHist(page_gray)
    query_norm = cv2.equalizeHist(query_gray)

    for scale in scales:
        new_h = int(query_norm.shape[0] * scale)
        new_w = int(query_norm.shape[1] * scale)

        if new_h > page_norm.shape[0] or new_w > page_norm.shape[1]:
            continue
        if new_h < 20 or new_w < 20:
            continue

        scaled_query = cv2.resize(query_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)

        try:
            for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
                match_result = cv2.matchTemplate(page_norm, scaled_query, method)
                _, max_val, _, max_loc = cv2.minMaxLoc(match_result)

                if max_val > best_score:
                    best_score = max_val
                    best_location = max_loc
                    best_scale = scale
        except Exception:
            continue

    return best_score, best_location, best_scale


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_blank_page(img_gray: np.ndarray, threshold: int = 30) -> bool:
    """Check if page is mostly blank (low variance)."""
    return np.std(img_gray) < threshold


def load_image_gray(path: str) -> Optional[np.ndarray]:
    """Load image as grayscale, return None if failed."""
    try:
        normalized = normalize_path(path)
        with open(normalized, "rb") as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


# ============================================================================
# PARALLEL PROCESSING FUNCTIONS
# ============================================================================

def quick_keypoint_check(args: Tuple[Dict, np.ndarray]) -> Dict:
    """
    Quick ORB keypoint check for filtering.
    Used only as a full fallback when query DISK extraction fails.
    """
    result, query_gray = args

    try:
        page_gray = load_image_gray(result["path"])
        if page_gray is None:
            result["keypoint_matches"] = 0
            result["is_blank"] = True
            result["rerank_method"] = "orb"
            return result

        if is_blank_page(page_gray):
            result["keypoint_matches"] = 0
            result["is_blank"] = True
            result["rerank_method"] = "orb"
            return result

        result["is_blank"] = False
        result["keypoint_matches"] = orb_keypoint_match(page_gray, query_gray)
        result["rerank_method"] = "orb"
    except Exception:
        result["keypoint_matches"] = 0
        result["is_blank"] = True
        result["rerank_method"] = "orb"

    return result


def quick_disk_check(args: Tuple[Dict, np.ndarray, np.ndarray, str]) -> Dict:
    """
    Quick DISK descriptor check for filtering.

    Falls back to ORB for candidates where .npz descriptors are missing.
    """
    result, query_descriptors, query_gray, collection = args

    try:
        image_path = result["path"]
        npz_path = image_path_to_npz(image_path, collection)

        if npz_path:
            candidate_desc = load_disk_descriptors(npz_path)
            if candidate_desc is not None and len(candidate_desc) > 0:
                result["is_blank"] = False
                result["keypoint_matches"] = disk_descriptor_match(
                    query_descriptors,
                    candidate_desc,
                    threshold=DISK_MATCH_THRESHOLD
                )
                result["rerank_method"] = "disk"
                return result

        # Fallback to ORB when DISK data is missing/unreadable for this candidate.
        page_gray = load_image_gray(image_path)
        if page_gray is None or is_blank_page(page_gray):
            result["keypoint_matches"] = 0
            result["is_blank"] = True
            result["rerank_method"] = "orb_fallback"
            return result

        result["is_blank"] = False
        result["keypoint_matches"] = orb_keypoint_match(page_gray, query_gray)
        result["rerank_method"] = "orb_fallback"
    except Exception:
        result["keypoint_matches"] = 0
        result["is_blank"] = True
        result["rerank_method"] = "disk_error"

    return result


def process_template_match(args: Tuple[Dict, np.ndarray]) -> Dict:
    """
    Full template matching for precise ranking.
    Run on top candidates by keypoint count.
    """
    result, query_gray = args

    try:
        page_gray = load_image_gray(result["path"])
        if page_gray is None:
            result["template_score"] = 0.0
            result["match_location"] = None
            result["best_scale"] = 1.0
            return result

        score, loc, scale = multi_scale_template_match(page_gray, query_gray)
        result["template_score"] = score
        result["match_location"] = loc
        result["best_scale"] = scale
    except Exception:
        result["template_score"] = 0.0
        result["match_location"] = None
        result["best_scale"] = 1.0

    return result


# ============================================================================
# COMBINED SCORING
# ============================================================================

def compute_combined_score(result: Dict, max_clip: float, max_keypoints: int, max_template: float) -> float:
    """
    Hierarchical scoring that heavily weights template matches.

    Scoring tiers:
    - Template >= 0.9: Dominant signal (base 1000)
    - Template >= 0.85: Strong match (base 500)
    - Template >= 0.75: Good match (base 200)
    - Keypoints >= 30: Strong geometric (base 150)
    - Keypoints >= 15: Moderate geometric (base 100)
    - Otherwise: Balanced fallback
    """
    clip_norm = result.get("score", 0.0) / max_clip if max_clip > 0 else 0.0
    kp = result.get("keypoint_matches", 0)
    template_score = result.get("template_score", 0.0)

    if template_score >= 0.9:
        return 1000 + template_score * 100 + kp
    if template_score >= 0.85:
        return 500 + template_score * 100 + kp
    if template_score >= 0.75:
        return 200 + template_score * 100 + kp
    if kp >= DISK_STRONG_MATCHES:
        return 150 + kp + template_score * 50
    if kp >= DISK_MODERATE_MATCHES:
        return 100 + kp + template_score * 50
    return clip_norm * 30 + template_score * 40 + kp * 2


def _extract_query_disk_descriptors(query_image_path: str) -> np.ndarray:
    """Extract DISK descriptors for query image from bytes."""
    try:
        from disk_searcher import extract_disk_features

        with open(query_image_path, "rb") as f:
            image_bytes = f.read()
        return extract_disk_features(image_bytes)
    except Exception:
        return np.empty((0, 128), dtype=np.float32)


# ============================================================================
# MAIN RE-RANKING FUNCTION
# ============================================================================

def rerank_with_orb_and_template(
    query_image_path: str,
    results: List[Dict],
    collection: str = "books",
    orb_workers: int = 12,
    template_workers: int = 8,
    top_for_template: int = 1000,
    verbose: bool = True
) -> List[Dict]:
    """
    Full re-ranking pipeline:
    1. DISK keypoint filtering on all candidates (parallel)
    2. ORB fallback for candidates missing DISK descriptors
    3. Template matching on top candidates by keypoints (parallel)
    4. Combined scoring with hierarchical weighting
    """
    if not results:
        return []

    query_gray = load_image_gray(query_image_path)
    if query_gray is None:
        print(f"Error: Could not load query image: {query_image_path}")
        return results

    query_descriptors = _extract_query_disk_descriptors(query_image_path)
    use_disk = len(query_descriptors) > 0

    if verbose:
        print(f"Re-ranking {len(results)} candidates...")
        if use_disk:
            print(f"  Query DISK descriptors: {len(query_descriptors):,}")
            print(f"  Running DISK keypoint matching ({orb_workers} workers)...")
        else:
            print("  Query DISK extraction failed, falling back to ORB filtering.")
            print(f"  Running ORB keypoint matching ({orb_workers} workers)...")

    with ThreadPoolExecutor(max_workers=orb_workers) as executor:
        if use_disk:
            args_list = [(r, query_descriptors, query_gray, collection) for r in results]
            results = list(executor.map(quick_disk_check, args_list))
        else:
            args_list = [(r, query_gray) for r in results]
            results = list(executor.map(quick_keypoint_check, args_list))

    results = [r for r in results if not r.get("is_blank", False)]
    results.sort(key=lambda x: x.get("keypoint_matches", 0), reverse=True)

    if verbose:
        non_zero = sum(1 for r in results if r.get("keypoint_matches", 0) > 0)
        disk_count = sum(1 for r in results if r.get("rerank_method") == "disk")
        print(f"    {non_zero} candidates with keypoint matches, {len(results)} non-blank")
        if use_disk:
            print(f"    {disk_count} matched with DISK, {len(results) - disk_count} ORB fallbacks")

    num_for_template = min(top_for_template, len(results))

    if verbose:
        print(f"  Running template matching on top {num_for_template} ({template_workers} workers)...")

    with ThreadPoolExecutor(max_workers=template_workers) as executor:
        args_list = [(r, query_gray) for r in results[:num_for_template]]
        top_results = list(executor.map(process_template_match, args_list))

    remaining = results[num_for_template:]
    for r in remaining:
        r["template_score"] = 0.0
        r["match_location"] = None
        r["best_scale"] = 1.0

    results = top_results + remaining

    if verbose:
        print("  Computing combined scores...")

    max_clip = max((r.get("score", 0.0) for r in results), default=1.0) or 1.0
    max_keypoints = max((r.get("keypoint_matches", 0) for r in results), default=1) or 1
    max_template = max((r.get("template_score", 0.0) for r in results), default=1.0) or 1.0

    for r in results:
        r["combined_score"] = compute_combined_score(r, max_clip, max_keypoints, max_template)

    results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)

    if verbose:
        print("  Re-ranking complete.")
        if results:
            top = results[0]
            print(f"    Top result: {os.path.basename(top['path'])}")
            print(
                f"      CLIP: {top.get('score', 0.0):.4f}, "
                f"Keypoints: {top.get('keypoint_matches', 0)}, "
                f"Template: {top.get('template_score', 0.0):.4f}, "
                f"Combined: {top.get('combined_score', 0.0):.2f}"
            )

    return results
