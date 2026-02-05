"""
CLIP Re-ranking module with ORB keypoints and template matching.
Ported from ImageSnippetSearch 3-tier pipeline.

Pipeline:
1. CLIP semantic search (fast) -> get 20K candidates
2. ORB keypoint filtering (parallel) -> filter to top 1K
3. Template matching (parallel) -> precise re-ranking
"""

import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import os


def normalize_path(path: str) -> str:
    """
    Normalize file paths to handle encoding mismatches.
    Fixes common issues like straight apostrophe (') vs curly apostrophe (').
    """
    if os.path.exists(path):
        return path
    # Try replacing straight apostrophe with curly apostrophe
    normalized = path.replace("'", "'")
    if os.path.exists(normalized):
        return normalized
    # Try the reverse
    normalized = path.replace("'", "'")
    if os.path.exists(normalized):
        return normalized
    return path


# ============================================================================
# TIER 2: ORB KEYPOINT MATCHING
# ============================================================================

def orb_keypoint_match(page_gray: np.ndarray, query_gray: np.ndarray) -> int:
    """
    ORB keypoint matching for geometric feature validation.

    Args:
        page_gray: Grayscale page image
        query_gray: Grayscale query image

    Returns:
        Number of good keypoint matches (distance < 50)
    """
    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(query_gray, None)
    kp2, des2 = orb.detectAndCompute(page_gray, None)

    # Need sufficient descriptors
    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return 0

    # Brute-force matching with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Count good matches (distance < 50)
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

    Args:
        page_gray: Grayscale page image
        query_gray: Grayscale query image
        scales: List of scales to try (default: 0.25 to 2.0)

    Returns:
        (best_score, best_location, best_scale)
    """
    if scales is None:
        scales = [0.25, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    best_score = 0
    best_location = None
    best_scale = 1.0

    # Normalize contrast with histogram equalization
    page_norm = cv2.equalizeHist(page_gray)
    query_norm = cv2.equalizeHist(query_gray)

    for scale in scales:
        new_h = int(query_norm.shape[0] * scale)
        new_w = int(query_norm.shape[1] * scale)

        # Skip invalid sizes
        if new_h > page_norm.shape[0] or new_w > page_norm.shape[1]:
            continue
        if new_h < 20 or new_w < 20:
            continue

        scaled_query = cv2.resize(query_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)

        try:
            # Try both methods - CCORR more robust to intensity variations
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
    """Load image as grayscale, return None if failed.

    Uses imdecode instead of imread to handle non-ASCII paths on Windows.
    cv2.imread fails with special characters like smart quotes (U+2019).
    """
    try:
        normalized = normalize_path(path)
        # Use imdecode to handle non-ASCII paths
        with open(normalized, 'rb') as f:
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
    Run on all CLIP candidates in parallel.
    """
    result, query_gray = args

    try:
        page_gray = load_image_gray(result['path'])
        if page_gray is None:
            result['keypoint_matches'] = 0
            result['is_blank'] = True
            return result

        # Check for blank pages
        if is_blank_page(page_gray):
            result['keypoint_matches'] = 0
            result['is_blank'] = True
            return result

        result['is_blank'] = False
        result['keypoint_matches'] = orb_keypoint_match(page_gray, query_gray)

    except Exception as e:
        result['keypoint_matches'] = 0
        result['is_blank'] = True

    return result


def process_template_match(args: Tuple[Dict, np.ndarray]) -> Dict:
    """
    Full template matching for precise ranking.
    Run on top candidates by keypoint count.
    """
    result, query_gray = args

    try:
        page_gray = load_image_gray(result['path'])
        if page_gray is None:
            result['template_score'] = 0
            result['match_location'] = None
            result['best_scale'] = 1.0
            return result

        score, loc, scale = multi_scale_template_match(page_gray, query_gray)
        result['template_score'] = score
        result['match_location'] = loc
        result['best_scale'] = scale

    except Exception as e:
        result['template_score'] = 0
        result['match_location'] = None
        result['best_scale'] = 1.0

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
    - Keypoints >= 15: Strong geometric (base 150)
    - Keypoints >= 10: Moderate geometric (base 100)
    - Otherwise: Balanced fallback
    """
    clip_norm = result.get('score', 0) / max_clip if max_clip > 0 else 0
    kp = result.get('keypoint_matches', 0)
    template_score = result.get('template_score', 0)

    if template_score >= 0.9:
        return 1000 + template_score * 100 + kp
    elif template_score >= 0.85:
        return 500 + template_score * 100 + kp
    elif template_score >= 0.75:
        return 200 + template_score * 100 + kp
    elif kp >= 15:
        return 150 + kp + template_score * 50
    elif kp >= 10:
        return 100 + kp + template_score * 50
    else:
        return clip_norm * 30 + template_score * 40 + kp * 2


# ============================================================================
# MAIN RE-RANKING FUNCTION
# ============================================================================

def rerank_with_orb_and_template(
    query_image_path: str,
    results: List[Dict],
    orb_workers: int = 12,
    template_workers: int = 8,
    top_for_template: int = 1000,
    verbose: bool = True
) -> List[Dict]:
    """
    Full re-ranking pipeline:
    1. ORB keypoint filtering on all candidates (parallel)
    2. Template matching on top candidates by keypoints (parallel)
    3. Combined scoring with hierarchical weighting

    Args:
        query_image_path: Path to query image
        results: List of CLIP search results with 'path' and 'score' keys
        orb_workers: Number of parallel workers for ORB
        template_workers: Number of parallel workers for template matching
        top_for_template: Number of top keypoint results to run template matching on
        verbose: Print progress

    Returns:
        Re-ranked results with keypoint_matches, template_score, and combined_score
    """
    if not results:
        return []

    # Load query image
    query_gray = load_image_gray(query_image_path)
    if query_gray is None:
        print(f"Error: Could not load query image: {query_image_path}")
        return results

    if verbose:
        print(f"Re-ranking {len(results)} candidates...")

    # -------------------------------------------------------------------------
    # TIER 2: ORB Keypoint Filtering (parallel)
    # -------------------------------------------------------------------------
    if verbose:
        print(f"  Running ORB keypoint matching ({orb_workers} workers)...")

    with ThreadPoolExecutor(max_workers=orb_workers) as executor:
        args_list = [(r, query_gray) for r in results]
        results = list(executor.map(quick_keypoint_check, args_list))

    # Filter out blanks and sort by keypoint count
    results = [r for r in results if not r.get('is_blank', False)]
    results.sort(key=lambda x: x.get('keypoint_matches', 0), reverse=True)

    if verbose:
        non_zero = sum(1 for r in results if r.get('keypoint_matches', 0) > 0)
        print(f"    {non_zero} candidates with keypoint matches, {len(results)} non-blank")

    # -------------------------------------------------------------------------
    # TIER 3: Template Matching (parallel, on top candidates)
    # -------------------------------------------------------------------------
    num_for_template = min(top_for_template, len(results))

    if verbose:
        print(f"  Running template matching on top {num_for_template} ({template_workers} workers)...")

    with ThreadPoolExecutor(max_workers=template_workers) as executor:
        args_list = [(r, query_gray) for r in results[:num_for_template]]
        top_results = list(executor.map(process_template_match, args_list))

    # Combine with remaining results (no template matching)
    remaining = results[num_for_template:]
    for r in remaining:
        r['template_score'] = 0
        r['match_location'] = None
        r['best_scale'] = 1.0

    results = top_results + remaining

    # -------------------------------------------------------------------------
    # Combined Scoring
    # -------------------------------------------------------------------------
    if verbose:
        print("  Computing combined scores...")

    max_clip = max(r.get('score', 0) for r in results) or 1
    max_keypoints = max(r.get('keypoint_matches', 0) for r in results) or 1
    max_template = max(r.get('template_score', 0) for r in results) or 1

    for r in results:
        r['combined_score'] = compute_combined_score(r, max_clip, max_keypoints, max_template)

    # Sort by combined score
    results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

    if verbose:
        print("  Re-ranking complete.")
        if results:
            top = results[0]
            print(f"    Top result: {os.path.basename(top['path'])}")
            print(f"      CLIP: {top.get('score', 0):.4f}, "
                  f"Keypoints: {top.get('keypoint_matches', 0)}, "
                  f"Template: {top.get('template_score', 0):.4f}, "
                  f"Combined: {top.get('combined_score', 0):.2f}")

    return results
