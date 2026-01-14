"""
Deception Lens Searcher
Performs similarity search using DINOv2 embeddings, InsightFace, and optional geometric verification.
"""
import torch
from PIL import Image
from opensearchpy import OpenSearch
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import cv2
import os
import time

# Try importing Kornia for geometric verification (optional)
try:
    import kornia.feature as KF
    import kornia as K
    KORNIA_AVAILABLE = True
except ImportError:
    KF = None
    K = None
    KORNIA_AVAILABLE = False
    print("Kornia not installed. Geometric verification will be disabled.")

# Try importing DISK feature cache - SQL Server (optional)
try:
    from disk_features import DiskFeatureStore, DiskFeatureData
    DISK_SQL_AVAILABLE = True
except ImportError:
    DiskFeatureStore = None
    DiskFeatureData = None
    DISK_SQL_AVAILABLE = False

# Try importing DISK feature cache - File-based (optional, preferred for NAS)
try:
    from disk_features_file import DiskFeatureFileStore, DiskFeatureData as DiskFeatureDataFile
    DISK_FILE_AVAILABLE = True
except ImportError:
    DiskFeatureFileStore = None
    DiskFeatureDataFile = None
    DISK_FILE_AVAILABLE = False

DISK_CACHE_AVAILABLE = DISK_SQL_AVAILABLE or DISK_FILE_AVAILABLE
if not DISK_CACHE_AVAILABLE:
    print("DISK cache not available. Geometric verification will be slower.")

# Try importing InsightFace for face search (optional)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not installed. Face search will be disabled.")


def normalize_path(path: str) -> str:
    """
    Normalize file paths to handle encoding mismatches.

    Fixes common issues like:
    - Straight apostrophe (') vs curly apostrophe (')
    - Other Unicode normalization issues

    If the original path doesn't exist, tries the normalized version.
    """
    if os.path.exists(path):
        return path

    # Try replacing straight apostrophe with curly apostrophe
    # ' (U+0027) -> ' (U+2019)
    normalized = path.replace("'", "'")
    if os.path.exists(normalized):
        return normalized

    # Try the reverse: curly to straight
    normalized = path.replace("'", "'")
    if os.path.exists(normalized):
        return normalized

    # Return original if neither works
    return path


class DinoSearcher:
    """
    Provides geometric verification using DISK + LightGlue.
    ChromaDB removed - use OpenSearchSearcher for actual search.
    """

    def __init__(self, db_path: str = None):
        # db_path kept for backwards compatibility but ignored
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Searcher using device: {self.device}")

        # Initialize DINOv2
        print("Loading DINOv2 model for search...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.eval()
        print("DINOv2 loaded.")

        # Initialize LightGlue for geometric verification (optional)
        self.extractor = None
        self.matcher = None
        if KORNIA_AVAILABLE:
            print("Loading DISK + LightGlue for geometric verification...")
            try:
                self.extractor = KF.DISK.from_pretrained('depth').to(self.device).eval()
                self.matcher = KF.LightGlue(features='disk').to(self.device).eval()
                print("DISK + LightGlue loaded.")
            except Exception as e:
                print(f"Failed to load DISK/LightGlue: {e}")

        # Initialize InsightFace for face search (optional, lazy-loaded)
        self.face_app = None
        self.face_app_loaded = False

        # Initialize DISK feature cache (file-based or SQL Server)
        self.disk_cache = None
        self.disk_file_cache = None

        # Try file-based cache first (preferred for NAS storage)
        if DISK_FILE_AVAILABLE:
            try:
                # Production: T:\disk-features\books (NAS)
                self.disk_file_cache = DiskFeatureFileStore(
                    category="books",
                    features_root=r"T:\disk-features",
                    source_image_root=r"T:\archiverelated\books"
                )
                print("DISK feature cache connected (file-based, NAS).")
            except Exception as e:
                print(f"File-based DISK cache unavailable: {e}")
                self.disk_file_cache = None

        # Try SQL Server cache as fallback
        if DISK_SQL_AVAILABLE and self.disk_file_cache is None:
            try:
                self.disk_cache = DiskFeatureStore()
                print("DISK feature cache connected (SQL Server).")
            except Exception as e:
                print(f"SQL DISK cache unavailable: {e}")
                self.disk_cache = None

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Generate DINOv2 embedding for a query image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            return embedding
        except Exception as e:
            print(f"Error getting embedding for {image_path}: {e}")
            return None

    def get_embedding_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Generate DINOv2 embedding from image bytes."""
        try:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            return embedding
        except Exception as e:
            print(f"Error getting embedding from bytes: {e}")
            return None

    # NOTE: search() and search_by_bytes() removed - use OpenSearchSearcher instead
    # This class is now only used for geometric verification via _verify_matches()

    def _load_face_app(self):
        """Lazy-load InsightFace to avoid memory conflicts with other models."""
        if self.face_app_loaded:
            return self.face_app is not None

        if not INSIGHTFACE_AVAILABLE:
            print("InsightFace not available for face search")
            self.face_app_loaded = True
            return False

        try:
            print("Loading InsightFace for face search...")
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace loaded.")
            self.face_app_loaded = True
            return True
        except Exception as e:
            print(f"Failed to load InsightFace: {e}")
            self.face_app_loaded = True
            return False

    def get_face_embedding(self, image_path: str) -> list:
        """Extract face embeddings from an image using InsightFace."""
        if not self._load_face_app():
            return []

        try:
            normalized = normalize_path(image_path)
            # Use imdecode to handle non-ASCII paths on Windows
            with open(normalized, 'rb') as f:
                data = f.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return []

            faces = self.face_app.get(img)
            embeddings = [face.embedding for face in faces]
            return embeddings
        except Exception as e:
            print(f"Error extracting faces from {image_path}: {e}")
            return []

    def get_face_embedding_from_bytes(self, image_bytes: bytes) -> list:
        """Extract face embeddings from image bytes using InsightFace."""
        if not self._load_face_app():
            return []

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return []

            faces = self.face_app.get(img)
            embeddings = [face.embedding for face in faces]
            return embeddings
        except Exception as e:
            print(f"Error extracting faces from bytes: {e}")
            return []

    # NOTE: search_faces() and search_faces_by_bytes() removed - use OpenSearchSearcher instead

    def _load_torch_image(self, path: str):
        """Load and prepare image for DISK/LightGlue."""
        normalized = normalize_path(path)

        # Use imdecode instead of imread to handle non-ASCII paths on Windows
        # cv2.imread fails with special characters like apostrophes
        try:
            with open(normalized, 'rb') as f:
                data = f.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None

        if img is None:
            return None

        # Pad to dimensions divisible by 16
        h, w = img.shape[:2]
        new_h = ((h + 15) // 16) * 16
        new_w = ((w + 15) // 16) * 16
        pad_bottom = new_h - h
        pad_right = new_w - w

        if pad_bottom > 0 or pad_right > 0:
            img = cv2.copyMakeBorder(
                img, 0, pad_bottom, 0, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = K.image_to_tensor(img, False).float() / 255.0
        return img.to(self.device)

    def _verify_matches(self, query_path: str, matches: list) -> list:
        """
        Perform geometric verification using DISK + LightGlue.
        Uses SQL-cached features when available for ~10x speedup.
        """
        if not self.extractor or not self.matcher:
            return matches

        query_img = self._load_torch_image(query_path)
        if query_img is None:
            return matches

        total = len(matches)
        print(f"Verifying {total} candidates...")
        start_time = time.time()

        # Extract query features
        with torch.no_grad():
            feats0_obj = self.extractor(query_img)[0]
            feats0 = {
                "keypoints": feats0_obj.keypoints.unsqueeze(0),
                "descriptors": feats0_obj.descriptors.unsqueeze(0),
                "image_size": torch.tensor(query_img.shape[-2:][::-1]).view(1, 2).to(self.device)
            }

        # Bulk load cached DISK features (file-based or SQL)
        cached_features = {}
        cache_hits = 0
        cache_misses = 0
        match_paths = [normalize_path(m['path']) for m in matches if os.path.exists(normalize_path(m.get('path', '')))]

        if self.disk_file_cache:
            # Try file-based cache first (NAS)
            load_start = time.time()
            cached_features = self.disk_file_cache.load_bulk(match_paths)
            load_time = time.time() - load_start
            cache_hits = len(cached_features)
            cache_misses = len(match_paths) - cache_hits
            print(f"  File cache: {cache_hits} hits, {cache_misses} misses (loaded in {load_time:.1f}s)")
        elif self.disk_cache:
            # Fall back to SQL cache
            load_start = time.time()
            cached_features = self.disk_cache.load_bulk(match_paths)
            load_time = time.time() - load_start
            cache_hits = len(cached_features)
            cache_misses = len(match_paths) - cache_hits
            print(f"  SQL cache: {cache_hits} hits, {cache_misses} misses (loaded in {load_time:.1f}s)")

        # Verify each candidate
        verify_start = time.time()
        for i, match in enumerate(matches):
            if i % 500 == 0 and i > 0:
                elapsed = time.time() - verify_start
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  Verified {i}/{total} ({rate:.1f}/s, ETA: {eta:.0f}s)...")

            try:
                match_path = normalize_path(match['path'])
                if not os.path.exists(match_path):
                    match['verified_matches'] = 0
                    continue

                # Try to use cached features
                if match_path in cached_features:
                    cached = cached_features[match_path]
                    feats1 = {
                        "keypoints": torch.from_numpy(cached.keypoints).unsqueeze(0).to(self.device),
                        "descriptors": torch.from_numpy(cached.descriptors).T.unsqueeze(0).to(self.device),
                        "image_size": torch.tensor([cached.padded_size[1], cached.padded_size[0]]).view(1, 2).to(self.device)
                    }
                else:
                    # Fall back to on-the-fly extraction
                    match_img = self._load_torch_image(match_path)
                    if match_img is None:
                        match['verified_matches'] = 0
                        continue

                    with torch.no_grad():
                        feats1_obj = self.extractor(match_img)[0]
                        feats1 = {
                            "keypoints": feats1_obj.keypoints.unsqueeze(0),
                            "descriptors": feats1_obj.descriptors.unsqueeze(0),
                            "image_size": torch.tensor(match_img.shape[-2:][::-1]).view(1, 2).to(self.device)
                        }

                # Run LightGlue matching
                with torch.no_grad():
                    matches01 = self.matcher({"image0": feats0, "image1": feats1})
                    matches_idx = matches01["matches"][0]
                    # Count only valid matches (LightGlue returns -1 for non-matches)
                    valid_matches = (matches_idx > -1).sum().item()
                    match['verified_matches'] = valid_matches

            except Exception as e:
                print(f"Verification failed for {match.get('path', 'unknown')}: {e}")
                match['verified_matches'] = 0

        total_time = time.time() - start_time
        verify_time = time.time() - verify_start
        rate = total / verify_time if verify_time > 0 else 0
        print(f"Verification complete: {total} images in {total_time:.1f}s ({rate:.1f}/s)")
        return matches

    def get_match_visualization(self, query_bytes: bytes, match_path: str) -> tuple:
        """
        Get visualization data for matched keypoints between query and result.

        Returns:
            Tuple of (match_count, bounding_box) where bounding_box is (x, y, w, h) or None
        """
        if not self.extractor or not self.matcher or not KORNIA_AVAILABLE:
            return (0, None)

        try:
            from io import BytesIO

            # Load query image from bytes
            query_pil = Image.open(BytesIO(query_bytes)).convert("RGB")
            query_cv = cv2.cvtColor(np.array(query_pil), cv2.COLOR_RGB2BGR)

            # Pad query image
            h, w = query_cv.shape[:2]
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            if new_h - h > 0 or new_w - w > 0:
                query_cv = cv2.copyMakeBorder(
                    query_cv, 0, new_h - h, 0, new_w - w,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            query_cv = cv2.cvtColor(query_cv, cv2.COLOR_BGR2RGB)
            query_tensor = K.image_to_tensor(query_cv, False).float() / 255.0
            query_tensor = query_tensor.to(self.device)

            # Load match image
            match_img = self._load_torch_image(match_path)
            if match_img is None:
                return (0, None)

            with torch.no_grad():
                # Extract features
                feats0_obj = self.extractor(query_tensor)[0]
                feats0 = {
                    "keypoints": feats0_obj.keypoints.unsqueeze(0),
                    "descriptors": feats0_obj.descriptors.unsqueeze(0),
                    "image_size": torch.tensor(query_tensor.shape[-2:][::-1]).view(1, 2).to(self.device)
                }

                feats1_obj = self.extractor(match_img)[0]
                feats1 = {
                    "keypoints": feats1_obj.keypoints.unsqueeze(0),
                    "descriptors": feats1_obj.descriptors.unsqueeze(0),
                    "image_size": torch.tensor(match_img.shape[-2:][::-1]).view(1, 2).to(self.device)
                }

                matches01 = self.matcher({"image0": feats0, "image1": feats1})
                matches_idx = matches01["matches"][0]

                # Get valid matches
                valid_mask = matches_idx > -1
                valid_count = valid_mask.sum().item()

                if valid_count < 4:
                    return (valid_count, None)

                # Get matched keypoint coordinates in the match image
                kpts0 = feats0["keypoints"][0]
                kpts1 = feats1["keypoints"][0]

                valid_indices = torch.where(valid_mask)[0]
                matched_kpts1_indices = matches_idx[valid_mask]
                matched_kpts1 = kpts1[matched_kpts1_indices].cpu().numpy()

                # Calculate bounding box around matched keypoints
                x_min = float(matched_kpts1[:, 0].min())
                y_min = float(matched_kpts1[:, 1].min())
                x_max = float(matched_kpts1[:, 0].max())
                y_max = float(matched_kpts1[:, 1].max())

                # Add padding (10%)
                w = x_max - x_min
                h = y_max - y_min
                padding = max(w, h) * 0.1
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = x_max + padding
                y_max = y_max + padding

                bbox = {
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min
                }

                return (valid_count, bbox)

        except Exception as e:
            print(f"Visualization failed: {e}")
            return (0, None)

    def generate_visualization_image(self, query_bytes: bytes, match_path: str) -> bytes:
        """
        Generate a visualization image with bounding box drawn on the match.

        Returns:
            PNG image bytes with the matched region highlighted
        """
        match_count, bbox = self.get_match_visualization(query_bytes, match_path)

        # Load the match image (use imdecode for non-ASCII paths on Windows)
        normalized = normalize_path(match_path)
        try:
            with open(normalized, 'rb') as f:
                data = f.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            return None

        if bbox and match_count >= 4:
            # Draw bounding box
            x1 = int(bbox["x"])
            y1 = int(bbox["y"])
            x2 = int(bbox["x"] + bbox["width"])
            y2 = int(bbox["y"] + bbox["height"])

            # Draw semi-transparent overlay outside the box
            overlay = img.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # Darken everything outside the box
            overlay[mask == 0] = (overlay[mask == 0] * 0.4).astype(np.uint8)

            # Draw bright box border
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Add match count text
            cv2.putText(
                overlay,
                f"{match_count} matches",
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            img = overlay

        # Encode as PNG
        _, buffer = cv2.imencode('.png', img)
        return buffer.tobytes()

    # NOTE: get_collection_stats() removed - use OpenSearchSearcher.get_counts() instead
