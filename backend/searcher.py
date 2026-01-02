"""
Deception Lens Searcher
Performs similarity search using DINOv2 embeddings, InsightFace, and optional geometric verification.
"""
import torch
from PIL import Image
import chromadb
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import cv2
import os

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

# Try importing InsightFace for face search (optional)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not installed. Face search will be disabled.")


class DinoSearcher:
    """
    Searches indexed images using DINOv2 visual similarity.
    Optionally performs geometric verification using LightGlue.
    """

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
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

    def search(
        self,
        query_path: str,
        top_k: int = 50,
        verify: bool = False,
        collection_name: str = "images"
    ) -> list:
        """
        Search for similar images.

        Args:
            query_path: Path to query image
            top_k: Number of results to return
            verify: Whether to perform geometric verification
            collection_name: Base name of the collection

        Returns:
            List of match dictionaries with path, score, verified_matches, metadata
        """
        query_emb = self.get_embedding(query_path)
        if query_emb is None:
            return []

        return self._search_with_embedding(query_emb, top_k, verify, collection_name, query_path)

    def search_by_bytes(
        self,
        image_bytes: bytes,
        top_k: int = 50,
        verify: bool = False,
        collection_name: str = "images"
    ) -> list:
        """Search for similar images using image bytes."""
        query_emb = self.get_embedding_from_bytes(image_bytes)
        if query_emb is None:
            return []

        return self._search_with_embedding(query_emb, top_k, verify, collection_name)

    def _search_with_embedding(
        self,
        query_emb: np.ndarray,
        top_k: int,
        verify: bool,
        collection_name: str,
        query_path: str = None
    ) -> list:
        """Internal search using pre-computed embedding."""
        try:
            collection = self.client.get_collection(name=f"{collection_name}_visual")
        except Exception as e:
            print(f"Collection {collection_name}_visual not found: {e}")
            return []

        # When verifying, check ALL candidates - accuracy over speed
        # The correct match could be ranked last by embedding but first by keypoints
        if verify:
            collection_size = collection.count()
            fetch_k = collection_size  # Check everything
        else:
            fetch_k = top_k

        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=fetch_k,
            include=["metadatas", "distances"]
        )

        matches = []
        if not results['ids'] or not results['ids'][0]:
            return matches

        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for id, dist, meta in zip(ids, distances, metadatas):
            # Convert cosine distance to similarity score
            # ChromaDB returns squared L2 distance by default, but we set cosine
            # For cosine: distance = 1 - similarity, so similarity = 1 - distance
            score = max(0, 1 - dist)

            match_data = {
                "path": meta.get("path", id),
                "score": score,
                "metadata": meta,
                "verified_matches": 0
            }
            matches.append(match_data)

        # Geometric verification (optional)
        if verify and self.matcher and query_path and os.path.exists(query_path):
            matches = self._verify_matches(query_path, matches)
            # Re-sort by verified matches then score
            matches.sort(key=lambda x: (x['verified_matches'], x['score']), reverse=True)

        # Return only top_k results
        return matches[:top_k]

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
            img = cv2.imread(image_path)
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

    def search_faces(
        self,
        query_path: str,
        top_k: int = 50,
        collection_name: str = "images"
    ) -> list:
        """
        Search for similar faces using InsightFace embeddings.

        Args:
            query_path: Path to query image containing face(s)
            top_k: Number of results to return
            collection_name: Base name of the collection

        Returns:
            List of match dictionaries with path, score, metadata
        """
        embeddings = self.get_face_embedding(query_path)
        if not embeddings:
            print("No faces detected in query image")
            return []

        # Use first detected face for search
        query_emb = embeddings[0]
        return self._search_faces_with_embedding(query_emb, top_k, collection_name)

    def search_faces_by_bytes(
        self,
        image_bytes: bytes,
        top_k: int = 50,
        collection_name: str = "images"
    ) -> list:
        """Search for similar faces using image bytes."""
        embeddings = self.get_face_embedding_from_bytes(image_bytes)
        if not embeddings:
            print("No faces detected in query image")
            return []

        # Use first detected face for search
        query_emb = embeddings[0]
        return self._search_faces_with_embedding(query_emb, top_k, collection_name)

    def _search_faces_with_embedding(
        self,
        query_emb: np.ndarray,
        top_k: int,
        collection_name: str
    ) -> list:
        """Internal face search using pre-computed embedding."""
        try:
            collection = self.client.get_collection(name=f"{collection_name}_faces")
        except Exception as e:
            print(f"Collection {collection_name}_faces not found: {e}")
            return []

        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        matches = []
        if not results['ids'] or not results['ids'][0]:
            return matches

        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for id, dist, meta in zip(ids, distances, metadatas):
            # Convert cosine distance to similarity score
            score = max(0, 1 - dist)

            match_data = {
                "path": meta.get("path", id),
                "score": score,
                "metadata": meta,
                "verified_matches": 0,
                "face_id": id
            }
            matches.append(match_data)

        return matches

    def _load_torch_image(self, path: str):
        """Load and prepare image for DISK/LightGlue."""
        img = cv2.imread(path)
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
        Checks ALL candidates for maximum accuracy.
        """
        if not self.extractor or not self.matcher:
            return matches

        query_img = self._load_torch_image(query_path)
        if query_img is None:
            return matches

        total = len(matches)
        print(f"Verifying {total} candidates...")

        with torch.no_grad():
            feats0_obj = self.extractor(query_img)[0]
            feats0 = {
                "keypoints": feats0_obj.keypoints.unsqueeze(0),
                "descriptors": feats0_obj.descriptors.unsqueeze(0),
                "image_size": torch.tensor(query_img.shape[-2:][::-1]).view(1, 2).to(self.device)
            }

        for i, match in enumerate(matches):
            if i % 100 == 0 and i > 0:
                print(f"  Verified {i}/{total}...")

            try:
                match_path = match['path']
                if not os.path.exists(match_path):
                    match['verified_matches'] = 0
                    continue

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

                    matches01 = self.matcher({"image0": feats0, "image1": feats1})
                    matches_idx = matches01["matches"][0]
                    # Count only valid matches (LightGlue returns -1 for non-matches)
                    valid_matches = (matches_idx > -1).sum().item()
                    match['verified_matches'] = valid_matches

            except Exception as e:
                print(f"Verification failed for {match.get('path', 'unknown')}: {e}")
                match['verified_matches'] = 0

        print(f"Verification complete.")
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

        # Load the match image
        img = cv2.imread(match_path)
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

    def get_collection_stats(self, collection_name: str = "images") -> dict:
        """Get statistics for a collection."""
        stats = {
            "visual_count": 0,
            "face_count": 0
        }

        try:
            visual_col = self.client.get_collection(f"{collection_name}_visual")
            stats["visual_count"] = visual_col.count()
        except:
            pass

        try:
            face_col = self.client.get_collection(f"{collection_name}_faces")
            stats["face_count"] = face_col.count()
        except:
            pass

        return stats
