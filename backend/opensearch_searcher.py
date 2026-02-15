"""
OpenSearch-based searcher for DINOv2 visual and InsightFace face embeddings.

This is the runtime module imported by server.py.
"""

import logging
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from opensearchpy import OpenSearch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-books"
FACES_INDEX = "faces-books"


class OpenSearchSearcher:
    """Search images using DINOv2 and InsightFace embeddings stored in OpenSearch."""

    def __init__(
        self,
        visual_index: str = VISUAL_INDEX,
        faces_index: str = FACES_INDEX
    ):
        self.visual_index = visual_index
        self.faces_index = faces_index
        self.client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            timeout=86400  # 24 hours
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"OpenSearch searcher using device: {self.device}")

        # DINOv2 model for visual search.
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.model.eval()

        # InsightFace is lazy-loaded for face search.
        self.face_app = None
        self.face_app_loaded = False

    def _load_face_app(self) -> bool:
        """Lazy-load InsightFace."""
        if self.face_app_loaded:
            return self.face_app is not None

        self.face_app_loaded = True
        if not INSIGHTFACE_AVAILABLE:
            logger.warning("InsightFace not available")
            return False

        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            self.face_app = None
            return False

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _get_visual_index(self, collection: Optional[str] = None) -> str:
        if collection:
            return f"dinov2-{collection}"
        return self.visual_index

    def _get_faces_index(self, collection: Optional[str] = None) -> str:
        if collection:
            return f"faces-{collection}"
        return self.faces_index

    def get_visual_embedding(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            return self._normalize_embedding(embedding)
        except Exception as e:
            logger.error(f"Error getting visual embedding for {image_path}: {e}")
            return None

    def get_visual_embedding_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            return self._normalize_embedding(embedding)
        except Exception as e:
            logger.error(f"Error getting visual embedding from bytes: {e}")
            return None

    def get_face_embedding_from_bytes(self, image_bytes: bytes) -> List[np.ndarray]:
        if not self._load_face_app():
            return []

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return []
            faces = self.face_app.get(img)
            return [face.embedding for face in faces]
        except Exception as e:
            logger.error(f"Error extracting faces from bytes: {e}")
            return []

    def search(self, query_path: str, top_k: int = 50, collection: Optional[str] = None) -> List[Dict]:
        embedding = self.get_visual_embedding(query_path)
        if embedding is None:
            return []
        return self._search_visual(embedding, top_k, collection)

    def search_by_bytes(self, image_bytes: bytes, top_k: int = 50, collection: Optional[str] = None) -> List[Dict]:
        embedding = self.get_visual_embedding_from_bytes(image_bytes)
        if embedding is None:
            return []
        return self._search_visual(embedding, top_k, collection)

    def _search_visual(self, embedding: np.ndarray, top_k: int, collection: Optional[str] = None) -> List[Dict]:
        index_name = self._get_visual_index(collection)
        capped_k = min(top_k, 10000)
        query = {
            "size": capped_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding.tolist(),
                        "k": capped_k
                    }
                }
            }
        }
        try:
            response = self.client.search(index=index_name, body=query)
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "path": hit["_source"]["path"],
                    "score": hit["_score"],
                    "metadata": {
                        "filename": hit["_source"].get("filename", ""),
                        "book": hit["_source"].get("book", ""),
                        "path": hit["_source"]["path"],
                        "collection": collection
                    },
                    "verified_matches": 0
                })
            return results
        except Exception as e:
            logger.error(f"OpenSearch visual search error: {e}")
            return []

    def _search_faces(self, embedding: np.ndarray, top_k: int, collection: Optional[str] = None) -> List[Dict]:
        index_name = self._get_faces_index(collection)
        capped_k = min(top_k, 10000)
        query = {
            "size": capped_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding.tolist(),
                        "k": capped_k
                    }
                }
            }
        }
        try:
            response = self.client.search(index=index_name, body=query)
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "path": hit["_source"]["path"],
                    "score": hit["_score"],
                    "metadata": {
                        "source_image": hit["_source"].get("source_image", ""),
                        "face_index": hit["_source"].get("face_index", 0),
                        "book": hit["_source"].get("book", ""),
                        "path": hit["_source"]["path"],
                        "collection": collection
                    },
                    "verified_matches": 0,
                    "face_id": hit["_id"]
                })
            return results
        except Exception as e:
            logger.error(f"OpenSearch face search error: {e}")
            return []

    def search_faces_by_bytes(
        self,
        image_bytes: bytes,
        top_k: int = 50,
        collection: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar faces using all detected query faces.

        Aggregates per-face hits so group photos and multi-face snippets are
        ranked by how many query faces they match, then by best score.
        """
        embeddings = self.get_face_embedding_from_bytes(image_bytes)
        detected_faces = len(embeddings)
        if detected_faces == 0:
            logger.info("No faces detected in query image")
            return []

        # Pull a wider candidate pool per query face before aggregation.
        per_face_k = min(max(top_k * 3, top_k), 500)
        aggregated: Dict[str, Dict] = {}

        for query_face_idx, emb in enumerate(embeddings):
            face_hits = self._search_faces(emb, per_face_k, collection)
            for hit in face_hits:
                score = float(hit.get("score", 0.0))
                if score < min_score:
                    continue

                meta = hit.get("metadata", {}) or {}
                source_image = meta.get("source_image") or hit.get("path")
                if not source_image:
                    continue

                entry = aggregated.get(source_image)
                if entry is None:
                    entry = {
                        "path": source_image,
                        "best_score": score,
                        "sum_score": score,
                        "candidate_face_hits": 1,
                        "matched_query_faces": {query_face_idx},
                        "best_hit": hit
                    }
                    aggregated[source_image] = entry
                else:
                    entry["sum_score"] += score
                    entry["candidate_face_hits"] += 1
                    entry["matched_query_faces"].add(query_face_idx)
                    if score > entry["best_score"]:
                        entry["best_score"] = score
                        entry["best_hit"] = hit

        if not aggregated:
            return []

        merged_results = []
        for source_image, data in aggregated.items():
            best_hit = data["best_hit"]
            best_meta = dict(best_hit.get("metadata", {}) or {})
            matched_query_faces = len(data["matched_query_faces"])
            best_meta.update({
                "source_image": source_image,
                "detected_query_faces": detected_faces,
                "matched_query_faces": matched_query_faces,
                "candidate_face_hits": data["candidate_face_hits"],
                "best_face_score": data["best_score"],
                "collection": collection
            })
            merged_results.append({
                "path": source_image,
                "score": data["best_score"],
                "verified_matches": matched_query_faces,
                "metadata": best_meta,
                "face_id": best_hit.get("face_id")
            })

        merged_results.sort(
            key=lambda r: (
                int((r.get("metadata") or {}).get("matched_query_faces", 0)),
                float(r.get("score", 0.0))
            ),
            reverse=True
        )
        return merged_results[:top_k]

    def get_counts(self, collection: Optional[str] = None) -> Dict[str, int]:
        counts = {"visual": 0, "faces": 0}
        visual_index = self._get_visual_index(collection)
        faces_index = self._get_faces_index(collection)
        try:
            counts["visual"] = self.client.count(index=visual_index)["count"]
        except Exception:
            pass
        try:
            counts["faces"] = self.client.count(index=faces_index)["count"]
        except Exception:
            pass
        return counts


# Backwards compatibility alias
OpenSearchVisualSearcher = OpenSearchSearcher
