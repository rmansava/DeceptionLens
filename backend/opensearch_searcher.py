"""
OpenSearch-based searcher for DINOv2 visual and InsightFace face embeddings.
Complete replacement for ChromaDB-based search.
"""
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch
import numpy as np
import cv2

# Try importing InsightFace (optional)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-books"
FACES_INDEX = "faces-books"


class OpenSearchSearcher:
    """Searches images using DINOv2 and InsightFace embeddings stored in OpenSearch."""

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
        print(f"OpenSearch Searcher using device: {self.device}")

        # Initialize DINOv2
        print("Loading DINOv2 model for search...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.eval()
        print("DINOv2 loaded.")

        # InsightFace lazy-loaded
        self.face_app = None
        self.face_app_loaded = False

    def _load_face_app(self):
        """Lazy-load InsightFace."""
        if self.face_app_loaded:
            return self.face_app is not None

        if not INSIGHTFACE_AVAILABLE:
            print("InsightFace not available")
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

    def get_visual_embedding(self, image_path: str) -> np.ndarray:
        """Generate DINOv2 embedding for a query image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"Error getting embedding for {image_path}: {e}")
            return None

    def get_visual_embedding_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Generate DINOv2 embedding from image bytes."""
        try:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"Error getting embedding from bytes: {e}")
            return None

    def get_face_embedding_from_bytes(self, image_bytes: bytes) -> list:
        """Extract face embeddings from image bytes."""
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
            print(f"Error extracting faces from bytes: {e}")
            return []

    def _get_visual_index(self, collection: str = None) -> str:
        """Get the visual index name for a collection."""
        if collection:
            return f"dinov2-{collection}"
        return self.visual_index

    def _get_faces_index(self, collection: str = None) -> str:
        """Get the faces index name for a collection."""
        if collection:
            return f"faces-{collection}"
        return self.faces_index

    def search(self, query_path: str, top_k: int = 50, collection: str = None) -> list:
        """Search for similar images by path (visual search)."""
        embedding = self.get_visual_embedding(query_path)
        if embedding is None:
            return []
        return self._search_visual(embedding, top_k, collection)

    def search_by_bytes(self, image_bytes: bytes, top_k: int = 50, collection: str = None) -> list:
        """Search for similar images using image bytes (visual search)."""
        embedding = self.get_visual_embedding_from_bytes(image_bytes)
        if embedding is None:
            return []
        return self._search_visual(embedding, top_k, collection)

    def _search_visual(self, embedding: np.ndarray, top_k: int, collection: str = None) -> list:
        """Internal visual search using pre-computed embedding."""
        index_name = self._get_visual_index(collection)
        # OpenSearch has max k=10000 for k-NN and max_result_window=10000 for size
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
                        "path": hit["_source"]["path"]
                    },
                    "verified_matches": 0
                })
            return results
        except Exception as e:
            print(f"OpenSearch visual search error: {e}")
            return []

    def search_faces_by_bytes(self, image_bytes: bytes, top_k: int = 50, collection: str = None) -> list:
        """Search for similar faces using image bytes."""
        embeddings = self.get_face_embedding_from_bytes(image_bytes)
        if not embeddings:
            print("No faces detected in query image")
            return []

        # Use first detected face
        return self._search_faces(embeddings[0], top_k, collection)

    def _search_faces(self, embedding: np.ndarray, top_k: int, collection: str = None) -> list:
        """Internal face search using pre-computed embedding."""
        index_name = self._get_faces_index(collection)
        # OpenSearch has max k=10000 for k-NN and max_result_window=10000 for size
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
                        "path": hit["_source"]["path"]
                    },
                    "verified_matches": 0,
                    "face_id": hit["_id"]
                })
            return results
        except Exception as e:
            print(f"OpenSearch face search error: {e}")
            return []

    def get_counts(self, collection: str = None) -> dict:
        """Get document counts for both indices."""
        counts = {"visual": 0, "faces": 0}
        visual_index = self._get_visual_index(collection)
        faces_index = self._get_faces_index(collection)
        try:
            counts["visual"] = self.client.count(index=visual_index)["count"]
        except:
            pass
        try:
            counts["faces"] = self.client.count(index=faces_index)["count"]
        except:
            pass
        return counts


# Backwards compatibility alias
OpenSearchVisualSearcher = OpenSearchSearcher


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python opensearch_searcher.py <query_image> [--faces]")
        sys.exit(1)

    query_path = sys.argv[1]
    search_faces = "--faces" in sys.argv

    searcher = OpenSearchSearcher()
    counts = searcher.get_counts()
    print(f"Index counts - Visual: {counts['visual']}, Faces: {counts['faces']}")

    if search_faces:
        with open(query_path, 'rb') as f:
            image_bytes = f.read()
        results = searcher.search_faces_by_bytes(image_bytes, top_k=10)
        print(f"\nTop 10 face matches:")
    else:
        results = searcher.search(query_path, top_k=10)
        print(f"\nTop 10 visual matches:")

    for i, r in enumerate(results):
        print(f"  {i+1}. {r['path']} (score: {r['score']:.4f})")
