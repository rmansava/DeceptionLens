"""
DINOv2 + InsightFace indexer using OpenSearch for vector storage.
More robust than ChromaDB for large-scale batch indexing.
"""
import os
import glob
import gc
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch, helpers
import numpy as np
import cv2

# Try importing InsightFace (optional)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not installed. Face indexing will be disabled.")

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-books"
FACES_INDEX = "faces-books"


class OpenSearchIndexer:
    """Indexes images using DINOv2 (visual) and InsightFace (faces) into OpenSearch."""

    def __init__(
        self,
        visual_index: str = VISUAL_INDEX,
        faces_index: str = FACES_INDEX,
        enable_visual: bool = True,
        enable_faces: bool = True
    ):
        self.visual_index = visual_index
        self.faces_index = faces_index
        self.enable_visual = enable_visual
        self.enable_faces = enable_faces

        self.client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            timeout=60
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize DINOv2 for visual embeddings
        self.processor = None
        self.model = None
        if self.enable_visual:
            print("Loading DINOv2 model (facebook/dinov2-base)...")
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
            self.model.eval()
            print("DINOv2 loaded successfully.")

        # Initialize InsightFace for face embeddings
        self.face_app = None
        if self.enable_faces and INSIGHTFACE_AVAILABLE:
            print("Loading InsightFace (buffalo_l / ArcFace)...")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace loaded successfully.")

    def get_visual_embedding(self, image_path: str) -> np.ndarray:
        """Generate DINOv2 embedding for an image."""
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
            print(f"Error processing {image_path}: {e}")
            return None

    def get_face_embeddings(self, image_path: str) -> list:
        """Extract face embeddings from an image using InsightFace."""
        if self.face_app is None:
            return []

        try:
            # Use imdecode to handle non-ASCII paths on Windows
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return []

            faces = self.face_app.get(img)
            return [face.embedding for face in faces]
        except Exception as e:
            print(f"Error extracting faces from {image_path}: {e}")
            return []

    def index_directory(self, dir_path: str, book_name: str = None, batch_size: int = 100):
        """Index all images in a directory (both visual and face embeddings)."""
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif', '*.bmp']
        files_set = set()
        for ext in image_extensions:
            for f in glob.glob(os.path.join(dir_path, '**', ext), recursive=True):
                files_set.add(os.path.normpath(f))
            for f in glob.glob(os.path.join(dir_path, '**', ext.upper()), recursive=True):
                files_set.add(os.path.normpath(f))

        files = sorted(list(files_set))
        if not files:
            print(f"No images found in {dir_path}")
            return {"visual": 0, "faces": 0}

        print(f"Found {len(files)} images to index.")

        if book_name is None:
            book_name = os.path.basename(dir_path)

        visual_actions = []
        face_actions = []
        visual_count = 0
        face_count = 0

        for file_path in tqdm(files, desc="Indexing"):
            # Visual embedding
            if self.enable_visual and self.model:
                embedding = self.get_visual_embedding(file_path)
                if embedding is not None:
                    visual_actions.append({
                        "_index": self.visual_index,
                        "_id": file_path,
                        "_source": {
                            "embedding": embedding.tolist(),
                            "path": file_path,
                            "filename": os.path.basename(file_path),
                            "book": book_name
                        }
                    })
                    visual_count += 1

            # Face embeddings
            if self.enable_faces and self.face_app:
                face_embeddings = self.get_face_embeddings(file_path)
                for i, face_emb in enumerate(face_embeddings):
                    face_id = f"{file_path}_face_{i}"
                    face_actions.append({
                        "_index": self.faces_index,
                        "_id": face_id,
                        "_source": {
                            "embedding": face_emb.tolist(),
                            "path": file_path,
                            "source_image": file_path,
                            "face_index": i,
                            "book": book_name
                        }
                    })
                    face_count += 1

            # Bulk insert visual when batch is full
            if len(visual_actions) >= batch_size:
                try:
                    helpers.bulk(self.client, visual_actions, refresh=False)
                except Exception as e:
                    print(f"Visual bulk insert error: {e}")
                visual_actions = []

            # Bulk insert faces when batch is full
            if len(face_actions) >= batch_size:
                try:
                    helpers.bulk(self.client, face_actions, refresh=False)
                except Exception as e:
                    print(f"Face bulk insert error: {e}")
                face_actions = []

            # Garbage collection
            gc.collect()

        # Insert remaining documents
        if visual_actions:
            try:
                helpers.bulk(self.client, visual_actions, refresh=False)
            except Exception as e:
                print(f"Final visual bulk insert error: {e}")

        if face_actions:
            try:
                helpers.bulk(self.client, face_actions, refresh=False)
            except Exception as e:
                print(f"Final face bulk insert error: {e}")

        # Refresh indices
        if self.enable_visual:
            self.client.indices.refresh(index=self.visual_index)
        if self.enable_faces:
            self.client.indices.refresh(index=self.faces_index)

        print(f"Indexed {visual_count} visual embeddings, {face_count} face embeddings")
        return {"visual": visual_count, "faces": face_count}

    def get_counts(self) -> dict:
        """Get document counts for both indices."""
        counts = {"visual": 0, "faces": 0}
        try:
            counts["visual"] = self.client.count(index=self.visual_index)["count"]
        except:
            pass
        try:
            counts["faces"] = self.client.count(index=self.faces_index)["count"]
        except:
            pass
        return counts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python opensearch_indexer.py <directory> [--visual-only] [--faces-only]")
        sys.exit(1)

    dir_path = sys.argv[1]
    enable_visual = "--faces-only" not in sys.argv
    enable_faces = "--visual-only" not in sys.argv

    indexer = OpenSearchIndexer(enable_visual=enable_visual, enable_faces=enable_faces)
    result = indexer.index_directory(dir_path)
    counts = indexer.get_counts()
    print(f"Total in index - Visual: {counts['visual']}, Faces: {counts['faces']}")
