"""
DinoDeceptionLens Indexer
Uses DINOv2 for visual embeddings and InsightFace for face detection.
Implements 2-pass system to avoid GPU conflicts between PyTorch and OnnxRuntime.
"""
import os
import sys
import glob
import gc
import torch
from PIL import Image
from tqdm import tqdm
import chromadb
from transformers import AutoImageProcessor, AutoModel
import numpy as np
import cv2

# Try importing InsightFace (optional)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not installed. Face recognition will be disabled.")


class DinoIndexer:
    """
    Indexes images using DINOv2 (visual) and InsightFace (faces).

    Due to GPU conflicts between PyTorch and OnnxRuntime, use 2-pass indexing:
    - Pass 1: --mode visual_only (DINOv2 on GPU)
    - Pass 2: --mode faces_only (InsightFace on GPU)
    """

    def __init__(
        self,
        collection_name: str = "images",
        db_path: str = "./chroma_db",
        enable_visual: bool = True,
        enable_faces: bool = True
    ):
        self.client = chromadb.PersistentClient(path=db_path)
        self.enable_visual = enable_visual
        self.enable_faces = enable_faces
        self.collection_name = collection_name

        # Initialize collections
        if self.enable_visual:
            self.visual_collection = self.client.get_or_create_collection(
                name=f"{collection_name}_visual",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.visual_collection = None

        if self.enable_faces:
            self.face_collection = self.client.get_or_create_collection(
                name=f"{collection_name}_faces",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.face_collection = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize DINOv2
        self.processor = None
        self.model = None
        if self.enable_visual:
            print("Loading DINOv2 model (facebook/dinov2-base)...")
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
            self.model.eval()
            print("DINOv2 loaded successfully.")

        # Initialize InsightFace
        self.face_app = None
        if self.enable_faces and INSIGHTFACE_AVAILABLE:
            print("Loading InsightFace (buffalo_l / ArcFace)...")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace loaded successfully.")

    def reset_collections(self):
        """Delete and recreate collections."""
        if self.enable_visual:
            try:
                self.client.delete_collection(f"{self.collection_name}_visual")
            except:
                pass
            self.visual_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_visual",
                metadata={"hnsw:space": "cosine"}
            )

        if self.enable_faces:
            try:
                self.client.delete_collection(f"{self.collection_name}_faces")
            except:
                pass
            self.face_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_faces",
                metadata={"hnsw:space": "cosine"}
            )
        print("Collections reset.")

    def get_dino_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate DINOv2 embedding for an image."""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            return embedding
        except Exception as e:
            print(f"DINOv2 Error: {e}")
            return None

    def get_face_embeddings(self, cv_image: np.ndarray) -> list:
        """Detect faces and return their embeddings."""
        embeddings = []
        if self.face_app is None:
            return embeddings

        try:
            faces = self.face_app.get(cv_image)
            for face in faces:
                embeddings.append(face.embedding)
        except Exception as e:
            print(f"InsightFace Error: {e}")

        return embeddings

    def process_image(self, image_path: str) -> dict:
        """Process a single image and return embeddings."""
        results = {
            "visual_embedding": None,
            "face_embeddings": []
        }

        try:
            # DINOv2 (requires PIL RGB image)
            if self.enable_visual and self.model:
                pil_image = Image.open(image_path).convert("RGB")
                results["visual_embedding"] = self.get_dino_embedding(pil_image)

            # InsightFace (requires CV2 BGR image)
            if self.enable_faces and self.face_app:
                cv_image = cv2.imread(image_path)
                if cv_image is not None:
                    results["face_embeddings"] = self.get_face_embeddings(cv_image)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        return results

    def index_directory(self, dir_path: str, path_mapping: tuple = None, batch_size: int = 10):
        """
        Index all images in a directory.

        Args:
            dir_path: Directory containing images
            path_mapping: Optional (source, target) tuple to remap stored paths
            batch_size: Number of images to batch before writing to DB
        """
        # Find all image files (case-insensitive, deduplicated)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif', '*.bmp']
        files_set = set()
        for ext in image_extensions:
            for f in glob.glob(os.path.join(dir_path, '**', ext), recursive=True):
                files_set.add(os.path.normpath(f))
            for f in glob.glob(os.path.join(dir_path, '**', ext.upper()), recursive=True):
                files_set.add(os.path.normpath(f))

        files = list(files_set)
        if not files:
            print(f"No images found in {dir_path}")
            return

        print(f"Found {len(files)} images to index.")
        files.sort()

        # Batch buffers
        vis_ids, vis_embs, vis_metas = [], [], []
        face_ids, face_embs, face_metas = [], [], []

        for file_path in tqdm(files, desc="Indexing"):
            try:
                data = self.process_image(file_path)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                continue

            # Determine stored path
            stored_path = file_path
            if path_mapping:
                local_root, target_root = path_mapping
                if stored_path.startswith(local_root):
                    stored_path = stored_path.replace(local_root, target_root, 1)

            # Add visual embedding
            if data["visual_embedding"] is not None:
                vis_ids.append(stored_path)
                vis_embs.append(data["visual_embedding"].tolist())
                vis_metas.append({
                    "path": stored_path,
                    "filename": os.path.basename(stored_path)
                })

            # Add face embeddings
            for i, face_emb in enumerate(data["face_embeddings"]):
                face_id = f"{stored_path}_face_{i}"
                face_ids.append(face_id)
                face_embs.append(face_emb.tolist())
                face_metas.append({
                    "path": stored_path,
                    "source_image": stored_path,
                    "face_index": i
                })

            # Flush visual batch
            if len(vis_ids) >= batch_size and self.visual_collection:
                try:
                    self.visual_collection.upsert(
                        ids=vis_ids,
                        embeddings=vis_embs,
                        metadatas=vis_metas
                    )
                except Exception as e:
                    print(f"Error upserting visual batch: {e}")
                vis_ids, vis_embs, vis_metas = [], [], []

            # Flush face batch
            if len(face_ids) >= batch_size and self.face_collection:
                try:
                    self.face_collection.upsert(
                        ids=face_ids,
                        embeddings=face_embs,
                        metadatas=face_metas
                    )
                except Exception as e:
                    print(f"Error upserting face batch: {e}")
                face_ids, face_embs, face_metas = [], [], []

            # Force garbage collection periodically
            gc.collect()

        # Flush remaining items
        if vis_ids and self.visual_collection:
            self.visual_collection.upsert(ids=vis_ids, embeddings=vis_embs, metadatas=vis_metas)
        if face_ids and self.face_collection:
            self.face_collection.upsert(ids=face_ids, embeddings=face_embs, metadatas=face_metas)

        # Print stats
        if self.visual_collection:
            print(f"Visual collection count: {self.visual_collection.count()}")
        if self.face_collection:
            print(f"Face collection count: {self.face_collection.count()}")
