"""
CLIP Searcher for DinoDeceptionLens
Uses pre-built FAISS index from ImageSnippetSearch
Supports both image-to-image and text-to-image search
"""
import os
import json
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from pathlib import Path


class ClipSearcher:
    """
    Searches using CLIP embeddings and FAISS index.
    Supports both visual (image) and text queries.
    """

    def __init__(
        self,
        index_path: str = "D:/faiss/books/index.faiss",
        paths_path: str = "D:/faiss/books/paths.json",
        model_name: str = "ViT-L/14"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CLIP Searcher using device: {self.device}")

        self.index_path = index_path
        self.paths_path = paths_path
        self.model_name = model_name

        # Lazy load - don't load until first search
        self.model = None
        self.preprocess = None
        self.index = None
        self.image_paths = None

    def _ensure_loaded(self):
        """Lazy load model and index on first use."""
        if self.model is None:
            print(f"Loading CLIP model ({self.model_name})...")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print("CLIP model loaded.")

        if self.index is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

            print(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            print(f"Index loaded: {self.index.ntotal:,} images")

            with open(self.paths_path) as f:
                self.image_paths = json.load(f)
            print(f"Paths loaded: {len(self.image_paths):,} entries")

    def search_by_image(self, image_path: str, top_k: int = 50) -> list:
        """Search using an image query."""
        self._ensure_loaded()

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_tensor).float()
                features /= features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy().astype('float32')

            distances, indices = self.index.search(features, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.image_paths):
                    continue
                path = self.image_paths[idx]
                results.append({
                    'path': path,
                    'score': float(dist),  # Inner product similarity
                    'verified_matches': 0,
                    'metadata': {
                        'path': path,
                        'filename': Path(path).name,
                        'folder': Path(path).parent.name
                    }
                })

            return results

        except Exception as e:
            print(f"CLIP image search error: {e}")
            return []

    def search_by_image_bytes(self, image_bytes: bytes, top_k: int = 50) -> list:
        """Search using image bytes."""
        self._ensure_loaded()

        try:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_tensor).float()
                features /= features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy().astype('float32')

            distances, indices = self.index.search(features, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.image_paths):
                    continue
                path = self.image_paths[idx]
                results.append({
                    'path': path,
                    'score': float(dist),
                    'verified_matches': 0,
                    'metadata': {
                        'path': path,
                        'filename': Path(path).name,
                        'folder': Path(path).parent.name
                    }
                })

            return results

        except Exception as e:
            print(f"CLIP image search error: {e}")
            return []

    def search_by_text(self, query_text: str, top_k: int = 50) -> list:
        """Search using a text query (e.g., 'truck', 'red car')."""
        self._ensure_loaded()

        try:
            # Tokenize and encode text
            text_tokens = clip.tokenize([query_text]).to(self.device)

            with torch.no_grad():
                features = self.model.encode_text(text_tokens).float()
                features /= features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy().astype('float32')

            distances, indices = self.index.search(features, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.image_paths):
                    continue
                path = self.image_paths[idx]
                results.append({
                    'path': path,
                    'score': float(dist),
                    'verified_matches': 0,
                    'metadata': {
                        'path': path,
                        'filename': Path(path).name,
                        'folder': Path(path).parent.name
                    }
                })

            return results

        except Exception as e:
            print(f"CLIP text search error: {e}")
            return []

    def get_stats(self) -> dict:
        """Get index statistics."""
        self._ensure_loaded()
        return {
            "total_images": self.index.ntotal,
            "model": self.model_name,
            "index_path": self.index_path
        }
