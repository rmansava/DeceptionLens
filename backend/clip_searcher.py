"""
CLIP Searcher for DinoDeceptionLens
Uses pre-built FAISS index from ImageSnippetSearch
Supports both image-to-image and text-to-image search
Supports multiple collections (books, print_ads, etc.)
"""
import os
import json
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from pathlib import Path

# Collection index paths
COLLECTION_PATHS = {
    "books": {
        "index": "D:/faiss/books/index.faiss",
        "paths": "D:/faiss/books/paths.json"
    },
    "print_ads": {
        "index": "D:/faiss/printads/index.faiss",
        "paths": "D:/faiss/printads/paths.json"
    },
    "board_games": {
        "index": "D:/faiss/board_games/index.faiss",
        "paths": "D:/faiss/board_games/paths.json"
    }
}


class ClipSearcher:
    """
    Searches using CLIP embeddings and FAISS index.
    Supports both visual (image) and text queries.
    Supports memory-mapping for low-RAM operation with NAS storage.
    """

    def __init__(
        self,
        index_path: str = "D:/faiss/books/index.faiss",
        paths_path: str = "D:/faiss/books/paths.json",
        model_name: str = "ViT-L/14",
        use_mmap: bool = False,
        collection: str = "books"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CLIP Searcher using device: {self.device}")

        self.index_path = index_path
        self.paths_path = paths_path
        self.model_name = model_name
        self.use_mmap = use_mmap
        self.collection = collection

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

            if self.use_mmap:
                # Memory-map the index - minimal RAM usage, good for NAS
                print(f"Memory-mapping FAISS index from {self.index_path}...")
                self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
                print(f"Index mmap'd: {self.index.ntotal:,} images (low RAM mode)")
            else:
                # Load fully into RAM - faster but uses more memory
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

    def search_with_rerank(
        self,
        image_path: str,
        top_k: int = 50,
        retrieval_k: int = 20000,
        rerank_k: int = 1000,
        verbose: bool = True
    ) -> list:
        """
        Search using CLIP + DISK keypoints + Template matching re-ranking.

        Pipeline:
        1. CLIP semantic search -> get retrieval_k candidates
        2. DISK keypoint filtering -> filter blanks, sort by matches
        3. Template matching on top rerank_k -> precise ranking
        4. Combined scoring -> final ranking

        Args:
            image_path: Path to query image
            top_k: Number of final results to return
            retrieval_k: Number of CLIP candidates to retrieve
            rerank_k: Number of candidates to run template matching on
            verbose: Print progress

        Returns:
            List of results with combined scoring
        """
        from clip_reranker import rerank_with_orb_and_template

        # Step 1: Get CLIP candidates
        if verbose:
            print(f"CLIP search: retrieving {retrieval_k} candidates...")

        results = self.search_by_image(image_path, top_k=retrieval_k)

        if not results:
            return []

        if verbose:
            print(f"  Got {len(results)} CLIP results")

        # Step 2-3: Apply ORB + Template re-ranking
        reranked = rerank_with_orb_and_template(
            query_image_path=image_path,
            results=results,
            collection=self.collection,
            top_for_template=rerank_k,
            verbose=verbose
        )

        # Return top_k
        return reranked[:top_k]

    def search_by_image_bytes_with_rerank(
        self,
        image_bytes: bytes,
        top_k: int = 50,
        retrieval_k: int = 20000,
        rerank_k: int = 1000,
        verbose: bool = True
    ) -> list:
        """
        Search with re-ranking using image bytes.
        Saves to temp file then calls search_with_rerank.
        """
        import tempfile
        import os

        # Save to temp file for re-ranking (needs path for cv2.imread)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        try:
            results = self.search_with_rerank(
                image_path=temp_path,
                top_k=top_k,
                retrieval_k=retrieval_k,
                rerank_k=rerank_k,
                verbose=verbose
            )
        finally:
            os.unlink(temp_path)

        return results

    def get_stats(self) -> dict:
        """Get index statistics."""
        self._ensure_loaded()
        return {
            "total_images": self.index.ntotal,
            "model": self.model_name,
            "index_path": self.index_path
        }


# Factory functions for multi-collection support

def get_clip_searcher(collection: str = "books") -> ClipSearcher:
    """
    Get a CLIP searcher for a specific collection.

    Args:
        collection: Collection name ("books", "print_ads", etc.)

    Returns:
        ClipSearcher instance configured for that collection
    """
    if collection not in COLLECTION_PATHS:
        raise ValueError(f"Unknown collection: {collection}. Available: {list(COLLECTION_PATHS.keys())}")

    paths = COLLECTION_PATHS[collection]
    return ClipSearcher(
        index_path=paths["index"],
        paths_path=paths["paths"],
        collection=collection
    )


def list_clip_collections() -> list:
    """
    List all available CLIP collections with their status.

    Returns:
        List of dicts with collection info and availability
    """
    result = []
    for name, paths in COLLECTION_PATHS.items():
        index_exists = os.path.exists(paths["index"])
        paths_exists = os.path.exists(paths["paths"])

        count = 0
        if index_exists:
            try:
                idx = faiss.read_index(paths["index"])
                count = idx.ntotal
            except:
                pass

        result.append({
            "name": name,
            "index_path": paths["index"],
            "paths_path": paths["paths"],
            "available": index_exists and paths_exists,
            "image_count": count
        })

    return result


# Cache of loaded searchers to avoid reloading
_searcher_cache = {}


def get_cached_clip_searcher(collection: str = "books") -> ClipSearcher:
    """
    Get a cached CLIP searcher for a collection.
    Searchers are reused to avoid reloading models and indexes.

    Args:
        collection: Collection name

    Returns:
        Cached ClipSearcher instance
    """
    if collection not in _searcher_cache:
        _searcher_cache[collection] = get_clip_searcher(collection)
    return _searcher_cache[collection]


def search_all_collections(
    image_bytes: bytes = None,
    image_path: str = None,
    text_query: str = None,
    top_k: int = 50,
    use_mmap: bool = True
) -> list:
    """
    Search across ALL available CLIP collections and merge results.

    Args:
        image_bytes: Image bytes for visual search
        image_path: Image path for visual search
        text_query: Text query for text search
        top_k: Number of results per collection (final results sorted by score)
        use_mmap: Use memory-mapped indices (recommended for NAS/low RAM)

    Returns:
        Merged list of results from all collections, sorted by score
    """
    all_results = []

    # Get available collections
    available = [c for c in list_clip_collections() if c["available"]]

    if not available:
        print("No CLIP collections available")
        return []

    print(f"Searching {len(available)} collections: {[c['name'] for c in available]}")

    for coll_info in available:
        coll_name = coll_info["name"]
        try:
            searcher = get_cached_clip_searcher(coll_name)

            if image_bytes:
                results = searcher.search_by_image_bytes(image_bytes, top_k=top_k)
            elif image_path:
                results = searcher.search_by_image(image_path, top_k=top_k)
            elif text_query:
                results = searcher.search_by_text(text_query, top_k=top_k)
            else:
                continue

            # Tag results with collection name
            for r in results:
                r["collection"] = coll_name

            all_results.extend(results)
            print(f"  {coll_name}: {len(results)} results")

        except Exception as e:
            print(f"  {coll_name}: Error - {e}")
            continue

    # Sort by score (descending) and return top results
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    print(f"Total: {len(all_results)} results from all collections")
    return all_results[:top_k * len(available)]  # Return more results for multi-collection
