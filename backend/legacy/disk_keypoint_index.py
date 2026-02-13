"""
DISK Keypoint FAISS Index with Path Management

Stores DISK keypoints in a FAISS index on NAS (T: drive) with automatic
path synchronization when books are renamed, deleted, or added.

Usage:
    index = DiskKeypointIndex()

    # Search
    results = index.search(query_image_path, top_k=10)

    # Maintenance
    index.add_book("Book Name")
    index.remove_book("Book Name")
    index.rename_book("Old Name", "New Name")
    index.rebuild()  # Full rebuild from disk features
"""

import faiss
import numpy as np
import json
import os
from glob import glob
from collections import Counter
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import threading
import shutil


class DiskKeypointIndex:
    """FAISS index for DISK keypoints with path management."""

    def __init__(
        self,
        index_dir: str = "T:/faiss/disk_keypoints",
        features_dir: str = "T:/disk-features/books",
        images_dir: str = "D:/books/pdf-images",
        dimension: int = 128
    ):
        self.index_dir = index_dir
        self.features_dir = features_dir
        self.images_dir = images_dir
        self.dimension = dimension

        self.index_path = os.path.join(index_dir, "index.faiss")
        self.paths_path = os.path.join(index_dir, "paths.json")
        self.metadata_path = os.path.join(index_dir, "metadata.json")

        self.index: Optional[faiss.Index] = None
        self.paths: List[str] = []
        self.metadata: Dict = {}

        self._lock = threading.Lock()

        # Load existing index or create directory
        if os.path.exists(self.index_path):
            self.load()
        else:
            os.makedirs(index_dir, exist_ok=True)

    def load(self):
        """Load existing index from disk."""
        with self._lock:
            print(f"Loading index from {self.index_dir}...")
            self.index = faiss.read_index(self.index_path)

            with open(self.paths_path, 'r') as f:
                self.paths = json.load(f)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)

            print(f"Loaded {self.index.ntotal:,} keypoints, {len(set(self.paths)):,} unique images")

    def save(self):
        """Save index to disk."""
        with self._lock:
            print(f"Saving index to {self.index_dir}...")

            # Backup existing files
            if os.path.exists(self.index_path):
                shutil.copy(self.index_path, self.index_path + ".backup")
            if os.path.exists(self.paths_path):
                shutil.copy(self.paths_path, self.paths_path + ".backup")

            faiss.write_index(self.index, self.index_path)

            with open(self.paths_path, 'w') as f:
                json.dump(self.paths, f)

            self.metadata['last_updated'] = datetime.now().isoformat()
            self.metadata['total_keypoints'] = self.index.ntotal
            self.metadata['unique_images'] = len(set(self.paths))

            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            print(f"Saved {self.index.ntotal:,} keypoints")

    def _get_book_features_dir(self, book_name: str) -> str:
        """Get the features directory for a book."""
        return os.path.join(self.features_dir, book_name)

    def _get_book_images_dir(self, book_name: str) -> str:
        """Get the images directory for a book."""
        return os.path.join(self.images_dir, book_name)

    def _load_book_features(self, book_name: str) -> Tuple[np.ndarray, List[str]]:
        """Load all DISK features for a book.

        Returns:
            descriptors: (N, 128) normalized float32 array
            paths: list of image paths for each keypoint
        """
        features_dir = self._get_book_features_dir(book_name)
        images_dir = self._get_book_images_dir(book_name)

        if not os.path.exists(features_dir):
            raise FileNotFoundError(f"Features not found: {features_dir}")

        npz_files = sorted(glob(os.path.join(features_dir, "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {features_dir}")

        all_descriptors = []
        all_paths = []

        for npz_path in npz_files:
            data = np.load(npz_path)
            descriptors = data['descriptors'].astype('float32')

            # Normalize
            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            descriptors = descriptors / (norms + 1e-8)

            # Get image path
            page_name = os.path.basename(npz_path).replace('.npz', '')
            image_path = os.path.join(images_dir, f"{page_name}.jpg")

            all_descriptors.append(descriptors)
            all_paths.extend([image_path] * len(descriptors))

        return np.vstack(all_descriptors), all_paths

    def add_book(self, book_name: str):
        """Add a book to the index."""
        print(f"Adding book: {book_name}")

        descriptors, paths = self._load_book_features(book_name)

        with self._lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.paths = []

            self.index.add(descriptors)
            self.paths.extend(paths)

        print(f"Added {len(descriptors):,} keypoints from {book_name}")
        self.save()

    def remove_book(self, book_name: str):
        """Remove a book from the index.

        Note: FAISS doesn't support efficient deletion, so this rebuilds
        the index without the specified book. For frequent deletions,
        consider using IndexIDMap.
        """
        print(f"Removing book: {book_name}")

        images_dir = self._get_book_images_dir(book_name)
        prefix = images_dir.replace("\\", "/")

        with self._lock:
            # Find indices to keep
            keep_indices = [
                i for i, p in enumerate(self.paths)
                if not p.replace("\\", "/").startswith(prefix)
            ]

            if len(keep_indices) == len(self.paths):
                print(f"Book not found in index: {book_name}")
                return

            removed_count = len(self.paths) - len(keep_indices)

            # Rebuild index without this book
            if keep_indices:
                # Get vectors to keep
                vectors = faiss.rev_swig_ptr(
                    self.index.get_xb(), self.index.ntotal * self.dimension
                ).reshape(self.index.ntotal, self.dimension).copy()

                keep_vectors = vectors[keep_indices]
                keep_paths = [self.paths[i] for i in keep_indices]

                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(keep_vectors)
                self.paths = keep_paths
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.paths = []

        print(f"Removed {removed_count:,} keypoints")
        self.save()

    def rename_book(self, old_name: str, new_name: str):
        """Rename a book in the index (updates paths only)."""
        print(f"Renaming book: {old_name} -> {new_name}")

        old_images_dir = self._get_book_images_dir(old_name).replace("\\", "/")
        new_images_dir = self._get_book_images_dir(new_name).replace("\\", "/")

        with self._lock:
            updated_count = 0
            for i, path in enumerate(self.paths):
                normalized_path = path.replace("\\", "/")
                if normalized_path.startswith(old_images_dir):
                    self.paths[i] = normalized_path.replace(old_images_dir, new_images_dir)
                    updated_count += 1

        if updated_count > 0:
            print(f"Updated {updated_count:,} paths")
            self.save()
        else:
            print(f"Book not found in index: {old_name}")

    def rebuild(self, progress_callback=None):
        """Rebuild entire index from disk features."""
        print("Rebuilding index from scratch...")

        book_dirs = [d for d in os.listdir(self.features_dir)
                     if os.path.isdir(os.path.join(self.features_dir, d))]

        print(f"Found {len(book_dirs)} books")

        all_descriptors = []
        all_paths = []

        for i, book_name in enumerate(book_dirs):
            try:
                descriptors, paths = self._load_book_features(book_name)
                all_descriptors.append(descriptors)
                all_paths.extend(paths)

                if (i + 1) % 100 == 0:
                    total_kp = sum(len(d) for d in all_descriptors)
                    print(f"  {i+1}/{len(book_dirs)} books ({total_kp:,} keypoints)")
                    if progress_callback:
                        progress_callback(i + 1, len(book_dirs), total_kp)

            except Exception as e:
                print(f"  Error loading {book_name}: {e}")

        with self._lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            if all_descriptors:
                self.index.add(np.vstack(all_descriptors))
            self.paths = all_paths
            self.metadata['books'] = book_dirs
            self.metadata['built_at'] = datetime.now().isoformat()

        print(f"Built index: {self.index.ntotal:,} keypoints from {len(book_dirs)} books")
        self.save()

    def search(
        self,
        query_descriptors: np.ndarray,
        top_k: int = 10,
        keypoint_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Search for images matching query keypoints.

        Args:
            query_descriptors: (N, 128) normalized float32 descriptors
            top_k: number of top images to return
            keypoint_k: number of nearest neighbors per keypoint
            threshold: minimum similarity for a keypoint match

        Returns:
            List of dicts with 'path' and 'votes' keys
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Ensure float32
        query_descriptors = query_descriptors.astype('float32')

        # Search
        distances, indices = self.index.search(query_descriptors, keypoint_k)

        # Vote for images
        votes = Counter()
        for i in range(len(query_descriptors)):
            for j in range(keypoint_k):
                idx = indices[i][j]
                if idx >= 0 and distances[i][j] >= threshold:
                    votes[self.paths[idx]] += 1

        # Return top results
        results = [
            {'path': path, 'votes': count}
            for path, count in votes.most_common(top_k)
        ]

        return results

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.index is None:
            return {'status': 'not_loaded'}

        unique_images = len(set(self.paths))
        return {
            'total_keypoints': self.index.ntotal,
            'unique_images': unique_images,
            'avg_keypoints_per_image': self.index.ntotal / max(unique_images, 1),
            'index_size_mb': os.path.getsize(self.index_path) / 1024 / 1024 if os.path.exists(self.index_path) else 0,
            'paths_size_mb': os.path.getsize(self.paths_path) / 1024 / 1024 if os.path.exists(self.paths_path) else 0,
            **self.metadata
        }


# Convenience function for searching with an image path
def search_by_image(
    query_image_path: str,
    index: DiskKeypointIndex = None,
    top_k: int = 10
) -> List[Dict]:
    """Search the index using an image file.

    Extracts DISK keypoints from the query image and searches.
    """
    import torch
    from PIL import Image
    import kornia.feature as KF

    # Load and pad image
    img = Image.open(query_image_path).convert('RGB')
    w, h = img.size
    new_w = ((w + 15) // 16) * 16
    new_h = ((h + 15) // 16) * 16
    padded = Image.new('RGB', (new_w, new_h), (0, 0, 0))
    padded.paste(img, (0, 0))

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(padded)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = img_tensor.to(device)

    # Extract DISK features
    extractor = KF.DISK.from_pretrained('depth').eval().to(device)
    with torch.no_grad():
        feats = extractor(img_tensor)
        descriptors = feats[0].descriptors.cpu().numpy()

    # Normalize
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = (descriptors / (norms + 1e-8)).astype('float32')

    # Search
    if index is None:
        index = DiskKeypointIndex()

    return index.search(descriptors, top_k=top_k)


if __name__ == "__main__":
    # Test/demo
    index = DiskKeypointIndex()

    print("\nIndex stats:")
    print(json.dumps(index.get_stats(), indent=2))

    # Example: rebuild for one book
    # index.add_book("encyclopedia of monsters")

    # Example: search
    # results = search_by_image("D:/trivpics/2023-5.jpg", index)
    # for r in results[:10]:
    #     print(f"{r['votes']:4d} votes: {r['path']}")
