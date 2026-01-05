r"""
File-based DISK Feature Storage and Retrieval.
Stores pre-computed DISK keypoints and descriptors as .npz files on NAS.
Alternative to SQL Server storage for faster NAS-based workflows.

Directory structure:
    T:\disk-features\{category}\{book_name}\{image_filename}.npz

Example:
    T:\disk-features\books\MyBook\page_001.npz
    T:\disk-features\printads\Magazine1965\ad_042.npz
"""
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


# Default root for DISK features on NAS
DEFAULT_FEATURES_ROOT = r"T:\disk-features"


@dataclass
class DiskFeatureData:
    """Container for DISK features."""
    keypoints: np.ndarray      # (N, 2) float32
    descriptors: np.ndarray    # (N, 128) float32
    image_size: Tuple[int, int]  # (height, width)
    padded_size: Tuple[int, int]  # (padded_height, padded_width)
    keypoint_count: int


class DiskFeatureFileStore:
    r"""
    File-based storage for pre-computed DISK features using .npz files.

    Usage:
        store = DiskFeatureFileStore(category="books")

        # Save features
        store.save(image_path, keypoints, descriptors, image_size, padded_size, book_name)

        # Load features for multiple images (bulk)
        features = store.load_bulk(image_paths)

        # Check if image has features
        exists = store.exists(image_path)
    """

    def __init__(
        self,
        category: str = "books",
        features_root: str = DEFAULT_FEATURES_ROOT,
        source_image_root: str = None
    ):
        """
        Initialize file-based DISK feature store.

        Args:
            category: Category folder (e.g., "books", "printads")
            features_root: Root directory for features (default: T:\disk-features)
            source_image_root: Root path of source images to strip when building feature paths
                              (e.g., "T:\\archiverelated\\books" or "D:\\books\\pdf-images")
        """
        self.category = category
        self.features_root = Path(features_root)
        self.category_root = self.features_root / category
        self.source_image_root = source_image_root

        # Ensure category directory exists
        self.category_root.mkdir(parents=True, exist_ok=True)

    def _get_feature_path(self, image_path: str, book_name: str = None) -> Path:
        """
        Convert image path to feature file path.

        Args:
            image_path: Absolute path to the image
            book_name: Book name (required for saving, optional for loading)

        Returns:
            Path to .npz feature file
        """
        image_path = Path(image_path)

        if book_name:
            # Use book_name directly
            filename = image_path.stem + ".npz"
            return self.category_root / book_name / filename

        # Try to infer from path structure
        # For paths like T:\archiverelated\books\BookName\page.jpg
        # or D:\books\pdf-images\BookName\page.jpg
        if self.source_image_root:
            rel_path = str(image_path).replace(self.source_image_root, "").lstrip("\\/")
            parts = Path(rel_path).parts
            if len(parts) >= 2:
                book_name = parts[0]
                filename = Path(parts[-1]).stem + ".npz"
                return self.category_root / book_name / filename

        # Fallback: use parent directory as book name
        book_name = image_path.parent.name
        filename = image_path.stem + ".npz"
        return self.category_root / book_name / filename

    def close(self):
        """No-op for file-based storage (for API compatibility)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(
        self,
        image_path: str,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        image_size: Tuple[int, int],
        padded_size: Tuple[int, int],
        book_name: str = None
    ) -> bool:
        """
        Save DISK features for an image as .npz file.

        Args:
            image_path: Absolute path to the image (used to derive filename)
            keypoints: (N, 2) array of keypoint coordinates
            descriptors: (N, 128) array of DISK descriptors
            image_size: (height, width) of original image
            padded_size: (height, width) of padded image (multiple of 16)
            book_name: Book name for directory organization

        Returns:
            True if successful
        """
        feature_path = self._get_feature_path(image_path, book_name)

        try:
            # Ensure book directory exists
            feature_path.parent.mkdir(parents=True, exist_ok=True)

            # Save compressed .npz (uses float16 for descriptors to save space)
            np.savez_compressed(
                feature_path,
                keypoints=keypoints.astype(np.float32),
                descriptors=descriptors.astype(np.float16),
                image_size=np.array(image_size, dtype=np.int32),
                padded_size=np.array(padded_size, dtype=np.int32),
                image_path=str(image_path)  # Store original path for reference
            )
            return True
        except Exception as e:
            print(f"Error saving DISK features for {image_path}: {e}")
            return False

    def save_batch(
        self,
        features_list: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int], str]]
    ) -> int:
        """
        Save multiple DISK features.

        Args:
            features_list: List of (image_path, keypoints, descriptors, image_size, padded_size, book_name)

        Returns:
            Number of successfully saved features
        """
        saved = 0
        for image_path, keypoints, descriptors, image_size, padded_size, book_name in features_list:
            if self.save(image_path, keypoints, descriptors, image_size, padded_size, book_name):
                saved += 1
        return saved

    def load(self, image_path: str, book_name: str = None) -> Optional[DiskFeatureData]:
        """
        Load DISK features for a single image.

        Args:
            image_path: Absolute path to the image
            book_name: Optional book name hint

        Returns:
            DiskFeatureData or None if not found
        """
        feature_path = self._get_feature_path(image_path, book_name)

        if not feature_path.exists():
            return None

        try:
            data = np.load(feature_path)
            return DiskFeatureData(
                keypoints=data['keypoints'].astype(np.float32),
                descriptors=data['descriptors'].astype(np.float32),
                image_size=tuple(data['image_size']),
                padded_size=tuple(data['padded_size']),
                keypoint_count=len(data['keypoints'])
            )
        except Exception as e:
            print(f"Error loading DISK features from {feature_path}: {e}")
            return None

    def _load_single(self, image_path: str) -> Tuple[str, Optional[DiskFeatureData]]:
        """Load single file (for parallel loading)."""
        return image_path, self.load(image_path)

    def load_bulk(self, image_paths: List[str], max_workers: int = 8) -> Dict[str, DiskFeatureData]:
        """
        Load DISK features for multiple images efficiently using parallel I/O.

        Args:
            image_paths: List of absolute paths to images
            max_workers: Number of parallel threads for loading

        Returns:
            Dictionary mapping image_path -> DiskFeatureData
        """
        if not image_paths:
            return {}

        results = {}

        # Use thread pool for parallel file I/O
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._load_single, path): path for path in image_paths}

            for future in as_completed(futures):
                try:
                    path, data = future.result()
                    if data is not None:
                        results[path] = data
                except Exception as e:
                    print(f"Error in bulk load: {e}")

        return results

    def exists(self, image_path: str, book_name: str = None) -> bool:
        """Check if DISK features exist for an image."""
        feature_path = self._get_feature_path(image_path, book_name)
        return feature_path.exists()

    def exists_bulk(self, image_paths: List[str]) -> Dict[str, bool]:
        """Check existence for multiple images."""
        return {path: self.exists(path) for path in image_paths}

    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_files = 0
        total_size = 0
        books = set()

        if self.category_root.exists():
            for book_dir in self.category_root.iterdir():
                if book_dir.is_dir():
                    books.add(book_dir.name)
                    for npz_file in book_dir.glob("*.npz"):
                        total_files += 1
                        total_size += npz_file.stat().st_size

        return {
            "total_images": total_files,
            "total_books": len(books),
            "total_keypoints": None,  # Would need to load all files to count
            "avg_keypoints_per_image": None,
            "total_storage_mb": total_size / (1024 * 1024),
            "avg_storage_per_image_kb": (total_size / total_files / 1024) if total_files > 0 else 0,
            "category": self.category,
            "features_root": str(self.features_root)
        }

    def get_book_images(self, book_name: str) -> List[str]:
        """Get all indexed feature file paths for a book."""
        book_dir = self.category_root / book_name
        if not book_dir.exists():
            return []
        return [str(f) for f in book_dir.glob("*.npz")]

    def delete(self, image_path: str, book_name: str = None) -> bool:
        """Delete DISK features for an image."""
        feature_path = self._get_feature_path(image_path, book_name)
        if feature_path.exists():
            feature_path.unlink()
            return True
        return False

    def delete_book(self, book_name: str) -> int:
        """Delete all DISK features for a book."""
        book_dir = self.category_root / book_name
        if not book_dir.exists():
            return 0

        count = 0
        for npz_file in book_dir.glob("*.npz"):
            npz_file.unlink()
            count += 1

        # Remove empty directory
        if not any(book_dir.iterdir()):
            book_dir.rmdir()

        return count


# Convenience function for quick operations
def get_disk_features_file(
    image_paths: List[str],
    category: str = "books",
    source_image_root: str = None
) -> Dict[str, DiskFeatureData]:
    """Quick function to load DISK features for multiple images from file storage."""
    with DiskFeatureFileStore(category=category, source_image_root=source_image_root) as store:
        return store.load_bulk(image_paths)


if __name__ == "__main__":
    # Test connection and show stats
    print("Testing DISK Feature File Store...")
    try:
        store = DiskFeatureFileStore(category="books")
        stats = store.get_stats()
        print(f"Store initialized at: {store.category_root}")
        print(f"Stats: {stats}")
        store.close()
    except Exception as e:
        print(f"Error: {e}")
