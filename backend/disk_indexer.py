"""
DISK Feature Indexer - Extracts and stores DISK keypoints/descriptors for fast LightGlue matching.
Run this after DINOv2/face indexing to pre-compute geometric verification features.
"""
import os
import glob
import gc
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional

# Kornia for DISK feature extraction
try:
    import kornia.feature as KF
    import kornia as K
    KORNIA_AVAILABLE = True
except ImportError:
    KF = None
    K = None
    KORNIA_AVAILABLE = False
    print("ERROR: Kornia not installed. Run: pip install kornia")

from disk_features import DiskFeatureStore


class DiskIndexer:
    """
    Extracts DISK keypoints and descriptors from images and stores them in SQL Server.

    This pre-computation speeds up LightGlue geometric verification from ~200ms to ~20ms per image.
    """

    def __init__(self, batch_size: int = 10, path_remap: Tuple[str, str] = None):
        """
        Initialize DISK indexer.

        Args:
            batch_size: Number of features to batch before SQL insert
            path_remap: Tuple of (source_prefix, target_prefix) to remap paths when storing.
                        E.g., ("D:\\books", "T:\\archiverelated\\books") reads from D: but stores T: paths.
        """
        if not KORNIA_AVAILABLE:
            raise RuntimeError("Kornia is required for DISK feature extraction")

        self.batch_size = batch_size
        self.path_remap = path_remap
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DISK Indexer using device: {self.device}")
        if path_remap:
            print(f"Path remapping: {path_remap[0]} -> {path_remap[1]}")

        # Initialize DISK extractor
        print("Loading DISK model...")
        self.extractor = KF.DISK.from_pretrained('depth').to(self.device).eval()
        print("DISK loaded successfully.")

        # Initialize SQL storage
        self.store = DiskFeatureStore()

    def _remap_path(self, path: str) -> str:
        """Remap path from source to target prefix if configured."""
        if self.path_remap:
            src, tgt = self.path_remap
            if path.lower().startswith(src.lower()):
                return tgt + path[len(src):]
        return path

    def close(self):
        """Clean up resources."""
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _load_and_pad_image(self, image_path: str) -> Optional[Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]]:
        """
        Load image and pad to dimensions divisible by 16 (required by DISK).

        Returns:
            Tuple of (tensor, original_size, padded_size) or None if failed
        """
        try:
            # Use cv2.imdecode for non-ASCII paths on Windows
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None

            h, w = img.shape[:2]
            original_size = (h, w)

            # Pad to multiples of 16
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            padded_size = (new_h, new_w)

            if new_h != h or new_w != w:
                img = cv2.copyMakeBorder(
                    img, 0, new_h - h, 0, new_w - w,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

            # Convert to tensor
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = K.image_to_tensor(img, False).float() / 255.0
            tensor = tensor.to(self.device)

            return tensor, original_size, padded_size

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def extract_features(self, image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]]:
        """
        Extract DISK features from an image.

        Returns:
            Tuple of (keypoints, descriptors, image_size, padded_size) or None if failed
        """
        result = self._load_and_pad_image(image_path)
        if result is None:
            return None

        tensor, original_size, padded_size = result

        try:
            with torch.no_grad():
                feats = self.extractor(tensor)[0]
                keypoints = feats.keypoints.cpu().numpy()  # (N, 2)
                descriptors = feats.descriptors.cpu().numpy()  # (N, 128)

            return keypoints, descriptors, original_size, padded_size

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def index_image(self, image_path: str, book_name: str = None) -> bool:
        """
        Extract and store DISK features for a single image.

        Args:
            image_path: Absolute path to the image
            book_name: Optional book name for organization

        Returns:
            True if successful
        """
        result = self.extract_features(image_path)
        if result is None:
            return False

        keypoints, descriptors, image_size, padded_size = result
        return self.store.save(image_path, keypoints, descriptors, image_size, padded_size, book_name)

    def index_directory(
        self,
        dir_path: str,
        book_name: str = None,
        skip_existing: bool = True
    ) -> dict:
        """
        Index all images in a directory.

        Args:
            dir_path: Directory containing images
            book_name: Book name (defaults to directory basename)
            skip_existing: Skip images that already have features stored

        Returns:
            Statistics dict with counts
        """
        # Find all images
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
            return {"total": 0, "indexed": 0, "skipped": 0, "failed": 0}

        if book_name is None:
            book_name = os.path.basename(dir_path)

        print(f"Found {len(files)} images in '{book_name}'")

        # Check which already exist (using remapped paths)
        if skip_existing:
            remapped_files = {f: self._remap_path(f) for f in files}
            existing = self.store.exists_bulk(list(remapped_files.values()))
            to_process = [f for f in files if not existing.get(remapped_files[f], False)]
            skipped = len(files) - len(to_process)
            print(f"Skipping {skipped} already indexed images")
        else:
            to_process = files
            skipped = 0

        if not to_process:
            print("All images already indexed!")
            return {"total": len(files), "indexed": 0, "skipped": skipped, "failed": 0}

        # Process images
        indexed = 0
        failed = 0
        batch = []

        for image_path in tqdm(to_process, desc=f"Indexing DISK features"):
            result = self.extract_features(image_path)

            if result is None:
                failed += 1
                continue

            keypoints, descriptors, image_size, padded_size = result
            # Use remapped path for storage
            store_path = self._remap_path(image_path)
            batch.append((store_path, keypoints, descriptors, image_size, padded_size, book_name))

            # Batch insert
            if len(batch) >= self.batch_size:
                saved = self.store.save_batch(batch)
                indexed += saved
                failed += len(batch) - saved
                batch = []

            # Garbage collection
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Insert remaining
        if batch:
            saved = self.store.save_batch(batch)
            indexed += saved
            failed += len(batch) - saved

        print(f"Indexed {indexed} images, {failed} failed, {skipped} skipped")
        return {
            "total": len(files),
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed
        }

    def get_stats(self) -> dict:
        """Get storage statistics."""
        return self.store.get_stats()


def main():
    """CLI for DISK indexing."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Index DISK features for LightGlue matching")
    parser.add_argument("directory", nargs="?", help="Directory containing images to index")
    parser.add_argument("--book", "-b", help="Book name (defaults to directory name)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for SQL inserts")
    parser.add_argument("--force", "-f", action="store_true", help="Re-index existing images")
    parser.add_argument("--stats", "-s", action="store_true", help="Show statistics and exit")
    parser.add_argument("--remap-from", help="Source path prefix to remap (e.g., D:\\books)")
    parser.add_argument("--remap-to", help="Target path prefix to store (e.g., T:\\archiverelated\\books)")

    args = parser.parse_args()

    # Build path remap tuple if both provided
    path_remap = None
    if args.remap_from and args.remap_to:
        path_remap = (args.remap_from, args.remap_to)
    elif args.remap_from or args.remap_to:
        print("Error: Both --remap-from and --remap-to must be provided together")
        sys.exit(1)

    with DiskIndexer(batch_size=args.batch_size, path_remap=path_remap) as indexer:
        if args.stats:
            stats = indexer.get_stats()
            print("\nDISK Feature Storage Statistics:")
            print(f"  Total images: {stats['total_images']:,}")
            print(f"  Total books: {stats['total_books']:,}")
            print(f"  Total keypoints: {stats['total_keypoints']:,}" if stats['total_keypoints'] else "  Total keypoints: 0")
            print(f"  Avg keypoints/image: {stats['avg_keypoints_per_image']:.0f}" if stats['avg_keypoints_per_image'] else "  Avg keypoints/image: 0")
            print(f"  Total storage: {stats['total_storage_mb']:.1f} MB")
            print(f"  Avg per image: {stats['avg_storage_per_image_kb']:.1f} KB")
            return

        if not args.directory:
            print("Error: directory is required (unless using --stats)")
            sys.exit(1)

        if not os.path.exists(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)

        result = indexer.index_directory(
            args.directory,
            book_name=args.book,
            skip_existing=not args.force
        )

        print(f"\nResult: {result}")
        print("\nCurrent storage stats:")
        stats = indexer.get_stats()
        print(f"  Total images indexed: {stats['total_images']:,}")
        print(f"  Storage used: {stats['total_storage_mb']:.1f} MB")


if __name__ == "__main__":
    main()
