"""
DISK Feature Storage and Retrieval for MS SQL Server.
Stores pre-computed DISK keypoints and descriptors for fast LightGlue matching.
"""
import gzip
import numpy as np
import pyodbc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Connection string for trivia database
# Adjust as needed for your environment
CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=trivia;"
    "Trusted_Connection=yes;"
)


@dataclass
class DiskFeatureData:
    """Container for DISK features."""
    keypoints: np.ndarray      # (N, 2) float32
    descriptors: np.ndarray    # (N, 128) float32
    image_size: Tuple[int, int]  # (height, width)
    padded_size: Tuple[int, int]  # (padded_height, padded_width)
    keypoint_count: int


class DiskFeatureStore:
    """
    SQL Server storage for pre-computed DISK features.

    Usage:
        store = DiskFeatureStore()

        # Save features
        store.save(image_path, keypoints, descriptors, image_size, padded_size, book_name)

        # Load features for multiple images (bulk)
        features = store.load_bulk(image_paths)

        # Check if image has features
        exists = store.exists(image_path)
    """

    def __init__(self, connection_string: str = CONNECTION_STRING):
        self.connection_string = connection_string
        self._conn = None

    def _get_connection(self) -> pyodbc.Connection:
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = pyodbc.connect(self.connection_string)
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _compress_array(arr: np.ndarray) -> bytes:
        """Compress numpy array to gzipped bytes."""
        return gzip.compress(arr.tobytes(), compresslevel=6)

    @staticmethod
    def _decompress_keypoints(data: bytes, count: int) -> np.ndarray:
        """Decompress keypoints from gzipped bytes."""
        arr = np.frombuffer(gzip.decompress(data), dtype=np.float32)
        return arr.reshape(count, 2)

    @staticmethod
    def _decompress_descriptors(data: bytes, count: int) -> np.ndarray:
        """Decompress descriptors from gzipped bytes (stored as float16, returned as float32)."""
        arr = np.frombuffer(gzip.decompress(data), dtype=np.float16)
        return arr.reshape(count, 128).astype(np.float32)

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
        Save DISK features for an image.

        Args:
            image_path: Absolute path to the image
            keypoints: (N, 2) array of keypoint coordinates
            descriptors: (N, 128) array of DISK descriptors
            image_size: (height, width) of original image
            padded_size: (height, width) of padded image (multiple of 16)
            book_name: Optional book name for organization

        Returns:
            True if successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Compress arrays (use float16 for descriptors to save space)
        kp_bytes = self._compress_array(keypoints.astype(np.float32))
        desc_bytes = self._compress_array(descriptors.astype(np.float16))

        try:
            cursor.execute(
                "EXEC dbo.sp_UpsertDiskFeatures ?, ?, ?, ?, ?, ?, ?, ?, ?",
                image_path,
                book_name,
                kp_bytes,
                desc_bytes,
                len(keypoints),
                image_size[1],  # width
                image_size[0],  # height
                padded_size[1],  # padded width
                padded_size[0],  # padded height
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving DISK features for {image_path}: {e}")
            conn.rollback()
            return False

    def save_batch(
        self,
        features_list: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int], str]]
    ) -> int:
        """
        Save multiple DISK features in a batch.

        Args:
            features_list: List of (image_path, keypoints, descriptors, image_size, padded_size, book_name)

        Returns:
            Number of successfully saved features
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        saved = 0

        for image_path, keypoints, descriptors, image_size, padded_size, book_name in features_list:
            kp_bytes = self._compress_array(keypoints.astype(np.float32))
            desc_bytes = self._compress_array(descriptors.astype(np.float16))

            try:
                cursor.execute(
                    "EXEC dbo.sp_UpsertDiskFeatures ?, ?, ?, ?, ?, ?, ?, ?, ?",
                    image_path,
                    book_name,
                    kp_bytes,
                    desc_bytes,
                    len(keypoints),
                    image_size[1],
                    image_size[0],
                    padded_size[1],
                    padded_size[0],
                )
                saved += 1
            except Exception as e:
                print(f"Error saving batch item {image_path}: {e}")

        conn.commit()
        return saved

    def load(self, image_path: str) -> Optional[DiskFeatureData]:
        """
        Load DISK features for a single image.

        Args:
            image_path: Absolute path to the image

        Returns:
            DiskFeatureData or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT Keypoints, Descriptors, KeypointCount,
                   ImageWidth, ImageHeight, PaddedWidth, PaddedHeight
            FROM dbo.DiskFeatures
            WHERE ImagePath = ?
        """, image_path)

        row = cursor.fetchone()
        if not row:
            return None

        kp_bytes, desc_bytes, count, width, height, pad_w, pad_h = row

        return DiskFeatureData(
            keypoints=self._decompress_keypoints(kp_bytes, count),
            descriptors=self._decompress_descriptors(desc_bytes, count),
            image_size=(height, width),
            padded_size=(pad_h, pad_w),
            keypoint_count=count
        )

    def load_bulk(self, image_paths: List[str]) -> Dict[str, DiskFeatureData]:
        """
        Load DISK features for multiple images efficiently.

        Args:
            image_paths: List of absolute paths to images

        Returns:
            Dictionary mapping image_path -> DiskFeatureData
        """
        if not image_paths:
            return {}

        conn = self._get_connection()
        cursor = conn.cursor()

        # Use table-valued parameter for efficiency with large lists
        # For smaller lists, use IN clause with parameterized query
        results = {}

        # Process in batches of 1000 to avoid SQL parameter limits
        batch_size = 1000
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            placeholders = ','.join(['?' for _ in batch])

            cursor.execute(f"""
                SELECT ImagePath, Keypoints, Descriptors, KeypointCount,
                       ImageWidth, ImageHeight, PaddedWidth, PaddedHeight
                FROM dbo.DiskFeatures
                WHERE ImagePath IN ({placeholders})
            """, batch)

            for row in cursor.fetchall():
                path, kp_bytes, desc_bytes, count, width, height, pad_w, pad_h = row
                results[path] = DiskFeatureData(
                    keypoints=self._decompress_keypoints(kp_bytes, count),
                    descriptors=self._decompress_descriptors(desc_bytes, count),
                    image_size=(height, width),
                    padded_size=(pad_h, pad_w),
                    keypoint_count=count
                )

        return results

    def exists(self, image_path: str) -> bool:
        """Check if DISK features exist for an image."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM dbo.DiskFeatures WHERE ImagePath = ?",
            image_path
        )
        return cursor.fetchone() is not None

    def exists_bulk(self, image_paths: List[str]) -> Dict[str, bool]:
        """Check existence for multiple images."""
        if not image_paths:
            return {}

        conn = self._get_connection()
        cursor = conn.cursor()

        results = {path: False for path in image_paths}

        batch_size = 1000
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            placeholders = ','.join(['?' for _ in batch])

            cursor.execute(f"""
                SELECT ImagePath FROM dbo.DiskFeatures
                WHERE ImagePath IN ({placeholders})
            """, batch)

            for row in cursor.fetchall():
                results[row[0]] = True

        return results

    def get_stats(self) -> dict:
        """Get storage statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dbo.vw_DiskFeaturesStats")
        row = cursor.fetchone()

        if not row:
            return {
                "total_images": 0,
                "total_books": 0,
                "total_keypoints": 0,
                "avg_keypoints_per_image": 0,
                "total_storage_mb": 0,
                "avg_storage_per_image_kb": 0
            }

        return {
            "total_images": row[0],
            "total_books": row[1],
            "total_keypoints": row[2],
            "avg_keypoints_per_image": row[3],
            "total_storage_mb": float(row[4]) if row[4] else 0,
            "avg_storage_per_image_kb": float(row[5]) if row[5] else 0,
            "first_indexed": row[6],
            "last_indexed": row[7]
        }

    def get_book_images(self, book_name: str) -> List[str]:
        """Get all indexed image paths for a book."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ImagePath FROM dbo.DiskFeatures WHERE BookName = ?",
            book_name
        )
        return [row[0] for row in cursor.fetchall()]

    def delete(self, image_path: str) -> bool:
        """Delete DISK features for an image."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.DiskFeatures WHERE ImagePath = ?", image_path)
        conn.commit()
        return cursor.rowcount > 0

    def delete_book(self, book_name: str) -> int:
        """Delete all DISK features for a book."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dbo.DiskFeatures WHERE BookName = ?", book_name)
        conn.commit()
        return cursor.rowcount


# Convenience function for quick operations
def get_disk_features(image_paths: List[str]) -> Dict[str, DiskFeatureData]:
    """Quick function to load DISK features for multiple images."""
    with DiskFeatureStore() as store:
        return store.load_bulk(image_paths)


if __name__ == "__main__":
    # Test connection and show stats
    print("Testing DISK Feature Store connection...")
    try:
        store = DiskFeatureStore()
        stats = store.get_stats()
        print(f"Connection successful!")
        print(f"Stats: {stats}")
        store.close()
    except Exception as e:
        print(f"Connection failed: {e}")
        print("\nMake sure to:")
        print("1. Run the SQL script: sql/create_disk_features_table.sql")
        print("2. Adjust CONNECTION_STRING if needed")
