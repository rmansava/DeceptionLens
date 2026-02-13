#!/usr/bin/env python3
"""
CLIP Indexer for OpenSearch - indexes directly to OpenSearch with per-book checkpointing.
Supports path remapping (D:\ local -> T:\ NAS) and resume from interruption.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from opensearchpy import OpenSearch, helpers

# Logging setup
LOG_FILE = os.path.join(os.path.dirname(__file__), "clip_opensearch_indexer.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


class ClipOpenSearchIndexer:
    """Index CLIP embeddings directly to OpenSearch with per-book checkpointing."""

    def __init__(self, model_name: str = "ViT-L/14", batch_size: int = 64,
                 opensearch_host: str = "localhost", opensearch_port: int = 9200):
        self.model_name = model_name
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.device}")

        # OpenSearch client
        self.os_client = OpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            http_compress=True,
            use_ssl=False
        )

        # Lazy load model
        self.model = None
        self.preprocess = None

    def _load_model(self):
        """Load CLIP model."""
        if self.model is None:
            log.info(f"Loading CLIP model ({self.model_name})...")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            log.info("CLIP model loaded.")

    def create_index(self, index_name: str, embedding_dim: int = 768):
        """Create OpenSearch index for CLIP embeddings."""
        if self.os_client.indices.exists(index=index_name):
            log.info(f"Index {index_name} already exists")
            return False

        settings = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "-1"  # Disable during bulk indexing
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }
                    },
                    "path": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "folder": {"type": "keyword"}
                }
            }
        }

        log.info(f"Creating index {index_name}...")
        self.os_client.indices.create(index=index_name, body=settings)
        log.info(f"Index {index_name} created.")
        return True

    def _encode_batch(self, image_paths: list) -> np.ndarray:
        """Encode a batch of images to CLIP embeddings."""
        batch_images = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(self.preprocess(img))
            except Exception:
                # Placeholder for failed loads
                blank = Image.new("RGB", (224, 224), (128, 128, 128))
                batch_images.append(self.preprocess(blank))

        if not batch_images:
            return None

        image_input = torch.stack(batch_images).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_input).float()
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    def _get_book_folders(self, source_path: str) -> list:
        """Get list of book folders (immediate subdirectories)."""
        folders = []
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            if os.path.isdir(item_path):
                folders.append(item)
        folders.sort()
        return folders

    def _get_images_in_folder(self, folder_path: str) -> list:
        """Get all image files in a folder (non-recursive)."""
        images = []
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(folder_path, file))
        images.sort()
        return images

    def index(self, source_path: str, index_name: str,
              remap_from: str = None, remap_to: str = None,
              checkpoint_file: str = None) -> dict:
        """
        Index all images to OpenSearch with per-book checkpointing.

        Args:
            source_path: Root directory containing book folders
            index_name: OpenSearch index name (e.g., "clip-books")
            remap_from: Local path prefix to replace (e.g., "D:\\books")
            remap_to: NAS path prefix (e.g., "T:\\archiverelated\\books\\pdf-images")
            checkpoint_file: Path to checkpoint file for resume

        Returns:
            dict with indexing statistics
        """
        self._load_model()

        if checkpoint_file is None:
            checkpoint_file = os.path.join(os.path.dirname(__file__), f"{index_name}_checkpoint.json")

        # Normalize remap paths
        if remap_from and remap_to:
            remap_from = os.path.normpath(remap_from)
            remap_to = os.path.normpath(remap_to)
            log.info(f"Path remapping: {remap_from} -> {remap_to}")

        def remap_path(local_path: str) -> str:
            if remap_from and remap_to:
                normalized = os.path.normpath(local_path)
                if normalized.startswith(remap_from):
                    return normalized.replace(remap_from, remap_to, 1)
            return local_path

        # Create index if needed
        self.create_index(index_name)

        # Get all book folders
        log.info(f"Scanning {source_path} for book folders...")
        all_books = self._get_book_folders(source_path)
        total_books = len(all_books)
        log.info(f"Found {total_books:,} book folders")

        # Load checkpoint
        completed_books = set()
        if os.path.exists(checkpoint_file):
            log.info("Loading checkpoint...")
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            completed_books = set(checkpoint.get("completed_books", []))
            log.info(f"Resuming: {len(completed_books):,} books already done")

        # Filter to remaining books
        remaining_books = [b for b in all_books if b not in completed_books]
        log.info(f"Books to process: {len(remaining_books):,}")

        # Stats
        total_images = 0
        total_indexed = 0
        errors = 0
        books_done = len(completed_books)

        try:
            for book_name in tqdm(remaining_books, desc="Books", initial=books_done, total=total_books):
                book_path = os.path.join(source_path, book_name)
                images = self._get_images_in_folder(book_path)

                if not images:
                    completed_books.add(book_name)
                    books_done += 1
                    continue

                total_images += len(images)

                # Process in batches
                actions = []
                for i in range(0, len(images), self.batch_size):
                    batch_paths = images[i:i + self.batch_size]
                    embeddings = self._encode_batch(batch_paths)

                    if embeddings is None:
                        errors += len(batch_paths)
                        continue

                    for j, local_path in enumerate(batch_paths):
                        stored_path = remap_path(local_path)
                        filename = os.path.basename(local_path)

                        actions.append({
                            "_op_type": "index",
                            "_index": index_name,
                            "_id": stored_path,  # Use path as ID
                            "_source": {
                                "embedding": embeddings[j].tolist(),
                                "path": stored_path,
                                "filename": filename,
                                "folder": book_name
                            }
                        })

                # Bulk insert for this book
                if actions:
                    try:
                        success, failed = helpers.bulk(
                            self.os_client, actions,
                            raise_on_error=False,
                            refresh=False
                        )
                        total_indexed += success
                        if failed:
                            errors += len(failed)
                    except Exception as e:
                        log.error(f"Bulk insert error for {book_name}: {e}")
                        errors += len(actions)

                # Mark book complete and save checkpoint
                completed_books.add(book_name)
                books_done += 1

                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "completed_books": list(completed_books),
                        "books_done": books_done,
                        "total_indexed": total_indexed,
                        "errors": errors,
                        "last_updated": datetime.now().isoformat()
                    }, f)

        except KeyboardInterrupt:
            log.info("\nInterrupted! Checkpoint saved.")
            return {"status": "interrupted", "books_done": books_done, "indexed": total_indexed}

        # Finalize
        log.info("Finalizing index...")
        self.os_client.indices.put_settings(
            index=index_name,
            body={"index": {"refresh_interval": "1s"}}
        )
        self.os_client.indices.refresh(index=index_name)

        # Cleanup checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        # Get final count
        final_count = self.os_client.count(index=index_name)["count"]

        log.info(f"\n{'='*60}")
        log.info("INDEXING COMPLETE")
        log.info(f"{'='*60}")
        log.info(f"Books processed:     {books_done:,}")
        log.info(f"Images found:        {total_images:,}")
        log.info(f"Indexed to OS:       {total_indexed:,}")
        log.info(f"Errors:              {errors:,}")
        log.info(f"Final index count:   {final_count:,}")
        log.info(f"\nOpenSearch index: {index_name}")

        return {
            "status": "success",
            "books_processed": books_done,
            "images_found": total_images,
            "indexed": total_indexed,
            "errors": errors,
            "final_count": final_count,
            "index_name": index_name
        }


def main():
    parser = argparse.ArgumentParser(
        description="Index CLIP embeddings directly to OpenSearch with per-book checkpointing"
    )
    parser.add_argument("--source", required=True, help="Source directory with book folders")
    parser.add_argument("--index", default="clip-books", help="OpenSearch index name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CLIP encoding")
    parser.add_argument("--remap-from", help="Local path prefix (e.g., D:\\books)")
    parser.add_argument("--remap-to", help="NAS path prefix (e.g., T:\\archiverelated\\books\\pdf-images)")
    parser.add_argument("--host", default="localhost", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source not found: {args.source}")
        return

    indexer = ClipOpenSearchIndexer(
        batch_size=args.batch_size,
        opensearch_host=args.host,
        opensearch_port=args.port
    )

    result = indexer.index(
        source_path=args.source,
        index_name=args.index,
        remap_from=args.remap_from,
        remap_to=args.remap_to
    )

    print(f"\nResult: {result['status']}")


if __name__ == "__main__":
    main()
