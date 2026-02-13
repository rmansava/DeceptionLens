#!/usr/bin/env python3
"""
Migrate CLIP embeddings from FAISS to OpenSearch.

This script transfers CLIP embeddings from a FAISS index to OpenSearch,
which provides proper ID-based document storage (no more position/path misalignment).

The FAISS index is NOT deleted - it's kept as a backup.

Prerequisites:
1. Run repair_faiss_paths.py first to fix paths.json alignment
2. OpenSearch running on localhost:9200

Usage:
    python migrate_clip_to_opensearch.py --index D:/faiss/books/index.faiss \
        --paths D:/faiss/books/paths_repaired.json \
        --collection books
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import faiss
import numpy as np
from tqdm import tqdm
from opensearchpy import OpenSearch, helpers

# Logging setup
LOG_FILE = os.path.join(os.path.dirname(__file__), "migrate_clip.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


class ClipMigrator:
    """Migrate CLIP embeddings from FAISS to OpenSearch."""

    def __init__(self, opensearch_host: str = "localhost", opensearch_port: int = 9200):
        self.os_client = OpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            http_compress=True,
            use_ssl=False
        )

    def create_index(self, index_name: str, embedding_dim: int = 768) -> bool:
        """
        Create OpenSearch index for CLIP embeddings.

        Uses the same settings as DINOv2 index for consistency.
        """
        if self.os_client.indices.exists(index=index_name):
            log.warning(f"Index {index_name} already exists!")
            return False

        # k-NN index settings optimized for high recall
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
                            "space_type": "innerproduct",  # Cosine similarity for normalized vectors
                            "engine": "nmslib",
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

    def migrate(self, index_path: str, paths_path: str, index_name: str,
                batch_size: int = 500, skip_nulls: bool = True) -> dict:
        """
        Migrate embeddings from FAISS to OpenSearch.

        Args:
            index_path: Path to index.faiss
            paths_path: Path to paths.json (repaired)
            index_name: OpenSearch index name (e.g., "clip-books")
            batch_size: Documents per bulk insert
            skip_nulls: Skip positions with null paths (deleted files)

        Returns:
            dict with migration statistics
        """
        # Load FAISS index
        log.info(f"Loading FAISS index from {index_path}...")
        faiss_index = faiss.read_index(index_path)
        num_vectors = faiss_index.ntotal
        embedding_dim = faiss_index.d
        log.info(f"FAISS index: {num_vectors:,} vectors, {embedding_dim} dimensions")

        # Load paths
        log.info(f"Loading paths from {paths_path}...")
        with open(paths_path, 'r') as f:
            paths = json.load(f)
        log.info(f"Paths loaded: {len(paths):,} entries")

        # Verify alignment
        if len(paths) != num_vectors:
            log.warning(f"Mismatch: {len(paths):,} paths vs {num_vectors:,} vectors")
            log.warning("Using min of both for migration")

        # Create index if needed
        if not self.os_client.indices.exists(index=index_name):
            self.create_index(index_name, embedding_dim)
        else:
            log.info(f"Index {index_name} exists, will append")

        # Extract all vectors from FAISS
        log.info("Extracting vectors from FAISS...")
        # For IndexFlatIP, we can reconstruct vectors
        all_vectors = faiss_index.reconstruct_n(0, num_vectors)
        log.info(f"Extracted {all_vectors.shape[0]:,} vectors")

        # Migration
        log.info(f"\nMigrating to OpenSearch index: {index_name}")
        log.info(f"Batch size: {batch_size}")

        indexed = 0
        skipped_null = 0
        skipped_exists = 0
        errors = 0
        actions = []

        for i in tqdm(range(min(num_vectors, len(paths))), desc="Migrating"):
            path = paths[i]

            # Skip null entries (deleted files)
            if path is None:
                skipped_null += 1
                continue

            embedding = all_vectors[i]
            filename = os.path.basename(path)
            folder = os.path.basename(os.path.dirname(path))

            action = {
                "_op_type": "index",
                "_index": index_name,
                "_id": path,  # Use path as document ID - this prevents duplicates!
                "_source": {
                    "embedding": embedding.tolist(),
                    "path": path,
                    "filename": filename,
                    "folder": folder
                }
            }
            actions.append(action)

            # Bulk insert
            if len(actions) >= batch_size:
                try:
                    success, failed = helpers.bulk(
                        self.os_client, actions,
                        raise_on_error=False,
                        refresh=False
                    )
                    indexed += success
                    if failed:
                        errors += len(failed)
                except Exception as e:
                    log.error(f"Bulk insert error: {e}")
                    errors += len(actions)
                actions = []

        # Final batch
        if actions:
            try:
                success, failed = helpers.bulk(
                    self.os_client, actions,
                    raise_on_error=False,
                    refresh=False
                )
                indexed += success
                if failed:
                    errors += len(failed)
            except Exception as e:
                log.error(f"Final bulk insert error: {e}")
                errors += len(actions)

        # Re-enable refresh and force refresh
        log.info("Finalizing index...")
        self.os_client.indices.put_settings(
            index=index_name,
            body={"index": {"refresh_interval": "1s"}}
        )
        self.os_client.indices.refresh(index=index_name)

        # Get final count
        final_count = self.os_client.count(index=index_name)["count"]

        # Summary
        log.info(f"\n{'='*60}")
        log.info("MIGRATION COMPLETE")
        log.info(f"{'='*60}")
        log.info(f"FAISS vectors:       {num_vectors:,}")
        log.info(f"Paths in file:       {len(paths):,}")
        log.info(f"Indexed to OS:       {indexed:,}")
        log.info(f"Skipped (null):      {skipped_null:,}")
        log.info(f"Errors:              {errors:,}")
        log.info(f"Final index count:   {final_count:,}")
        log.info(f"\nOpenSearch index: {index_name}")
        log.info("FAISS files NOT deleted (kept as backup)")

        return {
            "status": "success",
            "faiss_vectors": num_vectors,
            "paths_count": len(paths),
            "indexed": indexed,
            "skipped_null": skipped_null,
            "errors": errors,
            "final_count": final_count,
            "index_name": index_name
        }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate CLIP embeddings from FAISS to OpenSearch"
    )
    parser.add_argument("--index", required=True, help="Path to index.faiss")
    parser.add_argument("--paths", required=True, help="Path to paths.json (repaired)")
    parser.add_argument("--collection", required=True,
                        help="Collection name (creates clip-{collection} index)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for bulk indexing (default: 500)")
    parser.add_argument("--host", default="localhost", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")

    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"Error: Index not found: {args.index}")
        return

    if not os.path.exists(args.paths):
        print(f"Error: Paths file not found: {args.paths}")
        return

    index_name = f"clip-{args.collection}"

    migrator = ClipMigrator(args.host, args.port)
    result = migrator.migrate(
        index_path=args.index,
        paths_path=args.paths,
        index_name=index_name,
        batch_size=args.batch_size
    )

    print(f"\nResult: {result['status']}")
    print(f"OpenSearch index '{index_name}' now has {result['final_count']:,} documents")


if __name__ == "__main__":
    main()
