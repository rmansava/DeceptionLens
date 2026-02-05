#!/usr/bin/env python3
"""
Migrate FAISS CLIP indexes to OpenSearch.
Reads existing embeddings from FAISS and bulk-inserts to OpenSearch.
Much faster than re-encoding since embeddings already exist.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm
from opensearchpy import OpenSearch, helpers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def create_index(client: OpenSearch, index_name: str, embedding_dim: int = 768):
    """Create OpenSearch index for CLIP embeddings."""
    if client.indices.exists(index=index_name):
        log.info(f"Index {index_name} already exists")
        return False

    settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1"
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
    client.indices.create(index=index_name, body=settings)
    log.info(f"Index {index_name} created.")
    return True


def migrate_faiss_to_opensearch(
    faiss_path: str,
    paths_path: str,
    index_name: str,
    opensearch_host: str = "localhost",
    opensearch_port: int = 9200,
    batch_size: int = 5000,
    remap_from: str = None,
    remap_to: str = None
):
    """
    Migrate FAISS index to OpenSearch.

    Args:
        faiss_path: Path to index.faiss file
        paths_path: Path to paths.json file
        index_name: OpenSearch index name
        opensearch_host: OpenSearch host
        opensearch_port: OpenSearch port
        batch_size: Bulk insert batch size
        remap_from: Path prefix to replace
        remap_to: New path prefix
    """
    # Load FAISS index
    log.info(f"Loading FAISS index from {faiss_path}...")
    index = faiss.read_index(faiss_path)
    num_vectors = index.ntotal
    dim = index.d
    log.info(f"FAISS index: {num_vectors:,} vectors, {dim} dimensions")

    # Load paths
    log.info(f"Loading paths from {paths_path}...")
    with open(paths_path, 'r') as f:
        paths = json.load(f)
    log.info(f"Loaded {len(paths):,} paths")

    if len(paths) != num_vectors:
        log.warning(f"Path count ({len(paths):,}) != vector count ({num_vectors:,})")
        # Use the smaller count
        count = min(len(paths), num_vectors)
    else:
        count = num_vectors

    # Setup path remapping
    if remap_from and remap_to:
        remap_from = os.path.normpath(remap_from)
        remap_to = os.path.normpath(remap_to)
        log.info(f"Path remapping: {remap_from} -> {remap_to}")

    def remap_path(p: str) -> str:
        if not p:
            return p
        if remap_from and remap_to:
            normalized = os.path.normpath(p)
            if normalized.startswith(remap_from):
                return normalized.replace(remap_from, remap_to, 1)
        return p

    # Connect to OpenSearch
    client = OpenSearch(
        hosts=[{"host": opensearch_host, "port": opensearch_port}],
        http_compress=True,
        use_ssl=False
    )

    # Create index
    create_index(client, index_name, dim)

    # Extract all embeddings from FAISS
    log.info("Extracting embeddings from FAISS...")
    all_embeddings = index.reconstruct_n(0, count)
    log.info(f"Extracted {all_embeddings.shape[0]:,} embeddings")

    # Migrate in batches
    log.info(f"Migrating to OpenSearch index '{index_name}'...")
    total_indexed = 0
    errors = 0

    for start_idx in tqdm(range(0, count, batch_size), desc="Migrating"):
        end_idx = min(start_idx + batch_size, count)
        batch_embeddings = all_embeddings[start_idx:end_idx]
        batch_paths = paths[start_idx:end_idx]

        actions = []
        for i, (emb, path) in enumerate(zip(batch_embeddings, batch_paths)):
            if not path:
                continue

            stored_path = remap_path(path)
            filename = os.path.basename(path)
            folder = os.path.basename(os.path.dirname(path))

            actions.append({
                "_op_type": "index",
                "_index": index_name,
                "_id": stored_path,
                "_source": {
                    "embedding": emb.tolist(),
                    "path": stored_path,
                    "filename": filename,
                    "folder": folder
                }
            })

        if actions:
            try:
                success, failed = helpers.bulk(
                    client, actions,
                    raise_on_error=False,
                    refresh=False
                )
                total_indexed += success
                if failed:
                    errors += len(failed)
            except Exception as e:
                log.error(f"Bulk insert error: {e}")
                errors += len(actions)

    # Finalize
    log.info("Finalizing index...")
    client.indices.put_settings(
        index=index_name,
        body={"index": {"refresh_interval": "1s"}}
    )
    client.indices.refresh(index=index_name)

    final_count = client.count(index=index_name)["count"]

    log.info(f"\n{'='*60}")
    log.info("MIGRATION COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Source vectors:      {count:,}")
    log.info(f"Indexed to OS:       {total_indexed:,}")
    log.info(f"Errors:              {errors:,}")
    log.info(f"Final index count:   {final_count:,}")
    log.info(f"OpenSearch index:    {index_name}")

    return {
        "status": "success",
        "source_count": count,
        "indexed": total_indexed,
        "errors": errors,
        "final_count": final_count
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate FAISS CLIP index to OpenSearch"
    )
    parser.add_argument("--faiss", required=True, help="Path to index.faiss")
    parser.add_argument("--paths", required=True, help="Path to paths.json")
    parser.add_argument("--index", required=True, help="OpenSearch index name")
    parser.add_argument("--batch-size", type=int, default=5000, help="Bulk insert batch size")
    parser.add_argument("--host", default="localhost", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")
    parser.add_argument("--remap-from", help="Path prefix to replace")
    parser.add_argument("--remap-to", help="New path prefix")

    args = parser.parse_args()

    if not os.path.exists(args.faiss):
        print(f"Error: FAISS index not found: {args.faiss}")
        return

    if not os.path.exists(args.paths):
        print(f"Error: Paths file not found: {args.paths}")
        return

    result = migrate_faiss_to_opensearch(
        faiss_path=args.faiss,
        paths_path=args.paths,
        index_name=args.index,
        opensearch_host=args.host,
        opensearch_port=args.port,
        batch_size=args.batch_size,
        remap_from=args.remap_from,
        remap_to=args.remap_to
    )

    print(f"\nResult: {result['status']}")


if __name__ == "__main__":
    main()
