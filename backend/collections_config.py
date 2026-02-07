"""
Collections configuration for DinoDeceptionLens.
Defines paths and settings for different content categories (books, print_ads, etc.)
"""
import os

# Collection definitions
# Each collection has CLIP (FAISS), DINOv2 (OpenSearch), and DISK (file) indexes
COLLECTIONS = {
    "books": {
        "name": "Books",
        "description": "Scanned book pages",
        # CLIP FAISS index paths
        "clip_index": "D:/faiss/books/index.faiss",
        "clip_paths": "D:/faiss/books/paths.json",
        # OpenSearch index names
        "opensearch_visual": "dinov2-books",
        "opensearch_faces": "faces-books",
        # DISK features path
        "disk_features": "T:/disk-features/books",
        # DISK search chunks
        "disk_chunks_dir": "S:/faiss/disk_retrieval/chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/chunk_ids",
        # Source data
        "source_path": "T:/archiverelated/books",
    },
    "print_ads": {
        "name": "Print Ads",
        "description": "Vintage print advertisements",
        # CLIP FAISS index paths
        "clip_index": "D:/faiss/printads/index.faiss",
        "clip_paths": "D:/faiss/printads/paths.json",
        # OpenSearch index names
        "opensearch_visual": "dinov2-print_ads",
        "opensearch_faces": "faces-print_ads",
        # DISK features path
        "disk_features": "T:/disk-features/print_ads",
        # DISK search chunks
        "disk_chunks_dir": "S:/faiss/disk_retrieval/printads_chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/printads_chunk_ids",
        # Source data
        "source_path": "T:/archiverelated/print ads",
    },
    "board_games": {
        "name": "Board Games",
        "description": "Board game scans and photos",
        # CLIP FAISS index paths
        "clip_index": "D:/faiss/board_games/index.faiss",
        "clip_paths": "D:/faiss/board_games/paths.json",
        # OpenSearch index names
        "opensearch_visual": "dinov2-board_games",
        "opensearch_faces": "faces-board_games",
        # DISK features path
        "disk_features": "T:/disk-features/board_games",
        # DISK search chunks
        "disk_chunks_dir": "S:/faiss/disk_retrieval/boardgames_chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/boardgames_chunk_ids",
        # Source data
        "source_path": "T:/archiverelated/board games",
    },
    "albums": {
        "name": "Albums",
        "description": "Album art",
        # DISK search chunks
        "disk_chunks_dir": "S:/faiss/disk_retrieval/albums_chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/albums_chunk_ids",
        # Source data
        "source_path": "T:/albums",
    },
    "comics": {
        "name": "Comics",
        "description": "Comic book pages",
        # DISK search chunks
        "disk_chunks_dir": "S:/faiss/disk_retrieval/comics_chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/comics_chunk_ids",
        # Source data
        "source_path": "T:/comics",
    },
}

# Default collection
DEFAULT_COLLECTION = "books"


def get_collection_config(collection_name: str) -> dict:
    """Get configuration for a specific collection."""
    if collection_name not in COLLECTIONS:
        raise ValueError(f"Unknown collection: {collection_name}. Available: {list(COLLECTIONS.keys())}")
    return COLLECTIONS[collection_name]


def get_disk_collections(categories: list = None) -> dict:
    """Get DISK chunk paths for selected categories (or all if None).

    Returns dict of category -> {'chunks_dir': ..., 'ids_dir': ...}
    Only includes categories that have disk_chunks_dir configured.
    """
    result = {}
    for name, config in COLLECTIONS.items():
        if "disk_chunks_dir" not in config:
            continue
        if categories is not None and name not in categories:
            continue
        result[name] = {
            "chunks_dir": config["disk_chunks_dir"],
            "ids_dir": config["disk_chunk_ids_dir"],
        }
    return result


def list_collections() -> list:
    """List all available collections with their status."""
    result = []
    for name, config in COLLECTIONS.items():
        clip_exists = os.path.exists(config["clip_index"])
        result.append({
            "name": name,
            "display_name": config["name"],
            "description": config["description"],
            "clip_index_exists": clip_exists,
            "clip_index_path": config["clip_index"],
        })
    return result
