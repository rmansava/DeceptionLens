"""
Collections configuration for DinoDeceptionLens.
Defines paths and settings for different content categories (books, print_ads, etc.)
"""
import os

# Search priority order — books and print_ads first, then others
SEARCH_ORDER = ["board_games", "print_ads", "books", "comics", "albums", "cereal"]

# Collection definitions
# Each collection has CLIP (FAISS), face (OpenSearch), and DISK (file) indexes
COLLECTIONS = {
    "books": {
        "name": "Books",
        "description": "Scanned book pages",
        # CLIP FAISS index paths
        "clip_index": "D:/faiss/books/index.faiss",
        "clip_paths": "D:/faiss/books/paths.json",
        # OpenSearch face index name
        "opensearch_faces": "faces-books",
        # DISK features path
        "disk_features": "T:/disk-features/books",
        # DISK search chunks
        "disk_chunks_dir": "T:/faiss/disk_retrieval/chunks",
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
        # OpenSearch face index name
        "opensearch_faces": "faces-print_ads",
        # DISK features path
        "disk_features": "T:/disk-features/print_ads",
        # DISK search chunks
        "disk_chunks_dir": "T:/faiss/disk_retrieval/printads_chunks",
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
        # OpenSearch face index name
        "opensearch_faces": "faces-board_games",
        # DISK features path
        "disk_features": "T:/disk-features/board_games",
        # DISK search chunks
        "disk_chunks_dir": "T:/faiss/disk_retrieval/boardgames_chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/boardgames_chunk_ids",
        # Source data
        "source_path": "T:/archiverelated/board games",
    },
    "albums": {
        "name": "Albums",
        "description": "Album art",
        # DISK search chunks
        "disk_chunks_dir": "U:/faiss/disk_retrieval/albums_chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/albums_chunk_ids",
        # Source data
        "source_path": "T:/albums",
    },
    "comics": {
        "name": "Comics",
        "description": "Comic book pages",
        # DISK search chunks — split across T: (10GbE, ~1206 chunks) and U: (1GbE, remainder)
        "disk_chunks_dirs": [
            "T:/faiss/disk_retrieval/comics_chunks",
            "U:/faiss/disk_retrieval/comics_chunks",
        ],
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/comics_chunk_ids",
        # Source data
        "source_path": "T:/comics",
    },
    "cereal": {
        "name": "Cereal",
        "description": "Cereal boxes and related imagery",
        # DISK search chunks
        "disk_chunks_dir": "T:/faiss/disk_retrieval/cereal_chunks",
        "disk_chunk_ids_dir": "D:/faiss/disk_retrieval/cereal_chunk_ids",
        # Source data
        "source_path": "T:/archiverelated/cereal",
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

    Returns OrderedDict of category -> {'chunks_dirs': [...], 'ids_dir': ...}
    Only includes categories that have disk_chunks_dir or disk_chunks_dirs configured.
    Categories are returned in SEARCH_ORDER priority.
    For multi-dir collections, chunks are deduplicated (first dir wins).
    """
    result = {}
    # Use SEARCH_ORDER for priority, then any remaining collections
    ordered_names = list(SEARCH_ORDER) + [n for n in COLLECTIONS if n not in SEARCH_ORDER]
    for name in ordered_names:
        if name not in COLLECTIONS:
            continue
        config = COLLECTIONS[name]
        if "disk_chunks_dir" not in config and "disk_chunks_dirs" not in config:
            continue
        if categories is not None and name not in categories:
            continue
        # Support single dir or list of dirs
        if "disk_chunks_dirs" in config:
            chunks_dirs = list(config["disk_chunks_dirs"])
        else:
            chunks_dirs = [config["disk_chunks_dir"]]
        result[name] = {
            "chunks_dirs": chunks_dirs,
            "ids_dir": config["disk_chunk_ids_dir"],
        }
    return result


def list_collections() -> list:
    """List all available collections with their status."""
    result = []
    for name, config in COLLECTIONS.items():
        clip_index = config.get("clip_index")
        clip_exists = os.path.exists(clip_index) if clip_index else False
        entry = {
            "name": name,
            "display_name": config["name"],
            "description": config["description"],
            "clip_index_exists": clip_exists,
        }
        if clip_index:
            entry["clip_index_path"] = clip_index
        result.append(entry)
    return result
