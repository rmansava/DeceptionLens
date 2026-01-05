"""
Collections configuration for DinoDeceptionLens.
Defines paths and settings for different content categories (books, print_ads, etc.)
"""
import os

# Collection definitions
# Each collection has both CLIP (FAISS) and DINOv2 (ChromaDB) indexes
COLLECTIONS = {
    "books": {
        "name": "Books",
        "description": "Scanned book pages",
        # CLIP FAISS index paths
        "clip_index": "D:/faiss/books/index.faiss",
        "clip_paths": "D:/faiss/books/paths.json",
        # DINOv2 ChromaDB collection names
        "dino_collection": "images",  # Creates images_visual, images_faces
        "chroma_db_path": "./chroma_db",
        # Source data (for indexing)
        "source_path": None,  # Set when indexing
    },
    "print_ads": {
        "name": "Print Ads",
        "description": "Vintage print advertisements",
        # CLIP FAISS index paths
        "clip_index": "D:/faiss/print_ads/index.faiss",
        "clip_paths": "D:/faiss/print_ads/paths.json",
        # DINOv2 ChromaDB collection names
        "dino_collection": "print_ads",  # Creates print_ads_visual, print_ads_faces
        "chroma_db_path": "./chroma_db",
        # Source data (for indexing)
        "source_path": None,  # Set when indexing
    }
}

# Default collection
DEFAULT_COLLECTION = "books"


def get_collection_config(collection_name: str) -> dict:
    """Get configuration for a specific collection."""
    if collection_name not in COLLECTIONS:
        raise ValueError(f"Unknown collection: {collection_name}. Available: {list(COLLECTIONS.keys())}")
    return COLLECTIONS[collection_name]


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
