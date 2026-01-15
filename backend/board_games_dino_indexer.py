"""
Board Games DINOv2 Indexer for DeceptionLens
Creates DINOv2 embeddings in OpenSearch for board game images.
Uses SQL Server to track image hashes and skip duplicates.

Usage:
    python board_games_dino_indexer.py --source "T:/archiverelated/board games"

    # Visual only (skip faces)
    python board_games_dino_indexer.py --source "T:/archiverelated/board games" --visual-only

    # Faces only (skip visual)
    python board_games_dino_indexer.py --source "T:/archiverelated/board games" --faces-only
"""
import os
import sys
import argparse
import hashlib
import gc
from pathlib import Path
from tqdm import tqdm

# Database connection string
CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=rmdesk;"
    "DATABASE=DeceptionLens;"
    "Trusted_Connection=yes;"
)

# OpenSearch settings
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
VISUAL_INDEX = "dinov2-board_games"
FACES_INDEX = "faces-board_games"


def get_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_hash_exists(cursor, file_hash: str, collection: str) -> str:
    """Check if hash exists in DB. Returns existing path or None."""
    cursor.execute(
        "SELECT FilePath FROM ImageHashes WHERE FileHash = ? AND Collection = ?",
        (file_hash, collection)
    )
    row = cursor.fetchone()
    return row[0] if row else None


def add_hash_to_db(cursor, file_hash: str, file_path: str, collection: str, file_size: int):
    """Add a new hash to the database."""
    cursor.execute(
        """INSERT INTO ImageHashes (FileHash, FilePath, Collection, FileSize)
           VALUES (?, ?, ?, ?)""",
        (file_hash, file_path, collection, file_size)
    )


def scan_for_images(root_path: str) -> list:
    """Recursively scan for image files."""
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_paths = []

    for root, dirs, files in os.walk(root_path):
        for f in sorted(files):
            if Path(f).suffix.lower() in extensions:
                image_paths.append(os.path.join(root, f))

    image_paths.sort()
    return image_paths


def create_opensearch_indices(client):
    """Create OpenSearch indices if they don't exist."""
    # DINOv2 visual index (768 dimensions)
    visual_mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                },
                "path": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "folder": {"type": "keyword"}
            }
        }
    }

    # Face index (512 dimensions for ArcFace)
    faces_mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 512,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                },
                "path": {"type": "keyword"},
                "source_image": {"type": "keyword"},
                "face_index": {"type": "integer"},
                "folder": {"type": "keyword"}
            }
        }
    }

    if not client.indices.exists(index=VISUAL_INDEX):
        print(f"Creating index: {VISUAL_INDEX}")
        client.indices.create(index=VISUAL_INDEX, body=visual_mapping)

    if not client.indices.exists(index=FACES_INDEX):
        print(f"Creating index: {FACES_INDEX}")
        client.indices.create(index=FACES_INDEX, body=faces_mapping)


def index_dino_opensearch(
    source_path: str,
    collection: str = "board_games",
    enable_visual: bool = True,
    enable_faces: bool = False,
    batch_size: int = 100,
    no_dedup: bool = False
):
    """
    Index images using DINOv2 into OpenSearch with duplicate detection.
    """
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from opensearchpy import OpenSearch, helpers
    import cv2

    # Connect to SQL Server (for hash lookup) - skip if no_dedup
    conn = None
    cursor = None
    if not no_dedup:
        import pyodbc
        print("Connecting to SQL Server...")
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
    else:
        print("Skipping deduplication (--no-dedup mode)")

    # Connect to OpenSearch
    print("Connecting to OpenSearch...")
    os_client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        timeout=60
    )

    # Create indices if needed
    create_opensearch_indices(os_client)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DINOv2
    processor = None
    model = None
    if enable_visual:
        print("Loading DINOv2 model (facebook/dinov2-base)...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        model.eval()
        print("DINOv2 loaded.")

    # Load InsightFace
    face_app = None
    if enable_faces:
        try:
            from insightface.app import FaceAnalysis
            print("Loading InsightFace (buffalo_l / ArcFace)...")
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace loaded (DirectML).")
        except ImportError:
            print("InsightFace not installed. Skipping face indexing.")
            enable_faces = False

    # Scan for images
    print(f"\nScanning {source_path}...")
    all_image_paths = scan_for_images(source_path)
    total_images = len(all_image_paths)
    print(f"Found {total_images:,} images")

    if total_images == 0:
        print("No images found!")
        return

    # Filter out duplicates
    dino_collection = f"{collection}_dino"

    new_images = []
    skipped_duplicates = 0
    skipped_errors = 0

    if no_dedup:
        print("\nPreparing all images (no dedup)...")
        for path in all_image_paths:
            file_size = os.path.getsize(path)
            new_images.append((path, None, file_size))
        print(f"  Images to index: {len(new_images):,}")
    else:
        print("\nChecking for duplicates...")
        for path in tqdm(all_image_paths, desc="Hashing"):
            try:
                file_hash = get_file_hash(path)
                existing = check_hash_exists(cursor, file_hash, dino_collection)

                if existing:
                    skipped_duplicates += 1
                else:
                    file_size = os.path.getsize(path)
                    new_images.append((path, file_hash, file_size))
            except Exception:
                skipped_errors += 1
                continue

        print(f"\nDuplicate check complete:")
        print(f"  New images to index: {len(new_images):,}")
        print(f"  Skipped (duplicates): {skipped_duplicates:,}")
        print(f"  Skipped (errors): {skipped_errors:,}")

    if not new_images:
        print("No new images to index!")
        if conn:
            conn.close()
        return

    # Index images
    visual_actions = []
    face_actions = []
    visual_count = 0
    face_count = 0
    errors = 0
    hashes_to_add = []

    for file_path, file_hash, file_size in tqdm(new_images, desc="DINOv2 Indexing"):
        folder_name = os.path.basename(os.path.dirname(file_path))
        indexed_this_image = False

        # Visual embedding
        if enable_visual and model:
            try:
                image = Image.open(file_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                visual_actions.append({
                    "_index": VISUAL_INDEX,
                    "_id": file_path,
                    "_source": {
                        "embedding": embedding.tolist(),
                        "path": file_path,
                        "filename": os.path.basename(file_path),
                        "folder": folder_name
                    }
                })
                visual_count += 1
                indexed_this_image = True
            except Exception as e:
                errors += 1

        # Face embeddings
        if enable_faces and face_app:
            try:
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    faces = face_app.get(img)
                    for i, face in enumerate(faces):
                        face_id = f"{file_path}_face_{i}"
                        face_actions.append({
                            "_index": FACES_INDEX,
                            "_id": face_id,
                            "_source": {
                                "embedding": face.embedding.tolist(),
                                "path": file_path,
                                "source_image": file_path,
                                "face_index": i,
                                "folder": folder_name
                            }
                        })
                        face_count += 1
                        indexed_this_image = True
            except Exception:
                pass

        # Track hash for successfully indexed images
        if indexed_this_image and not no_dedup and file_hash:
            hashes_to_add.append((file_hash, file_path, dino_collection, file_size))

        # Bulk insert visual
        if len(visual_actions) >= batch_size:
            try:
                helpers.bulk(os_client, visual_actions, refresh=False)
            except Exception as e:
                print(f"Bulk insert error: {e}")
            visual_actions = []

        # Bulk insert faces
        if len(face_actions) >= batch_size:
            try:
                helpers.bulk(os_client, face_actions, refresh=False)
            except Exception as e:
                print(f"Face bulk insert error: {e}")
            face_actions = []

        # Commit hashes to DB periodically
        if not no_dedup and len(hashes_to_add) >= batch_size:
            for h in hashes_to_add:
                try:
                    add_hash_to_db(cursor, h[0], h[1], h[2], h[3])
                except Exception:
                    pass
            conn.commit()
            hashes_to_add = []

        gc.collect()

    # Insert remaining
    if visual_actions:
        try:
            helpers.bulk(os_client, visual_actions, refresh=False)
        except Exception as e:
            print(f"Final bulk insert error: {e}")

    if face_actions:
        try:
            helpers.bulk(os_client, face_actions, refresh=False)
        except Exception as e:
            print(f"Final face bulk insert error: {e}")

    # Commit remaining hashes
    if not no_dedup and hashes_to_add:
        for h in hashes_to_add:
            try:
                add_hash_to_db(cursor, h[0], h[1], h[2], h[3])
            except Exception:
                pass
        conn.commit()

    # Refresh indices
    if enable_visual:
        os_client.indices.refresh(index=VISUAL_INDEX)
    if enable_faces:
        os_client.indices.refresh(index=FACES_INDEX)

    if conn:
        conn.close()

    # Get final counts
    vis_total = os_client.count(index=VISUAL_INDEX)["count"] if enable_visual else 0
    face_total = os_client.count(index=FACES_INDEX)["count"] if enable_faces else 0

    print(f"\n{'='*60}")
    print(f"BOARD GAMES DINO INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Visual embeddings added: {visual_count:,}")
    print(f"Face embeddings added: {face_count:,}")
    print(f"Errors: {errors:,}")
    print(f"Total in {VISUAL_INDEX}: {vis_total:,}")
    print(f"Total in {FACES_INDEX}: {face_total:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Board Games DINOv2 Indexer (OpenSearch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index visual only (recommended first)
    python board_games_dino_indexer.py --source "T:/archiverelated/board games" --visual-only

    # Index faces only (run separately due to GPU memory)
    python board_games_dino_indexer.py --source "T:/archiverelated/board games" --faces-only

    # Index both
    python board_games_dino_indexer.py --source "T:/archiverelated/board games"

    # Skip SQL Server deduplication
    python board_games_dino_indexer.py --source "T:/archiverelated/board games" --no-dedup
        """
    )

    parser.add_argument("--source", required=True, help="Source directory with images")
    parser.add_argument("--collection", default="board_games", help="Collection name (default: board_games)")
    parser.add_argument("--visual-only", action="store_true", help="Only index DINOv2 visual")
    parser.add_argument("--faces-only", action="store_true", help="Only index faces")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size (default: 100)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip SQL Server deduplication")

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source path does not exist: {args.source}")
        sys.exit(1)

    enable_visual = not args.faces_only
    enable_faces = not args.visual_only

    if args.visual_only and args.faces_only:
        print("Error: Cannot specify both --visual-only and --faces-only")
        sys.exit(1)

    index_dino_opensearch(
        source_path=args.source,
        collection=args.collection,
        enable_visual=enable_visual,
        enable_faces=enable_faces,
        batch_size=args.batch_size,
        no_dedup=args.no_dedup
    )


if __name__ == "__main__":
    main()
