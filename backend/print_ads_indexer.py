"""
Print Ads Indexer for DinoDeceptionLens
Creates CLIP (FAISS) indexes for print advertisements.
Uses SQL Server to track image hashes and skip duplicates.

Usage:
    python print_ads_indexer.py --source "D:/PrintAds" --output "D:/faiss/print_ads"

    # Resume after interruption (uses hash DB to skip already indexed)
    python print_ads_indexer.py --source "D:/PrintAds" --output "D:/faiss/print_ads" --resume
"""
import os
import sys
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm

# Database connection string (same as db_helper.py)
CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=DeceptionLens;"
    "Trusted_Connection=yes;"
)


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


def index_with_dedup(
    source_path: str,
    output_dir: str,
    collection: str = "print_ads",
    batch_size: int = 64,
    resume: bool = False
):
    """
    Index images using CLIP with duplicate detection via SQL Server.

    Args:
        source_path: Directory containing images
        output_dir: Output directory for FAISS index
        collection: Collection name for hash DB
        batch_size: Batch size for CLIP encoding
        resume: If True, skip images already in hash DB
    """
    import pyodbc
    import torch
    import clip
    import faiss
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    # Connect to SQL Server
    print("Connecting to SQL Server...")
    conn = pyodbc.connect(CONNECTION_STRING)
    cursor = conn.cursor()

    # Setup CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model (ViT-L/14) on {device}...")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    # Scan for images
    print(f"\nScanning {source_path}...")
    all_image_paths = scan_for_images(source_path)
    total_images = len(all_image_paths)
    print(f"Found {total_images:,} images")

    if total_images == 0:
        print("No images found!")
        return

    # Filter out duplicates
    print("\nChecking for duplicates...")
    new_images = []
    skipped_duplicates = 0
    skipped_errors = 0

    for path in tqdm(all_image_paths, desc="Hashing"):
        try:
            file_hash = get_file_hash(path)
            existing = check_hash_exists(cursor, file_hash, collection)

            if existing:
                skipped_duplicates += 1
            else:
                file_size = os.path.getsize(path)
                new_images.append((path, file_hash, file_size))
        except Exception as e:
            skipped_errors += 1
            continue

    print(f"\nDuplicate check complete:")
    print(f"  New images to index: {len(new_images):,}")
    print(f"  Skipped (duplicates): {skipped_duplicates:,}")
    print(f"  Skipped (errors): {skipped_errors:,}")

    if not new_images:
        print("No new images to index!")
        conn.close()
        return

    # Encode images
    print(f"\nEncoding {len(new_images):,} images...")
    all_embeddings = []
    indexed_paths = []

    total_batches = (len(new_images) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="Encoding"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(new_images))
        batch_items = new_images[batch_start:batch_end]

        batch_images = []
        batch_paths = []
        batch_hashes = []
        batch_sizes = []

        for path, file_hash, file_size in batch_items:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(preprocess(img))
                batch_paths.append(path)
                batch_hashes.append(file_hash)
                batch_sizes.append(file_size)
            except Exception:
                # Use placeholder for failed loads
                blank = Image.new("RGB", (224, 224), (128, 128, 128))
                batch_images.append(preprocess(blank))
                batch_paths.append(path)
                batch_hashes.append(file_hash)
                batch_sizes.append(file_size)

        if not batch_images:
            continue

        # Encode batch
        image_input = torch.stack(batch_images).to(device)
        with torch.no_grad():
            features = model.encode_image(image_input).float()
            features /= features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())
        indexed_paths.extend(batch_paths)

        # Add hashes to DB
        for path, file_hash, file_size in zip(batch_paths, batch_hashes, batch_sizes):
            try:
                add_hash_to_db(cursor, file_hash, path, collection, file_size)
            except Exception:
                pass  # Ignore duplicate key errors

        # Commit every 10 batches
        if (batch_idx + 1) % 10 == 0:
            conn.commit()

    conn.commit()

    # Build FAISS index
    print("\nBuilding FAISS index...")
    all_embeddings = np.vstack(all_embeddings)

    # Check if we're adding to existing index
    index_path = os.path.join(output_dir, "index.faiss")
    paths_path = os.path.join(output_dir, "paths.json")

    if resume and os.path.exists(index_path):
        print("Loading existing index to append...")
        import json
        index = faiss.read_index(index_path)
        with open(paths_path) as f:
            existing_paths = json.load(f)

        index.add(all_embeddings.astype('float32'))
        all_paths = existing_paths + indexed_paths
    else:
        index = faiss.IndexFlatIP(all_embeddings.shape[1])
        index.add(all_embeddings.astype('float32'))
        all_paths = indexed_paths

    # Save
    print("Saving index...")
    faiss.write_index(index, index_path)

    import json
    with open(paths_path, "w") as f:
        json.dump(all_paths, f)

    conn.close()

    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"New images indexed: {len(indexed_paths):,}")
    print(f"Total in index: {index.ntotal:,}")
    print(f"Index file: {index_path}")
    print(f"Paths file: {paths_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Print Ads Indexer with Duplicate Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index a folder
    python print_ads_indexer.py --source "D:/PrintAds" --output "D:/faiss/print_ads"

    # Resume/add more images (skips already indexed)
    python print_ads_indexer.py --source "D:/MoreAds" --output "D:/faiss/print_ads" --resume

    # Different collection name
    python print_ads_indexer.py --source "D:/Magazines" --output "D:/faiss/magazines" --collection magazines
        """
    )

    parser.add_argument("--source", required=True, help="Source directory with images")
    parser.add_argument("--output", required=True, help="Output directory for FAISS index")
    parser.add_argument("--collection", default="print_ads", help="Collection name (default: print_ads)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--resume", action="store_true", help="Resume/append to existing index")

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source path does not exist: {args.source}")
        sys.exit(1)

    index_with_dedup(
        source_path=args.source,
        output_dir=args.output,
        collection=args.collection,
        batch_size=args.batch_size,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
