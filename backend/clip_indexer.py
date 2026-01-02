"""
CLIP Indexer for DinoDeceptionLens
Creates FAISS indexes from image folders using CLIP embeddings.
Supports checkpoint/resume for crash recovery.
"""
import os
import json
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path


class ClipIndexer:
    """
    Indexes images using CLIP embeddings and stores in FAISS.
    Supports batch processing with checkpoint/resume.
    """

    def __init__(self, model_name: str = "ViT-L/14", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint_interval = 100  # Save every N batches

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CLIP Indexer using device: {self.device}")

        # Lazy load model
        self.model = None
        self.preprocess = None

    def _ensure_model_loaded(self):
        """Lazy load CLIP model."""
        if self.model is None:
            print(f"Loading CLIP model ({self.model_name})...")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print("CLIP model loaded.")

    def index_folder(self, root_path: str, output_dir: str) -> dict:
        """
        Index all images in a folder recursively.

        Args:
            root_path: Root folder to scan for images
            output_dir: Directory to store index.faiss and paths.json

        Returns:
            dict with indexing stats
        """
        self._ensure_model_loaded()

        # Setup paths
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
        embeddings_file = os.path.join(checkpoint_dir, "embeddings.npy")

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Scan for images
        print(f"\nScanning {root_path} recursively...")
        all_image_paths = self._scan_for_images(root_path)
        total_images = len(all_image_paths)

        print(f"Found {total_images:,} images")

        if total_images == 0:
            return {"status": "error", "message": "No images found"}

        # Check for checkpoint
        start_batch, all_embeddings = self._load_checkpoint(
            checkpoint_file, embeddings_file, all_image_paths
        )

        # Calculate batches
        total_batches = (total_images + self.batch_size - 1) // self.batch_size

        if start_batch < total_batches:
            # Encode remaining images
            print(f"\nEncoding images (batch {start_batch + 1} to {total_batches})...")

            for batch_idx in tqdm(range(start_batch, total_batches),
                                   desc="Encoding", initial=start_batch, total=total_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, total_images)
                batch_paths = all_image_paths[batch_start:batch_end]

                batch_embeddings = self._encode_batch(batch_paths)
                if batch_embeddings is not None:
                    all_embeddings.append(batch_embeddings)

                # Save checkpoint
                if (batch_idx + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(
                        checkpoint_file, embeddings_file, all_embeddings,
                        batch_idx + 1, all_image_paths[:batch_end]
                    )

        # Build final index
        print("\nBuilding FAISS index...")
        all_embeddings = np.vstack(all_embeddings)

        index = faiss.IndexFlatIP(all_embeddings.shape[1])
        index.add(all_embeddings.astype('float32'))

        # Save
        index_path = os.path.join(output_dir, "index.faiss")
        paths_path = os.path.join(output_dir, "paths.json")

        faiss.write_index(index, index_path)
        with open(paths_path, "w") as f:
            json.dump(all_image_paths, f)

        # Cleanup checkpoints
        self._cleanup_checkpoints(checkpoint_dir, checkpoint_file, embeddings_file)

        print(f"\nIndexing complete: {len(all_image_paths):,} images")

        return {
            "status": "success",
            "total_images": len(all_image_paths),
            "index_path": index_path,
            "paths_path": paths_path,
            "embedding_dim": all_embeddings.shape[1]
        }

    def add_to_index(self, index_dir: str, new_folder: str) -> dict:
        """
        Add images from a new folder to an existing index.

        Args:
            index_dir: Directory with existing index.faiss and paths.json
            new_folder: Folder containing new images to add

        Returns:
            dict with update stats
        """
        self._ensure_model_loaded()

        index_path = os.path.join(index_dir, "index.faiss")
        paths_path = os.path.join(index_dir, "paths.json")

        if not os.path.exists(index_path):
            return {"status": "error", "message": "Index not found"}

        # Load existing
        print("Loading existing index...")
        index = faiss.read_index(index_path)
        with open(paths_path) as f:
            existing_paths = set(json.load(f))

        print(f"Existing index: {index.ntotal:,} images")

        # Scan new folder
        print(f"\nScanning {new_folder}...")
        new_paths = self._scan_for_images(new_folder)

        # Filter out already indexed
        new_paths = [p for p in new_paths if p not in existing_paths]
        print(f"New images to add: {len(new_paths):,}")

        if not new_paths:
            return {"status": "success", "added": 0, "message": "No new images"}

        # Encode new images
        print("\nEncoding new images...")
        all_embeddings = []

        total_batches = (len(new_paths) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(total_batches), desc="Encoding"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(new_paths))
            batch_paths = new_paths[batch_start:batch_end]

            batch_embeddings = self._encode_batch(batch_paths)
            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)

        # Add to index
        new_embeddings = np.vstack(all_embeddings)
        index.add(new_embeddings.astype('float32'))

        # Update paths
        all_paths = list(existing_paths) + new_paths

        # Save
        faiss.write_index(index, index_path)
        with open(paths_path, "w") as f:
            json.dump(all_paths, f)

        print(f"\nAdded {len(new_paths):,} images. Total: {index.ntotal:,}")

        return {
            "status": "success",
            "added": len(new_paths),
            "total": index.ntotal,
            "index_path": index_path
        }

    def _scan_for_images(self, root_path: str) -> list:
        """Recursively scan for image files."""
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_paths = []

        for root, dirs, files in os.walk(root_path):
            for f in sorted(files):
                if Path(f).suffix.lower() in extensions:
                    image_paths.append(os.path.join(root, f))

        image_paths.sort()
        return image_paths

    def _encode_batch(self, paths: list) -> np.ndarray:
        """Encode a batch of images."""
        batch_images = []

        for path in paths:
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

    def _load_checkpoint(self, checkpoint_file: str, embeddings_file: str,
                         all_paths: list) -> tuple:
        """Load checkpoint if valid."""
        if not os.path.exists(checkpoint_file):
            return 0, []

        print("\nCheckpoint found, validating...")

        with open(checkpoint_file) as f:
            checkpoint = json.load(f)

        checkpoint_paths = checkpoint.get("paths", [])
        start_batch = checkpoint.get("batch", 0)

        # Verify paths match
        if checkpoint_paths == all_paths[:len(checkpoint_paths)]:
            if os.path.exists(embeddings_file):
                embeddings = [np.load(embeddings_file)]
                print(f"Resuming from batch {start_batch} ({embeddings[0].shape[0]:,} embeddings)")
                return start_batch, embeddings

        print("Checkpoint invalid, starting fresh")
        return 0, []

    def _save_checkpoint(self, checkpoint_file: str, embeddings_file: str,
                         embeddings: list, batch: int, paths: list):
        """Save indexing checkpoint."""
        print(f"\n[Saving checkpoint at batch {batch}...]")

        combined = np.vstack(embeddings)
        np.save(embeddings_file, combined)

        with open(checkpoint_file, "w") as f:
            json.dump({
                "batch": batch,
                "paths": paths,
                "model": self.model_name
            }, f)

    def _cleanup_checkpoints(self, checkpoint_dir: str, checkpoint_file: str,
                             embeddings_file: str):
        """Remove checkpoint files."""
        for f in [embeddings_file, checkpoint_file]:
            if os.path.exists(f):
                os.remove(f)
        try:
            os.rmdir(checkpoint_dir)
        except:
            pass


def main():
    """CLI interface for CLIP indexing."""
    import argparse

    parser = argparse.ArgumentParser(description="CLIP Image Indexer")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a folder")
    index_parser.add_argument("--root", required=True, help="Folder to index")
    index_parser.add_argument("--output", required=True, help="Output directory")
    index_parser.add_argument("--batch-size", type=int, default=64)

    # Add command
    add_parser = subparsers.add_parser("add", help="Add to existing index")
    add_parser.add_argument("--index-dir", required=True, help="Existing index directory")
    add_parser.add_argument("--folder", required=True, help="New folder to add")

    args = parser.parse_args()

    indexer = ClipIndexer(batch_size=getattr(args, 'batch_size', 64))

    if args.command == "index":
        indexer.index_folder(args.root, args.output)
    elif args.command == "add":
        indexer.add_to_index(args.index_dir, args.folder)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
