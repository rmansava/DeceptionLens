#!/usr/bin/env python3
"""
Repair FAISS paths.json by matching embeddings.

Problem: paths.json got out of sync with index.faiss due to:
1. add_to_index() using set() which scrambles order
2. delete_from_faiss() removing paths without rebuilding index

Solution: For each image on disk, compute its CLIP embedding and find
its exact match in FAISS (similarity ≈ 1.0). This tells us the correct
position for each path.

After repair, optionally migrate to OpenSearch for proper ID-based indexing.
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
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

# Logging setup
LOG_FILE = os.path.join(os.path.dirname(__file__), "repair_faiss.log")
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


class FaissRepair:
    """Repair FAISS paths.json by matching embeddings to disk files."""

    def __init__(self, index_path: str, paths_path: str, model_name: str = "ViT-L/14"):
        self.index_path = index_path
        self.paths_path = paths_path
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.device}")

        # Will be loaded lazily
        self.model = None
        self.preprocess = None
        self.index = None

    def _load_model(self):
        """Load CLIP model."""
        if self.model is None:
            log.info(f"Loading CLIP model ({self.model_name})...")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            log.info("CLIP model loaded.")

    def _load_index(self):
        """Load FAISS index."""
        if self.index is None:
            log.info(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            log.info(f"Index loaded: {self.index.ntotal:,} vectors")

    def _compute_embedding(self, image_path: str) -> np.ndarray:
        """Compute CLIP embedding for an image."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(img_tensor).float()
                features /= features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().astype('float32')
        except Exception as e:
            log.warning(f"Failed to process {image_path}: {e}")
            return None

    def _find_exact_match(self, embedding: np.ndarray, threshold: float = 0.9999) -> int:
        """
        Find the FAISS position where this embedding exists.
        Returns -1 if no exact match found.
        """
        # Search for top 1 match
        distances, indices = self.index.search(embedding, 1)

        similarity = distances[0][0]
        position = indices[0][0]

        # For normalized vectors with IndexFlatIP, similarity of 1.0 means exact match
        if similarity >= threshold:
            return int(position)

        return -1

    def scan_images(self, source_path: str) -> list:
        """Scan directory for all image files."""
        log.info(f"Scanning {source_path} for images...")
        image_paths = []

        for root, _, files in os.walk(source_path):
            for file in files:
                if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, file))

        log.info(f"Found {len(image_paths):,} images on disk")
        return image_paths

    def repair(self, source_path: str, output_path: str = None,
               batch_size: int = 100, checkpoint_interval: int = 1000,
               remap_from: str = None, remap_to: str = None) -> dict:
        """
        Repair paths.json by matching disk images to FAISS positions.

        Args:
            source_path: Root directory containing images
            output_path: Where to save repaired paths.json (default: paths_repaired.json)
            batch_size: Images to process before progress update
            checkpoint_interval: Save checkpoint every N images
            remap_from: Local path prefix to replace (e.g., "D:\\books")
            remap_to: NAS path prefix to use instead (e.g., "T:\\archiverelated\\books\\pdf-images")

        Returns:
            dict with repair statistics
        """
        self._load_model()
        self._load_index()

        if output_path is None:
            output_path = self.paths_path.replace(".json", "_repaired.json")

        checkpoint_file = output_path.replace(".json", "_checkpoint.json")

        # Normalize remap paths
        if remap_from and remap_to:
            remap_from = os.path.normpath(remap_from)
            remap_to = os.path.normpath(remap_to)
            log.info(f"Path remapping: {remap_from} -> {remap_to}")

        def remap_path(local_path: str) -> str:
            """Convert local path to stored path format."""
            if remap_from and remap_to:
                normalized = os.path.normpath(local_path)
                if normalized.startswith(remap_from):
                    return normalized.replace(remap_from, remap_to, 1)
            return local_path

        # Get all images on disk
        all_images = self.scan_images(source_path)
        total_images = len(all_images)

        # Initialize position map (position -> path)
        # Using dict since positions may be sparse during repair
        position_map = {}

        # Load checkpoint if exists
        start_idx = 0
        if os.path.exists(checkpoint_file):
            log.info("Loading checkpoint...")
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            position_map = {int(k): v for k, v in checkpoint.get("position_map", {}).items()}
            start_idx = checkpoint.get("processed", 0)
            log.info(f"Resuming from image {start_idx:,}, {len(position_map):,} positions mapped")

        # Stats
        matched = len(position_map)
        not_found = 0
        errors = 0
        duplicates = 0

        log.info(f"\nRepairing paths.json...")
        log.info(f"FAISS index has {self.index.ntotal:,} vectors")
        log.info(f"Disk has {total_images:,} images")
        log.info(f"Starting from image {start_idx:,}")

        try:
            for i, image_path in enumerate(tqdm(all_images[start_idx:],
                                                 initial=start_idx,
                                                 total=total_images,
                                                 desc="Matching embeddings")):
                current_idx = start_idx + i

                # Compute embedding
                embedding = self._compute_embedding(image_path)
                if embedding is None:
                    errors += 1
                    continue

                # Find exact match in FAISS
                position = self._find_exact_match(embedding)

                if position >= 0:
                    stored_path = remap_path(image_path)
                    if position in position_map:
                        # Position already claimed by another path
                        existing = position_map[position]
                        if existing != stored_path:
                            log.warning(f"Duplicate position {position}: {existing} vs {stored_path}")
                            duplicates += 1
                    else:
                        position_map[position] = stored_path
                        matched += 1
                else:
                    not_found += 1
                    if not_found <= 10:  # Log first 10
                        log.warning(f"No match found for: {image_path}")

                # Checkpoint
                if (current_idx + 1) % checkpoint_interval == 0:
                    log.info(f"\nCheckpoint at {current_idx + 1:,}: {matched:,} matched, {not_found:,} not found")
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            "processed": current_idx + 1,
                            "position_map": position_map,
                            "matched": matched,
                            "not_found": not_found,
                            "errors": errors
                        }, f)

        except KeyboardInterrupt:
            log.info("\nInterrupted! Saving checkpoint...")
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "processed": current_idx + 1,
                    "position_map": position_map,
                    "matched": matched,
                    "not_found": not_found,
                    "errors": errors
                }, f)
            log.info(f"Checkpoint saved. Run again to resume.")
            return {"status": "interrupted", "matched": matched}

        # Convert position_map to ordered list
        log.info("\nBuilding final paths list...")
        max_position = max(position_map.keys()) if position_map else 0

        # Create list with None for missing positions
        repaired_paths = [None] * (max_position + 1)
        for pos, path in position_map.items():
            repaired_paths[pos] = path

        # Count gaps (positions in FAISS with no matching disk file)
        gaps = sum(1 for p in repaired_paths if p is None)

        # Save repaired paths
        log.info(f"Saving repaired paths to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(repaired_paths, f)

        # Cleanup checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        # Summary
        log.info(f"\n{'='*60}")
        log.info("REPAIR COMPLETE")
        log.info(f"{'='*60}")
        log.info(f"FAISS vectors:     {self.index.ntotal:,}")
        log.info(f"Images on disk:    {total_images:,}")
        log.info(f"Matched:           {matched:,}")
        log.info(f"Not in FAISS:      {not_found:,}")
        log.info(f"Gaps (no file):    {gaps:,}")
        log.info(f"Duplicates:        {duplicates:,}")
        log.info(f"Errors:            {errors:,}")
        log.info(f"\nRepaired paths saved to: {output_path}")

        if not_found > 0:
            log.info(f"\n{not_found:,} images on disk are NOT in FAISS - they need to be indexed")

        if gaps > 0:
            log.info(f"\n{gaps:,} FAISS positions have no matching file - those files were deleted")

        return {
            "status": "success",
            "faiss_vectors": self.index.ntotal,
            "disk_images": total_images,
            "matched": matched,
            "not_in_faiss": not_found,
            "gaps": gaps,
            "duplicates": duplicates,
            "errors": errors,
            "output_path": output_path
        }


def main():
    parser = argparse.ArgumentParser(
        description="Repair FAISS paths.json by matching embeddings to disk files"
    )
    parser.add_argument("--index", required=True, help="Path to index.faiss")
    parser.add_argument("--paths", required=True, help="Path to paths.json (will create _repaired.json)")
    parser.add_argument("--source", required=True, help="Source directory with images")
    parser.add_argument("--output", help="Output path for repaired paths.json")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save checkpoint every N images (default: 1000)")
    parser.add_argument("--remap-from", help="Local path prefix to replace (e.g., D:\\books)")
    parser.add_argument("--remap-to", help="NAS path prefix to use instead (e.g., T:\\archiverelated\\books\\pdf-images)")

    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(f"Error: Index not found: {args.index}")
        return

    if not os.path.exists(args.source):
        print(f"Error: Source directory not found: {args.source}")
        return

    repairer = FaissRepair(args.index, args.paths)
    result = repairer.repair(
        source_path=args.source,
        output_path=args.output,
        checkpoint_interval=args.checkpoint_interval,
        remap_from=args.remap_from,
        remap_to=args.remap_to
    )

    print(f"\nResult: {result['status']}")


if __name__ == "__main__":
    main()
