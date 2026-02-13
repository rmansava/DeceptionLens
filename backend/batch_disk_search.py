"""
Batch DISK Search - search a directory of images against all DISK chunks.

Standalone CLI tool. Each query image gets its own row in search history,
viewable in the web UI while the batch is still running.

Key optimization: loads each ~10GB chunk ONCE and searches ALL query images
against it. Total time ≈ single-image search time.

Usage:
    python batch_disk_search.py <directory> [--top_k 50] [--k 5] [--threshold 0.7] [--collections books,print_ads]
"""

import os
import sys
import time
import argparse
import logging

# Set up path so we can import from the backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from disk_searcher import search_disk_batch, get_total_chunks
from db_helper import create_search_session, update_search_progress, complete_search_session

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "batch_search.log")

# Log to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}


def find_images(directory):
    """Find all images in directory (recursive)."""
    images = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))
    return images


def main():
    parser = argparse.ArgumentParser(description="Batch DISK keypoint search")
    parser.add_argument("directory", help="Directory containing query images")
    parser.add_argument("--top_k", type=int, default=50, help="Results per image (default: 50)")
    parser.add_argument("--k", type=int, default=5, help="Nearest neighbors per keypoint (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum similarity (default: 0.7)")
    parser.add_argument("--collections", type=str, default=None, help="Comma-separated collections (default: all)")
    args = parser.parse_args()

    directory = args.directory
    if not os.path.isdir(directory):
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    # Parse collections
    categories = None
    if args.collections:
        categories = [c.strip() for c in args.collections.split(',') if c.strip()]
        if not categories:
            categories = None

    cat_label = ",".join(categories) if categories else "all"

    # Find images
    image_paths = find_images(directory)
    if not image_paths:
        print(f"No images found in {directory}")
        sys.exit(1)

    # Count chunks
    total_chunks = get_total_chunks(categories)

    print("=" * 70)
    print("  BATCH DISK SEARCH")
    print("=" * 70)
    print(f"  Directory:    {directory}")
    print(f"  Images:       {len(image_paths)}")
    print(f"  Collections:  {cat_label}")
    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Parameters:   top_k={args.top_k}, k={args.k}, threshold={args.threshold}")
    print("=" * 70)
    print()

    # Read all images and create search sessions
    print("Reading images and creating search sessions...")
    image_list = []       # [(image_bytes, image_name), ...]
    search_ids = {}       # {image_name: search_id}

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        image_list.append((image_bytes, image_name))

        # Create search session in DB
        search_id = create_search_session(
            search_type="DISK Keypoint (Batch)",
            query_image=image_bytes,
            query_image_name=image_name,
            collection=cat_label,
            total_chunks=total_chunks
        )
        search_ids[image_name] = search_id
        print(f"  [{i+1}/{len(image_paths)}] {image_name} -> session #{search_id}")

    print(f"\nAll {len(image_list)} sessions created. Results visible in web UI now.")
    print()

    # Progress callback: updates all search sessions + console
    start_time = time.time()

    def progress_callback(chunk_idx, total_chunks, per_image_results, elapsed_ms):
        # Update each image's search session in DB
        for image_name, results in per_image_results.items():
            if image_name in search_ids:
                try:
                    update_search_progress(
                        search_ids[image_name],
                        chunk_idx,
                        total_chunks,
                        results,
                        elapsed_ms
                    )
                except Exception as e:
                    pass  # Don't let DB errors stop the search

        # Console progress
        elapsed = time.time() - start_time
        pct = chunk_idx / total_chunks * 100
        avg = elapsed / chunk_idx if chunk_idx > 0 else 0
        eta = (total_chunks - chunk_idx) * avg
        eta_h = eta / 3600
        eta_m = eta / 60

        # Build vote summary (top vote per image)
        vote_parts = []
        for image_name, results in per_image_results.items():
            short_name = image_name[:12]
            top_votes = results[0]['votes'] if results else 0
            vote_parts.append(f"{short_name}: {top_votes}")

        vote_summary = " | ".join(vote_parts[:5])  # Show up to 5 images
        if len(vote_parts) > 5:
            vote_summary += f" | +{len(vote_parts)-5} more"

        if eta_h >= 1:
            eta_str = f"{eta_h:.1f}h"
        else:
            eta_str = f"{eta_m:.0f}m"

        print(f"\r  [{chunk_idx}/{total_chunks}] {pct:5.1f}% | "
              f"ETA: {eta_str} | {vote_summary}     ",
              end="", flush=True)

    # Run batch search
    print("Searching...")
    results = search_disk_batch(
        image_list,
        top_k=args.top_k,
        k=args.k,
        threshold=args.threshold,
        categories=categories,
        progress_callback=progress_callback
    )
    print()  # Newline after progress

    # Complete all search sessions
    duration_ms = int((time.time() - start_time) * 1000)
    print(f"\nCompleting search sessions...")
    for image_name, search_id in search_ids.items():
        try:
            complete_search_session(search_id, duration_ms)
        except Exception as e:
            print(f"  Warning: Failed to complete session #{search_id}: {e}")

    # Final update: write final results to each session
    for image_name, image_results in results.items():
        if image_name in search_ids:
            try:
                update_search_progress(
                    search_ids[image_name],
                    total_chunks,
                    total_chunks,
                    image_results,
                    duration_ms
                )
            except Exception:
                pass

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("  BATCH SEARCH COMPLETE!")
    print("=" * 70)
    print(f"  Images:       {len(image_list)}")
    print(f"  Chunks:       {total_chunks:,}")
    print(f"  Time:         {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print()

    # Per-image results
    print("  Results per image:")
    print("  " + "-" * 66)
    for image_name, image_results in results.items():
        sid = search_ids.get(image_name, "?")
        if image_results:
            top = image_results[0]
            top_path = os.path.basename(top['path'])
            if len(top_path) > 40:
                top_path = top_path[:37] + "..."
            print(f"  #{sid:<5} {image_name:<20} {top['votes']:>5} votes  {top_path}")
        else:
            print(f"  #{sid:<5} {image_name:<20} no matches")

    print("  " + "-" * 66)
    print(f"\n  View detailed results in the web UI search history.")
    print("=" * 70)


if __name__ == '__main__':
    main()
