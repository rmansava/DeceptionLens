"""
Search DISK Per-Book Shards

Searches all book indexes and merges votes to find matching images.
"""

import faiss
import numpy as np
import json
import os
from glob import glob
from collections import Counter
import time

# Config
NAS_INDEX = "T:/faiss/disk_retrieval/books"
LOCAL_INDEX = "C:/temp/disk-retrieval-index"  # Check here too for pending copies


def get_all_book_indexes():
    """Get all book index directories (NAS + local pending)."""
    books = []

    # NAS indexes
    if os.path.exists(NAS_INDEX):
        for book in os.listdir(NAS_INDEX):
            idx_path = os.path.join(NAS_INDEX, book, "index.faiss")
            if os.path.exists(idx_path):
                books.append((book, NAS_INDEX))

    # Local pending indexes
    if os.path.exists(LOCAL_INDEX):
        for book in os.listdir(LOCAL_INDEX):
            idx_path = os.path.join(LOCAL_INDEX, book, "index.faiss")
            if os.path.exists(idx_path):
                # Don't duplicate if already in NAS
                if not any(b[0] == book for b in books):
                    books.append((book, LOCAL_INDEX))

    return books


def search_single_book(book_name, base_dir, query_descriptors, k=5, threshold=0.7):
    """Search a single book's index, return votes per image."""
    idx_path = os.path.join(base_dir, book_name, "index.faiss")
    paths_path = os.path.join(base_dir, book_name, "paths.json")

    try:
        index = faiss.read_index(idx_path)
        with open(paths_path, 'r') as f:
            paths = json.load(f)

        # Search
        distances, indices = index.search(query_descriptors, k)

        # Count votes
        votes = Counter()
        for i in range(len(query_descriptors)):
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0 and distances[i][j] >= threshold:
                    votes[paths[idx]] += 1

        return votes

    except Exception as e:
        return Counter()


def search_all_books(query_descriptors, k=5, threshold=0.7, top_n=20):
    """Search all book indexes, merge votes, return top results."""
    books = get_all_book_indexes()
    print(f"Searching {len(books)} book indexes...")

    all_votes = Counter()

    t0 = time.time()
    for i, (book_name, base_dir) in enumerate(books):
        votes = search_single_book(book_name, base_dir, query_descriptors, k, threshold)
        all_votes.update(votes)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(books)} books searched...")

    elapsed = time.time() - t0
    print(f"Search complete in {elapsed:.1f}s ({len(books)/elapsed:.1f} books/sec)")

    # Return top results
    return all_votes.most_common(top_n)


def extract_query_descriptors(image_path):
    """Extract DISK descriptors from query image."""
    import torch
    from PIL import Image
    import kornia.feature as KF

    # Load and pad image
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    new_w = ((w + 15) // 16) * 16
    new_h = ((h + 15) // 16) * 16
    padded = Image.new('RGB', (new_w, new_h), (0, 0, 0))
    padded.paste(img, (0, 0))

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(padded)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = img_tensor.to(device)

    # Extract DISK features
    extractor = KF.DISK.from_pretrained('depth').eval().to(device)
    with torch.no_grad():
        feats = extractor(img_tensor)
        descriptors = feats[0].descriptors.cpu().numpy()

    # Normalize
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = (descriptors / (norms + 1e-8)).astype('float32')

    print(f"Extracted {len(descriptors)} keypoints from query image")
    return descriptors


def search_by_image(image_path, k=5, threshold=0.7, top_n=20):
    """Search all indexes using an image file."""
    print(f"\nQuery: {image_path}")
    print("-" * 60)

    # Extract query descriptors
    descriptors = extract_query_descriptors(image_path)

    # Search all books
    results = search_all_books(descriptors, k=k, threshold=threshold, top_n=top_n)

    # Display results
    print()
    print("Top matches:")
    print("-" * 60)
    for i, (path, votes) in enumerate(results):
        print(f"  {i+1:2d}. {votes:5d} votes: {path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default test
        test_image = "D:/trivpics/2023-5.jpg"
        if os.path.exists(test_image):
            search_by_image(test_image)
        else:
            print("Usage: python search_disk_shards.py <image_path>")
            print()
            books = get_all_book_indexes()
            print(f"Available: {len(books)} book indexes")
    else:
        search_by_image(sys.argv[1])
