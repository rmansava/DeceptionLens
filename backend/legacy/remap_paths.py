#!/usr/bin/env python3
"""
Remap paths in paths.json from local D:\ to NAS T:\ format.
Run this after clip_indexer.py finishes.
"""
import json
import os

INPUT_PATH = "D:/faiss/books_new/paths.json"
OUTPUT_PATH = "D:/faiss/books_new/paths_remapped.json"

REMAP_FROM = "D:\\books"
REMAP_TO = "T:\\archiverelated\\books\\pdf-images"

def main():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, 'r') as f:
        paths = json.load(f)

    print(f"Loaded {len(paths):,} paths")
    print(f"Remapping: {REMAP_FROM} -> {REMAP_TO}")

    remapped = []
    for p in paths:
        if p:
            normalized = os.path.normpath(p)
            if normalized.startswith(REMAP_FROM):
                remapped.append(normalized.replace(REMAP_FROM, REMAP_TO, 1))
            else:
                remapped.append(p)
        else:
            remapped.append(p)

    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(remapped, f)

    print(f"Done! {len(remapped):,} paths remapped")
    print(f"\nTo use the new index:")
    print(f"  1. Backup old: move D:\\faiss\\books D:\\faiss\\books_old")
    print(f"  2. Rename new: move D:\\faiss\\books_new D:\\faiss\\books")
    print(f"  3. Use remapped: copy D:\\faiss\\books\\paths_remapped.json D:\\faiss\\books\\paths.json")

if __name__ == "__main__":
    main()
