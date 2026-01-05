"""Test script to show pre/post verification rankings for Manglosaurus search.
Uses OpenSearch for initial search (books collection) + DISK/LightGlue verification.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opensearch_searcher import OpenSearchSearcher
from searcher import DinoSearcher

TARGET_PAGE = "encyclopedia of monsters-page210"

print("="*70)
print("TEST: Manglosaurus crop -> Encyclopedia of Monsters page210")
print("="*70)

# Initialize searchers
print("\nInitializing OpenSearch searcher (for DINOv2 k-NN search)...")
os_searcher = OpenSearchSearcher(visual_index="dinov2-books")
counts = os_searcher.get_counts()
print(f"OpenSearch index: {counts['visual']} visual embeddings, {counts['faces']} face embeddings")

print("\nInitializing DinoSearcher (for DISK + LightGlue verification)...")
dino_searcher = DinoSearcher(db_path="./chroma_db")

query_image = r"D:\trivpics\2023-5.jpg"
print(f"\nQuery image: {query_image}")

# STEP 1: Search OpenSearch WITHOUT verification to get initial DINOv2 ranking
print("\n" + "="*70)
print("STEP 1: Initial DINOv2/OpenSearch ranking (NO geometric verification)")
print("="*70)

# Get 5000 candidates from OpenSearch (same as server.py uses)
fetch_k = 5000
results_no_verify = os_searcher.search(query_image, top_k=fetch_k)

# Find target page rank
initial_rank = None
initial_score = None
for i, r in enumerate(results_no_verify, 1):
    path_lower = r['path'].lower()
    if "encyclopedia" in path_lower and "monsters" in path_lower and "page210" in path_lower:
        initial_rank = i
        initial_score = r['score']
        break

print(f"\nTotal candidates from OpenSearch: {len(results_no_verify)}")
print(f"\n>>> Target page (page210) initial rank: #{initial_rank} out of {len(results_no_verify)}")
print(f">>> Target page initial score: {initial_score:.4f}" if initial_score else ">>> Target not found!")

print("\nTop 10 by DINOv2 similarity (before verification):")
print("-"*70)
print(f"{'Rank':<6} {'Score':<10} {'Filename'}")
print("-"*70)
for i, r in enumerate(results_no_verify[:10], 1):
    filename = os.path.basename(r['path'])
    highlight = " <-- TARGET" if TARGET_PAGE in filename.lower() else ""
    print(f"{i:<6} {r['score']:<10.4f} {filename[:55]}{highlight}")

# STEP 2: Run DISK + LightGlue verification on all candidates
print("\n" + "="*70)
print("STEP 2: Running DISK + LightGlue geometric verification on all candidates")
print("="*70)

# Use DinoSearcher's _verify_matches (same as server.py does)
verified_results = dino_searcher._verify_matches(query_image, results_no_verify)

# Sort by (verified_matches, score) descending
verified_results.sort(key=lambda x: (x['verified_matches'], x['score']), reverse=True)

# Get top 20
top_results = verified_results[:20]

# Find target page in verified results
final_rank = None
final_score = None
final_matches = None
for i, r in enumerate(top_results, 1):
    path_lower = r['path'].lower()
    if "encyclopedia" in path_lower and "monsters" in path_lower and "page210" in path_lower:
        final_rank = i
        final_score = r['score']
        final_matches = r['verified_matches']
        break

print(f"\nTop 20 results after re-ranking by keypoint matches:")
print("-"*70)
print(f"{'Rank':<6} {'Score':<10} {'Matches':<10} {'Filename'}")
print("-"*70)
for i, r in enumerate(top_results, 1):
    filename = os.path.basename(r['path'])
    matches = r['verified_matches']
    path_lower = r['path'].lower()
    highlight = " <-- TARGET" if ("encyclopedia" in path_lower and "monsters" in path_lower and "page210" in path_lower) else ""
    print(f"{i:<6} {r['score']:<10.4f} {matches:<10} {filename[:50]}{highlight}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Query: Cropped Manglosaurus dinosaur image")
print(f"Target: encyclopedia of monsters-page210.jpg (full page with 3 dinosaur figures)")
print(f"")
print(f"BEFORE verification (DINOv2/OpenSearch only):")
print(f"  Rank: #{initial_rank} out of {len(results_no_verify)}")
print(f"  Score: {initial_score:.4f}" if initial_score else "  Not found")
print(f"")
print(f"AFTER verification (DISK + LightGlue):")
if final_rank:
    print(f"  Rank: #{final_rank}")
    print(f"  Score: {final_score:.4f}")
    print(f"  Keypoint matches: {final_matches}")
else:
    print(f"  Not in top 20")
print(f"")
if initial_rank and final_rank:
    print(f">>> RANK IMPROVEMENT: #{initial_rank} -> #{final_rank}")
print("="*70)
