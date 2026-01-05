"""
Test CLIP + ORB + Template matching re-ranking pipeline.
Tests the dinosaur image (Manglosaurus) to verify it finds Encyclopedia of Monsters page210.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clip_searcher import ClipSearcher
import time

# Target to find
TARGET_PAGE = "encyclopedia of monsters-page210"
QUERY_IMAGE = r"D:\trivpics\2023-5.jpg"

print("="*80)
print("TEST: CLIP + ORB + Template Matching Re-ranking Pipeline")
print("="*80)
print(f"\nQuery: {QUERY_IMAGE}")
print(f"Target: {TARGET_PAGE}")
print()

# Initialize
print("Initializing CLIP searcher...")
searcher = ClipSearcher()

# Get stats
stats = searcher.get_stats()
print(f"Index: {stats['total_images']:,} images")
print()

# ============================================================================
# TEST 1: Raw CLIP search (no re-ranking)
# ============================================================================
print("="*80)
print("TEST 1: Raw CLIP search (no re-ranking)")
print("="*80)

start = time.time()
raw_results = searcher.search_by_image(QUERY_IMAGE, top_k=100)
elapsed = time.time() - start

print(f"Searched in {elapsed:.2f}s")

# Find target
raw_rank = None
for i, r in enumerate(raw_results, 1):
    if TARGET_PAGE in r['path'].lower():
        raw_rank = i
        print(f"\n>>> TARGET FOUND at rank #{i}")
        print(f"    Score: {r['score']:.4f}")
        break

if not raw_rank:
    print(f"\n>>> TARGET NOT FOUND in top 100")

print(f"\nTop 10 (raw CLIP):")
print("-"*80)
for i, r in enumerate(raw_results[:10], 1):
    filename = os.path.basename(r['path'])
    marker = " <-- TARGET" if TARGET_PAGE in r['path'].lower() else ""
    print(f"{i:3}. [{r['score']:.4f}] {filename[:55]}{marker}")

# ============================================================================
# TEST 2: With ORB + Template re-ranking
# ============================================================================
print("\n" + "="*80)
print("TEST 2: CLIP + ORB + Template Matching Re-ranking")
print("="*80)

start = time.time()
reranked_results = searcher.search_with_rerank(
    QUERY_IMAGE,
    top_k=50,
    retrieval_k=20000,  # Get 20K CLIP candidates
    rerank_k=1000,      # Template match on top 1K by keypoints
    verbose=True
)
elapsed = time.time() - start

print(f"\nTotal time: {elapsed:.2f}s")

# Find target
rerank_rank = None
for i, r in enumerate(reranked_results, 1):
    if TARGET_PAGE in r['path'].lower():
        rerank_rank = i
        print(f"\n>>> TARGET FOUND at rank #{i}")
        print(f"    CLIP Score: {r['score']:.4f}")
        print(f"    Keypoints: {r.get('keypoint_matches', 0)}")
        print(f"    Template: {r.get('template_score', 0):.4f}")
        print(f"    Combined: {r.get('combined_score', 0):.2f}")
        break

if not rerank_rank:
    print(f"\n>>> TARGET NOT FOUND in top 50")

print(f"\nTop 20 (re-ranked):")
print("-"*80)
print(f"{'Rank':<5} {'CLIP':<8} {'KP':<5} {'Tmpl':<8} {'Combined':<10} {'Filename'}")
print("-"*80)

for i, r in enumerate(reranked_results[:20], 1):
    filename = os.path.basename(r['path'])
    marker = " <-- TARGET" if TARGET_PAGE in r['path'].lower() else ""
    print(f"{i:<5} {r['score']:<8.4f} {r.get('keypoint_matches', 0):<5} "
          f"{r.get('template_score', 0):<8.4f} {r.get('combined_score', 0):<10.2f} "
          f"{filename[:40]}{marker}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Query: Cropped Manglosaurus dinosaur image")
print(f"Target: {TARGET_PAGE}")
print(f"Index size: {stats['total_images']:,} images")
print()
print(f"Raw CLIP rank: {'#' + str(raw_rank) if raw_rank else 'NOT IN TOP 100'}")
print(f"Re-ranked rank: {'#' + str(rerank_rank) if rerank_rank else 'NOT IN TOP 50'}")

if raw_rank and rerank_rank:
    if rerank_rank < raw_rank:
        print(f"\n>>> IMPROVEMENT: #{raw_rank} -> #{rerank_rank}")
    else:
        print(f"\n>>> No improvement (raw was better)")
elif rerank_rank and not raw_rank:
    print(f"\n>>> TARGET FOUND with re-ranking (was not in top 100 raw)")

print("="*80)
