"""Quick test to show where page210 ranks in OpenSearch (NO DISK verification)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opensearch_searcher import OpenSearchSearcher

print("="*70)
print("TEST: Initial OpenSearch ranking for Manglosaurus crop")
print("(NO DISK verification - just DINOv2 embedding similarity)")
print("="*70)

# Initialize OpenSearch searcher only
print("\nInitializing OpenSearch searcher...")
os_searcher = OpenSearchSearcher(visual_index="dinov2-books")
counts = os_searcher.get_counts()
print(f"Index has {counts['visual']} visual embeddings")

query_image = r"D:\trivpics\2023-5.jpg"
print(f"\nQuery image: {query_image}")

# Get 5000 candidates from OpenSearch
print("\nSearching OpenSearch for top 5000 candidates...")
fetch_k = 5000
results = os_searcher.search(query_image, top_k=fetch_k)

print(f"Got {len(results)} results")

# Find target page rank
target_rank = None
target_score = None
for i, r in enumerate(results, 1):
    path_lower = r['path'].lower()
    if "encyclopedia" in path_lower and "monsters" in path_lower and "page210" in path_lower:
        target_rank = i
        target_score = r['score']
        print(f"\n>>> FOUND TARGET: encyclopedia of monsters-page210")
        print(f">>> Rank: #{target_rank} out of {len(results)}")
        print(f">>> Score: {target_score:.4f}")
        break

if not target_rank:
    print("\n>>> TARGET NOT FOUND in top 5000!")

# Show top 10
print("\n" + "="*70)
print("Top 10 by DINOv2 similarity:")
print("-"*70)
print(f"{'Rank':<6} {'Score':<10} {'Filename'}")
print("-"*70)
for i, r in enumerate(results[:10], 1):
    filename = os.path.basename(r['path'])
    print(f"{i:<6} {r['score']:<10.4f} {filename[:55]}")

# Show around target rank if found
if target_rank and target_rank > 10:
    print(f"\n... (ranks 11-{target_rank-3}) ...")
    print("-"*70)
    start = max(0, target_rank - 4)
    end = min(len(results), target_rank + 3)
    for i in range(start, end):
        r = results[i]
        filename = os.path.basename(r['path'])
        rank = i + 1
        marker = " <-- TARGET" if rank == target_rank else ""
        print(f"{rank:<6} {r['score']:<10.4f} {filename[:50]}{marker}")

print("="*70)
