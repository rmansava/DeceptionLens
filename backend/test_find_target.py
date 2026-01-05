"""Find where the target page actually ranks."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opensearch_searcher import OpenSearchSearcher

print("Initializing OpenSearch...")
os_searcher = OpenSearchSearcher(visual_index="dinov2-books")
counts = os_searcher.get_counts()
print(f"Index has {counts['visual']} visual embeddings")

query_image = r"D:\trivpics\2023-5.jpg"
print(f"\nQuery: {query_image}")

# Try fetching more candidates
for fetch_k in [5000, 10000, 20000, 50000]:
    print(f"\nSearching top {fetch_k}...")
    results = os_searcher.search(query_image, top_k=fetch_k)

    for i, r in enumerate(results, 1):
        path_lower = r['path'].lower()
        if "encyclopedia" in path_lower and "monsters" in path_lower and "page210" in path_lower:
            print(f"  >>> FOUND at rank #{i} with score {r['score']:.4f}")
            print(f"  >>> Path: {r['path']}")
            break
    else:
        print(f"  Not in top {len(results)}")
