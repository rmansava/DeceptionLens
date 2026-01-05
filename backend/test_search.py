"""Quick test script for the Manglosaurus search."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from searcher import DinoSearcher

print("="*60)
print("TEST SEARCH: Manglosaurus crop -> Encyclopedia of Monsters")
print("="*60)

# Initialize searcher (loads DINOv2, DISK, LightGlue)
searcher = DinoSearcher(db_path="./chroma_db")

# Test query
query_image = r"D:\trivpics\2023-5.jpg"
print(f"\nQuery image: {query_image}")
print("\nRunning search with verify=True (LightGlue geometric verification)...")
print("-"*60)

# Search with verification
results = searcher.search(
    query_path=query_image,
    top_k=20,
    verify=True,
    collection_name="books"
)

print(f"\nTop {len(results)} results:")
print("-"*60)
print(f"{'Rank':<5} {'Score':<8} {'Matches':<8} {'Filename'}")
print("-"*60)

for i, r in enumerate(results, 1):
    filename = os.path.basename(r['path'])
    score = r['score']
    matches = r['verified_matches']

    # Highlight the expected match
    highlight = " <-- EXPECTED MATCH" if "encyclopedia of monsters" in filename.lower() and "page210" in filename else ""
    print(f"{i:<5} {score:<8.4f} {matches:<8} {filename[:60]}{highlight}")

print("-"*60)
print("Done.")
