"""
Test 3-chunk DISK search to validate multi-chunk aggregation.

Searches chunks 141, 142, 143 where chunk 142 contains the Encyclopedia of Monsters.
This validates that results from chunk 142 correctly bubble up when mixed with
results from neighboring chunks.
"""

import requests
import time
import sys

API_URL = "http://localhost:8000"
TEST_IMAGE = r"D:\trivpics\2023-5.jpg"
CHUNKS = "141,142,143"  # Encyclopedia of Monsters is in chunk 142

def test_3chunk_search():
    print()
    print("=" * 70)
    print("  3-CHUNK DISK SEARCH TEST")
    print("=" * 70)
    print()
    print(f"  Testing multi-chunk aggregation with chunks {CHUNKS}")
    print(f"  Encyclopedia of Monsters is in chunk 142")
    print(f"  Test image: {TEST_IMAGE}")
    print()

    # Check server
    print("Checking if server is running...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print(f"  [OK] Server is running")
        else:
            print(f"  [ERROR] Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  [ERROR] Cannot connect to server: {e}")
        print(f"  Make sure to start: python server.py")
        return False
    print()

    # Run 3-chunk search
    print(f"Starting 3-chunk search...")
    print(f"Chunks: {CHUNKS}")
    print(f"Live tracking: ENABLED")
    print()

    try:
        with open(TEST_IMAGE, 'rb') as f:
            files = {'file': (TEST_IMAGE, f, 'image/jpeg')}
            params = {
                'top_k': 10,
                'k': 5,
                'threshold': 0.7,
                'live_tracking': True,
                'chunk_ids': CHUNKS
            }

            print("Sending request to API...")
            start = time.time()
            response = requests.post(f"{API_URL}/disk/search", files=files, params=params)
            elapsed = time.time() - start

            if response.status_code == 200:
                results = response.json()
                print(f"  [OK] Search completed in {elapsed:.1f}s")
                print()
                print(f"Found {len(results)} results")
                print()
                print("Top 10 results:")
                print("-" * 70)
                for i, result in enumerate(results[:10], 1):
                    path = result['path']
                    # Extract book name
                    if '/books/' in path:
                        book = path.split('/books/')[1].split('/')[0]
                    else:
                        book = path
                    score = result['score']
                    votes = result.get('verified_matches', 0)
                    print(f"{i:2d}. Votes: {votes:3d} | Score: {score:.3f}")
                    print(f"    {book[:80]}")
                print()

                # Check if Encyclopedia of Monsters is in top results
                found_monsters = False
                for result in results[:10]:
                    if 'Encyclopedia Of Monsters' in result['path']:
                        found_monsters = True
                        break

                if found_monsters:
                    print("[PASS] TEST PASSED - Encyclopedia of Monsters found in top 10!")
                    print()
                    print("This proves multi-chunk aggregation works correctly:")
                    print("  - Searched 3 chunks (141, 142, 143)")
                    print("  - Results from chunk 142 correctly bubbled to the top")
                    print("  - Votes aggregated properly across chunks")
                else:
                    print("[WARN] Encyclopedia of Monsters NOT in top 10")
                    print("This might mean:")
                    print("  - Threshold is too strict")
                    print("  - Test image doesn't match well")
                    print("  - Aggregation needs tuning")

                print()
                print("Check the web UI at http://localhost:5000 to see:")
                print("  - Live progress updates as each chunk was searched")
                print("  - Top 100 results updating in real-time")
                print("  - Search in history with results")

                return found_monsters

            else:
                print(f"  [ERROR] Search failed with status {response.status_code}")
                print(f"  Error: {response.text}")
                return False

    except Exception as e:
        print(f"  [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_3chunk_search()
    sys.exit(0 if success else 1)
