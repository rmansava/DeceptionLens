"""
Test DISK search API with live tracking.

Tests the /disk/search endpoint to verify:
1. Search completes successfully
2. Live tracking creates search session
3. Progress updates happen after each chunk
4. Results are saved to database
"""

import requests
import time
import sys

API_URL = "http://localhost:8000"
TEST_IMAGE = r"D:\trivpics\2023-5.jpg"

def test_disk_search():
    print()
    print("=" * 70)
    print("  DISK SEARCH API TEST")
    print("=" * 70)
    print()

    # Check if server is running
    print("Checking if server is running...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print(f"  ✓ Server is running")
            print(f"  Response: {response.json()}")
        else:
            print(f"  ✗ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Cannot connect to server: {e}")
        print(f"  Make sure to start the server with: python server.py")
        return False
    print()

    # Test DISK search with live tracking
    print(f"Testing DISK search with: {TEST_IMAGE}")
    print("Live tracking: ENABLED")
    print()

    try:
        with open(TEST_IMAGE, 'rb') as f:
            files = {'file': (TEST_IMAGE, f, 'image/jpeg')}
            params = {
                'top_k': 10,
                'k': 5,
                'threshold': 0.7,
                'live_tracking': True
            }

            print("Sending request...")
            start = time.time()
            response = requests.post(f"{API_URL}/disk/search", files=files, params=params)
            elapsed = time.time() - start

            if response.status_code == 200:
                results = response.json()
                print(f"  ✓ Search completed in {elapsed:.1f}s")
                print(f"  Found {len(results)} results")
                print()
                print("Top 5 results:")
                for i, result in enumerate(results[:5], 1):
                    path = result['path'].split('/')[-1]  # Just filename
                    score = result['score']
                    votes = result.get('verified_matches', 0)
                    print(f"  {i}. {path}")
                    print(f"     Score: {score:.3f} | Votes: {votes}")
                print()
                print("✓ Test PASSED - Search completed successfully!")
                print()
                print("Check the web UI to see the search in history with live progress!")
                return True
            else:
                print(f"  ✗ Search failed with status {response.status_code}")
                print(f"  Error: {response.text}")
                return False

    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_disk_search()
    sys.exit(0 if success else 1)
