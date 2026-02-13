"""
Rebuild chunk_1011_ids.npy from LOCAL data only (no NAS).

Strategy: Read chunks 1-1010 from D: drive (local SSD) to find exactly
where chunk 1010 ends in the book sequence. Then read only the few
remaining books from NAS to build chunk 1011.

Actually even simpler: chunks 1-1010 are each 21M vectors. We know the
exact sequence of compact_ids across all chunks. Chunk 1011 is just the
continuation. We can read chunk 1010 to find the last book being processed,
then figure out the remaining vectors.

Fastest approach: read ALL _ids.npy files locally, concatenate, and the
leftover after 1010*21M is chunk 1011's content.
"""
import os
import numpy as np
import time
import sys

CHUNK_IDS_DIR = r"D:\faiss\disk_retrieval\chunk_ids"
MAX_VECTORS_PER_CHUNK = 21_000_000
TARGET_CHUNK = 1011
TOTAL_GOOD_CHUNKS = 1010  # chunks 1-1010 are correct


def main():
    print(f"Rebuilding chunk_{TARGET_CHUNK:03d}_ids.npy (LOCAL data only)")
    print()

    ids_file = os.path.join(CHUNK_IDS_DIR, f"chunk_{TARGET_CHUNK:03d}_ids.npy")
    old_ids = np.load(ids_file)
    print(f"  Current chunk_{TARGET_CHUNK:03d}_ids.npy: {len(old_ids):,} entries (corrupt)")
    print()

    # The key insight: chunks are filled sequentially during consolidation.
    # Chunks 1-1010 each have exactly 21M vectors. The IDs in each chunk
    # file tell us the compact_id for each vector position.
    #
    # To rebuild chunk 1011, we need to know what comes AFTER chunk 1010.
    # The consolidation processes books in sorted order, filling chunks.
    # So chunk 1011 = the leftover vectors after 1010 full chunks.
    #
    # But we don't need to replay from scratch. We can read chunk 1010
    # to see which book it ends on, find where that book continues, and
    # then process the remaining books from NAS.
    #
    # EVEN SIMPLER: The consolidation state + path_lookup tells us the
    # full vector sequence. Let's just compute it.

    # Step 1: Read chunk 1010 to find the last compact_id
    print("  Reading chunk 1010 to find continuation point...")
    start = time.time()
    chunk_1010 = np.load(os.path.join(CHUNK_IDS_DIR, "chunk_1010_ids.npy"))
    last_id_in_1010 = chunk_1010[-1]
    print(f"  Chunk 1010: {len(chunk_1010):,} vectors, last compact_id = {last_id_in_1010}")

    # Count how many vectors of the last ID are at the end of chunk 1010
    # (the book may span the chunk boundary)
    tail_count = 0
    for i in range(len(chunk_1010) - 1, -1, -1):
        if chunk_1010[i] == last_id_in_1010:
            tail_count += 1
        else:
            break
    print(f"  Last ID {last_id_in_1010} has {tail_count:,} vectors at end of chunk 1010")

    # Step 2: Load path_lookup to get total vectors per compact_id
    # We need to know the total vectors for the book that spans the boundary
    import json
    lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    print("  Loading path_lookup.json...")
    with open(lookup_file) as f:
        id_to_path = json.load(f)
    total_paths = len(id_to_path)
    print(f"  {total_paths:,} total paths (compact_ids 0 to {total_paths-1})")

    # Step 3: Load consolidation state to get book processing order
    state_file = r"D:\faiss\disk_retrieval\consolidation_state.json"
    with open(state_file) as f:
        state = json.load(f)
    processed_books = sorted(state.get('processed_books', []))
    print(f"  {len(processed_books):,} books in consolidation order")

    # Step 4: We need per-book vector counts. We can get this by reading
    # each book's paths.json... but that's on NAS (slow).
    #
    # Alternative: read the _ids.npy files to count vectors per compact_id.
    # But reading 1010 files × 21M × 4 bytes = 84GB is also slow.
    #
    # Best alternative: use the DB! We have run-length encoded data.
    print()
    print("  Querying DB for vector counts per compact_id...")
    import pyodbc
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=trivia;Trusted_Connection=yes"
    )
    cursor = conn.cursor()

    # Get total vectors per compact_id across all books chunks
    # This sums VectorCount grouped by CompactId across all chunks
    cursor.execute("""
        SELECT CompactId, SUM(CAST(VectorCount AS BIGINT)) as TotalVectors
        FROM DiskChunkMap
        WHERE Collection = 'books'
        GROUP BY CompactId
        ORDER BY CompactId
    """)
    id_vector_counts = {}
    for row in cursor.fetchall():
        id_vector_counts[row[0]] = row[1]
    print(f"  Got vector counts for {len(id_vector_counts):,} compact_ids from DB")

    # Step 5: Replay the consolidation using just the vector counts
    # Process books in sorted order, tracking chunk boundaries
    print()
    print("  Replaying consolidation with local vector counts...")

    # Build path_to_id mapping
    path_to_id = {}
    for idx, path in enumerate(id_to_path):
        # Extract book name from path
        path_to_id[path] = idx

    # For each book, we need to know its compact_id and vector count.
    # The compact_id for a book = path_to_id[any path from that book]
    # Since all paths from a book share the same compact_id... wait, no.
    # Each IMAGE gets its own compact_id. A book has many images, each with
    # a different compact_id. The vectors for one image all share that image's
    # compact_id.
    #
    # So the consolidation adds ALL vectors from book1's images, then book2's, etc.
    # For each book, it iterates its paths.json which lists all image paths.
    # Each image path maps to a compact_id via path_to_id.
    # The number of vectors for that image = id_vector_counts[compact_id].
    #
    # But we don't know which compact_ids belong to which book without
    # reading paths.json from NAS...
    #
    # HOWEVER: the books are processed in sorted order, and compact_ids are
    # assigned incrementally. So book1 gets ids 0..N1-1, book2 gets N1..N1+N2-1, etc.
    # This means compact_ids are in sorted book order!
    #
    # So we can just iterate compact_ids 0, 1, 2, ... and sum their vector counts.
    # When we've counted 1010 * 21M vectors, the remainder is chunk 1011.

    vectors_before_1011 = TOTAL_GOOD_CHUNKS * MAX_VECTORS_PER_CHUNK
    print(f"  Vectors in chunks 1-1010: {vectors_before_1011:,}")

    # Count vectors by iterating compact_ids in order
    running_total = 0
    split_id = None
    split_offset = 0  # how many vectors of split_id are in chunks 1-1010

    for cid in range(total_paths):
        vec_count = id_vector_counts.get(cid, 0)
        if vec_count == 0:
            continue

        if running_total + vec_count <= vectors_before_1011:
            running_total += vec_count
        else:
            # This compact_id spans the boundary
            split_id = cid
            split_offset = vectors_before_1011 - running_total
            running_total = vectors_before_1011
            break

    print(f"  Boundary at compact_id {split_id}: {split_offset:,} vectors in chunk 1010, rest in 1011")

    # Step 6: Build chunk 1011's IDs
    # It starts with the remaining vectors from split_id, then continues
    # with all subsequent compact_ids
    chunk_1011_ids = []

    # Remaining vectors from the split book
    remaining = id_vector_counts.get(split_id, 0) - split_offset
    chunk_1011_ids.extend([split_id] * remaining)

    # Continue with subsequent compact_ids
    for cid in range(split_id + 1, total_paths):
        vec_count = id_vector_counts.get(cid, 0)
        if vec_count == 0:
            continue
        chunk_1011_ids.extend([cid] * vec_count)

    print(f"  Chunk 1011: {len(chunk_1011_ids):,} entries")
    print(f"  ID range: {chunk_1011_ids[0]} - {chunk_1011_ids[-1]}")

    # Verify against existing corrupt file
    overlap = min(len(old_ids), len(chunk_1011_ids))
    if overlap > 0:
        old_arr = old_ids[:overlap]
        new_arr = np.array(chunk_1011_ids[:overlap], dtype=np.int32)
        match = np.array_equal(old_arr, new_arr)
        print(f"  First {overlap:,} entries match existing corrupt file: {match}")
        if not match:
            # Find first mismatch
            for i in range(overlap):
                if old_arr[i] != new_arr[i]:
                    print(f"  First mismatch at position {i}: old={old_arr[i]}, new={new_arr[i]}")
                    break

    # Save
    ids_array = np.array(chunk_1011_ids, dtype=np.int32)
    np.save(ids_file, ids_array)
    saved = np.load(ids_file)
    print(f"\n  Saved: {len(saved):,} entries, ID range {saved.min()} - {saved.max()}")
    print(f"  File size: {os.path.getsize(ids_file) / (1024*1024):.1f} MB")
    print(f"  Done in {time.time()-start:.1f}s")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
