"""
Store and rebuild DISK chunk ID mappings in SQL Server.

Tables:
  DiskPathLookup  - compact ID -> image path (per collection)
  DiskChunkMap    - run-length encoded vector->ID mapping per chunk

This replaces replaying consolidation from per-book shards.
Any chunk's _ids.npy can be rebuilt with a single query.
"""

import os
import sys
import json
import time
import numpy as np
from glob import glob
from itertools import groupby
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_helper import get_connection
from collections_config import COLLECTIONS

logger = logging.getLogger(__name__)


def create_tables():
    """Create the DiskPathLookup and DiskChunkMap tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='DiskPathLookup' AND xtype='U')
        CREATE TABLE DiskPathLookup (
            Collection      NVARCHAR(50)   NOT NULL,
            CompactId       INT            NOT NULL,
            ImagePath       NVARCHAR(500)  NOT NULL,
            PRIMARY KEY (Collection, CompactId)
        )
    """)

    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='DiskChunkMap' AND xtype='U')
        CREATE TABLE DiskChunkMap (
            Collection      NVARCHAR(50)   NOT NULL,
            ChunkNumber     INT            NOT NULL,
            SequenceOrder   INT            NOT NULL,
            CompactId       INT            NOT NULL,
            VectorCount     INT            NOT NULL,
            PRIMARY KEY (Collection, ChunkNumber, SequenceOrder)
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("  Tables created/verified.")


def sync_chunk_to_db(collection_name, chunk_number, ids_array):
    """Sync a chunk's ID mapping to the DB after saving to disk.
    Non-fatal: logs warning on failure so builds aren't interrupted."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM DiskChunkMap WHERE Collection = ? AND ChunkNumber = ?",
            (collection_name, chunk_number))
        conn.commit()

        batch = []
        seq = 0
        for compact_id, group in groupby(ids_array.tolist()):
            count = sum(1 for _ in group)
            batch.append((collection_name, chunk_number, seq, int(compact_id), count))
            seq += 1
            if len(batch) >= 5000:
                cursor.executemany(
                    "INSERT INTO DiskChunkMap (Collection, ChunkNumber, SequenceOrder, CompactId, VectorCount) VALUES (?, ?, ?, ?, ?)",
                    batch)
                conn.commit()
                batch = []
        if batch:
            cursor.executemany(
                "INSERT INTO DiskChunkMap (Collection, ChunkNumber, SequenceOrder, CompactId, VectorCount) VALUES (?, ?, ?, ?, ?)",
                batch)
            conn.commit()

        cursor.close()
        conn.close()
        logger.info(f"Synced chunk {chunk_number} to DB: {len(ids_array):,} vectors, {seq:,} runs")
    except Exception as e:
        logger.warning(f"Failed to sync chunk {chunk_number} to DB: {e}")


def sync_paths_to_db(collection_name, path_to_id):
    """Sync new paths to the DB. Only inserts paths not already in DB.
    Non-fatal: logs warning on failure so builds aren't interrupted."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get current max ID in DB
        cursor.execute(
            "SELECT MAX(CompactId) FROM DiskPathLookup WHERE Collection = ?",
            (collection_name,))
        row = cursor.fetchone()
        max_db_id = row[0] if row[0] is not None else -1

        # Build id_to_path from path_to_id dict
        id_to_path = [''] * len(path_to_id)
        for path, pid in path_to_id.items():
            id_to_path[pid] = path

        # Only insert paths newer than what's in DB
        new_paths = [(collection_name, pid, id_to_path[pid])
                     for pid in range(max_db_id + 1, len(id_to_path))]

        if not new_paths:
            cursor.close()
            conn.close()
            return

        batch = []
        for item in new_paths:
            batch.append(item)
            if len(batch) >= 5000:
                cursor.executemany(
                    "INSERT INTO DiskPathLookup (Collection, CompactId, ImagePath) VALUES (?, ?, ?)",
                    batch)
                conn.commit()
                batch = []
        if batch:
            cursor.executemany(
                "INSERT INTO DiskPathLookup (Collection, CompactId, ImagePath) VALUES (?, ?, ?)",
                batch)
            conn.commit()

        cursor.close()
        conn.close()
        logger.info(f"Synced {len(new_paths):,} new paths to DB for {collection_name}")
    except Exception as e:
        logger.warning(f"Failed to sync paths to DB for {collection_name}: {e}")


def import_collection(collection_name):
    """Import a collection's path_lookup.json and _ids.npy files into DB."""
    config = COLLECTIONS[collection_name]
    ids_dir = config.get("disk_chunk_ids_dir")
    if not ids_dir or not os.path.exists(ids_dir):
        print(f"  {collection_name}: IDs dir not found, skipping")
        return

    lookup_file = os.path.join(ids_dir, "path_lookup.json")
    if not os.path.exists(lookup_file):
        print(f"  {collection_name}: no path_lookup.json, skipping")
        return

    conn = get_connection()
    cursor = conn.cursor()

    # Clear existing data for this collection
    cursor.execute("DELETE FROM DiskPathLookup WHERE Collection = ?", (collection_name,))
    cursor.execute("DELETE FROM DiskChunkMap WHERE Collection = ?", (collection_name,))
    conn.commit()

    # Import path_lookup
    print(f"  {collection_name}: Loading path_lookup.json...")
    with open(lookup_file) as f:
        id_to_path = json.load(f)

    print(f"  {collection_name}: Inserting {len(id_to_path):,} paths...")
    batch = []
    for compact_id, path in enumerate(id_to_path):
        batch.append((collection_name, compact_id, path))
        if len(batch) >= 5000:
            cursor.executemany(
                "INSERT INTO DiskPathLookup (Collection, CompactId, ImagePath) VALUES (?, ?, ?)",
                batch
            )
            conn.commit()
            batch = []
    if batch:
        cursor.executemany(
            "INSERT INTO DiskPathLookup (Collection, CompactId, ImagePath) VALUES (?, ?, ?)",
            batch
        )
        conn.commit()

    print(f"  {collection_name}: {len(id_to_path):,} paths imported.")

    # Import chunk maps (run-length encoded)
    id_files = sorted(glob(os.path.join(ids_dir, "chunk_*_ids.npy")))
    print(f"  {collection_name}: Processing {len(id_files)} chunk ID files...")

    total_runs = 0
    start = time.time()

    for file_idx, id_file in enumerate(id_files):
        chunk_name = os.path.basename(id_file).replace("_ids.npy", "")
        # Extract chunk number from name like "chunk_001"
        chunk_num = int(chunk_name.split("_")[1])

        ids = np.load(id_file)

        # Run-length encode
        batch = []
        seq = 0
        for compact_id, group in groupby(ids.tolist()):
            count = sum(1 for _ in group)
            batch.append((collection_name, chunk_num, seq, int(compact_id), count))
            seq += 1

            if len(batch) >= 5000:
                cursor.executemany(
                    "INSERT INTO DiskChunkMap (Collection, ChunkNumber, SequenceOrder, CompactId, VectorCount) VALUES (?, ?, ?, ?, ?)",
                    batch
                )
                conn.commit()
                batch = []

        if batch:
            cursor.executemany(
                "INSERT INTO DiskChunkMap (Collection, ChunkNumber, SequenceOrder, CompactId, VectorCount) VALUES (?, ?, ?, ?, ?)",
                batch
            )
            conn.commit()

        total_runs += seq
        elapsed = time.time() - start
        rate = (file_idx + 1) / elapsed if elapsed > 0 else 1
        eta = (len(id_files) - file_idx - 1) / rate
        print(f"\r    [{file_idx+1}/{len(id_files)}] {chunk_name} ({seq:,} runs) | "
              f"ETA: {eta/60:.1f}m   ", end="", flush=True)

    print(f"\n  {collection_name}: {total_runs:,} total runs across {len(id_files)} chunks.")

    cursor.close()
    conn.close()


def rebuild_chunk_ids(collection_name, chunk_number):
    """Rebuild a chunk's _ids.npy from the database."""
    config = COLLECTIONS[collection_name]
    ids_dir = config.get("disk_chunk_ids_dir")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT CompactId, VectorCount
        FROM DiskChunkMap
        WHERE Collection = ? AND ChunkNumber = ?
        ORDER BY SequenceOrder
    """, (collection_name, chunk_number))

    rows = cursor.fetchall()
    if not rows:
        print(f"  No data found for {collection_name} chunk {chunk_number}")
        cursor.close()
        conn.close()
        return False

    # Expand run-length encoding back to flat array
    ids = []
    for compact_id, vector_count in rows:
        ids.extend([compact_id] * vector_count)

    ids_array = np.array(ids, dtype=np.int32)

    ids_file = os.path.join(ids_dir, f"chunk_{chunk_number:03d}_ids.npy")

    # Compare with existing if present
    if os.path.exists(ids_file):
        old = np.load(ids_file)
        print(f"  Existing: {len(old):,} entries")
        print(f"  Rebuilt:  {len(ids_array):,} entries")
    else:
        print(f"  No existing file, creating new.")

    np.save(ids_file, ids_array)
    print(f"  Saved {ids_file}: {len(ids_array):,} entries, "
          f"ID range {ids_array.min()}-{ids_array.max()}")

    cursor.close()
    conn.close()
    return True


def import_single_chunk(collection_name, chunk_number):
    """Re-import a single chunk's _ids.npy into the DB (replaces existing data for that chunk)."""
    config = COLLECTIONS[collection_name]
    ids_dir = config.get("disk_chunk_ids_dir")

    ids_file = os.path.join(ids_dir, f"chunk_{chunk_number:03d}_ids.npy")
    if not os.path.exists(ids_file):
        print(f"  File not found: {ids_file}")
        return False

    ids = np.load(ids_file)
    print(f"  Loading {ids_file}: {len(ids):,} vectors")

    conn = get_connection()
    cursor = conn.cursor()

    # Delete existing data for this chunk
    cursor.execute(
        "DELETE FROM DiskChunkMap WHERE Collection = ? AND ChunkNumber = ?",
        (collection_name, chunk_number))
    conn.commit()

    # Run-length encode and insert
    batch = []
    seq = 0
    for compact_id, group in groupby(ids.tolist()):
        count = sum(1 for _ in group)
        batch.append((collection_name, chunk_number, seq, int(compact_id), count))
        seq += 1

        if len(batch) >= 5000:
            cursor.executemany(
                "INSERT INTO DiskChunkMap (Collection, ChunkNumber, SequenceOrder, CompactId, VectorCount) VALUES (?, ?, ?, ?, ?)",
                batch
            )
            conn.commit()
            batch = []

    if batch:
        cursor.executemany(
            "INSERT INTO DiskChunkMap (Collection, ChunkNumber, SequenceOrder, CompactId, VectorCount) VALUES (?, ?, ?, ?, ?)",
            batch
        )
        conn.commit()

    print(f"  Imported chunk {chunk_number}: {len(ids):,} vectors, {seq:,} runs")
    cursor.close()
    conn.close()
    return True


def verify_collection(collection_name):
    """Verify DB data matches the _ids.npy files on disk."""
    config = COLLECTIONS[collection_name]
    ids_dir = config.get("disk_chunk_ids_dir")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM DiskPathLookup WHERE Collection = ?
    """, (collection_name,))
    path_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT ChunkNumber, SUM(VectorCount) as TotalVectors, COUNT(*) as RunCount
        FROM DiskChunkMap WHERE Collection = ?
        GROUP BY ChunkNumber ORDER BY ChunkNumber
    """, (collection_name,))

    chunks = cursor.fetchall()
    mismatches = 0

    for chunk_num, total_vectors, run_count in chunks:
        ids_file = os.path.join(ids_dir, f"chunk_{chunk_num:03d}_ids.npy")
        if os.path.exists(ids_file):
            disk_count = len(np.load(ids_file))
            if disk_count != total_vectors:
                print(f"  MISMATCH chunk_{chunk_num:03d}: disk={disk_count:,}, db={total_vectors:,}")
                mismatches += 1

    print(f"  {collection_name}: {path_count:,} paths, {len(chunks)} chunks in DB, {mismatches} mismatches")
    cursor.close()
    conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DISK chunk ID database management")
    parser.add_argument("action", choices=["import", "import-chunk", "rebuild", "verify", "import-all"])
    parser.add_argument("--collection", "-c", help="Collection name")
    parser.add_argument("--chunk", type=int, help="Chunk number (for rebuild)")
    args = parser.parse_args()

    create_tables()

    if args.action == "import-all":
        for name in COLLECTIONS:
            if "disk_chunk_ids_dir" in COLLECTIONS[name]:
                print(f"\n{'='*60}")
                import_collection(name)
        print("\nAll collections imported.")

    elif args.action == "import":
        if not args.collection:
            parser.error("--collection required for import")
        import_collection(args.collection)

    elif args.action == "import-chunk":
        if not args.collection or args.chunk is None:
            parser.error("--collection and --chunk required for import-chunk")
        import_single_chunk(args.collection, args.chunk)

    elif args.action == "rebuild":
        if not args.collection or args.chunk is None:
            parser.error("--collection and --chunk required for rebuild")
        rebuild_chunk_ids(args.collection, args.chunk)

    elif args.action == "verify":
        if args.collection:
            verify_collection(args.collection)
        else:
            for name in COLLECTIONS:
                if "disk_chunk_ids_dir" in COLLECTIONS[name]:
                    verify_collection(name)


if __name__ == "__main__":
    main()
