"""Check which chunk actually contains Encyclopedia of Monsters page 210 vectors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from db_helper import get_connection
from collections_config import COLLECTIONS

conn = get_connection()
cursor = conn.cursor()

# Get compact IDs for Encyclopedia of Monsters page 210
cursor.execute(
    "SELECT CompactId, ImagePath FROM DiskPathLookup "
    "WHERE Collection = 'books' AND ImagePath LIKE '%Encyclopedia of Monsters%page210.jpg'"
)
targets = cursor.fetchall()
target_ids = [r[0] for r in targets]
print("Target compact IDs:")
for cid, path in targets:
    print(f"  ID {cid}: {os.path.basename(path)}")

# Check which chunks contain these IDs according to DB
print(f"\nDB DiskChunkMap entries for these IDs:")
for cid in target_ids:
    cursor.execute(
        "SELECT ChunkNumber, VectorCount FROM DiskChunkMap "
        "WHERE Collection = 'books' AND CompactId = ? ORDER BY ChunkNumber",
        (cid,)
    )
    rows = cursor.fetchall()
    if rows:
        for chunk, count in rows:
            print(f"  ID {cid} -> chunk {chunk} ({count:,} vectors)")
    else:
        print(f"  ID {cid} -> NOT IN ANY CHUNK!")

cursor.close()
conn.close()

# Now verify against actual _ids.npy files
ids_dir = COLLECTIONS["books"]["disk_chunk_ids_dir"]
print(f"\nVerifying against actual _ids.npy files in {ids_dir}...")

# Check a few chunks around 184
for chunk_num in [142, 143, 183, 184, 185]:
    ids_file = os.path.join(ids_dir, f"chunk_{chunk_num:03d}_ids.npy")
    if os.path.exists(ids_file):
        ids = np.load(ids_file)
        found = []
        for cid in target_ids:
            count = np.sum(ids == cid)
            if count > 0:
                found.append((cid, count))
        if found:
            print(f"  chunk_{chunk_num:03d}: FOUND! {found}")
        else:
            print(f"  chunk_{chunk_num:03d}: not found ({len(ids):,} vectors, ID range {ids.min()}-{ids.max()})")
    else:
        print(f"  chunk_{chunk_num:03d}: file not found")

# Brute force: scan ALL chunk ID files for these IDs
print(f"\nScanning ALL chunk ID files for target IDs...")
from glob import glob
all_id_files = sorted(glob(os.path.join(ids_dir, "chunk_*_ids.npy")))
print(f"  Total ID files: {len(all_id_files)}")

for ids_file in all_id_files:
    ids = np.load(ids_file)
    for cid in target_ids:
        count = np.sum(ids == cid)
        if count > 0:
            chunk_name = os.path.basename(ids_file)
            print(f"  {chunk_name}: ID {cid} found {count:,} times")
