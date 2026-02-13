"""Check FAISS chunk sizes vs ID file sizes for mismatches."""
import os
import sys
import numpy as np
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections_config import COLLECTIONS

def check_collection(name, config):
    chunks_dir = config.get("disk_chunks_dir")
    ids_dir = config.get("disk_chunk_ids_dir")
    if not chunks_dir or not os.path.exists(chunks_dir):
        print(f"  {name}: chunks dir not found ({chunks_dir})")
        return

    chunk_files = sorted(glob(os.path.join(chunks_dir, "chunk_*.faiss")))
    print(f"\n{'='*70}")
    print(f"  {name}: {len(chunk_files)} chunks")
    print(f"  Chunks: {chunks_dir}")
    print(f"  IDs:    {ids_dir}")
    print(f"{'='*70}")

    mismatches = []
    missing_ids = []
    total_faiss_vectors = 0
    total_id_entries = 0

    for cf in chunk_files:
        chunk_name = os.path.splitext(os.path.basename(cf))[0]
        faiss_size = os.path.getsize(cf)
        # FAISS Flat index: header (~200 bytes) + n_vectors * dim * 4
        # dim=128, so bytes_per_vector = 512
        faiss_vectors = max(0, (faiss_size - 200)) // 512

        # Check for IDs
        ids_file = os.path.join(ids_dir, f"{chunk_name}_ids.npy") if ids_dir else None
        paths_file = os.path.join(chunks_dir, f"{chunk_name}_paths.json")

        if ids_file and os.path.exists(ids_file):
            ids = np.load(ids_file)
            id_count = len(ids)
            total_id_entries += id_count
            total_faiss_vectors += faiss_vectors

            if id_count != faiss_vectors:
                ratio = id_count / faiss_vectors if faiss_vectors > 0 else 0
                mismatches.append((chunk_name, faiss_vectors, id_count, faiss_size / (1024**3)))
        elif os.path.exists(paths_file):
            import json
            with open(paths_file) as f:
                paths = json.load(f)
            id_count = len(paths)
            total_id_entries += id_count
            total_faiss_vectors += faiss_vectors

            if id_count != faiss_vectors:
                mismatches.append((chunk_name, faiss_vectors, id_count, faiss_size / (1024**3)))
        else:
            missing_ids.append(chunk_name)
            total_faiss_vectors += faiss_vectors

    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)}):")
        for chunk_name, fv, ic, size_gb in mismatches:
            diff = fv - ic
            print(f"    {chunk_name}: FAISS={fv:,} vectors, IDs={ic:,} entries (diff={diff:,}, {size_gb:.1f}GB)")
    else:
        print(f"\n  All chunks match!")

    if missing_ids:
        print(f"\n  MISSING ID FILES ({len(missing_ids)}):")
        for cn in missing_ids:
            print(f"    {cn}")

    print(f"\n  Totals: FAISS={total_faiss_vectors:,} vectors, IDs={total_id_entries:,} entries")
    if total_faiss_vectors > 0:
        print(f"  Coverage: {total_id_entries/total_faiss_vectors*100:.1f}%")

if __name__ == "__main__":
    for name, config in COLLECTIONS.items():
        if "disk_chunks_dir" in config:
            check_collection(name, config)
