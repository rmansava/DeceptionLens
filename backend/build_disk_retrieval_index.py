"""
Build DISK Retrieval Index (Per-Book Shards with Background NAS Sync)

Creates one FAISS index per book:
- Saves to local SSD instantly
- Queues NAS copy in background
- Processing never waits for NAS
"""

import faiss
import numpy as np
import json
import os
import shutil
from glob import glob
import time
import threading
from queue import Queue
import unicodedata
import re


def sanitize_dirname(name):
    """Convert Unicode to ASCII-safe directory name."""
    # Normalize Unicode (é -> e, ñ -> n, etc.)
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining problematic chars
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Clean up whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Config
NAS_FEATURES = "T:/disk-features/books"
LOCAL_BUFFER = "C:/temp/disk-retrieval-buffer"
LOCAL_INDEX = "C:/temp/disk-retrieval-index"
NAS_INDEX = "D:/faiss/disk_retrieval/books"
BACKUP_INDEX = "T:/faiss/disk_retrieval/books"  # NAS backup
IMAGES_DIR = "D:/books/pdf-images"

BATCH_SIZE = 50  # Books to copy to local at a time

# Background NAS copy queue
copy_queue = Queue()
copy_errors = []
nas_copy_count = 0
nas_copy_lock = threading.Lock()


def nas_copy_worker():
    """Background thread that copies book indexes from local to NAS."""
    global nas_copy_count
    while True:
        item = copy_queue.get()
        if item is None:  # Shutdown signal
            copy_queue.task_done()
            break

        book_name = item
        success = False
        for attempt in range(5):
            try:
                src_dir = os.path.join(LOCAL_INDEX, book_name)
                dst_dir = os.path.join(NAS_INDEX, book_name)
                os.makedirs(dst_dir, exist_ok=True)

                # Copy index and paths to primary (D:)
                shutil.copy2(
                    os.path.join(src_dir, "index.faiss"),
                    os.path.join(dst_dir, "index.faiss")
                )
                shutil.copy2(
                    os.path.join(src_dir, "paths.json"),
                    os.path.join(dst_dir, "paths.json")
                )

                # Also copy to backup (T:)
                backup_dir = os.path.join(BACKUP_INDEX, book_name)
                try:
                    os.makedirs(backup_dir, exist_ok=True)
                    shutil.copy2(
                        os.path.join(src_dir, "index.faiss"),
                        os.path.join(backup_dir, "index.faiss")
                    )
                    shutil.copy2(
                        os.path.join(src_dir, "paths.json"),
                        os.path.join(backup_dir, "paths.json")
                    )
                except Exception:
                    pass  # Backup failure is non-fatal

                # Remove local copy after successful primary copy
                shutil.rmtree(src_dir, ignore_errors=True)

                with nas_copy_lock:
                    nas_copy_count += 1
                success = True
                break

            except Exception as e:
                if attempt < 4:
                    time.sleep(5)  # Wait before retry
                else:
                    copy_errors.append((book_name, str(e)))

        copy_queue.task_done()


def start_copy_thread():
    """Start the background NAS copy thread."""
    thread = threading.Thread(target=nas_copy_worker, daemon=True)
    thread.start()
    return thread


def get_all_books():
    """Get list of all book directories on NAS."""
    return sorted([d for d in os.listdir(NAS_FEATURES)
                   if os.path.isdir(os.path.join(NAS_FEATURES, d))])


def get_indexed_books():
    """Get books already indexed (on NAS or pending local)."""
    indexed = set()

    # Check NAS
    if os.path.exists(NAS_INDEX):
        for d in os.listdir(NAS_INDEX):
            if os.path.exists(os.path.join(NAS_INDEX, d, "index.faiss")):
                indexed.add(d)

    # Check local (pending NAS copy)
    if os.path.exists(LOCAL_INDEX):
        for d in os.listdir(LOCAL_INDEX):
            if os.path.exists(os.path.join(LOCAL_INDEX, d, "index.faiss")):
                indexed.add(d)

    return indexed


def copy_batch_to_local(books):
    """Copy batch of books to local buffer with retry on network errors."""
    os.makedirs(LOCAL_BUFFER, exist_ok=True)
    for book in books:
        src = os.path.join(NAS_FEATURES, book)
        dst = os.path.join(LOCAL_BUFFER, book)
        if not os.path.exists(dst):
            for attempt in range(5):
                try:
                    shutil.copytree(src, dst)
                    break
                except (OSError, shutil.Error) as e:
                    if attempt < 4:
                        print(f"\n    Network error copying {book}, retry {attempt+1}/5...")
                        time.sleep(5)
                        # Clean up partial copy
                        if os.path.exists(dst):
                            shutil.rmtree(dst, ignore_errors=True)
                    else:
                        print(f"\n    SKIP: Failed to copy {book} after 5 attempts")
                        break


def clear_local_buffer():
    """Delete local buffer."""
    if os.path.exists(LOCAL_BUFFER):
        for attempt in range(3):
            try:
                shutil.rmtree(LOCAL_BUFFER)
                return
            except PermissionError:
                import gc
                gc.collect()
                time.sleep(2)
        try:
            os.rename(LOCAL_BUFFER, LOCAL_BUFFER + f"_old_{int(time.time())}")
        except:
            pass


def process_and_save_book(book):
    """Process book, save to local SSD, queue NAS copy."""
    # Ensure LOCAL_INDEX exists (might have been cleared)
    os.makedirs(LOCAL_INDEX, exist_ok=True)

    book_path = os.path.join(LOCAL_BUFFER, book)
    npz_files = sorted(glob(os.path.join(book_path, "*.npz")))

    if not npz_files:
        return 0

    # FIRST PASS: Count keypoints per file (no loading into memory)
    MAX_KEYPOINTS = 50_000_000  # 50M = ~25GB per shard
    file_kp_counts = []
    total_kp = 0

    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            count = data['descriptors'].shape[0]
            file_kp_counts.append((npz_path, count))
            total_kp += count
        except Exception:
            pass

    if not file_kp_counts:
        return 0

    # Check if we need to split
    if total_kp > MAX_KEYPOINTS:
        num_parts = (total_kp // MAX_KEYPOINTS) + 1
        print(f"\n    LARGE: {book} has {total_kp:,} kp, splitting into {num_parts} parts")

        # Group files into parts
        parts = []
        current_part = []
        current_kp = 0

        for npz_path, count in file_kp_counts:
            if current_kp + count > MAX_KEYPOINTS and current_part:
                parts.append(current_part)
                current_part = []
                current_kp = 0
            current_part.append(npz_path)
            current_kp += count

        if current_part:
            parts.append(current_part)

        # Process each part separately
        total_saved = 0
        for part_idx, part_files in enumerate(parts):
            part_descriptors = []
            part_paths = []

            for npz_path in part_files:
                try:
                    data = np.load(npz_path)
                    desc = data['descriptors'].astype('float32')
                    norms = np.linalg.norm(desc, axis=1, keepdims=True)
                    desc = desc / (norms + 1e-8)

                    page_name = os.path.basename(npz_path).replace('.npz', '')
                    img_path = f"{IMAGES_DIR}/{book}/{page_name}.jpg"

                    part_descriptors.append(desc)
                    part_paths.extend([img_path] * len(desc))
                except Exception:
                    pass

            if not part_descriptors:
                continue

            descriptors = np.vstack(part_descriptors)
            index = faiss.IndexFlatIP(128)
            index.add(descriptors)

            part_name = f"{book}_part{part_idx + 1}"
            book_index_dir = os.path.join(LOCAL_INDEX, part_name)

            try:
                os.makedirs(book_index_dir, exist_ok=True)
                faiss.write_index(index, os.path.join(book_index_dir, "index.faiss"))
                with open(os.path.join(book_index_dir, "paths.json"), 'w') as f:
                    json.dump(part_paths, f)
                copy_queue.put(part_name)
                total_saved += len(descriptors)
                print(f"      Part {part_idx + 1}/{len(parts)}: {len(descriptors):,} kp")
            except Exception as e:
                print(f"\n    ERROR: Could not save {part_name}: {e}")
                shutil.rmtree(book_index_dir, ignore_errors=True)

            # Free memory
            del part_descriptors, descriptors, index
            import gc
            gc.collect()

        return total_saved

    # Normal path: load all at once (fits in memory)
    all_descriptors = []
    all_paths = []

    for npz_path, _ in file_kp_counts:
        try:
            data = np.load(npz_path)
            desc = data['descriptors'].astype('float32')
            norms = np.linalg.norm(desc, axis=1, keepdims=True)
            desc = desc / (norms + 1e-8)

            page_name = os.path.basename(npz_path).replace('.npz', '')
            img_path = f"{IMAGES_DIR}/{book}/{page_name}.jpg"

            all_descriptors.append(desc)
            all_paths.extend([img_path] * len(desc))
        except Exception:
            pass

    if not all_descriptors:
        return 0

    # Stack all descriptors
    descriptors = np.vstack(all_descriptors)

    # Create simple flat index (no training needed)
    index = faiss.IndexFlatIP(128)
    index.add(descriptors)

    # Save to LOCAL SSD (fast!)
    book_index_dir = os.path.join(LOCAL_INDEX, book)

    # Robust directory creation with retry
    for attempt in range(3):
        try:
            os.makedirs(book_index_dir, exist_ok=True)
            if os.path.isdir(book_index_dir):
                break
        except Exception:
            time.sleep(1)

    if not os.path.isdir(book_index_dir):
        print(f"\n    ERROR: Could not create directory for {book}, skipping")
        return 0

    # Try to save - skip book if it fails (e.g., special characters in name)
    try:
        faiss.write_index(index, os.path.join(book_index_dir, "index.faiss"))
        with open(os.path.join(book_index_dir, "paths.json"), 'w') as f:
            json.dump(all_paths, f)
    except Exception as e:
        print(f"\n    ERROR: Could not save {book}: {e}")
        # Clean up partial save
        shutil.rmtree(book_index_dir, ignore_errors=True)
        return 0

    # Queue background copy to NAS (non-blocking)
    copy_queue.put(book)

    return len(descriptors)


def main():
    print()
    print("=" * 70)
    print("  DISK RETRIEVAL INDEX BUILDER (Per-Book Shards)")
    print("=" * 70)
    print()
    print("  Saves to local SSD instantly, copies to NAS in background.")
    print()
    print(f"  Source:  {NAS_FEATURES}")
    print(f"  Buffer:  {LOCAL_BUFFER}")
    print(f"  Local:   {LOCAL_INDEX} (fast saves)")
    print(f"  NAS:     {NAS_INDEX} (background copy)")
    print(f"  Batch:   {BATCH_SIZE} books at a time")
    print()
    print("-" * 70)

    # Start background NAS copy thread
    copy_thread = start_copy_thread()

    # Get all books and filter out already indexed
    all_books = get_all_books()
    indexed = get_indexed_books()
    books_to_process = [b for b in all_books if b not in indexed]

    total_books = len(all_books)
    already_done = len(indexed)
    remaining = len(books_to_process)

    print(f"  Total books:     {total_books:,}")
    print(f"  Already indexed: {already_done:,}")
    print(f"  To process:      {remaining:,}")
    print()

    if remaining == 0:
        print("  Nothing to do!")
        return

    print("-" * 70)

    os.makedirs(LOCAL_INDEX, exist_ok=True)
    os.makedirs(NAS_INDEX, exist_ok=True)

    start = time.time()
    books_done = 0
    total_keypoints = 0

    num_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, remaining)
        batch_books = books_to_process[batch_start:batch_end]

        # Copy batch to local
        clear_local_buffer()
        print(f"  Copying batch {batch_idx + 1}/{num_batches} to local...", end=" ", flush=True)
        t0 = time.time()
        copy_batch_to_local(batch_books)
        print(f"({time.time() - t0:.0f}s)")

        # Process each book
        for book in batch_books:
            kp = process_and_save_book(book)
            books_done += 1
            total_keypoints += kp

            elapsed = time.time() - start
            rate = books_done / elapsed * 60
            eta = (remaining - books_done) / (books_done / elapsed) / 60 if books_done > 0 else 0

            with nas_copy_lock:
                pending = books_done - nas_copy_count

            print(f"  Book {already_done + books_done:5,}/{total_books:,} | "
                  f"{kp:>10,} kp | "
                  f"{rate:5.1f} books/min | "
                  f"ETA {eta:5.0f}m | "
                  f"NAS queue: {pending}")

    # Cleanup local buffer
    clear_local_buffer()

    # Wait for NAS copies to complete
    print()
    print("-" * 70)
    print(f"  Waiting for {copy_queue.qsize()} pending NAS copies...", end=" ", flush=True)
    t0 = time.time()
    copy_queue.join()
    copy_queue.put(None)  # Signal shutdown
    copy_thread.join()
    print(f"({time.time() - t0:.0f}s)")

    # Report errors if any
    if copy_errors:
        print()
        print(f"  WARNING: {len(copy_errors)} NAS copy errors:")
        for book, err in copy_errors[:5]:
            print(f"    - {book}: {err}")
        if len(copy_errors) > 5:
            print(f"    ... and {len(copy_errors) - 5} more")

    # Stats
    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Books indexed:  {books_done:,}")
    print(f"  Keypoints:      {total_keypoints:,}")
    print(f"  Time:           {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print()
    print(f"  Saved to: {NAS_INDEX}")
    print(f"  ({total_books:,} book shards)")
    print()


if __name__ == "__main__":
    main()
