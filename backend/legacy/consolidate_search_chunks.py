"""
Consolidate per-book FAISS indexes into larger chunks for faster searching.

Workflow for 13TB index:
1. Load local state file (instant resume)
2. Scan NAS for all books, filter out already processed
3. Pipeline: Copy books to local buffer while processing previous ones
4. Consolidate into chunks, save to local SSD first
5. Move finished chunks to NAS (T:/faiss/disk_retrieval/chunks)
6. Update state file after each chunk

Each chunk is ~20GB (fits in 32GB RAM for searching).
Uses 200GB buffer with background copying for continuous processing.
State file on D: drive enables instant resume without NAS scanning.
"""

import faiss
import numpy as np
import json
import os
import shutil
from glob import glob
import time
import gc
import threading
from queue import Queue, Empty
from collections import deque
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from disk_chunk_db import sync_chunk_to_db, sync_paths_to_db, create_tables

COLLECTION_NAME = "books"

# Config - 13TB index requires NAS storage with local SSD for fast processing
NAS_BOOKS_DIRS = [                                    # Source: per-book indexes on NAS
    "T:/faiss/disk_retrieval/books",
    "S:/faiss/disk_retrieval/books",
]
NAS_BOOKS_DIR = NAS_BOOKS_DIRS[0]                     # Primary (for backward compat)
LOCAL_BOOKS_BUFFER = "D:/faiss/disk_retrieval/books" # Buffer: copy books here for fast reads
LOCAL_CHUNKS_DIR = "D:/faiss/disk_retrieval/chunks"  # Buffer: write chunks here first
NAS_CHUNKS_DIR = "T:/faiss/disk_retrieval/chunks"    # Final: move chunks to NAS

# Compact IDs output (stays on local SSD for fast reads during search)
CHUNK_IDS_DIR = "D:/faiss/disk_retrieval/chunk_ids"

# State file for instant resume (stored locally for speed)
STATE_FILE = "D:/faiss/disk_retrieval/consolidation_state.json"

# ~10GB chunks = ~21M vectors (128 dims * 4 bytes = 512 bytes/vector)
MAX_VECTORS_PER_CHUNK = 21_000_000

# Buffer settings - target ~200GB on local SSD
# Average book is ~2GB (index + paths), so ~100 books = 200GB
BUFFER_TARGET_BOOKS = 100  # Try to keep this many books in buffer
BUFFER_MIN_BOOKS = 20      # Start processing when we have at least this many
COPY_BATCH_SIZE = 10       # Copy this many books at a time in background


class BookBuffer:
    """Manages the local book buffer with background copying."""

    def __init__(self, all_books, book_source_map=None):
        self.all_books = deque(all_books)  # Books not yet copied
        self.book_source_map = book_source_map or {}  # book_name -> source_dir
        self.ready_books = deque()          # Books copied and ready to process
        self.copying = set()                # Books currently being copied
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.copy_thread = None
        self.books_copied = 0
        self.bytes_copied = 0
        self.copy_errors = []

    def start_copying(self):
        """Start background copy thread."""
        self.copy_thread = threading.Thread(target=self._copy_worker, daemon=True)
        self.copy_thread.start()

    def stop_copying(self):
        """Stop background copy thread."""
        self.stop_event.set()
        if self.copy_thread:
            self.copy_thread.join(timeout=5)

    def _copy_worker(self):
        """Background worker that copies books from NAS to local."""
        while not self.stop_event.is_set():
            # Check if we need more books in buffer
            with self.lock:
                ready_count = len(self.ready_books)
                need_more = ready_count < BUFFER_TARGET_BOOKS and len(self.all_books) > 0

            if not need_more:
                # Buffer is full enough, wait a bit
                time.sleep(0.5)
                continue

            # Get next batch to copy
            with self.lock:
                batch = []
                for _ in range(COPY_BATCH_SIZE):
                    if self.all_books:
                        book = self.all_books.popleft()
                        batch.append(book)
                        self.copying.add(book)

            if not batch:
                time.sleep(0.5)
                continue

            # Copy batch
            for book in batch:
                if self.stop_event.is_set():
                    break

                try:
                    source_base = self.book_source_map.get(book, NAS_BOOKS_DIR)
                    src_dir = os.path.join(source_base, book)
                    dst_dir = os.path.join(LOCAL_BOOKS_BUFFER, book)

                    os.makedirs(dst_dir, exist_ok=True)

                    src_index = os.path.join(src_dir, "index.faiss")
                    src_paths = os.path.join(src_dir, "paths.json")
                    dst_index = os.path.join(dst_dir, "index.faiss")
                    dst_paths = os.path.join(dst_dir, "paths.json")

                    shutil.copy2(src_index, dst_index)
                    shutil.copy2(src_paths, dst_paths)

                    # Track bytes copied
                    size = os.path.getsize(dst_index) + os.path.getsize(dst_paths)

                    with self.lock:
                        self.ready_books.append(book)
                        self.copying.discard(book)
                        self.books_copied += 1
                        self.bytes_copied += size

                except Exception as e:
                    with self.lock:
                        self.copying.discard(book)
                        self.copy_errors.append((book, str(e)))

    def get_next_book(self, timeout=30):
        """Get next book ready for processing. Returns None if no more books."""
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if self.ready_books:
                    return self.ready_books.popleft()
                if not self.all_books and not self.copying:
                    return None  # No more books coming
            time.sleep(0.1)
        return None  # Timeout

    def mark_processed(self, book):
        """Delete a processed book from local buffer."""
        try:
            book_dir = os.path.join(LOCAL_BOOKS_BUFFER, book)
            if os.path.exists(book_dir):
                shutil.rmtree(book_dir, ignore_errors=True)
        except Exception:
            pass

    def get_status(self):
        """Get current buffer status."""
        with self.lock:
            return {
                'ready': len(self.ready_books),
                'copying': len(self.copying),
                'remaining': len(self.all_books),
                'copied': self.books_copied,
                'bytes': self.bytes_copied,
                'errors': len(self.copy_errors)
            }

    def has_more(self):
        """Check if there are more books to process."""
        with self.lock:
            return bool(self.ready_books or self.all_books or self.copying)


def clear_local_buffer():
    """Remove all books from local buffer."""
    if os.path.exists(LOCAL_BOOKS_BUFFER):
        for d in os.listdir(LOCAL_BOOKS_BUFFER):
            path = os.path.join(LOCAL_BOOKS_BUFFER, d)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


def copy_with_retry(src, dst, max_retries=3, delay=5):
    """Copy file with retry logic for network errors."""
    for attempt in range(max_retries):
        try:
            shutil.copy2(src, dst)
            return True
        except OSError as e:
            if attempt < max_retries - 1:
                print(f"\n    Retry {attempt + 1}/{max_retries} for {os.path.basename(src)}: {e}")
                time.sleep(delay)
            else:
                raise
    return False


def save_chunk(chunk_num, vectors_list, ids_list, path_to_id):
    """Save a chunk with compact IDs to local SSD, then copy FAISS to NAS."""
    print(f"  Saving chunk {chunk_num}...", end=" ", flush=True)
    t0 = time.time()

    # Stack vectors and create index
    all_vectors = np.vstack(vectors_list)
    chunk_index = faiss.IndexFlatIP(128)
    chunk_index.add(all_vectors)

    # Save FAISS index to local first (fast), then copy to NAS
    os.makedirs(LOCAL_CHUNKS_DIR, exist_ok=True)
    local_index = os.path.join(LOCAL_CHUNKS_DIR, f"chunk_{chunk_num:03d}.faiss")
    faiss.write_index(chunk_index, local_index)

    os.makedirs(NAS_CHUNKS_DIR, exist_ok=True)
    nas_index = os.path.join(NAS_CHUNKS_DIR, f"chunk_{chunk_num:03d}.faiss")
    copy_with_retry(local_index, nas_index)

    try:
        os.remove(local_index)
    except Exception:
        pass

    # Save compact IDs (stays on local SSD)
    os.makedirs(CHUNK_IDS_DIR, exist_ok=True)
    ids_array = np.array(ids_list, dtype=np.int32)
    ids_file = os.path.join(CHUNK_IDS_DIR, f"chunk_{chunk_num:03d}_ids.npy")
    np.save(ids_file, ids_array)

    # Save path_lookup.json
    lookup = {v: k for k, v in path_to_id.items()}
    lookup_list = [lookup[i] for i in range(len(lookup))]
    lookup_file = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    with open(lookup_file, 'w') as f:
        json.dump(lookup_list, f)

    # Sync to SQL DB (non-fatal)
    sync_chunk_to_db(COLLECTION_NAME, chunk_num, ids_array)
    sync_paths_to_db(COLLECTION_NAME, path_to_id)

    # Get file sizes
    index_size = os.path.getsize(nas_index) / (1024**3)
    ids_size = os.path.getsize(ids_file) / (1024**2)

    print(f"({time.time() - t0:.0f}s) - {index_size:.1f}GB index + {ids_size:.0f}MB IDs")

    # Cleanup
    del all_vectors, chunk_index, ids_array
    gc.collect()


def load_state():
    """Load consolidation state from local file. Returns (next_chunk, processed_books_set)."""
    if not os.path.exists(STATE_FILE):
        return 1, set()

    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        return state.get('next_chunk', 1), set(state.get('processed_books', []))
    except Exception as e:
        print(f"  Warning: Could not load state file: {e}")
        return 1, set()


def save_state(next_chunk, processed_books):
    """Save consolidation state to local file."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state = {
        'next_chunk': next_chunk,
        'processed_books': list(processed_books),
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    # Write to temp file first, then rename (atomic)
    temp_file = STATE_FILE + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(state, f)
    shutil.move(temp_file, STATE_FILE)


def rebuild_state_from_chunks():
    """One-time rebuild of state file from existing chunks (slow, reads from NAS)."""
    print("  Rebuilding state from existing chunks...")
    chunk_files = sorted(glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*.faiss")))

    if not chunk_files:
        print("  No chunks found, nothing to rebuild")
        return 1, set()

    max_chunk = 0
    processed_books = set()

    for i, chunk_file in enumerate(chunk_files):
        name = os.path.basename(chunk_file)
        print(f"    Scanning {name} ({i+1}/{len(chunk_files)})...", end="\r", flush=True)

        try:
            num = int(name.replace('chunk_', '').replace('.faiss', '').split('_')[0])
            max_chunk = max(max_chunk, num)
        except:
            continue

        paths_file = chunk_file.replace('.faiss', '_paths.json')
        if os.path.exists(paths_file):
            try:
                with open(paths_file, 'r') as f:
                    paths = json.load(f)
                for path in paths:
                    book = path.split('/')[0] if '/' in path else path.split('\\')[0]
                    processed_books.add(book)
            except:
                pass

    print()
    save_state(max_chunk + 1, processed_books)
    print(f"  Rebuilt state: {len(processed_books):,} books in {max_chunk} chunks")
    return max_chunk + 1, processed_books


def main():
    create_tables()

    print()
    print("=" * 70)
    print("  CONSOLIDATE SEARCH CHUNKS (Pipelined NAS -> Local -> NAS)")
    print("=" * 70)
    print()
    print(f"  Source:      {NAS_BOOKS_DIR}")
    print(f"  Local buf:   {LOCAL_BOOKS_BUFFER}")
    print(f"  Output:      {NAS_CHUNKS_DIR}")
    print(f"  State file:  {STATE_FILE}")
    print(f"  Chunk size:  {MAX_VECTORS_PER_CHUNK:,} vectors (~{MAX_VECTORS_PER_CHUNK * 128 * 4 / (1024**3):.1f}GB)")
    print(f"  Buffer:      {BUFFER_TARGET_BOOKS} books (~{BUFFER_TARGET_BOOKS * 2}GB)")
    print()

    # Load state from local file (instant!)
    print("  Loading state file...")
    start_chunk, processed_books = load_state()

    if processed_books:
        print(f"  Found {len(processed_books):,} already processed books")
        print(f"  Will resume from chunk {start_chunk}")
    elif os.path.exists(STATE_FILE):
        print("  State file exists but empty, starting fresh")
    else:
        # No state file - check if chunks exist (first run after update)
        existing_chunks = glob(os.path.join(NAS_CHUNKS_DIR, "chunk_*.faiss"))
        if existing_chunks:
            print(f"  No state file but found {len(existing_chunks)} existing chunks")
            print("  Rebuilding state from chunks (one-time operation)...")
            start_chunk, processed_books = rebuild_state_from_chunks()
        else:
            print("  No previous state found, starting fresh")
    print()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Consolidate per-book FAISS indexes into chunks')
    parser.add_argument('--books-file', type=str, help='File containing list of books to process (one per line)')
    args = parser.parse_args()
    using_books_file = args.books_file is not None

    # Get all book indexes from NAS (or from file if specified)
    if args.books_file:
        print(f"  Loading book list from: {args.books_file}")
        with open(args.books_file, 'r', encoding='utf-8') as f:
            all_book_dirs = sorted([line.strip() for line in f if line.strip()])
        print(f"  Loaded {len(all_book_dirs):,} books from file")
    else:
        print("  Scanning NAS for books...")
        book_source_map = {}
        for nas_dir in NAS_BOOKS_DIRS:
            if not os.path.exists(nas_dir):
                print(f"    Skipping {nas_dir} (not found)")
                continue
            count = 0
            for d in os.listdir(nas_dir):
                if d not in book_source_map and os.path.isfile(os.path.join(nas_dir, d, "index.faiss")):
                    book_source_map[d] = nas_dir
                    count += 1
            print(f"    {nas_dir}: {count:,} books")
        all_book_dirs = sorted(book_source_map.keys())
        print(f"  Found {len(all_book_dirs):,} total book indexes across {len(NAS_BOOKS_DIRS)} sources")

    # Filter out already processed books
    book_dirs = [b for b in all_book_dirs if b not in processed_books]

    total_books = len(all_book_dirs)
    remaining_books = len(book_dirs)
    if processed_books:
        print(f"  Skipping {len(processed_books):,} already processed, {remaining_books:,} remaining")
    print()

    # Create directories
    os.makedirs(LOCAL_BOOKS_BUFFER, exist_ok=True)
    os.makedirs(LOCAL_CHUNKS_DIR, exist_ok=True)
    os.makedirs(NAS_CHUNKS_DIR, exist_ok=True)

    # Clear any leftover books in buffer
    print("  Clearing local buffer...")
    clear_local_buffer()

    print("-" * 70)
    print("  Starting pipelined processing...")
    print()

    # Initialize buffer manager
    source_map = book_source_map if not args.books_file else {}
    buffer = BookBuffer(book_dirs, source_map)
    buffer.start_copying()

    # Wait for initial buffer fill
    print(f"  Waiting for initial buffer fill ({BUFFER_MIN_BOOKS} books)...", end=" ", flush=True)
    t0 = time.time()
    while True:
        status = buffer.get_status()
        if status['ready'] >= BUFFER_MIN_BOOKS or (status['remaining'] == 0 and status['copying'] == 0):
            break
        time.sleep(0.5)
    print(f"({time.time() - t0:.0f}s)")
    print()

    start_time = time.time()
    chunk_num = start_chunk  # Resume from where we left off
    current_vectors = []
    current_ids = []
    current_count = 0
    books_in_chunk = 0
    books_in_current_chunk = []  # Track book names for state file
    books_processed = 0
    total_vectors = 0
    last_status_time = time.time()

    # Compact ID tracking - MUST load existing path_lookup on resume
    path_to_id = {}
    next_id = 0
    existing_lookup = os.path.join(CHUNK_IDS_DIR, "path_lookup.json")
    if os.path.exists(existing_lookup) and processed_books:
        print("  Loading existing path_lookup.json for resume...")
        with open(existing_lookup, 'r') as f:
            id_to_path = json.load(f)
        path_to_id = {p: i for i, p in enumerate(id_to_path)}
        next_id = len(id_to_path)
        print(f"  Loaded {len(path_to_id):,} existing paths (next_id={next_id})")
        del id_to_path

    def flush_chunk():
        """Save current chunk and reset accumulators."""
        nonlocal current_vectors, current_ids, current_count, books_in_chunk
        nonlocal books_in_current_chunk, chunk_num

        print(f"  Chunk {chunk_num}: {current_count:,} vectors from {books_in_chunk} books")
        save_chunk(chunk_num, current_vectors, current_ids, path_to_id)

        # Update state file with processed books
        processed_books.update(books_in_current_chunk)
        save_state(chunk_num + 1, processed_books)
        print(f"    State saved: {len(processed_books):,} books processed, "
              f"{len(path_to_id):,} unique paths")

        # Update the books file if we're using one
        if args.books_file:
            remaining_in_file = [b for b in book_dirs if b not in processed_books]
            with open(args.books_file, 'w', encoding='utf-8') as f:
                for b in sorted(remaining_in_file):
                    f.write(f"{b}\n")
            print(f"    Book list updated: {len(remaining_in_file):,} books remaining")

        # Reset for next chunk
        del current_vectors
        gc.collect()

        current_vectors = []
        current_ids = []
        current_count = 0
        books_in_chunk = 0
        books_in_current_chunk = []
        chunk_num += 1

    # Process books as they become ready
    while buffer.has_more():
        book = buffer.get_next_book(timeout=60)

        if book is None:
            if not buffer.has_more():
                break
            print("  Warning: Timeout waiting for next book")
            continue

        local_index_path = os.path.join(LOCAL_BOOKS_BUFFER, book, "index.faiss")
        local_paths_path = os.path.join(LOCAL_BOOKS_BUFFER, book, "paths.json")

        if not os.path.exists(local_index_path):
            buffer.mark_processed(book)
            continue

        try:
            # Load from local SSD (fast)
            index = faiss.read_index(local_index_path)
            with open(local_paths_path, 'r') as f:
                paths = json.load(f)

            # Extract vectors
            vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * 128)
            vectors = vectors.reshape(-1, 128).copy()
            book_vector_count = len(vectors)

            del index

            # Convert paths to compact IDs for this book's vectors
            book_ids = []
            for path in paths:
                if path not in path_to_id:
                    path_to_id[path] = next_id
                    next_id += 1
                book_ids.append(path_to_id[path])

            # Add vectors, splitting across chunk boundaries as needed
            offset = 0
            while offset < book_vector_count:
                space = MAX_VECTORS_PER_CHUNK - current_count
                take = min(space, book_vector_count - offset)

                current_vectors.append(vectors[offset:offset + take])
                current_ids.extend(book_ids[offset:offset + take])
                current_count += take
                offset += take
                total_vectors += take

                if current_count >= MAX_VECTORS_PER_CHUNK:
                    flush_chunk()

            books_in_chunk += 1
            books_in_current_chunk.append(book)
            books_processed += 1

        except Exception as e:
            print(f"    Warning: Failed to load {book}: {e}")
            buffer.mark_processed(book)
            continue

        # Delete processed book from buffer
        buffer.mark_processed(book)

        # Progress update every 30 seconds
        if time.time() - last_status_time > 30:
            status = buffer.get_status()
            elapsed = time.time() - start_time
            rate = books_processed / elapsed if elapsed > 0 else 0
            remaining = remaining_books - books_processed
            eta_hours = remaining / rate / 3600 if rate > 0 else 0

            # Show progress - simple when using books file, detailed otherwise
            if using_books_file:
                # Just show progress through the book list
                print(f"    Progress: {books_processed:,}/{remaining_books:,} books | "
                      f"{total_vectors:,} vectors | "
                      f"Chunk {chunk_num} | "
                      f"Buffer: {status['ready']} ready | "
                      f"ETA: {eta_hours:.1f}h")
            else:
                # Show batch and total when scanning all books
                batch_done = books_processed
                batch_total = remaining_books
                total_done = len(processed_books) + books_processed
                print(f"    Progress: Batch {batch_done:,}/{batch_total:,} | "
                      f"Total {total_done:,} books | "
                      f"{total_vectors:,} vectors | "
                      f"Chunk {chunk_num} | "
                      f"Buffer: {status['ready']} ready | "
                      f"ETA: {eta_hours:.1f}h")
            last_status_time = time.time()

    # Stop background copying
    buffer.stop_copying()

    # Save final chunk
    if current_vectors:
        print(f"  Chunk {chunk_num}: {current_count:,} vectors from {books_in_chunk} books (final)")
        save_chunk(chunk_num, current_vectors, current_ids, path_to_id)

        processed_books.update(books_in_current_chunk)
        save_state(chunk_num + 1, processed_books)
        print(f"    State saved: {len(processed_books):,} books processed")

        if args.books_file:
            remaining_in_file = [b for b in book_dirs if b not in processed_books]
            with open(args.books_file, 'w', encoding='utf-8') as f:
                for b in sorted(remaining_in_file):
                    f.write(f"{b}\n")
            print(f"    Book list updated: {len(remaining_in_file):,} books remaining")

    # Cleanup
    clear_local_buffer()

    # Summary
    elapsed = time.time() - start_time
    status = buffer.get_status()
    chunks_created = chunk_num - start_chunk + (1 if current_vectors else 0)

    print()
    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Total chunks:    {chunk_num}")
    print(f"  Total books:     {books_processed:,}")
    print(f"  Total vectors:   {total_vectors:,}")
    print(f"  Unique paths:    {len(path_to_id):,}")
    print(f"  Avg chunk size:  ~{total_vectors * 512 / max(chunks_created, 1) / (1024**3):.1f} GB")
    print(f"  Time:            {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"  Throughput:      {books_processed/(elapsed/60):.1f} books/min")
    if status['errors']:
        print(f"  Copy errors:     {status['errors']}")
    print()
    print(f"  FAISS chunks:    {NAS_CHUNKS_DIR}")
    print(f"  Compact IDs:     {CHUNK_IDS_DIR}")
    print()


if __name__ == "__main__":
    main()
