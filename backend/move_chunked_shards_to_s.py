"""
Move already-chunked per-book FAISS shards from T: to S: to free space.

Only moves books confirmed in the consolidation state file (already in 10GB chunks).
Verifies the copy before deleting from T:.
Uses 5 parallel workers for throughput.
"""

import os
import json
import shutil
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

SOURCE = r"T:\faiss\disk_retrieval\books"
DEST = r"S:\faiss\disk_retrieval\books"
STATE_FILE = r"D:\faiss\disk_retrieval\consolidation_state.json"
WORKERS = 5

# Shared counters
lock = threading.Lock()
moved = 0
skipped = 0
errors = 0
freed_bytes = 0
completed = 0

# Per-worker status
worker_status = ["idle"] * WORKERS
worker_ids = {}  # thread id -> worker index
worker_lock = threading.Lock()


def get_worker_id():
    """Get or assign a worker index for the current thread."""
    tid = threading.current_thread().ident
    with worker_lock:
        if tid not in worker_ids:
            worker_ids[tid] = len(worker_ids) % WORKERS
        return worker_ids[tid]


def get_dir_size(path):
    """Get total size of directory in bytes."""
    total = 0
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total


def verify_copy(src_dir, dst_dir):
    """Verify all files were copied correctly by checking sizes."""
    for f in os.listdir(src_dir):
        src_file = os.path.join(src_dir, f)
        dst_file = os.path.join(dst_dir, f)
        if os.path.isfile(src_file):
            if not os.path.exists(dst_file):
                return False
            if os.path.getsize(src_file) != os.path.getsize(dst_file):
                return False
    return True


def move_one_book(book):
    """Copy one book to S:, verify, delete from T:."""
    global moved, skipped, errors, freed_bytes, completed
    wid = get_worker_id()

    src_dir = os.path.join(SOURCE, book)
    dst_dir = os.path.join(DEST, book)

    if not os.path.isdir(src_dir):
        with lock:
            completed += 1
        worker_status[wid] = "idle"
        return

    src_size = get_dir_size(src_dir)
    size_gb = src_size / (1024**3)

    # Truncate long book names for display
    display = book[:50] + "..." if len(book) > 50 else book
    worker_status[wid] = f"copying {display} ({size_gb:.1f}GB)"

    try:
        if not os.path.exists(dst_dir):
            shutil.copytree(src_dir, dst_dir)

        worker_status[wid] = f"verifying {display}"

        if verify_copy(src_dir, dst_dir):
            shutil.rmtree(src_dir)
            with lock:
                moved += 1
                freed_bytes += src_size
                completed += 1
        else:
            with lock:
                skipped += 1
                completed += 1
            worker_status[wid] = "idle"
            return "VERIFY FAILED"

    except Exception as e:
        with lock:
            errors += 1
            completed += 1
        if os.path.exists(dst_dir):
            try:
                if not verify_copy(src_dir, dst_dir):
                    shutil.rmtree(dst_dir, ignore_errors=True)
            except Exception:
                pass
        worker_status[wid] = "idle"
        return f"ERROR: {e}"

    worker_status[wid] = "idle"
    return None


def render_status(total, start_time):
    """Render the status display (summary + 5 worker lines)."""
    with lock:
        pct = completed / total * 100 if total > 0 else 0
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = total - completed
        eta = remaining / rate / 60 if rate > 0 else 0

        lines = []
        lines.append(f"  [{completed:,}/{total:,}] {pct:5.1f}% | "
                     f"Moved: {moved:,} | "
                     f"Freed: {freed_bytes / (1024**3):.1f} GB | "
                     f"ETA: {eta:.0f}m")
        for i in range(WORKERS):
            lines.append(f"    Worker {i+1}: {worker_status[i]}")

    # Move cursor up and overwrite
    output = ""
    for line in lines:
        output += f"\r{line:<100}\n"
    # Move cursor back up
    output += f"\033[{len(lines)}A"
    print(output, end="", flush=True)


def main():
    global completed

    print("=" * 70)
    print("  MOVE CHUNKED BOOK SHARDS: T: -> S:")
    print("=" * 70)
    print()

    # Load consolidation state
    if not os.path.exists(STATE_FILE):
        print(f"ERROR: State file not found: {STATE_FILE}")
        sys.exit(1)

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    chunked_books = set(state.get('processed_books', []))
    print(f"  Books in consolidation state: {len(chunked_books):,}")

    # Find which chunked books exist on T:
    if not os.path.exists(SOURCE):
        print(f"ERROR: Source not found: {SOURCE}")
        sys.exit(1)

    all_dirs = os.listdir(SOURCE)
    to_move = sorted([d for d in all_dirs if d in chunked_books])
    to_keep = [d for d in all_dirs if d not in chunked_books]

    print(f"  Book shards on T:: {len(all_dirs):,}")
    print(f"  Already chunked (will move): {len(to_move):,}")
    print(f"  Not yet chunked (will keep): {len(to_keep):,}")
    print(f"  Workers: {WORKERS}")
    print()

    os.makedirs(DEST, exist_ok=True)

    already_on_s = sum(1 for d in to_move if os.path.exists(os.path.join(DEST, d)))
    if already_on_s:
        print(f"  Already on S:: {already_on_s:,} (will verify and delete from T:)")

    print()
    print("-" * 70)
    # Print blank lines for the status area
    for _ in range(WORKERS + 1):
        print()

    total = len(to_move)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(move_one_book, book): book for book in to_move}

        while completed < total:
            render_status(total, start_time)
            time.sleep(0.5)

        # Final render
        render_status(total, start_time)

        # Collect error messages
        error_msgs = []
        for future in futures:
            result = future.result()
            if result:
                error_msgs.append(f"    {futures[future]}: {result}")

    # Move past the status area
    print("\n" * (WORKERS + 1))

    if error_msgs:
        print("  Errors:")
        for msg in error_msgs:
            print(msg)
        print()

    elapsed = time.time() - start_time

    print("=" * 70)
    print("  COMPLETE!")
    print("=" * 70)
    print(f"  Moved:   {moved:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Errors:  {errors:,}")
    print(f"  Freed:   {freed_bytes / (1024**3):.1f} GB on T:")
    print(f"  Time:    {elapsed / 60:.1f} min")
    print()


if __name__ == '__main__':
    main()
