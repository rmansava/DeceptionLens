"""Quick test of decoupled save with 5 books."""
import faiss
import numpy as np
import json
import os
import shutil
from glob import glob
import time
import threading
from queue import Queue

# Config - TEST MODE: Only 5 books
NAS_FEATURES = 'T:/disk-features/books'
LOCAL_BUFFER = 'C:/temp/disk-retrieval-buffer'
LOCAL_INDEX = 'C:/temp/disk-retrieval-index'
NAS_INDEX = 'T:/faiss/disk_retrieval_test'  # Test location
IMAGES_DIR = 'D:/books/pdf-images'

TEST_BOOKS = 5  # Just 5 books for test

# Background copy
copy_queue = Queue()
copy_in_progress = threading.Event()

def background_copy_worker():
    while True:
        item = copy_queue.get()
        if item is None:
            copy_queue.task_done()
            break
        copy_in_progress.set()
        try:
            src_index = os.path.join(LOCAL_INDEX, 'index.faiss')
            dst_index = os.path.join(NAS_INDEX, 'index.faiss')
            if os.path.exists(src_index):
                shutil.copy2(src_index, dst_index)
            src_paths = os.path.join(LOCAL_INDEX, 'paths.json')
            dst_paths = os.path.join(NAS_INDEX, 'paths.json')
            if os.path.exists(src_paths):
                shutil.copy2(src_paths, dst_paths)
        except Exception as e:
            print(f'  [NAS COPY ERROR: {e}]')
        finally:
            copy_in_progress.clear()
            copy_queue.task_done()

# Start thread
copy_thread = threading.Thread(target=background_copy_worker, daemon=True)
copy_thread.start()

# Get 5 books
all_books = sorted([d for d in os.listdir(NAS_FEATURES) if os.path.isdir(os.path.join(NAS_FEATURES, d))])[:TEST_BOOKS]
print(f'Testing with {len(all_books)} books: {all_books}')

# Collect training data
print('Collecting training samples...')
os.makedirs(LOCAL_BUFFER, exist_ok=True)
for book in all_books:
    src = os.path.join(NAS_FEATURES, book)
    dst = os.path.join(LOCAL_BUFFER, book)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

train_data = []
for book in all_books[:3]:
    book_path = os.path.join(LOCAL_BUFFER, book)
    npz_files = glob(os.path.join(book_path, '*.npz'))[:10]
    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            desc = data['descriptors'].astype('float32')
            norms = np.linalg.norm(desc, axis=1, keepdims=True)
            train_data.append((desc / (norms + 1e-8))[:500])
        except:
            pass

train_data = np.vstack(train_data)
print(f'Training with {len(train_data):,} vectors')

# Create and train index
d = 128
nlist = min(256, len(train_data) // 40)  # Smaller for test
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(train_data)
del train_data
print(f'Trained IVF index ({nlist} clusters)')

# Process books
os.makedirs(LOCAL_INDEX, exist_ok=True)
os.makedirs(NAS_INDEX, exist_ok=True)
paths = []
start = time.time()

for i, book in enumerate(all_books):
    book_path = os.path.join(LOCAL_BUFFER, book)
    npz_files = glob(os.path.join(book_path, '*.npz'))
    kp_count = 0

    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            desc = data['descriptors'].astype('float32')
            norms = np.linalg.norm(desc, axis=1, keepdims=True)
            desc = desc / (norms + 1e-8)

            page_name = os.path.basename(npz_path).replace('.npz', '')
            img_path = f'{IMAGES_DIR}/{book}/{page_name}.jpg'

            index.add(desc)
            paths.extend([img_path] * len(desc))
            kp_count += len(desc)
        except Exception as e:
            pass

    # Wait if background copy is in progress (can't write while reading)
    while copy_in_progress.is_set():
        time.sleep(0.1)

    # Save to local (fast)
    t0 = time.time()
    faiss.write_index(index, os.path.join(LOCAL_INDEX, 'index.faiss'))
    with open(os.path.join(LOCAL_INDEX, 'paths.json'), 'w') as f:
        json.dump(paths, f)
    local_save_time = time.time() - t0

    # Queue NAS copy (won't block - happens in background)
    if copy_queue.empty():
        copy_queue.put('copy')

    nas_status = '[NAS syncing]' if copy_in_progress.is_set() else '[NAS synced]'
    print(f'  Book {i+1}/{len(all_books)}: {book[:40]:40s} | {kp_count:>8,} kp | local save: {local_save_time:.1f}s | {nas_status}')

# Wait for final NAS sync
print('Waiting for final NAS sync...', end=' ', flush=True)
t0 = time.time()
copy_queue.join()
copy_queue.put(None)
copy_queue.join()
print(f'done ({time.time()-t0:.1f}s)')

# Cleanup
shutil.rmtree(LOCAL_BUFFER, ignore_errors=True)

# Stats
elapsed = time.time() - start
idx_size = os.path.getsize(os.path.join(NAS_INDEX, 'index.faiss')) / 1024**2
paths_size = os.path.getsize(os.path.join(NAS_INDEX, 'paths.json')) / 1024**2

print()
print(f'COMPLETE!')
print(f'  Books: {len(all_books)}')
print(f'  Keypoints: {index.ntotal:,}')
print(f'  Time: {elapsed:.1f}s')
print(f'  Index: {idx_size:.1f} MB')
print(f'  Paths: {paths_size:.1f} MB')
print(f'  Saved to: {NAS_INDEX}')
