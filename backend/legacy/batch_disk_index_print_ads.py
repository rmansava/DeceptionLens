r"""
Batch DISK feature indexer for Print Ads - Double-buffered local processing.

Uses local temp for images, local storage for features, background move to NAS.

Flow:
1. Copy images from NAS to local SSD (double-buffered)
2. Extract DISK features to local SSD (fast writes)
3. Background thread moves each .npz file to NAS immediately
4. Delete local image folder after processing

Source: T:\archiverelated\print ads\ebay
Local images: C:\print ads\ebay\{folder}
Local features: C:\disk-features\print_ads\{folder}
NAS features: T:\disk-features\print_ads\{folder}
"""
import os
import sys
import time
import shutil
import threading
import queue
import re
from datetime import datetime
from pathlib import Path


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally (Subfolder2 before Subfolder10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Configuration
NAS_SOURCE = r"T:\archiverelated\print ads\ebay"
LOCAL_TEMP = r"C:\print ads\ebay"
LOCAL_FEATURES = r"C:\disk-features"  # Local SSD for fast writes
NAS_FEATURES = r"T:\disk-features"    # NAS destination
CATEGORY = "print_ads"
LOG_FILE = "batch_disk_print_ads.log"
PROGRESS_FILE = "batch_disk_print_ads_progress.txt"

# Copy queue for background transfers (images from NAS)
copy_queue = queue.Queue()
copy_result_queue = queue.Queue()
copy_thread = None

# Move queue for background transfers (features to NAS - per file)
move_queue = queue.Queue()
move_thread = None
move_stats = {"moved": 0, "failed": 0, "pending": 0}

# Save queue for background processing (save .npz, queue move, delete source)
save_queue = queue.Queue()
save_thread = None
save_stats = {"saved": 0, "failed": 0, "pending": 0}

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def log(msg, color=None):
    """Print to console and append to log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if color:
        print(f"{color}{line}{RESET}", flush=True)
    else:
        print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def get_all_folders():
    """Get all subfolders from NAS source (e.g., subfolder0, subfolder1, etc.)."""
    folders = []
    for entry in os.listdir(NAS_SOURCE):
        full_path = os.path.join(NAS_SOURCE, entry)
        if os.path.isdir(full_path):
            folders.append(entry)
    return sorted(folders, key=natural_sort_key)


def count_images(folder_path):
    """Count images in a folder (recursive)."""
    extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                count += 1
    return count


def copy_folder_to_local(src_folder_name):
    """Copy a subfolder from NAS to local temp directory. Skips if already exists with images.

    e.g., 'subfolder0' -> C:\\print ads\\ebay\\subfolder0
    """
    src_path = os.path.join(NAS_SOURCE, src_folder_name)
    dest_path = os.path.join(LOCAL_TEMP, src_folder_name)

    # Check if destination already has images (pre-staged data)
    if os.path.exists(dest_path):
        existing_count = count_images(dest_path)
        if existing_count > 0:
            # Already has data, skip copy
            return dest_path

    # Ensure parent exists
    os.makedirs(LOCAL_TEMP, exist_ok=True)

    # Copy from NAS
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    shutil.copytree(src_path, dest_path)
    return dest_path


def delete_local_folder(local_path):
    """Delete a local temp folder after processing."""
    if os.path.exists(local_path):
        shutil.rmtree(local_path)


def copy_worker():
    """Background worker that copies folders from NAS to local."""
    while True:
        folder_name = copy_queue.get()
        if folder_name is None:  # Poison pill
            copy_queue.task_done()
            break

        try:
            start = time.time()
            local_path = copy_folder_to_local(folder_name)
            elapsed = time.time() - start
            img_count = count_images(local_path)
            copy_result_queue.put((folder_name, local_path, img_count, elapsed, None))
        except Exception as e:
            copy_result_queue.put((folder_name, None, 0, 0, str(e)))

        copy_queue.task_done()


def start_copy_thread():
    """Start the background copy worker."""
    global copy_thread
    copy_thread = threading.Thread(target=copy_worker, daemon=True)
    copy_thread.start()
    log("Copy worker thread started")


def stop_copy_thread():
    """Stop the copy worker thread."""
    copy_queue.put(None)
    copy_queue.join()


def queue_copy(folder_name):
    """Queue a folder for background copy."""
    copy_queue.put(folder_name)


def move_worker():
    """Background worker that moves .npz files from local to NAS one at a time."""
    global move_stats
    while True:
        item = move_queue.get()
        if item is None:  # Poison pill
            move_queue.task_done()
            break

        local_npz, nas_npz = item
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(nas_npz), exist_ok=True)
            # Move file
            shutil.move(local_npz, nas_npz)
            move_stats["moved"] += 1
        except Exception as e:
            # If move fails, file stays local (can be retried later)
            move_stats["failed"] += 1

        move_stats["pending"] -= 1
        move_queue.task_done()


def start_move_thread():
    """Start the background move worker."""
    global move_thread
    move_thread = threading.Thread(target=move_worker, daemon=True)
    move_thread.start()
    log("Move worker thread started (per-file NAS transfers)")


def stop_move_thread():
    """Stop the move worker thread and wait for queue to drain."""
    if move_stats["pending"] > 0:
        log(f"Waiting for {move_stats['pending']} pending file moves...")
    move_queue.put(None)
    move_queue.join()
    log(f"All moves complete: {move_stats['moved']} moved, {move_stats['failed']} failed")


def queue_move(local_npz, nas_npz):
    """Queue a single .npz file for background move to NAS."""
    global move_stats
    move_stats["pending"] += 1
    move_queue.put((local_npz, nas_npz))


# Global reference to indexer store for save worker
_indexer_store = None


def save_worker():
    """Background worker that saves .npz, queues move, deletes source image."""
    global save_stats, _indexer_store
    while True:
        item = save_queue.get()
        if item is None:  # Poison pill
            save_queue.task_done()
            break

        img_path, store_path, keypoints, descriptors, image_size, padded_size, folder_name = item
        saved_ok = False
        try:
            # Save to local
            _indexer_store.save(store_path, keypoints, descriptors, image_size, padded_size, folder_name)

            # Get local .npz path and queue move to NAS
            local_npz = _indexer_store._get_feature_path(store_path, folder_name)
            if local_npz and os.path.exists(str(local_npz)):
                nas_npz = str(local_npz).replace(LOCAL_FEATURES, NAS_FEATURES)
                queue_move(str(local_npz), nas_npz)

            save_stats["saved"] += 1
            saved_ok = True
        except Exception as e:
            if save_stats["failed"] < 3:
                log(f"SAVE ERROR: {e}")
            save_stats["failed"] += 1

        # Only delete source image if save succeeded
        if saved_ok:
            try:
                os.remove(img_path)
                # Try to delete parent folder if empty
                os.rmdir(os.path.dirname(img_path))
            except:
                pass

        save_stats["pending"] -= 1
        save_queue.task_done()


def start_save_thread():
    """Start the background save worker."""
    global save_thread
    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()
    log("Save worker thread started (async disk I/O)")


def stop_save_thread():
    """Stop the save worker thread and wait for queue to drain."""
    if save_stats["pending"] > 0:
        log(f"Waiting for {save_stats['pending']} pending saves...")
    save_queue.put(None)
    save_queue.join()
    log(f"All saves complete: {save_stats['saved']} saved, {save_stats['failed']} failed")


def queue_save(img_path, store_path, keypoints, descriptors, image_size, padded_size, folder_name):
    """Queue extracted features for background save."""
    global save_stats
    save_stats["pending"] += 1
    save_queue.put((img_path, store_path, keypoints, descriptors, image_size, padded_size, folder_name))


def main():
    log("=" * 70)
    log("BATCH DISK FEATURE INDEXER - PRINT ADS")
    log("=" * 70)
    log(f"Source: {NAS_SOURCE}")
    log(f"Local images: {LOCAL_TEMP}")
    log(f"Local features: {LOCAL_FEATURES}\\{CATEGORY}")
    log(f"NAS features: {NAS_FEATURES}\\{CATEGORY}")
    log("=" * 70)

    # Create directories
    os.makedirs(LOCAL_TEMP, exist_ok=True)
    os.makedirs(os.path.join(LOCAL_FEATURES, CATEGORY), exist_ok=True)
    os.makedirs(os.path.join(NAS_FEATURES, CATEGORY), exist_ok=True)

    # Start background threads
    start_copy_thread()
    start_move_thread()
    start_save_thread()

    # Import indexer (loads DISK model)
    from disk_indexer_file import DiskIndexerFile

    log("Loading DISK model...")
    indexer = DiskIndexerFile(
        category=CATEGORY,
        features_root=LOCAL_FEATURES,  # Write to local SSD
        batch_size=20,
        path_remap=(LOCAL_TEMP, NAS_SOURCE),  # Remap local paths to NAS paths in stored references
        show_progress=False,  # We handle our own progress
        device="cpu"  # CPU can be faster than CUDA for DISK
    )
    log("DISK model loaded.")

    # Set global store reference for save worker
    global _indexer_store
    _indexer_store = indexer.store

    # Get all folders from NAS
    all_folders = get_all_folders()
    total = len(all_folders)
    log(f"Total folders on NAS: {total}")

    # Load progress
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            completed = set(line.strip() for line in f if line.strip())
        log(f"Resuming: {len(completed)} folders already completed")

    # Filter out completed
    to_process = [f for f in all_folders if f not in completed]
    log(f"Remaining to process: {len(to_process)}")

    if not to_process:
        log("All folders already processed!")
        stop_copy_thread()
        stop_save_thread()
        stop_move_thread()
        return

    log("=" * 70)

    start_time = time.time()
    processed = 0
    copying_folders = set()  # Folders currently being copied

    def get_local_folders_with_images():
        """Get list of local folders that have images."""
        if not os.path.exists(LOCAL_TEMP):
            return []
        result = []
        for entry in os.listdir(LOCAL_TEMP):
            folder_path = os.path.join(LOCAL_TEMP, entry)
            if os.path.isdir(folder_path) and count_images(folder_path) > 0:
                result.append(entry)
        return sorted(result, key=natural_sort_key)

    def queue_next_copies():
        """Queue copies to maintain at least 2 local folders."""
        local_folders = get_local_folders_with_images()
        local_count = len(local_folders) + len(copying_folders)

        # Find folders that need to be copied
        for folder_name in to_process:
            if local_count >= 2:
                break
            if folder_name not in local_folders and folder_name not in copying_folders:
                queue_copy(folder_name)
                copying_folders.add(folder_name)
                log(f"Queued copy: {folder_name}", color=CYAN)
                local_count += 1

    # Initial queue of copies
    queue_next_copies()

    while to_process:
        # Check for completed copies
        while not copy_result_queue.empty():
            try:
                fname, local_path, img_count, copy_time, error = copy_result_queue.get_nowait()
                copying_folders.discard(fname)
                if error:
                    log(f"COPY ERROR: {fname} - {error}", color=YELLOW)
                elif copy_time > 0.1:
                    log(f"Copied: {fname} ({img_count} imgs, {copy_time:.1f}s)", color=CYAN)
            except:
                break

        # Get local folders with images
        local_folders = get_local_folders_with_images()

        # Find a folder to process (prefer ones in to_process order)
        folder_to_process = None
        for folder_name in to_process:
            if folder_name in local_folders:
                folder_to_process = folder_name
                break

        if folder_to_process is None:
            # No local folder ready - wait for a copy to complete
            if copying_folders:
                log(f"Waiting for copy... ({len(copying_folders)} in progress)")
                result = copy_result_queue.get()  # Block
                fname, local_path, img_count, copy_time, error = result
                copying_folders.discard(fname)
                if error:
                    log(f"COPY ERROR: {fname} - {error}", color=YELLOW)
                    to_process.remove(fname)
                    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
                        f.write(fname + '\n')
                elif copy_time > 0.1:
                    log(f"Copied: {fname} ({img_count} imgs, {copy_time:.1f}s)", color=CYAN)
                continue
            else:
                # Nothing to copy and nothing local - we're done
                break

        folder_name = folder_to_process
        current_local_path = os.path.join(LOCAL_TEMP, folder_name)

        # Queue more copies if needed
        queue_next_copies()

        # Process folder - extract only, save is async
        img_count = count_images(current_local_path)
        if img_count == 0:
            log(f"SKIP (empty): {folder_name}")
        else:
            idx_start = time.time()
            extracted = 0
            skipped = 0
            failed = 0

            # Get list of images (recursive)
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            images = []
            total_files = 0
            sample_extensions = set()
            for root, dirs, files in os.walk(current_local_path):
                total_files += len(files)
                for f in files:
                    ext = Path(f).suffix.lower()
                    if len(sample_extensions) < 10:
                        sample_extensions.add(ext)
                    if ext in image_extensions:
                        images.append(os.path.join(root, f))
            if len(images) == 0 and total_files > 0:
                log(f"No images found in {folder_name} (extensions: {sample_extensions})")

            total_imgs = len(images)
            for i, img_path in enumerate(images):
                try:
                    # Remap path for storage reference (local -> NAS)
                    store_path = img_path.replace(LOCAL_TEMP, NAS_SOURCE)

                    # Check if already exists on NAS
                    local_npz = indexer.store._get_feature_path(store_path, folder_name)
                    if local_npz:
                        nas_npz = str(local_npz).replace(LOCAL_FEATURES, NAS_FEATURES)
                        if os.path.exists(nas_npz):
                            skipped += 1
                            try:
                                os.remove(img_path)
                            except:
                                pass
                            continue

                    # Show progress every 10 images
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"\r  {folder_name}: {i+1}/{total_imgs} | extracted:{extracted} skip:{skipped} fail:{failed}    ", end="", flush=True)

                    # Extract features (CPU work)
                    result = indexer.extract_features(img_path)
                    if result is None:
                        failed += 1
                        os.remove(img_path)
                        continue

                    keypoints, descriptors, image_size, padded_size = result

                    # Queue for background save (non-blocking)
                    queue_save(img_path, store_path, keypoints, descriptors, image_size, padded_size, folder_name)
                    extracted += 1

                except Exception as e:
                    failed += 1
                    try:
                        os.remove(img_path)
                    except:
                        pass

            idx_time = time.time() - idx_start
            processed += 1
            remaining = len(to_process) - 1
            print()  # Newline after progress
            log(f"[{processed}/{processed+remaining}] {folder_name} | {extracted} extracted, {skipped} skip, {failed} fail | {idx_time:.1f}s | save:{save_stats['pending']} move:{move_stats['pending']}", color=GREEN)

        # Remove from to_process and mark completed
        to_process.remove(folder_name)
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(folder_name + '\n')

        # ETA
        if processed % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed if processed > 0 else 0
            eta_hours = (avg_time * len(to_process)) / 3600 if avg_time > 0 else 0
            log(f"Progress: {processed} done, {len(to_process)} remaining | ETA: {eta_hours:.1f}h")

    # Cleanup
    log("=" * 70)
    log("Extraction complete! Waiting for background workers...")
    stop_copy_thread()
    stop_save_thread()
    stop_move_thread()

    total_time = time.time() - start_time
    log(f"Total time: {total_time/3600:.1f} hours")
    log(f"Processed: {processed} folders")
    log(f"Saved: {save_stats['saved']}, failed: {save_stats['failed']}")
    log(f"Moved to NAS: {move_stats['moved']}, failed: {move_stats['failed']}")

    indexer.close()


if __name__ == "__main__":
    main()
