r"""
File watcher for D:\books\pdf-images directory.

Handles three types of events:
1. RENAME (within pdf-images): Syncs rename to all locations
2. CATEGORY MOVE (to different D:\books subfolder): Moves in NAS, updates paths
3. DELETE (moved outside D:\books): Removes from all locations

Syncs to:
- T:\archiverelated\books\pdf-images (and other categories)
- T:\archive\books\pdf-tesseract-text
- D:\disk-features\books, T:\disk-features\books
- T:\archive\books\pdf-full-file-text (text files)
- OpenSearch indexes (dinov2-books, faces-books)
- FAISS paths.json
- Disk progress files

Usage:
    python file_watcher.py              # Run watcher
    python file_watcher.py --dry-run    # Preview mode (no changes)
"""

import argparse
import logging
import os
import sys
import time
import threading
import queue
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from rename_book import (
    FOLDER_LOCATIONS,
    TEXT_FILE_LOCATION,
    FAISS_PATHS,
    OPENSEARCH_INDEXES,
    get_opensearch_client,
    rename_folder_and_contents,
    rename_files_in_folder,
    rename_all_files_to_folder_name,
    rename_text_file,
    update_opensearch_index,
    update_faiss_paths,
    update_disk_progress,
    delete_folder,
    delete_text_file,
    delete_from_opensearch,
    delete_from_faiss,
    delete_from_disk_progress,
)

WATCH_PATH = r"D:\books\pdf-images"
BOOKS_ROOT = r"D:\books"
LOG_FILE = r"D:\books\pdf-images-watcher.log"

# NAS mappings for category moves
# Maps local D:\books\{category} to NAS T:\archiverelated\books\{category}
NAS_PDF_IMAGES_ROOT = r"T:\archiverelated\books"

# Other locations to sync (exclude the watch path itself)
SYNC_LOCATIONS = [loc for loc in FOLDER_LOCATIONS if loc != WATCH_PATH]


class BookFolderHandler(FileSystemEventHandler):
    """Handle file system events in the books directory."""

    def __init__(self, logger: logging.Logger, dry_run: bool = False):
        self.logger = logger
        self.dry_run = dry_run
        self.opensearch_client = None
        self._connect_opensearch()

        # Event queue for tracking pending operations
        self.event_queue = queue.Queue()
        self.total_queued = 0
        self.processed = 0
        self.queue_lock = threading.Lock()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

        super().__init__()

    def _process_queue(self):
        """Worker thread to process queued events."""
        while True:
            try:
                event_type, args = self.event_queue.get()

                with self.queue_lock:
                    self.processed += 1
                    remaining = self.total_queued - self.processed
                    progress = f"[{self.processed}/{self.total_queued}]"
                    if remaining > 0:
                        progress += f" ({remaining} queued)"

                if event_type == "rename":
                    old_name, new_name, new_folder_path = args
                    self.logger.info(f"{progress} [RENAMED BOOK] {old_name} -> {new_name}")
                    self._sync_rename(old_name, new_name, new_folder_path)
                elif event_type == "delete":
                    book_name, reason = args
                    self.logger.info(f"{progress} [DELETED BOOK] {book_name}{reason}")
                    self._sync_delete(book_name)
                elif event_type == "move":
                    book_name, old_category, new_category = args
                    self.logger.info(f"{progress} [MOVED BOOK] {book_name} -> {new_category}/")
                    self._sync_category_move(book_name, old_category, new_category)

                self.event_queue.task_done()

                # Reset counters when queue is empty
                with self.queue_lock:
                    if self.event_queue.empty():
                        self.total_queued = 0
                        self.processed = 0

            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")

    def _queue_event(self, event_type: str, args: tuple):
        """Add an event to the queue."""
        with self.queue_lock:
            self.total_queued += 1
            queued = self.total_queued - self.processed
            if queued > 1:
                self.logger.info(f"Queued: {queued} events pending")
        self.event_queue.put((event_type, args))

    def _connect_opensearch(self):
        """Connect to OpenSearch."""
        try:
            self.opensearch_client = get_opensearch_client()
            self.logger.info("Connected to OpenSearch")
        except Exception as e:
            self.logger.warning(f"OpenSearch connection failed: {e}")
            self.opensearch_client = None

    def _is_book_folder_event(self, path: str) -> bool:
        """Check if event is for a direct child folder (book folder)."""
        rel_path = os.path.relpath(path, WATCH_PATH)
        return os.sep not in rel_path and rel_path != "."

    def _get_book_name(self, path: str) -> str:
        """Extract book folder name from path."""
        rel_path = os.path.relpath(path, WATCH_PATH)
        return rel_path.split(os.sep)[0]

    def _sync_rename(self, old_name: str, new_name: str, new_folder_path: str):
        """Sync a book rename to all other locations."""
        self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Syncing rename: {old_name} -> {new_name}")

        errors = 0

        # Step 0: Rename ALL files inside the watched folder to match new folder name
        # This handles both files with old name AND files with different names (from PDF metadata)
        result = rename_all_files_to_folder_name(new_folder_path, dry_run=self.dry_run)
        status = result.get("status", "unknown")
        if status == "success":
            renamed = result.get('renamed', 0)
            already_ok = result.get('already_correct', 0)
            self.logger.info(f"  [OK] pdf-images (local): {renamed} renamed, {already_ok} already correct")
        elif status == "dry-run":
            renamed = result.get('renamed', 0)
            already_ok = result.get('already_correct', 0)
            self.logger.info(f"  [DRY] pdf-images (local): would rename {renamed}, {already_ok} already correct")
        elif status == "partial":
            renamed = result.get('renamed', 0)
            errs = result.get('errors', 0)
            self.logger.warning(f"  [!!] pdf-images (local): {renamed} renamed, {errs} errors")
            if result.get('error_details'):
                for err in result['error_details'][:3]:
                    self.logger.warning(f"       {err}")
        elif status == "skipped":
            self.logger.debug(f"  [--] pdf-images (local): {result.get('message')}")
        else:
            errors += 1
            self.logger.error(f"  [ERR] pdf-images (local): {result.get('message')}")

        # Step 1: Rename in other folder locations
        for base_path in SYNC_LOCATIONS:
            result = rename_folder_and_contents(base_path, old_name, new_name, dry_run=self.dry_run)
            status = result.get("status", "unknown")
            if status == "success":
                files = result.get("files", {}).get("renamed", 0)
                self.logger.info(f"  [OK] {os.path.basename(base_path)}: {files} files")
            elif status == "dry-run":
                files = result.get("files", {}).get("renamed", 0)
                self.logger.info(f"  [DRY] {os.path.basename(base_path)}: would rename {files} files")
            elif status == "skipped":
                self.logger.debug(f"  [--] {os.path.basename(base_path)}: {result.get('message')}")
            else:
                errors += 1
                self.logger.error(f"  [ERR] {os.path.basename(base_path)}: {result.get('message')}")

        # Step 2: Rename text file
        result = rename_text_file(old_name, new_name, dry_run=self.dry_run)
        status = result.get("status", "unknown")
        if status == "success":
            self.logger.info(f"  [OK] text file renamed")
        elif status == "dry-run":
            self.logger.info(f"  [DRY] text file would be renamed")
        elif status == "skipped":
            self.logger.debug(f"  [--] text file: {result.get('message')}")
        else:
            errors += 1
            self.logger.error(f"  [ERR] text file: {result.get('message')}")

        # Step 3: Update OpenSearch indexes
        if self.opensearch_client:
            for idx in OPENSEARCH_INDEXES:
                result = update_opensearch_index(self.opensearch_client, idx, old_name, new_name, dry_run=self.dry_run)
                status = result.get("status", "unknown")
                if status == "success":
                    self.logger.info(f"  [OK] {idx}: {result.get('updated')} docs")
                elif status == "dry-run":
                    self.logger.info(f"  [DRY] {idx}: would update {result.get('would_update')} docs")
                elif status == "skipped":
                    self.logger.debug(f"  [--] {idx}: {result.get('message')}")
                else:
                    errors += 1
                    self.logger.error(f"  [ERR] {idx}: {result.get('message')}")

        # Step 4: Update FAISS paths
        for pf in FAISS_PATHS:
            result = update_faiss_paths(pf, old_name, new_name, dry_run=self.dry_run)
            status = result.get("status", "unknown")
            if status == "success":
                self.logger.info(f"  [OK] FAISS: {result.get('updated')} paths")
            elif status == "dry-run":
                self.logger.info(f"  [DRY] FAISS: would update {result.get('would_update')} paths")
            elif status == "skipped":
                self.logger.debug(f"  [--] FAISS: {result.get('message')}")
            else:
                errors += 1
                self.logger.error(f"  [ERR] FAISS: {result.get('message')}")

        # Step 5: Update disk indexer progress file
        result = update_disk_progress(old_name, new_name, dry_run=self.dry_run)
        status = result.get("status", "unknown")
        if status == "success":
            self.logger.info(f"  [OK] disk progress: updated")
        elif status == "dry-run":
            self.logger.info(f"  [DRY] disk progress: would update")
        elif status == "skipped":
            self.logger.debug(f"  [--] disk progress: {result.get('message')}")
        else:
            errors += 1
            self.logger.error(f"  [ERR] disk progress: {result.get('message')}")

        if errors == 0:
            self.logger.info(f"Sync complete for: {new_name}")
        else:
            self.logger.warning(f"Sync completed with {errors} error(s) for: {new_name}")

    def _check_moved_to_other_category(self, book_name: str):
        """Check if a book was moved to another category within D:\books."""
        import time
        time.sleep(0.5)  # Brief delay for file system to settle

        old_category = os.path.basename(WATCH_PATH)  # pdf-images

        # Check all subfolders of D:\books for this book
        try:
            for subfolder in os.listdir(BOOKS_ROOT):
                if subfolder == old_category:
                    continue  # Skip the source folder
                subfolder_path = os.path.join(BOOKS_ROOT, subfolder)
                if os.path.isdir(subfolder_path):
                    book_path = os.path.join(subfolder_path, book_name)
                    if os.path.exists(book_path):
                        return subfolder  # Found it in another category
        except Exception:
            pass
        return None

    def _sync_delete(self, book_name: str):
        """Delete a book from all locations."""
        # First check if this was actually a move to another category (cut/paste)
        new_category = self._check_moved_to_other_category(book_name)
        if new_category:
            old_category = os.path.basename(WATCH_PATH)
            self.logger.info(f"Detected category move (cut/paste): {book_name} -> {new_category}/")
            self._sync_category_move(book_name, old_category, new_category)
            return

        self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Syncing delete: {book_name}")

        errors = 0

        # Step 1: Delete from other folder locations
        for base_path in SYNC_LOCATIONS:
            result = delete_folder(base_path, book_name, dry_run=self.dry_run)
            status = result.get("status", "unknown")
            if status == "success":
                self.logger.info(f"  [OK] {os.path.basename(base_path)}: {result.get('deleted')} files deleted")
            elif status == "dry-run":
                self.logger.info(f"  [DRY] {os.path.basename(base_path)}: would delete {result.get('would_delete')} files")
            elif status == "skipped":
                self.logger.debug(f"  [--] {os.path.basename(base_path)}: {result.get('message')}")
            else:
                errors += 1
                self.logger.error(f"  [ERR] {os.path.basename(base_path)}: {result.get('message')}")

        # Step 2: Delete text file
        result = delete_text_file(book_name, dry_run=self.dry_run)
        status = result.get("status", "unknown")
        if status == "success":
            self.logger.info(f"  [OK] text file deleted")
        elif status == "dry-run":
            self.logger.info(f"  [DRY] text file would be deleted")
        elif status == "skipped":
            self.logger.debug(f"  [--] text file: {result.get('message')}")
        else:
            errors += 1
            self.logger.error(f"  [ERR] text file: {result.get('message')}")

        # Step 3: Delete from OpenSearch indexes
        if self.opensearch_client:
            for idx in OPENSEARCH_INDEXES:
                result = delete_from_opensearch(self.opensearch_client, idx, book_name, dry_run=self.dry_run)
                status = result.get("status", "unknown")
                if status == "success":
                    self.logger.info(f"  [OK] {idx}: {result.get('deleted')} docs deleted")
                elif status == "dry-run":
                    self.logger.info(f"  [DRY] {idx}: would delete {result.get('would_delete')} docs")
                elif status == "skipped":
                    self.logger.debug(f"  [--] {idx}: {result.get('message')}")
                else:
                    errors += 1
                    self.logger.error(f"  [ERR] {idx}: {result.get('message')}")

        # Step 4: Delete from FAISS paths
        for pf in FAISS_PATHS:
            result = delete_from_faiss(pf, book_name, dry_run=self.dry_run)
            status = result.get("status", "unknown")
            if status == "success":
                self.logger.info(f"  [OK] FAISS: {result.get('deleted')} paths removed")
            elif status == "dry-run":
                self.logger.info(f"  [DRY] FAISS: would remove {result.get('would_delete')} paths")
            elif status == "skipped":
                self.logger.debug(f"  [--] FAISS: {result.get('message')}")
            else:
                errors += 1
                self.logger.error(f"  [ERR] FAISS: {result.get('message')}")

        # Step 5: Remove from disk indexer progress file
        result = delete_from_disk_progress(book_name, dry_run=self.dry_run)
        status = result.get("status", "unknown")
        if status == "success":
            self.logger.info(f"  [OK] disk progress: removed")
        elif status == "dry-run":
            self.logger.info(f"  [DRY] disk progress: would remove")
        elif status == "skipped":
            self.logger.debug(f"  [--] disk progress: {result.get('message')}")
        else:
            errors += 1
            self.logger.error(f"  [ERR] disk progress: {result.get('message')}")

        if errors == 0:
            self.logger.info(f"Delete complete for: {book_name}")
        else:
            self.logger.warning(f"Delete completed with {errors} error(s) for: {book_name}")

    def _sync_category_move(self, book_name: str, old_category: str, new_category: str):
        """Move a book from one category to another in NAS locations."""
        self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Syncing category move: {book_name}")
        self.logger.info(f"  From: {old_category} -> To: {new_category}")

        errors = 0
        import shutil

        # Move in NAS pdf-images location
        old_nas_path = os.path.join(NAS_PDF_IMAGES_ROOT, old_category, book_name)
        new_nas_path = os.path.join(NAS_PDF_IMAGES_ROOT, new_category, book_name)

        if os.path.exists(old_nas_path):
            if self.dry_run:
                self.logger.info(f"  [DRY] NAS pdf-images: would move to {new_category}")
            else:
                try:
                    os.makedirs(os.path.dirname(new_nas_path), exist_ok=True)
                    shutil.move(old_nas_path, new_nas_path)
                    self.logger.info(f"  [OK] NAS pdf-images: moved to {new_category}")
                except Exception as e:
                    errors += 1
                    self.logger.error(f"  [ERR] NAS pdf-images: {e}")
        else:
            self.logger.debug(f"  [--] NAS pdf-images: not found in {old_category}")

        # Move pdf-tesseract-text folder
        tesseract_base = r"T:\archive\books\pdf-tesseract-text"
        old_tesseract = os.path.join(tesseract_base, book_name)
        # Note: tesseract doesn't have category subfolders, so we just leave it
        # The path stays the same since it's organized by book name, not category

        # Move disk-features folders (local and NAS)
        for df_base in [r"D:\disk-features\books", r"T:\disk-features\books"]:
            old_df = os.path.join(df_base, book_name)
            if os.path.exists(old_df):
                # disk-features doesn't have category subfolders either, just delete since
                # it would need re-indexing for the new category anyway
                if self.dry_run:
                    self.logger.info(f"  [DRY] {os.path.basename(df_base)}: would delete (needs re-indexing)")
                else:
                    try:
                        shutil.rmtree(old_df)
                        self.logger.info(f"  [OK] {os.path.basename(df_base)}: deleted (needs re-indexing)")
                    except Exception as e:
                        errors += 1
                        self.logger.error(f"  [ERR] {os.path.basename(df_base)}: {e}")

        # Delete from disk progress files (book needs re-indexing for new category)
        result = delete_from_disk_progress(book_name, dry_run=self.dry_run)
        status = result.get("status", "unknown")
        if status == "success":
            self.logger.info(f"  [OK] disk progress: removed (needs re-indexing)")
        elif status == "dry-run":
            self.logger.info(f"  [DRY] disk progress: would remove")
        elif status == "skipped":
            self.logger.debug(f"  [--] disk progress: {result.get('message')}")

        # Update OpenSearch paths (change category in path)
        if self.opensearch_client:
            old_path_prefix = f"T:/archiverelated/books/{old_category}/{book_name}"
            new_path_prefix = f"T:/archiverelated/books/{new_category}/{book_name}"
            for idx in OPENSEARCH_INDEXES:
                try:
                    # Search for documents with old path
                    search_body = {
                        "query": {"prefix": {"image_path": old_path_prefix}},
                        "size": 10000
                    }
                    result = self.opensearch_client.search(index=idx, body=search_body)
                    hits = result.get("hits", {}).get("hits", [])

                    if hits:
                        if self.dry_run:
                            self.logger.info(f"  [DRY] {idx}: would update {len(hits)} docs")
                        else:
                            # Update each document
                            for hit in hits:
                                doc_id = hit["_id"]
                                old_path = hit["_source"]["image_path"]
                                new_path = old_path.replace(f"/{old_category}/", f"/{new_category}/")
                                self.opensearch_client.update(
                                    index=idx,
                                    id=doc_id,
                                    body={"doc": {"image_path": new_path}}
                                )
                            self.logger.info(f"  [OK] {idx}: {len(hits)} docs updated")
                    else:
                        self.logger.debug(f"  [--] {idx}: no docs found")
                except Exception as e:
                    errors += 1
                    self.logger.error(f"  [ERR] {idx}: {e}")

        # Update FAISS paths
        for pf in FAISS_PATHS:
            try:
                if os.path.exists(pf):
                    import json
                    with open(pf, 'r', encoding='utf-8') as f:
                        paths = json.load(f)

                    old_prefix = f"T:/archiverelated/books/{old_category}/{book_name}"
                    updated = 0
                    new_paths = []
                    for p in paths:
                        if p.startswith(old_prefix):
                            new_p = p.replace(f"/{old_category}/", f"/{new_category}/")
                            new_paths.append(new_p)
                            updated += 1
                        else:
                            new_paths.append(p)

                    if updated > 0:
                        if self.dry_run:
                            self.logger.info(f"  [DRY] FAISS: would update {updated} paths")
                        else:
                            with open(pf, 'w', encoding='utf-8') as f:
                                json.dump(new_paths, f)
                            self.logger.info(f"  [OK] FAISS: {updated} paths updated")
                    else:
                        self.logger.debug(f"  [--] FAISS: no matching paths")
            except Exception as e:
                errors += 1
                self.logger.error(f"  [ERR] FAISS: {e}")

        if errors == 0:
            self.logger.info(f"Category move complete for: {book_name}")
        else:
            self.logger.warning(f"Category move completed with {errors} error(s) for: {book_name}")

    def on_created(self, event: FileSystemEvent):
        """Handle file/folder creation - ignored."""
        pass

    def on_deleted(self, event: FileSystemEvent):
        """Handle file/folder deletion."""
        # Check if it's a direct child of the watch path
        if self._is_book_folder_event(event.src_path):
            book_name = os.path.basename(event.src_path)
            self._queue_event("delete", (book_name, ""))

    def on_moved(self, event: FileSystemEvent):
        """Handle file/folder rename/move."""
        if event.is_directory and self._is_book_folder_event(event.src_path):
            old_name = os.path.basename(event.src_path)
            new_folder_path = event.dest_path

            # Check if moved within same directory (rename)
            if new_folder_path.startswith(WATCH_PATH):
                new_name = os.path.basename(event.dest_path)
                self._queue_event("rename", (old_name, new_name, new_folder_path))

            # Check if moved to a different category within D:\books
            elif new_folder_path.startswith(BOOKS_ROOT):
                # Extract the new category (subfolder name)
                rel_path = os.path.relpath(new_folder_path, BOOKS_ROOT)
                new_category = rel_path.split(os.sep)[0]
                old_category = os.path.basename(WATCH_PATH)  # pdf-images
                book_name = os.path.basename(new_folder_path)
                self._queue_event("move", (book_name, old_category, new_category))

            # Moved completely outside D:\books (Recycle Bin, etc.)
            else:
                self._queue_event("delete", (old_name, " (moved outside books folder)"))

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification - ignored for books."""
        pass


def setup_logger(log_to_file: bool = True) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("BookWatcher")
    logger.setLevel(logging.DEBUG)

    # Console handler - INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler - DEBUG and above
    if log_to_file:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description="Watch D:\\books\\pdf-images for changes and sync renames")
    parser.add_argument("--no-log-file", action="store_true", help="Disable logging to file")
    parser.add_argument("--dry-run", action="store_true", help="Preview mode - don't make changes")
    args = parser.parse_args()

    if not os.path.exists(WATCH_PATH):
        print(f"Error: Watch path does not exist: {WATCH_PATH}")
        sys.exit(1)

    logger = setup_logger(log_to_file=not args.no_log_file)

    # Count current books
    book_count = len([d for d in os.listdir(WATCH_PATH) if os.path.isdir(os.path.join(WATCH_PATH, d))])

    print("=" * 60)
    print("  Book Folder Watcher + Auto-Sync (Rename & Delete)")
    print("=" * 60)
    print(f"  Watching: {WATCH_PATH}")
    print(f"  Current books: {book_count}")
    print(f"  Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will sync renames)'}")
    if not args.no_log_file:
        print(f"  Log file: {LOG_FILE}")
    print()
    print("  Syncs to:")
    for loc in SYNC_LOCATIONS:
        print(f"    - {loc}")
    print(f"    - {TEXT_FILE_LOCATION} (text files)")
    print(f"    - OpenSearch: {', '.join(OPENSEARCH_INDEXES)}")
    print(f"    - FAISS: {', '.join(FAISS_PATHS)}")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print()

    event_handler = BookFolderHandler(logger, dry_run=args.dry_run)
    observer = Observer()
    observer.schedule(event_handler, WATCH_PATH, recursive=False)  # Only watch top level
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()

    observer.join()
    print("Done.")


if __name__ == "__main__":
    main()
