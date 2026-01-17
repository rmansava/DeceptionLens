r"""
File watcher for D:\books\pdf-images directory.

Automatically syncs book folder renames to all other locations:
- T:\archiverelated\books\pdf-images
- T:\archive\books\pdf-tesseract-text
- T:\disk-features\books
- T:\archive\books\pdf-full-file-text (text files)
- OpenSearch indexes (dinov2-books, faces-books)
- FAISS paths.json

Usage:
    python file_watcher.py              # Run watcher
    python file_watcher.py --dry-run    # Preview mode (no changes)
"""

import argparse
import logging
import os
import sys
import time
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
LOG_FILE = r"D:\books\pdf-images-watcher.log"

# Other locations to sync (exclude the watch path itself)
SYNC_LOCATIONS = [loc for loc in FOLDER_LOCATIONS if loc != WATCH_PATH]


class BookFolderHandler(FileSystemEventHandler):
    """Handle file system events in the books directory."""

    def __init__(self, logger: logging.Logger, dry_run: bool = False):
        self.logger = logger
        self.dry_run = dry_run
        self.opensearch_client = None
        self._connect_opensearch()
        super().__init__()

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

    def _sync_delete(self, book_name: str):
        """Delete a book from all locations."""
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

    def on_created(self, event: FileSystemEvent):
        """Handle file/folder creation - ignored."""
        pass

    def on_deleted(self, event: FileSystemEvent):
        """Handle file/folder deletion."""
        # Check if it's a direct child of the watch path
        if self._is_book_folder_event(event.src_path):
            book_name = os.path.basename(event.src_path)
            self.logger.info(f"[DELETED BOOK] {book_name}")
            self._sync_delete(book_name)

    def on_moved(self, event: FileSystemEvent):
        """Handle file/folder rename/move."""
        if event.is_directory and self._is_book_folder_event(event.src_path):
            old_name = os.path.basename(event.src_path)
            new_folder_path = event.dest_path

            # Check if moved outside watched directory (e.g., to Recycle Bin)
            if not new_folder_path.startswith(WATCH_PATH):
                self.logger.info(f"[DELETED BOOK] {old_name} (moved to Recycle Bin)")
                self._sync_delete(old_name)
            else:
                new_name = os.path.basename(event.dest_path)
                self.logger.info(f"[RENAMED BOOK] {old_name} -> {new_name}")
                self._sync_rename(old_name, new_name, new_folder_path)

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
