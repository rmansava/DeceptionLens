"""
Rename a book across all file locations and indexes.

Usage:
    python rename_book.py                      # Interactive mode
    python rename_book.py --old "X" --new "Y"  # Command line mode
    python rename_book.py --dry-run            # Preview changes only

Note: Apostrophe encoding is auto-detected (handles ' vs ' differences).
"""

import argparse
import json
import os
import shutil
import sys
import time
from opensearchpy import OpenSearch

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200

# Quote characters to normalize
QUOTE_CHARS = "''\u2019\u2018`"


def strip_quotes(s):
    """Remove all quote characters for comparison."""
    return "".join(c for c in s if c not in QUOTE_CHARS)


def normalize_apostrophes(name):
    """Find actual folder name by matching with quotes stripped."""
    base_path = r"D:\books\pdf-images"
    if os.path.exists(os.path.join(base_path, name)):
        return name

    # Strip quotes from input for comparison
    name_stripped = strip_quotes(name)

    # Search for matching folder
    try:
        for folder in os.listdir(base_path):
            if strip_quotes(folder) == name_stripped:
                return folder
    except Exception:
        pass

    return name


FOLDER_LOCATIONS = [
    r"D:\books\pdf-images",
    r"T:\archiverelated\books\pdf-images",
    r"T:\archive\books\pdf-tesseract-text",
    r"D:\disk-features\books",
    r"T:\disk-features\books",
]

TEXT_FILE_LOCATION = r"T:\archive\books\pdf-full-file-text"
FAISS_PATHS = ["D:/faiss/books/paths.json"]
OPENSEARCH_INDEXES = ["dinov2-books", "faces-books"]
DISK_PROGRESS_FILE = r"C:\Users\rmans\source\repos\DinoDeceptionLens\backend\batch_disk_progress.txt"
DISK_PROGRESS_FILE_BASED = r"C:\Users\rmans\source\repos\DinoDeceptionLens\backend\batch_disk_progress_file.txt"


def get_opensearch_client():
    return OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}], http_compress=True, timeout=300)


def rename_all_files_to_folder_name(folder_path, dry_run):
    """Rename ALL files in folder to match the folder name (extracts page numbers).

    Used when folder is renamed but file names don't contain the old folder name
    (e.g., files named from PDF metadata).
    """
    import re
    if not os.path.exists(folder_path):
        return {"status": "skipped", "message": "Folder not found"}

    folder_name = os.path.basename(folder_path)
    folder_name_lower = folder_name.lower()
    renamed_count = 0
    already_correct = 0
    errors = []

    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))]
        total = len(files)

        for i, fn in enumerate(files):
            # Skip if already starts with folder name (case-insensitive)
            if fn.lower().startswith(folder_name_lower):
                already_correct += 1
                continue

            # Extract page number
            match = re.search(r'-page(\d+)\.', fn, re.IGNORECASE)
            if not match:
                errors.append(f"No page number: {fn}")
                continue

            page_num = match.group(1)
            ext = os.path.splitext(fn)[1]
            new_fn = f"{folder_name}-page{page_num}{ext}"

            old_fp = os.path.join(folder_path, fn)
            new_fp = os.path.join(folder_path, new_fn)

            if os.path.exists(new_fp):
                errors.append(f"Target exists: {new_fn}")
                continue

            if dry_run:
                renamed_count += 1
            else:
                try:
                    os.rename(old_fp, new_fp)
                    renamed_count += 1
                except Exception as e:
                    errors.append(str(e))

            if (i + 1) % 50 == 0:
                print(f"      Progress: {i+1}/{total} files...")

        if errors:
            return {"status": "partial", "renamed": renamed_count, "already_correct": already_correct, "errors": len(errors), "error_details": errors[:5]}
        return {"status": "success" if not dry_run else "dry-run", "renamed": renamed_count, "already_correct": already_correct, "total": total}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def rename_files_in_folder(folder_path, old_name, new_name, dry_run):
    if not os.path.exists(folder_path):
        return {"status": "skipped", "message": "Folder not found"}
    renamed_count = 0
    errors = []
    try:
        files = os.listdir(folder_path)
        total = len(files)
        for i, fn in enumerate(files):
            if old_name in fn:
                old_fp = os.path.join(folder_path, fn)
                new_fn = fn.replace(old_name, new_name)
                new_fp = os.path.join(folder_path, new_fn)
                if dry_run:
                    renamed_count += 1
                else:
                    try:
                        os.rename(old_fp, new_fp)
                        renamed_count += 1
                    except Exception as e:
                        errors.append(str(e))
            if (i + 1) % 50 == 0:
                print(f"      Progress: {i+1}/{total} files...")
        if errors:
            return {"status": "partial", "renamed": renamed_count, "errors": len(errors)}
        # Verify: check no files with old_name remain (skip if new_name contains old_name)
        if not dry_run and old_name not in new_name:
            remaining = sum(1 for fn in os.listdir(folder_path) if old_name in fn)
            if remaining > 0:
                return {"status": "error", "message": f"Verify failed: {remaining} files still have old name", "renamed": renamed_count}
        return {"status": "success" if not dry_run else "dry-run", "renamed": renamed_count, "total": total}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def rename_folder_and_contents(base_path, old_name, new_name, dry_run):
    old_folder = os.path.join(base_path, old_name)
    new_folder = os.path.join(base_path, new_name)
    if not os.path.exists(old_folder):
        return {"path": base_path, "status": "skipped", "message": "Folder not found"}
    if os.path.exists(new_folder):
        return {"path": base_path, "status": "error", "message": "Target exists"}
    print(f"      Renaming files inside folder...")
    file_result = rename_files_in_folder(old_folder, old_name, new_name, dry_run)
    if not dry_run:
        # Try to rename folder, with retry on failure
        folder_renamed = False
        for attempt in range(3):
            try:
                os.rename(old_folder, new_folder)
                folder_renamed = True
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5)  # Brief pause before retry
                else:
                    last_error = str(e)
        if not folder_renamed:
            return {"path": base_path, "status": "error", "message": last_error, "files": file_result}
        # Verify folder was renamed
        if not os.path.exists(new_folder):
            return {"path": base_path, "status": "error", "message": "Verify failed: new folder not found", "files": file_result}
    return {"path": base_path, "status": "success" if not dry_run else "dry-run", "renamed_to": new_folder, "files": file_result}


def rename_text_file(old_name, new_name, dry_run):
    old_file = os.path.join(TEXT_FILE_LOCATION, f"{old_name}.txt")
    new_file = os.path.join(TEXT_FILE_LOCATION, f"{new_name}.txt")
    if not os.path.exists(old_file):
        return {"path": TEXT_FILE_LOCATION, "status": "skipped", "message": "File not found"}
    if os.path.exists(new_file):
        return {"path": TEXT_FILE_LOCATION, "status": "error", "message": "Target exists"}
    if dry_run:
        return {"path": TEXT_FILE_LOCATION, "status": "dry-run", "would_rename_to": new_file}
    try:
        os.rename(old_file, new_file)
        # Verify
        if not os.path.exists(new_file):
            return {"path": TEXT_FILE_LOCATION, "status": "error", "message": "Verify failed: new file not found"}
        return {"path": TEXT_FILE_LOCATION, "status": "success", "renamed_to": new_file}
    except Exception as e:
        return {"path": TEXT_FILE_LOCATION, "status": "error", "message": str(e)}


def update_opensearch_index(client, index_name, old_name, new_name, dry_run):
    count_query = {"query": {"bool": {"should": [{"match_phrase": {"book": old_name}}, {"wildcard": {"path": f"*{old_name}*"}}], "minimum_should_match": 1}}}
    try:
        count_result = client.count(index=index_name, body=count_query)
        doc_count = count_result.get("count", 0)
    except Exception as e:
        return {"index": index_name, "status": "error", "message": str(e)}
    if doc_count == 0:
        return {"index": index_name, "status": "skipped", "message": "No matching documents"}
    if dry_run:
        return {"index": index_name, "status": "dry-run", "would_update": doc_count}
    update_script = "if (ctx._source.book != null && ctx._source.book.contains(params.old_name)) { ctx._source.book = ctx._source.book.replace(params.old_name, params.new_name); } if (ctx._source.path != null && ctx._source.path.contains(params.old_name)) { ctx._source.path = ctx._source.path.replace(params.old_name, params.new_name); } if (ctx._source.filename != null && ctx._source.filename.contains(params.old_name)) { ctx._source.filename = ctx._source.filename.replace(params.old_name, params.new_name); }"
    update_body = {"query": count_query["query"], "script": {"source": update_script, "params": {"old_name": old_name, "new_name": new_name}}}
    try:
        print(f"      Updating {doc_count} documents...")
        result = client.update_by_query(index=index_name, body=update_body, wait_for_completion=True, refresh=True)
        updated = result.get("updated", 0)
        failures = result.get("failures", [])
        if failures:
            return {"index": index_name, "status": "partial", "updated": updated, "failures": len(failures)}
        # Verify: check no old_name docs remain (skip if new_name contains old_name)
        if old_name not in new_name:
            verify_count = client.count(index=index_name, body=count_query).get("count", 0)
            if verify_count > 0:
                return {"index": index_name, "status": "error", "message": f"Verify failed: {verify_count} docs still have old name"}
        return {"index": index_name, "status": "success", "updated": updated}
    except Exception as e:
        return {"index": index_name, "status": "error", "message": str(e)}


def update_faiss_paths(paths_file, old_name, new_name, dry_run):
    if not os.path.exists(paths_file):
        return {"file": paths_file, "status": "skipped", "message": "File not found"}
    try:
        with open(paths_file, "r", encoding="utf-8") as f:
            paths = json.load(f)
    except Exception as e:
        return {"file": paths_file, "status": "error", "message": f"Read failed: {e}"}
    updated_count = 0
    new_paths = []
    for path in paths:
        if old_name in path:
            new_paths.append(path.replace(old_name, new_name))
            updated_count += 1
        else:
            new_paths.append(path)
    if updated_count == 0:
        return {"file": paths_file, "status": "skipped", "message": "No matching paths"}
    if dry_run:
        return {"file": paths_file, "status": "dry-run", "would_update": updated_count}
    backup = paths_file + ".backup"
    try:
        shutil.copy2(paths_file, backup)
        with open(paths_file, "w", encoding="utf-8") as f:
            json.dump(new_paths, f)
        # Verify: re-read and check no old_name entries remain (skip if new_name contains old_name)
        if old_name not in new_name:
            with open(paths_file, "r", encoding="utf-8") as f:
                verify_paths = json.load(f)
            remaining = sum(1 for p in verify_paths if old_name in p)
            if remaining > 0:
                return {"file": paths_file, "status": "error", "message": f"Verify failed: {remaining} paths still have old name"}
        return {"file": paths_file, "status": "success", "updated": updated_count}
    except Exception as e:
        if os.path.exists(backup):
            shutil.copy2(backup, paths_file)
        return {"file": paths_file, "status": "error", "message": str(e)}


def _update_single_progress_file(filepath, old_name, new_name, dry_run):
    """Update a single progress file."""
    if not os.path.exists(filepath):
        return {"status": "skipped", "message": "File not found"}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return {"status": "error", "message": f"Read failed: {e}"}

    updated_count = 0
    new_lines = []
    for line in lines:
        stripped = line.rstrip('\n')
        if stripped == old_name:
            new_lines.append(new_name + '\n')
            updated_count += 1
        else:
            new_lines.append(line)

    if updated_count == 0:
        return {"status": "skipped", "message": "No matching entry"}
    if dry_run:
        return {"status": "dry-run", "would_update": updated_count}

    backup = filepath + ".backup"
    try:
        shutil.copy2(filepath, backup)
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return {"status": "success", "updated": updated_count}
    except Exception as e:
        if os.path.exists(backup):
            shutil.copy2(backup, filepath)
        return {"status": "error", "message": str(e)}


def update_disk_progress(old_name, new_name, dry_run):
    """Update both disk indexer progress files (SQL and file-based)."""
    results = []
    total_updated = 0

    for filepath in [DISK_PROGRESS_FILE, DISK_PROGRESS_FILE_BASED]:
        result = _update_single_progress_file(filepath, old_name, new_name, dry_run)
        results.append((os.path.basename(filepath), result))
        if result.get("status") == "success":
            total_updated += result.get("updated", 0)
        elif result.get("status") == "dry-run":
            total_updated += result.get("would_update", 0)

    # Return combined result
    if any(r[1].get("status") == "error" for r in results):
        errors = [f"{name}: {r.get('message')}" for name, r in results if r.get("status") == "error"]
        return {"status": "error", "message": "; ".join(errors)}
    if total_updated == 0:
        return {"status": "skipped", "message": "No matching entry in any file"}
    if dry_run:
        return {"status": "dry-run", "would_update": total_updated}
    return {"status": "success", "updated": total_updated}


def _delete_from_single_progress_file(filepath, book_name, dry_run):
    """Delete from a single progress file."""
    if not os.path.exists(filepath):
        return {"status": "skipped", "message": "File not found"}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return {"status": "error", "message": f"Read failed: {e}"}

    new_lines = [line for line in lines if line.rstrip('\n') != book_name]
    deleted_count = len(lines) - len(new_lines)

    if deleted_count == 0:
        return {"status": "skipped", "message": "No matching entry"}
    if dry_run:
        return {"status": "dry-run", "would_delete": deleted_count}

    backup = filepath + ".backup"
    try:
        shutil.copy2(filepath, backup)
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return {"status": "success", "deleted": deleted_count}
    except Exception as e:
        if os.path.exists(backup):
            shutil.copy2(backup, filepath)
        return {"status": "error", "message": str(e)}


def delete_from_disk_progress(book_name, dry_run):
    """Remove a book from both disk indexer progress files (SQL and file-based)."""
    results = []
    total_deleted = 0

    for filepath in [DISK_PROGRESS_FILE, DISK_PROGRESS_FILE_BASED]:
        result = _delete_from_single_progress_file(filepath, book_name, dry_run)
        results.append((os.path.basename(filepath), result))
        if result.get("status") == "success":
            total_deleted += result.get("deleted", 0)
        elif result.get("status") == "dry-run":
            total_deleted += result.get("would_delete", 0)

    # Return combined result
    if any(r[1].get("status") == "error" for r in results):
        errors = [f"{name}: {r.get('message')}" for name, r in results if r.get("status") == "error"]
        return {"status": "error", "message": "; ".join(errors)}
    if total_deleted == 0:
        return {"status": "skipped", "message": "No matching entry in any file"}
    if dry_run:
        return {"status": "dry-run", "would_delete": total_deleted}
    return {"status": "success", "deleted": total_deleted}


def delete_folder(base_path, book_name, dry_run):
    """Delete a book folder and all its contents."""
    folder_path = os.path.join(base_path, book_name)
    if not os.path.exists(folder_path):
        return {"path": base_path, "status": "skipped", "message": "Folder not found"}
    try:
        file_count = len(os.listdir(folder_path))
        if dry_run:
            return {"path": base_path, "status": "dry-run", "would_delete": file_count}
        shutil.rmtree(folder_path)
        if os.path.exists(folder_path):
            return {"path": base_path, "status": "error", "message": "Delete failed - folder still exists"}
        return {"path": base_path, "status": "success", "deleted": file_count}
    except Exception as e:
        return {"path": base_path, "status": "error", "message": str(e)}


def delete_text_file(book_name, dry_run):
    """Delete the text file for a book."""
    text_file = os.path.join(TEXT_FILE_LOCATION, f"{book_name}.txt")
    if not os.path.exists(text_file):
        return {"path": TEXT_FILE_LOCATION, "status": "skipped", "message": "Text file not found"}
    try:
        if dry_run:
            return {"path": TEXT_FILE_LOCATION, "status": "dry-run", "would_delete": 1}
        os.remove(text_file)
        if os.path.exists(text_file):
            return {"path": TEXT_FILE_LOCATION, "status": "error", "message": "Delete failed - file still exists"}
        return {"path": TEXT_FILE_LOCATION, "status": "success", "deleted": 1}
    except Exception as e:
        return {"path": TEXT_FILE_LOCATION, "status": "error", "message": str(e)}


def delete_from_opensearch(client, index_name, book_name, dry_run):
    """Delete all documents for a book from OpenSearch index."""
    delete_query = {
        "query": {
            "bool": {
                "should": [
                    {"wildcard": {"book": f"*{book_name}*"}},
                    {"wildcard": {"path": f"*{book_name}*"}},
                    {"wildcard": {"filename": f"*{book_name}*"}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    try:
        count_result = client.count(index=index_name, body=delete_query)
        doc_count = count_result.get("count", 0)
    except Exception as e:
        return {"index": index_name, "status": "error", "message": str(e)}
    if doc_count == 0:
        return {"index": index_name, "status": "skipped", "message": "No matching documents"}
    if dry_run:
        return {"index": index_name, "status": "dry-run", "would_delete": doc_count}
    try:
        print(f"      Deleting {doc_count} documents...")
        result = client.delete_by_query(index=index_name, body=delete_query, wait_for_completion=True, refresh=True)
        deleted = result.get("deleted", 0)
        failures = result.get("failures", [])
        if failures:
            return {"index": index_name, "status": "partial", "deleted": deleted, "failures": len(failures)}
        return {"index": index_name, "status": "success", "deleted": deleted}
    except Exception as e:
        return {"index": index_name, "status": "error", "message": str(e)}


def delete_from_faiss(paths_file, book_name, dry_run):
    """Remove paths for a book from FAISS paths.json."""
    if not os.path.exists(paths_file):
        return {"file": paths_file, "status": "skipped", "message": "File not found"}
    try:
        with open(paths_file, "r", encoding="utf-8") as f:
            paths = json.load(f)
    except Exception as e:
        return {"file": paths_file, "status": "error", "message": f"Read failed: {e}"}

    # Filter out paths containing book_name
    new_paths = [p for p in paths if book_name not in p]
    deleted_count = len(paths) - len(new_paths)

    if deleted_count == 0:
        return {"file": paths_file, "status": "skipped", "message": "No matching paths"}
    if dry_run:
        return {"file": paths_file, "status": "dry-run", "would_delete": deleted_count}

    backup = paths_file + ".backup"
    try:
        shutil.copy2(paths_file, backup)
        with open(paths_file, "w", encoding="utf-8") as f:
            json.dump(new_paths, f)
        return {"file": paths_file, "status": "success", "deleted": deleted_count}
    except Exception as e:
        if os.path.exists(backup):
            shutil.copy2(backup, paths_file)
        return {"file": paths_file, "status": "error", "message": str(e)}


def print_status(name, result):
    status = result.get("status", "unknown")
    if status == "success":
        files_info = result.get("files", {})
        if files_info and files_info.get("renamed"):
            print(f"    [OK] {name}: folder + {files_info['renamed']} files")
        else:
            print(f"    [OK] {name}: {result.get('updated', 'done')}")
    elif status == "dry-run":
        files_info = result.get("files", {})
        if files_info and files_info.get("renamed"):
            print(f"    [DRY] {name}: would rename folder + {files_info['renamed']} files")
        else:
            print(f"    [DRY] {name}: would update {result.get('would_update', 'yes')}")
    elif status == "skipped":
        print(f"    [--] {name}: {result.get('message', 'skipped')}")
    elif status == "partial":
        print(f"    [!!] {name}: {result.get('updated', 0)} ok, {result.get('failures', 0)} failed")
    else:
        print(f"    [ERR] {name}: {result.get('message', 'failed')}")


def main():
    parser = argparse.ArgumentParser(description="Rename book across all locations")
    parser.add_argument("--old", help="Old book name")
    parser.add_argument("--new", help="New book name")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    if not args.old:
        print("=" * 70)
        print("  Book Rename Tool")
        print("=" * 70)
        print("\nEnter OLD book name (copy-paste from explorer for exact encoding):")
        args.old = input("> ").strip()
        if not args.old:
            print("Error: Old name required")
            sys.exit(1)

    if not args.new:
        print("\nEnter NEW book name:")
        args.new = input("> ").strip()
        if not args.new:
            print("Error: New name required")
            sys.exit(1)

    old_name, new_name, dry_run = args.old, args.new, args.dry_run

    # Auto-detect correct apostrophe encoding
    original_old = old_name
    old_name = normalize_apostrophes(old_name)
    if old_name != original_old:
        print(f"\n  [Auto-detected apostrophe encoding]")

    print("\n" + "=" * 70)
    print(f"  Old: {old_name}")
    print(f"  New: {new_name}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 70)

    results = []

    print("\n" + "-" * 70)
    print("  STEP 1: Renaming folders and files")
    print("-" * 70)
    for i, base_path in enumerate(FOLDER_LOCATIONS):
        print(f"\n  [{i+1}/{len(FOLDER_LOCATIONS)}] {base_path}")
        result = rename_folder_and_contents(base_path, old_name, new_name, dry_run)
        results.append((base_path, result))
        print_status(os.path.basename(base_path), result)

    print("\n" + "-" * 70)
    print("  STEP 2: Renaming text file")
    print("-" * 70)
    print(f"\n  {TEXT_FILE_LOCATION}")
    result = rename_text_file(old_name, new_name, dry_run)
    results.append((TEXT_FILE_LOCATION, result))
    print_status("text file", result)

    print("\n" + "-" * 70)
    print("  STEP 3: Updating OpenSearch indexes")
    print("-" * 70)
    try:
        client = get_opensearch_client()
        for idx in OPENSEARCH_INDEXES:
            print(f"\n  {idx}")
            result = update_opensearch_index(client, idx, old_name, new_name, dry_run)
            results.append((idx, result))
            print_status(idx, result)
    except Exception as e:
        print(f"\n  [ERR] OpenSearch connection failed: {e}")

    print("\n" + "-" * 70)
    print("  STEP 4: Updating FAISS paths")
    print("-" * 70)
    for pf in FAISS_PATHS:
        print(f"\n  {pf}")
        result = update_faiss_paths(pf, old_name, new_name, dry_run)
        results.append((pf, result))
        print_status("paths.json", result)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    success = sum(1 for _, r in results if r.get("status") == "success")
    skipped = sum(1 for _, r in results if r.get("status") in ("skipped", "dry-run"))
    errors = sum(1 for _, r in results if r.get("status") not in ("success", "skipped", "dry-run"))
    print(f"  Success: {success}, Skipped: {skipped}, Errors: {errors}")
    print()
    if dry_run:
        print("  DRY RUN - No changes made. Run without --dry-run to apply.")
    else:
        print("  Done!" if errors == 0 else f"  Done with {errors} error(s).")
    print()


if __name__ == "__main__":
    main()
