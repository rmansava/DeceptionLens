"""
Fix folders where file names don't match the folder name.

This happens when PDF extraction uses different metadata than the folder name.
Renames files to use the folder name as prefix.

Usage:
    python fix_mismatched_filenames.py                    # Scan and report
    python fix_mismatched_filenames.py --fix              # Actually rename
    python fix_mismatched_filenames.py --folder "name"    # Check specific folder
"""
import os
import re
import argparse
from pathlib import Path

BOOKS_ROOT = Path(r"D:\books\pdf-images")


def get_page_number(filename: str) -> str:
    """Extract page number from filename like 'BookName-page123.jpg'"""
    match = re.search(r'-page(\d+)\.', filename)
    if match:
        return match.group(1)
    return None


def check_folder(folder_path: Path, fix: bool = False) -> dict:
    """Check if files in folder match the folder name."""
    folder_name = folder_path.name

    files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
    if not files:
        return {"status": "empty", "count": 0}

    mismatched = []
    folder_name_lower = folder_name.lower()
    for f in files:
        # Check if file starts with folder name (case-insensitive on Windows)
        if not f.name.lower().startswith(folder_name_lower):
            mismatched.append(f)

    if not mismatched:
        return {"status": "ok", "count": len(files)}

    if not fix:
        return {
            "status": "mismatched",
            "count": len(files),
            "mismatched": len(mismatched),
            "sample": mismatched[0].name if mismatched else None
        }

    # Fix: rename files to match folder name
    renamed = 0
    errors = []
    for f in mismatched:
        page_num = get_page_number(f.name)
        if page_num is None:
            errors.append(f"No page number: {f.name}")
            continue

        new_name = f"{folder_name}-page{page_num}{f.suffix}"
        new_path = f.parent / new_name

        if new_path.exists():
            errors.append(f"Target exists: {new_name}")
            continue

        try:
            f.rename(new_path)
            renamed += 1
        except Exception as e:
            errors.append(f"Error renaming {f.name}: {e}")

    return {
        "status": "fixed",
        "count": len(files),
        "renamed": renamed,
        "errors": len(errors)
    }


def main():
    parser = argparse.ArgumentParser(description="Fix mismatched file names")
    parser.add_argument("--fix", action="store_true", help="Actually rename files")
    parser.add_argument("--folder", type=str, help="Check specific folder")
    args = parser.parse_args()

    if args.folder:
        # Check specific folder
        folder_path = BOOKS_ROOT / args.folder
        if not folder_path.exists():
            # Try partial match
            matches = [d for d in BOOKS_ROOT.iterdir() if d.is_dir() and args.folder.lower() in d.name.lower()]
            if not matches:
                print(f"Folder not found: {args.folder}")
                return
            folder_path = matches[0]

        print(f"Checking: {folder_path.name}")
        result = check_folder(folder_path, fix=args.fix)
        print(f"  Status: {result['status']}")
        print(f"  Files: {result['count']}")
        if result.get('mismatched'):
            print(f"  Mismatched: {result['mismatched']}")
            print(f"  Sample: {result.get('sample', 'N/A')}")
        if result.get('renamed'):
            print(f"  Renamed: {result['renamed']}")
        if result.get('errors'):
            print(f"  Errors: {result['errors']}")
        return

    # Scan all folders
    folders = sorted([d for d in BOOKS_ROOT.iterdir() if d.is_dir()])
    print(f"Scanning {len(folders)} folders...")
    print()

    mismatched_folders = []

    for i, folder in enumerate(folders):
        if (i + 1) % 100 == 0:
            print(f"\r  Progress: {i+1}/{len(folders)}...", end="", flush=True)

        result = check_folder(folder, fix=args.fix)

        if result['status'] == 'mismatched':
            mismatched_folders.append((folder.name, result))
            print(f"\rMISMATCHED: {folder.name}")
            print(f"  Files: {result['count']}, Mismatched: {result['mismatched']}")
            print(f"  Sample: {result.get('sample', 'N/A')[:80]}")
        elif result['status'] == 'fixed':
            print(f"\rFIXED: {folder.name} - {result['renamed']} renamed")

    print("\r" + " " * 80)
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total folders: {len(folders)}")
    print(f"Mismatched: {len(mismatched_folders)}")

    if mismatched_folders and not args.fix:
        print()
        print("Run with --fix to rename files")


if __name__ == "__main__":
    main()
