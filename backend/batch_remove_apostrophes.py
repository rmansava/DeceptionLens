"""
Batch remove apostrophes from book names.

Finds all books with ' in their name and renames them to remove the apostrophe.
Updates all file locations and indexes.

Usage:
    python batch_remove_apostrophes.py --dry-run    # Preview changes
    python batch_remove_apostrophes.py              # Apply changes
"""

import argparse
import os
import sys
from rename_book import (
    FOLDER_LOCATIONS,
    TEXT_FILE_LOCATION,
    FAISS_PATHS,
    OPENSEARCH_INDEXES,
    get_opensearch_client,
    rename_folder_and_contents,
    rename_text_file,
    update_opensearch_index,
    update_faiss_paths,
    print_status,
)

# Characters to remove from book names
APOSTROPHE_CHARS = "'"  # Just the straight apostrophe


def find_books_with_apostrophe(base_path):
    """Find all book folders containing apostrophes."""
    books = []
    try:
        for folder in os.listdir(base_path):
            if APOSTROPHE_CHARS in folder:
                books.append(folder)
    except Exception as e:
        print(f"Error listing {base_path}: {e}")
    return books


def generate_new_name(old_name):
    """Generate new name by removing apostrophes."""
    return old_name.replace("'", "")


def main():
    parser = argparse.ArgumentParser(description="Batch remove apostrophes from book names")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--limit", type=int, help="Limit number of books to process")
    args = parser.parse_args()

    base_path = r"D:\books\pdf-images"

    print("=" * 70)
    print("  Batch Apostrophe Removal")
    print("=" * 70)

    # Find all books with apostrophes
    print(f"\nScanning {base_path}...")
    books = find_books_with_apostrophe(base_path)
    print(f"Found {len(books)} books with apostrophes")

    if args.limit:
        books = books[:args.limit]
        print(f"Limited to {len(books)} books")

    if not books:
        print("No books to process.")
        return

    # Preview first few
    print("\nSample renames:")
    for book in books[:5]:
        new_name = generate_new_name(book)
        print(f"  {book[:60]}...")
        print(f"    -> {new_name[:60]}...")

    if len(books) > 5:
        print(f"  ... and {len(books) - 5} more")

    if args.dry_run:
        print(f"\nDRY RUN - Would rename {len(books)} books")
        return

    # Confirm
    if not args.yes:
        print(f"\nThis will rename {len(books)} books across all locations.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    # Connect to OpenSearch once
    try:
        client = get_opensearch_client()
    except Exception as e:
        print(f"Warning: OpenSearch connection failed: {e}")
        client = None

    # Process each book
    success_count = 0
    error_count = 0

    for i, old_name in enumerate(books):
        new_name = generate_new_name(old_name)

        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(books)}] {old_name[:50]}...")
        print(f"       -> {new_name[:50]}...")
        print("=" * 70)

        book_errors = 0

        # Step 1: Rename folders
        for base in FOLDER_LOCATIONS:
            result = rename_folder_and_contents(base, old_name, new_name, dry_run=False)
            if result.get("status") == "error":
                book_errors += 1
                print(f"    [ERR] {base}: {result.get('message')}")
            elif result.get("status") == "success":
                files = result.get("files", {}).get("renamed", 0)
                print(f"    [OK] {os.path.basename(base)}: {files} files")

        # Step 2: Rename text file
        result = rename_text_file(old_name, new_name, dry_run=False)
        if result.get("status") == "error":
            book_errors += 1
            print(f"    [ERR] text file: {result.get('message')}")
        elif result.get("status") == "success":
            print(f"    [OK] text file")

        # Step 3: Update OpenSearch
        if client:
            for idx in OPENSEARCH_INDEXES:
                result = update_opensearch_index(client, idx, old_name, new_name, dry_run=False)
                if result.get("status") == "error":
                    book_errors += 1
                    print(f"    [ERR] {idx}: {result.get('message')}")
                elif result.get("status") == "success":
                    print(f"    [OK] {idx}: {result.get('updated')} docs")

        # Step 4: Update FAISS
        for pf in FAISS_PATHS:
            result = update_faiss_paths(pf, old_name, new_name, dry_run=False)
            if result.get("status") == "error":
                book_errors += 1
                print(f"    [ERR] FAISS: {result.get('message')}")
            elif result.get("status") == "success":
                print(f"    [OK] FAISS: {result.get('updated')} paths")

        if book_errors == 0:
            success_count += 1
        else:
            error_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total processed: {len(books)}")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print()


if __name__ == "__main__":
    main()
