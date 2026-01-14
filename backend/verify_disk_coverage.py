"""
Verify DISK feature coverage - ensure every image has a corresponding .npz file.

Usage:
    python verify_disk_coverage.py           # Full report
    python verify_disk_coverage.py --summary # Just totals
    python verify_disk_coverage.py --missing # List books with missing features
    python verify_disk_coverage.py --fix     # Index any missing images
"""
import os
import sys
import time
from pathlib import Path

BOOKS_ROOT = r"D:\books\pdf-images"
NAS_FEATURES_ROOT = r"T:\disk-features\books"
LOCAL_FEATURES_ROOT = r"D:\disk-features\books"
FEATURES_ROOT = r"D:\disk-features"
CATEGORY = "books"
PATH_REMAP = (r"D:\books", r"T:\archiverelated\books")

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


def count_images(book_path: Path) -> set:
    """Get set of image basenames (without extension) in a book folder."""
    images = set()
    for f in book_path.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.add(f.stem.lower())
    return images


def count_features(features_path: Path) -> set:
    """Get set of feature basenames (without .npz) in a features folder."""
    features = set()
    if features_path.exists():
        for f in features_path.glob("*.npz"):
            # Remove .npz extension to get original image stem
            features.add(f.stem.lower())
    return features


def get_missing_image_paths(book_path: Path, missing_stems: set) -> list:
    """Get full paths of images that are missing features."""
    missing_paths = []
    for f in book_path.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            if f.stem.lower() in missing_stems:
                missing_paths.append(str(f))
    return missing_paths


def cleanup_orphaned_features(extra_features_books: list, books_path: Path) -> dict:
    """Delete orphaned .npz files that have no corresponding image."""
    deleted = 0
    failed = 0

    for book, extra_count in extra_features_books:
        book_path = books_path / book

        # Get actual image stems
        images = set()
        for f in book_path.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.add(f.stem.lower())

        # Check NAS first, then local
        nas_features_path = Path(NAS_FEATURES_ROOT) / book
        local_features_path = Path(LOCAL_FEATURES_ROOT) / book

        features_path = nas_features_path if nas_features_path.exists() else local_features_path

        if not features_path.exists():
            continue

        # Delete orphaned .npz files
        for npz_file in features_path.glob("*.npz"):
            if npz_file.stem.lower() not in images:
                try:
                    npz_file.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"  Failed to delete {npz_file.name}: {e}")
                    failed += 1

    return {"deleted": deleted, "failed": failed}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify DISK feature coverage")
    parser.add_argument("--summary", action="store_true", help="Just show totals")
    parser.add_argument("--missing", action="store_true", help="List books with missing features")
    parser.add_argument("--fix", action="store_true", help="Index any missing images")
    parser.add_argument("--book", type=str, help="Check a specific book")
    args = parser.parse_args()

    books_path = Path(BOOKS_ROOT)
    if not books_path.exists():
        print(f"Books root not found: {BOOKS_ROOT}")
        sys.exit(1)

    # Get all books
    all_books = sorted([d.name for d in books_path.iterdir() if d.is_dir()])

    if args.book:
        all_books = [b for b in all_books if args.book.lower() in b.lower()]
        if not all_books:
            print(f"No books match: {args.book}")
            sys.exit(1)

    total_images = 0
    total_features = 0
    missing_books = []  # (book, img_count, feat_count, missing_count, location, missing_stems)
    extra_features_books = []

    print(f"Checking {len(all_books)} books...")
    print()

    for idx, book in enumerate(all_books):
        # Show progress with count of books needing fixes
        needs_fix = len(missing_books)
        fix_str = f" | {needs_fix} need fixing" if needs_fix > 0 else ""
        print(f"\r  Scanning [{idx+1}/{len(all_books)}] {book[:45]:<45}{fix_str:<20}", end="", flush=True)
        book_path = books_path / book
        images = count_images(book_path)

        # Check NAS first, then local
        nas_features_path = Path(NAS_FEATURES_ROOT) / book
        local_features_path = Path(LOCAL_FEATURES_ROOT) / book

        if nas_features_path.exists():
            features = count_features(nas_features_path)
            location = "NAS"
        elif local_features_path.exists():
            features = count_features(local_features_path)
            location = "local"
        else:
            features = set()
            location = "NONE"

        total_images += len(images)
        total_features += len(features)

        missing = images - features
        extra = features - images

        if missing:
            missing_books.append((book, len(images), len(features), len(missing), location, missing))

        if extra:
            extra_features_books.append((book, len(extra)))

        if not args.summary and not args.missing and not args.fix:
            status = "✓" if not missing else f"✗ ({len(missing)} missing)"
            if location == "NONE":
                status = "✗ (no features)"
            print(f"\r  {book[:60]:<60} | {len(images):>4} imgs | {len(features):>4} feats | {location:<5} | {status}")

    # Clear progress line
    print("\r" + " " * 80 + "\r", end="")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total books:    {len(all_books):,}")
    print(f"Total images:   {total_images:,}")
    print(f"Total features: {total_features:,}")

    coverage = (total_features / total_images * 100) if total_images > 0 else 0
    print(f"Coverage:       {coverage:.2f}%")

    if missing_books:
        print()
        print(f"Books with missing features: {len(missing_books)}")
        if args.missing or (not args.summary and not args.fix):
            print()
            for book, imgs, feats, missing_count, loc, _ in missing_books[:50]:
                print(f"  {book[:55]:<55} | {imgs} imgs, {feats} feats, {missing_count} missing ({loc})")
            if len(missing_books) > 50:
                print(f"  ... and {len(missing_books) - 50} more")

        # Calculate total missing
        total_missing = sum(m[3] for m in missing_books)
        print()
        print(f"Total missing features: {total_missing:,}")

        # Fix missing features if requested
        if args.fix:
            print()
            print("=" * 80)
            print("FIXING MISSING FEATURES")
            print("=" * 80)
            print()

            from disk_indexer_file import DiskIndexerFile

            indexer = DiskIndexerFile(
                category=CATEGORY,
                features_root=FEATURES_ROOT,
                batch_size=20,
                path_remap=PATH_REMAP,
                show_progress=True,
                device="cuda"
            )

            total_indexed = 0
            total_failed = 0
            start_time = time.time()

            for i, (book, imgs, feats, missing_count, loc, missing_stems) in enumerate(missing_books):
                book_path = books_path / book
                missing_paths = get_missing_image_paths(book_path, missing_stems)

                print(f"[{i+1}/{len(missing_books)}] {book[:50]} - {len(missing_paths)} missing")

                for image_path in missing_paths:
                    try:
                        success = indexer.index_image(image_path, book_name=book)
                        if success:
                            total_indexed += 1
                        else:
                            total_failed += 1
                    except Exception as e:
                        print(f"  Error: {e}")
                        total_failed += 1

            indexer.close()

            elapsed = time.time() - start_time
            print()
            print("=" * 80)
            print("FIX COMPLETE")
            print("=" * 80)
            print(f"Indexed: {total_indexed:,}")
            print(f"Failed:  {total_failed:,}")
            print(f"Time:    {elapsed:.1f}s ({total_indexed/elapsed:.1f} img/s)" if elapsed > 0 else "")
            print()
            print("Note: Run again to verify, or run batch indexer to move to NAS.")
    else:
        print()
        print("All images have DISK features!")

    if extra_features_books:
        total_extra = sum(e[1] for e in extra_features_books)
        print(f"\nNote: {total_extra} orphaned .npz files (images deleted?)")

        # Prompt to clean up
        response = input("\nClean up orphaned .npz files? (y/n): ").strip().lower()
        if response == 'y':
            print()
            print("Cleaning up orphaned features...")
            result = cleanup_orphaned_features(extra_features_books, books_path)
            print(f"Deleted: {result['deleted']}")
            if result['failed'] > 0:
                print(f"Failed:  {result['failed']}")


if __name__ == "__main__":
    main()
