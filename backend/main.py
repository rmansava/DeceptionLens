"""
DinoDeceptionLens CLI
Main entry point for indexing and searching.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="DinoDeceptionLens: Visual similarity search using DINOv2 and InsightFace"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a directory of images")
    index_parser.add_argument(
        "--dir", required=True,
        help="Path to the directory containing images"
    )
    index_parser.add_argument(
        "--collection", default="images",
        help="Name of the ChromaDB collection (default: images)"
    )
    index_parser.add_argument(
        "--db-path", default="./chroma_db",
        help="Path to ChromaDB storage (default: ./chroma_db)"
    )
    index_parser.add_argument(
        "--reset", action="store_true",
        help="Reset the collection before indexing"
    )
    index_parser.add_argument(
        "--map-source",
        help="Original source root to be replaced in stored paths"
    )
    index_parser.add_argument(
        "--map-target",
        help="Target root to replace with in stored paths"
    )
    index_parser.add_argument(
        "--mode",
        choices=['all', 'visual_only', 'faces_only'],
        default='all',
        help="Indexing mode: 'all', 'visual_only' (DINOv2), or 'faces_only' (InsightFace)"
    )
    index_parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size for writing to database (default: 10)"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for a query image")
    search_parser.add_argument(
        "--query", required=True,
        help="Path to the query image"
    )
    search_parser.add_argument(
        "--collection", default="images",
        help="Name of the ChromaDB collection (default: images)"
    )
    search_parser.add_argument(
        "--db-path", default="./chroma_db",
        help="Path to ChromaDB storage (default: ./chroma_db)"
    )
    search_parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of results to return (default: 20)"
    )
    search_parser.add_argument(
        "--verify", action="store_true",
        help="Perform geometric verification (requires Kornia)"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show collection statistics")
    stats_parser.add_argument(
        "--collection", default="images",
        help="Name of the ChromaDB collection (default: images)"
    )
    stats_parser.add_argument(
        "--db-path", default="./chroma_db",
        help="Path to ChromaDB storage (default: ./chroma_db)"
    )

    args = parser.parse_args()

    if args.command == "index":
        from indexer import DinoIndexer

        print(f"Starting indexing for directory: {args.dir}")
        print(f"Mode: {args.mode}")
        print(f"Collection: {args.collection}")
        print(f"Database path: {args.db_path}")

        enable_visual = args.mode in ('all', 'visual_only')
        enable_faces = args.mode in ('all', 'faces_only')

        indexer = DinoIndexer(
            collection_name=args.collection,
            db_path=args.db_path,
            enable_visual=enable_visual,
            enable_faces=enable_faces
        )

        if args.reset:
            indexer.reset_collections()

        mapping = None
        if args.map_source and args.map_target:
            mapping = (args.map_source, args.map_target)
            print(f"Path mapping: '{args.map_source}' -> '{args.map_target}'")
        elif args.map_source or args.map_target:
            print("Warning: Both --map-source and --map-target required. Ignoring.")

        indexer.index_directory(args.dir, path_mapping=mapping, batch_size=args.batch_size)
        print("Indexing complete.")

    elif args.command == "search":
        from searcher import DinoSearcher

        print(f"Searching for: {args.query}")

        searcher = DinoSearcher(db_path=args.db_path)
        results = searcher.search(
            args.query,
            top_k=args.top_k,
            verify=args.verify,
            collection_name=args.collection
        )

        print(f"\nFound {len(results)} matches:")
        for i, res in enumerate(results):
            verified = f", Verified: {res['verified_matches']}" if args.verify else ""
            print(f"{i+1}. {res['path']} (Score: {res['score']:.4f}{verified})")

    elif args.command == "stats":
        from searcher import DinoSearcher

        searcher = DinoSearcher(db_path=args.db_path)
        stats = searcher.get_collection_stats(args.collection)

        print(f"\nCollection '{args.collection}' statistics:")
        print(f"  Visual embeddings: {stats['visual_count']}")
        print(f"  Face embeddings: {stats['face_count']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
