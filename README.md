# DinoDeceptionLens

Visual similarity search engine using DINOv2 and InsightFace for image indexing and retrieval.

## Description

DinoDeceptionLens is a visual search tool that indexes image directories and enables similarity-based searching. It combines DINOv2 for general visual embeddings and InsightFace for face recognition, storing vectors in ChromaDB for efficient retrieval.

## Features

- **Multi-mode indexing**: Visual-only (DINOv2), faces-only (InsightFace), or combined mode
- **Similarity search**: Find visually similar images using vector embeddings
- **Geometric verification**: Optional Kornia-based verification for improved accuracy
- **Path mapping**: Remap source paths for portable database storage
- **Batch processing**: Configurable batch sizes for efficient database writes
- **Collection statistics**: View indexed image and face counts

## Tech Stack

- **Backend**: Python with FastAPI
- **Frontend**: .NET 8.0 Blazor Server (C#)
- **ML Models**: DINOv2, InsightFace
- **Vector DB**: ChromaDB
- **Verification**: Kornia (optional)

## Usage

```bash
# Index images
python main.py index --dir /path/to/images --mode all

# Search for similar images
python main.py search --query /path/to/query.jpg --top-k 20

# View statistics
python main.py stats --collection images
```

## DISK Chunk Builder Docs

- See `backend/CHUNK_BUILDERS.md` for how the category chunk batch files work (`run_build_*_chunks.bat`), including source modes, paths, resume behavior, and CUDA auto-restart handling.
