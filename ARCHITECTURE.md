# Deception Lens - Technical Architecture

An image finder using CLIP for text-to-image and visual similarity search, plus DINOv2 for fine-grained visual matching with geometric verification.

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Deception Lens                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐         ┌────────────────────────────────────────┐│
│  │  Blazor Server   │  HTTP   │           FastAPI Backend              ││
│  │  (C# Frontend)   │◄───────►│           (Python)                     ││
│  │  Port 5000       │         │           Port 8000                    ││
│  └──────────────────┘         └────────────────────────────────────────┘│
│                                          │                               │
│                    ┌─────────────────────┼─────────────────────┐        │
│                    ▼                     ▼                     ▼        │
│         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐│
│         │    FAISS Index   │  │     ChromaDB     │  │ Geometric Verify ││
│         │  (CLIP Vectors)  │  │ (DINOv2 Vectors) │  │ (DISK+LightGlue) ││
│         │  D:/faiss/books  │  │   ./chroma_db    │  │                  ││
│         └──────────────────┘  └──────────────────┘  └──────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

## Search Modes

1. **CLIP Text Search** - Search by text description (e.g., "truck", "red car")
2. **CLIP Visual Search** - Search by image using CLIP embeddings
3. **DINOv2 Visual Search** - Search by image with optional geometric verification

## Directory Structure

```
DeceptionLens/
├── backend/
│   ├── main.py           # CLI entry point for indexing/searching
│   ├── indexer.py        # DinoIndexer class - DINOv2 indexing
│   ├── searcher.py       # DinoSearcher class - DINOv2 searching
│   ├── clip_indexer.py   # ClipIndexer class - CLIP/FAISS indexing
│   ├── clip_searcher.py  # ClipSearcher class - CLIP text/image search
│   ├── server.py         # FastAPI REST API server
│   ├── batch_index.py    # Batch processing script for DINOv2
│   └── chroma_db/        # ChromaDB persistent storage (DINOv2)
│
├── web/
│   ├── Pages/
│   │   └── Index.razor   # Main search UI page
│   ├── Services/
│   │   └── SearchService.cs  # HTTP client for backend API
│   ├── Models/
│   │   └── SearchResult.cs   # Data models
│   └── Program.cs        # ASP.NET Core startup
│
├── D:/faiss/books/       # CLIP FAISS index (external)
│   ├── index.faiss       # CLIP embeddings (~9GB)
│   └── paths.json        # Image path mapping
│
└── ARCHITECTURE.md       # This file
```

---

## Core Components

### 1. Indexer (`backend/indexer.py`)

The `DinoIndexer` class processes images and stores embeddings in ChromaDB.

#### Models Used:
- **DINOv2** (`facebook/dinov2-base`): Self-supervised vision transformer
  - Output: 768-dimensional embedding per image
  - Uses mean pooling of last hidden state
- **InsightFace** (`buffalo_l` / ArcFace): Face detection and recognition
  - Output: 512-dimensional embedding per detected face
  - Can detect multiple faces per image

#### Two-Pass Indexing System:
Due to GPU memory conflicts between PyTorch (DINOv2) and ONNX Runtime (InsightFace), indexing is done in two passes:

```bash
# Pass 1: Visual embeddings only (DINOv2 on GPU)
python main.py index --dir "path/to/images" --collection books --mode visual_only

# Pass 2: Face embeddings only (InsightFace on GPU)
python main.py index --dir "path/to/images" --collection books --mode faces_only
```

#### Collection Naming:
Each collection creates two sub-collections in ChromaDB:
- `{collection_name}_visual` - DINOv2 image embeddings
- `{collection_name}_faces` - InsightFace face embeddings

#### Embedding Generation:

```python
# DINOv2 Visual Embedding
def get_dino_embedding(self, image: Image.Image) -> np.ndarray:
    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = self.model(**inputs)
        # Mean pooling of patch tokens
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    return embedding  # Shape: (768,)

# InsightFace Face Embedding
def get_face_embeddings(self, cv_image: np.ndarray) -> list:
    faces = self.face_app.get(cv_image)
    embeddings = [face.embedding for face in faces]  # Each: (512,)
    return embeddings
```

#### Database Storage:
- Uses `upsert` (not `add`) to handle duplicate IDs gracefully
- Deduplicates file paths using `set()` before processing
- Batches writes (default batch_size=10) for performance
- Stores metadata: `{"path": absolute_path, "filename": basename}`

---

### 2. Searcher (`backend/searcher.py`)

The `DinoSearcher` class performs similarity search with optional geometric verification.

#### Search Flow:

```
Query Image
     │
     ▼
┌─────────────────┐
│ DINOv2 Embedding│  (768-dim vector)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ChromaDB Query  │  Cosine similarity search
│ (fetch all if   │  Returns candidates sorted by similarity
│  verify=true)   │
└────────┬────────┘
         │
         ▼ (if verify=true)
┌─────────────────┐
│ DISK + LightGlue│  Geometric verification
│ Feature Match   │  Count valid keypoint matches
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Re-rank Results │  Sort by (verified_matches, score)
└────────┬────────┘
         │
         ▼
    Top-K Results
```

#### Similarity Scoring:
```python
# ChromaDB returns cosine distance
# Convert to similarity: score = 1 - distance
score = max(0, 1 - dist)  # Range: 0.0 to 1.0
```

#### Geometric Verification (DISK + LightGlue):

When `verify=true`, the searcher:
1. Extracts DISK keypoints and descriptors from query image
2. For EACH candidate (entire collection), extracts features
3. Uses LightGlue to match keypoints between query and candidate
4. Counts valid matches (LightGlue returns -1 for non-matches)
5. Re-ranks results by verified_matches (descending), then score

```python
# Critical: Count only VALID matches
matches01 = self.matcher({"image0": feats0, "image1": feats1})
matches_idx = matches01["matches"][0]
valid_matches = (matches_idx > -1).sum().item()  # NOT len(matches_idx)!
```

**Why check entire collection?**
The exact match might rank poorly by embedding similarity but have high keypoint matches. Checking all candidates ensures the correct image is found.

#### Image Preprocessing for DISK:
Images must have dimensions divisible by 16:
```python
new_h = ((h + 15) // 16) * 16
new_w = ((w + 15) // 16) * 16
# Pad with black pixels if needed
```

---

### 3. CLIP Searcher (`backend/clip_searcher.py`)

The `ClipSearcher` class provides text-to-image and image-to-image search using CLIP and FAISS.

#### Models Used:
- **CLIP** (`ViT-L/14`): Vision-Language model from OpenAI
  - Output: 768-dimensional embedding for both text and images
  - Supports text-to-image and image-to-image search

#### Search Flow:

```
Query (Text or Image)
        │
        ▼
┌─────────────────┐
│ CLIP Encoding   │  Text: tokenize + encode
│                 │  Image: preprocess + encode
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FAISS Search    │  Inner product similarity
│ IndexFlatIP     │  Returns top-k nearest
└────────┬────────┘
         │
         ▼
    Top-K Results
```

#### Usage:
```python
from clip_searcher import ClipSearcher

searcher = ClipSearcher(
    index_path="D:/faiss/books/index.faiss",
    paths_path="D:/faiss/books/paths.json"
)

# Text search
results = searcher.search_by_text("truck", top_k=50)

# Image search
results = searcher.search_by_image("query.jpg", top_k=50)
```

---

### 4. CLIP Indexer (`backend/clip_indexer.py`)

The `ClipIndexer` class creates FAISS indices from image folders.

#### Features:
- Batch processing (default 64 images per batch)
- GPU-accelerated CLIP encoding
- Checkpoint/resume support for crash recovery
- Inner product similarity (IndexFlatIP)

#### Usage:
```python
from clip_indexer import ClipIndexer

indexer = ClipIndexer(model_name="ViT-L/14", batch_size=64)

# Create new index
result = indexer.index_folder("D:/books/pdf-images", "D:/faiss/books")

# Add to existing index
result = indexer.add_to_index("D:/faiss/books", "D:/new_images")
```

---

### 5. API Server (`backend/server.py`)

FastAPI REST endpoints:

#### DINOv2 Endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns searcher status |
| `/stats?collection=X` | GET | Get visual/face counts for collection |
| `/collections` | GET | List all collections |
| `/search` | POST | DINOv2 image search |
| `/search/bytes` | POST | DINOv2 search by bytes |
| `/image?path=X` | GET | Serve image from filesystem |
| `/collections/{name}` | DELETE | Delete a collection |

#### CLIP Endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/clip/stats` | GET | Get CLIP index statistics |
| `/clip/search` | POST | CLIP image search |
| `/clip/text` | POST/GET | CLIP text-to-image search |
| `/clip/index` | POST | Start CLIP indexing (background) |
| `/clip/index/status` | GET | Get indexing progress |
| `/clip/index/add` | POST | Add folder to existing index |

#### Search Request:
```
POST /search
Content-Type: multipart/form-data

Query Parameters:
- top_k: int (1-500, default 50)
- collection: string (default "images")
- verify: bool (default false)

Body:
- file: image file
```

#### Search Response:
```json
[
  {
    "path": "D:\\books\\pdf-images\\BookName\\page_001.jpg",
    "score": 0.9234,
    "verified_matches": 156,
    "metadata": {
      "path": "D:\\books\\pdf-images\\BookName\\page_001.jpg",
      "filename": "page_001.jpg"
    }
  }
]
```

---

### 6. Web Frontend (`web/`)

Blazor Server application (C# / ASP.NET Core).

#### Key Files:
- **`Pages/Index.razor`**: Main search UI
  - Search engine selector: CLIP Text, CLIP Visual, DINOv2 Visual
  - Text input for CLIP text search
  - Drag-and-drop or click to upload query image (visual modes)
  - Collection selector (DINOv2 only)
  - Results count slider (top_k)
  - Geometric verification toggle (DINOv2 only, default: ON)
  - Results grid with rank badges and scores

- **`Services/SearchService.cs`**: HTTP client wrapper
  - Typed HttpClient injection
  - Handles multipart/form-data uploads
  - CLIP text/image search methods
  - DINOv2 search methods
  - BaseAddress fallback to `http://localhost:8000`

#### Configuration (`appsettings.json`):
```json
{
  "ApiUrl": "http://localhost:8000"
}
```

---

## Data Flow

### Indexing Flow:

```
1. User runs: python main.py index --dir "D:\books\pdf-images\BookName" --collection books --mode visual_only

2. DinoIndexer:
   a. Scans directory for images (jpg, png, webp, gif, bmp)
   b. Deduplicates file list
   c. For each image:
      - Load as PIL RGB image
      - Generate DINOv2 embedding (768-dim)
      - Add to batch buffer
   d. Upsert batches to ChromaDB (books_visual collection)

3. User runs: python main.py index --dir "D:\books\pdf-images\BookName" --collection books --mode faces_only

4. DinoIndexer:
   a. Same file scan
   b. For each image:
      - Load as OpenCV BGR image
      - Detect faces with InsightFace
      - Extract ArcFace embeddings (512-dim each)
      - Add to batch buffer with ID: "{path}_face_{index}"
   d. Upsert batches to ChromaDB (books_faces collection)
```

### Search Flow:

```
1. User uploads image in Blazor UI

2. SearchService.cs:
   a. Reads image into MemoryStream
   b. POSTs to /search?top_k=50&collection=books&verify=true

3. server.py:
   a. Saves upload to temp file
   b. Calls searcher.search()

4. searcher.py:
   a. Generates query DINOv2 embedding
   b. Queries books_visual collection (ALL candidates when verify=true)
   c. For each candidate:
      - Load candidate image
      - Extract DISK features
      - Match with LightGlue
      - Count valid matches
   d. Sort by (verified_matches DESC, score DESC)
   e. Return top_k results

5. Results displayed in UI with rank badges
```

---

## Key Fixes Applied

### 1. LightGlue Match Counting Bug
**Problem**: Original code counted all keypoints, not valid matches.
```python
# WRONG:
valid_matches = len(matches_idx)  # Counts ALL keypoints

# CORRECT:
valid_matches = (matches_idx > -1).sum().item()  # Counts only matched keypoints
```

### 2. ChromaDB Duplicate ID Errors
**Problem**: Using `add()` failed on duplicate IDs.
**Solution**: Changed to `upsert()` which updates existing or inserts new.

### 3. HttpClient BaseAddress
**Problem**: Typed HttpClient had null BaseAddress.
**Solution**: Added fallback in constructor:
```csharp
if (_httpClient.BaseAddress == null)
{
    _httpClient.BaseAddress = new Uri(_apiBaseUrl);
}
```

---

## Dependencies

### Python (backend):
```
# Core
torch
transformers
fastapi
uvicorn
pillow
opencv-python
numpy
tqdm

# DINOv2 / Geometric Verification
chromadb
insightface
onnxruntime-gpu
kornia

# CLIP / FAISS
clip (git+https://github.com/openai/CLIP.git)
faiss-cpu (or faiss-gpu)
```

### C# (web):
```
Microsoft.AspNetCore.Components
Microsoft.Extensions.Http
System.Text.Json
```

---

## Running the System

### Start Backend:
```bash
cd backend
python server.py
# Runs on http://localhost:8000
```

### Start Frontend:
```bash
cd web
dotnet run
# Runs on http://localhost:5000
```

### Index a Book:
```bash
cd backend

# Visual pass
python main.py index --dir "D:\books\pdf-images\BookName" --collection books --mode visual_only

# Faces pass
python main.py index --dir "D:\books\pdf-images\BookName" --collection books --mode faces_only
```

### Batch Index All Books:
```bash
cd backend
python batch_index.py
# Processes all books in D:\books\pdf-images
# Saves progress to batch_progress.txt (resumable)
```

### Check Stats:
```bash
python main.py stats --collection books
```

---

## ChromaDB Collections

Current state of the `books` collection:
- `books_visual`: ~928+ visual embeddings (768-dim each)
- `books_faces`: ~124+ face embeddings (512-dim each)

Collections use HNSW index with cosine similarity metric.

---

## Performance Notes

1. **Geometric verification is slow** but accurate - checks entire collection
2. **Indexing is GPU-bound** - processes ~1-3 images/second on RTX 3090
3. **Face detection adds overhead** - only finds faces in ~15% of book pages
4. **ChromaDB is fast** - cosine search is nearly instant even with 1M+ vectors
5. **Batch indexing resumes** - progress saved to `batch_progress.txt`

---

## Future Improvements

1. Add face search endpoint (currently visual only)
2. Implement thumbnail caching
3. Add pagination for large result sets
4. Support PDF direct upload (extract pages automatically)
5. Add collection management UI
6. Implement incremental indexing (skip already indexed files)
