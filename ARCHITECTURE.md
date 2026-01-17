# Deception Lens - Technical Architecture

An image finder using CLIP for text-to-image and visual similarity search, plus DINOv2 for fine-grained visual matching with geometric verification.

## System Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                              Deception Lens                                    │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐         ┌─────────────────────────────────────────────┐│
│  │  Blazor Server   │  HTTP   │              FastAPI Backend                ││
│  │  (C# Frontend)   │◄───────►│              (Python)                       ││
│  │  Port 5000       │         │              Port 8000                      ││
│  └──────────────────┘         └─────────────────────────────────────────────┘│
│                                             │                                 │
│              ┌──────────────────────────────┼──────────────────────────────┐ │
│              ▼                              ▼                              ▼ │
│   ┌──────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   │    FAISS Index   │         │    OpenSearch    │         │  Geometric      │
│   │  (CLIP Vectors)  │         │ (DINOv2 Vectors) │         │  Verification   │
│   │  D:/faiss/books  │         │ dinov2-books idx │         │ (DISK+LightGlue)│
│   └──────────────────┘         └──────────────────┘         └────────┬────────┘
│                                                                      │        │
│                                                                      ▼        │
│                                                         ┌──────────────────┐  │
│                                                         │   SQL Server     │  │
│                                                         │ trivia.DiskFeatures│
│                                                         │ (cached keypoints)│  │
│                                                         └──────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Search Modes

1. **CLIP Text Search** - Search by text description (e.g., "truck", "red car")
2. **CLIP Visual Search** - Search by image using CLIP embeddings
3. **DINOv2 Visual Search** - Search by image with optional geometric verification

## Directory Structure

```
DeceptionLens/
├── backend/
│   ├── main.py                    # CLI entry point for indexing/searching
│   ├── indexer.py                 # DinoIndexer class - DINOv2 indexing
│   ├── searcher.py                # DinoSearcher class - DINOv2 searching
│   ├── clip_indexer.py            # ClipIndexer class - CLIP/FAISS indexing
│   ├── clip_searcher.py           # ClipSearcher class - CLIP text/image search
│   ├── server.py                  # FastAPI REST API server
│   ├── batch_index.py             # Batch processing script for DINOv2
│   │
│   ├── # DISK Feature Cache (SQL Server)
│   ├── disk_features.py           # SQL storage for DISK keypoints
│   ├── disk_indexer.py            # DISK extraction (SQL storage)
│   ├── batch_disk_index.py        # Batch DISK indexing (SQL)
│   │
│   ├── # DISK Feature Cache (File-Based / NAS)
│   ├── disk_features_file.py      # File storage for DISK keypoints (.npz)
│   ├── disk_indexer_file.py       # DISK extraction (file storage)
│   ├── batch_disk_index_file.py   # Batch DISK indexing (files)
│   │
│   ├── sql/
│   │   └── create_disk_features_table.sql  # SQL schema
│   └── chroma_db/                 # ChromaDB persistent storage (DINOv2)
│
├── web/
│   ├── Pages/
│   │   └── Index.razor            # Main search UI page
│   ├── Services/
│   │   └── SearchService.cs       # HTTP client for backend API
│   ├── Models/
│   │   └── SearchResult.cs        # Data models
│   └── Program.cs                 # ASP.NET Core startup
│
├── D:/faiss/books/                # CLIP FAISS index (external)
│   ├── index.faiss                # CLIP embeddings (~9GB)
│   └── paths.json                 # Image path mapping
│
├── D:/disk-features/              # DISK features (local indexing)
│   └── books/                     # Category folder
│       └── {BookName}/*.npz       # Per-image feature files
│
├── T:/disk-features/              # DISK features (NAS production)
│   └── books/                     # Move from D: after indexing
│
├── SQL Server (trivia DB)         # DISK feature cache (alternative)
│   └── dbo.DiskFeatures           # Pre-computed keypoints/descriptors
│
└── ARCHITECTURE.md                # This file
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

# DISK Feature Cache (SQL Server)
pyodbc
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

## OpenSearch Backend (books collection)

The `books` collection uses OpenSearch for vector storage instead of ChromaDB for better scalability.

### OpenSearch Indices:
- `dinov2-books`: Visual embeddings (768-dim, HNSW with cosinesimil)
- `faces-books`: Face embeddings (512-dim, HNSW with cosinesimil)

### Two-Stage Search Pipeline:

```
Query Image (e.g., cropped snippet)
         │
         ▼
┌─────────────────────┐
│ Stage 1: OpenSearch │  DINOv2 embedding similarity
│ Fetch top 5000      │  Global image similarity
│ candidates          │  (crop vs full page = LOW ~0.25)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Stage 2: LightGlue  │  Local keypoint matching
│ Geometric verify    │  (crop vs full page = HIGH matches)
│ all 5000 candidates │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Re-rank by matches  │  Sort by (verified_matches, score) DESC
│ Return top K        │
└─────────────────────┘
```

---

## Test Case: Cropped Image Search

This demonstrates the two-stage search finding a cropped snippet within a full page.

### Test Setup:
- **Query Image**: `D:/trivpics/2023-5.jpg` (cropped Manglosaurus dinosaur)
- **Target Page**: `D:\books\pdf-images\encyclopedia of monsters\encyclopedia of monsters-page210.jpg`
- **Target Content**: Full page with 3 figures (Manglosaurus, Manglord, Manglodactyl) + text

### The Problem with Global Embeddings:

DINOv2 computes a **global embedding** for the entire image. When comparing:
- Cropped dinosaur image vs full page with multiple figures + text
- **Cosine similarity = 0.249** (very low!)

The crop would NOT appear in top search results based on embedding alone.

### How LightGlue Fixes This:

LightGlue performs **local keypoint matching** using DISK features:
- Finds matching keypoints between crop and full page
- The Manglosaurus dinosaur in both images produces **662 verified matches**
- Even though global similarity is low, local matches are HIGH

### Initial OpenSearch Ranking (Before LightGlue):

- **Page210 initial rank: 2589** out of 5000 candidates
- **Score: 0.6246** (low due to crop vs full page)

Without LightGlue verification, page210 would be buried at rank 2589 and never shown to the user.

### Final Results (After LightGlue Re-ranking):

| Rank | Image | DINOv2 Score | Keypoint Matches |
|------|-------|--------------|------------------|
| **1** | encyclopedia of monsters-page210.jpg | 0.625 | **662** |
| 2 | The encyclopedia of monsters -- Jeff Rovin-page212.jpg | 0.625 | 524 |
| 3 | 50 Great Comedy Film Posters-page19.jpg | 0.611 | 202 |
| ... | (other results) | ... | ... |

**Rank improvement: 2589 → 1** (thanks to 662 keypoint matches)

### Key Configuration:

```python
# server.py - Must fetch MANY candidates for crops to work
fetch_k = 5000  # Large pool for LightGlue to search
```

**Why 5000?** With only embedding similarity, the target page might rank #3000+.
LightGlue needs it in the candidate pool to find the keypoint matches.

---

## Performance Notes

1. **Geometric verification is slow** but accurate - checks entire candidate pool
2. **DISK indexing is optimized** - parallel prefetch + async saves achieves ~10-13 img/s on RTX 4070 Super
3. **Face detection adds overhead** - only finds faces in ~15% of book pages
4. **OpenSearch k-NN is fast** - uses HNSW algorithm for approximate nearest neighbors
5. **Batch indexing resumes** - progress saved to `batch_progress_opensearch.txt`
6. **Crop search requires large candidate pool** - fetch_k=5000 for snippets to work
7. **DISK feature cache** - pre-computed features reduce verification from ~17min to ~2-3min

### DISK Indexing Optimization

The `disk_indexer_file.py` uses a multi-threaded pipeline to maximize GPU utilization:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  File Feeder    │────►│  Prefetch Queue │────►│  GPU Inference  │
│  (1 thread)     │     │  (8 workers)    │     │  (main thread)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌─────────────────┐              │
                        │  Save Worker    │◄─────────────┘
                        │  (async disk)   │
                        └─────────────────┘
```

| Configuration | Rate | Notes |
|---------------|------|-------|
| Sequential (original) | ~8 img/s | GPU idle 50%+ waiting on I/O |
| Parallel prefetch only | ~6 img/s | Batch saves still block |
| **Parallel + async save** | **~10-13 img/s** | GPU stays fed, saves non-blocking |

Key parameters in `DiskIndexerFile`:
- `num_workers=8` - CPU threads for image decode/resize
- `prefetch_size=32` - Images buffered ahead of GPU
- `batch_size=20` - Features batched before async save

### DISK Coverage Verification

The `verify_disk_coverage.py` tool verifies that NAS storage has DISK features for every image:

```bash
# Check coverage (NAS only)
python verify_disk_coverage.py --summary

# Fix missing features and sync to NAS automatically
python verify_disk_coverage.py --fix
```

**What it does:**
1. Scans all images in `D:\books\pdf-images`
2. Checks if corresponding `.npz` exists on NAS (`T:\disk-features\books`)
3. With `--fix`:
   - Checks local storage (`D:\disk-features\books`) for existing features
   - Reuses local features instead of re-indexing (fast!)
   - Indexes only truly missing features
   - Moves all to NAS and merges if needed
   - Prompts to clean up orphaned features
4. Ensures NAS is the source of truth and stays in sync

Run `verify_disk_coverage.bat` after batch indexing to ensure 100% coverage. The tool is smart about reusing work - if features exist locally but aren't on NAS yet, it moves them instead of re-computing.

### OpenSearch Coverage Verification

The `verify_opensearch_coverage.py` tool verifies that OpenSearch has embeddings for every image:

```bash
# Check coverage
python verify_opensearch_coverage.py --summary

# Fix missing embeddings automatically
python verify_opensearch_coverage.py --fix
```

**What it does:**
1. Scans all images in `D:\books\pdf-images`
2. Queries OpenSearch to check which images are indexed in `dinov2-books` and `faces-books`
3. Reports missing visual and face embeddings separately
4. With `--fix`:
   - Re-indexes books with missing embeddings
   - Uses the existing `OpenSearchIndexer` to ensure consistency
   - Reports total indexed visual and face embeddings
   - Prompts to clean up orphaned entries from renamed/deleted books
5. Ensures OpenSearch indexes stay in sync with source images

Run `verify_opensearch_coverage.bat` after batch indexing to ensure 100% coverage. The tool queries OpenSearch efficiently using scroll API and term aggregations.

---

## DISK Feature Cache (SQL Server)

The geometric verification step (DISK + LightGlue) was the main bottleneck. Pre-computing DISK features provides ~6-8x speedup.

### The Problem

Without caching, each verification requires:
1. Load image from disk (~5-10ms)
2. Run DISK feature extraction (~150-250ms)
3. Run LightGlue matching (~15-30ms)

**Total: ~200ms × 5000 candidates = ~17 minutes per search**

### The Solution

Pre-compute and store DISK features in SQL Server:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISK Feature Cache                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐     ┌────────────────────────────────────┐ │
│  │  disk_indexer   │────►│  SQL Server: trivia.dbo.DiskFeatures│ │
│  │  (batch index)  │     │  - Keypoints (gzipped float32)     │ │
│  └─────────────────┘     │  - Descriptors (gzipped float16)   │ │
│                          │  - Image dimensions                 │ │
│  ┌─────────────────┐     └────────────────────────────────────┘ │
│  │  searcher.py    │◄────────────────┘                          │
│  │  (bulk load)    │     Verification: ~20-35ms per image       │
│  └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### SQL Table Schema

```sql
-- Database: trivia
CREATE TABLE dbo.DiskFeatures (
    Id INT IDENTITY PRIMARY KEY,
    ImagePath NVARCHAR(500) NOT NULL UNIQUE,
    BookName NVARCHAR(200),
    Keypoints VARBINARY(MAX),      -- gzipped (N,2) float32
    Descriptors VARBINARY(MAX),    -- gzipped (N,128) float16
    KeypointCount SMALLINT,
    ImageWidth SMALLINT,
    ImageHeight SMALLINT,
    PaddedWidth SMALLINT,          -- DISK requires multiples of 16
    PaddedHeight SMALLINT,
    CreatedAt DATETIME2 DEFAULT SYSUTCDATETIME()
);
```

### Storage Estimates

| Compression | Per Image | 2.9M Images |
|-------------|-----------|-------------|
| None (float32) | ~260 KB | ~750 GB |
| float16 descriptors | ~140 KB | ~400 GB |
| float16 + gzip | ~60-80 KB | ~175-230 GB |

### Performance Comparison

| Step | Without Cache | With Cache |
|------|---------------|------------|
| Load image | 5-10ms | 0 |
| DISK extraction | 150-250ms | 0 |
| SQL bulk load | 0 | ~2-5ms/image |
| Decompress | 0 | ~1ms |
| LightGlue | 15-30ms | 15-30ms |
| **Total per image** | **~200ms** | **~20-35ms** |
| **5000 candidates** | **~17 min** | **~2-3 min** |

### Components

#### 1. disk_features.py - SQL Storage Layer

```python
from disk_features import DiskFeatureStore

store = DiskFeatureStore()

# Save features
store.save(image_path, keypoints, descriptors, image_size, padded_size, book_name)

# Bulk load for search (efficient)
features = store.load_bulk(image_paths)  # Returns dict[path -> DiskFeatureData]

# Check statistics
stats = store.get_stats()
```

#### 2. disk_indexer.py - Feature Extraction

```python
from disk_indexer import DiskIndexer

# Single directory
indexer = DiskIndexer(batch_size=20)
result = indexer.index_directory("D:/books/pdf-images/BookName")

# With path remapping (read from D:, store as T:)
indexer = DiskIndexer(
    batch_size=20,
    path_remap=("D:\\books", "T:\\archiverelated\\books")
)
```

CLI usage:
```bash
# Index single book
python disk_indexer.py "D:\books\pdf-images\BookName"

# With path remapping
python disk_indexer.py "D:\books\pdf-images\BookName" \
    --remap-from "D:\books" \
    --remap-to "T:\archiverelated\books"

# Show stats
python disk_indexer.py --stats
```

#### 3. batch_disk_index.py - Batch Processing

```bash
python batch_disk_index.py
```

Features:
- Processes all books in `D:\books\pdf-images`
- Path remapping built-in: `D:\books` → `T:\archiverelated\books`
- Resume support via `batch_disk_progress.txt`
- Logs to `batch_disk_index.log`
- Skips already indexed images

#### 4. searcher.py Integration

The `DinoSearcher` automatically uses cached features when available:

```python
# In _verify_matches():
if self.disk_cache:
    # Bulk load from SQL (fast)
    cached_features = self.disk_cache.load_bulk(match_paths)

for match in matches:
    if match_path in cached_features:
        # Use cached features (fast path)
        feats1 = cached_to_tensor(cached_features[match_path])
    else:
        # Fall back to on-the-fly extraction (slow path)
        feats1 = self.extractor(load_image(match_path))

    # LightGlue matching (same either way)
    matches01 = self.matcher({"image0": feats0, "image1": feats1})
```

### Path Remapping

For NAS storage, images may be on a different drive during indexing vs. production:

| Phase | Local Path | NAS Path |
|-------|------------|----------|
| Indexing | `D:\books\pdf-images\...` | - |
| Production | - | `T:\archiverelated\books\pdf-images\...` |

The indexer reads from D: but stores T: paths in SQL, so no migration needed later.

### Setup Steps (SQL Server)

1. **Create SQL table**:
```bash
sqlcmd -S localhost -d trivia -E -i backend/sql/create_disk_features_table.sql
```

2. **Install pyodbc**:
```bash
pip install pyodbc
```

3. **Run batch indexer**:
```bash
cd backend
python batch_disk_index.py
```

4. **Monitor progress**:
```sql
SELECT * FROM trivia.dbo.vw_DiskFeaturesStats;
```

---

## DISK Feature Cache (File-Based / NAS)

Alternative to SQL Server for NAS-based storage. Uses compressed `.npz` files for maximum portability.

### Directory Structure

```
T:\disk-features\                    # NAS root
├── books\                           # Category folder
│   ├── Encyclopedia of Monsters\    # Book folder
│   │   ├── page_001.npz            # Compressed features
│   │   ├── page_002.npz
│   │   └── ...
│   └── ...
├── printads\                        # Another category
│   └── ...
└── ...
```

### File Format

Each `.npz` file contains:
- `keypoints`: (N, 2) float32 array - keypoint x,y coordinates
- `descriptors`: (N, 128) float16 array - DISK descriptors
- `image_size`: (2,) int32 - original (height, width)
- `padded_size`: (2,) int32 - padded dimensions (multiples of 16)
- `image_path`: string - reference to source image path

### Indexing Workflow

The batch indexer writes to local SSD for speed, then automatically moves completed books to NAS:

```bash
# Run the batch indexer (or use run_disk_indexer.bat)
cd backend
python batch_disk_index_file.py
```

**What happens:**
1. Indexes to `D:\disk-features\books\{BookName}\` (local SSD, fast writes)
2. After each book completes, queues it for background move to `T:\disk-features\books\`
3. Background thread moves one book at a time to NAS (doesn't block indexing)
4. Console shows progress with [MOVE] messages in cyan

**Features:**
- Resume support: tracks completed books in `batch_disk_progress_file.txt`
- Logs to `batch_disk_index_file.log`
- ETA display based on average processing time
- Graceful shutdown: waits for pending moves before exiting

### Components

#### 1. disk_features_file.py - File Storage Layer

```python
from disk_features_file import DiskFeatureFileStore

store = DiskFeatureFileStore(
    category="books",
    features_root=r"T:\disk-features",
    source_image_root=r"T:\archiverelated\books"
)

# Bulk load for search (parallel I/O)
features = store.load_bulk(image_paths)  # Returns dict[path -> DiskFeatureData]

# Check statistics
stats = store.get_stats()
```

#### 2. disk_indexer_file.py - File-Based Feature Extraction

```python
from disk_indexer_file import DiskIndexerFile

indexer = DiskIndexerFile(
    category="books",
    features_root=r"D:\disk-features",  # Local for speed
    batch_size=20,
    path_remap=("D:\\books", "T:\\archiverelated\\books")
)
result = indexer.index_directory("D:/books/pdf-images/BookName")
```

CLI usage:
```bash
# Index single book
python disk_indexer_file.py "D:\books\pdf-images\BookName" \
    --category books \
    --features-root "D:\disk-features"

# Show stats
python disk_indexer_file.py --stats --category books
```

#### 3. batch_disk_index_file.py - Batch Processing

```bash
# Run directly or via batch file
python batch_disk_index_file.py
# Or: run_disk_indexer.bat
```

Features:
- Processes all books in `D:\books\pdf-images`
- Writes to `D:\disk-features\books` (local SSD for speed)
- **Auto-moves** each completed book to `T:\disk-features\books` (NAS)
- Background move queue: 1 book at a time, doesn't block indexing
- Path remapping: stores `T:\archiverelated\books` paths in .npz files
- Resume support via `batch_disk_progress_file.txt`
- Logs to `batch_disk_index_file.log`

### Storage Comparison

| Storage | Per Image | 2.9M Images | Notes |
|---------|-----------|-------------|-------|
| SQL Server | ~1.7 MB | ~4.9 TB | Requires local SQL |
| .npz files | ~1.5 MB | ~4.3 TB | Portable, NAS-friendly |
| .npz (aggressive) | ~0.8-1.0 MB | ~2.5-3 TB | Lower compression level |

### Search Integration

The `DinoSearcher` automatically tries file cache first, falls back to SQL:

```python
# Priority order:
# 1. File-based cache (T:\disk-features\books)
# 2. SQL Server cache (trivia.dbo.DiskFeatures)
# 3. On-the-fly extraction (slow)

# In searcher.py __init__:
if DISK_FILE_AVAILABLE:
    self.disk_file_cache = DiskFeatureFileStore(
        category="books",
        features_root=r"T:\disk-features",
        source_image_root=r"T:\archiverelated\books"
    )
elif DISK_SQL_AVAILABLE:
    self.disk_cache = DiskFeatureStore()
```

### Benefits of File-Based Storage

1. **NAS Compatible**: No SQL Server required on NAS
2. **Portable**: Can move/copy directories freely
3. **Parallel I/O**: ThreadPoolExecutor for bulk loading
4. **Easy Backup**: Standard file copy/robocopy
5. **Incremental Updates**: Add new books by adding folders
6. **Multiple Categories**: books, printads, etc. in separate folders

### Performance Notes

- **Local SSD writes**: ~50-100 images/second during indexing
- **NAS bulk load**: ~10-20ms per image (parallel threads)
- **Total verification time**: Similar to SQL (~2-3 min for 5000 candidates)

---

## DINOv2/ArcFace Indexing Deduplication Strategy

The `board_games_dino_indexer.py` implements an optimized multi-stage deduplication strategy to avoid reprocessing images for both **DINOv2 visual embeddings** (768-dim) and **ArcFace face embeddings** (512-dim).

### The Problem

Initial indexing runs were extremely slow due to:
1. Reading and hashing every file from NAS (~4 hours for 878k images)
2. Checking SQL Server for each hash
3. Only then discovering which files were already processed

### The Solution: Four-Stage Pipeline

```
┌───────────────────────────────────────────────────────────────────────┐
│               DINOv2/ArcFace Indexing Deduplication                    │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Stage 1: Load Existing Paths (fast, no file I/O)                     │
│  ┌─────────────────────┐     ┌─────────────────────┐                  │
│  │  OpenSearch Scroll  │     │  SQL Server Query   │                  │
│  │  (document IDs)     │     │  (ImageHashes)      │                  │
│  │  ~510k paths        │     │  ~878k paths        │                  │
│  └──────────┬──────────┘     └──────────┬──────────┘                  │
│             │                            │                            │
│             └──────────┬─────────────────┘                            │
│                        ▼                                              │
│            ┌─────────────────────┐                                    │
│            │  Combined Set       │  Union of both sources             │
│            │  ~878k unique paths │  O(1) lookup per file              │
│            └──────────┬──────────┘                                    │
│                       │                                               │
│  Stage 2: Path-Based Skip (instant, in-memory)                        │
│                       ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  for path in all_image_paths:                                    │  │
│  │      if path in existing_paths:                                  │  │
│  │          skip  # No file read needed!                            │  │
│  │      else:                                                       │  │
│  │          add to paths_to_hash                                    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                       │                                               │
│  Stage 3: Content Hash (only for truly NEW files)                     │
│                       ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  for path in paths_to_hash:  # Only ~10-20% of files             │  │
│  │      hash = SHA256(file)     # Now we read the file              │  │
│  │      if hash in SQL:                                             │  │
│  │          skip  # Content duplicate in different location         │  │
│  │      else:                                                       │  │
│  │          add to new_images  # Ready for indexing                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                       │                                               │
│  Stage 4: Generate Embeddings (both in single pass)                   │
│                       ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  for image in new_images:                                        │  │
│  │      DINOv2 → 768-dim visual embedding → board_games_visual      │  │
│  │      ArcFace → 512-dim face embedding(s) → board_games_faces     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Why Both OpenSearch AND SQL Server?

| Source | Contains | Purpose |
|--------|----------|---------|
| OpenSearch | Successfully indexed images | Skip already-indexed paths |
| SQL Server | All hashed images (saved before indexing) | Skip files we've already read/hashed |

**Key insight**: SQL Server tracks files we've hashed immediately after hashing completes (before indexing starts). If indexing crashes, the next run skips all hashed files by path - no re-hashing needed.

### Embedding Types and OpenSearch Indices

| Embedding | Model | Dimensions | Index Name | Purpose |
|-----------|-------|------------|------------|---------|
| **Visual** | DINOv2 (ViT-B/14) | 768 | `board_games_visual` | Scene/object similarity search |
| **Face** | ArcFace (buffalo_l) | 512 | `board_games_faces` | Face recognition/similarity |

Both embeddings are generated in a **single pass** over new images:
- Each image gets one DINOv2 visual embedding (stored with image path as doc ID)
- Each image may have 0-N face embeddings (stored with `{path}_face_{i}` as doc ID)

### Performance Comparison

| Scenario | Old Method | New Method |
|----------|------------|------------|
| **First run (empty DB)** | 4 hours (hash all) | 4 hours (same) |
| **Resume after interrupt** | 4 hours (re-hash all) | ~1 min (path skip) |
| **Incremental (new files)** | 4 hours (re-hash all) | ~10 min (hash only new) |
| **Re-run after completion** | 4 hours (re-hash all) | ~1 min (all paths skip) |

### Code Flow

```python
# Stage 1: Load existing paths from both sources
opensearch_paths = load_opensearch_paths(os_client, VISUAL_INDEX)
sql_paths = load_existing_paths(cursor, dino_collection)
existing_paths = opensearch_paths | sql_paths  # Union

# Stage 2: Fast path-based skip (no file I/O)
for path in all_image_paths:
    if path in existing_paths:
        skipped_by_path += 1
    else:
        paths_to_hash.append(path)

# Stage 3: Hash only NEW files for content dedup
for path in paths_to_hash:
    file_hash = get_file_hash(path)  # NOW we read the file
    if check_hash_exists(cursor, file_hash, collection):
        skipped_duplicates += 1  # Same content, different path
    else:
        new_images.append((path, file_hash, file_size))

# Save hashes to SQL BEFORE indexing (crash-safe checkpoint)
for path, file_hash, file_size in new_images:
    add_hash_to_db(cursor, file_hash, path, collection, file_size)
conn.commit()

# Stage 4: Generate embeddings (both in single pass)
for image_path in new_images:
    # DINOv2 visual embedding (768-dim)
    embedding = dinov2_model(image)
    opensearch.index(index="board_games_visual", id=image_path, body={"embedding": embedding})

    # ArcFace face embeddings (512-dim each)
    faces = face_app.get(image)
    for i, face in enumerate(faces):
        opensearch.index(index="board_games_faces", id=f"{image_path}_face_{i}", body={"embedding": face.embedding})
```

### Output Example

```
Loading existing paths from OpenSearch...
  Found 509,700 paths in OpenSearch
Loading existing paths from SQL Server...
  Found 878,448 paths in SQL Server
  Total unique paths to skip: 878,448
  Skipped by path: 878,448
  New paths to check: 0

No new images to index!
```

Or when there are new images:

```
Loading existing paths from OpenSearch...
  Found 509,700 paths in OpenSearch
Loading existing paths from SQL Server...
  Found 550,000 paths in SQL Server
  Total unique paths to skip: 550,000
  Skipped by path: 550,000
  New paths to check: 328,448

Hashing new files for content dedup...
Hashing new files: 100%|████████| 328448/328448 [1:23:45<00:00, 65.3it/s]

Duplicate check complete:
  New images to index: 325,000
  Skipped (already in OpenSearch): 550,000
  Skipped (same content): 3,448
  Skipped (errors): 0

Saving 325,000 hashes to SQL Server...
  Saved 325,000 hashes - safe to resume if indexing interrupted

DINOv2 Indexing: 100%|████████| 325000/325000 [2:15:30<00:00, 40.0it/s]

============================================================
BOARD GAMES DINO INDEXING COMPLETE
============================================================
Visual embeddings added: 325,000
Face embeddings added: 87,234
Errors: 12
Total in board_games_visual: 834,700
Total in board_games_faces: 198,456
```

### Crash-Safe Hash Persistence

Hashes are saved to SQL Server **immediately after hashing completes**, before indexing begins. This ensures:

1. **No re-hashing on crash**: If indexing crashes or is interrupted, the next run skips all hashed files by path
2. **Work is never lost**: The expensive hashing phase (~4 hours for 878k files) is preserved
3. **Fast recovery**: Resume indexing from where it left off without re-reading files

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Crash-Safe Workflow                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐  │
│  │ Hash Files   │────▶│ Save to SQL      │────▶│ Index to        │  │
│  │ (~4 hours)   │     │ (checkpoint!)    │     │ OpenSearch      │  │
│  └──────────────┘     └──────────────────┘     └─────────────────┘  │
│                              │                         │            │
│                              │                    ┌────┴────┐       │
│                              │                    │ CRASH!  │       │
│                              │                    └─────────┘       │
│                              ▼                                      │
│                    ┌─────────────────────────┐                      │
│                    │ Next run: Skip by path  │                      │
│                    │ (SQL has all hashes)    │                      │
│                    │ Resume from crash point │                      │
│                    └─────────────────────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### OpenSearch Scroll API

The `load_opensearch_paths()` function uses the scroll API to efficiently retrieve all document IDs:

```python
def load_opensearch_paths(os_client, index_name: str) -> set:
    """Load all existing document IDs (file paths) from OpenSearch index."""
    paths = set()
    query = {"query": {"match_all": {}}, "_source": False}  # IDs only, no content

    response = os_client.search(index=index_name, body=query, scroll="2m", size=10000)
    scroll_id = response.get("_scroll_id")
    hits = response["hits"]["hits"]

    while hits:
        for hit in hits:
            paths.add(hit["_id"])  # Document ID is the file path
        response = os_client.scroll(scroll_id=scroll_id, scroll="2m")
        hits = response["hits"]["hits"]

    os_client.clear_scroll(scroll_id=scroll_id)
    return paths
```

**Performance**: ~30-60 seconds to load 500k+ document IDs (just metadata, no vectors).

### SQL Server Path Query

```python
def load_existing_paths(cursor, collection: str) -> set:
    """Load all existing file paths for a collection into a set (single query)."""
    cursor.execute(
        "SELECT FilePath FROM ImageHashes WHERE Collection = ?",
        (collection,)
    )
    paths = set()
    for row in cursor.fetchall():
        paths.add(row[0])
    return paths
```

**Performance**: ~5-10 seconds to load 500k+ paths.

### Usage

```bash
# Run the optimized board games indexer
cd backend
python board_games_dino_indexer.py --source "T:/archiverelated/board games"

# Visual only (skip faces)
python board_games_dino_indexer.py --source "T:/archiverelated/board games" --visual-only

# Faces only (skip visual)
python board_games_dino_indexer.py --source "T:/archiverelated/board games" --faces-only

# Skip all dedup (not recommended, will reprocess everything)
python board_games_dino_indexer.py --source "T:/archiverelated/board games" --no-dedup
```

### Batch Files

- `run_board_games_dino.bat` - Runs visual then faces indexing
- `snapshot_boardgames.bat` - Creates OpenSearch snapshots to NAS
- `restore_boardgames.bat` - Restores snapshots from NAS
- `verify_board_games.bat` - Verifies indexing completeness, finds/deletes duplicates, indexes missing files

---

## Indexing Verification & Duplicate Detection

The `verify_indexing.py` script provides post-indexing verification to ensure all NAS images are indexed and to identify/remove content duplicates.

### The Problem

After indexing 878k+ images, several issues can occur:
1. **Missing files**: Some images may not be in OpenSearch (indexing errors, crashes, new files added)
2. **Content duplicates**: Same image content exists at multiple paths (copies, backups, renamed files)
3. **Wasted storage**: Duplicate files consume NAS space unnecessarily

### The Solution: Three-Phase Verification

```
┌───────────────────────────────────────────────────────────────────────┐
│               Indexing Verification Pipeline                           │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Phase 1: Path Comparison (fast, no file I/O)                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  NAS Scan              OpenSearch Scroll                        │  │
│  │  ┌──────────────┐      ┌──────────────┐                         │  │
│  │  │ 878,448      │      │ 877,100      │                         │  │
│  │  │ image files  │      │ document IDs │                         │  │
│  │  └──────┬───────┘      └──────┬───────┘                         │  │
│  │         │                      │                                │  │
│  │         └──────────┬───────────┘                                │  │
│  │                    ▼                                            │  │
│  │         ┌─────────────────────┐                                 │  │
│  │         │ Set Difference      │                                 │  │
│  │         │ NAS - OpenSearch    │                                 │  │
│  │         │ = 1,348 missing     │                                 │  │
│  │         └─────────────────────┘                                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                       │                                               │
│  Phase 2: Content Hash Dedup (reads missing files only)               │
│                       ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  for path in missing_files:  # Only 1,348 files, not 878k!      │  │
│  │      hash = SHA256(file)                                        │  │
│  │      original = SQL.lookup(hash, collection)                    │  │
│  │                                                                 │  │
│  │      if original exists and original != path:                   │  │
│  │          → DUPLICATE (same content at different path)           │  │
│  │      else:                                                      │  │
│  │          → TRULY MISSING (needs indexing)                       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                       │                                               │
│                       ▼                                               │
│         ┌─────────────────────────────────────────┐                   │
│         │  Results:                               │                   │
│         │  - 1,200 duplicates (delete optional)   │                   │
│         │  - 148 truly missing (index these)      │                   │
│         └─────────────────────────────────────────┘                   │
│                       │                                               │
│  Phase 3: Fix Issues (optional)                                       │
│                       ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  --delete-duplicates:                                           │  │
│  │      Confirm → Delete 1,200 duplicate files → Free X GB         │  │
│  │                                                                 │  │
│  │  --index-missing:                                               │  │
│  │      Load DINOv2 + ArcFace → Index 148 files → Update indices   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Why This Works

| Phase | What It Checks | File I/O Required | Speed |
|-------|---------------|-------------------|-------|
| **Path comparison** | Is path in OpenSearch? | None (metadata only) | ~1 min for 878k |
| **Content hash** | Is content a duplicate? | Only missing files | ~5 min for 1k files |
| **Indexing** | Generate embeddings | Only truly missing | ~10 min for 100 files |

**Key insight**: By comparing paths first (fast), we reduce the files needing content hashing from 878k to just ~1k. Then content hashing identifies duplicates vs truly missing files.

### Duplicate Detection Logic

```python
# A file is a DUPLICATE if:
# 1. It's not in OpenSearch (missing from index)
# 2. Its SHA256 hash exists in SQL ImageHashes table
# 3. The original path (from SQL) is different from this path

def is_duplicate(path, cursor, collection):
    file_hash = sha256(path)
    original = SQL.get_path_for_hash(file_hash, collection)

    if original and original != path:
        return True, original  # Duplicate of 'original'
    return False, None  # Truly missing, needs indexing
```

### Output Example

```
============================================
Board Games Indexing Verification
============================================
Source: T:\archiverelated\board games

Scanning T:\archiverelated\board games...
  Found 878,448 images on NAS

Loading paths from OpenSearch...
  Found 877,100 in dinov2-board_games
  Found 498,345 face embeddings from 312,000 images in faces-board_games

============================================================
RESULTS
============================================================
Total images on NAS:        878,448
Total in visual index:      877,100
Missing from visual index:  1,348
Images with faces indexed:  312,000

Visual coverage: 99.85%

*** 1,348 IMAGES NOT IN OPENSEARCH ***

============================================================
DUPLICATE DETECTION
============================================================
Checking 1,348 missing files for content duplicates...
Checking duplicates: 100%|████████| 1348/1348 [00:45<00:00, 30.0it/s]

============================================================
DUPLICATE ANALYSIS RESULTS
============================================================
Content duplicates found:   1,200
Space used by duplicates:   2.34 GB
Truly missing (not dupes):  148
Errors (couldn't check):    0

============================================================
DELETING DUPLICATES
============================================================
Are you sure you want to DELETE 1,200 duplicate files? (yes/no): yes
Deleting: 100%|████████| 1200/1200 [00:12<00:00, 100.0it/s]

Deleted: 1,200 files
Errors:  0 files
Space freed: 2.34 GB

============================================================
INDEXING MISSING FILES
============================================================
Files to index: 148

Loading DINOv2 model...
Loading ArcFace model...
Indexing: 100%|████████| 148/148 [02:30<00:00, 1.0it/s]

============================================================
INDEXING COMPLETE
============================================================
Visual embeddings added: 148
Face embeddings added:   23
Errors:                  0
```

### Usage

```bash
# Quick verification only (no file reads, just path comparison)
verify_board_games.bat --quick

# Find duplicates but don't delete or index
verify_board_games.bat --find-duplicates

# FULL FIX (default): Find duplicates, delete them, index missing
verify_board_games.bat
```

### Logging

All output is logged to `backend/verify_indexing.log` for review if the console window closes.

---

## Future Improvements

1. Add face search endpoint (currently visual only)
2. Implement thumbnail caching
3. Add pagination for large result sets
4. Support PDF direct upload (extract pages automatically)
5. Add collection management UI
6. ~~Implement incremental indexing (skip already indexed files)~~ ✅ Implemented
