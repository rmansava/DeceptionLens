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
2. **Indexing is GPU-bound** - processes ~1-3 images/second on RTX 3090
3. **Face detection adds overhead** - only finds faces in ~15% of book pages
4. **OpenSearch k-NN is fast** - uses HNSW algorithm for approximate nearest neighbors
5. **Batch indexing resumes** - progress saved to `batch_progress_opensearch.txt`
6. **Crop search requires large candidate pool** - fetch_k=5000 for snippets to work
7. **DISK feature cache** - pre-computed features reduce verification from ~17min to ~2-3min

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

## Future Improvements

1. Add face search endpoint (currently visual only)
2. Implement thumbnail caching
3. Add pagination for large result sets
4. Support PDF direct upload (extract pages automatically)
5. Add collection management UI
6. Implement incremental indexing (skip already indexed files)
