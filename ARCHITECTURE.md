# Deception Lens - System Architecture

A visual search engine for scanned book pages, board games, and print ads. Given an image (or text), it finds visually similar or matching pages across millions of indexed images.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Deception Lens                                │
│                                                                            │
│  ┌──────────────────┐         ┌──────────────────────────────────────────┐  │
│  │  Blazor Server   │  HTTP   │             FastAPI Backend              │  │
│  │  (C# / .NET 8)   │◄──────►│             (Python 3.11+)              │  │
│  │  Port 5000       │         │             Port 8000                   │  │
│  └──────────────────┘         └──────┬──────┬──────┬──────┬────────────┘  │
│                                      │      │      │      │              │
│              ┌───────────────────────┘      │      │      └────────┐     │
│              ▼                              ▼      ▼               ▼     │
│   ┌──────────────────┐         ┌────────────────┐  ┌────────────────┐   │
│   │    FAISS Index   │         │   OpenSearch   │  │   SQL Server   │   │
│   │  (CLIP Vectors)  │         │ (DINOv2+Faces) │  │ (History/Hash) │   │
│   │  D:/faiss/       │         │ localhost:9200 │  │ localhost/trivia│   │
│   └──────────────────┘         └────────────────┘  └────────────────┘   │
│              │                                                           │
│              │         ┌─────────────────────────────────────────┐       │
│              └────────►│  DISK Chunk Search (Keypoint Matching)  │       │
│                        │  441 chunks × 22GB = ~9.7TB on NAS     │       │
│                        │  Rolling buffer: copy → search → delete │       │
│                        │  Queue: 1 search at a time              │       │
│                        └─────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Technologies Used

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Blazor Server (.NET 8, C#) | Search UI, results display, search history |
| **Backend API** | FastAPI (Python) | REST API, model inference, search orchestration |
| **Semantic Search** | CLIP ViT-L/14 + FAISS | Text-to-image and image-to-image similarity (768-dim) |
| **Visual Search** | DINOv2 ViT-B/14 + OpenSearch | Fine-grained visual similarity (768-dim) |
| **Face Search** | ArcFace/InsightFace buffalo_l + OpenSearch | Face recognition and matching (512-dim) |
| **Keypoint Search** | DISK (Kornia) + FAISS chunks | Cropped image source finding via keypoint voting |
| **Geometric Verification** | DISK + LightGlue | Local keypoint matching for re-ranking |
| **Vector Database** | OpenSearch 2.x (HNSW) | DINOv2 and face embeddings |
| **Vector Index** | FAISS (IndexFlatIP) | CLIP embeddings, DISK descriptors |
| **Relational Database** | SQL Server (trivia) | Search history, image hashes, deduplication |
| **GPU** | NVIDIA RTX 4070 Super (CUDA) | Model inference, feature extraction |

---

## How to Run

### Start Backend
```bash
cd backend
python server.py
# Runs on http://localhost:8000
# DISK search queue starts automatically
```

### Start Frontend
```bash
cd web
dotnet run
# Runs on http://localhost:5000
```

### Required Services
- **OpenSearch** on `localhost:9200` (for DINOv2 and face search)
- **SQL Server** on `localhost` database `trivia` (Windows Auth)
- **NAS** mapped to `T:` drive (source images and DISK chunks)

---

## Search Modes

The web UI offers 5 search modes:

### 1. Keypoint Search (DISK Chunks)
**UI Label:** "Keypoint Search" — "Find the source of cropped images."

**What it does:** Given a cropped/zoomed portion of a page, finds the original full source page by matching local keypoint features across all indexed books.

**How it works:**
1. Extract DISK keypoints from query image (128-dim descriptors, normalized)
2. Search each FAISS chunk (441 chunks, ~22GB each) for nearest neighbors per keypoint
3. Each matched keypoint "votes" for the page it belongs to
4. Pages with the most votes are the best matches
5. Results aggregated across all chunks

**Key parameters:**
- `k=5` — nearest neighbors per keypoint
- `threshold=0.7` — minimum similarity to count as a vote
- `top_k=50` — number of results returned

**Performance:** ~5.5 minutes per chunk × 441 chunks = ~40 hours for full search (network-bound at ~14 MB/s from NAS). With 10GbE upgrade, ~5-6 hours.

**Queue system:** Only 1 keypoint search runs at a time. Others wait in queue to prevent GPU memory exhaustion and SSD overflow.

---

### 2. CLIP Text Search
**UI Label:** "CLIP Search" — text input

**What it does:** Search by text description (e.g., "red truck", "sunset beach").

**How it works:**
1. Encode text with CLIP ViT-L/14 → 768-dim vector
2. Search FAISS index (IndexFlatIP, inner product similarity)
3. Return top-k results

**Supports:** "All Collections" mode — searches books, board_games, print_ads, albums in parallel and merges results.

---

### 3. CLIP Visual Search
**UI Label:** "CLIP Search" — image upload

**What it does:** Find semantically similar images (same concept, not exact match).

**How it works:**
1. Encode image with CLIP ViT-L/14 → 768-dim vector
2. Search FAISS index
3. Return top-k results

---

### 4. Face Search
**UI Label:** "Face Search"

**What it does:** Find matching faces using ArcFace recognition.

**How it works:**
1. Detect faces in query image using InsightFace
2. Extract ArcFace embedding (512-dim) for each face
3. Search OpenSearch faces index (HNSW, cosine similarity)
4. Return top-k results

---

### 5. Deep Search
**UI Label:** "Deeper Search" button (appears after CLIP or DINOv2 search)

**What it does:** Multi-stage search combining CLIP and DINOv2 with re-ranking for more accurate results.

**How it works:**
1. Parallel CLIP + DINOv2 similarity search (retrieval_k=20000)
2. Merge and deduplicate results
3. Re-rank top candidates (rerank_k=1000)
4. Return final top-k

---

## How Indexing Works

### CLIP Indexing (FAISS)
```bash
cd backend
python clip_indexer.py  # or via API: POST /clip/index
```
- Model: CLIP ViT-L/14 (768-dim embeddings)
- Batch size: 64 images
- Output: `index.faiss` + `paths.json` per collection
- Checkpoint/resume support for crash recovery

### DINOv2 + Face Indexing (OpenSearch)
```bash
cd backend
# Visual pass (DINOv2 on GPU)
python main.py index --dir "D:\books\pdf-images\BookName" --collection books --mode visual_only

# Face pass (InsightFace on GPU) - separate pass due to GPU memory conflicts
python main.py index --dir "D:\books\pdf-images\BookName" --collection books --mode faces_only
```
- DINOv2: 768-dim visual embedding per image → OpenSearch HNSW index
- ArcFace: 512-dim face embedding per detected face → OpenSearch HNSW index
- Two-pass system: PyTorch (DINOv2) and ONNX Runtime (InsightFace) conflict on GPU

### DISK Feature Indexing (for Keypoint Search)
```bash
cd backend
python batch_disk_index_file.py
```
- Extracts DISK keypoints + 128-dim descriptors per image
- Stores as `.npz` files on NAS (`T:/disk-features/books/`)
- Multi-threaded pipeline: 8 workers for image decode, async saves
- Rate: ~10-13 images/sec on RTX 4070 Super
- Total: ~19.6 billion keypoints across 4,735 books

### DISK Chunk Building (FAISS chunks)
After DISK features are extracted, they're consolidated into searchable FAISS chunks:
- Each chunk: ~22GB FAISS index + ~6.5GB paths JSON
- 441 chunks total on NAS at `T:/faiss/disk_retrieval/chunks/`
- Each chunk contains keypoint descriptors for a batch of books

### Deduplication Strategy
Four-stage pipeline prevents reprocessing:
1. **Path-based skip** — Check OpenSearch + SQL for known paths (instant, no I/O)
2. **Content hash** — SHA256 only for new files
3. **Hash checkpoint** — Save hashes to SQL before indexing (crash-safe)
4. **Embedding generation** — Only for truly new images

---

## Storage Map

### Local SSD (D: drive)
| Path | Contents | Size |
|------|----------|------|
| `D:/faiss/books/` | CLIP FAISS index + paths | ~15 GB |
| `D:/faiss/board_games/` | CLIP FAISS index + paths | ~10 GB |
| `D:/faiss/printads/` | CLIP FAISS index + paths | ~2.7 GB |
| `D:/faiss/albums/` | CLIP FAISS index + paths | ~2.3 GB |
| `D:/faiss/disk_retrieval/chunk_buffer/` | Rolling buffer for DISK chunks (temp) | ~110 GB max |
| `D:/books/pdf-images/` | Source book page images | ~1.5 TB |

### NAS (T: drive)
| Path | Contents | Size |
|------|----------|------|
| `T:/faiss/disk_retrieval/chunks/` | 441 DISK FAISS chunks | ~9.7 TB |
| `T:/disk-features/books/` | DISK `.npz` feature files per book | ~2 TB |
| `T:/disk-features/board_games/` | DISK `.npz` features | varies |
| `T:/disk-features/print_ads/` | DISK `.npz` features | varies |
| `T:/archiverelated/books/` | Source images (NAS copy) | varies |
| `T:/faiss/` | Backup of D:/faiss/ | varies |

### OpenSearch (localhost:9200)
| Index | Dimensions | Documents | Purpose |
|-------|-----------|-----------|---------|
| `dinov2-books` | 768 | ~5M | DINOv2 visual search |
| `dinov2-board_games` | 768 | ~877k | DINOv2 visual search |
| `dinov2-print_ads` | 768 | TBD | DINOv2 visual search |
| `faces-books` | 512 | TBD | Face search |
| `faces-board_games` | 512 | ~498k | Face search |
| `faces-print_ads` | 512 | TBD | Face search |

### SQL Server (localhost/trivia)
| Table | Purpose |
|-------|---------|
| `ImageHashes` | SHA256 hashes for deduplication |
| `ImageSearchHistory` | Search sessions with progress tracking |
| `ImageSearchResults` | Results for each search session |

---

## DISK Chunk-Based Keypoint Search (Deep Dive)

This is the most complex feature. It enables finding the source page of a cropped image across millions of pages.

### Why Chunks?
- Each book page has ~4,000 DISK keypoints (128-dim descriptors)
- ~5M pages × 4,000 keypoints = ~19.6 billion descriptors
- Too large to fit in memory → split into 441 FAISS chunks (~22GB each)
- Only 1 chunk can be loaded at a time (GPU/RAM constraint)

### Rolling Buffer Strategy
```
┌─────────────────────────────────────────────────────────────────┐
│                    Rolling Buffer (5 slots)                      │
│                                                                  │
│  Copy Thread (background):     Search Thread (main):             │
│  ┌──────────────┐              ┌──────────────┐                  │
│  │ Copy chunk 6 │              │ Search chunk 1│ ← votes added   │
│  │ from NAS     │              │ then delete   │                 │
│  │ to local SSD │              └──────────────┘                  │
│  └──────────────┘                                                │
│                                                                  │
│  Buffer on SSD:                                                  │
│  [chunk_2] [chunk_3] [chunk_4] [chunk_5] [chunk_6_copying...]   │
│                                                                  │
│  After search:                                                   │
│  [deleted] [chunk_3] [chunk_4] [chunk_5] [chunk_6] [chunk_7...]│
└─────────────────────────────────────────────────────────────────┘
```

1. Background thread copies chunks from NAS (`T:`) to local SSD (`D:`)
2. Main thread searches completed chunks, accumulates votes
3. After searching, chunk is deleted from SSD to free space
4. Copy and search overlap — while searching chunk N, chunk N+5 is being copied
5. **FAISS indexes** are copied to local SSD (faster random access)
6. **Paths JSON files** are read directly from NAS (avoids corruption with large files)

### Vote Aggregation
```python
# For each chunk:
#   1. Search FAISS index: for each query keypoint, find k=5 nearest neighbors
#   2. If similarity > threshold (0.7), that neighbor's page gets a vote
#   3. Votes accumulate across all 441 chunks
#
# Final ranking: pages sorted by total vote count (descending)
```

### Network Copy Issue
Large JSON files (6.5GB `paths.json`) were corrupted during copy from NAS (known robocopy issue with large files over network — matching file sizes but different MD5 hashes). Solution: read paths files directly from NAS instead of copying them.

---

## Queue System

Only one DISK keypoint search can run at a time. The queue system prevents:
- **GPU OOM**: Multiple DISK models loaded simultaneously
- **SSD overflow**: Multiple 22GB chunks being copied at once (5 × 22GB per search)
- **File conflicts**: Multiple searches writing to same chunk_buffer directory
- **Network congestion**: Multiple 14 MB/s copy streams saturating NAS

### How It Works
```
User 1: POST /disk/search → Queue position 1 → Runs immediately
User 2: POST /disk/search → Queue position 2 → Sees "Waiting in queue (position: 1)"
User 3: POST /disk/search → Queue position 3 → Sees "Waiting in queue (position: 2)"

User 1 finishes → User 2 starts automatically → User 3 moves to position 1
```

- Queue processor runs as async background task
- Searches executed via `asyncio.to_thread()` (non-blocking)
- Queue status visible in web UI through database Notes field
- Check queue: `GET /disk/queue`

---

## Live Search Tracking

Long-running searches (especially keypoint) show real-time progress in the web UI.

### How It Works
1. Backend creates a search session in SQL (`ImageSearchHistory`)
2. During search, progress is updated every chunk: current chunk, total chunks, top results so far
3. Frontend polls the database for progress updates
4. User sees: "Searching chunk 142/441..." with a progress bar
5. On completion, final results are stored in `ImageSearchResults`

### Database Flow
```
create_search_session() → Status: "in_progress"
    ↓
update_search_progress() → CurrentProgress: 142, TotalChunks: 441, interim results
    ↓ (repeated for each chunk)
complete_search_session() → Status: "completed", Duration: 145232ms
```

---

## Search History

All searches are saved to SQL Server and viewable in the web UI's History panel.

**Stored per search:**
- Search type (CLIP Text, CLIP Visual, DINOv2, Face, Keypoint, Deep Search)
- Query image (stored as binary)
- Query text (for text searches)
- All results with paths and scores
- Search duration
- Collection searched
- Notes (queue position, etc.)

---

## API Endpoints

### Search
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | DINOv2 image search (with optional geometric verification) |
| `/clip/search` | POST | CLIP image search |
| `/clip/text` | POST/GET | CLIP text-to-image search |
| `/face/search` | POST | Face similarity search |
| `/disk/search` | POST | Keypoint search (queued, live tracking) |
| `/disk/queue` | GET | Check keypoint search queue status |
| `/deep-search` | POST | Multi-stage CLIP + DINOv2 search |

### Stats & Info
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check |
| `/stats` | GET | Collection statistics |
| `/clip/stats` | GET | CLIP index statistics |
| `/collections` | GET | List all collections |
| `/image` | GET | Serve image from filesystem |

### Search History
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search-history` | GET | Paginated search history |
| `/search-history/{id}` | GET | Full search details |
| `/search-history/{id}` | DELETE | Delete search record |
| `/search-history/{id}/image` | GET | Query image for search |
| `/search-history/{id}/note` | PUT | Add note to search |

### Indexing
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/clip/index` | POST | Start CLIP indexing (background) |
| `/clip/index/status` | GET | Get indexing progress |
| `/clip/index/add` | POST | Add folder to existing index |

---

## Directory Structure

```
DinoDeceptionLens/
├── backend/
│   ├── server.py                    # FastAPI REST API server
│   ├── disk_searcher.py             # DISK chunk-based keypoint search
│   ├── disk_queue.py                # Search queue (1 at a time)
│   ├── db_helper.py                 # SQL Server helpers (history, progress)
│   │
│   ├── clip_searcher.py             # CLIP text/image search (FAISS)
│   ├── clip_indexer.py              # CLIP FAISS index builder
│   ├── searcher.py                  # DINOv2 search + LightGlue verification
│   ├── indexer.py                   # DINOv2 indexing to ChromaDB/OpenSearch
│   ├── opensearch_searcher.py       # OpenSearch vector search
│   │
│   ├── disk_features_file.py        # DISK feature file storage (.npz)
│   ├── disk_indexer_file.py         # DISK feature extraction (file-based)
│   ├── batch_disk_index_file.py     # Batch DISK indexing
│   ├── disk_features.py             # DISK features (SQL storage, legacy)
│   ├── disk_indexer.py              # DISK extraction (SQL, legacy)
│   │
│   ├── batch_index.py               # Batch DINOv2 indexing
│   ├── board_games_dino_indexer.py   # Board games indexer with dedup
│   ├── verify_disk_coverage.py       # Verify DISK feature completeness
│   ├── verify_opensearch_coverage.py # Verify OpenSearch completeness
│   ├── verify_indexing.py           # Post-indexing verification + dedup
│   │
│   ├── main.py                      # CLI entry point
│   └── test_3chunk.py               # Test script for 3-chunk validation
│
├── web/
│   ├── Pages/
│   │   └── Index.razor              # Main search UI
│   ├── Services/
│   │   └── SearchService.cs         # HTTP client for backend API
│   ├── Models/
│   │   └── SearchResult.cs          # Data models
│   └── Program.cs                   # ASP.NET Core startup
│
├── ARCHITECTURE.md                  # This file
├── DISK_SEARCH_QUICKSTART.md        # Quick start for DISK search
├── TRIVIA-IMAGE-SEARCH.md           # Detailed trivia search docs
└── backend/
    ├── DISK_SEARCH_STRATEGY.md      # DISK indexing strategy
    └── DISK_SEARCH_STATUS.md        # Implementation status
```

---

## Dependencies

### Python (backend)
```
# Core
torch, torchvision
transformers
fastapi, uvicorn
pillow, opencv-python
numpy, tqdm

# CLIP / FAISS
clip (git+https://github.com/openai/CLIP.git)
faiss-cpu (or faiss-gpu)

# DINOv2 / Geometric Verification
kornia                    # DISK keypoint extractor
insightface              # ArcFace face recognition
onnxruntime-gpu          # InsightFace inference
chromadb                 # Legacy vector store

# OpenSearch
opensearch-py

# Database
pyodbc                   # SQL Server connection
```

### C# (web)
```
.NET 8.0
Microsoft.AspNetCore.Components
Microsoft.Extensions.Http
System.Text.Json
```

---

## Key Configuration

| Setting | Value | Where |
|---------|-------|-------|
| Backend port | 8000 | `server.py` or `PORT` env var |
| Frontend port | 5000 | `web/Properties/launchSettings.json` |
| CLIP model | ViT-L/14 | `clip_searcher.py` |
| DINOv2 model | facebook/dinov2-base | `indexer.py` |
| DISK chunk dir | `T:/faiss/disk_retrieval/chunks/` | `disk_searcher.py` |
| Chunk buffer dir | `D:/faiss/disk_retrieval/chunk_buffer/` | `disk_searcher.py` |
| Buffer size | 5 chunks | `disk_searcher.py` |
| DISK features dir | `T:/disk-features/` | `disk_features_file.py` |
| SQL Server | `localhost/trivia` (Windows Auth) | `db_helper.py` |
| OpenSearch | `localhost:9200` | `opensearch_searcher.py` |

---

## Test Case: Cropped Image Search

**Scenario:** Find the source page of a cropped dinosaur image.

- **Query:** `D:/trivpics/2023-5.jpg` (cropped Manglosaurus)
- **Target:** `encyclopedia of monsters-page210.jpg` (full page with 3 figures + text)

**Results with Keypoint Search (3-chunk test, chunks 141-143):**
- Encyclopedia of Monsters pages received **145 votes** — correctly identified
- Search completed in ~16.5 minutes for 3 chunks

**Why embedding-only search fails here:**
- DINOv2 cosine similarity between crop and full page = **0.249** (very low)
- The crop would rank at position ~2589 out of 5000 candidates
- Keypoint matching finds **662 verified local matches** and promotes it to rank #1

---

## Current Scale

| Metric | Value |
|--------|-------|
| Books indexed | 4,735 (69.4% of total) |
| Total images (books) | ~5M pages |
| DISK keypoints | ~19.6 billion |
| FAISS chunks | 441 × ~22GB |
| Total chunk storage | ~9.7 TB |
| CLIP index (books) | ~15 GB |
| Collections | books, board_games, print_ads, albums |
