# DISK Search - Status & Implementation

## Architecture

### Chunk Format (10 GB target, GPU FAISS compatible)
- FAISS `IndexFlatIP(128)` chunks, ~19.5M vectors each (~10 GB)
- Sized to fit in 16 GB VRAM (4070 Ti Super) with headroom for DISK model + scratch
- Compact IDs: `chunk_XXX_ids.npy` (int32) + `path_lookup.json` (ID→path)
- Both books and print ads use identical chunk + compact ID format

### Storage Layout

**Books:**
- Per-book shards (source): `T:/faiss/disk_retrieval/books/` (4,238 books)
- FAISS chunks (NAS): `T:/faiss/disk_retrieval/chunks/`
- Compact IDs (local SSD): `D:/faiss/disk_retrieval/chunk_ids/`

**Print Ads:**
- Source images (local copy): `C:\printads` (copied from NAS for GPU extraction)
- Source images (NAS): `T:\archiverelated\print ads`
- FAISS chunks (NAS): `S:\faiss\disk_retrieval\printads_chunks\`
- Compact IDs (local SSD): `D:\faiss\disk_retrieval\printads_chunk_ids\`

### Scripts

| Script | Purpose |
|--------|---------|
| `consolidate_search_chunks.py` | Re-chunk book shards → 10 GB chunks + compact IDs |
| `build_printads_disk_chunks.py` | Extract DISK features from print ads → 10 GB chunks + compact IDs |
| `disk_searcher.py` | Search chunks (CLI, no history) |
| `test_disk_api.py` | Search via API (saves to history + live tracking) |
| `disk_queue.py` | Queue system, 1 search at a time |
| `convert_paths_to_ids.py` | One-time conversion of old paths.json → compact IDs |

### Chunk Sizing Rationale
- Vector: 128 dims × 4 bytes = 512 bytes
- 19.5M vectors × 512 bytes = ~10 GB
- Fits in 16 GB VRAM (4070 Ti Super) or 32 GB (5090) with headroom
- Print ads: variable keypoints per image, script flushes when vector count hits 19.5M
- Books: accumulates per-book vectors, flushes when exceeding 19.5M

## Current Index Stats

**Books (existing, will be re-chunked after 10 Gbps upgrade):**
- 606 chunks (old 20 GB format, will become ~1,200 at 10 GB)
- 23.6 billion keypoints, 2.9M unique pages
- Compact IDs: 95 GB on local SSD

**Print Ads (building now, 10 GB chunks from start):**
- 1,221,287 images → ~940 chunks estimated
- ~15,000 keypoints per image average
- FAISS chunks → `S:\faiss\disk_retrieval\printads_chunks\`

## Performance

### Search Times (single image against all books + print ads)

| Scenario | Books | Print Ads | Total |
|----------|-------|-----------|-------|
| Current (CPU, 113 MB/s actual) | 36 hrs | 32 hrs | **~68 hrs** |
| CPU, 113 MB/s (nothing else on NAS) | ~12 hrs | ~10 hrs | **~22 hrs** |
| faiss-gpu (10 GB chunks) + 1 Gbps | 3 hrs | 2 hrs | **~5 hrs** (estimated) |
| faiss-gpu (10 GB chunks) + 10 Gbps | 1 hr | 0.5 hr | **~1.5 hrs** |
| faiss-gpu + 5090 + 10 Gbps | 15 min | 10 min | **~25 min** |

### Bottleneck Analysis
- **Current**: Network copy dominates (NAS→SSD). 113 MB/s when NAS is idle, ~14 MB/s when build is running
- **With faiss-gpu**: GPU search ~2-3s per 10 GB chunk vs 30-60s on CPU
- **Hard floor**: GPU compute time. No network upgrade helps below ~1.5 hrs (4070 Ti) or ~25 min (5090)

### Network
- Link: 1 Gbps, actual throughput: 113 MB/s (verified with chunk copy test)
- 14 MB/s observed during builds = NAS I/O contention, not network limit
- 10 Gbps upgrade planned

## Search Methods

### CLI (no history)
```bash
python disk_searcher.py "D:\trivpics\2023-5.jpg" 140,141,142,143,144
```

### API (saves to history + live tracking)
```bash
curl -X POST "http://localhost:8000/disk/search?top_k=10&chunk_ids=140,141,142,143,144&live_tracking=true" \
  -F "file=@D:\trivpics\2023-5.jpg"
```

**API endpoint**: `POST /disk/search`
- `chunk_ids` (optional): Comma-separated chunk numbers
- `top_k`: Number of results (default 50)
- `k`: Nearest neighbors per keypoint (default 5)
- `threshold`: Minimum similarity for voting (default 0.7)
- `live_tracking`: Enable live progress in DB (default true)

### Web UI
- DISK search panel on `https://localhost:5001`
- Upload cropped image → "Find Source Page"
- Searches all chunks by default (no chunk filter in UI yet)

## Test Images
- **T-Rex dinosaur crop**: `D:\trivpics\2023-5.jpg`
  - Target: Encyclopedia of Monsters, page 206
  - Expected: 145 votes (top result), found in chunk 142
  - Chunks to test: 140-144

## Build Scripts

### Print Ads (run now)
```bash
run_build_printads_chunks.bat
```
- Extracts DISK features on GPU, builds chunks by vector count (not fixed image count)
- Flushes chunk when accumulator hits 19.5M vectors (~10 GB)
- Resumable via `build_progress.json`

### Books (run after 10 Gbps upgrade)
```bash
python consolidate_search_chunks.py
```
- Re-chunks existing per-book shards into 10 GB chunks with compact IDs
- Same vector-count flush logic as print ads
- Will replace current 606 × 20 GB chunks with ~1,200 × 10 GB chunks

## GPU FAISS Notes
- `faiss-gpu` (`pip install faiss-gpu-cu12`) required for GPU search
- Cannot mmap on GPU - entire index must fit in VRAM
- 10 GB chunks fit in 16 GB VRAM (4070 Ti Super) with ~4 GB headroom
- 5090 (32 GB) could do ~24 GB chunks but 10 GB is the sweet spot (works on both)
- GPU search: ~2-3s per 10 GB chunk vs 30-60s on CPU

## Hardware Notes

### Current System

- iBUYPOWER: Ryzen 9 7950X, MSI X670E Tomahawk, 64 GB DDR5-4800
- GPU: NVIDIA 4070 Ti Super (16 GB VRAM)
- PSU: 1000W Corsair RM1000e (PCIe Gen 5 ready)
- Storage: 4 TB Samsung 990 PRO NVMe, Lian Li Lancool 216 case
- 5090 compatible (PSU/slot/case all support it) but $4K — not worth it

### 5090 Assessment

- 32 GB VRAM could do 24 GB chunks, but only saves ~10 min vs 10 GB chunks on 4070 Ti
- The 4070 Ti Super + faiss-gpu + 10 Gbps gets 95% of the benefit at $0 extra cost
- If 5090 prices drop, it's a plug-and-play upgrade (same 10 GB chunks work fine)

### Strix Halo (ASUS ROG 395+, 128 GB RAM)

- Only ~64 GB usable in Windows (iGPU reserves half)
- Explored as CPU FAISS search node (10 chunks parallel in RAM)
- Killed by 1 Gbps network — loading 10 GB per chunk from NAS takes longer than CPU search
- Would need all chunks on local NVMe (~33 TB) to be useful, not practical
- Conclusion: not worth the complexity for DISK search

### Upgrade Path

1. **Now**: faiss-gpu on 4070 Ti Super (free, ~5 hrs at 1 Gbps)
2. **Soon**: 10 Gbps network upgrade (~1.5 hrs total search)
3. **Maybe later**: 5090 if prices drop (~25 min total search)

## Things to Consider

### DINOv2 May Not Be Useful for Trivia Search

- 47 GB across 3 collections in OpenSearch (books 10 GB, print_ads 22 GB, board_games 15 GB)
- Tried using DINOv2 as a pre-filter for DISK — it filtered too aggressively and excluded correct results
- For crop-based trivia search, DISK brute-force against everything is the right approach
- DINOv2 overlaps with CLIP but has no text search capability
- CLIP is strictly better for this use case (semantic understanding + text-to-image)
- Keeping DINOv2 for now since 47 GB is negligible, but it's the weakest link in the stack

### Pre-Filtering Doesn't Work for Crop Search

- The trivia use case is always a small crop from a larger image
- Pre-filters (DINOv2, CLIP) compare whole-image embeddings — a crop looks nothing like the full source page
- DISK works because it matches local keypoints, not global image features
- Brute-force DISK against all chunks is the correct strategy, not narrowing candidates first

### Print Ads Not Yet Wired Into Searcher

- `disk_searcher.py` and `server.py` only search book chunks
- When print ads build finishes, need to add print ads chunks to the search path
- Print ads chunks: `S:\faiss\disk_retrieval\printads_chunks\`
- Print ads compact IDs: `D:\faiss\disk_retrieval\printads_chunk_ids\`

### Book Chunk Paths Need Updating

- Code still references `T:/faiss/disk_retrieval/chunks` for book chunks
- Book chunks being moved to `S:/faiss/disk_retrieval/chunks` via robocopy
- Update `disk_searcher.py`, `server.py`, and `consolidate_search_chunks.py` when ready

## Remaining Work
- [ ] Finish print ads build (~6 days estimated)
- [ ] Upgrade to 10 Gbps network
- [ ] Install faiss-gpu and update disk_searcher.py
- [ ] Re-chunk books to 10 GB with compact IDs
- [ ] Run database migration for live tracking
- [ ] Add chunk filter to web UI
- [ ] Index remaining ~2,213 unprocessed books
