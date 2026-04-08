# DISK Search Test Plan

Known query -> source-page pairs and performance benchmarks for validating
search correctness and optimizing contest-day workflow.

## Known Pairs

| Test Image | Expected Source | Expected Page | Collection |
|---|---|---|---|
| `D:\trivpics\2023-5.jpg` | Encyclopedia Of Monsters, The | page 206 | books |
| `D:\trivpics\2024-8.png` | Ad boy Vintage advertising with character - Warren Dotz | page 98 | books |
| `D:\stcloudtrivia\2023-19.jpg` | Television cartoon shows an illustrated encyclopedia | page 365 | books |

---

## Test 1: Canonical Single-Chunk Check

Purpose: Confirm DISK correctness on a known pair (< 1 minute).

```bash
curl -X POST "http://localhost:8000/disk/search?chunk_ids=183&collections=books&top_k=20&k=5&threshold=0.7" -F "file=@D:/trivpics/2023-5.jpg"
```

Expected:
- #1 path ends with `Encyclopedia Of Monsters...-page206.jpg`
- Top votes ~130-170
- Strong separation from #2 (> 100 vote margin)

Fail conditions:
- Correct page is not rank #1
- Margin vs #2 < 30

---

## Test 2: GPU vs CPU Search Speed

Purpose: Measure actual PyTorch GPU search time per chunk vs CPU fallback.

### GPU search (default — auto-detected)
```bash
curl -X POST "http://localhost:8000/disk/search?chunk_ids=183&collections=books&top_k=20" -F "file=@D:/trivpics/2023-5.jpg"
```

### CPU search (force disable GPU)
```bash
set CUDA_VISIBLE_DEVICES=
curl -X POST "http://localhost:8000/disk/search?chunk_ids=183&collections=books&top_k=20" -F "file=@D:/trivpics/2023-5.jpg"
```

Record from server logs:
- `Searched in X.Xs (GPU)` vs `Searched in X.Xs (CPU)`
- Expected: GPU ~2-5s, CPU ~30-60s per chunk

---

## Test 3: Direct NAS Read vs Rolling Buffer

Purpose: Determine if we can skip local SSD copy at 10GbE speeds.

### A. Current mode (rolling buffer — copies chunk to D: first)
Default behavior. Measure total time per chunk including copy.

### B. Direct NAS mode (load straight from NAS into RAM)
Requires code change: remove `IO_FLAG_MMAP` and read directly from NAS path.

```python
# Current (mmap from local SSD):
index = faiss.read_index(local_chunk_file, faiss.IO_FLAG_MMAP)

# Direct NAS (full load into RAM):
index = faiss.read_index(nas_chunk_file)  # no mmap, sequential read
```

Record:
- Time to load chunk from NAS into RAM
- Total per-chunk time (load + search)
- Compare to rolling buffer total per-chunk time
- Watch for NAS I/O issues or timeouts

Expected: ~9s load at 1 GB/s + ~4s search = ~13s total.
Rolling buffer hides copy behind search, so may still be faster.

---

## Test 4: Multi-Collection Search

Purpose: Verify all collections return results and config paths are correct.

Search each collection individually with the dinosaur image:

```bash
# Books (1,012 chunks on T:)
curl -X POST "http://localhost:8000/disk/search?collections=books&chunk_ids=183&top_k=5" -F "file=@D:/trivpics/2023-5.jpg"

# Board Games (672 chunks on T:)
curl -X POST "http://localhost:8000/disk/search?collections=board_games&chunk_ids=1&top_k=5" -F "file=@D:/trivpics/2023-5.jpg"

# Print Ads (1,022 chunks on T:)
curl -X POST "http://localhost:8000/disk/search?collections=print_ads&chunk_ids=1&top_k=5" -F "file=@D:/trivpics/2023-5.jpg"

# Albums (102 chunks on U:)
curl -X POST "http://localhost:8000/disk/search?collections=albums&chunk_ids=1&top_k=5" -F "file=@D:/trivpics/2023-5.jpg"

# Comics (chunks on U:)
curl -X POST "http://localhost:8000/disk/search?collections=comics&chunk_ids=1&top_k=5" -F "file=@D:/trivpics/2023-5.jpg"
```

Pass conditions:
- Each collection loads without errors
- Path lookup resolves correctly (no "path not found" errors)
- Results come back (even if low votes — the image is from books)

Fail conditions:
- "No chunks found" for any collection
- Path errors (wrong drive letter in collections_config.py)
- Chunk file not found

---

## Test 5: Full-Category Single Image Search

Purpose: Measure real wall-clock time for searching all chunks in one category.

```bash
curl -X POST "http://localhost:8000/disk/search?collections=books&top_k=50&live_tracking=true" -F "file=@D:/trivpics/2023-5.jpg"
```

Record:
- Total chunks searched (1,012 for books)
- Total wall-clock time
- Per-chunk average (total time / chunks)
- Breakdown: time spent on copy vs search vs vote aggregation
- Top result and vote count
- Check search history at http://localhost:5000/history for live progress

---

## Test 6: Batch Search (Contest Simulation)

Purpose: Verify batch mode works and estimate contest-day timing.

Put 3+ test images in a folder and run batch search:

```bash
# Create test folder
mkdir D:\test_batch
copy D:\trivpics\2023-5.jpg D:\test_batch\
copy D:\trivpics\2024-8.png D:\test_batch\
copy D:\stcloudtrivia\2023-19.jpg D:\test_batch\
```

Run batch:
```bash
cd backend
python batch_disk_search.py D:\test_batch
```

Record:
- Total time for all images
- Per-image results (correct source found?)
- Verify chunk is loaded once and all images searched against it

---

## Test 7: Hub-Page False Positive Check

Purpose: Verify dense page-0 images don't dominate results.

Run batch search on all known test pairs. For each result:
- Check if #1 is the correct source
- Check if any page-0 results appear in top 5
- Check vote margins

Baseline from prior run (search #2228, 1014 chunks):
- `2024-8.png` ranked #2 with 297 votes, #1 had 305 (8-vote margin)
- "Encyclopedia of Television Shows page 0" was a repeat false positive

---

## Reporting Template

For each test, log:

| Field | Value |
|---|---|
| Test # | |
| Query image | |
| Search ID | |
| Collections searched | |
| Total chunks | |
| Wall-clock time | |
| Per-chunk average | |
| Search method (GPU/CPU) | |
| Rank of expected page | |
| Expected page votes | |
| #1 path | |
| #1 votes | |
| #2 votes | |
| Margin (top1 - top2) | |
| Notes | |

---

## Current Chunk Inventory

| Category | Chunks | Location | Status |
|----------|--------|----------|--------|
| Books | 1,012 | T: | Done |
| Board Games | 672 | T: | Done |
| Print Ads | 1,022 | T: | Done (moved from S:) |
| Albums | 102 | U: | Stopped (1.3% done) |
| Comics | ~2,500 est | U: | Building |
| Cereal | 0 | — | Not started |
