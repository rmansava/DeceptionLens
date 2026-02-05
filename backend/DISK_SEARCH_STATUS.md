# DISK Search - Status & Implementation

## What We Accomplished Today

### 1. Fixed DISK Searcher
- **Problem**: Script used outdated `torch.hub.load()` that doesn't work with DISK
- **Solution**: Updated to use Kornia's DISK implementation (`KF.DISK.from_pretrained('depth')`)
- **Result**: DISK model loads correctly on CUDA

### 2. Fixed FAISS Memory-Mapped Loading
- **Problem**: Loading 22GB indexes took 20+ minutes and often hung
- **Solution**: Use `faiss.read_index(file, faiss.IO_FLAG_MMAP)` for instant loading
- **Result**: Load time reduced from 20+ min to ~40 seconds

### 3. Added Chunk Caching
- **Problem**: Re-copying same chunks wasted 25+ minutes each time
- **Solution**: Check if chunk exists locally with same size, skip copy if cached
- **Result**: Subsequent searches of same chunk are instant

### 4. Successfully Tested DISK Search
- **Test Image**: `D:\trivpics\2023-5.jpg` (dinosaur crop)
- **Target**: Encyclopedia of Monsters, page 206
- **Result**: **FOUND with 145 votes!** (top result)
- **Chunk**: 142 (contains 15 encyclopedia books, 44.5M vectors)
- **Time**: 27.1 minutes (25.7 min copy + 1.4 min search/load)

### 5. Implemented Rolling Buffer Strategy
- **Concept**: Keep 5 chunks (~100GB) in buffer, copy next while searching current
- **Benefits**: Parallelizes copy and search operations
- **Location**: `_search_chunks_rolling_buffer()` in disk_searcher.py

### 6. Added Live Search Tracking
- **Features**:
  - Create search session when search starts
  - Update progress after each chunk with top 100 results
  - See aggregated votes grow as chunks are searched
  - Track status: in_progress, completed, failed
- **Database Migration**: `migrations/add_live_search_tracking.sql`
- **Functions**: `create_search_session()`, `update_search_progress()`, `complete_search_session()`

### 7. Created Unprocessed Books Finder
- **Script**: `find_unprocessed_books.py`
- **Output**: `D:/faiss/disk_retrieval/unprocessed_books.txt` (2,213 books)
- **Coverage**: 69.4% indexed (4,735 / 6,820 books)

### 8. Compact ID Conversion (paths.json → compact IDs)
- **Problem**: 606 paths.json files = 3.3 TB (same path string repeated per keypoint, ~11,206x redundancy)
- **Solution**: Convert to int32 numpy arrays + global path_lookup.json
- **Result**: 3.3 TB → ~95 GB (35x smaller), loads in 0.1s vs multi-GB JSON from NAS
- **Script**: `convert_paths_to_ids.py` + `run_convert_paths.bat`
- **Output**: `D:/faiss/disk_retrieval/chunk_ids/chunk_XXX_ids.npy` + `path_lookup.json`
- **Time**: 12.3 hours for 605 chunks, 2,886,439 unique paths

### 9. Re-tested DISK Search with Compact IDs (2026-02-05)
- **Test Image**: `temp_uploads/ec4a14d7-9d4f-4c32-bdbe-72b9e3f0b058.jpg` (T-Rex dinosaur crop)
- **Target**: Encyclopedia of Monsters, page 206
- **Chunks searched**: 140, 141, 142, 143, 144
- **Result**: **FOUND with 145 votes!** (top result, same as original test)
- **Chunk containing the book**: 142 (44,502,894 vectors)
- **path_lookup.json**: loaded once in 1.5s (441 MB, 2.9M paths), cached for subsequent chunks
- **Per-chunk ID load**: 0.1s (vs minutes for NAS paths.json before)
- **Time**: 18.1 minutes for 5 chunks (mostly NAS→SSD copy, not path loading)
- **Backward compatible**: `load_chunk_paths()` tries IDs first, falls back to NAS paths.json

### 10. Search Queue System
- **Script**: `disk_queue.py` - ensures only 1 DISK search runs at a time
- **Integration**: FastAPI lifespan in `server.py`, `/disk/queue` status endpoint
- **Purpose**: Prevents GPU OOM and SSD overflow from concurrent searches

## Current Index Stats

**Chunk Index (Books):**
- Total chunks: 606
- Indexed books: ~4,735+
- Total keypoints: 23,599,882,172 (~23.6 billion)
- Index size: ~13 TB (606 chunks × ~22GB avg)
- Compact IDs: ~95 GB on local SSD (D:/faiss/disk_retrieval/chunk_ids/)
- Unique paths: 2,886,439
- path_lookup.json: 441 MB

**Book-to-Chunks Mapping:**
- File: `D:/faiss/disk_retrieval/book_to_chunks.json`
- File: `D:/faiss/disk_retrieval/chunk_to_books.json`
- Created by: `build_chunk_index.py`

## Performance Analysis

### Current Network (estimates ~14 MB/s):
- Copy 22GB chunk: ~25 minutes
- Load 22GB index (mmap): ~40 seconds
- Search 44.5M vectors: ~38 seconds
- **Total per chunk: ~27 minutes**
- **Full 441 chunks: ~198 hours (8.3 days)**

### With 10 Gigabit Network (~1 GB/s):
- Copy 22GB chunk: ~22 seconds
- Load + Search: ~78 seconds
- **Total per chunk: ~100 seconds**
- **Full 441 chunks: ~12.3 hours**

### With Rolling Buffer (5 chunks ahead):
- Copy and search happen in parallel
- Estimated speedup: 2-3x (depends on network vs search time ratio)
- **With 10GbE + Rolling Buffer: ~5-6 hours for full search**

## Network Upgrade Impact

**Current bottleneck:** Network copy (93% of search time)

**10 Gigabit Ethernet Benefits:**
- 70x faster copy speed (14 MB/s → 1 GB/s)
- Full search: 198 hours → 12.3 hours
- With rolling buffer: ~5-6 hours
- Makes full-corpus DISK search practical!

## Next Steps

### Database Migration
Run this SQL on your database before using live search:
```bash
sqlcmd -S localhost -d ImageSearch -i migrations/add_live_search_tracking.sql
```

### Index Remaining Books
```bash
# Use the unprocessed books list with your consolidation script
python consolidate_search_chunks.py --books-file D:/faiss/disk_retrieval/unprocessed_books.txt
```

### Test Rolling Buffer Search
```bash
python disk_searcher.py "D:\trivpics\2023-5.jpg"
# Or specify chunks:
python disk_searcher.py "D:\trivpics\2023-5.jpg" "100,150,200"
```

### Enable Live Search in Web UI
1. Run database migration
2. Update DISK search endpoint in server.py to use live tracking
3. Frontend will show active searches with real-time progress

## File Locations

**Scripts:**
- `disk_searcher.py` - Main DISK search (command-line, no history saving)
- `test_disk_api.py` - DISK search via API (saves to search history + live tracking)
- `find_unprocessed_books.py` - Find books to index
- `build_chunk_index.py` - Build book↔chunk mapping

**Data:**
- `T:/faiss/disk_retrieval/chunks/` - 441 FAISS chunks (9.7TB)
- `D:/faiss/disk_retrieval/chunk_buffer/` - Local SSD buffer (rolling)
- `D:/faiss/disk_retrieval/book_to_chunks.json` - Book→chunks index
- `D:/faiss/disk_retrieval/unprocessed_books.txt` - Books to index

**Database:**
- `migrations/add_live_search_tracking.sql` - Schema updates for live search

## Key Code Changes

**disk_searcher.py:**
- Changed: `torch.hub.load()` → `KF.DISK.from_pretrained('depth')`
- Changed: `faiss.read_index(file)` → `faiss.read_index(file, faiss.IO_FLAG_MMAP)`
- Added: Chunk caching (skip copy if exists)
- Added: `_search_chunks_rolling_buffer()` with 5-chunk buffer
- Added: `search_id` and `progress_callback` parameters
- Added: `specific_chunks` parameter for targeted search

**db_helper.py:**
- Added: `create_search_session()` - Start live search
- Added: `update_search_progress()` - Update after each chunk
- Added: `complete_search_session()` - Mark search complete

## Usage Examples

### Search via CLI (no history)
```bash
# Search specific chunks from command line (results printed, NOT saved to DB)
python disk_searcher.py "D:\trivpics\2023-5.jpg" 140,141,142,143,144
```

### Search via API (saves to history + live tracking)
```bash
# Uses test_disk_api.py - hits /disk/search endpoint, saves results to search history
python test_disk_api.py

# Or use curl directly with chunk_ids parameter:
curl -X POST "http://localhost:8000/disk/search?top_k=10&chunk_ids=140,141,142,143,144&live_tracking=true" \
  -F "file=@D:\trivpics\2023-5.jpg"
```

**API endpoint**: `POST /disk/search`
- `chunk_ids` (optional): Comma-separated chunk numbers, e.g. `140,141,142,143,144`
- `top_k`: Number of results (default 50)
- `k`: Nearest neighbors per keypoint (default 5)
- `threshold`: Minimum similarity for voting (default 0.7)
- `live_tracking`: Enable live progress in DB (default true)
- Without `chunk_ids`, searches ALL 606 chunks (~8 days at current network speed)

### Find Which Chunks Contain a Book
```python
import json
with open('D:/faiss/disk_retrieval/book_to_chunks.json') as f:
    book_to_chunks = json.load(f)

chunks = book_to_chunks.get('Encyclopedia Of Monsters, The (ISBN 0816023034)', [])
print(f"Book is in chunks: {chunks}")
# Output: [142]
```

### Test Images
- **T-Rex dinosaur crop**: `D:\trivpics\2023-5.jpg` or `temp_uploads/ec4a14d7-9d4f-4c32-bdbe-72b9e3f0b058.jpg`
  - Target: Encyclopedia of Monsters, page 206
  - Expected: 145 votes (top result), found in chunk 142

## Success Metrics

✅ DISK model loads successfully
✅ Memory-mapped FAISS loading works
✅ Test search found correct result (145 votes)
✅ Rolling buffer strategy implemented
✅ Live search tracking ready (needs DB migration)
✅ Can identify 2,213 unprocessed books quickly

## Remaining Work

- [ ] Run database migration
- [ ] Update server.py DISK endpoint to use live tracking
- [ ] Test rolling buffer with multiple chunks
- [ ] Index remaining 2,213 books
- [ ] Upgrade to 10 Gigabit network
- [ ] Test full 441-chunk search
- [ ] Frontend UI for viewing active searches
