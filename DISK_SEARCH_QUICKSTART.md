# DISK Search - Quick Start Guide

## Setup (One-Time)

### 1. Run Database Migration
```bash
cd backend
setup_disk_search.bat
```

This adds the required columns for live tracking:
- `Status` - 'in_progress', 'completed', 'failed'
- `CurrentProgress` - "Searching chunk 5/441"
- `TotalChunks` - For progress calculation

### 2. Start the Services

**Backend:**
```bash
cd backend
python server.py
```
Backend runs at: `http://localhost:8000`

**Frontend:**
```bash
cd web
dotnet run
```
Frontend runs at: `http://localhost:5000`

## Usage

### Option 1: Web UI

1. Go to `http://localhost:5000`
2. Upload an image (e.g., cropped screenshot)
3. Select "DISK Keypoint" search
4. Watch live progress as chunks are searched!

**Live Progress Shows:**
- Current chunk being searched (e.g., "Searching chunk 142/441")
- Top 100 results updating in real-time
- Aggregated votes from all chunks searched so far
- ETA for completion

### Option 2: API

```bash
curl -X POST "http://localhost:8000/disk/search?top_k=10&live_tracking=true" \
  -F "file=@D:\trivpics\2023-5.jpg"
```

**Parameters:**
- `top_k`: Number of results (1-500, default: 50)
- `k`: Neighbors per keypoint (1-20, default: 5)
- `threshold`: Min similarity (0.0-1.0, default: 0.7)
- `live_tracking`: Enable progress (true/false, default: true)

### Option 3: Command Line

**Search specific chunks:**
```bash
cd backend
python disk_searcher.py "D:\trivpics\2023-5.jpg" 142
```

**Search all chunks:**
```bash
python disk_searcher.py "D:\trivpics\2023-5.jpg"
```

**Find which chunks contain a book:**
```bash
python -c "import json; data = json.load(open('D:/faiss/disk_retrieval/book_to_chunks.json')); print(data['Encyclopedia Of Monsters, The (ISBN 0816023034)'])"
```

## Test the Setup

```bash
cd backend
python test_disk_api.py
```

This will:
1. Check if server is running
2. Send test search (dinosaur image)
3. Verify results come back
4. Confirm search is in history

### Verified Test: Dino Image vs Chunk 183

The canonical test is searching `D:\trivpics\2023-5.jpg` (a cropped dinosaur from Encyclopedia of Monsters page 206) against the chunk that contains that book:

```bash
curl -X POST "http://localhost:8000/disk/search?chunk_ids=183&collections=books&top_k=20" -F "file=@D:/trivpics/2023-5.jpg"
```

**Expected result:** Encyclopedia of Monsters page 206 is #1 with ~157 votes, massively ahead of #2 at ~6 votes.

**What went wrong before (Feb 2026):** All chunk build scripts had `MAX_IMAGE_DIM = 1600`, which downscaled source images before DISK feature extraction. Since the query side extracts at full resolution, the descriptors didn't match well -- votes were low and results were unreliable. The per-book FAISS shards on T: and S: were built by an older pipeline at full resolution and were fine. The bug only affected the consolidated 10GB chunks.

**The fix:** Removed the 1600px resize cap from all build scripts. Books are rebuilt via `consolidate_search_chunks.py` which reads the original full-resolution per-book shards (no re-extraction needed). Other categories (board_games, print_ads, cereal, albums, comics) use `MAX_IMAGE_DIM = 4096` to cap only outlier-huge images while preserving full quality for the vast majority. All old bad chunks and SQL data were deleted and rebuilt from scratch.

**How to find which chunk a book is in:**
```bash
python -c "
import json, numpy as np, os
from glob import glob
ids_dir = 'D:/faiss/disk_retrieval/chunk_ids'
paths = json.load(open(os.path.join(ids_dir, 'path_lookup.json')))
ids = [i for i, p in enumerate(paths) if 'Encyclopedia Of Monsters' in p]
print(f'Compact IDs: {ids[0]}-{ids[-1]} ({len(ids)} images)')
for f in sorted(glob(os.path.join(ids_dir, 'chunk_*_ids.npy'))):
    if ids[0] in np.load(f):
        print(f'Found in {os.path.basename(f).replace(\"_ids.npy\",\"\")}')
        break
"
```

## Performance

**Current (14 MB/s network):**
- Per chunk: ~27 minutes (25.7 min copy + 1.4 min search)
- Full 441 chunks: ~198 hours (8.3 days)

**With 10 Gigabit Ethernet:**
- Per chunk: ~100 seconds (22s copy + 78s search)
- Full 441 chunks: ~12.3 hours
- With rolling buffer: ~5-6 hours

## Index Status

Run this to check your index:
```bash
python -c "from glob import glob; chunks = glob('T:/faiss/disk_retrieval/chunks/chunk_*.faiss'); print(f'Chunks: {len(chunks)}'); print(f'Size: ~{len(chunks) * 22}GB')"
```

**Current stats:**
- 441 chunks indexed
- 4,735 books (69.4% coverage)
- ~19.6 billion DISK vectors
- ~9.7 TB total index size

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill the process if needed
taskkill /PID <pid> /F
```

### Database connection fails
```bash
# Test SQL Server connection
sqlcmd -S localhost -d ImageSearch -Q "SELECT @@VERSION"
```

### Chunks not found
```bash
# Check NAS connection
dir T:\faiss\disk_retrieval\chunks\chunk_*.faiss
```

### Search is slow
- Check network speed to NAS (should be 10GbE for good performance)
- Verify chunks are on T: drive (NAS), not D: drive
- Check if rolling buffer is enabled (default: yes)

## What's Next

### Index Remaining Books
```bash
# Find unprocessed books
python find_unprocessed_books.py

# Start indexing them
consolidate_chunks_from_list.bat
```

This will process the remaining 2,213 books (~31% remaining).

### Upgrade Network
Installing 10 Gigabit Ethernet will reduce full-corpus search time from 8+ days to 5-6 hours.

### View Search History
Go to `http://localhost:5000/history` to see all searches including:
- Active searches with live progress
- Completed searches with results
- Search duration and top results
- Query images

## Files

**Scripts:**
- `setup_disk_search.bat` - One-time database setup
- `test_disk_api.py` - Test the API endpoint
- `disk_searcher.py` - Command-line search
- `find_unprocessed_books.py` - Find books to index
- `consolidate_chunks_from_list.bat` - Index remaining books

**Data:**
- `T:/faiss/disk_retrieval/chunks/` - 441 FAISS chunks (9.7TB)
- `D:/faiss/disk_retrieval/chunk_buffer/` - Local buffer for rolling copy
- `D:/faiss/disk_retrieval/book_to_chunks.json` - Book→chunk mapping
- `D:/faiss/disk_retrieval/consolidation_state.json` - Processing state

**Database:**
- `migrations/add_live_search_tracking.sql` - Schema migration

## Support

For issues or questions, check:
- `backend/DISK_SEARCH_STATUS.md` - Detailed implementation notes
- `backend/DISK_SEARCH_STRATEGY.md` - Technical details
- Git commit history for recent changes
