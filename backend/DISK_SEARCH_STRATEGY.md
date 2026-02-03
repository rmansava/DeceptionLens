# DISK Retrieval Index Strategy

## Overview

Two-phase approach for DISK keypoint retrieval across ~7000 books:
1. **Build Phase**: Per-book shards (crash resilient, never rebuild)
2. **Search Phase**: Consolidated chunks (fast copy, fast search)

## All Storage Locations

### Source Data
| Location | Contents | Size |
|----------|----------|------|
| `T:/disk-features/books/` | DISK feature .npz files (source) | ~2TB |
| `D:/books/pdf-images/` | Book page images (.jpg) | ~1.5TB |

### FAISS Indexes (Primary - Local SSD)
| Location | Contents | Size |
|----------|----------|------|
| `D:/faiss/disk_retrieval/books/` | Per-book FAISS indexes (~7000 books) | ~360GB |
| `D:/faiss/disk_retrieval/chunks/` | Consolidated search chunks | ~360GB |
| `D:/faiss/albums/` | Album cover indexes | ~2.3GB |
| `D:/faiss/board_games/` | Board game indexes | ~2.7GB |
| `D:/faiss/books/` | Book CLIP indexes | ~15GB |
| `D:/faiss/printads/` | Print ad indexes | ~2.7GB |

### Backup (NAS)
| Location | Contents | Notes |
|----------|----------|-------|
| `T:/faiss/disk_retrieval/books/` | Mirror of D:/faiss/disk_retrieval/books | Per-book shards backup |

### Temp/Buffer (Local SSD)
| Location | Purpose |
|----------|---------|
| `C:/temp/disk-retrieval-buffer/` | Batch copy buffer during build |
| `C:/temp/disk-retrieval-index/` | Index staging before NAS copy |
| `C:/temp/consolidate-buffer/` | Chunk consolidation temp |
| `C:/temp/disk-search/` | Search working directory |

## Directory Structure

```
D:/faiss/disk_retrieval/          # PRIMARY (local SSD - fast search)
├── books/                        # Per-book shards (~7000 folders, ~360GB)
│   ├── Book Name 1/
│   │   ├── index.faiss
│   │   └── paths.json
│   ├── Book Name 2/
│   │   └── ...
│   └── ...
│
└── chunks/                       # Consolidated for search (post-build)
    ├── chunk_001.faiss           # ~50GB each
    ├── chunk_001_paths.json
    ├── chunk_002.faiss
    └── ...                       # ~8 chunks total

T:/faiss/disk_retrieval/          # BACKUP (NAS)
└── books/                        # Mirror of D:/faiss/disk_retrieval/books
```

## Phase 1: Building (Current)

**Script**: `build_disk_retrieval_index.py`

**Strategy**: One FAISS IndexFlatIP per book
- Saves to local SSD instantly
- Background thread copies to NAS
- Auto-resumes from where it left off
- Crash = lose at most 1 book

**Why per-book**:
- No training needed (IndexFlatIP)
- Instant saves (~1 sec per book to local SSD)
- Never need to rebuild - just add new books
- Easy to fix/remove individual books

**Output**: `D:/faiss/disk_retrieval/books/{book_name}/` (with backup on T:)

## Phase 2: Consolidation (After Build)

**Script**: `consolidate_search_chunks.py` (to be created)

**Strategy**: Merge per-book indexes into ~50GB chunks (limited by 64GB RAM)

```python
# Pseudocode
chunk_size_gb = 50  # Must fit in RAM for searching
current_chunk = faiss.IndexFlatIP(128)
current_paths = []
chunk_num = 1

for book in all_books:
    # Load book index
    book_index = faiss.read_index(f"books/{book}/index.faiss")
    book_paths = json.load(f"books/{book}/paths.json")

    # Add to current chunk
    vectors = faiss.rev_swig_ptr(book_index.get_xb(), book_index.ntotal * 128)
    current_chunk.add(vectors.reshape(-1, 128))
    current_paths.extend(book_paths)

    # If chunk is big enough, save it
    if get_index_size(current_chunk) >= chunk_size_gb * 1024**3:
        faiss.write_index(current_chunk, f"chunks/chunk_{chunk_num:03d}.faiss")
        json.dump(current_paths, f"chunks/chunk_{chunk_num:03d}_paths.json")
        chunk_num += 1
        current_chunk = faiss.IndexFlatIP(128)
        current_paths = []
```

**Why consolidate**:
- Fewer large files copy faster than many small files
- Single index search is faster than 100 sequential searches
- ~35 chunks vs ~7000 book folders

## Phase 3: Searching

**Script**: `search_disk_chunks.py` (to be created)

**Strategy**: Copy one chunk to local SSD, search it, accumulate votes, repeat

```python
# Pseudocode
LOCAL_SEARCH = "C:/temp/disk-search"
all_votes = Counter()

for chunk_file in sorted(glob("T:/faiss/disk_retrieval/chunks/chunk_*.faiss")):
    # Copy chunk to local SSD
    shutil.copy(chunk_file, LOCAL_SEARCH)
    shutil.copy(chunk_file.replace('.faiss', '_paths.json'), LOCAL_SEARCH)

    # Search locally (fast!)
    index = faiss.read_index(f"{LOCAL_SEARCH}/chunk.faiss")
    paths = json.load(f"{LOCAL_SEARCH}/chunk_paths.json")

    distances, indices = index.search(query_descriptors, k=5)

    # Accumulate votes
    for i in range(len(query_descriptors)):
        for j in range(k):
            if indices[i][j] >= 0 and distances[i][j] >= threshold:
                all_votes[paths[indices[i][j]]] += 1

    # Clean up
    os.remove(f"{LOCAL_SEARCH}/chunk.faiss")
    os.remove(f"{LOCAL_SEARCH}/chunk_paths.json")

# Return top results
return all_votes.most_common(top_n)
```

**Speed comparison**:
- Current (NAS reads): ~37 sec/book = ~12 hours for 1200 books
- Batched chunks: ~5 min copy + ~10 sec search per 50GB chunk
- Estimated: ~35 chunks × 6 min = ~3.5 hours total

## File Size Estimates (Actual)

- Total books indexed: ~7,000
- Total keypoints: ~700 million
- Index size: ~360 GB (all per-book shards)
- paths.json: ~6 GB total
- **Total: ~360 GB**

Per 50GB chunk (~8 chunks total):
- ~87M keypoints per chunk
- Covers ~800-1000 books per chunk

## When to Re-consolidate

Re-run consolidation when:
- Many new books added
- Books removed/updated
- Want different chunk sizes

Per-book shards remain the source of truth. Chunks are derived/disposable.

## Alternative: Parallel Chunk Search

Could copy multiple chunks to different local paths and search in parallel:

```python
# Copy chunk 1 to C:/temp/search1/
# Copy chunk 2 to C:/temp/search2/
# Search both in parallel threads
# Merge votes at end
```

Limited by:
- Local SSD space
- NAS bandwidth for parallel copies
- Memory for multiple indexes

## Trivia Contest Batch Search

**Use case**: Search ~30 contest images at once to find their source pages.

**Script**: `search_trivia_batch.py` (to be created)

**Strategy**: Extract all query features upfront, search all queries against each chunk

```
queries/                      # Stage contest images here
├── 2023-1.jpg
├── 2023-2.jpg
├── ...
└── 2023-30.jpg
```

**Algorithm**:

```python
# 1. Extract DISK features for ALL query images (once)
query_descriptors = {}
for img_path in glob("queries/*.jpg"):
    query_descriptors[img_path] = extract_disk_features(img_path)
print(f"Extracted features for {len(query_descriptors)} query images")

# 2. Initialize per-image vote counters
query_votes = {img: Counter() for img in query_descriptors}

# 3. Process each chunk
for chunk_num, chunk_file in enumerate(sorted(glob("T:/chunks/chunk_*.faiss"))):
    print(f"Processing chunk {chunk_num + 1}...")

    # Copy chunk to local SSD
    shutil.copy(chunk_file, LOCAL_SEARCH)
    shutil.copy(chunk_file.replace('.faiss', '_paths.json'), LOCAL_SEARCH)

    # Load chunk (fast - local SSD)
    index = faiss.read_index(f"{LOCAL_SEARCH}/chunk.faiss")
    paths = json.load(open(f"{LOCAL_SEARCH}/chunk_paths.json"))

    # Search ALL queries against this chunk
    for img_path, descriptors in query_descriptors.items():
        distances, indices = index.search(descriptors, k=5)

        for i in range(len(descriptors)):
            for j in range(5):
                if indices[i][j] >= 0 and distances[i][j] >= 0.7:
                    query_votes[img_path][paths[indices[i][j]]] += 1

    # Clean up chunk
    os.remove(f"{LOCAL_SEARCH}/chunk.faiss")
    os.remove(f"{LOCAL_SEARCH}/chunk_paths.json")

# 4. Output results for each query image
for img_path in sorted(query_descriptors.keys()):
    print(f"\n{img_path}:")
    for match_path, votes in query_votes[img_path].most_common(5):
        print(f"  {votes:4d} votes: {match_path}")
```

**Benefits**:
- Feature extraction: 30 images × ~2 sec = ~1 minute (done once)
- Copy each chunk: ~5 min × 35 chunks = ~3 hours total
- Search 30 queries per chunk: ~10 sec × 35 = ~6 minutes total
- **Total: ~3.5 hours for all 30 images** (vs 30 × 12 hours = 360 hours individually)

**Output format**:

```
queries/2023-1.jpg:
   662 votes: D:/books/pdf-images/Encyclopedia Of Monsters/page210.jpg
    45 votes: D:/books/pdf-images/Some Other Book/page12.jpg
    ...

queries/2023-2.jpg:
   891 votes: D:/books/pdf-images/Movie Monsters Encyclopedia/page88.jpg
    ...
```

**Contest workflow**:
1. Get contest images, put in `queries/` folder
2. Run `search_trivia_batch.py`
3. Wait ~3.5 hours
4. Get ranked results for each image
5. Verify top matches visually

## Implementation Order

1. [x] Per-book build (running now)
2. [ ] Wait for build to complete
3. [ ] Create consolidation script
4. [ ] Run consolidation
5. [ ] Create batch search script (`search_trivia_batch.py`)
6. [ ] Test with sample contest images
7. [ ] Optional: parallel chunk processing
