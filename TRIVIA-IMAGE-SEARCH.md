# Trivia Image Search System

## The Problem

In trivia contests, participants receive **image snippets** - cropped portions of larger source images - and must identify the original source to answer questions about it.

### Challenge
- **Query**: A small crop (e.g., a cartoon dinosaur character)
- **Target**: The full page/image it came from (e.g., page 210 of "Encyclopedia of Monsters")
- **Index**: 3+ million images from books, magazines, albums, ads, etc.

### Media Types
| Type | Examples | Characteristics |
|------|----------|-----------------|
| Book pages | Encyclopedia entries, textbook diagrams | Text + images, consistent layouts |
| Print ads | Magazine advertisements, posters | Logos, products, stylized text |
| Album covers | Music albums, compilation art | Artistic, iconic imagery |
| Screenshots | TV shows, movies, games | Frames, UI elements |
| Artwork | Paintings, illustrations, cartoons | Stylized, varying quality |

---

## Technology Stack

### Indexing Methods

We use multiple embedding models to capture different aspects of images:

#### 1. DINOv2 (Visual Similarity)
- **Model**: facebook/dinov2-large
- **Dimension**: 1024
- **Storage**: OpenSearch vector index
- **Strength**: Finding visually similar images (color, texture, composition)
- **Weakness**: Struggles when crop is small portion of page

```
Query: [crop of dinosaur]
DINOv2 sees: "small colorful illustration"
Page has: "full page with text + small dinosaur in corner"
Result: Poor match (global embeddings don't align)
```

#### 2. CLIP (Semantic Similarity)
- **Model**: ViT-L/14
- **Dimension**: 768
- **Storage**: FAISS index
- **Strength**: Understanding "what" is in the image semantically
- **Weakness**: May match conceptually similar but wrong images

``` 
Query: [crop of dinosaur]
CLIP sees: "dinosaur character, cartoon style"
Page has: "encyclopedia page about dinosaur"
Result: Good match (semantic concept survives cropping)
```

#### 3. Face Embeddings
- **Model**: InsightFace (buffalo_l)
- **Storage**: OpenSearch vector index
- **Strength**: Finding specific people across different photos
- **Use case**: Actor headshots, celebrity images

---

## Search Strategies

### Strategy 1: Raw DINOv2 Search
**Best for**: Large crops that represent significant portion of source image

```
Upload image → DINOv2 embedding → OpenSearch KNN → Results
```

- Fast (~1 second)
- Works when query visually resembles full page
- Fails for small semantic crops

### Strategy 2: DINOv2 + Geometric Verification
**Best for**: When you need to confirm visual matches

```
Upload image → DINOv2 embedding → OpenSearch KNN → LightGlue verification → Re-ranked results
```

- Verifies matches using keypoint correspondence
- Filters false positives
- Slower but more precise

### Strategy 3: CLIP Search
**Best for**: Semantic queries, text descriptions, conceptual matching

```
Upload image → CLIP embedding → FAISS search → Results
```

- Understands "what" the image is about
- Works for cropped snippets
- May return conceptually similar but wrong matches

### Strategy 4: CLIP + ORB + Template Matching (Re-ranking)
**Best for**: Small crops that need precise pixel-level verification

```
Upload image → CLIP (20K candidates) → ORB keypoint filter → Template matching (top 1K) → Combined scoring → Results
```

**Pipeline Details:**

| Stage | Purpose | Output |
|-------|---------|--------|
| CLIP retrieval | Semantic candidates | 20,000 results |
| ORB keypoints | Geometric filtering | Filter blanks, rank by matches |
| Template matching | Pixel-level verification | Score 0-1 for exact match |
| Combined scoring | Hierarchical ranking | Final ranked list |

**Scoring Tiers:**
- Template >= 0.9: Base 1000 (dominant signal)
- Template >= 0.85: Base 500 (strong match)
- Template >= 0.75: Base 200 (good match)
- Keypoints >= 15: Base 150 (geometric evidence)
- Keypoints >= 10: Base 100 (moderate evidence)
- Otherwise: Balanced fallback using all signals

### Strategy 5: Deep Search (Parallel CLIP + DINOv2)
**Best for**: Unknown query type, hedging bets

```
                    ┌─→ CLIP (20K) ──────┐
Upload image ───────┤                    ├─→ Merge unique → ORB + Template → Results
                    └─→ DINOv2 (20K) ────┘
```

- Runs both retrieval methods in parallel
- Merges unique candidates
- Applies verification pipeline
- Slowest but most comprehensive

---

## Performance Comparison

### Test Case 1: Cropped Dinosaur → Encyclopedia of Monsters page 210

| Method | Target Rank | Notes |
|--------|-------------|-------|
| Raw DINOv2 | Not in top 10,000 | Visual embedding fails for small crop |
| Raw CLIP | Not in top 100 | Semantic match buried by noise |
| CLIP + Rerank | **#19** | Template matching finds exact crop |

### Test Case 2: King Koko Dog → Ad boy Vintage advertising page 98

Query: Grayscale crop of "King Koko" dog mascot (Puppy Palace hot dogs)
Source: `D:\books\pdf-images\Ad boy Vintage advertising with character - Warren Dotz\...-page98.jpg`

| Method | Target Rank | Score | Verified Matches |
|--------|-------------|-------|------------------|
| Raw DINOv2 | **#70** | 0.7631 | N/A |
| DINOv2 + LightGlue | **#1** | 0.7631 | **1920** |

**Key Finding**: Geometric verification (DISK + LightGlue) dramatically improves ranking:
- Page 140 had higher DINOv2 score (0.7753) but only 8 keypoint matches
- Page 98 had lower DINOv2 score but 1920 keypoint matches → correct source
- Sorting by `verified_matches` moves correct result from #70 to #1

### When to Use What

| Query Type | Recommended Strategy |
|------------|---------------------|
| Large visual crop (>50% of page) | DINOv2 |
| Small semantic crop | CLIP + Rerank |
| Face/person | Face Search |
| Unknown/mixed | Deep Search |
| Text description | CLIP Text Search |

---

## Technical Implementation

### Index Sizes
- DINOv2/OpenSearch: ~3 million images
- CLIP/FAISS: ~3 million images
- Face embeddings: Extracted from indexed images

### Re-ranking Parameters
```python
retrieval_k = 20000   # CLIP candidates to retrieve
rerank_k = 1000       # Candidates for template matching
orb_features = 500    # ORB keypoints per image
distance_threshold = 50  # Good match threshold
template_scales = [0.25, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
```

### Parallel Processing
- ORB matching: 12 workers
- Template matching: 8 workers
- Typical re-rank time: 30-60 seconds for 20K candidates

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Blazor)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ DINOv2   │  │  CLIP    │  │  Face    │  │  Deep Search     │ │
│  │ Search   │  │  Search  │  │  Search  │  │  (CLIP+DINOv2)   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└───────┼─────────────┼─────────────┼─────────────────┼───────────┘
        │             │             │                 │
        ▼             ▼             ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ /search      │  │ /clip/search │  │ /search/deep           │ │
│  │ (DINOv2)     │  │ (CLIP+Rerank)│  │ (Parallel+Verify)      │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘ │
└─────────┼─────────────────┼──────────────────────┼──────────────┘
          │                 │                      │
          ▼                 ▼                      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│   OpenSearch    │  │     FAISS       │  │   Re-ranking        │
│   (DINOv2 KNN)  │  │   (CLIP KNN)    │  │   (ORB+Template)    │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
```

---

## Known Problems & Challenges

### Problem 1: Small Crops in Large Pages
**Symptom**: DINOv2 can't find a cropped snippet even though the source page is indexed.

**Example**: Cropped dinosaur illustration (Manglosaurus) from Encyclopedia of Monsters page 210.
- DINOv2 result: Not found in top 10,000
- CLIP + Rerank result: Found at rank #19

**Why**: DINOv2 creates a single global embedding for the entire image. A small crop's embedding doesn't match the full page's embedding because the page contains much more visual information.

**Solution**: Use CLIP for semantic retrieval + template matching for verification.

---

### Problem 2: Semantic vs Visual Mismatch
**Symptom**: CLIP returns conceptually similar but wrong images.

**Example**: Query for "red car" might return any red car, not THE specific red car from your source.

**Why**: CLIP understands concepts, not specific instances. Two different dinosaur illustrations are "semantically" similar.

**Solution**: Template matching re-ranking verifies pixel-level matches after semantic retrieval.

---

### Problem 3: Re-ranking Speed
**Symptom**: Deep search takes 30-60 seconds.

**Why**:
- Retrieving 20K+ candidates from each index
- ORB keypoint extraction on thousands of images
- Multi-scale template matching is CPU-intensive

**Tradeoffs**:
- Fewer candidates = faster but might miss target
- Fewer template scales = faster but less robust to size variations
- Current balance: 20K retrieval, 1K for template matching

---

### Problem 4: Blank/Low-Contrast Pages
**Symptom**: Mostly-white or mostly-black pages rank highly due to low variance.

**Why**: Template matching on uniform regions can produce false high scores.

**Solution**: Blank page detection (`np.std(img) < 30`) filters these before ranking.

---

### Problem 5: Scale Variance
**Symptom**: Crop might be displayed at different size than in source.

**Example**: A logo that's 100x100 in query but 50x50 in source page.

**Why**: Direct template matching fails if sizes don't match.

**Solution**: Multi-scale template matching tries 8 different scales (0.25x to 2.0x).

---

### Problem 6: Contrast/Lighting Differences
**Symptom**: Same image with different brightness/contrast doesn't match.

**Example**: Scanned page vs digital original, or photo of a page.

**Why**: Raw pixel comparison is sensitive to intensity variations.

**Solution**: Histogram equalization normalizes contrast before template matching.

---

### Problem 7: DINOv2 + CLIP Index Mismatch
**Symptom**: Deep search merges results but paths might not align perfectly.

**Why**: Indexes might be built at different times or from different source directories.

**Current state**: Both indexes should cover same images, but edge cases may exist.

---

### Problem 8: Memory/GPU Constraints
**Symptom**: Can't load all 3M embeddings into GPU memory.

**Solution**:
- FAISS index on CPU with GPU for encoding only
- OpenSearch handles DINOv2 index server-side
- Re-ranking done on CPU with parallel workers

---

### Problem 9: cv2.imread Fails on Windows with Non-ASCII Paths
**Symptom**: Geometric verification shows 0 verified matches for ALL results, even when files exist.

**Example**:
```
Path: D:\books\pdf-images\...\Anna's Archive\...-page100.jpg
os.path.exists(): True
cv2.imread(): None (FAILED)
cv2.imdecode(): Success
```

**Why**: OpenCV's `cv2.imread()` on Windows cannot handle paths with special characters like curly apostrophes (`'`), certain Unicode characters, or non-ASCII filenames. This is a known OpenCV bug.

**Solution**: Replace all `cv2.imread(path)` calls with:
```python
with open(path, 'rb') as f:
    data = f.read()
img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
```

**Fixed in**: `searcher.py` - `_load_torch_image()`, `get_face_embedding()`, `generate_visualization_image()`

---

## Future Improvements

1. **Hybrid retrieval**: Single query that intelligently routes to best method
2. **Query classification**: Auto-detect if crop is "semantic" vs "visual"
3. **Ensemble scoring**: Weight multiple methods based on confidence
4. **Incremental indexing**: Add new media without full re-index
5. **OCR integration**: Search by text visible in images
