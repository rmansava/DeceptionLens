"""
Deception Lens API Server
FastAPI backend for the web application.
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import shutil
import os
import uuid
import logging
import time
from typing import List, Optional
from contextlib import asynccontextmanager
import re

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup: Ensure DB indexes exist
    from db_helper import ensure_indexes
    logger.info("Ensuring DB indexes...")
    ensure_indexes()

    # Startup: Initialize DISK search queue
    from disk_queue import initialize_disk_queue
    logger.info("Initializing DISK search queue...")
    await initialize_disk_queue()
    logger.info("DISK search queue initialized")

    yield

    # Shutdown: Stop the queue
    from disk_queue import get_disk_queue
    queue = get_disk_queue()
    await queue.stop()
    logger.info("DISK search queue stopped")


app = FastAPI(
    title="Deception Lens API",
    description="Visual similarity search using CLIP and DINOv2",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Blazor frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Searchers (lazy loading)
searcher = None
clip_searcher = None
opensearch_visual_searcher = None
DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
CLIP_INDEX_PATH = os.environ.get("CLIP_INDEX_PATH", "D:/faiss/books/index.faiss")
CLIP_PATHS_PATH = os.environ.get("CLIP_PATHS_PATH", "D:/faiss/books/paths.json")
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global progress tracking for long-running searches
search_progress = {
    "stage": "idle",  # idle, searching, loading_cache, verifying, complete
    "message": "",
    "current": 0,
    "total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "rate": 0.0,
    "eta_seconds": 0
}

# Collections that use OpenSearch for visual search
OPENSEARCH_VISUAL_COLLECTIONS = {"books", "print_ads", "board_games"}


def get_searcher():
    """Lazy-load the DINOv2 searcher (used for geometric verification with DISK+LightGlue)."""
    global searcher
    if searcher is None:
        from searcher import DinoSearcher
        logger.info(f"Initializing DINOv2 searcher for geometric verification")
        searcher = DinoSearcher(db_path=DB_PATH)
    return searcher


def get_opensearch_visual_searcher():
    """Lazy-load the OpenSearch visual searcher (for books collection)."""
    global opensearch_visual_searcher
    if opensearch_visual_searcher is None:
        from opensearch_searcher import OpenSearchSearcher
        logger.info("Initializing OpenSearch visual searcher")
        opensearch_visual_searcher = OpenSearchSearcher(visual_index="dinov2-books")
    return opensearch_visual_searcher


def get_clip_searcher(collection: str = "books"):
    """Get a CLIP searcher for a specific collection."""
    from clip_searcher import get_cached_clip_searcher
    logger.info(f"Getting CLIP searcher for collection: {collection}")
    return get_cached_clip_searcher(collection)


def save_search_to_history(
    search_type: str,
    results: list,
    search_duration_ms: int,
    query_image: bytes = None,
    query_image_name: str = None,
    query_text: str = None,
    collection: str = None
):
    """Save a search to history (called in background)."""
    try:
        from db_helper import save_search_history
        save_search_history(
            search_type=search_type,
            query_image=query_image,
            query_image_name=query_image_name,
            query_text=query_text,
            results=results,
            search_duration_ms=search_duration_ms,
            collection=collection
        )
        logger.info(f"Saved {search_type} search to history ({len(results)} results)")
    except Exception as e:
        logger.error(f"Failed to save search to history: {e}")


class SearchResult(BaseModel):
    path: str
    score: float
    verified_matches: int
    metadata: dict


class StatsResponse(BaseModel):
    visual_count: int
    face_count: int


class HealthResponse(BaseModel):
    status: str
    searcher_loaded: bool
    db_path: str
    lightglue_ready: bool = False
    disk_cache_ready: bool = False


class SearchProgressResponse(BaseModel):
    stage: str
    message: str
    current: int
    total: int
    cache_hits: int
    cache_misses: int
    rate: float
    eta_seconds: int


class DiskSearchStartResponse(BaseModel):
    search_id: int
    status: str
    queue_position: int
    total_chunks: int
    message: str


def _parse_progress_chunk(progress_text: Optional[str]) -> Optional[int]:
    """Parse 'Searching chunk X/Y' and return X."""
    if not progress_text:
        return None
    m = re.search(r"(\d+)\s*/\s*(\d+)", progress_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is healthy."""
    lightglue_ready = False
    disk_cache_ready = False
    if searcher is not None:
        lightglue_ready = searcher.extractor is not None and searcher.matcher is not None
        disk_cache_ready = searcher.disk_file_cache is not None or searcher.disk_cache is not None
    return HealthResponse(
        status="ok",
        searcher_loaded=searcher is not None,
        db_path=DB_PATH,
        lightglue_ready=lightglue_ready,
        disk_cache_ready=disk_cache_ready
    )


@app.get("/search/progress", response_model=SearchProgressResponse)
def get_search_progress():
    """Get current search progress for long-running verification."""
    return SearchProgressResponse(**search_progress)


@app.get("/stats", response_model=StatsResponse)
def get_stats(collection: str = "books"):
    """Get statistics for a collection."""
    try:
        # "all" doesn't have stats - just return zeros
        if collection == "all":
            return StatsResponse(visual_count=0, face_count=0)

        if collection in OPENSEARCH_VISUAL_COLLECTIONS:
            # Use OpenSearch for stats
            os_searcher = get_opensearch_visual_searcher()
            counts = os_searcher.get_counts(collection)
            return StatsResponse(
                visual_count=counts.get("visual", 0),
                face_count=counts.get("faces", 0)
            )
        else:
            # Return zeros for unknown collections
            return StatsResponse(visual_count=0, face_count=0)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=500),
    collection: str = Query(default="books"),
    verify: bool = Query(default=False),
    background_tasks: BackgroundTasks = None
):
    """
    Search for similar images.

    - **file**: Query image to search for
    - **top_k**: Number of results to return (1-500)
    - **collection**: Collection name to search in (use CLIP search for "all" collections)
    - **verify**: Whether to perform geometric verification
    """
    # DINOv2/OpenSearch doesn't support "all" - use CLIP search-all for that
    if collection == "all":
        raise HTTPException(
            status_code=400,
            detail="DINOv2 search doesn't support 'all' collections. Use CLIP search for multi-collection search, or select a specific collection."
        )

    start_time = time.time()

    # Use OpenSearch for specified collections (more robust)
    use_opensearch = collection in OPENSEARCH_VISUAL_COLLECTIONS

    # Read image bytes for history before consuming file
    image_bytes = await file.read()
    image_filename = file.filename

    # Save uploaded file temporarily
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(image_bytes)

        logger.info(f"Searching with query: {temp_path} (OpenSearch: {use_opensearch})")

        # Progress callback for updating global state
        def update_progress(stage, message, current, total, cache_hits, cache_misses, rate, eta):
            global search_progress
            search_progress = {
                "stage": stage,
                "message": message,
                "current": current,
                "total": total,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "rate": round(rate, 1),
                "eta_seconds": eta
            }

        if use_opensearch:
            # Use OpenSearch for visual search
            os_searcher = get_opensearch_visual_searcher()
            if verify:
                # Reset progress
                update_progress("searching", f"Searching {collection} with DINOv2...", 0, 0, 0, 0, 0, 0)

                # When verifying, fetch MANY more candidates from OpenSearch
                # Crops have low global similarity but high keypoint matches
                # Need large pool for LightGlue to find the right page
                fetch_k = 5000
                matches = os_searcher.search(temp_path, top_k=fetch_k, collection=collection)

                update_progress("searching", f"Found {len(matches)} candidates", len(matches), len(matches), 0, 0, 0, 0)

                # Show progress during model loading (can take a while on first search)
                update_progress("loading_models", "Loading DINOv2 + DISK + LightGlue models...", 0, len(matches), 0, 0, 0, 0)

                # Use DinoSearcher's geometric verification with progress callback
                # require_verification=True ensures search fails if LightGlue is not available
                dino_searcher = get_searcher()
                matches = dino_searcher._verify_matches(
                    temp_path, matches,
                    progress_callback=update_progress,
                    require_verification=True  # Fail loudly if verification not available
                )
                matches.sort(key=lambda x: (x['verified_matches'], x['score']), reverse=True)
                matches = matches[:top_k]

                # Reset progress to idle
                update_progress("idle", "", 0, 0, 0, 0, 0, 0)
            else:
                matches = os_searcher.search(temp_path, top_k=top_k, collection=collection)
        else:
            # Collection not in OpenSearch - return error
            s = get_searcher()
            if s is None:
                raise HTTPException(status_code=503, detail="Searcher not initialized")
            matches = s.search(
                temp_path,
                top_k=top_k,
                verify=verify,
                collection_name=collection
            )

        results = []
        for m in matches:
            results.append(SearchResult(
                path=m['path'],
                score=m['score'],
                verified_matches=m.get('verified_matches', 0),
                metadata=m.get('metadata', {})
            ))

        # Save to history in background
        duration_ms = int((time.time() - start_time) * 1000)
        history_results = [{'path': m['path'], 'score': m['score'], 'verified_matches': m.get('verified_matches', 0)} for m in matches]
        if background_tasks:
            background_tasks.add_task(
                save_search_to_history,
                search_type="DINOv2 Visual",
                results=history_results,
                search_duration_ms=duration_ms,
                query_image=image_bytes,
                query_image_name=image_filename,
                collection=collection
            )

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/search/bytes", response_model=List[SearchResult])
async def search_image_bytes(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=500),
    collection: str = Query(default="books"),
    verify: bool = Query(default=False)
):
    """
    Search using image bytes directly (alternative endpoint).
    """
    s = get_searcher()
    if s is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    try:
        image_bytes = await file.read()
        logger.info(f"Searching with {len(image_bytes)} bytes")

        matches = s.search_by_bytes(
            image_bytes,
            top_k=top_k,
            verify=verify,
            collection_name=collection
        )

        results = []
        for m in matches:
            results.append(SearchResult(
                path=m['path'],
                score=m['score'],
                verified_matches=m.get('verified_matches', 0),
                metadata=m.get('metadata', {})
            ))

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image")
def get_image(path: str = Query(..., description="Absolute path to the image")):
    """
    Serve an image from the local filesystem.

    WARNING: This allows reading any file accessible to the process.
    Only use in local/trusted environments.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path)


@app.get("/collections")
def list_collections():
    """List all available collections."""
    try:
        # Return all collections from config
        from collections_config import COLLECTIONS
        return {
            "collections": list(COLLECTIONS.keys())
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    """Delete a collection from OpenSearch (NOTE: deletes all documents in the index)."""
    # NOTE: This is a destructive operation - disabled for safety
    # OpenSearch indices should be managed separately
    raise HTTPException(
        status_code=501,
        detail="Collection deletion via API is disabled. Use OpenSearch directly to manage indices."
    )


# ============== Face Search Endpoints ==============

@app.post("/search/faces", response_model=List[SearchResult])
async def search_faces(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=500),
    min_score: float = Query(default=0.0, ge=0.0, description="Minimum face similarity score"),
    collection: str = Query(default="books"),
    background_tasks: BackgroundTasks = None
):
    """
    Search for similar faces using InsightFace.

    - **file**: Query image containing face(s)
    - **top_k**: Number of results to return (1-500)
    - **min_score**: Minimum score to keep a match
    - **collection**: Collection name to search in (select a specific collection)
    """
    # Face search doesn't support "all" collections
    if collection == "all":
        raise HTTPException(
            status_code=400,
            detail="Face search doesn't support 'all' collections. Please select a specific collection."
        )

    start_time = time.time()

    # Face search currently supported for collections with OpenSearch face indexes.
    use_opensearch = collection in OPENSEARCH_VISUAL_COLLECTIONS

    try:
        image_bytes = await file.read()
        image_filename = file.filename
        logger.info(
            f"Face searching with {len(image_bytes)} bytes "
            f"(OpenSearch: {use_opensearch}, min_score={min_score}, collection={collection})"
        )

        if not use_opensearch:
            raise HTTPException(
                status_code=400,
                detail=f"Face search is only available for: {', '.join(sorted(OPENSEARCH_VISUAL_COLLECTIONS))}."
            )

        os_searcher = get_opensearch_visual_searcher()
        matches = os_searcher.search_faces_by_bytes(
            image_bytes,
            top_k=top_k,
            collection=collection,
            min_score=min_score
        )

        if not matches:
            logger.info("No faces detected or no matches found")

        results = []
        for m in matches:
            results.append(SearchResult(
                path=m['path'],
                score=m['score'],
                verified_matches=m.get('verified_matches', 0),
                metadata=m.get('metadata', {})
            ))

        # Save to history in background
        duration_ms = int((time.time() - start_time) * 1000)
        history_results = [{'path': m['path'], 'score': m['score']} for m in matches]
        if background_tasks:
            background_tasks.add_task(
                save_search_to_history,
                search_type="Face Search",
                results=history_results,
                search_duration_ms=duration_ms,
                query_image=image_bytes,
                query_image_name=image_filename,
                collection=collection
            )

        return results

    except Exception as e:
        logger.error(f"Face search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Visualization Endpoints ==============

@app.post("/visualize")
async def visualize_match(
    file: UploadFile = File(...),
    match_path: str = Query(..., description="Path to the matched image")
):
    """
    Generate a visualization showing where the query matches on the result image.

    - **file**: Query image
    - **match_path**: Path to the matched result image

    Returns: PNG image with the matched region highlighted
    """
    s = get_searcher()
    if s is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    try:
        query_bytes = await file.read()
        logger.info(f"Generating visualization for match: {match_path}")

        vis_bytes = s.generate_visualization_image(query_bytes, match_path)

        if vis_bytes is None:
            raise HTTPException(status_code=404, detail="Could not generate visualization")

        return Response(content=vis_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== CLIP Search Endpoints ==============

class ClipStatsResponse(BaseModel):
    total_images: int
    model: str
    index_path: str


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 50


@app.get("/clip/collections")
def list_clip_collections():
    """List all available CLIP collections."""
    try:
        from clip_searcher import list_clip_collections as list_collections
        collections = list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to list CLIP collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clip/stats", response_model=ClipStatsResponse)
def get_clip_stats(collection: str = Query(default="books", description="Collection name")):
    """Get CLIP index statistics for a collection."""
    try:
        cs = get_clip_searcher(collection)
        stats = cs.get_stats()
        return ClipStatsResponse(
            total_images=stats["total_images"],
            model=stats["model"],
            index_path=stats["index_path"]
        )
    except Exception as e:
        logger.error(f"Failed to get CLIP stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clip/search", response_model=List[SearchResult])
async def clip_search_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=500),
    collection: str = Query(default="books", description="Collection to search (books, print_ads)"),
    rerank: bool = Query(default=False, description="Apply ORB + Template matching re-ranking"),
    retrieval_k: int = Query(default=20000, ge=100, le=50000, description="CLIP candidates to retrieve (when rerank=true)"),
    rerank_k: int = Query(default=1000, ge=100, le=5000, description="Candidates for template matching (when rerank=true)"),
    background_tasks: BackgroundTasks = None
):
    """
    Search for similar images using CLIP.

    - **file**: Query image to search for
    - **top_k**: Number of results to return (1-500)
    - **collection**: Collection to search (books, print_ads)
    - **rerank**: If true, apply ORB keypoint + Template matching re-ranking (slower but more accurate for crops)
    - **retrieval_k**: Number of CLIP candidates to retrieve for re-ranking
    - **rerank_k**: Number of candidates to run template matching on
    """
    start_time = time.time()

    try:
        cs = get_clip_searcher(collection)
        image_bytes = await file.read()
        image_filename = file.filename

        if rerank:
            logger.info(f"CLIP search with re-ranking: {len(image_bytes)} bytes, retrieval_k={retrieval_k}, rerank_k={rerank_k}")
            matches = cs.search_by_image_bytes_with_rerank(
                image_bytes,
                top_k=top_k,
                retrieval_k=retrieval_k,
                rerank_k=rerank_k,
                verbose=True
            )
        else:
            logger.info(f"CLIP searching with {len(image_bytes)} bytes")
            matches = cs.search_by_image_bytes(image_bytes, top_k=top_k)

        results = []
        for m in matches:
            # Use combined_score if available (reranked), otherwise use CLIP score
            score = m.get('combined_score', m['score'])
            results.append(SearchResult(
                path=m['path'],
                score=score,
                verified_matches=m.get('keypoint_matches', m.get('verified_matches', 0)),
                metadata=m.get('metadata', {})
            ))

        # Save to history in background
        duration_ms = int((time.time() - start_time) * 1000)
        history_results = [{'path': m['path'], 'score': m['score'],
                           'keypoint_matches': m.get('keypoint_matches', 0)} for m in matches]
        if background_tasks:
            background_tasks.add_task(
                save_search_to_history,
                search_type="CLIP Visual",
                results=history_results,
                search_duration_ms=duration_ms,
                query_image=image_bytes,
                query_image_name=image_filename,
                collection=collection
            )

        return results

    except Exception as e:
        logger.error(f"CLIP search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clip/search-all", response_model=List[SearchResult])
async def clip_search_all_collections(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=500),
    background_tasks: BackgroundTasks = None
):
    """
    Search for similar images across ALL available CLIP collections.
    Uses memory-mapping for minimal RAM usage.

    - **file**: Query image to search for
    - **top_k**: Number of results per collection (results merged and sorted by score)
    """
    start_time = time.time()

    try:
        from clip_searcher import search_all_collections
        image_bytes = await file.read()
        image_filename = file.filename

        logger.info(f"CLIP search ALL collections with {len(image_bytes)} bytes")
        matches = search_all_collections(
            image_bytes=image_bytes,
            top_k=top_k,
            use_mmap=True  # Use memory-mapping for low RAM usage
        )

        results = []
        for m in matches:
            metadata = m.get('metadata', {})
            metadata['collection'] = m.get('collection', 'unknown')
            results.append(SearchResult(
                path=m['path'],
                score=m['score'],
                verified_matches=m.get('verified_matches', 0),
                metadata=metadata
            ))

        # Save to history in background
        duration_ms = int((time.time() - start_time) * 1000)
        history_results = [{'path': m['path'], 'score': m['score'], 'collection': m.get('collection')} for m in matches]
        if background_tasks:
            background_tasks.add_task(
                save_search_to_history,
                search_type="CLIP Visual (All Collections)",
                results=history_results,
                search_duration_ms=duration_ms,
                query_image=image_bytes,
                query_image_name=image_filename,
                collection="all"
            )

        logger.info(f"CLIP search ALL: {len(results)} results in {duration_ms}ms")
        return results

    except Exception as e:
        logger.error(f"CLIP search all collections failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TextSearchRequestWithCollection(BaseModel):
    query: str
    top_k: int = 50
    collection: str = "books"


@app.post("/clip/text", response_model=List[SearchResult])
async def clip_text_search(request: TextSearchRequestWithCollection, background_tasks: BackgroundTasks = None):
    """
    Search for images using a text query (e.g., "truck", "red car").

    - **query**: Text description to search for
    - **top_k**: Number of results to return
    - **collection**: Collection to search (books, print_ads)
    """
    start_time = time.time()

    try:
        cs = get_clip_searcher(request.collection)

        logger.info(f"CLIP text search: '{request.query}' in collection '{request.collection}'")
        matches = cs.search_by_text(request.query, top_k=request.top_k)

        results = []
        for m in matches:
            results.append(SearchResult(
                path=m['path'],
                score=m['score'],
                verified_matches=m.get('verified_matches', 0),
                metadata=m.get('metadata', {})
            ))

        # Save to history in background
        duration_ms = int((time.time() - start_time) * 1000)
        history_results = [{'path': m['path'], 'score': m['score']} for m in matches]
        if background_tasks:
            background_tasks.add_task(
                save_search_to_history,
                search_type="Text",
                results=history_results,
                search_duration_ms=duration_ms,
                query_text=request.query,
                collection=request.collection
            )

        return results

    except Exception as e:
        logger.error(f"CLIP text search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TextSearchAllRequest(BaseModel):
    query: str
    top_k: int = 50


@app.post("/clip/text-all", response_model=List[SearchResult])
async def clip_text_search_all(request: TextSearchAllRequest, background_tasks: BackgroundTasks = None):
    """
    Search for images using a text query across ALL collections.
    Uses memory-mapping for minimal RAM usage.

    - **query**: Text description to search for
    - **top_k**: Number of results per collection
    """
    start_time = time.time()

    try:
        from clip_searcher import search_all_collections

        logger.info(f"CLIP text search ALL: '{request.query}'")
        matches = search_all_collections(
            text_query=request.query,
            top_k=request.top_k,
            use_mmap=True
        )

        results = []
        for m in matches:
            metadata = m.get('metadata', {})
            metadata['collection'] = m.get('collection', 'unknown')
            results.append(SearchResult(
                path=m['path'],
                score=m['score'],
                verified_matches=m.get('verified_matches', 0),
                metadata=metadata
            ))

        # Save to history
        duration_ms = int((time.time() - start_time) * 1000)
        history_results = [{'path': m['path'], 'score': m['score'], 'collection': m.get('collection')} for m in matches]
        if background_tasks:
            background_tasks.add_task(
                save_search_to_history,
                search_type="Text (All Collections)",
                results=history_results,
                search_duration_ms=duration_ms,
                query_text=request.query,
                collection="all"
            )

        logger.info(f"CLIP text search ALL: {len(results)} results in {duration_ms}ms")
        return results

    except Exception as e:
        logger.error(f"CLIP text search all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clip/text", response_model=List[SearchResult])
async def clip_text_search_get(
    query: str = Query(..., description="Text to search for"),
    top_k: int = Query(default=50, ge=1, le=500),
    collection: str = Query(default="books", description="Collection to search (books, print_ads)"),
    background_tasks: BackgroundTasks = None
):
    """
    Search for images using a text query (GET version).
    """
    start_time = time.time()

    try:
        cs = get_clip_searcher(collection)

        logger.info(f"CLIP text search: '{query}' in collection '{collection}'")
        matches = cs.search_by_text(query, top_k=top_k)

        results = []
        for m in matches:
            results.append(SearchResult(
                path=m['path'],
                score=m['score'],
                verified_matches=m.get('verified_matches', 0),
                metadata=m.get('metadata', {})
            ))

        # Save to history in background
        duration_ms = int((time.time() - start_time) * 1000)
        history_results = [{'path': m['path'], 'score': m['score']} for m in matches]
        if background_tasks:
            background_tasks.add_task(
                save_search_to_history,
                search_type="Text",
                results=history_results,
                search_duration_ms=duration_ms,
                query_text=query,
                collection=collection
            )

        return results

    except Exception as e:
        logger.error(f"CLIP text search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== DISK Search Endpoints ==============

@app.post("/disk/search", response_model=DiskSearchStartResponse)
async def disk_search_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=500),
    k: int = Query(default=5, ge=1, le=20, description="Nearest neighbors per keypoint"),
    threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="Minimum similarity for voting"),
    live_tracking: bool = Query(default=True, description="Enable live progress tracking"),
    chunk_ids: str = Query(default=None, description="Comma-separated chunk IDs for testing (e.g. '141,142,143')"),
    collections: str = Query(default=None, description="Comma-separated collections to search (e.g. 'books,print_ads'). Default: all"),
    background_tasks: BackgroundTasks = None
):
    """
    Queue a DISK keypoint search and return immediately.

    Best for finding the source of cropped images. Client should poll
    `/history/{search_id}` for progress and results.
    In-progress updates store a trimmed leaderboard (default top 20);
    final completion writes the full leaderboard.

    IMPORTANT: Uses a queue to ensure only one search runs at a time.
    Other requests will wait in queue to prevent GPU/memory issues.

    - **file**: Query image (cropped image to find source of)
    - **top_k**: Number of results to return (1-500)
    - **k**: Nearest neighbors per keypoint for voting
    - **threshold**: Minimum similarity score to count as vote
    - **live_tracking**: Enable live progress updates (default: true)
    - **chunk_ids**: Comma-separated chunk IDs for testing (e.g. '141,142,143')
    - **collections**: Comma-separated collections to search (e.g. 'books,print_ads'). Default: all
    """
    try:
        from disk_searcher import search_disk, get_total_chunks
        from db_helper import (
            create_search_session,
            update_search_progress,
            complete_search_session,
            fail_search_session,
            add_search_note
        )
        from disk_queue import get_disk_queue

        image_bytes = await file.read()
        image_filename = file.filename

        # Parse collections
        categories = None
        if collections:
            categories = [c.strip() for c in collections.split(',') if c.strip()]
            if not categories:
                categories = None

        # Parse chunk_ids if provided
        specific_chunks = None
        if chunk_ids:
            specific_chunks = [int(x.strip()) for x in chunk_ids.split(',')]

        cat_label = ",".join(categories) if categories else "all"
        logger.info(f"DISK search: {len(image_bytes)} bytes, top_k={top_k}, collections={cat_label}, live_tracking={live_tracking}" +
                    (f", chunks={specific_chunks}" if specific_chunks else ""))

        # Count total chunks for progress tracking
        if specific_chunks:
            total_chunks = len(specific_chunks)
        else:
            total_chunks = get_total_chunks(categories)

        # Async DISK search always creates a history row for polling.
        search_id = create_search_session(
            search_type="DISK Keypoint",
            query_image=image_bytes,
            query_image_name=image_filename,
            collection=cat_label,
            total_chunks=total_chunks
        )
        logger.info(f"Created search session #{search_id} with {total_chunks} chunks across {cat_label}")

        # Progress callback for live updates
        def progress_callback(current_chunk, total_chunks, top_results, elapsed_ms):
            try:
                update_search_progress(search_id, current_chunk, total_chunks, top_results, elapsed_ms)
            except Exception as e:
                logger.error(f"Failed to update search progress: {e}")

        # Define the search function to be executed (categories captured in closure)
        def run_search(image_bytes, top_k, k, threshold, specific_chunks, progress_callback, check_stopped=None):
            run_start = time.time()
            try:
                matches = search_disk(
                    image_bytes,
                    top_k=top_k,
                    k=k,
                    threshold=threshold,
                    specific_chunks=specific_chunks,
                    categories=categories,
                    progress_callback=progress_callback,
                    check_stopped=check_stopped,
                    search_id=search_id
                )
                if not (check_stopped and check_stopped()):
                    duration_ms = int((time.time() - run_start) * 1000)
                    # Keep progress writes small, but store the full final leaderboard.
                    update_search_progress(
                        search_id,
                        total_chunks,
                        total_chunks,
                        matches,
                        duration_ms,
                        max_results=100
                    )
                    complete_search_session(search_id, duration_ms)
                return matches
            except Exception as e:
                fail_search_session(search_id, str(e))
                raise

        # Add to queue
        queue = get_disk_queue()
        queue_position = await queue.add_search(
            search_id=search_id if search_id else 0,
            image_bytes=image_bytes,
            top_k=top_k,
            k=k,
            threshold=threshold,
            specific_chunks=specific_chunks,
            progress_callback=progress_callback,
            search_function=run_search
        )

        status = await queue.get_status(search_id)
        status_label = status['status'] if status else 'queued'
        if status_label == 'queued':
            queue_pos = status.get('position', queue_position) if status else queue_position
            add_search_note(search_id, f"Waiting in queue (position: {queue_pos})")
            message = f"Queued at position {queue_pos}"
        else:
            add_search_note(search_id, "")
            message = "Search started"

        logger.info(f"DISK search #{search_id} accepted ({message})")
        return DiskSearchStartResponse(
            search_id=search_id,
            status=status_label,
            queue_position=status.get('position', queue_position) if status else queue_position,
            total_chunks=total_chunks,
            message=message
        )

    except Exception as e:
        logger.error(f"DISK search failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/disk/resume/{source_search_id}", response_model=DiskSearchStartResponse)
async def resume_disk_search(
    source_search_id: int,
    top_k: int = Query(default=50, ge=1, le=500),
    k: int = Query(default=5, ge=1, le=20, description="Nearest neighbors per keypoint"),
    threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="Minimum similarity for voting"),
    background_tasks: BackgroundTasks = None
):
    """
    Resume a previously interrupted DISK search from the next chunk.

    Creates a new search session and seeds vote counts from checkpoint data
    when available. If no checkpoint exists, falls back to current DB results.
    """
    try:
        from db_helper import (
            get_search_details,
            get_search_query_image,
            create_search_session,
            update_search_progress,
            complete_search_session,
            fail_search_session,
            add_search_note
        )
        from disk_searcher import search_disk, load_search_checkpoint, get_total_chunks
        from disk_queue import get_disk_queue

        source = get_search_details(source_search_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source search not found")

        source_type = source.get("SearchType", "")
        if "DISK" not in source_type:
            raise HTTPException(status_code=400, detail="Only DISK searches can be resumed")

        source_status = (source.get("Status") or "").lower()
        if source_status not in ("stopped", "failed"):
            raise HTTPException(
                status_code=400,
                detail=f"Search status is '{source_status or 'unknown'}'; only stopped/failed searches can be resumed"
            )

        image_bytes = get_search_query_image(source_search_id)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Source query image is not available for resume")

        source_collection = source.get("Collection") or "all"
        categories = None
        if source_collection and source_collection != "all":
            categories = [c.strip() for c in source_collection.split(",") if c.strip()]
            if not categories:
                categories = None

        total_chunks = source.get("TotalChunks")
        if not total_chunks:
            total_chunks = get_total_chunks(categories)

        checkpoint = load_search_checkpoint(source_search_id)
        initial_votes = {}
        if checkpoint and isinstance(checkpoint.get("votes"), dict):
            initial_votes = checkpoint["votes"]
            resume_from = int(checkpoint.get("current_chunk", 0)) + 1
        else:
            resume_from = (_parse_progress_chunk(source.get("CurrentProgress")) or 0) + 1
            for r in source.get("Results", []):
                path = r.get("ImagePath")
                votes = r.get("VerifiedMatches")
                if path and isinstance(votes, int) and votes > 0:
                    initial_votes[path] = votes

        resume_from = max(1, resume_from)
        if resume_from > total_chunks:
            raise HTTPException(status_code=400, detail="Search already reached the last chunk; nothing to resume")

        image_name = source.get("QueryImageName")
        new_search_id = create_search_session(
            search_type="DISK Keypoint",
            query_image=image_bytes,
            query_image_name=image_name,
            collection=source_collection,
            total_chunks=total_chunks
        )

        add_search_note(
            new_search_id,
            f"Resumed from search #{source_search_id} at chunk {resume_from}/{total_chunks}"
        )

        def progress_callback(current_chunk, total_chunks, top_results, elapsed_ms):
            try:
                update_search_progress(new_search_id, current_chunk, total_chunks, top_results, elapsed_ms)
            except Exception as e:
                logger.error(f"Failed to update resumed search progress: {e}")

        def run_search(image_bytes, top_k, k, threshold, specific_chunks, progress_callback, check_stopped=None):
            run_start = time.time()
            try:
                matches = search_disk(
                    image_bytes,
                    top_k=top_k,
                    k=k,
                    threshold=threshold,
                    specific_chunks=None,
                    categories=categories,
                    progress_callback=progress_callback,
                    check_stopped=check_stopped,
                    search_id=new_search_id,
                    start_chunk=resume_from,
                    initial_votes=initial_votes
                )
                if not (check_stopped and check_stopped()):
                    duration_ms = int((time.time() - run_start) * 1000)
                    update_search_progress(
                        new_search_id,
                        total_chunks,
                        total_chunks,
                        matches,
                        duration_ms,
                        max_results=100
                    )
                    complete_search_session(new_search_id, duration_ms)
                return matches
            except Exception as e:
                fail_search_session(new_search_id, str(e))
                raise

        queue = get_disk_queue()
        queue_position = await queue.add_search(
            search_id=new_search_id,
            image_bytes=image_bytes,
            top_k=top_k,
            k=k,
            threshold=threshold,
            specific_chunks=None,
            progress_callback=progress_callback,
            search_function=run_search
        )

        status = await queue.get_status(new_search_id)
        status_label = status["status"] if status else "queued"
        queue_pos = status.get("position", queue_position) if status else queue_position

        if status_label == "queued":
            add_search_note(
                new_search_id,
                f"Resumed from #{source_search_id}. Waiting in queue (position: {queue_pos})"
            )
            message = f"Resumed from chunk {resume_from}. Queued at position {queue_pos}"
        else:
            message = f"Resumed from chunk {resume_from}. Search started"

        logger.info(
            f"Resumed DISK search #{source_search_id} -> #{new_search_id} "
            f"(from chunk {resume_from}/{total_chunks})"
        )
        return DiskSearchStartResponse(
            search_id=new_search_id,
            status=status_label,
            queue_position=queue_pos,
            total_chunks=total_chunks,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume DISK search #{source_search_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/disk/gpu-status")
async def get_disk_gpu_status():
    """Check if GPU search is enabled for DISK searches."""
    from disk_searcher import _gpu_search_available, _check_gpu_search
    import torch
    return {
        "cached_value": _gpu_search_available,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "check_result": _check_gpu_search()
    }


@app.get("/disk/queue")
async def get_disk_queue_info():
    """
    Get information about the DISK search queue.

    Returns:
    - queue_length: Number of searches waiting
    - current_search_id: ID of currently running search (null if none)
    - completed_count: Number of completed searches in cache
    """
    try:
        from disk_queue import get_disk_queue

        queue = get_disk_queue()
        info = await queue.get_queue_info()

        return info
    except Exception as e:
        logger.error(f"Failed to get queue info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== CLIP Indexing Endpoints ==============

clip_indexer = None


def get_clip_indexer():
    """Lazy-load the CLIP indexer."""
    global clip_indexer
    if clip_indexer is None:
        from clip_indexer import ClipIndexer
        clip_indexer = ClipIndexer()
    return clip_indexer


class IndexRequest(BaseModel):
    root_path: str
    output_dir: str


class AddToIndexRequest(BaseModel):
    index_dir: str
    folder: str


class IndexResponse(BaseModel):
    status: str
    message: str = ""
    total_images: int = 0
    index_path: str = ""


# Track indexing status
indexing_status = {"running": False, "progress": "", "result": None}


def run_indexing_task(root_path: str, output_dir: str):
    """Background task for indexing."""
    global indexing_status
    indexing_status = {"running": True, "progress": "Starting...", "result": None}

    try:
        indexer = get_clip_indexer()
        indexing_status["progress"] = f"Indexing {root_path}..."
        result = indexer.index_folder(root_path, output_dir)
        indexing_status["result"] = result
        indexing_status["progress"] = "Complete"
    except Exception as e:
        indexing_status["result"] = {"status": "error", "message": str(e)}
        indexing_status["progress"] = f"Error: {e}"
    finally:
        indexing_status["running"] = False


@app.post("/clip/index", response_model=IndexResponse)
async def start_clip_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Start CLIP indexing for a folder (runs in background).

    - **root_path**: Folder to scan recursively for images
    - **output_dir**: Directory to store index.faiss and paths.json
    """
    if indexing_status["running"]:
        raise HTTPException(status_code=409, detail="Indexing already in progress")

    if not os.path.exists(request.root_path):
        raise HTTPException(status_code=400, detail=f"Path not found: {request.root_path}")

    background_tasks.add_task(run_indexing_task, request.root_path, request.output_dir)

    return IndexResponse(
        status="started",
        message=f"Indexing started for {request.root_path}"
    )


@app.get("/clip/index/status")
def get_indexing_status():
    """Get current indexing status."""
    return indexing_status


@app.post("/clip/index/add", response_model=IndexResponse)
async def add_to_clip_index(request: AddToIndexRequest):
    """
    Add images from a new folder to an existing CLIP index.

    - **index_dir**: Directory with existing index.faiss
    - **folder**: New folder to add
    """
    if indexing_status["running"]:
        raise HTTPException(status_code=409, detail="Indexing already in progress")

    if not os.path.exists(request.folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder}")

    try:
        indexer = get_clip_indexer()
        result = indexer.add_to_index(request.index_dir, request.folder)

        return IndexResponse(
            status=result["status"],
            message=result.get("message", ""),
            total_images=result.get("total", result.get("added", 0)),
            index_path=result.get("index_path", "")
        )
    except Exception as e:
        logger.error(f"Add to index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Search History Endpoints ==============

class SearchHistoryEntry(BaseModel):
    Id: int
    SearchDate: str
    SearchType: str
    QueryText: Optional[str] = None
    QueryImageName: Optional[str] = None
    ResultCount: int
    SearchDurationMs: Optional[int] = None
    Collection: Optional[str] = None
    TopResultPath: Optional[str] = None
    TopResultScore: Optional[float] = None
    Status: Optional[str] = "completed"
    CurrentProgress: Optional[str] = None
    TotalChunks: Optional[int] = None
    TopResultVotes: Optional[int] = None


class SearchHistoryListResponse(BaseModel):
    entries: List[SearchHistoryEntry]
    total: int
    page: int
    page_size: int


class SearchHistoryDetailResult(BaseModel):
    Rank: int
    ImagePath: str
    Score: float
    VerifiedMatches: Optional[int] = None
    KeypointMatches: Optional[int] = None
    TemplateScore: Optional[float] = None
    CombinedScore: Optional[float] = None


class SearchHistoryDetail(BaseModel):
    Id: int
    SearchDate: str
    SearchType: str
    QueryText: Optional[str] = None
    QueryImageName: Optional[str] = None
    ResultCount: int
    SearchDurationMs: Optional[int] = None
    Collection: Optional[str] = None
    Notes: Optional[str] = None
    Status: Optional[str] = "completed"
    CurrentProgress: Optional[str] = None
    TotalChunks: Optional[int] = None
    Results: List[SearchHistoryDetailResult]


class SaveSearchResponse(BaseModel):
    id: int
    message: str


@app.get("/history", response_model=SearchHistoryListResponse)
def get_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search_type: Optional[str] = Query(default=None)
):
    """
    Get search history with pagination.

    - **page**: Page number (1-based)
    - **page_size**: Results per page (1-100)
    - **search_type**: Filter by type (DINOv2, CLIP, DISK, Face, Text)
    """
    try:
        from db_helper import get_search_history as db_get_history, get_search_history_count

        entries = db_get_history(page=page, page_size=page_size, search_type=search_type)
        total = get_search_history_count(search_type=search_type)

        return SearchHistoryListResponse(
            entries=[SearchHistoryEntry(**e) for e in entries],
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"Failed to get search history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{search_id}", response_model=SearchHistoryDetail)
def get_history_detail(search_id: int):
    """
    Get full details for a specific search including all results.
    """
    try:
        from db_helper import get_search_details

        details = get_search_details(search_id)
        if not details:
            raise HTTPException(status_code=404, detail="Search not found")

        return SearchHistoryDetail(
            Id=details['Id'],
            SearchDate=details['SearchDate'],
            SearchType=details['SearchType'],
            QueryText=details.get('QueryText'),
            QueryImageName=details.get('QueryImageName'),
            ResultCount=details['ResultCount'],
            SearchDurationMs=details.get('SearchDurationMs'),
            Collection=details.get('Collection'),
            Notes=details.get('Notes'),
            Status=details.get('Status', 'completed'),
            CurrentProgress=details.get('CurrentProgress'),
            TotalChunks=details.get('TotalChunks'),
            Results=[SearchHistoryDetailResult(**r) for r in details['Results']]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get search details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{search_id}/image")
def get_history_query_image(search_id: int):
    """
    Get the query image for a specific search.
    """
    try:
        from db_helper import get_search_query_image as db_get_image

        image_bytes = db_get_image(search_id)
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Image not found")

        return Response(content=image_bytes, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get search image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history", response_model=SaveSearchResponse)
async def save_history(
    file: UploadFile = File(None),
    search_type: str = Form(...),
    query_text: Optional[str] = Form(default=None),
    results_json: str = Form(..., description="JSON array of results"),
    search_duration_ms: Optional[int] = Form(default=None),
    collection: Optional[str] = Form(default=None)
):
    """
    Save a search to history.

    - **file**: The query image (optional)
    - **search_type**: Type of search
    - **query_text**: Text query (for text searches)
    - **results_json**: JSON string of search results
    - **search_duration_ms**: Search duration in milliseconds
    - **collection**: Collection that was searched
    """
    import json

    try:
        from db_helper import save_search_history as db_save_history

        # Parse results
        results = json.loads(results_json)

        # Read image if provided
        image_bytes = None
        image_name = None
        if file:
            image_bytes = await file.read()
            image_name = file.filename

        search_id = db_save_history(
            search_type=search_type,
            query_image=image_bytes,
            query_image_name=image_name,
            query_text=query_text,
            results=results,
            search_duration_ms=search_duration_ms,
            collection=collection
        )

        return SaveSearchResponse(
            id=search_id,
            message=f"Saved search with {len(results)} results"
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid results JSON: {e}")
    except Exception as e:
        logger.error(f"Failed to save search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{search_id}")
def delete_history_entry(search_id: int):
    """Delete a search history entry."""
    try:
        from db_helper import delete_search_history

        if delete_search_history(search_id):
            return {"message": "Deleted"}
        else:
            raise HTTPException(status_code=404, detail="Search not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history/{search_id}/stop")
async def stop_history_entry(search_id: int):
    """Stop an in-progress search session."""
    try:
        from db_helper import stop_search_session
        from disk_queue import get_disk_queue

        # Signal the search thread to stop
        queue = get_disk_queue()
        await queue.stop_search(search_id)

        # Mark as stopped in DB
        if stop_search_session(search_id):
            return {"message": "Stopped"}
        else:
            raise HTTPException(status_code=404, detail="Search not found or not in progress")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/history/{search_id}/note")
def update_history_note(search_id: int, note: str = Query(...)):
    """Add or update a note on a search."""
    try:
        from db_helper import add_search_note

        if add_search_note(search_id, note):
            return {"message": "Note updated"}
        else:
            raise HTTPException(status_code=404, detail="Search not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
