"""
Deception Lens API Server
FastAPI backend for the web application.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import uuid
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deception Lens API",
    description="Visual similarity search using CLIP and DINOv2",
    version="1.0.0"
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
DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
CLIP_INDEX_PATH = os.environ.get("CLIP_INDEX_PATH", "D:/faiss/books/index.faiss")
CLIP_PATHS_PATH = os.environ.get("CLIP_PATHS_PATH", "D:/faiss/books/paths.json")
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_searcher():
    """Lazy-load the DINOv2 searcher."""
    global searcher
    if searcher is None:
        from searcher import DinoSearcher
        logger.info(f"Initializing DINOv2 searcher with DB path: {DB_PATH}")
        searcher = DinoSearcher(db_path=DB_PATH)
    return searcher


def get_clip_searcher():
    """Lazy-load the CLIP searcher."""
    global clip_searcher
    if clip_searcher is None:
        from clip_searcher import ClipSearcher
        logger.info(f"Initializing CLIP searcher with index: {CLIP_INDEX_PATH}")
        clip_searcher = ClipSearcher(
            index_path=CLIP_INDEX_PATH,
            paths_path=CLIP_PATHS_PATH
        )
    return clip_searcher


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


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is healthy."""
    return HealthResponse(
        status="ok",
        searcher_loaded=searcher is not None,
        db_path=DB_PATH
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats(collection: str = "images"):
    """Get statistics for a collection."""
    try:
        s = get_searcher()
        stats = s.get_collection_stats(collection)
        return StatsResponse(
            visual_count=stats["visual_count"],
            face_count=stats["face_count"]
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=50, ge=1, le=500),
    collection: str = Query(default="images"),
    verify: bool = Query(default=False)
):
    """
    Search for similar images.

    - **file**: Query image to search for
    - **top_k**: Number of results to return (1-500)
    - **collection**: Collection name to search in
    - **verify**: Whether to perform geometric verification
    """
    s = get_searcher()
    if s is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    # Save uploaded file temporarily
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Searching with query: {temp_path}")
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
    collection: str = Query(default="images"),
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
        s = get_searcher()
        collections = s.client.list_collections()
        return {
            "collections": [c.name for c in collections]
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    """Delete a collection (both visual and faces variants)."""
    try:
        s = get_searcher()
        deleted = []

        # Try to delete both visual and faces variants
        for suffix in ["_visual", "_faces"]:
            full_name = f"{collection_name}{suffix}"
            try:
                s.client.delete_collection(name=full_name)
                deleted.append(full_name)
                logger.info(f"Deleted collection: {full_name}")
            except Exception as e:
                logger.warning(f"Could not delete {full_name}: {e}")

        if not deleted:
            raise HTTPException(status_code=404, detail=f"No collections found for {collection_name}")

        return {"deleted": deleted}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== CLIP Search Endpoints ==============

class ClipStatsResponse(BaseModel):
    total_images: int
    model: str
    index_path: str


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 50


@app.get("/clip/stats", response_model=ClipStatsResponse)
def get_clip_stats():
    """Get CLIP index statistics."""
    try:
        cs = get_clip_searcher()
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
    top_k: int = Query(default=50, ge=1, le=500)
):
    """
    Search for similar images using CLIP.

    - **file**: Query image to search for
    - **top_k**: Number of results to return (1-500)
    """
    try:
        cs = get_clip_searcher()
        image_bytes = await file.read()

        logger.info(f"CLIP searching with {len(image_bytes)} bytes")
        matches = cs.search_by_image_bytes(image_bytes, top_k=top_k)

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
        logger.error(f"CLIP search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clip/text", response_model=List[SearchResult])
async def clip_text_search(request: TextSearchRequest):
    """
    Search for images using a text query (e.g., "truck", "red car").

    - **query**: Text description to search for
    - **top_k**: Number of results to return
    """
    try:
        cs = get_clip_searcher()

        logger.info(f"CLIP text search: '{request.query}'")
        matches = cs.search_by_text(request.query, top_k=request.top_k)

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
        logger.error(f"CLIP text search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clip/text", response_model=List[SearchResult])
async def clip_text_search_get(
    query: str = Query(..., description="Text to search for"),
    top_k: int = Query(default=50, ge=1, le=500)
):
    """
    Search for images using a text query (GET version).
    """
    try:
        cs = get_clip_searcher()

        logger.info(f"CLIP text search: '{query}'")
        matches = cs.search_by_text(query, top_k=top_k)

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
        logger.error(f"CLIP text search failed: {e}")
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
