"""
DISK Search Queue Manager

Ensures only one DISK search runs at a time to prevent:
- GPU memory exhaustion (multiple DISK models)
- SSD space overflow (multiple 22GB chunks)
- File conflicts in chunk_buffer directory
- Network congestion
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SearchStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SearchRequest:
    """A queued search request."""
    search_id: int
    image_bytes: bytes
    top_k: int
    k: int
    threshold: float
    specific_chunks: Optional[list]
    progress_callback: Optional[Callable]
    search_function: Callable

    # Status tracking
    status: SearchStatus
    queued_time: float
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class DiskSearchQueue:
    """
    Queue manager for DISK searches.

    Ensures only one search runs at a time. Other requests wait in queue.
    """

    def __init__(self):
        self.queue: list[SearchRequest] = []
        self.current_search: Optional[SearchRequest] = None
        self.completed_searches: Dict[int, SearchRequest] = {}  # Keep last 100
        self.lock = asyncio.Lock()
        self.processing_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the queue processor."""
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._process_queue())
            logger.info("DISK search queue processor started")

    async def stop(self):
        """Stop the queue processor."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
            logger.info("DISK search queue processor stopped")

    async def add_search(
        self,
        search_id: int,
        image_bytes: bytes,
        top_k: int,
        k: int,
        threshold: float,
        specific_chunks: Optional[list],
        progress_callback: Optional[Callable],
        search_function: Callable
    ) -> int:
        """
        Add a search to the queue.

        Returns the position in queue (0 = running now, 1 = next, etc.)
        """
        async with self.lock:
            request = SearchRequest(
                search_id=search_id,
                image_bytes=image_bytes,
                top_k=top_k,
                k=k,
                threshold=threshold,
                specific_chunks=specific_chunks,
                progress_callback=progress_callback,
                search_function=search_function,
                status=SearchStatus.QUEUED,
                queued_time=time.time()
            )

            self.queue.append(request)
            position = len(self.queue)  # Position in queue (1-indexed)

            logger.info(f"Added search #{search_id} to queue at position {position}")

            return position

    async def get_status(self, search_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a search by ID."""
        async with self.lock:
            # Check if it's the current search
            if self.current_search and self.current_search.search_id == search_id:
                return {
                    'search_id': search_id,
                    'status': SearchStatus.RUNNING.value,
                    'position': 0,
                    'elapsed_seconds': time.time() - self.current_search.started_time if self.current_search.started_time else 0
                }

            # Check if it's in the queue
            for i, req in enumerate(self.queue):
                if req.search_id == search_id:
                    return {
                        'search_id': search_id,
                        'status': SearchStatus.QUEUED.value,
                        'position': i + 1,
                        'wait_seconds': time.time() - req.queued_time
                    }

            # Check if it's completed
            if search_id in self.completed_searches:
                req = self.completed_searches[search_id]
                return {
                    'search_id': search_id,
                    'status': req.status.value,
                    'result': req.result,
                    'error': req.error,
                    'elapsed_seconds': (req.completed_time - req.started_time) if req.started_time and req.completed_time else 0
                }

            return None

    async def get_queue_info(self) -> Dict[str, Any]:
        """Get overall queue information."""
        async with self.lock:
            return {
                'queue_length': len(self.queue),
                'current_search_id': self.current_search.search_id if self.current_search else None,
                'completed_count': len(self.completed_searches)
            }

    async def _process_queue(self):
        """Background task that processes the queue."""
        logger.info("Queue processor started")

        while True:
            try:
                # Get next search from queue
                request = None
                async with self.lock:
                    if self.queue:
                        request = self.queue.pop(0)
                        self.current_search = request
                        request.status = SearchStatus.RUNNING
                        request.started_time = time.time()

                if request:
                    logger.info(f"Starting search #{request.search_id} (waited {time.time() - request.queued_time:.1f}s in queue)")

                    try:
                        # Run the search function
                        result = await asyncio.to_thread(
                            request.search_function,
                            request.image_bytes,
                            request.top_k,
                            request.k,
                            request.threshold,
                            request.specific_chunks,
                            request.progress_callback
                        )

                        request.result = result
                        request.status = SearchStatus.COMPLETED
                        logger.info(f"Search #{request.search_id} completed successfully")

                    except Exception as e:
                        logger.error(f"Search #{request.search_id} failed: {e}", exc_info=True)
                        request.error = str(e)
                        request.status = SearchStatus.FAILED

                    finally:
                        request.completed_time = time.time()

                        # Move to completed searches
                        async with self.lock:
                            self.completed_searches[request.search_id] = request

                            # Keep only last 100 completed searches
                            if len(self.completed_searches) > 100:
                                oldest_id = min(self.completed_searches.keys())
                                del self.completed_searches[oldest_id]

                            self.current_search = None

                else:
                    # No searches in queue, sleep a bit
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Queue processor cancelled")
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}", exc_info=True)
                await asyncio.sleep(1)


# Global queue instance
_disk_queue: Optional[DiskSearchQueue] = None


def get_disk_queue() -> DiskSearchQueue:
    """Get the global DISK queue instance."""
    global _disk_queue
    if _disk_queue is None:
        _disk_queue = DiskSearchQueue()
    return _disk_queue


async def initialize_disk_queue():
    """Initialize and start the disk queue."""
    queue = get_disk_queue()
    await queue.start()
    return queue
