"""
Database helper for SQL Server connection and search history operations.
"""

import os
import pyodbc
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Connection settings from environment or defaults
DB_SERVER = os.environ.get("DB_SERVER", "localhost")
DB_NAME = os.environ.get("DB_NAME", "trivia")
DB_TRUSTED = os.environ.get("DB_TRUSTED", "yes").lower() == "yes"
DB_USER = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")


def get_connection_string() -> str:
    """Build SQL Server connection string."""
    if DB_TRUSTED:
        return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_NAME};Trusted_Connection=yes;"
    else:
        return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD};"


def get_connection() -> pyodbc.Connection:
    """Get a database connection."""
    return pyodbc.connect(get_connection_string())


def create_search_session(
    search_type: str,
    query_image: Optional[bytes] = None,
    query_image_name: Optional[str] = None,
    query_text: Optional[str] = None,
    collection: Optional[str] = None,
    total_chunks: Optional[int] = None
) -> int:
    """
    Create a new search session (for live tracking).

    Returns the search ID that can be updated as search progresses.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO ImageSearchHistory (
                SearchType, QueryText, QueryImage, QueryImageName,
                ResultCount, SearchDurationMs, Collection, Status, CurrentProgress, TotalChunks
            )
            OUTPUT INSERTED.Id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            search_type,
            query_text,
            query_image,
            query_image_name,
            0,  # Will be updated as we search
            None,  # Duration not known yet
            collection,
            'in_progress',
            'Starting...',
            total_chunks
        ))

        row = cursor.fetchone()
        search_id = row[0]
        conn.commit()
        logger.info(f"Created search session #{search_id}")
        return search_id

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create search session: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def update_search_progress(
    search_id: int,
    current_chunk: int,
    total_chunks: int,
    top_results: List[Dict[str, Any]],
    elapsed_ms: Optional[int] = None
):
    """Update search progress with current top results."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Update search history
        progress_text = f"Searching chunk {current_chunk}/{total_chunks}"
        cursor.execute("""
            UPDATE ImageSearchHistory
            SET CurrentProgress = ?,
                ResultCount = ?,
                SearchDurationMs = ?
            WHERE Id = ?
        """, (progress_text, len(top_results), elapsed_ms, search_id))

        # Delete old results and insert new ones
        cursor.execute("DELETE FROM ImageSearchResults WHERE SearchHistoryId = ?", (search_id,))

        for rank, result in enumerate(top_results[:100], 1):  # Top 100
            cursor.execute("""
                INSERT INTO ImageSearchResults (
                    SearchHistoryId, Rank, ImagePath, Score,
                    VerifiedMatches, KeypointMatches, TemplateScore, CombinedScore
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                search_id,
                rank,
                result.get('path', ''),
                result.get('score', 0.0),
                result.get('verified_matches', result.get('votes', None)),
                result.get('keypoint_matches', None),
                result.get('template_score', None),
                result.get('combined_score', None)
            ))

        conn.commit()

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update search progress: {e}")
    finally:
        cursor.close()
        conn.close()


def complete_search_session(search_id: int, duration_ms: int):
    """Mark search as completed."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE ImageSearchHistory
            SET Status = 'completed',
                CurrentProgress = 'Complete',
                SearchDurationMs = ?
            WHERE Id = ?
        """, (duration_ms, search_id))
        conn.commit()
        logger.info(f"Completed search session #{search_id}")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to complete search session: {e}")
    finally:
        cursor.close()
        conn.close()


def stop_search_session(search_id: int):
    """Mark a search session as stopped."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE ImageSearchHistory
            SET Status = 'stopped',
                CurrentProgress = 'Stopped by user'
            WHERE Id = ? AND Status = 'in_progress'
        """, (search_id,))
        conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"Stopped search session #{search_id}")
        return updated

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to stop search session: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def get_search_status(search_id: int) -> Optional[str]:
    """Get the current status of a search session."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT Status FROM ImageSearchHistory WHERE Id = ?", (search_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()
        conn.close()


def save_search_history(
    search_type: str,
    query_image: Optional[bytes] = None,
    query_image_name: Optional[str] = None,
    query_text: Optional[str] = None,
    results: List[Dict[str, Any]] = None,
    search_duration_ms: Optional[int] = None,
    collection: Optional[str] = None
) -> int:
    """
    Save a search to history with its results.

    Args:
        search_type: Type of search ('DINOv2', 'CLIP', 'Deep Search', 'Face', 'Text')
        query_image: The uploaded query image bytes
        query_image_name: Original filename
        query_text: Text query (for text searches)
        results: List of search results with 'path', 'score', etc.
        search_duration_ms: How long the search took
        collection: Which collection was searched

    Returns:
        The new SearchHistory ID
    """
    results = results or []

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Insert search history
        cursor.execute("""
            INSERT INTO ImageSearchHistory (
                SearchType, QueryText, QueryImage, QueryImageName,
                ResultCount, SearchDurationMs, Collection
            )
            OUTPUT INSERTED.Id
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            search_type,
            query_text,
            query_image,
            query_image_name,
            len(results),
            search_duration_ms,
            collection
        ))

        row = cursor.fetchone()
        search_id = row[0]

        # Insert results
        for rank, result in enumerate(results, 1):
            cursor.execute("""
                INSERT INTO ImageSearchResults (
                    SearchHistoryId, Rank, ImagePath, Score,
                    VerifiedMatches, KeypointMatches, TemplateScore, CombinedScore
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                search_id,
                rank,
                result.get('path', ''),
                result.get('score', 0.0),
                result.get('verified_matches', None),
                result.get('keypoint_matches', None),
                result.get('template_score', None),
                result.get('combined_score', None)
            ))

        conn.commit()
        logger.info(f"Saved search history #{search_id} with {len(results)} results")
        return search_id

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save search history: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_search_history(
    page: int = 1,
    page_size: int = 20,
    search_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get search history with pagination.

    Args:
        page: Page number (1-based)
        page_size: Results per page
        search_type: Filter by search type (optional)

    Returns:
        List of search history entries with top result
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if search_type:
            cursor.execute("""
                SELECT
                    h.Id,
                    h.SearchDate,
                    h.SearchType,
                    h.QueryText,
                    h.QueryImageName,
                    h.ResultCount,
                    h.SearchDurationMs,
                    h.Collection,
                    r.ImagePath AS TopResultPath,
                    r.Score AS TopResultScore,
                    h.Status,
                    h.CurrentProgress,
                    h.TotalChunks,
                    r.VerifiedMatches AS TopResultVotes
                FROM ImageSearchHistory h
                LEFT JOIN ImageSearchResults r ON h.Id = r.SearchHistoryId AND r.Rank = 1
                WHERE h.SearchType = ?
                ORDER BY h.SearchDate DESC
                OFFSET ? ROWS
                FETCH NEXT ? ROWS ONLY
            """, (search_type, (page - 1) * page_size, page_size))
        else:
            cursor.execute("""
                SELECT
                    h.Id,
                    h.SearchDate,
                    h.SearchType,
                    h.QueryText,
                    h.QueryImageName,
                    h.ResultCount,
                    h.SearchDurationMs,
                    h.Collection,
                    r.ImagePath AS TopResultPath,
                    r.Score AS TopResultScore,
                    h.Status,
                    h.CurrentProgress,
                    h.TotalChunks,
                    r.VerifiedMatches AS TopResultVotes
                FROM ImageSearchHistory h
                LEFT JOIN ImageSearchResults r ON h.Id = r.SearchHistoryId AND r.Rank = 1
                ORDER BY h.SearchDate DESC
                OFFSET ? ROWS
                FETCH NEXT ? ROWS ONLY
            """, ((page - 1) * page_size, page_size))

        columns = [column[0] for column in cursor.description]
        results = []

        for row in cursor.fetchall():
            entry = dict(zip(columns, row))
            # Convert datetime for JSON serialization
            if entry.get('SearchDate'):
                entry['SearchDate'] = entry['SearchDate'].isoformat()
            results.append(entry)

        return results

    finally:
        cursor.close()
        conn.close()


def get_search_history_count(search_type: Optional[str] = None) -> int:
    """Get total count of search history entries."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if search_type:
            cursor.execute(
                "SELECT COUNT(*) FROM ImageSearchHistory WHERE SearchType = ?",
                (search_type,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM ImageSearchHistory")

        return cursor.fetchone()[0]

    finally:
        cursor.close()
        conn.close()


def get_search_details(search_id: int) -> Optional[Dict[str, Any]]:
    """
    Get full details for a specific search including all results.

    Args:
        search_id: The SearchHistory ID

    Returns:
        Dict with search metadata and results, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Get search metadata
        cursor.execute("""
            SELECT
                Id, SearchDate, SearchType, QueryText,
                QueryImageName, ResultCount, SearchDurationMs, Collection, Notes,
                Status, CurrentProgress, TotalChunks
            FROM ImageSearchHistory
            WHERE Id = ?
        """, (search_id,))

        row = cursor.fetchone()
        if not row:
            return None

        columns = [column[0] for column in cursor.description]
        search = dict(zip(columns, row))

        if search.get('SearchDate'):
            search['SearchDate'] = search['SearchDate'].isoformat()

        # Get results
        cursor.execute("""
            SELECT
                Rank, ImagePath, Score, VerifiedMatches,
                KeypointMatches, TemplateScore, CombinedScore
            FROM ImageSearchResults
            WHERE SearchHistoryId = ?
            ORDER BY Rank
        """, (search_id,))

        result_columns = [column[0] for column in cursor.description]
        search['Results'] = []

        for row in cursor.fetchall():
            search['Results'].append(dict(zip(result_columns, row)))

        return search

    finally:
        cursor.close()
        conn.close()


def get_search_query_image(search_id: int) -> Optional[bytes]:
    """Get the query image bytes for a search."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT QueryImage FROM ImageSearchHistory WHERE Id = ?",
            (search_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    finally:
        cursor.close()
        conn.close()


def delete_search_history(search_id: int) -> bool:
    """Delete a search history entry and its results."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "DELETE FROM ImageSearchHistory WHERE Id = ?",
            (search_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to delete search history: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def add_search_note(search_id: int, note: str) -> bool:
    """Add or update a note on a search history entry."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "UPDATE ImageSearchHistory SET Notes = ? WHERE Id = ?",
            (note, search_id)
        )
        conn.commit()
        return cursor.rowcount > 0

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to add note: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
