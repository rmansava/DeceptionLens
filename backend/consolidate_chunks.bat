@echo off
title DISK Chunk Consolidation (Pipelined)
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   DISK CHUNK CONSOLIDATION (Pipelined NAS -^> Local -^> NAS)
echo ================================================================
echo.
echo   Consolidates ~7000 per-book indexes into chunks for fast searching.
echo.
echo   Source:      T:\faiss\disk_retrieval\books\     (NAS, 13TB)
echo   Local buf:   D:\faiss\disk_retrieval\books\     (SSD, ~200GB buffer)
echo   Output:      T:\faiss\disk_retrieval\chunks\    (NAS)
echo.
echo   Pipelined workflow:
echo     - Background thread copies books from NAS to local buffer
echo     - Main thread processes books into chunks
echo     - Processed books deleted immediately to make room
echo     - Buffer maintains ~100 books (~200GB) ready for processing
echo.
echo   Chunk size: ~20GB (40M vectors) - fits in 32GB RAM for searching
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u consolidate_search_chunks.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
