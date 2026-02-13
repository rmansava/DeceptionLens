@echo off
title Books DISK Chunk Builder (Direct)
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   BOOKS DISK CHUNK BUILDER (Direct to Chunks)
echo ================================================================
echo.
echo   Builds 10GB DISK chunks directly from book page images.
echo   Use this for new books instead of the old shard+consolidate pipeline.
echo.
echo   Source:  D:\books\pdf-images
echo   Chunks:  T:\faiss\disk_retrieval\chunks\
echo   IDs:     D:\faiss\disk_retrieval\chunk_ids\
echo.
echo   Loads existing path_lookup.json to continue IDs.
echo   Auto-detects highest chunk number and continues from there.
echo   Resumable via progress file.
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u build_books_disk_chunks.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
