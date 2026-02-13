@echo off
title Build Chunk Index
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   BUILD CHUNK INDEX
echo ================================================================
echo.
echo   Scans all chunk paths.json files to create:
echo     - book_to_chunks.json (which chunks contain each book)
echo     - chunk_to_books.json (which books are in each chunk)
echo.
echo   Source: T:\faiss\disk_retrieval\chunks\
echo   Output: D:\faiss\disk_retrieval\
echo.
echo   This takes ~30-60 min due to NAS reads.
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u build_chunk_index.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
