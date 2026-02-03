@echo off
title Get Chunked Books List
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   GET CHUNKED BOOKS LIST
echo ================================================================
echo.
echo   Scans all chunk paths.json files to find unique books
echo   Faster than build_chunk_index.bat (only counts, no mapping)
echo.
echo   Source: T:\faiss\disk_retrieval\chunks\
echo   Output: D:\faiss\disk_retrieval\chunked_books.txt
echo.
echo   Takes ~20-30 min
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u get_chunked_books.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
