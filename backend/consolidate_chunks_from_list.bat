@echo off
title DISK Chunk Consolidation (From Unprocessed List)
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   DISK CHUNK CONSOLIDATION - Process Unprocessed Books
echo ================================================================
echo.
echo   This will process the 2,213 unprocessed books found by
echo   find_unprocessed_books.py
echo.
echo   Book list: D:\faiss\disk_retrieval\unprocessed_books.txt
echo.
echo   Source:      T:\faiss\disk_retrieval\books\     (NAS, 13TB)
echo   Local buf:   D:\faiss\disk_retrieval\books\     (SSD, ~200GB buffer)
echo   Output:      T:\faiss\disk_retrieval\chunks\    (NAS)
echo.
echo   Estimated time: ~48-72 hours for 2,213 books
echo   Will create ~50-60 new chunks (~1.1TB)
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u consolidate_search_chunks.py --books-file "D:\faiss\disk_retrieval\unprocessed_books.txt"

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
