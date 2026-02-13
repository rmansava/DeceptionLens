@echo off
title DISK Retrieval Index Builder
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   DISK RETRIEVAL INDEX BUILDER
echo ================================================================
echo.
echo   This builds a searchable index from your existing DISK features.
echo   Enables finding images by keypoint matching.
echo.
echo   Process:
echo     1. Copy 50 books from NAS (T:) to local buffer (C:)
echo     2. Process locally (fast SSD reads)
echo     3. Save checkpoint to NAS every 5 batches
echo     4. Clear buffer and repeat
echo.
echo   Output: D:\faiss\disk_retrieval\books\ (with backup on T:)
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u build_disk_retrieval_index.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
