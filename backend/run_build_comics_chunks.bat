@echo off
title Building Comics DISK Chunks
echo ============================================
echo  Comics DISK Chunk Builder
echo  Images -> FAISS chunks + compact IDs
echo  Chunks:  S:\faiss\disk_retrieval\comics_chunks\
echo  IDs:     D:\faiss\disk_retrieval\comics_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Source: C:\comics (local copy, all subfolders)
echo Paths stored as: T:\comics
echo.
echo NOTE: Copy comics from NAS first if not already done:
echo   robocopy "T:\comics" "C:\comics" /E /R:2 /W:5
echo.
cd /d "%~dp0"
python build_comics_disk_chunks.py
echo.
pause
