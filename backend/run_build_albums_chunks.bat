@echo off
title Building Albums DISK Chunks
echo ============================================
echo  Albums DISK Chunk Builder
echo  Images -> FAISS chunks + compact IDs
echo  Chunks:  S:\faiss\disk_retrieval\albums_chunks\
echo  IDs:     D:\faiss\disk_retrieval\albums_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Source: C:\albums (local copy, all subfolders)
echo Paths stored as: T:\albums
echo.
echo NOTE: Copy albums from NAS first if not already done:
echo   robocopy "T:\albums" "C:\albums" /E /R:2 /W:5 /XF *.txt
echo.
cd /d "%~dp0"
python build_albums_disk_chunks.py
echo.
pause
