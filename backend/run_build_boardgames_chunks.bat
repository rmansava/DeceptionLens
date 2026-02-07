@echo off
title Building Board Games DISK Chunks
echo ============================================
echo  Board Games DISK Chunk Builder
echo  Images -> FAISS chunks + compact IDs
echo  Chunks:  S:\faiss\disk_retrieval\boardgames_chunks\
echo  IDs:     D:\faiss\disk_retrieval\boardgames_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Source: C:\boardgames (local copy, all subfolders)
echo Paths stored as: T:\archiverelated\board games
echo.
echo NOTE: Copy board games from NAS first if not already done:
echo   robocopy "T:\archiverelated\board games" "C:\boardgames" /E /R:2 /W:5
echo.
cd /d "%~dp0"
python build_boardgames_disk_chunks.py
echo.
pause
