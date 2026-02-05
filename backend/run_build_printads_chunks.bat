@echo off
title Building Print Ads DISK Chunks
echo ============================================
echo  Print Ads DISK Chunk Builder
echo  908K images -> FAISS chunks + compact IDs
echo  Chunks:  T:\faiss\disk_retrieval\printads_chunks\
echo  IDs:     D:\faiss\disk_retrieval\printads_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Source: C:\printads (local copy, all subfolders)
echo Paths stored as: T:\archiverelated\print ads
echo.
cd /d "%~dp0"
python build_printads_disk_chunks.py
echo.
pause
