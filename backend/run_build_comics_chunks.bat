@echo off
title Building Comics DISK Chunks
echo ============================================
echo  Comics DISK Chunk Builder (Pipelined)
echo  NAS -> Local buffer -> GPU -> Chunks
echo  Chunks:  T:\faiss\disk_retrieval\comics_chunks\
echo  IDs:     D:\faiss\disk_retrieval\comics_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Source: T:\archiverelated\comics (streamed via buffer)
echo Buffer: C:\comics_buffer (200 folders/batch, 3 batches max)
echo Paths stored as: T:\archiverelated\comics
echo.
cd /d "%~dp0"
python build_comics_disk_chunks.py
echo.
pause
