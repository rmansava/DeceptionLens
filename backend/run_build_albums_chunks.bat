@echo off
title Building Albums DISK Chunks
echo ============================================
echo  Albums DISK Chunk Builder (Pipelined)
echo  NAS -> Local buffer -> GPU -> Chunks
echo  Chunks:  T:\faiss\disk_retrieval\albums_chunks\
echo  IDs:     D:\faiss\disk_retrieval\albums_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Source: T:\archiverelated\albums (streamed via buffer)
echo Buffer: C:\albums_buffer (5000 folders/batch, 3 batches max)
echo Paths stored as: T:\archiverelated\albums
echo.
cd /d "%~dp0"
python build_albums_disk_chunks.py
echo.
pause
