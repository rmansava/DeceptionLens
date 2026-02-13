@echo off
title CEREAL DISK Chunk Builder
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   CEREAL DISK CHUNK BUILDER
echo ================================================================
echo.
echo   Source:   C:\cereal                (local SSD, 100K images)
echo   Chunks:  T:\faiss\disk_retrieval\cereal_chunks\    (NAS 10GbE)
echo   IDs:     D:\faiss\disk_retrieval\cereal_chunk_ids\ (local SSD)
echo.
echo   ~10 GB per chunk, GPU DISK feature extraction
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u build_cereal_disk_chunks.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
