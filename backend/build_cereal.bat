@echo off
title CEREAL DISK Chunk Builder (Pipelined)
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   CEREAL DISK CHUNK BUILDER
echo ================================================================
echo.
echo   Mode A: C:\cereal exists -> process local directly
echo   Mode B: otherwise NAS -> local SSD staging -> GPU -> chunks
echo.
echo   Local source: C:\cereal              (if present)
echo   NAS source:   T:\archiverelated\cereal (fallback mode)
echo   Buffer:       C:\cereal_buffer       (local SSD staging)
echo   Chunks:  T:\faiss\disk_retrieval\cereal_chunks\   (NAS)
echo   IDs:     D:\faiss\disk_retrieval\cereal_chunk_ids\ (local SSD)
echo.
echo   20 GB folder batches, up to 100 GB buffered
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
