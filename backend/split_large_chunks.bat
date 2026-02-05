@echo off
title Split Large Chunks
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   SPLIT LARGE CHUNKS
echo ================================================================
echo.
echo   Splits oversized chunks (^>30GB) into ~20GB pieces.
echo   Run this after consolidation completes.
echo.
echo   Source: T:\faiss\disk_retrieval\chunks\
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u split_large_chunks.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
