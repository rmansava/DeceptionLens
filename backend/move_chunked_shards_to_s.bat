@echo off
title Move Chunked Book Shards T: -> S:
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   MOVE CHUNKED BOOK SHARDS FROM T: TO S:
echo ================================================================
echo.
echo   Moves per-book FAISS shards that are already in 10GB chunks.
echo   Only moves books confirmed in consolidation_state.json.
echo   Verifies each copy before deleting from T:.
echo.
echo   Source: T:\faiss\disk_retrieval\books\
echo   Dest:   S:\faiss\disk_retrieval\books\
echo.
echo   Expected to free ~8 TB on T:
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u move_chunked_shards_to_s.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
