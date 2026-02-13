@echo off
title Repair Path Lookup
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   REPAIR PATH_LOOKUP.JSON
echo ================================================================
echo.
echo   Fixes the ID mapping after consolidation resume bug.
echo   Reads per-book shards from T: and S: to reconstruct.
echo   Rewrites chunk_*_ids.npy and path_lookup.json.
echo   FAISS chunks are NOT modified.
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

python -u repair_path_lookup.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
