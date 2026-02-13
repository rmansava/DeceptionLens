@echo off
title Converting paths.json to compact IDs
echo ============================================
echo  Paths to IDs Conversion
echo  3.3 TB paths.json -^> ~95 GB compact IDs
echo  Output: D:\faiss\disk_retrieval\chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
cd /d "%~dp0"
python convert_paths_to_ids.py
echo.
pause
