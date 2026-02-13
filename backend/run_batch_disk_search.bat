@echo off
title Batch DISK Search
echo ============================================
echo  Batch DISK Keypoint Search
echo  Search a directory of images against all
echo  DISK chunks. Results appear in web UI
echo  search history as they're found.
echo ============================================
echo.
echo Enter the directory path containing query images:
set /p SEARCH_DIR="> "
echo.
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend
python -u batch_disk_search.py "%SEARCH_DIR%"
echo.
pause
