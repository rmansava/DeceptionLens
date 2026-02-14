@echo off
title Consolidate Book Chunks (T: + S: -> T:/chunks)
cd /d "%~dp0"
python legacy/consolidate_search_chunks.py
echo.
echo Done! Press any key to close...
pause >nul
