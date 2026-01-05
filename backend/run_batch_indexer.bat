@echo off
title DeceptionLens Batch Indexer
cd /d "%~dp0"

echo ============================================================
echo   DeceptionLens Batch Indexer
echo ============================================================
echo.
echo Starting batch indexing of D:\books\pdf-images
echo Progress is saved to batch_progress.txt
echo.
echo Press Ctrl+C to stop (progress will be saved)
echo ============================================================
echo.

python batch_index.py

echo.
echo ============================================================
echo Batch indexing complete!
echo ============================================================
pause
