@echo off
echo ============================================
echo Fast DINOv2 Batch Indexer (GPU)
echo ============================================
echo.
echo This indexer:
echo   - Loads DINOv2 model ONCE (not per book)
echo   - Processes 16 images at a time on GPU
echo   - Skips books from batch_progress_opensearch.txt
echo   - Skips books from batch_progress_fast.txt
echo.
echo Press Ctrl+C to stop at any time.
echo ============================================
echo.

cd /d "%~dp0"
python batch_index_fast.py

echo.
echo ============================================
echo Batch indexing finished!
pause
