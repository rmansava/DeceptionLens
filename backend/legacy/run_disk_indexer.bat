@echo off
echo ============================================================
echo DISK Feature Indexer - File-Based (NAS)
echo ============================================================
echo.
echo This will:
echo   1. Index DISK features for all books in D:\books\pdf-images
echo   2. Save to D:\disk-features\books (local SSD)
echo   3. Auto-move each book to T:\disk-features\books (NAS)
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause >nul

cd /d "%~dp0"
.venv\Scripts\python.exe batch_disk_index_file.py

echo.
echo ============================================================
echo Done! Check batch_disk_index_file.log for details.
echo ============================================================
pause
