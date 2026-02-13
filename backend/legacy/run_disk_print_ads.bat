@echo off
echo ============================================================
echo DISK Feature Indexer - Print Ads
echo ============================================================
echo.
echo This will:
echo   1. Copy folders from T:\archiverelated\print ads\ebay to C:\print ads\ebay
echo   2. Extract DISK features (CPU)
echo   3. Save .npz to C:\disk-features\print_ads
echo   4. Move .npz files to T:\disk-features\print_ads (NAS)
echo   5. Delete processed source images
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause >nul

cd /d "%~dp0"
.venv\Scripts\python.exe batch_disk_index_print_ads.py

echo.
echo ============================================================
echo Done! Check batch_disk_print_ads.log for details.
echo ============================================================
pause
