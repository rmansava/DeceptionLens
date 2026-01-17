@echo off
echo ============================================================
echo DISK Feature Indexer - Board Games
echo ============================================================
echo.
echo This will:
echo   1. Extract DISK features for all images in T:\archiverelated\board games
echo   2. Save to D:\disk-features\board_games (local SSD)
echo   3. Auto-move each folder to T:\disk-features\board_games (NAS)
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause >nul

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python batch_disk_index_board_games.py

echo.
echo ============================================================
echo Done! Check batch_disk_board_games.log for details.
echo ============================================================
pause
