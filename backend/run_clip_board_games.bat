@echo off
echo ============================================================
echo CLIP Indexer - Board Games
echo ============================================================
echo.
echo This will:
echo   1. Load CLIP model (ViT-L/14)
echo   2. Encode all images in T:\archiverelated\board games
echo   3. Build FAISS index at D:\faiss\board_games
echo.
echo This may take several hours for ~878k images.
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause >nul

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Create output directory
if not exist "D:\faiss\board_games" mkdir "D:\faiss\board_games"

python clip_indexer.py index --root "T:\archiverelated\board games" --output "D:\faiss\board_games" --batch-size 64

echo.
echo ============================================================
echo Done! Index saved to D:\faiss\board_games
echo ============================================================
pause
