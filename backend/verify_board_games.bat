@echo off
REM Verify Board Games Indexing and Find Duplicates
REM Usage: verify_board_games.bat [--find-duplicates] [--delete-duplicates]

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

set SOURCE=T:\archiverelated\board games

echo.
echo ============================================
echo Board Games Indexing Verification
echo ============================================
echo Source: %SOURCE%
echo.

REM Check command line args
if "%1"=="--delete-duplicates" (
    echo Mode: FIND AND DELETE DUPLICATES
    echo WARNING: This will permanently delete duplicate files!
    echo.
    python verify_indexing.py --source "%SOURCE%" --find-duplicates --delete-duplicates
) else if "%1"=="--find-duplicates" (
    echo Mode: Find duplicates (no delete)
    python verify_indexing.py --source "%SOURCE%" --find-duplicates
) else (
    echo Mode: Quick verification (no file reads)
    echo.
    echo To find duplicates:        verify_board_games.bat --find-duplicates
    echo To delete duplicates:      verify_board_games.bat --delete-duplicates
    echo.
    python verify_indexing.py --source "%SOURCE%"
)

echo.
pause
