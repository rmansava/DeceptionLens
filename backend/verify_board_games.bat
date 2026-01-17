@echo off
REM Verify Board Games Indexing and Find/Fix Issues
REM Usage: verify_board_games.bat [--find-duplicates] [--delete-duplicates] [--fix]

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
if "%1"=="--fix" (
    echo Mode: FULL FIX - Find duplicates, delete them, index missing
    echo WARNING: This will permanently delete duplicate files!
    echo.
    python verify_indexing.py --source "%SOURCE%" --find-duplicates --delete-duplicates --index-missing
) else if "%1"=="--delete-duplicates" (
    echo Mode: FIND AND DELETE DUPLICATES
    echo WARNING: This will permanently delete duplicate files!
    echo.
    python verify_indexing.py --source "%SOURCE%" --find-duplicates --delete-duplicates
) else if "%1"=="--index-missing" (
    echo Mode: Find duplicates and index truly missing files
    python verify_indexing.py --source "%SOURCE%" --find-duplicates --index-missing
) else if "%1"=="--find-duplicates" (
    echo Mode: Find duplicates (no delete, no index)
    python verify_indexing.py --source "%SOURCE%" --find-duplicates
) else (
    echo Mode: Quick verification (no file reads)
    echo.
    echo Options:
    echo   --find-duplicates    Check if missing files are duplicates
    echo   --delete-duplicates  Delete duplicate files
    echo   --index-missing      Index truly missing files after verification
    echo   --fix                Full fix: delete duplicates + index missing
    echo.
    python verify_indexing.py --source "%SOURCE%"
)

echo.
pause
