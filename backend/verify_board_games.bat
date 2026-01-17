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

REM Check command line args - default is full fix
if "%1"=="--quick" (
    echo Mode: Quick verification only (no file reads)
    python verify_indexing.py --source "%SOURCE%"
) else if "%1"=="--find-duplicates" (
    echo Mode: Find duplicates (no delete, no index)
    python verify_indexing.py --source "%SOURCE%" --find-duplicates
) else (
    echo Mode: FULL FIX - Find duplicates, delete them, index missing
    echo WARNING: This will ask before deleting duplicate files!
    echo.
    python verify_indexing.py --source "%SOURCE%" --find-duplicates --delete-duplicates --index-missing
)

echo.
pause
