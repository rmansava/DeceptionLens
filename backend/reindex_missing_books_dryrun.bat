@echo off
echo ========================================
echo Re-indexing Missing Books (DRY RUN)
echo ========================================
echo.

cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo Preview of what will be indexed:
echo.

python reindex_missing_books.py --dry-run

echo.
pause
