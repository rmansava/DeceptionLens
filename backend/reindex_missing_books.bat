@echo off
echo ========================================
echo Re-indexing Missing Books
echo ========================================
echo.

cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo Starting re-index...
echo.

python reindex_missing_books.py

echo.
echo ========================================
echo Done!
echo ========================================
pause
