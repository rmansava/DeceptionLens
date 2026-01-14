@echo off
echo ========================================
echo Creating Snapshots of Books Indexes
echo ========================================
echo.
echo This will snapshot:
echo   - dinov2-books -^> T:\opensearch-dino-books
echo   - faces-books  -^> T:\opensearch-faces-books
echo.

cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

python snapshot_books_indexes.py

echo.
pause
