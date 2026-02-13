@echo off
echo ============================================
echo CLIP Re-indexing from D:\books
echo ============================================
echo.
echo This will create a new FAISS index at D:\faiss\books_new
echo Estimated time: 2-4 hours for ~2.9M images
echo.

cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

python clip_indexer.py index --root "D:\books" --output "D:\faiss\books_new" --batch-size 64

echo.
echo ============================================
echo Indexing complete!
echo ============================================
echo.
echo Next step: Remap paths from D:\ to T:\
echo Run: python remap_paths.py
echo.
pause
