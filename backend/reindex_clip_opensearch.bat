@echo off
echo ============================================
echo CLIP Direct to OpenSearch Indexer
echo ============================================
echo.
echo Source: D:\books
echo Target: OpenSearch index "clip-books"
echo Path remap: D:\books -> T:\archiverelated\books\pdf-images
echo.
echo Checkpoint: clip-books_checkpoint.json (auto-resume on restart)
echo.

cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

python clip_opensearch_indexer.py --source "D:\books" --index "clip-books" --remap-from "D:\books" --remap-to "T:\archiverelated\books\pdf-images" --batch-size 64

echo.
echo ============================================
echo Done!
echo ============================================
pause
