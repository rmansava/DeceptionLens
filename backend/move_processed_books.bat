@echo off
title Move Processed Books
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   MOVE PROCESSED BOOKS
echo ================================================================
echo.
echo   This will MOVE 4,743 confirmed processed books from:
echo     T:\faiss\disk_retrieval\books\
echo   to:
echo     \\disk80\backup\ds923 backup\faiss\disk_retrieval\books\
echo.
echo   Frees up ~9TB on T: drive
echo   Takes ~24-36 hours
echo.
echo   Books will be DELETED from T: drive after successful copy
echo.
echo ================================================================

python -u move_processed_books.py

echo.
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
