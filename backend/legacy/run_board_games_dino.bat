@echo off
cd /d %~dp0
echo Starting Board Games DINO + Faces Indexer...
echo Source: T:\archiverelated\board games
echo.
.venv\Scripts\python board_games_dino_indexer.py --source "T:\archiverelated\board games"
pause
