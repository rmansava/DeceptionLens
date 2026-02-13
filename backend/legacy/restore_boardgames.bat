@echo off
echo ========================================
echo Restore Board Games Indexes from NAS
echo ========================================
echo.
echo This will restore from:
echo   - T:\opensearch-dino-boardgames -^> dinov2-board_games
echo   - T:\opensearch-faces-boardgames -^> faces-board_games
echo.
echo Prerequisites:
echo   1. D:\opensearch-boardgames must exist
echo   2. Docker must have volume mount: -v D:\opensearch-boardgames:/boardgames-snapshots
echo.

cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

python restore_boardgames_indexes.py

echo.
pause
