@echo off
echo ========================================
echo Creating Snapshots of Board Games Indexes
echo ========================================
echo.
echo This will snapshot:
echo   - dinov2-board_games -^> T:\opensearch-dino-boardgames
echo   - faces-board_games  -^> T:\opensearch-faces-boardgames
echo.

cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

python snapshot_boardgames_indexes.py

echo.
pause
