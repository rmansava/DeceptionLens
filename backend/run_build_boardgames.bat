@echo off
title Build Board Games DISK Chunks (Full Resolution)
cd /d "%~dp0"
python build_boardgames_disk_chunks.py
echo.
echo Done! Press any key to close...
pause >nul
