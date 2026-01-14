@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe verify_disk_coverage.py --fix
pause
