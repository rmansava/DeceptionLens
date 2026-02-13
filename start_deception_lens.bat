@echo off
title DinoDeceptionLens Launcher
echo ============================================
echo  DinoDeceptionLens Launcher
echo  Backend: http://localhost:8000 (FastAPI)
echo  Frontend: http://localhost:5000 (Blazor)
echo ============================================
echo.

set "ROOT=%~dp0"

:: Check if backend is already running on port 8000
set BACKEND_RUNNING=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000.*LISTENING" 2^>nul') do (
    set BACKEND_RUNNING=1
)

:: Check if frontend is already running on port 5000
set FRONTEND_RUNNING=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000.*LISTENING" 2^>nul') do (
    set FRONTEND_RUNNING=1
)

if %BACKEND_RUNNING%==1 (
    echo  Backend already running on port 8000
) else (
    echo  Starting backend...
    start "DinoDeceptionLens Backend" cmd /k "cd /d "%ROOT%backend" && call .venv\Scripts\activate.bat && set CHROMA_DB_PATH=%ROOT%backend\chroma_db && python server.py"
    echo  Backend starting in new window
)

if %FRONTEND_RUNNING%==1 (
    echo  Frontend already running on port 5000
) else (
    echo  Starting frontend...
    start "DinoDeceptionLens Frontend" cmd /k "cd /d "%ROOT%web" && dotnet run"
    echo  Frontend starting in new window
)

echo.
echo  Both services launching. Close this window anytime.
echo  Backend: http://localhost:8000
echo  Frontend: http://localhost:5000
echo.
pause
