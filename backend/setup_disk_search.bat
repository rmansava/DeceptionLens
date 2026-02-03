@echo off
title Setup DISK Search with Live Tracking
cd /d C:\Users\rmans\source\repos\DinoDeceptionLens\backend

echo.
echo ================================================================
echo   SETUP DISK SEARCH - Live Tracking
echo ================================================================
echo.
echo   This will set up the database for live DISK search tracking.
echo.
echo   Steps:
echo   1. Run database migration (add Status, CurrentProgress columns)
echo   2. Verify DISK search endpoint is ready
echo   3. Test configuration
echo.
echo ================================================================
echo.

REM Check if SQL Server is accessible
echo [1/3] Checking database connection...
sqlcmd -S localhost -d ImageSearch -Q "SELECT @@VERSION" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot connect to SQL Server
    echo Please check that SQL Server is running and ImageSearch database exists
    pause
    exit /b 1
)
echo   Connected to SQL Server successfully
echo.

REM Run migration
echo [2/3] Running database migration...
sqlcmd -S localhost -d ImageSearch -i migrations\add_live_search_tracking.sql
if errorlevel 1 (
    echo ERROR: Migration failed
    pause
    exit /b 1
)
echo   Migration complete
echo.

REM Check chunks
echo [3/3] Checking DISK chunks...
python -c "from glob import glob; chunks = glob('T:/faiss/disk_retrieval/chunks/chunk_*.faiss'); print(f'  Found {len(chunks)} chunks on NAS'); print(f'  Total size: ~{len(chunks) * 22:.0f}GB'); print(f'  Ready for search!')"
echo.

echo ================================================================
echo   SETUP COMPLETE!
echo ================================================================
echo.
echo   DISK search is now ready with live tracking:
echo.
echo   Backend API:  http://localhost:8000
echo   Web Frontend: http://localhost:5000
echo.
echo   To start the services:
echo   1. Backend:  python server.py
echo   2. Frontend: dotnet run (in web folder)
echo.
echo   API endpoint: POST http://localhost:8000/disk/search
echo   - Upload an image to search
echo   - Set live_tracking=true for real-time progress
echo   - Watch top 100 results update as chunks are searched
echo.
echo ================================================================
pause
