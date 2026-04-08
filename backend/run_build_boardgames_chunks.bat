@echo off
title Building Board Games DISK Chunks
setlocal EnableDelayedExpansion

set "CUDA_POISON_EXIT_CODE=86"
set "MAX_RESTARTS=20"
set "RESTART_DELAY_SEC=8"
set "DISK_MAX_VECTORS_PER_CHUNK=19500000"
set "DISK_MAX_IMAGE_DIM=2048"
set /a RESTART_COUNT=0

echo ============================================
echo  Board Games DISK Chunk Builder
echo  Mode A: C:\boardgames exists -> process local directly
echo  Mode B: otherwise NAS -> Local Temp Buffer -> GPU -> Chunks
echo  Chunks:  T:\faiss\disk_retrieval\boardgames_chunks\
echo  IDs:     D:\faiss\disk_retrieval\boardgames_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Local source: C:\boardgames (if present)
echo NAS source:   T:\archiverelated\board games (fallback mode)
echo Buffer: C:\boardgames-temp (20 GB per batch, up to 100 GB buffered)
echo Resize mode: images over %DISK_MAX_IMAGE_DIM% px are downscaled (aspect ratio preserved)
echo Paths stored as: T:\archiverelated\board games
echo Chunk cap: %DISK_MAX_VECTORS_PER_CHUNK% vectors (~10 GB FAISS payload)
echo.
cd /d "%~dp0"
set "DISK_CUDA_POISON_EXIT_CODE=%CUDA_POISON_EXIT_CODE%"

:run_build
python -u build_boardgames_disk_chunks.py
set "EXITCODE=%ERRORLEVEL%"

if "!EXITCODE!"=="%CUDA_POISON_EXIT_CODE%" (
  set /a RESTART_COUNT+=1
  echo.
  echo CUDA poison exit detected ^(code %CUDA_POISON_EXIT_CODE%^). Auto-restart !RESTART_COUNT!/%MAX_RESTARTS%...
  if !RESTART_COUNT! GEQ %MAX_RESTARTS% (
    echo Reached max auto-restarts. Stopping.
    goto :done
  )
  timeout /t %RESTART_DELAY_SEC% /nobreak >nul
  goto :run_build
)

:done
echo.
pause
endlocal
