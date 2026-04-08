@echo off
title Books DISK Chunk Builder (Direct)
setlocal EnableDelayedExpansion

set "CUDA_POISON_EXIT_CODE=86"
set "MAX_RESTARTS=20"
set "RESTART_DELAY_SEC=8"
set "DISK_MAX_VECTORS_PER_CHUNK=19500000"
set "DISK_MAX_IMAGE_DIM=2048"
set /a RESTART_COUNT=0

cd /d "%~dp0"

echo.
echo ================================================================
echo   BOOKS DISK CHUNK BUILDER (Direct to Chunks)
echo ================================================================
echo.
echo   Builds 10GB DISK chunks directly from book page images.
echo   Use this for new books instead of the old shard+consolidate pipeline.
echo.
echo   Source:  D:\books\pdf-images
echo   Chunks:  T:\faiss\disk_retrieval\chunks\
echo   IDs:     D:\faiss\disk_retrieval\chunk_ids\
echo   Chunk cap: %DISK_MAX_VECTORS_PER_CHUNK% vectors (~10 GB FAISS payload)
echo   Resize mode: images over %DISK_MAX_IMAGE_DIM% px are downscaled (aspect ratio preserved)
echo.
echo   Loads existing path_lookup.json to continue IDs.
echo   Auto-detects highest chunk number and continues from there.
echo   Resumable via progress file.
echo.
echo ----------------------------------------------------------------
echo   Press any key to start, or Ctrl+C to cancel...
pause > nul

set "DISK_CUDA_POISON_EXIT_CODE=%CUDA_POISON_EXIT_CODE%"

:run_build
python -u build_books_disk_chunks.py
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
echo ================================================================
echo   Done! Press any key to exit...
pause > nul
endlocal
