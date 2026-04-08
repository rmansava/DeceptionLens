@echo off
title Building Comics DISK Chunks
setlocal EnableDelayedExpansion

set "CUDA_POISON_EXIT_CODE=86"
set "MAX_RESTARTS=20"
set "RESTART_DELAY_SEC=8"
set "DISK_MAX_VECTORS_PER_CHUNK=18000000"
set "DISK_MAX_IMAGE_DIM=2048"
set /a RESTART_COUNT=0

echo ============================================
echo  Comics DISK Chunk Builder (Pipelined)
echo  NAS -> Local buffer -> GPU -> Chunks
echo  Chunks:  T:\faiss\disk_retrieval\comics_chunks\
echo  IDs:     D:\faiss\disk_retrieval\comics_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Source: T:\archiverelated\comics (streamed via buffer)
echo Buffer: C:\comics_buffer (20 GB/batch, up to 100 GB buffered)
echo Resize mode: images over %DISK_MAX_IMAGE_DIM% px are downscaled (aspect ratio preserved)
echo Paths stored as: T:\archiverelated\comics
echo Chunk cap: %DISK_MAX_VECTORS_PER_CHUNK% vectors (~10 GB FAISS payload)
echo.
cd /d "%~dp0"
set "DISK_CUDA_POISON_EXIT_CODE=%CUDA_POISON_EXIT_CODE%"

:run_build
python -u build_comics_disk_chunks.py
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
