@echo off
title Building Print Ads DISK Chunks
setlocal EnableDelayedExpansion

set "CUDA_POISON_EXIT_CODE=86"
set "MAX_RESTARTS=20"
set "RESTART_DELAY_SEC=8"
set "DISK_MAX_VECTORS_PER_CHUNK=19500000"
set "DISK_COPY_THREADS=32"
set "DISK_CLEAN_BUFFER_ON_START=0"
set "DISK_MAX_IMAGE_DIM=2048"
set "DISK_SKIP_IF_MAX_DIM_OVER=0"
set /a RESTART_COUNT=0

echo ============================================
echo  Print Ads DISK Chunk Builder
echo  Mode A: C:\printads or C:\print ads exists -> process local directly
echo  Mode B: otherwise NAS -> Local Buffer -> GPU -> Chunks
echo  Chunks:  S:\faiss\disk_retrieval\printads_chunks\
echo  IDs:     D:\faiss\disk_retrieval\printads_chunk_ids\
echo  Resumable - safe to stop and restart
echo ============================================
echo.
echo Local source: C:\printads or C:\print ads (if present)
echo NAS source:   T:\archiverelated\print ads (fallback mode)
echo Buffer: C:\printads_buffer (20 GB/batch, up to 100 GB buffered)
echo Copy threads: %DISK_COPY_THREADS%  (robocopy MT)
echo Resize mode: images over %DISK_MAX_IMAGE_DIM% px are downscaled (aspect ratio preserved)
echo Paths stored as: T:\archiverelated\print ads
echo Chunk cap: %DISK_MAX_VECTORS_PER_CHUNK% vectors (~10 GB FAISS payload)
echo.
cd /d "%~dp0"
set "DISK_CUDA_POISON_EXIT_CODE=%CUDA_POISON_EXIT_CODE%"

:run_build
python -u build_printads_disk_chunks.py
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
