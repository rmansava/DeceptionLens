@echo off
title DinoDeceptionLens Backend (Auto-Reload)
set "ROOT=%~dp0"
cd /d "%ROOT%backend"

if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

set CHROMA_DB_PATH=%ROOT%backend\chroma_db
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

