@echo off
title DinoDeceptionLens Web (Hot Reload)
set "ROOT=%~dp0"
cd /d "%ROOT%web"

dotnet watch run --urls http://0.0.0.0:5000

