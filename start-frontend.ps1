# DinoDeceptionLens - Start Frontend
# Run this script to start the Blazor web application

$ErrorActionPreference = "Stop"

Write-Host "DinoDeceptionLens Frontend" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

# Navigate to web directory
$webPath = Join-Path $PSScriptRoot "web"
Set-Location $webPath

Write-Host ""
Write-Host "Starting Blazor Server..." -ForegroundColor Green
Write-Host "Access at: https://localhost:5001 or http://localhost:5000" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

dotnet run
