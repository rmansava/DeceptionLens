# DinoDeceptionLens - Start Backend Server
# Run this script to start the FastAPI backend

$ErrorActionPreference = "Stop"

Write-Host "DinoDeceptionLens Backend Server" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Navigate to backend directory
$backendPath = Join-Path $PSScriptRoot "backend"
Set-Location $backendPath

# Check for virtual environment
$venvPath = Join-Path $backendPath ".venv"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "$venvPath\Scripts\Activate.ps1"
} else {
    Write-Host "No virtual environment found. Using system Python." -ForegroundColor Yellow
    Write-Host "Consider creating one with: python -m venv .venv" -ForegroundColor Gray
}

# Set environment variables
$env:CHROMA_DB_PATH = Join-Path $backendPath "chroma_db"
Write-Host "Database path: $env:CHROMA_DB_PATH" -ForegroundColor Gray

# Optionally disable GPU for stable server operation
# Uncomment if you have GPU conflicts:
# $env:CUDA_VISIBLE_DEVICES = "-1"

Write-Host ""
Write-Host "Starting server on http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

python server.py
