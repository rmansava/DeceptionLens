# DinoDeceptionLens - Setup Script
# Run this once to set up the development environment

$ErrorActionPreference = "Stop"

Write-Host "DinoDeceptionLens Setup" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan
Write-Host ""

$rootPath = $PSScriptRoot
$backendPath = Join-Path $rootPath "backend"
$webPath = Join-Path $rootPath "web"

# ============================================
# Backend Setup
# ============================================
Write-Host "Setting up Python backend..." -ForegroundColor Yellow
Set-Location $backendPath

# Create virtual environment
$venvPath = Join-Path $backendPath ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Gray
    python -m venv .venv
}

# Activate and install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Gray
& "$venvPath\Scripts\Activate.ps1"

# Check for CUDA
$cudaAvailable = $false
try {
    python -c "import torch; print(torch.cuda.is_available())" 2>$null | Out-Null
    $cudaCheck = python -c "import torch; print(torch.cuda.is_available())"
    if ($cudaCheck -eq "True") {
        $cudaAvailable = $true
    }
} catch {}

if ($cudaAvailable) {
    Write-Host "CUDA detected! GPU acceleration will be available." -ForegroundColor Green
} else {
    Write-Host "CUDA not detected. Using CPU (slower)." -ForegroundColor Yellow
    Write-Host "For GPU support, install PyTorch with CUDA:" -ForegroundColor Gray
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Gray
}

pip install -r requirements.txt

Write-Host "Backend setup complete!" -ForegroundColor Green
Write-Host ""

# ============================================
# Frontend Setup
# ============================================
Write-Host "Setting up .NET frontend..." -ForegroundColor Yellow
Set-Location $webPath

# Check for .NET SDK
$dotnetVersion = $null
try {
    $dotnetVersion = dotnet --version
} catch {
    Write-Host "ERROR: .NET SDK not found. Please install .NET 8.0 SDK." -ForegroundColor Red
    Write-Host "Download from: https://dotnet.microsoft.com/download/dotnet/8.0" -ForegroundColor Yellow
    exit 1
}

Write-Host ".NET SDK version: $dotnetVersion" -ForegroundColor Gray

# Restore packages
Write-Host "Restoring NuGet packages..." -ForegroundColor Gray
dotnet restore

Write-Host "Frontend setup complete!" -ForegroundColor Green
Write-Host ""

# ============================================
# Summary
# ============================================
Set-Location $rootPath

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Index some images:" -ForegroundColor White
Write-Host "   .\index-images.ps1 -Dir 'D:\MyImages' -Collection 'my_images'" -ForegroundColor Gray
Write-Host ""
Write-Host "   For GPU conflict avoidance, use 2-pass indexing:" -ForegroundColor White
Write-Host "   .\index-images.ps1 -Dir 'D:\MyImages' -Mode 'visual_only'" -ForegroundColor Gray
Write-Host "   .\index-images.ps1 -Dir 'D:\MyImages' -Mode 'faces_only'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start the backend server:" -ForegroundColor White
Write-Host "   .\start-backend.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start the frontend (in another terminal):" -ForegroundColor White
Write-Host "   .\start-frontend.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Open your browser to:" -ForegroundColor White
Write-Host "   https://localhost:5001" -ForegroundColor Cyan
Write-Host ""
