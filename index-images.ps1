# DinoDeceptionLens - Index Images
# Usage: .\index-images.ps1 -Dir "D:\MyImages" -Collection "my_collection" -Mode "all"

param(
    [Parameter(Mandatory=$true)]
    [string]$Dir,

    [string]$Collection = "images",

    [ValidateSet("all", "visual_only", "faces_only")]
    [string]$Mode = "all",

    [switch]$Reset,

    [string]$MapSource,
    [string]$MapTarget
)

$ErrorActionPreference = "Stop"

Write-Host "DinoDeceptionLens Indexer" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# Navigate to backend directory
$backendPath = Join-Path $PSScriptRoot "backend"
Set-Location $backendPath

# Check for virtual environment
$venvPath = Join-Path $backendPath ".venv"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "$venvPath\Scripts\Activate.ps1"
}

Write-Host "Directory: $Dir" -ForegroundColor Gray
Write-Host "Collection: $Collection" -ForegroundColor Gray
Write-Host "Mode: $Mode" -ForegroundColor Gray

$args = @("main.py", "index", "--dir", $Dir, "--collection", $Collection, "--mode", $Mode)

if ($Reset) {
    $args += "--reset"
    Write-Host "Reset: Yes" -ForegroundColor Yellow
}

if ($MapSource -and $MapTarget) {
    $args += "--map-source"
    $args += $MapSource
    $args += "--map-target"
    $args += $MapTarget
    Write-Host "Path Mapping: $MapSource -> $MapTarget" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Starting indexing..." -ForegroundColor Green
Write-Host ""

& python $args
