# Jalankan ShelfSense Edge
# .\run.ps1            -> kamera asli
# .\run.ps1 -Synthetic -> kamera palsu (uji UI tanpa webcam)

param(
    [switch]$Synthetic,
    [int]$Port = 8000,
    [int]$Camera = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Virtualenv belum ada. Jalankan dulu:" -ForegroundColor Yellow
    Write-Host "  py -3.11 -m venv .venv"
    Write-Host "  .venv\Scripts\python -m pip install -r requirements.txt"
    exit 1
}

$env:SSE_CAMERA = "$Camera"
$env:SSE_SYNTHETIC = if ($Synthetic) { "1" } else { "0" }

Write-Host "ShelfSense Edge -> http://127.0.0.1:$Port" -ForegroundColor Green
if ($Synthetic) { Write-Host "Mode: kamera sintetis" -ForegroundColor DarkYellow }

& .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
