# Hentikan server ShelfSense Edge yang berjalan di port tertentu.
#   .\stop.ps1          -> port 8000
#   .\stop.ps1 -Port 8080

param([int]$Port = 8000)

$procs = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique

if (-not $procs) {
    Write-Host "Tidak ada server di port $Port." -ForegroundColor DarkYellow
    exit 0
}

foreach ($procId in $procs) {
    $name = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
    Write-Host "Menghentikan PID $procId ($name) di port $Port …" -ForegroundColor Yellow
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 1

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "Gagal menghentikan. Coba jalankan PowerShell sebagai Administrator." -ForegroundColor Red
    exit 1
}

Write-Host "Server di port $Port sudah berhenti." -ForegroundColor Green
