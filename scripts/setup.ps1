$ErrorActionPreference = 'Stop'

Write-Host "=== Sky Observer Local Setup (Windows) ==="
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
}

$hasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasNvidia) {
  Write-Host "NVIDIA GPU detected; building and starting with GPU compose override..."
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
} else {
  Write-Host "No NVIDIA GPU detected; building and starting with CPU fallback..."
  docker compose build
  docker compose up -d
}

Write-Host "Sky Observer dashboard available at: http://localhost:8080"
