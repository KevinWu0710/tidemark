# tidemark — build + smoke on Docker Desktop
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .env)) {
    Write-Host "缺少 .env — 請 copy .env.example .env 並填入金鑰"
    exit 1
}
docker compose build
docker compose --profile tools run --rm smoke
