$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontendIndex = Join-Path $projectRoot "frontend\dist\index.html"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project environment is missing. Run .\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $frontendIndex)) {
    throw "Frontend build is missing. Run .\setup.ps1 first."
}

Write-Host "Agentic Lang Studio" -ForegroundColor Green
Write-Host "Site: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "API docs: http://127.0.0.1:8000/docs" -ForegroundColor DarkGray
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray

Push-Location (Join-Path $projectRoot "backend")
try {
    & $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}

