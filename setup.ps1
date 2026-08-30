$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$venvPath = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

Write-Host "[1/4] Preparing Python environment" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $pythonPath)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $venvPath
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $venvPath
    }
    else {
        throw "Python 3.11+ was not found. Install Python, then run setup.ps1 again."
    }
}

Write-Host "[2/4] Installing backend dependencies" -ForegroundColor Cyan
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -r (Join-Path $projectRoot "backend\requirements.txt")

Write-Host "[3/4] Installing frontend dependencies" -ForegroundColor Cyan
Push-Location (Join-Path $projectRoot "frontend")
try {
    & npm install
    Write-Host "[4/4] Building the production site" -ForegroundColor Cyan
    & npm run build
}
finally {
    Pop-Location
}

Write-Host "" 
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run .\run-site.ps1 and open http://127.0.0.1:8000" -ForegroundColor Green

