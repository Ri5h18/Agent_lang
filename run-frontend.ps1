$ErrorActionPreference = "Stop"

$frontendPath = Join-Path $PSScriptRoot "frontend"
if (-not (Test-Path -LiteralPath (Join-Path $frontendPath "node_modules"))) {
    throw "Frontend dependencies are missing. Run .\setup.ps1 first."
}

Push-Location $frontendPath
try {
    & npm run dev
}
finally {
    Pop-Location
}

