#Requires -Version 5.1
<#
.SYNOPSIS
  在 sisinf_challenge2026 目录创建 .venv，并安装 numpy + ortools（与全局 opentelemetry/protobuf 隔离）。

.USAGE
  cd sisinf_challenge2026
  .\setup_venv.ps1

  之后用 .venv 里的 Python 运行求解器：
  .\.venv\Scripts\python.exe scripts\solve_sisinf.py --input saiti1\instances_all.json --output results\out.json
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in $Root\.venv ..."
    py -3 -m venv "$Root\.venv" 2>$null
    if (-not (Test-Path $VenvPython)) {
        python -m venv "$Root\.venv"
    }
}

& $VenvPython -m pip install -U pip
& $VenvPython -m pip install -r "$Root\requirements.txt" -r "$Root\requirements-ortools.txt"

Write-Host "Verifying ortools CP-SAT import..."
& $VenvPython -c "from ortools.sat.python import cp_model; print('ortools OK:', cp_model.__name__)"

Write-Host ""
Write-Host "Done. Activate in PowerShell:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Or run directly:"
Write-Host "  .\.venv\Scripts\python.exe scripts\solve_sisinf.py --help"
