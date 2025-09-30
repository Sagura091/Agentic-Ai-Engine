# PowerShell Setup Script for Agentic AI System
# This script runs the complete system setup

param(
    [switch]$SkipDocker,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AGENTIC AI SYSTEM SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Blue
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.11+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "setup_system.py")) {
    Write-Host "Error: setup_system.py not found" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    exit 1
}

# Run the Python setup script
Write-Host ""
Write-Host "Running setup script..." -ForegroundColor Blue
Write-Host ""

try {
    if ($Verbose) {
        python setup_system.py
    } else {
        python setup_system.py
    }
    
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  SETUP SUCCESSFUL!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host "  SETUP COMPLETED WITH WARNINGS" -ForegroundColor Yellow
        Write-Host "========================================" -ForegroundColor Yellow
        Write-Host ""
    }
    
    exit $exitCode
    
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  SETUP FAILED" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Make sure Docker Desktop is running" -ForegroundColor White
    Write-Host "  2. Check that port 5432 is not in use" -ForegroundColor White
    Write-Host "  3. Verify Python dependencies are installed:" -ForegroundColor White
    Write-Host "     pip install -r requirements.txt" -ForegroundColor White
    Write-Host ""
    exit 1
}

