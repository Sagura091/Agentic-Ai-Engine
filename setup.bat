@echo off
REM Windows Batch Setup Script for Agentic AI System
REM This script runs the complete system setup

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   AGENTIC AI SYSTEM SETUP
echo ========================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Python is not installed or not in PATH
    echo   Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   %PYTHON_VERSION%

REM Check if we're in the right directory
if not exist "setup_system.py" (
    echo [ERROR] setup_system.py not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Run the Python setup script
echo.
echo Running setup script...
echo.

python setup_system.py

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% equ 0 (
    echo.
    echo ========================================
    echo   SETUP SUCCESSFUL!
    echo ========================================
    echo.
) else (
    echo.
    echo ========================================
    echo   SETUP COMPLETED WITH WARNINGS
    echo ========================================
    echo.
)

echo Press any key to exit...
pause >nul
exit /b %EXIT_CODE%

