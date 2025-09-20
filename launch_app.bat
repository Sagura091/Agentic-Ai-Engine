@echo off
REM Revolutionary AI Agent Builder - Quick Launcher
REM Launches both frontend and backend in separate PowerShell windows

echo.
echo ========================================
echo 🚀 Revolutionary AI Agent Builder
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js 18+ and try again
    pause
    exit /b 1
)

echo ✅ Python and Node.js found!
echo.

REM Launch the Python launcher
echo 🔥 Starting application launcher...
python launch_app.py

echo.
echo 👋 Application launcher finished
pause
