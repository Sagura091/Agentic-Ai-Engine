@echo off
cd /d "C:\Users\Bab18\Desktop\Agents"
echo Starting Agentic AI Engine Backend...
echo Directory: C:\Users\Bab18\Desktop\Agents
echo.

REM Check and install dependencies if needed
if exist "requirements.txt" (
    echo Checking dependencies...
    python -m pip install -r requirements.txt --quiet >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Some dependencies may have issues, continuing...
    ) else (
        echo Dependencies ready!
    )
)

echo.
echo Backend Server: http://localhost:8888
echo API Documentation: http://localhost:8888/docs
echo Interactive API: http://localhost:8888/redoc
echo Health Check: http://localhost:8888/health
echo.
echo Starting server with hot reload...
echo.

REM Start the backend server with graceful reload handling
python -m uvicorn app.main:socketio_app --host localhost --port 8888 --reload --log-level warning --no-access-log --reload-delay 2 --timeout-graceful-shutdown 10
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start backend server!
    echo TIP: Make sure you have all dependencies installed:
    echo    pip install -r requirements.txt
    echo.
    echo TIP: Or try running manually:
    echo    python -m uvicorn app.main:socketio_app --host localhost --port 8888 --reload
)

echo.
echo Backend server stopped. Press any key to close...
pause >nul
