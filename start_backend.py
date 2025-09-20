#!/usr/bin/env python3
"""
Backend Launcher - Agentic AI Engine
Simple script to launch just the backend server on localhost:8888
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

class BackendLauncher:
    def __init__(self):
        self.root_dir = Path(__file__).parent.absolute()
        self.app_dir = self.root_dir / "app"
        self.requirements_file = self.root_dir / "requirements.txt"
        
    def check_backend_files(self):
        """Check if required backend files exist"""
        print("🔍 Checking backend files...")
        
        if not self.app_dir.exists():
            print(f"❌ App directory not found: {self.app_dir}")
            return False
            
        main_file = self.app_dir / "main.py"
        if not main_file.exists():
            print(f"❌ Backend main.py not found: {main_file}")
            return False
            
        print("✅ Backend files found!")
        return True
        
    def install_dependencies(self):
        """Install backend dependencies if requirements.txt exists"""
        if not self.requirements_file.exists():
            print("ℹ️  No requirements.txt found, skipping dependency installation")
            return True
            
        print("📦 Installing backend dependencies...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                print(f"⚠️  Some dependencies may have issues:")
                print(result.stderr)
                print("🔄 Continuing anyway...")
            else:
                print("✅ Dependencies installed successfully!")
                
        except subprocess.TimeoutExpired:
            print("⚠️  Dependency installation timed out, continuing anyway...")
        except Exception as e:
            print(f"⚠️  Error installing dependencies: {e}")
            print("🔄 Continuing anyway...")
            
        return True
        
    def launch_backend(self):
        """Launch backend in a new terminal window"""
        print("🚀 Launching backend server...")

        # Create a batch file to run the backend
        batch_content = f"""@echo off
cd /d "{self.root_dir}"
echo Starting Agentic AI Engine Backend...
echo Directory: {self.root_dir}
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
"""

        # Write batch file
        batch_file = self.root_dir / "start_backend_temp.bat"
        try:
            with open(batch_file, 'w') as f:
                f.write(batch_content)

            # Launch new command prompt window with the batch file
            process = subprocess.Popen([
                "cmd.exe", "/c", "start", "cmd.exe", "/k", str(batch_file)
            ], shell=True, cwd=self.root_dir)

            print("✅ Backend launched in new terminal window!")
            print("")
            print("🎉 Backend Server Starting!")
            print("=" * 50)
            print("🔥 Backend Server: http://localhost:8888")
            print("📚 API Documentation: http://localhost:8888/docs")
            print("🔧 Interactive API: http://localhost:8888/redoc")
            print("❤️  Health Check: http://localhost:8888/health")
            print("=" * 50)
            print("")
            print("💡 The backend is running in the new terminal window.")
            print("💡 Close the terminal window to stop the server.")
            print("💡 Press Ctrl+C in the terminal window for graceful shutdown.")

            # Clean up batch file after a delay
            def cleanup_batch_file():
                time.sleep(5)  # Wait 5 seconds before cleanup
                try:
                    if batch_file.exists():
                        batch_file.unlink()
                except:
                    pass  # Ignore cleanup errors

            cleanup_thread = threading.Thread(target=cleanup_batch_file)
            cleanup_thread.daemon = True
            cleanup_thread.start()

            return True

        except Exception as e:
            print(f"❌ Failed to launch backend: {e}")
            # Clean up batch file on error
            try:
                if batch_file.exists():
                    batch_file.unlink()
            except:
                pass
            return False
            
    def run(self):
        """Main run method"""
        print("🚀 Agentic AI Engine - Backend Launcher")
        print("=" * 50)
        
        # Check backend files
        if not self.check_backend_files():
            print("❌ Backend file check failed. Please fix the issues above.")
            return False
            
        # Install dependencies
        if not self.install_dependencies():
            print("❌ Dependency installation failed.")
            return False
            
        print("")
        print("🎯 Launching backend server...")
        print("-" * 30)
        
        # Launch backend
        if not self.launch_backend():
            print("❌ Failed to launch backend")
            return False
            
        return True

def main():
    """Main entry point"""
    launcher = BackendLauncher()
    
    try:
        success = launcher.run()
        if not success:
            print("\n❌ Backend launch failed!")
            input("Press Enter to exit...")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Launch cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
