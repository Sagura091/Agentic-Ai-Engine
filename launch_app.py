#!/usr/bin/env python3
"""
Revolutionary AI Agent Builder - Application Launcher
Launches both frontend and backend services in PowerShell windows
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

class AppLauncher:
    def __init__(self):
        self.root_dir = Path(__file__).parent.absolute()
        self.frontend_dir = self.root_dir / "frontend"
        self.backend_dir = self.root_dir / "app"
        self.processes = []
        self.running = True
        
    def check_dependencies(self):
        """Check if required directories and files exist"""
        print("🔍 Checking dependencies...")
        
        # Check directories
        if not self.frontend_dir.exists():
            print(f"❌ Frontend directory not found: {self.frontend_dir}")
            return False
            
        if not self.backend_dir.exists():
            print(f"❌ Backend directory not found: {self.backend_dir}")
            return False
            
        # Check frontend package.json
        frontend_package = self.frontend_dir / "package.json"
        if not frontend_package.exists():
            print(f"❌ Frontend package.json not found: {frontend_package}")
            return False
            
        # Check backend requirements or main files
        backend_main = self.backend_dir / "main.py"
        backend_requirements = self.backend_dir / "requirements.txt"
        
        if not backend_main.exists():
            print(f"❌ Backend main.py not found: {backend_main}")
            return False
            
        print("✅ All dependencies found!")
        return True
        
    def install_frontend_deps(self):
        """Install frontend dependencies if needed"""
        print("📦 Checking frontend dependencies...")
        
        node_modules = self.frontend_dir / "node_modules"
        if not node_modules.exists():
            print("📥 Installing frontend dependencies...")

            # Try multiple npm install strategies
            install_commands = [
                ["npm", "install", "--legacy-peer-deps", "--force", "--no-audit"],
                ["npm", "install", "--legacy-peer-deps", "--no-audit"],
                ["npm", "install", "--force", "--no-audit"],
                ["npm", "ci", "--legacy-peer-deps", "--force"],
                ["npm", "install", "--no-audit"]
            ]

            success = False
            for i, cmd in enumerate(install_commands):
                try:
                    print(f"🔄 Attempt {i+1}: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        cwd=self.frontend_dir,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes timeout
                    )

                    if result.returncode == 0:
                        print("✅ Frontend dependencies installed!")
                        success = True
                        break
                    else:
                        print(f"⚠️  Attempt {i+1} failed, trying next method...")
                        if i == len(install_commands) - 1:  # Last attempt
                            print(f"❌ All installation attempts failed. Last error:")
                            print(result.stderr)

                except subprocess.TimeoutExpired:
                    print(f"⚠️  Attempt {i+1} timed out, trying next method...")
                    if i == len(install_commands) - 1:  # Last attempt
                        print("❌ All installation attempts timed out")
                except Exception as e:
                    print(f"⚠️  Attempt {i+1} error: {e}")
                    if i == len(install_commands) - 1:  # Last attempt
                        print(f"❌ All installation attempts failed")

            if not success:
                print("⚠️  Frontend dependency installation failed, but continuing...")
                print("💡 You may need to run 'npm install' manually in the frontend directory")
                print("💡 Or try: npm install --legacy-peer-deps --force")
                return True  # Continue anyway, let user handle it manually
        else:
            print("✅ Frontend dependencies already installed!")
            
        return True
        
    def install_backend_deps(self):
        """Install backend dependencies if needed"""
        print("🐍 Checking backend dependencies...")
        
        requirements_file = self.backend_dir / "requirements.txt"
        if requirements_file.exists():
            print("📥 Installing backend dependencies...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    cwd=self.backend_dir,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                if result.returncode != 0:
                    print(f"❌ Failed to install backend dependencies:")
                    print(result.stderr)
                    return False
                    
                print("✅ Backend dependencies installed!")
            except subprocess.TimeoutExpired:
                print("❌ Backend dependency installation timed out")
                return False
            except Exception as e:
                print(f"❌ Error installing backend dependencies: {e}")
                return False
        else:
            print("ℹ️  No requirements.txt found, skipping backend dependency installation")
            
        return True
        
    def launch_backend(self):
        """Launch backend in a new PowerShell window"""
        print("🚀 Launching backend server...")

        # PowerShell command to run the backend with dependency check
        ps_command = f"""
        cd '{self.backend_dir}';
        Write-Host '🔥 Starting Revolutionary AI Agent Builder Backend...' -ForegroundColor Green;
        Write-Host '📍 Directory: {self.backend_dir}' -ForegroundColor Cyan;

        # Check if requirements.txt exists and install dependencies
        if (Test-Path 'requirements.txt') {{
            Write-Host '📦 Checking backend dependencies...' -ForegroundColor Cyan;
            try {{
                python -m pip install -r requirements.txt --quiet;
                Write-Host '✅ Backend dependencies ready!' -ForegroundColor Green;
            }} catch {{
                Write-Host '⚠️  Some dependencies may have issues, continuing...' -ForegroundColor Yellow;
            }}
        }} else {{
            Write-Host 'ℹ️  No requirements.txt found, skipping dependency check' -ForegroundColor Gray;
        }}

        Write-Host '';
        Write-Host '🌐 Server will be available at: http://localhost:8888' -ForegroundColor Yellow;
        Write-Host '📚 API Documentation: http://localhost:8888/docs' -ForegroundColor Yellow;
        Write-Host '';
        Write-Host '🚀 Starting server...' -ForegroundColor Green;
        python main.py;
        Write-Host '';
        Write-Host '❌ Backend server stopped. Press any key to close...' -ForegroundColor Red;
        Read-Host
        """
        
        try:
            process = subprocess.Popen([
                "powershell.exe",
                "-NoExit",
                "-Command", ps_command
            ], shell=True)
            
            self.processes.append(("Backend", process))
            print("✅ Backend launched in PowerShell window!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to launch backend: {e}")
            return False
            
    def launch_frontend(self):
        """Launch frontend in a new PowerShell window"""
        print("🎨 Launching frontend development server...")

        # Check if node_modules exists
        node_modules = self.frontend_dir / "node_modules"

        # PowerShell command to run the frontend with dependency check
        ps_command = f"""
        cd '{self.frontend_dir}';
        Write-Host '🎨 Starting Revolutionary AI Agent Builder Frontend...' -ForegroundColor Green;
        Write-Host '📍 Directory: {self.frontend_dir}' -ForegroundColor Cyan;

        # Check if node_modules exists
        if (-not (Test-Path 'node_modules')) {{
            Write-Host '⚠️  node_modules not found, attempting to install dependencies...' -ForegroundColor Yellow;
            Write-Host '📦 Running: npm install --legacy-peer-deps --force --no-audit' -ForegroundColor Cyan;
            try {{
                npm install --legacy-peer-deps --force --no-audit;
                if ($LASTEXITCODE -eq 0) {{
                    Write-Host '✅ Dependencies installed successfully!' -ForegroundColor Green;
                }} else {{
                    Write-Host '❌ Dependency installation failed, but trying to start anyway...' -ForegroundColor Red;
                }}
            }} catch {{
                Write-Host '❌ Dependency installation failed, but trying to start anyway...' -ForegroundColor Red;
            }}
        }} else {{
            Write-Host '✅ Dependencies found!' -ForegroundColor Green;
        }}

        Write-Host '';
        Write-Host '🌐 Development server will be available at: http://localhost:5173' -ForegroundColor Yellow;
        Write-Host '🔧 Vite dev server with hot reload enabled' -ForegroundColor Yellow;
        Write-Host '';
        Write-Host '🚀 Starting development server...' -ForegroundColor Green;

        # Try to start the dev server
        try {{
            npm run dev;
        }} catch {{
            Write-Host '';
            Write-Host '❌ Failed to start development server!' -ForegroundColor Red;
            Write-Host '💡 Try running these commands manually:' -ForegroundColor Yellow;
            Write-Host '   cd frontend' -ForegroundColor Gray;
            Write-Host '   npm install --legacy-peer-deps --force' -ForegroundColor Gray;
            Write-Host '   npm run dev' -ForegroundColor Gray;
        }}

        Write-Host '';
        Write-Host '❌ Frontend server stopped. Press any key to close...' -ForegroundColor Red;
        Read-Host
        """
        
        try:
            process = subprocess.Popen([
                "powershell.exe",
                "-NoExit",
                "-Command", ps_command
            ], shell=True)
            
            self.processes.append(("Frontend", process))
            print("✅ Frontend launched in PowerShell window!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to launch frontend: {e}")
            return False
            
    def monitor_processes(self):
        """Monitor running processes"""
        print("\n🔍 Monitoring services...")
        print("Press Ctrl+C to stop all services\n")
        
        try:
            while self.running:
                time.sleep(2)
                
                # Check if processes are still running
                active_processes = []
                for name, process in self.processes:
                    if process.poll() is None:  # Process is still running
                        active_processes.append((name, process))
                    else:
                        print(f"⚠️  {name} process has stopped")
                        
                self.processes = active_processes
                
                if not self.processes:
                    print("❌ All processes have stopped")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            self.shutdown()
            
    def shutdown(self):
        """Shutdown all processes"""
        print("🔄 Stopping all services...")
        self.running = False
        
        for name, process in self.processes:
            try:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    print(f"✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {name}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {name} force stopped")
                    
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
                
        print("✅ All services stopped!")
        
    def run(self):
        """Main run method"""
        print("🚀 Revolutionary AI Agent Builder - Application Launcher")
        print("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed. Please fix the issues above.")
            return False
            
        # Install frontend dependencies
        if not self.install_frontend_deps():
            print("❌ Frontend dependency installation failed.")
            return False

        # Backend dependencies will be installed in the PowerShell window
        print("🐍 Backend dependencies will be checked during launch...")
            
        print("\n🎯 Launching services...")
        print("-" * 40)
        
        # Launch backend
        if not self.launch_backend():
            print("❌ Failed to launch backend")
            return False
            
        # Wait a moment for backend to start
        time.sleep(2)
        
        # Launch frontend
        if not self.launch_frontend():
            print("❌ Failed to launch frontend")
            self.shutdown()
            return False
            
        # Wait a moment for frontend to start
        time.sleep(3)
        
        print("\n🎉 Revolutionary AI Agent Builder is starting up!")
        print("=" * 60)
        print("🔥 Backend Server: http://localhost:8888")
        print("📚 API Documentation: http://localhost:8888/docs")
        print("🎨 Frontend App: http://localhost:5173")
        print("🔧 Visual Agent Builder: http://localhost:5173/agents/create-visual")
        print("=" * 60)
        
        # Monitor processes
        self.monitor_processes()
        
        return True

def main():
    """Main entry point"""
    launcher = AppLauncher()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        launcher.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = launcher.run()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        launcher.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()
