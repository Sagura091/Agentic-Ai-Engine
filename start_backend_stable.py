#!/usr/bin/env python3
"""
Stable Backend Launcher - Agentic AI Engine
Launch backend without hot reload for maximum stability
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch backend without hot reload for stability."""
    root_dir = Path(__file__).parent.absolute()
    
    print("🚀 Agentic AI Engine - Stable Backend Launcher")
    print("=" * 50)
    print("🔒 Hot reload DISABLED for maximum stability")
    print("💡 Restart manually after code changes")
    print("=" * 50)
    print("")
    
    # Set environment variable to disable reload
    os.environ["DISABLE_RELOAD"] = "1"
    
    print("🔧 Backend Server: http://localhost:8888")
    print("📚 API Documentation: http://localhost:8888/docs")
    print("🔧 Interactive API: http://localhost:8888/redoc")
    print("❤️  Health Check: http://localhost:8888/health")
    print("")
    print("🚀 Starting stable backend server...")
    print("💡 Press Ctrl+C to stop the server")
    print("")
    
    try:
        # Start the backend server without reload
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:socketio_app",
            "--host", "localhost",
            "--port", "8888",
            "--log-level", "warning",
            "--no-access-log",
            "--timeout-graceful-shutdown", "10"
        ], cwd=root_dir, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        print("💡 Try installing dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
