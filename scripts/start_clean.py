#!/usr/bin/env python3
"""
Clean Backend Launcher - Agentic AI Engine
Launches the backend with minimal console logging and detailed file logging.
"""

import os
import sys
import logging
from pathlib import Path

def setup_clean_logging():
    """Setup clean console logging with detailed file logging."""
    # Ensure logs directory exists
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger to WARNING level for console
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler for detailed logging
    file_handler = logging.FileHandler(logs_dir / "backend.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    
    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    print("✅ Clean logging configured:")
    print(f"   📺 Console: WARNING level and above")
    print(f"   📁 File: INFO level and above -> {logs_dir / 'backend.log'}")

def main():
    """Main entry point with clean logging."""
    print("🚀 Starting Agentic AI Engine with Clean Logging")
    print("=" * 60)
    
    # Setup clean logging first
    setup_clean_logging()
    
    # Set environment variables for clean startup
    os.environ["AGENTIC_LOG_LEVEL"] = "WARNING"
    os.environ["AGENTIC_LOG_TO_CONSOLE"] = "true"
    os.environ["AGENTIC_LOG_TO_FILE"] = "true"
    os.environ["AGENTIC_LOG_CONSOLE_FORMAT"] = "simple"
    
    print("\n🔧 Configuration:")
    print("   🖥️  Host: localhost")
    print("   🔌 Port: 8000")
    print("   📋 Docs: http://localhost:8000/docs")
    print("   ❤️  Health: http://localhost:8000/health")
    print("   📊 Metrics: http://localhost:8000/metrics")
    
    print("\n🎯 Starting server...")
    print("-" * 40)
    
    try:
        # Import and run the application
        import uvicorn
        from app.main import socketio_app
        
        # Run with clean configuration
        uvicorn.run(
            socketio_app,
            host="localhost",
            port=8000,
            reload=False,  # Disable reload for cleaner output
            log_level="warning",  # Only warnings and errors
            access_log=False,  # No access logs
            server_header=False,  # No server header
            date_header=False,  # No date header
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
