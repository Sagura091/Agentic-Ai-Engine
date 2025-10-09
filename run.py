#!/usr/bin/env python3
"""
Agentic AI Engine - Backend Launcher

Simple, clean launcher for the backend application.
"""

import asyncio
import sys
import logging
import secrets
from pathlib import Path


def generate_secret_key():
    """Generate a secure secret key and save it to .env file."""
    env_file = Path(".env")

    # Generate a secure random key
    secret_key = secrets.token_urlsafe(32)

    print("\n🔐 SECURITY: Generating secure secret key...")
    print(f"   Generated key: {secret_key[:16]}... (truncated for security)")

    # Read existing .env or create new one
    env_lines = []
    key_exists = False

    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip().startswith('AGENTIC_SECRET_KEY='):
                    # Replace existing key
                    env_lines.append(f'AGENTIC_SECRET_KEY={secret_key}\n')
                    key_exists = True
                else:
                    env_lines.append(line)

    # Add key if it doesn't exist
    if not key_exists:
        env_lines.append(f'\n# Security Configuration\nAGENTIC_SECRET_KEY={secret_key}\n')

    # Write back to .env
    with open(env_file, 'w') as f:
        f.writelines(env_lines)

    print(f"   ✅ Secret key saved to .env file")
    print(f"   🔒 Your application is now secure!\n")


def main():
    """Main entry point."""
    try:
        # Import configuration
        from app.config.settings import get_settings

        # Load configuration
        settings = get_settings()

        # Check if it's the default secret key issue
        if settings.SECRET_KEY == "your-secret-key-change-this":
            print("\n🔴 CRITICAL: Using default secret key!")
            print("   This is a security risk for production deployments.")
            response = input("\n   Would you like to generate a secure secret key now? (y/n): ")
            if response.lower() in ['y', 'yes']:
                generate_secret_key()
                print("   Please restart the application to use the new key.")
                sys.exit(0)
            else:
                print("   ⚠️  Continuing with default key (NOT RECOMMENDED for production)\n")

        # Display startup info
        print(f"🚀 Starting {settings.APP_NAME}")
        print(f"   Environment: {settings.ENVIRONMENT}")
        print(f"   Host: {settings.HOST}")
        print(f"   Port: {settings.PORT}")
        print(f"   Docs: http://{settings.HOST}:{settings.PORT}/docs")
        print()

        # Configure clean console logging
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
            stream=sys.stdout,
        )

        # Start uvicorn with socketio_app (same as app.main:main())
        import uvicorn

        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level="warning",
            access_log=False,
            loop="uvloop" if sys.platform != "win32" else "asyncio",
        )

    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

