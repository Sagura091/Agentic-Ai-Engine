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
                if line.strip().startswith('SECURITY__SECRET_KEY='):
                    # Replace existing key
                    env_lines.append(f'SECURITY__SECRET_KEY={secret_key}\n')
                    key_exists = True
                else:
                    env_lines.append(line)

    # Add key if it doesn't exist
    if not key_exists:
        env_lines.append(f'\n# Security Configuration\nSECURITY__SECRET_KEY={secret_key}\n')

    # Write back to .env
    with open(env_file, 'w') as f:
        f.writelines(env_lines)

    print(f"   ✅ Secret key saved to .env file")
    print(f"   🔒 Your application is now secure!\n")


def main():
    """Main entry point."""
    try:
        # Import configuration
        from app.config.unified_config import get_config

        # Load configuration
        config = get_config()

        # Quick validation (skip connectivity for faster startup)
        print("🔍 Validating configuration...")

        # Run validation in event loop
        async def validate():
            return await config.validate(skip_connectivity=True)

        is_valid = asyncio.run(validate())

        if not is_valid:
            print("⚠️  Configuration has warnings. Check logs for details.")

            # Check if it's the default secret key issue
            if config.security.secret_key == "dev-secret-key-change-in-production":
                print("\n🔴 CRITICAL: Using default secret key!")
                print("   This is a security risk for production deployments.")
                response = input("\n   Would you like to generate a secure secret key now? (y/n): ")
                if response.lower() in ['y', 'yes']:
                    generate_secret_key()
                    print("   Please restart the application to use the new key.")
                    sys.exit(0)
                else:
                    print("   ⚠️  Continuing with default key (NOT RECOMMENDED for production)\n")
            else:
                print("   Continuing anyway...\n")

        # Display startup info
        print(f"🚀 Starting {config.server.app_name}")
        print(f"   Environment: {config.server.environment}")
        print(f"   Host: {config.server.host}")
        print(f"   Port: {config.server.port}")
        print(f"   Docs: http://{config.server.host}:{config.server.port}/docs")
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
            host=config.server.host,
            port=config.server.port,
            reload=config.server.debug,
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

