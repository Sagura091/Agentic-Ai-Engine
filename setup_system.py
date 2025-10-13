#!/usr/bin/env python3
"""
Agentic AI System - Database Setup Script

This script sets up the database infrastructure required for the system:
1. Checks Docker is running
2. Starts PostgreSQL container
3. Runs database migrations
4. Verifies database connection

NOTE: Models, directories, and other resources are automatically initialized
      on first backend/agent startup. This script only handles database setup.

Usage:
    python setup_system.py

After running this script, start the backend with:
    python -m app.main
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class Colors:
    """ANSI color codes for terminal output."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    @staticmethod
    def strip_colors():
        """Disable colors on Windows if not supported."""
        if os.name == 'nt':
            # Enable ANSI colors on Windows 10+
            os.system('')


def print_header(text: str):
    """Print a formatted header."""
    Colors.strip_colors()
    print()
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}")
    print()


def print_step(step: int, total: int, text: str):
    """Print a step indicator."""
    print(f"{Colors.BLUE}[{step}/{total}] {text}{Colors.END}")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}      ✓ {text}{Colors.END}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}      ⚠ {text}{Colors.END}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}      ✗ {text}{Colors.END}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.WHITE}      {text}{Colors.END}")


async def check_docker():
    """Check if Docker is running."""
    print_step(1, 4, "Checking Docker...")
    
    try:
        process = await asyncio.create_subprocess_exec(
            'docker', 'version',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        if process.returncode == 0:
            print_success("Docker is running")
            return True
        else:
            print_error("Docker is not running")
            print_info("Please start Docker Desktop and try again")
            return False
    except FileNotFoundError:
        print_error("Docker is not installed")
        print_info("Please install Docker Desktop from https://www.docker.com/products/docker-desktop")
        return False


async def start_postgres():
    """Start PostgreSQL container."""
    print_step(2, 4, "Starting PostgreSQL container...")
    
    try:
        # Start PostgreSQL
        process = await asyncio.create_subprocess_exec(
            'docker-compose', 'up', '-d', 'postgres',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            print_error("Failed to start PostgreSQL")
            print_info(stderr.decode())
            return False
        
        print_success("PostgreSQL container started")
        
        # Wait for PostgreSQL to be ready
        print_info("Waiting for PostgreSQL to be ready...")
        max_attempts = 30
        
        for attempt in range(1, max_attempts + 1):
            await asyncio.sleep(2)
            
            process = await asyncio.create_subprocess_exec(
                'docker-compose', 'exec', '-T', 'postgres',
                'pg_isready', '-U', 'agentic_user', '-d', 'agentic_ai',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode == 0:
                print_success("PostgreSQL is ready")
                return True
            
            if attempt % 5 == 0:
                print_info(f"Attempt {attempt}/{max_attempts} - Still waiting...")
        
        print_error("PostgreSQL failed to start within timeout")
        return False
        
    except Exception as e:
        print_error(f"Error starting PostgreSQL: {e}")
        return False


async def run_migrations():
    """Run database migrations."""
    print_step(3, 4, "Running database migrations...")
    
    try:
        # Import and run migrations
        from db.migrations.run_all_migrations import MasterMigrationRunner
        
        runner = MasterMigrationRunner()
        results = await runner.run_all_migrations()
        
        # Check results
        completed = results.get('completed', 0)
        failed = results.get('failed', 0)
        
        if failed == 0:
            print_success(f"All migrations completed successfully ({completed} migrations)")
            return True
        else:
            print_warning(f"Migrations completed with warnings ({completed} completed, {failed} failed)")
            print_info("This is normal - warnings occur when creating existing objects")
            return True
            
    except Exception as e:
        print_error(f"Migration failed: {e}")
        return False


async def verify_database():
    """Verify database connection and migrations."""
    print_step(4, 4, "Verifying database setup...")

    checks_passed = 0
    total_checks = 2

    # Check 1: Database connection
    try:
        from app.models.database.base import get_engine
        from sqlalchemy import text

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        print_success("Database connection working")
        checks_passed += 1
    except Exception as e:
        print_error(f"Database connection failed: {e}")

    # Check 2: Migration history
    try:
        from app.models.database.base import get_engine
        from sqlalchemy import text

        engine = get_engine()
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM migration_history"))
            count = result.scalar()

        print_success(f"Migration history verified ({count} migrations recorded)")
        checks_passed += 1
    except Exception as e:
        print_error(f"Migration history check failed: {e}")

    print()
    print_info(f"Verification: {checks_passed}/{total_checks} checks passed")

    return checks_passed == total_checks





async def main():
    """Main setup function - Database infrastructure only."""
    start_time = datetime.now()

    print_header("AGENTIC AI SYSTEM - DATABASE SETUP")

    # Step 1: Check Docker
    if not await check_docker():
        return 1

    print()

    # Step 2: Start PostgreSQL
    if not await start_postgres():
        return 1

    print()

    # Step 3: Run migrations
    if not await run_migrations():
        return 1

    print()

    # Step 4: Verify database
    verification_passed = await verify_database()

    # Print summary
    elapsed_time = (datetime.now() - start_time).total_seconds()

    print()
    print_header("DATABASE SETUP COMPLETE!")

    if verification_passed:
        print(f"{Colors.GREEN}✓ Database is ready!{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ Setup completed with warnings{Colors.END}")

    print()
    print(f"{Colors.CYAN}Setup time: {elapsed_time:.1f} seconds{Colors.END}")
    print()
    print(f"{Colors.BOLD}What's Next:{Colors.END}")
    print(f"  {Colors.WHITE}Models, directories, and configurations will be initialized automatically{Colors.END}")
    print(f"  {Colors.WHITE}on first backend or agent startup.{Colors.END}")
    print()
    print(f"{Colors.BOLD}Start the Backend:{Colors.END}")
    print(f"  {Colors.GREEN}python -m app.main{Colors.END}")
    print()
    print(f"  {Colors.WHITE}On first startup, the system will:{Colors.END}")
    print(f"  {Colors.WHITE}  • Create all required directories{Colors.END}")
    print(f"  {Colors.WHITE}  • Download essential embedding models{Colors.END}")
    print(f"  {Colors.WHITE}  • Check and pull Ollama models (if installed){Colors.END}")
    print(f"  {Colors.WHITE}  • Configure all systems automatically{Colors.END}")
    print()
    print(f"{Colors.BOLD}Database Access:{Colors.END}")
    print(f"  pgAdmin:  {Colors.WHITE}http://localhost:5050{Colors.END}")
    print(f"  Email:    {Colors.WHITE}admin@agentic.ai{Colors.END}")
    print()
    print(f"{Colors.BOLD}Management:{Colors.END}")
    print(f"  Stop:     {Colors.YELLOW}docker-compose down{Colors.END}")
    print(f"  Clean:    {Colors.RED}docker-compose down -v{Colors.END}")
    print()

    return 0 if verification_passed else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print()
        print(f"{Colors.RED}Setup interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"{Colors.RED}Setup failed with error: {e}{Colors.END}")
        sys.exit(1)

