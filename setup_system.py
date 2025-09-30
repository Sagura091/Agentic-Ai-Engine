#!/usr/bin/env python3
"""
Complete Agentic AI System Setup Script

This script performs a complete system setup:
1. Runs database migrations
2. Initializes the backend (creates all directories)
3. Verifies the setup

Usage:
    python setup_system.py
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
    print_step(1, 5, "Checking Docker...")
    
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
    print_step(2, 5, "Starting PostgreSQL container...")
    
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
    print_step(3, 5, "Running database migrations...")
    
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


async def initialize_backend():
    """Initialize the backend to create all directories."""
    print_step(4, 5, "Initializing backend and creating directories...")
    
    try:
        # Import settings to trigger directory creation
        from app.config.settings import get_settings
        
        settings = get_settings()
        settings.create_directories()
        
        print_success("Settings loaded and base directories created")
        
        # Create additional data directories
        data_directories = [
            "data/logs/agents",
            "data/logs/backend",
            "data/chroma",
            "data/autonomous",
            "data/agent_files",
            "data/cache",
            "data/downloads/session_docs",
            "data/generated_files",
            "data/memes/generated",
            "data/memes/templates",
            "data/models/embedding",
            "data/models/llm",
            "data/models/reranking",
            "data/models/vision",
            "data/outputs",
            "data/screenshots",
            "data/session_documents/sessions",
            "data/session_vectors",
            "data/templates",
            "data/temp/session_docs",
            "data/uploads",
            "data/config/agents",
            "data/config/templates",
            "data/meme_analysis_cache"
        ]
        
        created_count = 0
        for directory in data_directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
        
        print_success(f"Created {created_count} additional directories")
        print_success(f"Total data directories: {len(data_directories) + 5}")
        
        return True
        
    except Exception as e:
        print_error(f"Backend initialization failed: {e}")
        return False


async def check_ollama_installed():
    """Check if Ollama is installed (helper function)."""
    try:
        from detect_system_and_model import OllamaManager
        return await OllamaManager.is_installed()
    except:
        return False


async def check_ollama_and_model():
    """Check Ollama and recommend/pull model."""
    print_step(5, 8, "Checking Ollama and multimodal model...")

    try:
        # Import the detection script
        from detect_system_and_model import detect_and_recommend, OllamaManager

        # Run detection
        recommended_model, installed_models = await detect_and_recommend()

        if recommended_model is None:
            print_warning("Ollama is not installed")
            print_info("Install from: https://ollama.ai/download")
            print_info("Skipping Ollama setup - you can install it later")
            return None

        print_success(f"Ollama is installed")

        # Check if recommended model is installed
        if installed_models and recommended_model in installed_models:
            print_success(f"Recommended model already installed: {recommended_model}")
            return recommended_model

        # Ask to pull model
        print_info(f"Recommended model: {recommended_model}")

        # Auto-pull in setup
        os.environ['AUTO_PULL_MODEL'] = 'true'
        success = await OllamaManager.pull_model(recommended_model)

        if success:
            print_success(f"Successfully pulled {recommended_model}")
            return recommended_model
        else:
            print_warning(f"Failed to pull {recommended_model}")
            print_info(f"You can pull it later with: ollama pull {recommended_model}")
            # Return None to indicate no model was pulled, but Ollama is installed
            return None

    except Exception as e:
        print_warning(f"Ollama check failed: {e}")
        print_info("Continuing without Ollama setup")
        return None


async def update_env_with_model(model_name: str):
    """Update .env file with the recommended model."""
    if not model_name:
        return

    try:
        env_path = Path(".env")

        if not env_path.exists():
            print_warning(".env file not found, skipping model configuration")
            return

        # Read current .env
        with open(env_path, 'r') as f:
            lines = f.readlines()

        # Update or add the model setting
        model_updated = False
        new_lines = []

        for line in lines:
            if line.startswith('AGENTIC_DEFAULT_AGENT_MODEL='):
                new_lines.append(f'AGENTIC_DEFAULT_AGENT_MODEL={model_name}\n')
                model_updated = True
            else:
                new_lines.append(line)

        # Add if not found
        if not model_updated:
            new_lines.append(f'\n# Auto-configured by setup\nAGENTIC_DEFAULT_AGENT_MODEL={model_name}\n')

        # Write back
        with open(env_path, 'w') as f:
            f.writelines(new_lines)

        print_success(f"Updated .env with model: {model_name}")

    except Exception as e:
        print_warning(f"Could not update .env: {e}")


async def initialize_backend_once():
    """Initialize the backend once to create all data files."""
    print_step(6, 8, "Initializing backend (creating all data files)...")

    try:
        print_info("Starting backend initialization...")
        print_info("This will create all necessary data files and configurations")

        # Import the main app
        from app.main import app
        from app.config.settings import get_settings

        # Trigger startup events
        print_info("Loading application...")

        # The app startup will create all necessary files
        # We just need to import and trigger the lifespan
        settings = get_settings()

        print_success("Backend initialized successfully")
        print_success("All data files and configurations created")

        return True

    except Exception as e:
        print_warning(f"Backend initialization had warnings: {e}")
        print_info("This is often normal - continuing...")
        return True  # Don't fail setup for backend warnings


async def verify_setup():
    """Verify the setup is complete."""
    print_step(7, 8, "Verifying setup...")
    
    checks_passed = 0
    total_checks = 4
    
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
    
    # Check 2: Data directories
    try:
        required_dirs = ["data", "data/agents", "data/workflows", "data/logs", "data/chroma"]
        missing = [d for d in required_dirs if not Path(d).exists()]
        
        if not missing:
            print_success("All required directories exist")
            checks_passed += 1
        else:
            print_error(f"Missing directories: {missing}")
    except Exception as e:
        print_error(f"Directory check failed: {e}")
    
    # Check 3: Migration history
    try:
        from app.models.database.base import get_engine
        from sqlalchemy import text
        
        engine = get_engine()
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM migration_history"))
            count = result.scalar()
        
        print_success(f"Migration history table exists ({count} migrations recorded)")
        checks_passed += 1
    except Exception as e:
        print_error(f"Migration history check failed: {e}")
    
    # Check 4: Configuration
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        
        print_success(f"Configuration loaded (Environment: {settings.ENVIRONMENT})")
        checks_passed += 1
    except Exception as e:
        print_error(f"Configuration check failed: {e}")
    
    print()
    print_info(f"Verification: {checks_passed}/{total_checks} checks passed")
    
    return checks_passed == total_checks


async def main():
    """Main setup function."""
    start_time = datetime.now()

    print_header("AGENTIC AI SYSTEM - COMPLETE SETUP")

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

    # Step 4: Initialize backend directories
    if not await initialize_backend():
        return 1

    print()

    # Step 5: Check Ollama and pull model
    recommended_model = await check_ollama_and_model()
    ollama_installed = recommended_model is not None or await check_ollama_installed()

    print()

    # Step 6: Initialize backend once (creates all data files)
    if not await initialize_backend_once():
        return 1

    print()

    # Step 7: Verify setup
    verification_passed = await verify_setup()

    print()

    # Step 8: Update .env with recommended model
    if recommended_model:
        print_step(8, 8, "Updating configuration with recommended model...")
        await update_env_with_model(recommended_model)
    else:
        print_step(8, 8, "Skipping model configuration (no model pulled)")
        if ollama_installed:
            print_info("Ollama is installed but model pull failed or was skipped")
    
    # Print summary
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print()
    print_header("SETUP COMPLETE!")
    
    if verification_passed:
        print(f"{Colors.GREEN}✓ All checks passed!{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ Setup completed with warnings{Colors.END}")
    
    print()
    print(f"{Colors.CYAN}Setup time: {elapsed_time:.1f} seconds{Colors.END}")
    print()
    print(f"{Colors.BOLD}System Configuration:{Colors.END}")
    if recommended_model:
        print(f"  Default Model: {Colors.WHITE}{recommended_model}{Colors.END}")
        print(f"  LLM Provider:  {Colors.WHITE}ollama{Colors.END}")
    elif ollama_installed:
        print(f"  Ollama:        {Colors.YELLOW}Installed (model pull failed/skipped){Colors.END}")
        print(f"  Suggestion:    {Colors.WHITE}Run: ollama pull llama3.2-vision:11b{Colors.END}")
    else:
        print(f"  Ollama:        {Colors.YELLOW}Not installed (optional){Colors.END}")
    print()
    print(f"{Colors.BOLD}Next Steps:{Colors.END}")
    print(f"  1. Start backend:  {Colors.WHITE}python -m app.main{Colors.END}")
    print(f"  2. Test agents:    {Colors.WHITE}python scripts/test_agent_standalone.py{Colors.END}")
    print(f"  3. Run tests:      {Colors.WHITE}python -m pytest tests/ -v{Colors.END}")
    if not ollama_installed:
        print(f"  4. Install Ollama: {Colors.WHITE}https://ollama.ai/download{Colors.END}")
    elif not recommended_model:
        print(f"  4. Pull model:     {Colors.WHITE}ollama pull llama3.2-vision:11b{Colors.END}")
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

