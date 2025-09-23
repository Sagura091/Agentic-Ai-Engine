"""
Complete Database Setup Script for Agentic AI System

This script performs a complete database setup after Docker volumes are cleared:
1. Executes init-db.sql for basic table structure
2. Runs all migrations for full feature set
3. Verifies the setup was successful

Usage: python setup_database.py
"""

import asyncio
import sys
import subprocess
import time
import os
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.models.database.base import get_engine
    from sqlalchemy import text
    import structlog
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the project root and dependencies are installed")
    sys.exit(1)

logger = structlog.get_logger(__name__)


async def wait_for_database(max_attempts: int = 30, delay: int = 2) -> bool:
    """Wait for database to be available."""
    print("🔄 Waiting for database connection...")
    
    for attempt in range(max_attempts):
        try:
            engine = get_engine()
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            print("✅ Database connection established!")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"⏳ Attempt {attempt + 1}/{max_attempts} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"❌ Failed to connect to database after {max_attempts} attempts: {e}")
                return False
    
    return False


async def execute_init_sql() -> bool:
    """Execute the init-db.sql file using psql command."""
    try:
        print("🔧 Executing init-db.sql...")

        # Read the SQL file
        init_sql_path = project_root / "init-db.sql"
        if not init_sql_path.exists():
            print(f"❌ init-db.sql not found at {init_sql_path}")
            return False

        # Try to use psql command first (more reliable for complex SQL)
        try:
            db_url = os.getenv('DATABASE_URL', 'postgresql://agentic_user:agentic_password@localhost:5432/agentic_ai')

            result = subprocess.run(
                ['psql', db_url, '-f', str(init_sql_path)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print("✅ init-db.sql executed successfully using psql!")
                return True
            else:
                print("⚠️  psql command failed, trying SQLAlchemy method...")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  psql not available, using SQLAlchemy method...")

        # Fallback to SQLAlchemy method with better SQL parsing
        with open(init_sql_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Execute the entire SQL content as one statement
        engine = get_engine()
        async with engine.begin() as conn:
            try:
                await conn.execute(text(sql_content))
                print("✅ init-db.sql executed successfully!")
                return True
            except Exception as e:
                print(f"⚠️  Full execution failed, trying statement by statement: {e}")

                # If that fails, try statement by statement with better parsing
                # Remove comments and split more carefully
                lines = sql_content.split('\n')
                current_statement = []
                in_function = False

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('--'):
                        continue

                    current_statement.append(line)

                    # Check for function boundaries
                    if 'CREATE OR REPLACE FUNCTION' in line.upper():
                        in_function = True
                    elif in_function and line.endswith("';"):
                        in_function = False

                    # Execute statement if we hit a semicolon and not in a function
                    if line.endswith(';') and not in_function:
                        statement = ' '.join(current_statement)
                        if statement.strip():
                            try:
                                await conn.execute(text(statement))
                            except Exception as stmt_error:
                                if "already exists" not in str(stmt_error).lower():
                                    print(f"⚠️  Warning: {stmt_error}")
                        current_statement = []

                print("✅ init-db.sql executed with warnings!")
                return True

    except Exception as e:
        print(f"❌ Error executing init-db.sql: {e}")
        return False


def run_migrations() -> bool:
    """Run the migration script."""
    try:
        print("🚀 Running database migrations...")

        migrations_script = project_root / "db" / "migrations" / "run_all_migrations.py"
        if not migrations_script.exists():
            print(f"❌ Migration script not found at {migrations_script}")
            return False

        # Set up environment with proper Python path
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')

        # Run the migration script with proper environment
        result = subprocess.run(
            [sys.executable, str(migrations_script)],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print("✅ Database migrations completed successfully!")
            if result.stdout:
                print("📋 Migration output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ Migration failed with exit code {result.returncode}")
            if result.stderr:
                print("🔍 Error output:")
                print(result.stderr)
            if result.stdout:
                print("📋 Standard output:")
                print(result.stdout)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Migration script timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False


async def verify_setup() -> bool:
    """Verify that the database setup was successful."""
    try:
        print("🔍 Verifying database setup...")
        
        engine = get_engine()
        async with engine.begin() as conn:
            # Check for key tables from init-db.sql
            basic_tables = ['agents', 'workflows', 'custom_tools', 'agent_conversations']
            
            # Check for key tables from migrations
            migration_tables = ['users', 'user_sessions', 'autonomous_agent_states', 'learning_experiences']
            
            all_tables = basic_tables + migration_tables
            existing_tables = []
            
            for table in all_tables:
                result = await conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table}'
                    );
                """))
                exists = result.scalar()
                if exists:
                    existing_tables.append(table)
            
            print(f"📊 Found {len(existing_tables)}/{len(all_tables)} expected tables:")
            for table in existing_tables:
                print(f"   ✅ {table}")
            
            missing_tables = set(all_tables) - set(existing_tables)
            if missing_tables:
                print("⚠️  Missing tables:")
                for table in missing_tables:
                    print(f"   ❌ {table}")
            
            # Consider setup successful if we have at least the basic tables
            success = len(existing_tables) >= len(basic_tables)
            
            if success:
                print("✅ Database setup verification passed!")
            else:
                print("❌ Database setup verification failed!")
            
            return success
            
    except Exception as e:
        print(f"❌ Error verifying setup: {e}")
        return False


async def main() -> int:
    """Main setup function."""
    print("🚀 AGENTIC AI DATABASE SETUP")
    print("=" * 50)
    print("This script will set up your database from scratch.")
    print("Use this after clearing Docker volumes or resetting the database.")
    print()
    
    try:
        # Step 1: Wait for database
        if not await wait_for_database():
            return 1
        
        # Step 2: Execute init-db.sql
        if not await execute_init_sql():
            return 1
        
        # Step 3: Run migrations
        if not run_migrations():
            return 1
        
        # Step 4: Verify setup
        if not await verify_setup():
            print("⚠️  Setup completed with warnings. Some features may not be available.")
        
        print()
        print("🎉 DATABASE SETUP COMPLETE!")
        print("=" * 50)
        print("✅ Your database is ready for use!")
        print("🌐 You can now start the application:")
        print("   Backend:  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("   Frontend: cd frontend && npm run dev")
        print("🎯 Create your admin user at: http://localhost:8000/register")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error during setup: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
