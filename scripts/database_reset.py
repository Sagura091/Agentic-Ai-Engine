"""
Database reset utility for purging and recreating the database.

This script will:
1. Drop all existing tables
2. Recreate the database schema
3. Allow you to create a fresh admin user
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.database.base import get_engine, Base
from app.models.database.migrations.run_all_migrations import MasterMigrationRunner
from sqlalchemy import text
import structlog

logger = structlog.get_logger(__name__)


async def drop_all_tables():
    """Drop all tables in the database."""
    try:
        engine = get_engine()
        
        async with engine.begin() as conn:
            # Get all table names
            result = await conn.execute(text("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """))
            tables = result.fetchall()
            
            if tables:
                # Drop all tables
                table_names = [table[0] for table in tables]
                logger.info(f"Dropping {len(table_names)} tables: {table_names}")
                
                # Disable foreign key checks temporarily
                await conn.execute(text("SET session_replication_role = replica;"))
                
                for table_name in table_names:
                    await conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                    logger.info(f"Dropped table: {table_name}")
                
                # Re-enable foreign key checks
                await conn.execute(text("SET session_replication_role = DEFAULT;"))
                
                logger.info("✅ All tables dropped successfully")
            else:
                logger.info("No tables found to drop")
                
    except Exception as e:
        logger.error(f"❌ Error dropping tables: {e}")
        raise


async def recreate_schema():
    """Recreate the database schema."""
    try:
        logger.info("🔄 Recreating database schema...")

        # Run all migrations to recreate tables
        migration_runner = MasterMigrationRunner()
        results = await migration_runner.run_all_migrations()

        if results.get('overall_success', False):
            logger.info("✅ Database schema recreated successfully")
        else:
            logger.error(f"❌ Some migrations failed: {results}")
            raise Exception("Migration failed")

    except Exception as e:
        logger.error(f"❌ Error recreating schema: {e}")
        raise


async def main():
    """Main function to reset the database."""
    print("🚨 DATABASE RESET UTILITY 🚨")
    print("This will PERMANENTLY DELETE all data in your database!")
    print()
    
    # Confirmation
    confirm = input("Are you sure you want to proceed? Type 'YES' to continue: ")
    if confirm != "YES":
        print("❌ Database reset cancelled.")
        return
    
    try:
        print("\n🔄 Starting database reset...")
        
        # Step 1: Drop all tables
        print("1️⃣ Dropping all existing tables...")
        await drop_all_tables()
        
        # Step 2: Recreate schema
        print("2️⃣ Recreating database schema...")
        await recreate_schema()
        
        print("\n✅ Database reset completed successfully!")
        print("🎯 You can now create a fresh admin user through the application.")
        print("🌐 Visit: http://localhost:8000/register to create your admin account")
        
    except Exception as e:
        print(f"\n❌ Database reset failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
