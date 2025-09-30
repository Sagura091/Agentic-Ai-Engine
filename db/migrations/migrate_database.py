#!/usr/bin/env python3
"""
Database Migration Script for Agentic AI Platform.

This script provides a simple command-line interface for running
database migrations and managing the database schema.

Location: db/migrations/migrate_database.py

Usage:
    python db/migrations/migrate_database.py [command]

Commands:
    migrate     - Run all pending migrations
    status      - Show migration status
    health      - Check database health
    kb-migrate  - Migrate knowledge bases from JSON to database
    help        - Show this help message

Examples:
    python db/migrations/migrate_database.py migrate
    python db/migrations/migrate_database.py status
    python db/migrations/migrate_database.py health
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path (go up two levels: migrations/ -> db/ -> project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import structlog
from run_all_migrations import MasterMigrationRunner
from app.services.knowledge_base_migration_service import knowledge_base_migration_service

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()  # Pretty console output
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class DatabaseMigrationCLI:
    """Command-line interface for database migrations."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.runner = MasterMigrationRunner()
    
    async def run_migrations(self) -> int:
        """Run all database migrations."""
        try:
            print("🚀 Starting database migrations...")
            print("=" * 60)
            
            results = await self.runner.run_all_migrations()
            
            print("=" * 60)
            print("📊 Migration Results:")
            print(f"   Total Migrations: {results['total_migrations']}")
            print(f"   ✅ Completed: {results['completed_migrations']}")
            print(f"   ❌ Failed: {results['failed_migrations']}")
            print(f"   🎯 Overall Success: {results['success']}")
            
            if results["errors"]:
                print("\n❌ Errors encountered:")
                for error in results["errors"]:
                    print(f"   - {error}")
            
            print("=" * 60)
            
            if results["success"]:
                print("🎉 All migrations completed successfully!")
                return 0
            else:
                print("💔 Some migrations failed. Check the output above for details.")
                return 1
                
        except Exception as e:
            print(f"💥 Migration failed with exception: {str(e)}")
            return 1
    
    async def show_status(self) -> int:
        """Show migration status."""
        try:
            print("📋 Database Migration Status")
            print("=" * 60)
            
            status = await self.runner.get_migration_status()
            
            # Autonomous tables
            autonomous = status.get("autonomous_tables", {})
            autonomous_status = "✅ Applied" if autonomous.get("applied") else "⏳ Pending"
            print(f"Autonomous Tables: {autonomous_status}")
            
            # Enhanced tables
            enhanced = status.get("enhanced_tables", {})
            enhanced_status = "✅ Applied" if enhanced.get("applied") else "⏳ Pending"
            print(f"Enhanced Tables: {enhanced_status}")
            
            # Knowledge base data
            kb_data = status.get("knowledge_base_data", {})
            kb_status = kb_data.get("status", "unknown")
            kb_status_icon = {
                "completed": "✅",
                "pending": "⏳",
                "no_data": "ℹ️",
                "error": "❌"
            }.get(kb_status, "❓")
            print(f"Knowledge Base Data: {kb_status_icon} {kb_status.title()}")
            
            if kb_data.get("json_count", 0) > 0:
                print(f"   - JSON file has {kb_data['json_count']} knowledge bases")
            if kb_data.get("database_count", 0) > 0:
                print(f"   - Database has {kb_data['database_count']} knowledge bases")
            
            # Overall status
            overall = status.get("overall_status", "unknown")
            overall_icon = {
                "completed": "✅",
                "pending": "⏳",
                "error": "❌"
            }.get(overall, "❓")
            
            print("=" * 60)
            print(f"Overall Status: {overall_icon} {overall.title()}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to get migration status: {str(e)}")
            return 1
    
    async def check_health(self) -> int:
        """Check database health."""
        try:
            print("🏥 Database Health Check")
            print("=" * 60)
            
            # Test database connection
            from app.models.database.base import get_session_factory
            from sqlalchemy import text
            
            session_factory = get_session_factory()
            
            try:
                async with session_factory() as session:
                    # Test connection
                    result = await session.execute(text("SELECT 1"))
                    result.scalar()
                    print("✅ Database Connection: Healthy")
                    
                    # Count tables
                    result = await session.execute(text("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                    """))
                    table_count = result.scalar()
                    print(f"📊 Total Tables: {table_count}")
                    
                    # Check migration status
                    status = await self.runner.get_migration_status()
                    overall_status = status.get("overall_status", "unknown")
                    status_icon = {
                        "completed": "✅",
                        "pending": "⏳",
                        "error": "❌"
                    }.get(overall_status, "❓")
                    print(f"🔄 Migration Status: {status_icon} {overall_status.title()}")
                    
                    print("=" * 60)
                    print("🎉 Database is healthy!")
                    return 0
                    
            except Exception as e:
                print(f"❌ Database Connection: Failed ({str(e)})")
                print("=" * 60)
                print("💔 Database is unhealthy!")
                return 1
                
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return 1
    
    async def migrate_knowledge_bases(self) -> int:
        """Migrate knowledge bases from JSON to database."""
        try:
            print("📚 Migrating Knowledge Bases from JSON to Database")
            print("=" * 60)
            
            result = await knowledge_base_migration_service.migrate_from_json()
            
            print(f"✅ Success: {result['success']}")
            print(f"📊 Migrated: {result['migrated_count']} knowledge bases")
            print(f"⏭️ Skipped: {result['skipped_count']} knowledge bases")
            
            if result["errors"]:
                print(f"❌ Errors: {len(result['errors'])}")
                for error in result["errors"]:
                    print(f"   - {error}")
            
            print("=" * 60)
            print(result["message"])
            
            return 0 if result["success"] else 1
            
        except Exception as e:
            print(f"❌ Knowledge base migration failed: {str(e)}")
            return 1
    
    def show_help(self) -> int:
        """Show help message."""
        print(__doc__)
        return 0


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Database Migration Tool for Agentic AI Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=["migrate", "status", "health", "kb-migrate", "help"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    cli = DatabaseMigrationCLI()
    
    # Execute the requested command
    if args.command == "migrate":
        return await cli.run_migrations()
    elif args.command == "status":
        return await cli.show_status()
    elif args.command == "health":
        return await cli.check_health()
    elif args.command == "kb-migrate":
        return await cli.migrate_knowledge_bases()
    elif args.command == "help":
        return cli.show_help()
    else:
        print(f"❌ Unknown command: {args.command}")
        cli.show_help()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Migration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        sys.exit(1)
