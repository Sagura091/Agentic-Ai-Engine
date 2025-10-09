"""
Master Migration Runner for All Database Migrations.

This script runs all database migrations in the correct order:
1. 001_init_database.sql - Database initialization (SQL)
2. 002_create_autonomous_tables.py - Autonomous agent tables (Python)
3. 003_fix_schema_issues.sql - Schema bug fixes (SQL)
4. 004_create_enhanced_tables.py - Enhanced platform tables (Python)
5. 005_add_document_tables.py - Document storage tables (skipped - in 001)
6. 006_add_admin_settings_tables.py - Admin settings (skipped - optional)
7. 007_add_tool_system_tables.py - Tool system (skipped - in 001)
8. 008_add_workflow_system_tables.py - Workflow system (skipped - in 001)
9. Knowledge base data migration - JSON to database (Python)

Location: db/migrations/run_all_migrations.py
Usage: python db/migrations/run_all_migrations.py
"""

import asyncio
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List

import structlog

# Add the project root to the path (go up two levels from migrations/ to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also add to PYTHONPATH for submodule imports
import os
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')

# Import migration classes using importlib (numeric filenames)
migrations_dir = Path(__file__).parent

# Load autonomous tables migration (002)
spec_autonomous = importlib.util.spec_from_file_location(
    "autonomous_migration", migrations_dir / "002_create_autonomous_tables.py"
)
autonomous_module = importlib.util.module_from_spec(spec_autonomous)
spec_autonomous.loader.exec_module(autonomous_module)
DatabaseMigrationManager = autonomous_module.DatabaseMigrationManager

# Load auth tables migration (004)
spec_auth = importlib.util.spec_from_file_location(
    "auth_migration", migrations_dir / "004_create_auth_tables.py"
)
auth_module = importlib.util.module_from_spec(spec_auth)
spec_auth.loader.exec_module(auth_module)
create_auth_tables = auth_module.create_auth_tables

# Load enhanced tables migration (005)
spec_enhanced = importlib.util.spec_from_file_location(
    "enhanced_migration", migrations_dir / "005_create_enhanced_tables.py"
)
enhanced_module = importlib.util.module_from_spec(spec_enhanced)
spec_enhanced.loader.exec_module(enhanced_module)
EnhancedTablesMigration = enhanced_module.EnhancedTablesMigration

# Load admin settings migration (006)
spec_admin = importlib.util.spec_from_file_location(
    "admin_migration", migrations_dir / "006_add_admin_settings_tables.py"
)
admin_module = importlib.util.module_from_spec(spec_admin)
spec_admin.loader.exec_module(admin_module)
create_admin_settings_tables = admin_module.create_admin_settings_tables
insert_default_settings = admin_module.insert_default_settings

# Load tool system migration (007)
spec_tool = importlib.util.spec_from_file_location(
    "tool_migration", migrations_dir / "007_add_tool_system_tables.py"
)
tool_module = importlib.util.module_from_spec(spec_tool)
spec_tool.loader.exec_module(tool_module)
create_tool_system_tables = tool_module.create_tool_system_tables

# Load workflow system migration (008)
spec_workflow = importlib.util.spec_from_file_location(
    "workflow_migration", migrations_dir / "008_add_workflow_system_tables.py"
)
workflow_module = importlib.util.module_from_spec(spec_workflow)
spec_workflow.loader.exec_module(workflow_module)
create_workflow_system_tables = workflow_module.create_workflow_system_tables

# Import knowledge base service
sys.path.insert(0, str(project_root))
from app.services.knowledge_base_migration_service import knowledge_base_migration_service

logger = structlog.get_logger(__name__)


class MasterMigrationRunner:
    """Master migration runner for all database migrations."""
    
    def __init__(self):
        """Initialize the master migration runner."""
        self.migrations = [
            {
                "name": "001_init_database",
                "description": "Run SQL database initialization (001_init_database.sql)",
                "runner": lambda: self._run_sql_file("001_init_database.sql")
            },
            {
                "name": "002_autonomous_tables",
                "description": "Create autonomous agent tables (002_create_autonomous_tables.py)",
                "runner": self._run_autonomous_tables_migration
            },
            {
                "name": "003_fix_schema_issues",
                "description": "Fix schema bugs and inconsistencies (003_fix_schema_issues.sql)",
                "runner": lambda: self._run_sql_file("003_fix_schema_issues.sql")
            },
            {
                "name": "004_auth_tables",
                "description": "Create authentication and user tables (004_create_auth_tables.py)",
                "runner": self._run_auth_tables_migration
            },
            {
                "name": "005_enhanced_tables",
                "description": "Create enhanced platform tables (005_create_enhanced_tables.py)",
                "runner": self._run_enhanced_tables_migration
            },
            {
                "name": "006_admin_settings",
                "description": "Create admin settings tables (006_add_admin_settings_tables.py)",
                "runner": self._run_admin_settings_migration
            },
            {
                "name": "007_tool_system",
                "description": "Verify tool system tables (007_add_tool_system_tables.py)",
                "runner": self._run_tool_system_migration
            },
            {
                "name": "008_workflow_system",
                "description": "Verify workflow system tables (008_add_workflow_system_tables.py)",
                "runner": self._run_workflow_system_migration
            },
            {
                "name": "knowledge_base_data",
                "description": "Migrate knowledge base data from JSON to database",
                "runner": self._run_knowledge_base_migration
            }
        ]
    
    async def run_all_migrations(self) -> Dict[str, Any]:
        """
        Run all migrations in order.

        Returns:
            Dict containing overall migration results
        """
        logger.info("Starting master migration process")

        results = {
            "success": True,
            "total_migrations": len(self.migrations),
            "completed_migrations": 0,
            "failed_migrations": 0,
            "migration_results": {},
            "errors": []
        }

        for migration in self.migrations:
            migration_name = migration["name"]
            migration_description = migration["description"]

            logger.info(f"Running migration: {migration_name}", description=migration_description)

            try:
                migration_result = await migration["runner"]()
                results["migration_results"][migration_name] = migration_result

                if migration_result.get("success", False):
                    results["completed_migrations"] += 1
                    logger.info(f"[SUCCESS] Migration completed: {migration_name}")
                else:
                    results["failed_migrations"] += 1
                    results["success"] = False
                    error_msg = f"Migration failed: {migration_name} - {migration_result.get('message', 'Unknown error')}"
                    results["errors"].append(error_msg)
                    logger.error(f"[FAILED] Migration failed: {migration_name}", error=migration_result.get('message'))

            except Exception as e:
                results["failed_migrations"] += 1
                results["success"] = False
                error_msg = f"Migration exception: {migration_name} - {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"[ERROR] Migration exception: {migration_name}", error=str(e))

        # Summary
        if results["success"]:
            logger.info(f"All migrations completed successfully! ({results['completed_migrations']}/{results['total_migrations']})")
        else:
            logger.error(f"Migration process failed. Completed: {results['completed_migrations']}, Failed: {results['failed_migrations']}")

        return results

    async def _run_sql_file(self, filename: str) -> Dict[str, Any]:
        """
        Run a SQL file migration.

        Args:
            filename: Name of the SQL file to execute

        Returns:
            Dict containing migration results
        """
        try:
            from app.models.database.base import get_engine
            from sqlalchemy import text
            import re

            # Path to SQL file
            sql_file_path = Path(__file__).parent / filename

            if not sql_file_path.exists():
                logger.warning(f"SQL file not found: {sql_file_path} - skipping")
                return {
                    "success": True,
                    "message": f"SQL file not found: {filename} - skipped",
                    "skipped": True
                }

            logger.info(f"Executing SQL file: {filename}")

            # Read SQL file
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            # Execute SQL
            engine = get_engine()
            async with engine.begin() as conn:
                # Handle DO $$ blocks specially - they need to be executed as complete units
                # Split by DO $$ blocks first
                parts = re.split(r'(DO \$\$.*?\$\$;)', sql_content, flags=re.DOTALL)

                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # If it's a DO block, execute it as-is
                    if part.startswith('DO $$'):
                        try:
                            await conn.execute(text(part))
                        except Exception as e:
                            error_msg = str(e)
                            if "already exists" not in error_msg.lower() and "duplicate" not in error_msg.lower():
                                logger.warning(f"SQL DO block warning: {error_msg[:100]}")
                    else:
                        # Split regular statements by semicolon
                        statements = [s.strip() for s in part.split(';') if s.strip()]

                        for statement in statements:
                            # Skip comments, empty statements, and psql commands
                            if statement.startswith('--') or statement.startswith('\\') or not statement:
                                continue

                            try:
                                await conn.execute(text(statement))
                            except Exception as e:
                                # Log but continue - some statements might fail if already exists
                                error_msg = str(e)
                                if "already exists" not in error_msg.lower() and "duplicate" not in error_msg.lower():
                                    logger.warning(f"SQL statement warning: {error_msg[:100]}")

            logger.info(f"SQL file {filename} completed successfully")
            return {
                "success": True,
                "message": f"SQL file {filename} executed successfully"
            }

        except Exception as e:
            logger.error(f"SQL file {filename} failed: {str(e)}")
            return {
                "success": False,
                "message": f"SQL file {filename} failed: {str(e)}"
            }

    async def _run_autonomous_tables_migration(self) -> Dict[str, Any]:
        """Run autonomous tables migration (002)."""
        try:
            migration = DatabaseMigrationManager()
            await migration.initialize()
            success = await migration.create_autonomous_tables()

            return {
                "success": success,
                "message": "Autonomous tables migration completed" if success else "Autonomous tables migration failed"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Autonomous tables migration failed: {str(e)}"
            }

    async def _run_auth_tables_migration(self) -> Dict[str, Any]:
        """Run auth tables migration (004)."""
        try:
            success = await create_auth_tables()

            return {
                "success": success if success is not None else True,
                "message": "Auth tables migration completed" if success else "Auth tables migration failed"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Auth tables migration failed: {str(e)}"
            }

    async def _run_enhanced_tables_migration(self) -> Dict[str, Any]:
        """Run enhanced tables migration (005)."""
        try:
            migration = EnhancedTablesMigration()
            success = await migration.create_enhanced_tables()

            return {
                "success": success,
                "message": "Enhanced tables migration completed" if success else "Enhanced tables migration failed"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Enhanced tables migration failed: {str(e)}"
            }
    
    async def _run_knowledge_base_migration(self) -> Dict[str, Any]:
        """Run knowledge base data migration."""
        try:
            result = await knowledge_base_migration_service.migrate_from_json()
            return result

        except Exception as e:
            return {
                "success": False,
                "message": f"Knowledge base migration failed: {str(e)}"
            }

    async def _run_admin_settings_migration(self) -> Dict[str, Any]:
        """Run admin settings tables migration (006)."""
        try:
            success = await create_admin_settings_tables()
            if success:
                # Insert default settings
                await insert_default_settings()

            return {
                "success": success,
                "message": "Admin settings migration completed" if success else "Admin settings migration failed"
            }
        except Exception as e:
            logger.error(f"Admin settings migration failed: {str(e)}")
            return {
                "success": False,
                "message": f"Admin settings migration failed: {str(e)}"
            }

    async def _run_tool_system_migration(self) -> Dict[str, Any]:
        """Run tool system tables migration (007) - Verification only."""
        try:
            success = await create_tool_system_tables()
            return {
                "success": success,
                "message": "Tool system tables verified" if success else "Tool system tables not found"
            }
        except Exception as e:
            logger.error(f"Tool system migration failed: {str(e)}")
            return {
                "success": False,
                "message": f"Tool system migration failed: {str(e)}"
            }

    async def _run_workflow_system_migration(self) -> Dict[str, Any]:
        """Run workflow system tables migration (008) - Verification only."""
        try:
            success = await create_workflow_system_tables()
            return {
                "success": success,
                "message": "Workflow system tables verified" if success else "Workflow system tables not found"
            }
        except Exception as e:
            logger.error(f"Workflow system migration failed: {str(e)}")
            return {
                "success": False,
                "message": f"Workflow system migration failed: {str(e)}"
            }
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get status of all migrations."""
        try:
            # Check autonomous tables
            autonomous_migration = DatabaseMigrationManager()
            await autonomous_migration.initialize()  # Initialize engine and session factory
            autonomous_applied = await autonomous_migration.is_migration_applied("create_autonomous_tables_v1")

            # Check enhanced tables
            enhanced_migration = EnhancedTablesMigration()
            enhanced_applied = await enhanced_migration.is_migration_applied("create_enhanced_tables_v1")
            
            # Check knowledge base migration
            kb_status = await knowledge_base_migration_service.get_migration_status()
            
            return {
                "autonomous_tables": {
                    "applied": autonomous_applied,
                    "status": "completed" if autonomous_applied else "pending"
                },
                "enhanced_tables": {
                    "applied": enhanced_applied,
                    "status": "completed" if enhanced_applied else "pending"
                },
                "knowledge_base_data": kb_status,
                "overall_status": "completed" if (autonomous_applied and enhanced_applied and kb_status.get("status") == "completed") else "pending"
            }
            
        except Exception as e:
            logger.error("Failed to get migration status", error=str(e))
            return {
                "error": str(e),
                "overall_status": "error"
            }
    
    async def rollback_migration(self, migration_name: str) -> Dict[str, Any]:
        """
        Rollback a specific migration (if supported).
        
        Args:
            migration_name: Name of migration to rollback
            
        Returns:
            Dict containing rollback results
        """
        logger.warning(f"Rollback requested for migration: {migration_name}")
        
        # For now, rollback is not implemented as it's complex and dangerous
        # In production, you would implement proper rollback scripts
        return {
            "success": False,
            "message": f"Rollback not implemented for migration: {migration_name}. Manual intervention required.",
            "migration_name": migration_name
        }


async def main():
    """Main function to run all migrations."""
    try:
        logger.info("Master Database Migration Runner")
        logger.info("=" * 50)

        runner = MasterMigrationRunner()
        results = await runner.run_all_migrations()

        logger.info("=" * 50)
        logger.info("Migration Summary:")
        logger.info(f"   Total Migrations: {results['total_migrations']}")
        logger.info(f"   Completed: {results['completed_migrations']}")
        logger.info(f"   Failed: {results['failed_migrations']}")
        logger.info(f"   Overall Success: {results['success']}")

        if results["errors"]:
            logger.info("Errors:")
            for error in results["errors"]:
                logger.info(f"   - {error}")

        logger.info("=" * 50)

        if results["success"]:
            logger.info("All migrations completed successfully!")
            return 0
        else:
            logger.error("Some migrations failed. Check logs for details.")
            return 1

    except Exception as e:
        logger.error(f"Master migration failed: {str(e)}")
        return 1


if __name__ == "__main__":
    # Configure logging
    import structlog
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
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run migrations
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
