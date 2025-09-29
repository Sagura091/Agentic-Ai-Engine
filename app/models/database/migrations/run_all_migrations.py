"""
Master Migration Runner for All Database Migrations.

This script runs all database migrations in the correct order:
1. Autonomous agent tables (existing)
2. Enhanced platform tables (new)
3. Knowledge base data migration (JSON to database)
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

import structlog

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# from app.models.database.migrations.create_autonomous_tables import AutonomousTablesMigration
from app.models.database.migrations.create_autonomous_tables import DatabaseMigrationManager
from app.models.database.migrations.create_enhanced_tables import EnhancedTablesMigration
from app.services.knowledge_base_migration_service import knowledge_base_migration_service

logger = structlog.get_logger(__name__)


class MasterMigrationRunner:
    """Master migration runner for all database migrations."""
    
    def __init__(self):
        """Initialize the master migration runner."""
        self.migrations = [
            {
                "name": "autonomous_tables",
                "description": "Create autonomous agent tables",
                "runner": self._run_autonomous_tables_migration
            },
            {
                "name": "enhanced_tables", 
                "description": "Create enhanced platform tables",
                "runner": self._run_enhanced_tables_migration
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
        logger.info("🚀 Starting master migration process")
        
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
            
            logger.info(f"📋 Running migration: {migration_name}", description=migration_description)
            
            try:
                migration_result = await migration["runner"]()
                results["migration_results"][migration_name] = migration_result
                
                if migration_result.get("success", False):
                    results["completed_migrations"] += 1
                    logger.info(f"✅ Migration completed: {migration_name}")
                else:
                    results["failed_migrations"] += 1
                    results["success"] = False
                    error_msg = f"Migration failed: {migration_name} - {migration_result.get('message', 'Unknown error')}"
                    results["errors"].append(error_msg)
                    logger.error(f"❌ Migration failed: {migration_name}", error=migration_result.get('message'))
                    
            except Exception as e:
                results["failed_migrations"] += 1
                results["success"] = False
                error_msg = f"Migration exception: {migration_name} - {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"💥 Migration exception: {migration_name}", error=str(e))
        
        # Summary
        if results["success"]:
            logger.info(f"🎉 All migrations completed successfully! ({results['completed_migrations']}/{results['total_migrations']})")
        else:
            logger.error(f"💔 Migration process failed. Completed: {results['completed_migrations']}, Failed: {results['failed_migrations']}")
        
        return results
    
    async def _run_autonomous_tables_migration(self) -> Dict[str, Any]:
        """Run autonomous tables migration."""
        try:
            migration = AutonomousTablesMigration()
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
    
    async def _run_enhanced_tables_migration(self) -> Dict[str, Any]:
        """Run enhanced tables migration."""
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
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get status of all migrations."""
        try:
            # Check autonomous tables
            autonomous_migration = AutonomousTablesMigration()
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
        logger.info("🎯 Master Database Migration Runner")
        logger.info("=" * 50)
        
        runner = MasterMigrationRunner()
        results = await runner.run_all_migrations()
        
        logger.info("=" * 50)
        logger.info("📊 Migration Summary:")
        logger.info(f"   Total Migrations: {results['total_migrations']}")
        logger.info(f"   Completed: {results['completed_migrations']}")
        logger.info(f"   Failed: {results['failed_migrations']}")
        logger.info(f"   Overall Success: {results['success']}")
        
        if results["errors"]:
            logger.info("❌ Errors:")
            for error in results["errors"]:
                logger.info(f"   - {error}")
        
        logger.info("=" * 50)
        
        if results["success"]:
            logger.info("🎉 All migrations completed successfully!")
            return 0
        else:
            logger.error("💔 Some migrations failed. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Master migration failed: {str(e)}")
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
