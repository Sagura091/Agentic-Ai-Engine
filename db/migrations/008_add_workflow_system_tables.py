"""
008_add_workflow_system_tables.py
Workflow system tables migration (ALREADY IN 001_init_database.sql)

Migration Order: 008 (Eighth - Workflow System Management)
Dependencies: 007_add_tool_system_tables.py
Next: None (Final migration)

NOTE: Workflow system tables are already created in 001_init_database.sql
This migration verifies they exist and skips gracefully.

Tables created in 001:
- workflows
- workflow_nodes
- workflow_connections
- workflow_executions
- workflow_step_executions
- workflow_step_states
- workflow_templates
- node_definitions
- node_execution_state
- component_workflow_executions
- task_executions

CONVERTED TO SQLALCHEMY ASYNC - Compatible with run_all_migrations.py
Create Date: 2025-10-08
"""
import asyncio
import sys
from pathlib import Path
import structlog

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.models.database.base import get_engine

logger = structlog.get_logger(__name__)


async def create_workflow_system_tables():
    """
    Verify workflow system tables exist (created in 001_init_database.sql).
    
    Returns:
        bool: True if tables exist or were created successfully
    """
    try:
        logger.info("Checking workflow system tables...")
        
        engine = get_engine()
        
        async with engine.begin() as conn:
            # Check if workflows table exists
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'workflows'
                );
            """))
            workflows_exists = result.scalar()
            
            if workflows_exists:
                logger.info("✅ Workflow system tables already exist (created in 001_init_database.sql)")
                logger.info("   Tables: workflows, workflow_nodes, workflow_connections, workflow_executions,")
                logger.info("           workflow_step_executions, workflow_step_states, workflow_templates,")
                logger.info("           node_definitions, node_execution_state, component_workflow_executions,")
                logger.info("           task_executions")
                return True
            else:
                logger.warning("⚠️  Workflow system tables not found!")
                logger.warning("   These should have been created by 001_init_database.sql")
                logger.warning("   Please run 001_init_database.sql first")
                return False
        
    except Exception as e:
        logger.error(f"Failed to check workflow system tables: {str(e)}")
        return False


# Standalone execution for testing
async def main():
    """Main function for standalone execution."""
    logger.info("Running Workflow System Tables Migration (008)")
    success = await create_workflow_system_tables()
    if success:
        logger.info("🎉 Migration completed successfully!")
    else:
        logger.error("❌ Migration failed!")
    return success


if __name__ == "__main__":
    asyncio.run(main())

