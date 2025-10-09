"""
007_add_tool_system_tables.py
Tool system tables migration (ALREADY IN 001_init_database.sql)

Migration Order: 007 (Seventh - Tool System Management)
Dependencies: 006_add_admin_settings_tables.py
Next: 008_add_workflow_system_tables.py

NOTE: Tool system tables are already created in 001_init_database.sql
This migration verifies they exist and skips gracefully.

Tables created in 001:
- tools
- tool_categories
- tool_templates
- tool_executions
- agent_tools

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


async def create_tool_system_tables():
    """
    Verify tool system tables exist (created in 001_init_database.sql).
    
    Returns:
        bool: True if tables exist or were created successfully
    """
    try:
        logger.info("Checking tool system tables...")
        
        engine = get_engine()
        
        async with engine.begin() as conn:
            # Check if tools table exists
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'tools'
                );
            """))
            tools_exists = result.scalar()
            
            if tools_exists:
                logger.info("✅ Tool system tables already exist (created in 001_init_database.sql)")
                logger.info("   Tables: tools, tool_categories, tool_templates, tool_executions, agent_tools")
                return True
            else:
                logger.warning("⚠️  Tool system tables not found!")
                logger.warning("   These should have been created by 001_init_database.sql")
                logger.warning("   Please run 001_init_database.sql first")
                return False
        
    except Exception as e:
        logger.error(f"Failed to check tool system tables: {str(e)}")
        return False


# Standalone execution for testing
async def main():
    """Main function for standalone execution."""
    logger.info("Running Tool System Tables Migration (007)")
    success = await create_tool_system_tables()
    if success:
        logger.info("🎉 Migration completed successfully!")
    else:
        logger.error("❌ Migration failed!")
    return success


if __name__ == "__main__":
    asyncio.run(main())

