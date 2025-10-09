"""
003_create_auth_tables.py
Database migration to create authentication and user management tables.

Migration Order: 003 (Third - Authentication & User Management)
Dependencies: 002_create_autonomous_tables.py
Next: 004_create_enhanced_tables.py

This migration creates the core tables needed for user authentication,
project management, conversations, and notifications.
"""

import asyncio
import structlog
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from app.config.settings import get_settings
from app.models.database.base import Base
from app.models.auth import (
    UserDB, ConversationDB, MessageDB
    # REMOVED: UserAPIKeyDB, UserAgentDB, UserWorkflowDB - these models don't exist
    # REMOVED: ProjectDB, ProjectMemberDB, NotificationDB, KeycloakConfigDB
    # Reason: Project management and notifications not implemented, SSO complexity removed
)
from app.models.enhanced_user import UserSession

logger = structlog.get_logger(__name__)


async def create_auth_tables():
    """Create authentication and user management tables."""
    try:
        settings = get_settings()
        
        # Create async engine
        engine = create_async_engine(
            settings.database_url_async,
            echo=True  # Show SQL for debugging
        )
        
        logger.info("Creating essential authentication tables...")
        logger.info("OPTIMIZED SCHEMA: Removed project management, notifications, and SSO complexity")

        # Create tables
        async with engine.begin() as conn:
            # Create tables in the correct order (respecting foreign keys)
            await conn.run_sync(Base.metadata.create_all)
            
            # Create essential indexes individually (SQLAlchemy already creates most indexes from model definitions)
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)"))
            
            logger.info("Database indexes created successfully")
            
        await engine.dispose()
        logger.info("Authentication tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create authentication tables: {str(e)}")
        raise


async def create_default_admin_user():
    """Create a default admin user for initial setup."""
    try:
        from app.services.auth_service import auth_service
        from app.models.auth import UserCreate
        
        # Check if admin user already exists
        settings = get_settings()
        admin_email = getattr(settings, 'DEFAULT_ADMIN_EMAIL', 'admin@localhost')
        admin_username = getattr(settings, 'DEFAULT_ADMIN_USERNAME', 'admin')
        admin_password = getattr(settings, 'DEFAULT_ADMIN_PASSWORD', 'admin123')
        
        try:
            # Try to create admin user
            admin_user_data = UserCreate(
                username=admin_username,
                email=admin_email,
                password=admin_password,
                full_name="System Administrator",
                is_admin=True
            )
            
            user_response, token_response = await auth_service.register_user(admin_user_data)
            logger.info(f"Default admin user created: {admin_username}")
            
        except ValueError as e:
            if "already exists" in str(e).lower():
                logger.info("Default admin user already exists")
            else:
                logger.warning(f"Could not create default admin user: {str(e)}")
        
    except Exception as e:
        logger.warning(f"Failed to create default admin user: {str(e)}")
        # Don't raise - this is optional


async def run_migration():
    """Run the complete authentication migration."""
    logger.info("Starting authentication tables migration...")
    
    try:
        # Create tables and indexes
        await create_auth_tables()
        
        # Create default admin user
        await create_default_admin_user()
        
        logger.info("Authentication migration completed successfully")
        
    except Exception as e:
        logger.error(f"Authentication migration failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run migration directly
    asyncio.run(run_migration())
