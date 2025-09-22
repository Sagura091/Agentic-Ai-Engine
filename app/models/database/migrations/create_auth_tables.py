"""
Database migration to create authentication and user management tables.

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
    UserDB, ProjectDB, ProjectMemberDB,
    ConversationDB, MessageDB, NotificationDB,
    UserAPIKeyDB, UserAgentDB, UserWorkflowDB, KeycloakConfigDB
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
        
        logger.info("Creating authentication tables...")
        
        # Create tables
        async with engine.begin() as conn:
            # Create tables in the correct order (respecting foreign keys)
            await conn.run_sync(Base.metadata.create_all)
            
            # Create indexes for better performance
            await conn.execute(text("""
                -- User table indexes
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
                CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
                
                -- User sessions indexes
                CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active);
                
                -- Projects indexes
                CREATE INDEX IF NOT EXISTS idx_projects_owner_id ON projects(owner_id);
                CREATE INDEX IF NOT EXISTS idx_projects_is_public ON projects(is_public);
                CREATE INDEX IF NOT EXISTS idx_projects_is_archived ON projects(is_archived);
                CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at);
                CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);
                
                -- Project members indexes
                CREATE INDEX IF NOT EXISTS idx_project_members_project_id ON project_members(project_id);
                CREATE INDEX IF NOT EXISTS idx_project_members_user_id ON project_members(user_id);
                CREATE INDEX IF NOT EXISTS idx_project_members_role ON project_members(role);
                CREATE INDEX IF NOT EXISTS idx_project_members_active ON project_members(is_active);
                
                -- Conversations indexes
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_project_id ON conversations(project_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_agent_id ON conversations(agent_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_is_pinned ON conversations(is_pinned);
                CREATE INDEX IF NOT EXISTS idx_conversations_is_archived ON conversations(is_archived);
                
                -- Messages indexes
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
                CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
                
                -- Notifications indexes
                CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
                CREATE INDEX IF NOT EXISTS idx_notifications_type ON notifications(type);
                CREATE INDEX IF NOT EXISTS idx_notifications_priority ON notifications(priority);
                CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON notifications(is_read);
                CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at);

                -- User API Keys indexes
                CREATE INDEX IF NOT EXISTS idx_user_api_keys_user_id ON user_api_keys(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_api_keys_provider ON user_api_keys(provider);
                CREATE INDEX IF NOT EXISTS idx_user_api_keys_is_active ON user_api_keys(is_active);
                CREATE INDEX IF NOT EXISTS idx_user_api_keys_is_default ON user_api_keys(is_default);
                CREATE INDEX IF NOT EXISTS idx_user_api_keys_key_hash ON user_api_keys(key_hash);

                -- User Agents indexes
                CREATE INDEX IF NOT EXISTS idx_user_agents_user_id ON user_agents(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_agents_agent_id ON user_agents(agent_id);
                CREATE INDEX IF NOT EXISTS idx_user_agents_agent_type ON user_agents(agent_type);
                CREATE INDEX IF NOT EXISTS idx_user_agents_is_active ON user_agents(is_active);

                -- User Workflows indexes
                CREATE INDEX IF NOT EXISTS idx_user_workflows_user_id ON user_workflows(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_workflows_workflow_id ON user_workflows(workflow_id);
                CREATE INDEX IF NOT EXISTS idx_user_workflows_is_active ON user_workflows(is_active);
            """))
            
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
