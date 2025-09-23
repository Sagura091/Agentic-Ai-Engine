"""
Create basic authentication tables for a fresh start.

This script creates only the essential tables needed for user authentication.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.database.base import get_engine
from sqlalchemy import text
import structlog

logger = structlog.get_logger(__name__)


async def create_basic_auth_tables():
    """Create basic authentication tables."""
    try:
        engine = get_engine()
        
        async with engine.begin() as conn:
            # Create users table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'moderator', 'admin')),
                    is_active BOOLEAN DEFAULT true,
                    is_verified BOOLEAN DEFAULT false,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_login TIMESTAMP WITH TIME ZONE,
                    profile_data JSONB DEFAULT '{}'::jsonb
                );
            """))
            
            # Create user_sessions table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_token VARCHAR(255) UNIQUE NOT NULL,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    ip_address INET,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT true
                );
            """))
            
            # Create indexes
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);"))
            
            logger.info("✅ Basic authentication tables created successfully")
            
    except Exception as e:
        logger.error(f"❌ Error creating auth tables: {e}")
        raise


async def main():
    """Main function to create basic auth tables."""
    try:
        print("🔧 Creating basic authentication tables...")
        await create_basic_auth_tables()
        print("✅ Authentication tables created successfully!")
        print("🎯 You can now register a new admin user at: http://localhost:8000/register")
        
    except Exception as e:
        print(f"❌ Failed to create auth tables: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
