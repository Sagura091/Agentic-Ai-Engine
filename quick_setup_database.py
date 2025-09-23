"""
Quick Database Setup Script for Agentic AI System

This script provides a simple, reliable way to set up the database
after Docker volumes are cleared. It focuses on creating the essential
tables needed to get the system running.

Usage: python quick_setup_database.py
"""

import asyncio
import sys
import time
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.models.database.base import get_engine
    from sqlalchemy import text
    import structlog
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure you're in the project root and dependencies are installed")
    sys.exit(1)

logger = structlog.get_logger(__name__)


async def wait_for_database(max_attempts: int = 30, delay: int = 2) -> bool:
    """Wait for database to be available."""
    print("Waiting for database connection...")
    
    for attempt in range(max_attempts):
        try:
            engine = get_engine()
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            print("Database connection established!")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1}/{max_attempts} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Failed to connect to database after {max_attempts} attempts: {e}")
                return False
    
    return False


async def create_essential_tables() -> bool:
    """Create essential tables for the system."""
    try:
        print("Creating essential database tables...")
        
        engine = get_engine()
        async with engine.begin() as conn:
            # Create extensions
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pg_trgm"'))
            
            # Create users table (essential for authentication) - MATCHES UserDB model
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    username VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    hashed_password VARCHAR(255) NOT NULL,
                    password_salt VARCHAR(255),
                    is_active BOOLEAN DEFAULT true,
                    user_group VARCHAR(50) DEFAULT 'user' CHECK (user_group IN ('user', 'moderator', 'admin')),
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP WITH TIME ZONE,
                    last_login TIMESTAMP WITH TIME ZONE,
                    login_count INTEGER DEFAULT 0,
                    api_keys JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            
            # Create user_sessions table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_token VARCHAR(255) UNIQUE NOT NULL,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    ip_address INET,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT true
                )
            """))
            
            # Create agents table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS agents (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    agent_type VARCHAR(100) NOT NULL DEFAULT 'general',
                    model VARCHAR(255) NOT NULL DEFAULT 'llama3.2:latest',
                    model_provider VARCHAR(50) NOT NULL DEFAULT 'ollama',
                    temperature FLOAT DEFAULT 0.7,
                    max_tokens INTEGER DEFAULT 2048,
                    capabilities JSONB DEFAULT '[]',
                    tools JSONB DEFAULT '[]',
                    system_prompt TEXT,
                    status VARCHAR(50) DEFAULT 'active',
                    last_activity TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    agent_metadata JSONB DEFAULT '{}'
                )
            """))
            
            # Create conversations table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
                    title VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """))
            
            # Create messages table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """))
            
            # Create basic indexes
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_users_user_group ON users(user_group)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)"))
            
        print("Essential tables created successfully!")
        return True
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False


async def verify_setup() -> bool:
    """Verify that the database setup was successful."""
    try:
        print("Verifying database setup...")
        
        engine = get_engine()
        async with engine.begin() as conn:
            # Check for essential tables
            essential_tables = ['users', 'user_sessions', 'agents', 'conversations', 'messages']
            existing_tables = []
            
            for table in essential_tables:
                result = await conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table}'
                    )
                """))
                exists = result.scalar()
                if exists:
                    existing_tables.append(table)
                    print(f"  Found: {table}")
            
            success = len(existing_tables) == len(essential_tables)
            
            if success:
                print("Database setup verification passed!")
            else:
                missing = set(essential_tables) - set(existing_tables)
                print(f"Missing tables: {missing}")
            
            return success
            
    except Exception as e:
        print(f"Error verifying setup: {e}")
        return False


async def main() -> int:
    """Main setup function."""
    print("AGENTIC AI QUICK DATABASE SETUP")
    print("=" * 50)
    print("This script creates essential tables for the system.")
    print()
    
    try:
        # Step 1: Wait for database
        if not await wait_for_database():
            return 1
        
        # Step 2: Create essential tables
        if not await create_essential_tables():
            return 1
        
        # Step 3: Verify setup
        if not await verify_setup():
            print("Setup completed with warnings.")
        
        print()
        print("DATABASE SETUP COMPLETE!")
        print("=" * 50)
        print("Your database is ready for basic use!")
        print("You can now:")
        print("1. Start the backend: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("2. Start the frontend: cd frontend && npm run dev")
        print("3. Register your admin user at: http://localhost:8000/register")
        print()
        print("Note: For full features, you may need to run additional migrations later.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error during setup: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
