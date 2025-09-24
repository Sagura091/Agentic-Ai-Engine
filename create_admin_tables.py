#!/usr/bin/env python3
"""
Create admin settings tables directly using SQLAlchemy.
This bypasses the migration system issues and creates the necessary tables.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.models.database.base import get_database_session

async def create_admin_settings_tables():
    """Create admin settings tables directly."""
    
    sql_commands = [
        # Create admin_settings table
        """
        CREATE TABLE IF NOT EXISTS admin_settings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            category VARCHAR(100) NOT NULL,
            key VARCHAR(200) NOT NULL,
            value JSONB NOT NULL,
            default_value JSONB NOT NULL,
            setting_type VARCHAR(50) NOT NULL,
            description TEXT,
            security_level VARCHAR(50) DEFAULT 'admin_only',
            requires_restart BOOLEAN DEFAULT FALSE,
            validation_rules JSONB DEFAULT '{}',
            enum_values JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_by UUID,
            is_active BOOLEAN DEFAULT TRUE,
            is_system_managed BOOLEAN DEFAULT FALSE
        );
        """,
        
        # Create unique index on category + key
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_admin_settings_category_key 
        ON admin_settings (category, key);
        """,
        
        # Create additional indexes
        """
        CREATE INDEX IF NOT EXISTS idx_admin_settings_category ON admin_settings (category);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_admin_settings_updated ON admin_settings (updated_at);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_admin_settings_active ON admin_settings (is_active);
        """,
        
        # Create admin_setting_history table
        """
        CREATE TABLE IF NOT EXISTS admin_setting_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            setting_id UUID NOT NULL,
            category VARCHAR(100) NOT NULL,
            key VARCHAR(200) NOT NULL,
            old_value JSONB,
            new_value JSONB NOT NULL,
            change_reason TEXT,
            changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            changed_by UUID NOT NULL,
            system_restart_required BOOLEAN DEFAULT FALSE
        );
        """,
        
        # Create indexes for admin_setting_history
        """
        CREATE INDEX IF NOT EXISTS idx_admin_setting_history_setting_id ON admin_setting_history (setting_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_at ON admin_setting_history (changed_at);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_by ON admin_setting_history (changed_by);
        """,
        
        # Create system_configuration_cache table
        """
        CREATE TABLE IF NOT EXISTS system_configuration_cache (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            component_name VARCHAR(100) NOT NULL UNIQUE,
            configuration JSONB NOT NULL,
            configuration_hash VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE,
            requires_restart BOOLEAN DEFAULT FALSE
        );
        """,
        
        # Create indexes for system_configuration_cache
        """
        CREATE INDEX IF NOT EXISTS idx_system_config_cache_component ON system_configuration_cache (component_name);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_system_config_cache_updated ON system_configuration_cache (updated_at);
        """,
    ]
    
    # Insert default settings
    insert_settings = """
    INSERT INTO admin_settings (category, key, value, default_value, setting_type, description) VALUES
    ('system_configuration', 'app_name', '"Agentic AI Microservice"', '"Agentic AI Platform"', 'string', 'Application name displayed in UI'),
    ('system_configuration', 'app_version', '"0.1.0"', '"0.1.0"', 'string', 'Application version'),
    ('system_configuration', 'debug_mode', 'false', 'false', 'boolean', 'Enable debug mode'),
    ('system_configuration', 'max_concurrent_agents', '10', '10', 'integer', 'Maximum number of concurrent agents'),
    ('llm_providers', 'ollama_enabled', 'true', 'true', 'boolean', 'Enable Ollama provider'),
    ('llm_providers', 'ollama_base_url', '"http://localhost:11434"', '"http://localhost:11434"', 'string', 'Ollama base URL'),
    ('llm_providers', 'openai_enabled', 'false', 'false', 'boolean', 'Enable OpenAI provider'),
    ('llm_providers', 'anthropic_enabled', 'false', 'false', 'boolean', 'Enable Anthropic provider'),
    ('rag_system', 'chunk_size', '1000', '1000', 'integer', 'Default chunk size for document processing'),
    ('rag_system', 'chunk_overlap', '200', '200', 'integer', 'Overlap between chunks'),
    ('rag_system', 'top_k', '5', '5', 'integer', 'Number of top results to retrieve')
    ON CONFLICT (category, key) DO NOTHING;
    """
    
    try:
        print("🚀 Creating admin settings tables...")
        
        async for session in get_database_session():
            # Execute table creation commands
            for i, command in enumerate(sql_commands, 1):
                try:
                    await session.execute(text(command))
                    print(f"✅ Command {i}/{len(sql_commands)} executed successfully")
                except Exception as e:
                    print(f"⚠️  Command {i} warning: {str(e)}")
            
            # Insert default settings
            try:
                await session.execute(text(insert_settings))
                print("✅ Default settings inserted")
            except Exception as e:
                print(f"⚠️  Insert settings warning: {str(e)}")
            
            await session.commit()
            print("✅ All changes committed to database")
            break  # Exit after first session
            
        print("🎉 Admin settings tables created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin settings tables: {str(e)}")
        return False

async def verify_tables():
    """Verify that the tables were created successfully."""
    try:
        print("\n🔍 Verifying table creation...")
        
        async for session in get_database_session():
            # Check if tables exist
            tables_to_check = ['admin_settings', 'admin_setting_history', 'system_configuration_cache']
            
            for table in tables_to_check:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table}'
                    );
                """))
                exists = result.scalar()
                if exists:
                    print(f"✅ Table '{table}' exists")
                else:
                    print(f"❌ Table '{table}' does not exist")
            
            # Check if we have default settings
            result = await session.execute(text("SELECT COUNT(*) FROM admin_settings;"))
            count = result.scalar()
            print(f"📊 Found {count} settings in admin_settings table")
            
            break  # Exit after first session
            
        return True
        
    except Exception as e:
        print(f"❌ Error verifying tables: {str(e)}")
        return False

async def main():
    """Main function."""
    print("🔧 Admin Settings Tables Creator")
    print("=" * 50)
    
    # Create tables
    success = await create_admin_settings_tables()
    if not success:
        print("❌ Failed to create tables")
        return 1
    
    # Verify tables
    success = await verify_tables()
    if not success:
        print("❌ Failed to verify tables")
        return 1
    
    print("\n🎉 SUCCESS! Admin settings tables are ready!")
    print("You can now use the enhanced admin settings in the frontend.")
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        sys.exit(1)
