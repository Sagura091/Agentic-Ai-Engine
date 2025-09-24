#!/usr/bin/env python3
"""
Standalone script to run the admin settings migration.
This ensures the admin settings tables are properly created in the database.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Boolean, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid
import os

# Database configuration from environment variables
DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
DATABASE_PORT = os.getenv("DATABASE_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "agentic_ai")
DATABASE_USER = os.getenv("DATABASE_USER", "agentic_user")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "agentic_password")

async def create_admin_settings_tables():
    """Create admin settings tables using direct SQLAlchemy."""
    
    try:
        print("🔧 Creating admin settings management tables...")
        
        # Create database engine
        database_url = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # ============================================================================
                # 1. ADMIN SETTINGS TABLE - Core settings storage
                # ============================================================================
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS admin_settings (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        category VARCHAR(100) NOT NULL,
                        key VARCHAR(200) NOT NULL,
                        value JSONB NOT NULL,
                        value_type VARCHAR(50) NOT NULL,
                        description TEXT,
                        default_value JSONB,
                        validation_rules JSONB,
                        is_sensitive BOOLEAN NOT NULL DEFAULT FALSE,
                        requires_restart BOOLEAN NOT NULL DEFAULT FALSE,
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        is_system_managed BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                        created_by UUID NOT NULL,
                        updated_by UUID NOT NULL,
                        UNIQUE(category, key)
                    );
                """))
                
                # Create indexes for admin_settings
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_category ON admin_settings(category);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_key ON admin_settings(key);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_active ON admin_settings(is_active);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_system_managed ON admin_settings(is_system_managed);"))
                
                # ============================================================================
                # 2. ADMIN SETTING HISTORY TABLE - Audit trail for all changes
                # ============================================================================
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS admin_setting_history (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        setting_id UUID NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        key VARCHAR(200) NOT NULL,
                        old_value JSONB,
                        new_value JSONB NOT NULL,
                        change_reason TEXT,
                        changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                        changed_by UUID NOT NULL,
                        system_restart_required BOOLEAN NOT NULL DEFAULT FALSE,
                        applied_at TIMESTAMP WITH TIME ZONE
                    );
                """))
                
                # Create indexes for admin_setting_history
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_setting_id ON admin_setting_history(setting_id);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_at ON admin_setting_history(changed_at);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_by ON admin_setting_history(changed_by);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_category ON admin_setting_history(category);"))
                
                # ============================================================================
                # 3. SYSTEM CONFIGURATION CACHE TABLE - Performance optimization
                # ============================================================================
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS system_configuration_cache (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        cache_key VARCHAR(255) NOT NULL UNIQUE,
                        cache_value JSONB NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
                    );
                """))
                
                # Create indexes for system_configuration_cache
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_config_cache_key ON system_configuration_cache(cache_key);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_config_cache_category ON system_configuration_cache(category);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_config_cache_expires ON system_configuration_cache(expires_at);"))
                
                print("✅ Admin settings tables created successfully!")
                
                # ============================================================================
                # 4. INSERT DEFAULT SETTINGS
                # ============================================================================
                print("📝 Inserting default admin settings...")
                
                # Default admin user ID (this should be replaced with actual admin user ID)
                default_admin_id = "2f84c9c6-a978-4aff-b8ce-1a5e6bf22506"
                
                # System Configuration Settings
                default_settings = [
                    ('system_configuration', 'app_name', '"Agentic AI Engine"', 'string', 'Application name displayed in the UI', 'false'),
                    ('system_configuration', 'debug_mode', 'false', 'boolean', 'Enable debug mode for development', 'true'),
                    ('system_configuration', 'max_agents', '100', 'integer', 'Maximum number of agents allowed', 'false'),
                    ('system_configuration', 'session_timeout', '3600', 'integer', 'Session timeout in seconds', 'false'),
                    ('llm_providers', 'enable_ollama', 'true', 'boolean', 'Enable Ollama local LLM provider', 'false'),
                    ('llm_providers', 'enable_openai', 'false', 'boolean', 'Enable OpenAI API provider', 'false'),
                    ('llm_providers', 'ollama_base_url', '"http://localhost:11434"', 'string', 'Ollama server base URL', 'true'),
                    ('llm_providers', 'default_temperature', '0.7', 'float', 'Default temperature for LLM responses', 'false'),
                    ('rag_system', 'enable_rag', 'true', 'boolean', 'Enable RAG system', 'false'),
                    ('rag_system', 'chunk_size', '1000', 'integer', 'Default chunk size for document processing', 'false'),
                    ('rag_system', 'chunk_overlap', '200', 'integer', 'Overlap between document chunks', 'false'),
                    ('rag_system', 'max_results', '10', 'integer', 'Maximum number of search results', 'false')
                ]
                
                for category, key, value, value_type, description, requires_restart in default_settings:
                    conn.execute(text("""
                        INSERT INTO admin_settings (
                            category, key, value, value_type, description, 
                            default_value, requires_restart, created_by, updated_by
                        ) VALUES (
                            :category, :key, :value::jsonb, :value_type, :description,
                            :value::jsonb, :requires_restart::boolean, :admin_id::uuid, :admin_id::uuid
                        ) ON CONFLICT (category, key) DO NOTHING;
                    """), {
                        'category': category,
                        'key': key,
                        'value': value,
                        'value_type': value_type,
                        'description': description,
                        'requires_restart': requires_restart,
                        'admin_id': default_admin_id
                    })
                
                print(f"✅ Inserted {len(default_settings)} default admin settings!")
                
                # Commit transaction
                trans.commit()
                print("🎉 Admin settings migration completed successfully!")
                
                return True
                
            except Exception as e:
                trans.rollback()
                raise e
                
    except Exception as e:
        print(f"❌ Error creating admin settings tables: {str(e)}")
        return False

async def main():
    """Main function."""
    print("🚀 Admin Settings Migration")
    print("=" * 50)
    
    success = await create_admin_settings_tables()
    if not success:
        return 1
    
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
