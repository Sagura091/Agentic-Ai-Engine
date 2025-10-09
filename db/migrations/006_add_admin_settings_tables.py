"""
006_add_admin_settings_tables.py
Add comprehensive admin settings management tables

Migration Order: 006 (Sixth - Admin Settings Management)
Dependencies: 005_create_enhanced_tables.py
Next: 007_add_tool_system_tables.py

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
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


async def create_admin_settings_tables():
    """Create comprehensive admin settings management tables."""
    try:
        logger.info("Creating admin settings management tables...")

        engine = get_engine()

        async with engine.begin() as conn:
            # ============================================================================
            # 1. ADMIN SETTINGS TABLE - Core settings storage
            # ============================================================================
            await conn.execute(text("""
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
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_by UUID NOT NULL,
                    updated_by UUID NOT NULL
                );
            """))

            # Create unique constraint for category + key combination
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'uq_admin_settings_category_key'
                    ) THEN
                        ALTER TABLE admin_settings ADD CONSTRAINT uq_admin_settings_category_key UNIQUE (category, key);
                    END IF;
                END $$;
            """))

            # Create indexes for performance
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_category ON admin_settings(category);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_key ON admin_settings(key);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_active ON admin_settings(is_active);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_settings_system_managed ON admin_settings(is_system_managed);"))

            # ============================================================================
            # 2. ADMIN SETTING HISTORY TABLE - Audit trail for all changes
            # ============================================================================
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS admin_setting_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    setting_id UUID NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    key VARCHAR(200) NOT NULL,
                    old_value JSONB,
                    new_value JSONB NOT NULL,
                    change_reason TEXT,
                    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    changed_by UUID NOT NULL,
                    system_restart_required BOOLEAN NOT NULL DEFAULT FALSE,
                    applied_at TIMESTAMPTZ
                );
            """))

            # Create indexes for history table
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_setting_id ON admin_setting_history(setting_id);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_at ON admin_setting_history(changed_at);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_changed_by ON admin_setting_history(changed_by);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_admin_setting_history_category ON admin_setting_history(category);"))

            # ============================================================================
            # 3. SYSTEM CONFIGURATION CACHE TABLE - Performance optimization
            # ============================================================================
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS system_configuration_cache (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    cache_key VARCHAR(255) NOT NULL UNIQUE,
                    cache_value JSONB NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    expires_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """))

            # Create indexes for cache table
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_config_cache_key ON system_configuration_cache(cache_key);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_config_cache_category ON system_configuration_cache(category);"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_config_cache_expires ON system_configuration_cache(expires_at);"))

        logger.info("[SUCCESS] Admin settings tables created successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to create admin settings tables: {str(e)}")
        return False


async def insert_default_settings():
    """Insert default admin settings data."""
    try:
        logger.info("Inserting default admin settings...")

        engine = get_engine()

        # Default admin user ID (this should be replaced with actual admin user ID)
        default_admin_id = "2f84c9c6-a978-4aff-b8ce-1a5e6bf22506"

        # System Configuration Settings
        system_settings = [
            {
                'category': 'system_configuration',
                'key': 'app_name',
                'value': '"Agentic AI Engine"',
                'value_type': 'string',
                'description': 'Application name displayed in the UI',
                'default_value': '"Agentic AI Engine"',
                'requires_restart': False
            },
            {
                'category': 'system_configuration',
                'key': 'debug_mode',
                'value': 'false',
                'value_type': 'boolean',
                'description': 'Enable debug mode for development',
                'default_value': 'false',
                'requires_restart': True
            },
            {
                'category': 'system_configuration',
                'key': 'max_agents',
                'value': '100',
                'value_type': 'integer',
                'description': 'Maximum number of agents allowed',
                'default_value': '100',
                'requires_restart': False
            },
            {
                'category': 'system_configuration',
                'key': 'session_timeout',
                'value': '3600',
                'value_type': 'integer',
                'description': 'Session timeout in seconds',
                'default_value': '3600',
                'requires_restart': False
            }
        ]

        # LLM Provider Settings
        llm_settings = [
            {
                'category': 'llm_providers',
                'key': 'enable_ollama',
                'value': 'true',
                'value_type': 'boolean',
                'description': 'Enable Ollama local LLM provider',
                'default_value': 'true',
                'requires_restart': False
            },
            {
                'category': 'llm_providers',
                'key': 'enable_openai',
                'value': 'false',
                'value_type': 'boolean',
                'description': 'Enable OpenAI API provider',
                'default_value': 'false',
                'requires_restart': False
            },
            {
                'category': 'llm_providers',
                'key': 'ollama_base_url',
                'value': '"http://localhost:11434"',
                'value_type': 'string',
                'description': 'Ollama server base URL',
                'default_value': '"http://localhost:11434"',
                'requires_restart': True
            },
            {
                'category': 'llm_providers',
                'key': 'default_temperature',
                'value': '0.7',
                'value_type': 'float',
                'description': 'Default temperature for LLM responses',
                'default_value': '0.7',
                'requires_restart': False
            }
        ]

        # RAG System Settings
        rag_settings = [
            {
                'category': 'rag_system',
                'key': 'enable_rag',
                'value': 'true',
                'value_type': 'boolean',
                'description': 'Enable RAG system',
                'default_value': 'true',
                'requires_restart': False
            },
            {
                'category': 'rag_system',
                'key': 'chunk_size',
                'value': '1000',
                'value_type': 'integer',
                'description': 'Default chunk size for document processing',
                'default_value': '1000',
                'requires_restart': False
            },
            {
                'category': 'rag_system',
                'key': 'chunk_overlap',
                'value': '200',
                'value_type': 'integer',
                'description': 'Overlap between document chunks',
                'default_value': '200',
                'requires_restart': False
            },
            {
                'category': 'rag_system',
                'key': 'max_results',
                'value': '10',
                'value_type': 'integer',
                'description': 'Maximum number of search results',
                'default_value': '10',
                'requires_restart': False
            }
        ]
    
        # Combine all settings
        all_settings = system_settings + llm_settings + rag_settings

        # Insert settings using raw SQL
        async with engine.begin() as conn:
            for setting in all_settings:
                await conn.execute(text(f"""
                    INSERT INTO admin_settings (
                        id, category, key, value, value_type, description,
                        default_value, requires_restart, created_by, updated_by
                    ) VALUES (
                        gen_random_uuid(),
                        '{setting['category']}',
                        '{setting['key']}',
                        '{setting['value']}'::jsonb,
                        '{setting['value_type']}',
                        '{setting['description']}',
                        '{setting['default_value']}'::jsonb,
                        {str(setting['requires_restart']).lower()},
                        '{default_admin_id}'::uuid,
                        '{default_admin_id}'::uuid
                    ) ON CONFLICT (category, key) DO NOTHING;
                """))

        logger.info(f"[SUCCESS] Inserted {len(all_settings)} default admin settings!")
        return True

    except Exception as e:
        logger.error(f"Failed to insert default settings: {str(e)}")
        return False


# Standalone execution for testing
async def main():
    """Main function for standalone execution."""
    logger.info("Running Admin Settings Migration (006)")
    success = await create_admin_settings_tables()
    if success:
        await insert_default_settings()
        logger.info("🎉 Migration completed successfully!")
    else:
        logger.error("❌ Migration failed!")
    return success


if __name__ == "__main__":
    asyncio.run(main())
