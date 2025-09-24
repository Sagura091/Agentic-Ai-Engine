"""
006_add_admin_settings_tables.py
Add comprehensive admin settings management tables

Migration Order: 006 (Sixth - Admin Settings Management)
Dependencies: 005_add_document_tables.py
Next: None (Latest migration)

Original Revision ID: 006_add_admin_settings_tables
Create Date: 2025-09-23 15:45:00.000000
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from alembic import op
    import sqlalchemy as sa
    from sqlalchemy.dialects import postgresql
    import uuid
except ImportError:
    # Fallback for direct execution without alembic
    import sqlalchemy as sa
    from sqlalchemy.dialects import postgresql
    import uuid
    from sqlalchemy import create_engine, text
    from app.core.config import settings

    # Create a mock op object for direct execution
    class MockOp:
        def __init__(self, engine):
            self.engine = engine

        def create_table(self, table_name, *columns, **kwargs):
            # This is a simplified version - in real usage, alembic handles this
            pass

        def create_unique_constraint(self, name, table, columns):
            pass

        def create_index(self, name, table, columns):
            pass

        def drop_table(self, table_name):
            pass

        def execute(self, sql):
            with self.engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()

    # Create engine for direct execution
    engine = create_engine(f"postgresql://{settings.DATABASE_USER}:{settings.DATABASE_PASSWORD}@{settings.DATABASE_HOST}:{settings.DATABASE_PORT}/{settings.DATABASE_NAME}")
    op = MockOp(engine)

# revision identifiers, used by Alembic.
revision = '006_add_admin_settings_tables'
down_revision = '001_add_document_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Create comprehensive admin settings management tables."""
    
    print("🔧 Creating admin settings management tables...")
    
    # ============================================================================
    # 1. ADMIN SETTINGS TABLE - Core settings storage
    # ============================================================================
    op.create_table(
        'admin_settings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('key', sa.String(200), nullable=False, index=True),
        sa.Column('value', postgresql.JSONB, nullable=False),
        sa.Column('value_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('default_value', postgresql.JSONB, nullable=True),
        sa.Column('validation_rules', postgresql.JSONB, nullable=True),
        sa.Column('is_sensitive', sa.Boolean, nullable=False, default=False),
        sa.Column('requires_restart', sa.Boolean, nullable=False, default=False),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('is_system_managed', sa.Boolean, nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=False),
        schema=None
    )
    
    # Create unique constraint for category + key combination
    op.create_unique_constraint('uq_admin_settings_category_key', 'admin_settings', ['category', 'key'])
    
    # Create indexes for performance
    op.create_index('idx_admin_settings_category', 'admin_settings', ['category'])
    op.create_index('idx_admin_settings_key', 'admin_settings', ['key'])
    op.create_index('idx_admin_settings_active', 'admin_settings', ['is_active'])
    op.create_index('idx_admin_settings_system_managed', 'admin_settings', ['is_system_managed'])
    
    # ============================================================================
    # 2. ADMIN SETTING HISTORY TABLE - Audit trail for all changes
    # ============================================================================
    op.create_table(
        'admin_setting_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('setting_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('key', sa.String(200), nullable=False),
        sa.Column('old_value', postgresql.JSONB, nullable=True),
        sa.Column('new_value', postgresql.JSONB, nullable=False),
        sa.Column('change_reason', sa.Text, nullable=True),
        sa.Column('changed_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('changed_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('system_restart_required', sa.Boolean, nullable=False, default=False),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
        schema=None
    )
    
    # Create indexes for history table
    op.create_index('idx_admin_setting_history_setting_id', 'admin_setting_history', ['setting_id'])
    op.create_index('idx_admin_setting_history_changed_at', 'admin_setting_history', ['changed_at'])
    op.create_index('idx_admin_setting_history_changed_by', 'admin_setting_history', ['changed_by'])
    op.create_index('idx_admin_setting_history_category', 'admin_setting_history', ['category'])
    
    # ============================================================================
    # 3. SYSTEM CONFIGURATION CACHE TABLE - Performance optimization
    # ============================================================================
    op.create_table(
        'system_configuration_cache',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('cache_key', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('cache_value', postgresql.JSONB, nullable=False),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        schema=None
    )
    
    # Create indexes for cache table
    op.create_index('idx_system_config_cache_key', 'system_configuration_cache', ['cache_key'])
    op.create_index('idx_system_config_cache_category', 'system_configuration_cache', ['category'])
    op.create_index('idx_system_config_cache_expires', 'system_configuration_cache', ['expires_at'])
    
    print("✅ Admin settings tables created successfully!")


def downgrade():
    """Drop admin settings management tables."""
    
    print("🗑️ Dropping admin settings management tables...")
    
    # Drop tables in reverse order
    op.drop_table('system_configuration_cache')
    op.drop_table('admin_setting_history')
    op.drop_table('admin_settings')
    
    print("✅ Admin settings tables dropped successfully!")


def insert_default_settings():
    """Insert default admin settings data."""
    
    print("📝 Inserting default admin settings...")
    
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
    for setting in all_settings:
        op.execute(f"""
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
        """)
    
    print(f"✅ Inserted {len(all_settings)} default admin settings!")


# Standalone execution for testing
if __name__ == "__main__":
    print("🚀 Running Admin Settings Migration (006)")
    upgrade()
    insert_default_settings()
    print("🎉 Migration completed successfully!")
