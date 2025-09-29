"""
007_add_tool_system_tables.py
Add comprehensive tool system management tables

Migration Order: 007 (Seventh - Tool System Management)
Dependencies: 006_add_admin_settings_tables.py
Next: 008_add_workflow_system_tables.py

Original Revision ID: 007_add_tool_system_tables
Create Date: 2025-09-23 16:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '007_add_tool_system_tables'
down_revision = '006_add_admin_settings_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Create comprehensive tool system management tables."""
    
    print("🔧 Creating tool system management tables...")
    
    # ============================================================================
    # 1. TOOLS TABLE - Core tool definitions
    # ============================================================================
    op.create_table(
        'tools',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('implementation', sa.Text, nullable=False),
        sa.Column('parameters_schema', postgresql.JSONB, nullable=False, default={}),
        sa.Column('return_schema', postgresql.JSONB, nullable=False, default={}),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('author', sa.String(255), nullable=True),
        sa.Column('source_type', sa.String(50), nullable=False, default='generated'),
        sa.Column('original_filename', sa.String(255), nullable=True),
        sa.Column('file_hash', sa.String(255), nullable=True),
        sa.Column('validation_status', sa.String(50), nullable=False, default='pending'),
        sa.Column('validation_score', sa.Float, nullable=False, default=0.0),
        sa.Column('validation_issues', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('validation_warnings', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('dependencies', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('system_requirements', postgresql.JSONB, nullable=False, default={}),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('is_system_tool', sa.Boolean, nullable=False, default=False),
        sa.Column('is_public', sa.Boolean, nullable=False, default=False),
        sa.Column('usage_count', sa.Integer, nullable=False, default=0),
        sa.Column('success_rate', sa.Float, nullable=False, default=0.0),
        sa.Column('average_execution_time_ms', sa.Float, nullable=False, default=0.0),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('tool_metadata', postgresql.JSONB, nullable=False, default={}),
        sa.Column('configuration', postgresql.JSONB, nullable=False, default={}),
        sa.Column('tags', postgresql.JSONB, nullable=False, default=[]),
        schema=None
    )
    
    # Create indexes for tools table
    op.create_index('idx_tools_name', 'tools', ['name'])
    op.create_index('idx_tools_category', 'tools', ['category'])
    op.create_index('idx_tools_active', 'tools', ['is_active'])
    op.create_index('idx_tools_system', 'tools', ['is_system_tool'])
    op.create_index('idx_tools_validation_status', 'tools', ['validation_status'])
    
    # ============================================================================
    # 2. TOOL_CATEGORIES TABLE - Tool categorization
    # ============================================================================
    op.create_table(
        'tool_categories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(100), nullable=False, unique=True, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('parent_category_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('icon', sa.String(100), nullable=True),
        sa.Column('color', sa.String(50), nullable=True),
        sa.Column('sort_order', sa.Integer, nullable=False, default=0),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        schema=None
    )
    
    # Create foreign key for parent category
    op.create_foreign_key('fk_tool_categories_parent', 'tool_categories', 'tool_categories', ['parent_category_id'], ['id'])
    
    # Create indexes for tool_categories table
    op.create_index('idx_tool_categories_name', 'tool_categories', ['name'])
    op.create_index('idx_tool_categories_parent', 'tool_categories', ['parent_category_id'])
    op.create_index('idx_tool_categories_active', 'tool_categories', ['is_active'])
    
    # ============================================================================
    # 3. AGENT_TOOLS TABLE - Agent-tool associations
    # ============================================================================
    op.create_table(
        'agent_tools',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('tool_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('is_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('configuration', postgresql.JSONB, nullable=False, default={}),
        sa.Column('usage_count', sa.Integer, nullable=False, default=0),
        sa.Column('success_count', sa.Integer, nullable=False, default=0),
        sa.Column('failure_count', sa.Integer, nullable=False, default=0),
        sa.Column('average_execution_time_ms', sa.Float, nullable=False, default=0.0),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        schema=None
    )
    
    # Create foreign keys
    op.create_foreign_key('fk_agent_tools_agent', 'agent_tools', 'agents', ['agent_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_agent_tools_tool', 'agent_tools', 'tools', ['tool_id'], ['id'], ondelete='CASCADE')
    
    # Create unique constraint for agent-tool combination
    op.create_unique_constraint('uq_agent_tools_agent_tool', 'agent_tools', ['agent_id', 'tool_id'])
    
    # Create indexes for agent_tools table
    op.create_index('idx_agent_tools_agent', 'agent_tools', ['agent_id'])
    op.create_index('idx_agent_tools_tool', 'agent_tools', ['tool_id'])
    op.create_index('idx_agent_tools_enabled', 'agent_tools', ['is_enabled'])
    
    # ============================================================================
    # 4. TOOL_EXECUTIONS TABLE - Tool execution history
    # ============================================================================
    op.create_table(
        'tool_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('execution_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('tool_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True),
        sa.Column('inputs', postgresql.JSONB, nullable=False, default={}),
        sa.Column('outputs', postgresql.JSONB, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_details', postgresql.JSONB, nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
        sa.Column('memory_usage_mb', sa.Float, nullable=True),
        sa.Column('cpu_usage_percent', sa.Float, nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('execution_metadata', postgresql.JSONB, nullable=False, default={}),
        schema=None
    )
    
    # Create foreign keys
    op.create_foreign_key('fk_tool_executions_tool', 'tool_executions', 'tools', ['tool_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_tool_executions_agent', 'tool_executions', 'agents', ['agent_id'], ['id'], ondelete='SET NULL')
    
    # Create indexes for tool_executions table
    op.create_index('idx_tool_executions_execution_id', 'tool_executions', ['execution_id'])
    op.create_index('idx_tool_executions_tool', 'tool_executions', ['tool_id'])
    op.create_index('idx_tool_executions_agent', 'tool_executions', ['agent_id'])
    op.create_index('idx_tool_executions_status', 'tool_executions', ['status'])
    op.create_index('idx_tool_executions_created_at', 'tool_executions', ['created_at'])
    
    # ============================================================================
    # 5. TOOL_TEMPLATES TABLE - Reusable tool templates
    # ============================================================================
    op.create_table(
        'tool_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('template_code', sa.Text, nullable=False),
        sa.Column('parameters', postgresql.JSONB, nullable=False, default={}),
        sa.Column('placeholders', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('author', sa.String(255), nullable=True),
        sa.Column('complexity', sa.String(50), nullable=False, default='simple'),
        sa.Column('difficulty_level', sa.String(50), nullable=False, default='beginner'),
        sa.Column('estimated_time_minutes', sa.Integer, nullable=False, default=30),
        sa.Column('prerequisites', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('learning_objectives', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('is_public', sa.Boolean, nullable=False, default=False),
        sa.Column('is_featured', sa.Boolean, nullable=False, default=False),
        sa.Column('usage_count', sa.Integer, nullable=False, default=0),
        sa.Column('rating', sa.Float, nullable=False, default=0.0),
        sa.Column('rating_count', sa.Integer, nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('template_metadata', postgresql.JSONB, nullable=False, default={}),
        sa.Column('tags', postgresql.JSONB, nullable=False, default=[]),
        schema=None
    )
    
    # Create indexes for tool_templates table
    op.create_index('idx_tool_templates_name', 'tool_templates', ['name'])
    op.create_index('idx_tool_templates_category', 'tool_templates', ['category'])
    op.create_index('idx_tool_templates_complexity', 'tool_templates', ['complexity'])
    op.create_index('idx_tool_templates_public', 'tool_templates', ['is_public'])
    op.create_index('idx_tool_templates_featured', 'tool_templates', ['is_featured'])
    op.create_index('idx_tool_templates_rating', 'tool_templates', ['rating'])
    
    print("✅ Tool system tables created successfully!")


def downgrade():
    """Drop tool system management tables."""
    
    print("🗑️ Dropping tool system management tables...")
    
    # Drop tables in reverse order
    op.drop_table('tool_templates')
    op.drop_table('tool_executions')
    op.drop_table('agent_tools')
    op.drop_table('tool_categories')
    op.drop_table('tools')
    
    print("✅ Tool system tables dropped successfully!")


# Standalone execution for testing
if __name__ == "__main__":
    print("🚀 Running Tool System Migration (007)")
    upgrade()
    print("🎉 Migration completed successfully!")
