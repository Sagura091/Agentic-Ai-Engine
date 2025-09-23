"""
Database migration to add admin settings tables.

This migration creates the necessary tables for storing and managing
admin settings with full audit trail and configuration cache.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


def upgrade():
    """Create admin settings tables."""
    
    # Create admin_settings table
    op.create_table(
        'admin_settings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('key', sa.String(200), nullable=False, index=True),
        sa.Column('value', sa.JSON(), nullable=False),
        sa.Column('default_value', sa.JSON(), nullable=False),
        sa.Column('setting_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('security_level', sa.String(50), default='admin_only'),
        sa.Column('requires_restart', sa.Boolean(), default=False),
        sa.Column('validation_rules', sa.JSON(), default={}),
        sa.Column('enum_values', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, index=True),
        sa.Column('is_system_managed', sa.Boolean(), default=False),
    )
    
    # Create unique index on category + key
    op.create_index(
        'idx_admin_settings_category_key',
        'admin_settings',
        ['category', 'key'],
        unique=True
    )
    
    # Create additional indexes
    op.create_index('idx_admin_settings_category', 'admin_settings', ['category'])
    op.create_index('idx_admin_settings_updated', 'admin_settings', ['updated_at'])
    
    # Create admin_setting_history table
    op.create_table(
        'admin_setting_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('setting_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('key', sa.String(200), nullable=False),
        sa.Column('old_value', sa.JSON(), nullable=True),
        sa.Column('new_value', sa.JSON(), nullable=False),
        sa.Column('change_reason', sa.Text(), nullable=True),
        sa.Column('changed_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('changed_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('system_restart_required', sa.Boolean(), default=False),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create indexes for history table
    op.create_index('idx_admin_setting_history_setting_id', 'admin_setting_history', ['setting_id'])
    op.create_index('idx_admin_setting_history_changed_at', 'admin_setting_history', ['changed_at'])
    op.create_index('idx_admin_setting_history_changed_by', 'admin_setting_history', ['changed_by'])
    
    # Create system_configuration_cache table
    op.create_table(
        'system_configuration_cache',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('component_name', sa.String(100), nullable=False, unique=True, index=True),
        sa.Column('configuration', sa.JSON(), nullable=False),
        sa.Column('configuration_hash', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, index=True),
        sa.Column('requires_restart', sa.Boolean(), default=False),
    )
    
    # Create indexes for configuration cache
    op.create_index('idx_system_config_cache_component', 'system_configuration_cache', ['component_name'])
    op.create_index('idx_system_config_cache_updated', 'system_configuration_cache', ['updated_at'])


def downgrade():
    """Drop admin settings tables."""
    
    # Drop indexes first
    op.drop_index('idx_system_config_cache_updated', 'system_configuration_cache')
    op.drop_index('idx_system_config_cache_component', 'system_configuration_cache')
    op.drop_index('idx_admin_setting_history_changed_by', 'admin_setting_history')
    op.drop_index('idx_admin_setting_history_changed_at', 'admin_setting_history')
    op.drop_index('idx_admin_setting_history_setting_id', 'admin_setting_history')
    op.drop_index('idx_admin_settings_updated', 'admin_settings')
    op.drop_index('idx_admin_settings_category', 'admin_settings')
    op.drop_index('idx_admin_settings_category_key', 'admin_settings')
    
    # Drop tables
    op.drop_table('system_configuration_cache')
    op.drop_table('admin_setting_history')
    op.drop_table('admin_settings')
