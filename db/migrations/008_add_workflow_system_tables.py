"""
008_add_workflow_system_tables.py
Add comprehensive workflow system management tables

Migration Order: 008 (Eighth - Workflow System Management)
Dependencies: 007_add_tool_system_tables.py
Next: 009_add_meme_system_tables.py

Original Revision ID: 008_add_workflow_system_tables
Create Date: 2025-09-23 16:15:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '008_add_workflow_system_tables'
down_revision = '007_add_tool_system_tables'
branch_labels = None
depends_on = None


def upgrade():
    """Create comprehensive workflow system management tables."""

    print("Creating workflow system management tables...")
    
    # ============================================================================
    # 1. WORKFLOWS TABLE - Core workflow definitions
    # ============================================================================
    op.create_table(
        'workflows',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('workflow_type', sa.String(100), nullable=False, default='sequential', index=True),
        sa.Column('nodes', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('edges', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('configuration', postgresql.JSONB, nullable=False, default={}),
        sa.Column('status', sa.String(50), nullable=False, default='draft', index=True),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('is_template', sa.Boolean, nullable=False, default=False),
        sa.Column('is_public', sa.Boolean, nullable=False, default=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('total_executions', sa.Integer, nullable=False, default=0),
        sa.Column('successful_executions', sa.Integer, nullable=False, default=0),
        sa.Column('failed_executions', sa.Integer, nullable=False, default=0),
        sa.Column('average_execution_time_ms', sa.Float, nullable=False, default=0.0),
        sa.Column('last_executed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('workflow_metadata', postgresql.JSONB, nullable=False, default={}),
        sa.Column('tags', postgresql.JSONB, nullable=False, default=[]),
        schema=None
    )
    
    # Create foreign key for created_by
    op.create_foreign_key('fk_workflows_created_by', 'workflows', 'users', ['created_by'], ['id'], ondelete='SET NULL')
    
    # Create indexes for workflows table
    op.create_index('idx_workflows_name', 'workflows', ['name'])
    op.create_index('idx_workflows_type', 'workflows', ['workflow_type'])
    op.create_index('idx_workflows_status', 'workflows', ['status'])
    op.create_index('idx_workflows_template', 'workflows', ['is_template'])
    op.create_index('idx_workflows_public', 'workflows', ['is_public'])
    op.create_index('idx_workflows_created_by', 'workflows', ['created_by'])
    
    # ============================================================================
    # 2. WORKFLOW_TEMPLATES TABLE - Reusable workflow templates
    # ============================================================================
    op.create_table(
        'workflow_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('template_data', postgresql.JSONB, nullable=False, default={}),
        sa.Column('parameters', postgresql.JSONB, nullable=False, default={}),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('author', sa.String(255), nullable=True),
        sa.Column('complexity', sa.String(50), nullable=False, default='simple'),
        sa.Column('estimated_time_minutes', sa.Integer, nullable=False, default=60),
        sa.Column('prerequisites', postgresql.JSONB, nullable=False, default=[]),
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
    
    # Create indexes for workflow_templates table
    op.create_index('idx_workflow_templates_name', 'workflow_templates', ['name'])
    op.create_index('idx_workflow_templates_category', 'workflow_templates', ['category'])
    op.create_index('idx_workflow_templates_complexity', 'workflow_templates', ['complexity'])
    op.create_index('idx_workflow_templates_public', 'workflow_templates', ['is_public'])
    op.create_index('idx_workflow_templates_featured', 'workflow_templates', ['is_featured'])
    
    # ============================================================================
    # 3. WORKFLOW_EXECUTIONS TABLE - Workflow execution history
    # ============================================================================
    op.create_table(
        'workflow_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('execution_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True),
        sa.Column('inputs', postgresql.JSONB, nullable=False, default={}),
        sa.Column('outputs', postgresql.JSONB, nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_details', postgresql.JSONB, nullable=True),
        sa.Column('agent_assignments', postgresql.JSONB, nullable=False, default={}),
        sa.Column('agent_results', postgresql.JSONB, nullable=False, default={}),
        sa.Column('total_steps', sa.Integer, nullable=False, default=0),
        sa.Column('completed_steps', sa.Integer, nullable=False, default=0),
        sa.Column('current_step', sa.String(255), nullable=True),
        sa.Column('total_tokens_used', sa.Integer, nullable=False, default=0),
        sa.Column('total_api_calls', sa.Integer, nullable=False, default=0),
        sa.Column('context', postgresql.JSONB, nullable=False, default={}),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('execution_metadata', postgresql.JSONB, nullable=False, default={}),
        schema=None
    )
    
    # Create foreign key
    op.create_foreign_key('fk_workflow_executions_workflow', 'workflow_executions', 'workflows', ['workflow_id'], ['id'], ondelete='CASCADE')
    
    # Create indexes for workflow_executions table
    op.create_index('idx_workflow_executions_execution_id', 'workflow_executions', ['execution_id'])
    op.create_index('idx_workflow_executions_workflow', 'workflow_executions', ['workflow_id'])
    op.create_index('idx_workflow_executions_status', 'workflow_executions', ['status'])
    op.create_index('idx_workflow_executions_created_at', 'workflow_executions', ['created_at'])
    
    # ============================================================================
    # 4. WORKFLOW_STEP_EXECUTIONS TABLE - Individual step execution tracking
    # ============================================================================
    op.create_table(
        'workflow_step_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('step_id', sa.String(255), nullable=False, index=True),
        sa.Column('workflow_execution_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('step_name', sa.String(255), nullable=False),
        sa.Column('step_type', sa.String(100), nullable=False, index=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending', index=True),
        sa.Column('inputs', postgresql.JSONB, nullable=False, default={}),
        sa.Column('outputs', postgresql.JSONB, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_details', postgresql.JSONB, nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
        sa.Column('retry_count', sa.Integer, nullable=False, default=0),
        sa.Column('max_retries', sa.Integer, nullable=False, default=3),
        sa.Column('depends_on_steps', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('blocks_steps', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('autonomous_decisions', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('reasoning_trace', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('step_metadata', postgresql.JSONB, nullable=False, default={}),
        schema=None
    )
    
    # Create foreign key
    op.create_foreign_key('fk_workflow_step_executions_workflow_execution', 'workflow_step_executions', 'workflow_executions', ['workflow_execution_id'], ['id'], ondelete='CASCADE')
    
    # Create indexes for workflow_step_executions table
    op.create_index('idx_workflow_step_executions_step_id', 'workflow_step_executions', ['step_id'])
    op.create_index('idx_workflow_step_executions_workflow_execution', 'workflow_step_executions', ['workflow_execution_id'])
    op.create_index('idx_workflow_step_executions_step_type', 'workflow_step_executions', ['step_type'])
    op.create_index('idx_workflow_step_executions_status', 'workflow_step_executions', ['status'])
    
    # ============================================================================
    # 5. NODE_DEFINITIONS TABLE - Reusable node definitions
    # ============================================================================
    op.create_table(
        'node_definitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('node_type', sa.String(100), nullable=False, unique=True, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('category', sa.String(100), nullable=False, index=True),
        sa.Column('input_schema', postgresql.JSONB, nullable=False, default={}),
        sa.Column('output_schema', postgresql.JSONB, nullable=False, default={}),
        sa.Column('configuration_schema', postgresql.JSONB, nullable=False, default={}),
        sa.Column('implementation_class', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False, default='1.0.0'),
        sa.Column('is_system_node', sa.Boolean, nullable=False, default=False),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('node_metadata', postgresql.JSONB, nullable=False, default={}),
        schema=None
    )

    # Create indexes for node_definitions table
    op.create_index('idx_node_definitions_node_type', 'node_definitions', ['node_type'])
    op.create_index('idx_node_definitions_category', 'node_definitions', ['category'])
    op.create_index('idx_node_definitions_system', 'node_definitions', ['is_system_node'])
    op.create_index('idx_node_definitions_active', 'node_definitions', ['is_active'])

    # ============================================================================
    # 6. WORKFLOW_NODES TABLE - Nodes within workflows
    # ============================================================================
    op.create_table(
        'workflow_nodes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('node_id', sa.String(255), nullable=False, index=True),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('node_type', sa.String(100), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('position_x', sa.Float, nullable=False, default=0.0),
        sa.Column('position_y', sa.Float, nullable=False, default=0.0),
        sa.Column('configuration', postgresql.JSONB, nullable=False, default={}),
        sa.Column('is_start_node', sa.Boolean, nullable=False, default=False),
        sa.Column('is_end_node', sa.Boolean, nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('node_metadata', postgresql.JSONB, nullable=False, default={}),
        schema=None
    )

    # Create foreign keys
    op.create_foreign_key('fk_workflow_nodes_workflow', 'workflow_nodes', 'workflows', ['workflow_id'], ['id'], ondelete='CASCADE')

    # Create unique constraint for node_id within workflow
    op.create_unique_constraint('uq_workflow_nodes_workflow_node', 'workflow_nodes', ['workflow_id', 'node_id'])

    # Create indexes for workflow_nodes table
    op.create_index('idx_workflow_nodes_node_id', 'workflow_nodes', ['node_id'])
    op.create_index('idx_workflow_nodes_workflow', 'workflow_nodes', ['workflow_id'])
    op.create_index('idx_workflow_nodes_type', 'workflow_nodes', ['node_type'])
    op.create_index('idx_workflow_nodes_start', 'workflow_nodes', ['is_start_node'])
    op.create_index('idx_workflow_nodes_end', 'workflow_nodes', ['is_end_node'])

    # ============================================================================
    # 7. WORKFLOW_CONNECTIONS TABLE - Connections between nodes
    # ============================================================================
    op.create_table(
        'workflow_connections',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('source_node_id', sa.String(255), nullable=False, index=True),
        sa.Column('target_node_id', sa.String(255), nullable=False, index=True),
        sa.Column('source_port', sa.String(100), nullable=True),
        sa.Column('target_port', sa.String(100), nullable=True),
        sa.Column('connection_type', sa.String(50), nullable=False, default='data'),
        sa.Column('condition', postgresql.JSONB, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('connection_metadata', postgresql.JSONB, nullable=False, default={}),
        schema=None
    )

    # Create foreign key
    op.create_foreign_key('fk_workflow_connections_workflow', 'workflow_connections', 'workflows', ['workflow_id'], ['id'], ondelete='CASCADE')

    # Create indexes for workflow_connections table
    op.create_index('idx_workflow_connections_workflow', 'workflow_connections', ['workflow_id'])
    op.create_index('idx_workflow_connections_source', 'workflow_connections', ['source_node_id'])
    op.create_index('idx_workflow_connections_target', 'workflow_connections', ['target_node_id'])
    op.create_index('idx_workflow_connections_type', 'workflow_connections', ['connection_type'])
    op.create_index('idx_workflow_connections_active', 'workflow_connections', ['is_active'])

    # ============================================================================
    # 8. NODE_EXECUTION_STATES TABLE - Node execution state tracking
    # ============================================================================
    op.create_table(
        'node_execution_states',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('workflow_execution_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('node_id', sa.String(255), nullable=False, index=True),
        sa.Column('execution_mode', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='created', index=True),
        sa.Column('agent_config', postgresql.JSONB, nullable=False, default={}),
        sa.Column('execution_context', postgresql.JSONB, nullable=False, default={}),
        sa.Column('input_data', postgresql.JSONB, nullable=False, default={}),
        sa.Column('output_data', postgresql.JSONB, nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
        sa.Column('tokens_used', sa.Integer, nullable=False, default=0),
        sa.Column('api_calls', sa.Integer, nullable=False, default=0),
        sa.Column('memory_usage_mb', sa.Float, nullable=True),
        sa.Column('autonomous_decisions', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('reasoning_steps', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('goal_achievements', postgresql.JSONB, nullable=False, default=[]),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('execution_metadata', postgresql.JSONB, nullable=False, default={}),
        schema=None
    )

    # Create foreign key
    op.create_foreign_key('fk_node_execution_states_workflow_execution', 'node_execution_states', 'workflow_executions', ['workflow_execution_id'], ['id'], ondelete='CASCADE')

    # Create unique constraint for workflow_execution + node combination
    op.create_unique_constraint('uq_node_execution_states_workflow_node', 'node_execution_states', ['workflow_execution_id', 'node_id'])

    # Create indexes for node_execution_states table
    op.create_index('idx_node_execution_states_workflow_execution', 'node_execution_states', ['workflow_execution_id'])
    op.create_index('idx_node_execution_states_node', 'node_execution_states', ['node_id'])
    op.create_index('idx_node_execution_states_status', 'node_execution_states', ['status'])
    op.create_index('idx_node_execution_states_execution_mode', 'node_execution_states', ['execution_mode'])

    print("[SUCCESS] Workflow system tables created successfully!")


def downgrade():
    """Drop workflow system management tables."""

    print("Dropping workflow system management tables...")

    # Drop tables in reverse order
    op.drop_table('node_execution_states')
    op.drop_table('workflow_connections')
    op.drop_table('workflow_nodes')
    op.drop_table('node_definitions')
    op.drop_table('workflow_step_executions')
    op.drop_table('workflow_executions')
    op.drop_table('workflow_templates')
    op.drop_table('workflows')

    print("[SUCCESS] Workflow system tables dropped successfully!")


# Standalone execution for testing
if __name__ == "__main__":
    print("Running Workflow System Migration (008)")
    upgrade()
    print("Migration completed successfully!")
