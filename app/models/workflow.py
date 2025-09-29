"""
SQLAlchemy models for workflow management.

This module defines the database models for storing and managing workflows,
their execution history, and related components.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database.base import Base


class Workflow(Base):
    """Workflow model for storing workflow definitions."""
    
    __tablename__ = "workflows"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    workflow_type = Column(String(100), nullable=False, default='sequential', index=True)
    
    # Workflow definition
    nodes = Column(JSON, default=list)  # List of workflow nodes
    edges = Column(JSON, default=list)  # List of workflow edges
    configuration = Column(JSON, default=dict)  # Workflow configuration
    
    # Status and metadata
    status = Column(String(50), default='draft', index=True)  # draft, active, archived
    version = Column(String(50), default='1.0.0')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_executed = Column(DateTime(timezone=True))
    
    # Execution statistics
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    average_execution_time = Column(Float, default=0.0)
    
    # Metadata
    workflow_metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)  # List of tags for categorization
    
    # Relationships
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Workflow(id={self.id}, name='{self.name}', type='{self.workflow_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "workflow_type": self.workflow_type,
            "nodes": self.nodes,
            "edges": self.edges,
            "configuration": self.configuration,
            "status": self.status,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "average_execution_time": self.average_execution_time,
            "metadata": self.workflow_metadata,
            "tags": self.tags
        }


class WorkflowExecution(Base):
    """Workflow execution model for tracking workflow runs."""
    
    __tablename__ = "workflow_executions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to workflow
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False, index=True)
    
    # Execution information
    execution_id = Column(String(255), nullable=False, index=True)  # External execution ID
    status = Column(String(50), default='pending', index=True)  # pending, running, completed, failed, cancelled
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Input and output
    inputs = Column(JSON, default=dict)
    outputs = Column(JSON, default=dict)
    
    # Execution details
    execution_time_ms = Column(Float)
    error_message = Column(Text)
    error_details = Column(JSON)
    
    # Agent assignments and results
    agent_assignments = Column(JSON, default=dict)  # Which agents were assigned to which tasks
    agent_results = Column(JSON, default=dict)  # Results from each agent
    
    # Progress tracking
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    current_step = Column(String(255))
    
    # Resource usage
    total_tokens_used = Column(Integer, default=0)
    total_api_calls = Column(Integer, default=0)
    
    # Context and metadata
    context = Column(JSON, default=dict)
    execution_metadata = Column(JSON, default=dict)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    step_executions = relationship("WorkflowStepExecution", back_populates="workflow_execution", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WorkflowExecution(id={self.id}, workflow_id={self.workflow_id}, status='{self.status}')>"


class WorkflowStepExecution(Base):
    """Individual step execution within a workflow."""
    
    __tablename__ = "workflow_step_executions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to workflow execution
    workflow_execution_id = Column(UUID(as_uuid=True), ForeignKey("workflow_executions.id"), nullable=False, index=True)
    
    # Step information
    step_name = Column(String(255), nullable=False)
    step_type = Column(String(100), nullable=False)  # agent, decision, subgraph, etc.
    step_order = Column(Integer, nullable=False)
    
    # Agent information (if applicable)
    agent_id = Column(String(255))  # Reference to agent (may not be in DB)
    agent_type = Column(String(100))
    
    # Execution details
    status = Column(String(50), default='pending', index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Input and output
    inputs = Column(JSON, default=dict)
    outputs = Column(JSON, default=dict)
    
    # Performance metrics
    execution_time_ms = Column(Float)
    tokens_used = Column(Integer, default=0)
    api_calls_made = Column(Integer, default=0)
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Metadata
    step_metadata = Column(JSON, default=dict)
    
    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="step_executions")
    
    def __repr__(self):
        return f"<WorkflowStepExecution(id={self.id}, step_name='{self.step_name}', status='{self.status}')>"


class WorkflowTemplate(Base):
    """Workflow template model for storing reusable workflow patterns."""
    
    __tablename__ = "workflow_templates"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Template information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    category = Column(String(100), nullable=False, index=True)
    
    # Template definition
    template_data = Column(JSON, nullable=False)  # Complete template structure
    required_agents = Column(JSON, default=list)  # List of required agent types
    optional_agents = Column(JSON, default=list)  # List of optional agent types
    
    # Configuration
    default_configuration = Column(JSON, default=dict)
    parameters = Column(JSON, default=dict)  # Configurable parameters
    
    # Metadata
    version = Column(String(50), default='1.0.0')
    author = Column(String(255))
    tags = Column(JSON, default=list)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Status
    is_public = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Metadata
    template_metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<WorkflowTemplate(id={self.id}, name='{self.name}', category='{self.category}')>"


# ============================================================================
# REVOLUTIONARY COMPONENT WORKFLOW EXECUTION MODELS
# ============================================================================

class ComponentWorkflowExecution(Base):
    """Component-based workflow execution model."""

    __tablename__ = "component_workflow_executions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Workflow information
    workflow_id = Column(String(255), nullable=False, index=True)
    workflow_name = Column(String(255))
    workflow_description = Column(Text)

    # Execution details
    execution_mode = Column(String(50), nullable=False, default='sequential')  # sequential, parallel, autonomous
    status = Column(String(50), default='queued', index=True)  # queued, running, completed, failed, cancelled

    # Component information
    components = Column(JSON, default=list)  # List of component configurations
    component_count = Column(Integer, default=0)

    # Execution tracking
    current_step = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    failed_steps = Column(Integer, default=0)

    # Results and outputs
    execution_results = Column(JSON, default=dict)  # Results from each component
    final_output = Column(JSON, default=dict)  # Final workflow output
    context_data = Column(JSON, default=dict)  # Execution context

    # Performance metrics
    execution_time_ms = Column(Float)
    total_tokens_used = Column(Integer, default=0)
    total_api_calls = Column(Integer, default=0)

    # Error handling
    error_message = Column(Text)
    error_details = Column(JSON)
    error_step = Column(Integer)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Worker information
    worker_id = Column(String(100))
    node_id = Column(String(100))  # For distributed execution

    # Metadata
    execution_metadata = Column(JSON, default=dict)

    def __repr__(self):
        return f"<ComponentWorkflowExecution(id={self.id}, workflow_id='{self.workflow_id}', status='{self.status}')>"


class WorkflowStepState(Base):
    """Individual workflow step state tracking model."""

    __tablename__ = "workflow_step_states"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Step identification
    step_id = Column(String(255), nullable=False, index=True)
    workflow_execution_id = Column(UUID(as_uuid=True), ForeignKey('component_workflow_executions.id'), nullable=False)
    step_number = Column(Integer, nullable=False)

    # Component information
    component_id = Column(String(255))
    component_type = Column(String(100))  # TOOL, CAPABILITY, PROMPT, WORKFLOW_STEP
    component_name = Column(String(255))
    component_config = Column(JSON, default=dict)

    # Execution state
    status = Column(String(50), default='pending', index=True)  # pending, running, completed, failed, skipped
    execution_mode = Column(String(50))  # autonomous, instruction_based, default

    # Agent information
    component_agent_id = Column(String(255))
    assigned_worker = Column(String(100))

    # Execution details
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    context_updates = Column(JSON, default=dict)

    # Performance metrics
    execution_time_ms = Column(Float)
    tokens_used = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)

    # Error handling
    error_message = Column(Text)
    error_details = Column(JSON)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Dependencies and relationships
    depends_on_steps = Column(JSON, default=list)  # List of step IDs this step depends on
    blocks_steps = Column(JSON, default=list)  # List of step IDs that depend on this step

    # Autonomous execution data
    autonomous_decisions = Column(JSON, default=list)  # Decisions made by autonomous agents
    reasoning_trace = Column(JSON, default=list)  # Reasoning steps for autonomous execution

    # Metadata
    step_metadata = Column(JSON, default=dict)

    # Relationships
    workflow_execution = relationship("ComponentWorkflowExecution", back_populates="step_states")

    def __repr__(self):
        return f"<WorkflowStepState(id={self.id}, step_id='{self.step_id}', status='{self.status}')>"


# Add relationship to ComponentWorkflowExecution
ComponentWorkflowExecution.step_states = relationship("WorkflowStepState", back_populates="workflow_execution", cascade="all, delete-orphan")


class ComponentAgentExecution(Base):
    """Component agent execution tracking model."""

    __tablename__ = "component_agent_executions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Agent identification
    component_agent_id = Column(String(255), nullable=False, index=True)
    agent_name = Column(String(255))
    agent_type = Column(String(100))

    # Template information
    template_name = Column(String(255))
    template_version = Column(String(50))

    # Execution context
    workflow_execution_id = Column(UUID(as_uuid=True), ForeignKey('component_workflow_executions.id'))
    step_state_id = Column(UUID(as_uuid=True), ForeignKey('workflow_step_states.id'))

    # Execution details
    execution_mode = Column(String(50), nullable=False)  # autonomous, instruction_based, default
    status = Column(String(50), default='created', index=True)  # created, queued, running, completed, failed

    # Configuration and context
    agent_config = Column(JSON, default=dict)
    execution_context = Column(JSON, default=dict)
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)

    # Performance metrics
    execution_time_ms = Column(Float)
    tokens_used = Column(Integer, default=0)
    api_calls = Column(Integer, default=0)
    memory_usage_mb = Column(Float)

    # Autonomous execution data
    autonomous_decisions = Column(JSON, default=list)
    reasoning_steps = Column(JSON, default=list)
    goal_achievements = Column(JSON, default=list)

    # Error handling
    error_message = Column(Text)
    error_details = Column(JSON)
    error_stack_trace = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    queued_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Worker and node information
    worker_id = Column(String(100))
    node_id = Column(String(100))

    # Metadata
    execution_metadata = Column(JSON, default=dict)

    # Relationships
    workflow_execution = relationship("ComponentWorkflowExecution")
    step_state = relationship("WorkflowStepState")

    def __repr__(self):
        return f"<ComponentAgentExecution(id={self.id}, component_agent_id='{self.component_agent_id}', status='{self.status}')>"


# ============================================================================
# ADVANCED NODE SYSTEM MODELS
# ============================================================================

class NodeDefinition(Base):
    """Node type definitions for the advanced workflow system."""

    __tablename__ = "node_definitions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Node type information
    node_type = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100), nullable=False, index=True)

    # Port definitions
    input_ports = Column(JSON, default=list)  # Array of input port definitions
    output_ports = Column(JSON, default=list)  # Array of output port definitions

    # Configuration and validation
    configuration_schema = Column(JSON, default=dict)  # JSON schema for node configuration
    default_configuration = Column(JSON, default=dict)  # Default configuration values

    # Execution information
    execution_handler = Column(String(255), nullable=False)  # Backend handler function name
    execution_timeout = Column(Integer, default=300)  # Timeout in seconds

    # Connection rules
    connection_rules = Column(JSON, default=dict)  # Port compatibility and connection rules
    max_input_connections = Column(Integer, default=-1)  # -1 for unlimited
    max_output_connections = Column(Integer, default=-1)  # -1 for unlimited

    # Metadata
    icon = Column(String(100))  # Icon identifier
    color = Column(String(50))  # Color for UI
    tags = Column(JSON, default=list)  # Tags for categorization

    # Status
    is_active = Column(Boolean, default=True)
    is_experimental = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    node_instances = relationship("WorkflowNode", back_populates="node_definition")

    def __repr__(self):
        return f"<NodeDefinition(node_type='{self.node_type}', name='{self.name}')>"


class WorkflowNode(Base):
    """Individual node instances within workflows."""

    __tablename__ = "workflow_nodes"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # References
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id'), nullable=False, index=True)
    node_definition_id = Column(UUID(as_uuid=True), ForeignKey('node_definitions.id'), nullable=False, index=True)

    # Node instance information
    node_id = Column(String(255), nullable=False, index=True)  # Frontend node ID (unique within workflow)
    name = Column(String(255), nullable=False)  # User-defined name

    # Position and layout
    position = Column(JSON, nullable=False)  # {x, y} coordinates
    size = Column(JSON, default=dict)  # {width, height} if needed

    # Configuration
    configuration = Column(JSON, default=dict)  # Node-specific configuration

    # Status
    is_enabled = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    workflow = relationship("Workflow")
    node_definition = relationship("NodeDefinition", back_populates="node_instances")
    source_connections = relationship("WorkflowConnection", foreign_keys="WorkflowConnection.source_node_id", back_populates="source_node")
    target_connections = relationship("WorkflowConnection", foreign_keys="WorkflowConnection.target_node_id", back_populates="target_node")
    execution_states = relationship("NodeExecutionState", back_populates="workflow_node")

    def __repr__(self):
        return f"<WorkflowNode(node_id='{self.node_id}', workflow_id='{self.workflow_id}')>"


class WorkflowConnection(Base):
    """Connections between nodes in workflows."""

    __tablename__ = "workflow_connections"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # References
    workflow_id = Column(UUID(as_uuid=True), ForeignKey('workflows.id'), nullable=False, index=True)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey('workflow_nodes.id'), nullable=False, index=True)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey('workflow_nodes.id'), nullable=False, index=True)

    # Connection details
    connection_id = Column(String(255), nullable=False, index=True)  # Frontend connection ID
    source_port = Column(String(100), nullable=False)  # Source port identifier
    target_port = Column(String(100), nullable=False)  # Target port identifier

    # Connection properties
    connection_type = Column(String(50), default='data')  # data, control, etc.
    is_animated = Column(Boolean, default=False)
    style = Column(JSON, default=dict)  # Visual styling properties

    # Data transformation
    data_transformation = Column(JSON, default=dict)  # Optional data transformation rules

    # Status
    is_enabled = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    workflow = relationship("Workflow")
    source_node = relationship("WorkflowNode", foreign_keys=[source_node_id], back_populates="source_connections")
    target_node = relationship("WorkflowNode", foreign_keys=[target_node_id], back_populates="target_connections")

    def __repr__(self):
        return f"<WorkflowConnection(connection_id='{self.connection_id}', source='{self.source_port}', target='{self.target_port}')>"


class NodeExecutionState(Base):
    """Execution state tracking for individual nodes."""

    __tablename__ = "node_execution_state"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # References
    workflow_execution_id = Column(UUID(as_uuid=True), ForeignKey('workflow_executions.id'), nullable=False, index=True)
    workflow_node_id = Column(UUID(as_uuid=True), ForeignKey('workflow_nodes.id'), nullable=False, index=True)

    # Execution identification
    execution_id = Column(String(255), nullable=False, index=True)  # Unique execution identifier
    node_id = Column(String(255), nullable=False, index=True)  # Frontend node ID

    # Execution status
    status = Column(String(50), default='pending', index=True)  # pending, running, success, error, cancelled

    # Data
    input_data = Column(JSON, default=dict)  # Input data received by the node
    output_data = Column(JSON, default=dict)  # Output data produced by the node
    intermediate_data = Column(JSON, default=dict)  # Intermediate processing data

    # Performance metrics
    execution_time_ms = Column(Integer)  # Execution time in milliseconds
    memory_usage_mb = Column(Float)  # Memory usage in MB
    cpu_usage_percent = Column(Float)  # CPU usage percentage

    # Error handling
    error_message = Column(Text)
    error_details = Column(JSON, default=dict)
    error_stack_trace = Column(Text)
    retry_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    queued_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Metadata
    execution_metadata = Column(JSON, default=dict)

    # Relationships
    workflow_execution = relationship("WorkflowExecution")
    workflow_node = relationship("WorkflowNode", back_populates="execution_states")

    def __repr__(self):
        return f"<NodeExecutionState(node_id='{self.node_id}', status='{self.status}', execution_id='{self.execution_id}')>"
