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
