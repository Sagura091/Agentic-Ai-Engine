"""
SQLAlchemy models for tool management.

This module defines the database models for storing and managing dynamic tools,
their configurations, and usage statistics.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database.base import Base


class Tool(Base):
    """Tool model for storing dynamic tool definitions."""
    
    __tablename__ = "tools"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic information
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)
    
    # Tool definition
    implementation = Column(Text, nullable=False)  # Python code implementation
    parameters_schema = Column(JSON, default=dict)  # JSON schema for parameters
    return_schema = Column(JSON, default=dict)  # JSON schema for return value
    
    # Tool metadata
    version = Column(String(50), default='1.0.0')
    author = Column(String(255))
    complexity = Column(String(50), default='simple')  # simple, moderate, complex
    safety_level = Column(String(50), default='safe')  # safe, caution, restricted

    # Upload and validation metadata
    source_type = Column(String(50), default='generated')  # generated, uploaded, template
    original_filename = Column(String(255))  # Original uploaded filename
    file_hash = Column(String(255))  # SHA256 hash of uploaded file
    validation_status = Column(String(50), default='pending')  # pending, validated, rejected
    validation_score = Column(Float, default=0.0)  # Security/quality score (0-1)
    validation_issues = Column(JSON, default=list)  # List of validation issues
    validation_warnings = Column(JSON, default=list)  # List of validation warnings
    
    # Dependencies and requirements
    dependencies = Column(JSON, default=list)  # List of required packages
    system_requirements = Column(JSON, default=dict)  # System requirements
    
    # Status and availability
    status = Column(String(50), default='active', index=True)  # active, deprecated, disabled
    is_global = Column(Boolean, default=False)  # Available to all agents
    is_verified = Column(Boolean, default=False)  # Verified by system admin
    is_public = Column(Boolean, default=False)  # Available in public marketplace
    requires_approval = Column(Boolean, default=True)  # Requires admin approval for public use
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    average_execution_time = Column(Float, default=0.0)
    
    # Performance metrics
    total_execution_time = Column(Float, default=0.0)
    last_used = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Metadata and configuration
    tool_metadata = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    # Relationships
    tool_executions = relationship("ToolExecution", back_populates="tool", cascade="all, delete-orphan")
    agent_tools = relationship("AgentTool", back_populates="tool", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Tool(id={self.id}, name='{self.name}', category='{self.category}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "implementation": self.implementation,
            "parameters_schema": self.parameters_schema,
            "return_schema": self.return_schema,
            "version": self.version,
            "author": self.author,
            "complexity": self.complexity,
            "safety_level": self.safety_level,
            "source_type": self.source_type,
            "original_filename": self.original_filename,
            "validation_status": self.validation_status,
            "validation_score": self.validation_score,
            "validation_issues": self.validation_issues,
            "validation_warnings": self.validation_warnings,
            "dependencies": self.dependencies,
            "system_requirements": self.system_requirements,
            "status": self.status,
            "is_global": self.is_global,
            "is_verified": self.is_verified,
            "is_public": self.is_public,
            "requires_approval": self.requires_approval,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "average_execution_time": self.average_execution_time,
            "total_execution_time": self.total_execution_time,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.tool_metadata,
            "configuration": self.configuration,
            "tags": self.tags
        }


class AgentTool(Base):
    """Association table for agent-tool relationships."""
    
    __tablename__ = "agent_tools"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    tool_id = Column(UUID(as_uuid=True), ForeignKey("tools.id"), nullable=False, index=True)
    
    # Assignment details
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    assigned_by = Column(String(255))  # Who assigned the tool
    
    # Tool-specific configuration for this agent
    agent_specific_config = Column(JSON, default=dict)
    
    # Usage permissions
    can_modify = Column(Boolean, default=False)
    can_share = Column(Boolean, default=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Relationships
    tool = relationship("Tool", back_populates="agent_tools")
    
    def __repr__(self):
        return f"<AgentTool(agent_id={self.agent_id}, tool_id={self.tool_id})>"


class ToolExecution(Base):
    """Tool execution model for tracking tool usage and performance."""
    
    __tablename__ = "tool_executions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to tool
    tool_id = Column(UUID(as_uuid=True), ForeignKey("tools.id"), nullable=False, index=True)
    
    # Execution context
    agent_id = Column(String(255), index=True)  # Which agent used the tool
    task_execution_id = Column(UUID(as_uuid=True), index=True)  # Related task execution
    
    # Execution details
    status = Column(String(50), default='pending', index=True)  # pending, running, completed, failed
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Input and output
    inputs = Column(JSON, default=dict)
    outputs = Column(JSON, default=dict)
    
    # Performance metrics
    execution_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    cpu_usage_percent = Column(Float)
    
    # Error handling
    error_message = Column(Text)
    error_type = Column(String(100))
    stack_trace = Column(Text)
    
    # Context and metadata
    context = Column(JSON, default=dict)
    execution_metadata = Column(JSON, default=dict)
    
    # Relationships
    tool = relationship("Tool", back_populates="tool_executions")
    
    def __repr__(self):
        return f"<ToolExecution(id={self.id}, tool_id={self.tool_id}, status='{self.status}')>"


class ToolCategory(Base):
    """Tool category model for organizing tools."""
    
    __tablename__ = "tool_categories"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Category information
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    parent_category_id = Column(UUID(as_uuid=True), ForeignKey("tool_categories.id"))
    
    # Display properties
    icon = Column(String(100))
    color = Column(String(50))
    sort_order = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Metadata
    category_metadata = Column(JSON, default=dict)
    
    # Self-referential relationship for hierarchical categories
    parent_category = relationship("ToolCategory", remote_side=[id], backref="subcategories")
    
    def __repr__(self):
        return f"<ToolCategory(id={self.id}, name='{self.name}')>"


class ToolTemplate(Base):
    """Tool template model for storing reusable tool patterns."""
    
    __tablename__ = "tool_templates"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Template information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    category = Column(String(100), nullable=False, index=True)
    
    # Template definition
    template_code = Column(Text, nullable=False)  # Template implementation
    parameters = Column(JSON, default=dict)  # Configurable parameters
    placeholders = Column(JSON, default=list)  # List of placeholders to fill
    
    # Template metadata
    version = Column(String(50), default='1.0.0')
    author = Column(String(255))
    complexity = Column(String(50), default='simple')
    
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
    tags = Column(JSON, default=list)
    
    def __repr__(self):
        return f"<ToolTemplate(id={self.id}, name='{self.name}', category='{self.category}')>"
