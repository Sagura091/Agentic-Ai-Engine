"""
SQLAlchemy models for agent management.

This module defines the database models for storing and managing AI agents,
their configurations, and execution history.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database.base import Base


class Agent(Base):
    """Agent model for storing AI agent configurations."""

    __tablename__ = "agents"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    agent_type = Column(String(100), nullable=False, default='general', index=True)
    
    # LLM configuration
    model = Column(String(255), nullable=False, default='llama3.2:latest')
    model_provider = Column(String(50), nullable=False, default='ollama')
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2048)
    
    # Agent capabilities and tools
    capabilities = Column(JSON, default=list)  # List of capability strings
    tools = Column(JSON, default=list)  # List of tool names
    system_prompt = Column(Text)
    
    # Status and metadata
    status = Column(String(50), default='active', index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_activity = Column(DateTime(timezone=True))
    
    # Configuration and metadata
    agent_metadata = Column(JSON, default=dict)
    
    # Autonomous agent specific fields
    autonomy_level = Column(String(50), default='basic')
    learning_mode = Column(String(50), default='passive')
    decision_threshold = Column(Float, default=0.6)
    
    # Performance tracking
    total_tasks_completed = Column(Integer, default=0)
    total_tasks_failed = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="agent", cascade="all, delete-orphan")
    task_executions = relationship("TaskExecution", back_populates="agent", cascade="all, delete-orphan")
    autonomous_states = relationship("AutonomousAgentState", back_populates="agent", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Agent(id={self.id}, name='{self.name}', type='{self.agent_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "model": self.model,
            "model_provider": self.model_provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "metadata": self.agent_metadata,
            "autonomy_level": self.autonomy_level,
            "learning_mode": self.learning_mode,
            "decision_threshold": self.decision_threshold,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "average_response_time": self.average_response_time
        }


class Conversation(Base):
    """Conversation model for storing chat history with agents."""

    __tablename__ = "conversations"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to agent
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    
    # Conversation metadata
    title = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Session information
    session_id = Column(String(255), index=True)
    user_id = Column(String(255), index=True)
    
    # Conversation state
    is_active = Column(Boolean, default=True)
    message_count = Column(Integer, default=0)
    
    # Metadata
    conversation_metadata = Column(JSON, default=dict)
    
    # Relationships
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, agent_id={self.agent_id}, title='{self.title}')>"


class Message(Base):
    """Message model for storing individual messages in conversations."""

    __tablename__ = "messages"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to conversation
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    
    # Message content
    content = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Message metadata
    token_count = Column(Integer)
    response_time_ms = Column(Float)
    message_metadata = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"


class TaskExecution(Base):
    """Task execution model for tracking agent task performance."""

    __tablename__ = "task_executions"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to agent
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    
    # Task information
    task_description = Column(Text, nullable=False)
    task_type = Column(String(100), index=True)
    
    # Execution details
    status = Column(String(50), default='pending', index=True)  # pending, running, completed, failed
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Results and performance
    result = Column(JSON)
    error_message = Column(Text)
    execution_time_ms = Column(Float)
    token_usage = Column(Integer)
    
    # Context and metadata
    context = Column(JSON, default=dict)
    task_metadata = Column(JSON, default=dict)
    
    # Relationships
    agent = relationship("Agent", back_populates="task_executions")
    
    def __repr__(self):
        return f"<TaskExecution(id={self.id}, agent_id={self.agent_id}, status='{self.status}')>"
