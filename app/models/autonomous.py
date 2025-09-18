"""
SQLAlchemy models for autonomous agent persistence.

This module defines database models for storing autonomous agent state,
goals, decisions, memories, and learning experiences to enable true
persistent agentic behavior across sessions.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database.base import Base


class AutonomousAgentState(Base):
    """Persistent state for autonomous agents."""

    __tablename__ = "autonomous_agent_states"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False, index=True)
    
    # State information
    session_id = Column(String(255), nullable=True, index=True)
    autonomy_level = Column(String(50), nullable=False, default='adaptive')
    decision_confidence = Column(Float, default=0.0)
    learning_enabled = Column(Boolean, default=True)
    
    # Complex state data
    current_task = Column(Text)
    tools_available = Column(JSON, default=list)
    outputs = Column(JSON, default=dict)
    errors = Column(JSON, default=list)
    iteration_count = Column(Integer, default=0)
    max_iterations = Column(Integer, default=50)
    custom_state = Column(JSON, default=dict)
    
    # Autonomous capabilities state
    goal_stack = Column(JSON, default=list)
    context_memory = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)
    self_initiated_tasks = Column(JSON, default=list)
    proactive_actions = Column(JSON, default=list)
    emergent_behaviors = Column(JSON, default=list)
    collaboration_state = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_accessed = Column(DateTime(timezone=True))
    
    # Relationships
    agent = relationship("Agent", back_populates="autonomous_states")
    goals = relationship("AutonomousGoalDB", back_populates="agent_state", cascade="all, delete-orphan")
    decisions = relationship("AutonomousDecisionDB", back_populates="agent_state", cascade="all, delete-orphan")
    memories = relationship("AgentMemoryDB", back_populates="agent_state", cascade="all, delete-orphan")


class AutonomousGoalDB(Base):
    """Persistent storage for autonomous agent goals."""

    __tablename__ = "autonomous_goals"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    goal_id = Column(String(255), nullable=False, unique=True, index=True)
    agent_state_id = Column(UUID(as_uuid=True), ForeignKey('autonomous_agent_states.id'), nullable=False)
    
    # Goal definition
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    goal_type = Column(String(50), nullable=False)  # learning, optimization, exploration, etc.
    priority = Column(String(20), nullable=False, default='medium')  # low, medium, high
    status = Column(String(20), nullable=False, default='pending')  # pending, active, completed, failed, paused
    
    # Goal details
    target_outcome = Column(JSON, default=dict)
    success_criteria = Column(JSON, default=list)
    context = Column(JSON, default=dict)
    goal_metadata = Column(JSON, default=dict)
    
    # Progress tracking
    progress = Column(Float, default=0.0)
    completion_confidence = Column(Float, default=0.0)
    estimated_effort = Column(Float, default=1.0)
    actual_effort = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deadline = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    agent_state = relationship("AutonomousAgentState", back_populates="goals")


class AutonomousDecisionDB(Base):
    """Persistent storage for autonomous agent decisions."""

    __tablename__ = "autonomous_decisions"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    decision_id = Column(String(255), nullable=False, unique=True, index=True)
    agent_state_id = Column(UUID(as_uuid=True), ForeignKey('autonomous_agent_states.id'), nullable=False)
    
    # Decision information
    decision_type = Column(String(100), nullable=False)
    context = Column(JSON, default=dict)
    options_considered = Column(JSON, default=list)
    chosen_option = Column(JSON, default=dict)
    confidence = Column(Float, nullable=False)
    reasoning = Column(JSON, default=list)
    
    # Outcomes
    expected_outcome = Column(JSON, default=dict)
    actual_outcome = Column(JSON, default=dict)
    learning_value = Column(Float, default=0.0)
    success = Column(Boolean, default=None)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    outcome_recorded_at = Column(DateTime(timezone=True))
    
    # Relationships
    agent_state = relationship("AutonomousAgentState", back_populates="decisions")


class AgentMemoryDB(Base):
    """Persistent storage for agent episodic and semantic memory."""

    __tablename__ = "agent_memories"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    memory_id = Column(String(255), nullable=False, unique=True, index=True)
    agent_state_id = Column(UUID(as_uuid=True), ForeignKey('autonomous_agent_states.id'), nullable=False)
    
    # Memory content
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False, default='episodic')  # episodic, semantic, procedural
    context = Column(JSON, default=dict)
    
    # Memory properties
    importance = Column(Float, default=0.5)
    emotional_valence = Column(Float, default=0.0)
    tags = Column(JSON, default=list)
    
    # Access patterns
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    
    # Temporal information
    session_id = Column(String(255), index=True)
    expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    agent_state = relationship("AutonomousAgentState", back_populates="memories")


class LearningExperienceDB(Base):
    """Persistent storage for agent learning experiences."""

    __tablename__ = "learning_experiences"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experience_id = Column(String(255), nullable=False, unique=True, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False, index=True)
    
    # Experience data
    experience_type = Column(String(50), nullable=False)  # decision, outcome, pattern, insight
    content = Column(JSON, nullable=False)
    context = Column(JSON, default=dict)
    
    # Learning metrics
    learning_value = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    impact_score = Column(Float, default=0.0)
    
    # Pattern information
    pattern_type = Column(String(50))
    pattern_data = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent")


class PerformanceMetricDB(Base):
    """Persistent storage for agent performance metrics."""

    __tablename__ = "performance_metrics"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=False, index=True)
    
    # Metric information
    metric_type = Column(String(50), nullable=False)  # decision_quality, learning_rate, goal_achievement
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_data = Column(JSON, default=dict)
    
    # Context
    session_id = Column(String(255), index=True)
    task_context = Column(JSON, default=dict)
    
    # Aggregation period
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent")
