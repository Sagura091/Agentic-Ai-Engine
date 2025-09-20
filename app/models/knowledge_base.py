"""
SQLAlchemy models for knowledge base management.

This module defines database models for storing knowledge base metadata,
replacing the JSON file-based storage with proper database persistence.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database.base import Base


class KnowledgeBase(Base):
    """Knowledge base metadata and configuration."""

    __tablename__ = "knowledge_bases"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key - using string ID for compatibility
    id = Column(String(255), primary_key=True)
    
    # Basic information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    use_case = Column(String(100), nullable=False, default='general', index=True)
    
    # Categorization and discovery
    tags = Column(JSON, default=list)  # List of tags for categorization
    category = Column(String(100), index=True)
    domain = Column(String(100))  # Domain-specific classification
    
    # Access control
    is_public = Column(Boolean, default=False, index=True)
    created_by = Column(String(255), nullable=False, index=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Content statistics
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    size_mb = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    
    # Model configuration
    embedding_model = Column(String(255))
    embedding_dimension = Column(Integer)
    chunking_strategy = Column(String(100), default='recursive')
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    
    # Performance metrics
    avg_query_time_ms = Column(Float, default=0.0)
    total_queries = Column(Integer, default=0)
    last_query_at = Column(DateTime(timezone=True))
    
    # Status and health
    status = Column(String(50), default='active', index=True)  # active, inactive, processing, error
    health_score = Column(Float, default=1.0)  # 0.0 to 1.0
    last_health_check = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_accessed = Column(DateTime(timezone=True))
    
    # Configuration and metadata
    configuration = Column(JSON, default=dict)
    kb_metadata = Column(JSON, default=dict)
    
    # Relationships
    access_permissions = relationship("KnowledgeBaseAccess", back_populates="knowledge_base", cascade="all, delete-orphan")
    usage_logs = relationship("KnowledgeBaseUsageLog", back_populates="knowledge_base", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', name='{self.name}', use_case='{self.use_case}')>"


class KnowledgeBaseAccess(Base):
    """Access permissions for knowledge bases."""

    __tablename__ = "knowledge_base_access"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    knowledge_base_id = Column(String(255), ForeignKey('rag.knowledge_bases.id'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=True, index=True)
    
    # Access control
    access_level = Column(String(50), nullable=False, default='read')  # read, write, admin
    permission_type = Column(String(50), nullable=False, default='user')  # user, agent, role, public
    
    # Constraints and limits
    query_limit_per_hour = Column(Integer)
    query_limit_per_day = Column(Integer)
    data_limit_mb = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    expires_at = Column(DateTime(timezone=True))
    
    # Audit trail
    granted_by = Column(String(255), nullable=False)
    granted_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True))
    
    # Metadata
    access_metadata = Column(JSON, default=dict)
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="access_permissions")
    
    def __repr__(self):
        return f"<KnowledgeBaseAccess(kb_id='{self.knowledge_base_id}', access_level='{self.access_level}')>"


class KnowledgeBaseUsageLog(Base):
    """Usage tracking for knowledge bases."""

    __tablename__ = "knowledge_base_usage_logs"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    knowledge_base_id = Column(String(255), ForeignKey('rag.knowledge_bases.id'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=True, index=True)
    
    # Operation details
    operation_type = Column(String(50), nullable=False, index=True)  # query, upload, delete, update
    query_text = Column(Text)
    results_count = Column(Integer)
    
    # Performance metrics
    processing_time_ms = Column(Float)
    tokens_used = Column(Integer)
    embedding_time_ms = Column(Float)
    search_time_ms = Column(Float)
    
    # Status and results
    success = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text)
    error_type = Column(String(100))
    
    # Context
    session_id = Column(String(255), index=True)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    operation_metadata = Column(JSON, default=dict)
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="usage_logs")
    
    def __repr__(self):
        return f"<KnowledgeBaseUsageLog(kb_id='{self.knowledge_base_id}', operation='{self.operation_type}', success={self.success})>"


class KnowledgeBaseTemplate(Base):
    """Templates for creating knowledge bases."""

    __tablename__ = "knowledge_base_templates"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Template information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    category = Column(String(100), index=True)
    use_case = Column(String(100), nullable=False, index=True)
    
    # Template configuration
    default_configuration = Column(JSON, nullable=False)
    recommended_models = Column(JSON, default=list)  # List of recommended embedding models
    chunking_strategy = Column(String(100), default='recursive')
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    
    # Template metadata
    version = Column(String(50), default='1.0.0')
    author = Column(String(255))
    tags = Column(JSON, default=list)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    
    # Status
    is_public = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Metadata
    template_metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<KnowledgeBaseTemplate(name='{self.name}', use_case='{self.use_case}')>"
