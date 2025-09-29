"""
SQLAlchemy models for model management and tracking.

This module defines database models for storing model registry,
usage tracking, and download history to support the UniversalModelManager.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database.base import Base


class ModelRegistry(Base):
    """Registry for all downloaded and available models."""

    __tablename__ = "model_registry"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Model identification
    model_id = Column(String(255), nullable=False, unique=True, index=True)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False, index=True)  # embedding, reranking, vision, llm
    model_source = Column(String(50), nullable=False)  # huggingface, ollama, openai_api, anthropic_api
    
    # Model details
    description = Column(Text)
    dimension = Column(Integer)  # For embedding models
    max_sequence_length = Column(Integer)
    size_mb = Column(Float)
    
    # Download and storage info
    download_url = Column(Text)
    local_path = Column(Text)
    is_downloaded = Column(Boolean, default=False, index=True)
    is_available = Column(Boolean, default=True, index=True)
    
    # Usage tracking
    download_date = Column(DateTime(timezone=True))
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    
    # Performance metrics
    average_inference_time_ms = Column(Float, default=0.0)
    total_tokens_processed = Column(Integer, default=0)
    success_rate = Column(Float, default=1.0)
    
    # Metadata and configuration
    model_metadata = Column(JSON, default=dict)
    configuration = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    usage_logs = relationship("ModelUsageLog", back_populates="model", cascade="all, delete-orphan")
    download_history = relationship("ModelDownloadHistory", back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ModelRegistry(id={self.model_id}, name='{self.model_name}', type='{self.model_type}')>"


class ModelUsageLog(Base):
    """Tracking individual model usage events."""

    __tablename__ = "model_usage_logs"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    model_id = Column(String(255), ForeignKey('rag.model_registry.model_id'), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=True, index=True)
    
    # Usage details
    usage_type = Column(String(50), nullable=False, index=True)  # embedding, inference, reranking, vision
    operation = Column(String(100))  # embed_text, embed_image, generate, rerank
    
    # Performance metrics
    tokens_processed = Column(Integer)
    input_length = Column(Integer)
    output_length = Column(Integer)
    processing_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    
    # Status and results
    success = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text)
    error_type = Column(String(100))
    
    # Context and metadata
    session_id = Column(String(255), index=True)
    request_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    model = relationship("ModelRegistry", back_populates="usage_logs")
    
    def __repr__(self):
        return f"<ModelUsageLog(model_id={self.model_id}, usage_type='{self.usage_type}', success={self.success})>"


class ModelDownloadHistory(Base):
    """Tracking model download attempts and progress."""

    __tablename__ = "model_download_history"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    model_id = Column(String(255), ForeignKey('rag.model_registry.model_id'), nullable=False, index=True)
    
    # Download details
    download_status = Column(String(50), nullable=False, index=True)  # started, in_progress, completed, failed, cancelled
    download_progress = Column(Float, default=0.0)  # 0.0 to 100.0
    download_speed_mbps = Column(Float)
    
    # File information
    total_size_mb = Column(Float)
    downloaded_size_mb = Column(Float, default=0.0)
    file_path = Column(Text)
    checksum = Column(String(255))
    
    # Error handling
    error_message = Column(Text)
    error_type = Column(String(100))
    retry_count = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    
    # Context
    initiated_by = Column(String(255))  # user_id or system
    download_source = Column(String(255))  # URL or source identifier
    download_metadata = Column(JSON, default=dict)
    
    # Relationships
    model = relationship("ModelRegistry", back_populates="download_history")
    
    def __repr__(self):
        return f"<ModelDownloadHistory(model_id={self.model_id}, status='{self.download_status}', progress={self.download_progress}%)>"


class ModelPerformanceMetrics(Base):
    """Aggregated performance metrics for models."""

    __tablename__ = "model_performance_metrics"
    __table_args__ = {'schema': 'rag', 'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    model_id = Column(String(255), ForeignKey('rag.model_registry.model_id'), nullable=False, index=True)
    
    # Time period for metrics
    metric_date = Column(DateTime(timezone=True), nullable=False, index=True)
    metric_period = Column(String(20), nullable=False)  # hourly, daily, weekly, monthly
    
    # Usage metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Performance metrics
    avg_processing_time_ms = Column(Float, default=0.0)
    min_processing_time_ms = Column(Float, default=0.0)
    max_processing_time_ms = Column(Float, default=0.0)
    p95_processing_time_ms = Column(Float, default=0.0)
    
    # Throughput metrics
    total_tokens_processed = Column(Integer, default=0)
    avg_tokens_per_second = Column(Float, default=0.0)
    peak_tokens_per_second = Column(Float, default=0.0)
    
    # Resource usage
    avg_memory_usage_mb = Column(Float, default=0.0)
    peak_memory_usage_mb = Column(Float, default=0.0)
    avg_cpu_usage_percent = Column(Float, default=0.0)
    
    # Quality metrics (if available)
    avg_quality_score = Column(Float)
    user_satisfaction_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<ModelPerformanceMetrics(model_id={self.model_id}, period='{self.metric_period}', date={self.metric_date})>"
