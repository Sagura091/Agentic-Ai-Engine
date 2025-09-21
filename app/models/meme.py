"""
Meme Models - Database models for meme storage and management.

This module contains SQLAlchemy models for storing meme data, analysis results,
and generation history in the unified database system.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4

from sqlalchemy import Column, String, Integer, Float, Text, JSON, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from app.models.database.base import Base

# Meme storage models


class MemeDB(Base):
    """Database model for storing collected memes."""
    
    __tablename__ = "memes"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    meme_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Basic meme information
    title = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    image_url = Column(Text, nullable=False)
    local_path = Column(String(500), nullable=True)
    
    # Source information
    source = Column(String(100), nullable=False)  # reddit, imgur, etc.
    subreddit = Column(String(100), nullable=True)
    author = Column(String(100), nullable=True)
    
    # Metrics
    score = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    
    # Content analysis
    text_content = Column(JSON, default=list)  # List of extracted text
    template_type = Column(String(100), nullable=True)
    content_category = Column(String(100), default='general')
    
    # Image properties
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    file_size = Column(Integer, default=0)
    content_hash = Column(String(64), nullable=True, index=True)
    
    # Timestamps
    created_utc = Column(Float, nullable=True)  # Original creation time
    collected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    processed = Column(Boolean, default=False)
    
    # Relationships
    analysis_results = relationship("MemeAnalysisDB", back_populates="meme", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MemeDB(meme_id='{self.meme_id}', title='{self.title[:50]}...')>"


class MemeAnalysisDB(Base):
    """Database model for storing meme analysis results."""
    
    __tablename__ = "meme_analysis"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Foreign key to meme
    meme_id = Column(UUID(as_uuid=True), ForeignKey('memes.id'), nullable=False)
    
    # Text analysis
    extracted_text = Column(JSON, default=list)  # List of text elements
    text_regions = Column(JSON, default=list)  # Bounding boxes for text
    readability_score = Column(Float, default=0.0)
    
    # Template matching
    template_matches = Column(JSON, default=list)  # List of (template_id, confidence)
    best_template_match = Column(String(100), nullable=True)
    template_confidence = Column(Float, default=0.0)
    
    # Visual analysis
    visual_features = Column(JSON, default=dict)
    dominant_colors = Column(JSON, default=list)
    complexity_score = Column(Float, default=0.0)
    
    # Sentiment and content analysis
    sentiment_score = Column(Float, default=0.0)
    humor_score = Column(Float, default=0.0)
    content_category = Column(String(100), default='unknown')
    detected_objects = Column(JSON, default=list)
    
    # Quality metrics
    overall_quality_score = Column(Float, default=0.0)
    virality_prediction = Column(Float, default=0.0)
    
    # Analysis metadata
    analysis_version = Column(String(50), default='1.0')
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, default=0.0)
    
    # Additional metadata
    metadata = Column(JSON, default=dict)
    
    # Relationships
    meme = relationship("MemeDB", back_populates="analysis_results")
    
    def __repr__(self):
        return f"<MemeAnalysisDB(analysis_id='{self.analysis_id}', quality={self.overall_quality_score})>"


class GeneratedMemeDB(Base):
    """Database model for storing AI-generated memes."""
    
    __tablename__ = "generated_memes"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    meme_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Generation request information
    prompt = Column(Text, nullable=False)
    style = Column(String(100), default='funny')
    target_audience = Column(String(100), default='general')
    
    # Generation details
    generation_method = Column(String(100), nullable=False)  # ai, template, hybrid
    template_used = Column(String(100), nullable=True)
    ai_model_used = Column(String(100), nullable=True)
    
    # Generated content
    image_path = Column(String(500), nullable=False)
    text_elements = Column(JSON, default=list)
    
    # Quality metrics
    quality_score = Column(Float, default=0.0)
    humor_score = Column(Float, default=0.0)
    creativity_score = Column(Float, default=0.0)
    
    # Generation metadata
    generation_time = Column(Float, default=0.0)  # Time taken to generate
    generation_parameters = Column(JSON, default=dict)
    
    # Agent information
    agent_id = Column(String(255), nullable=True)
    generation_session_id = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Performance tracking
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    engagement_score = Column(Float, default=0.0)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<GeneratedMemeDB(meme_id='{self.meme_id}', method='{self.generation_method}')>"


class MemeTemplateDB(Base):
    """Database model for storing meme templates."""
    
    __tablename__ = "meme_templates"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    template_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Template information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), default='general')
    
    # Template structure
    text_regions = Column(JSON, default=list)  # List of text region coordinates
    typical_text_count = Column(Integer, default=2)
    
    # Template files
    template_image_path = Column(String(500), nullable=True)
    example_images = Column(JSON, default=list)
    
    # Usage statistics
    usage_count = Column(Integer, default=0)
    popularity_score = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)  # Quality of memes generated with this template
    
    # Template metadata
    keywords = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    difficulty_level = Column(String(50), default='easy')  # easy, medium, hard
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<MemeTemplateDB(template_id='{self.template_id}', name='{self.name}')>"


class MemeAgentStateDB(Base):
    """Database model for storing meme agent state."""
    
    __tablename__ = "meme_agent_states"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Agent statistics
    total_memes_collected = Column(Integer, default=0)
    total_memes_analyzed = Column(Integer, default=0)
    total_memes_generated = Column(Integer, default=0)
    
    # Current state
    current_trends = Column(JSON, default=list)
    favorite_templates = Column(JSON, default=list)
    quality_scores = Column(JSON, default=list)
    
    # Learning progress
    learning_progress = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)
    
    # Last activity timestamps
    last_collection_time = Column(DateTime, nullable=True)
    last_analysis_time = Column(DateTime, nullable=True)
    last_generation_time = Column(DateTime, nullable=True)
    
    # Agent configuration
    configuration = Column(JSON, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    status = Column(String(100), default='idle')
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<MemeAgentStateDB(agent_id='{self.agent_id}', status='{self.status}')>"


class MemeTrendDB(Base):
    """Database model for tracking meme trends."""
    
    __tablename__ = "meme_trends"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    trend_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Trend information
    topic = Column(String(255), nullable=False)
    keywords = Column(JSON, default=list)
    description = Column(Text, nullable=True)
    
    # Trend metrics
    popularity_score = Column(Float, default=0.0)
    growth_rate = Column(Float, default=0.0)
    peak_score = Column(Float, default=0.0)
    current_score = Column(Float, default=0.0)
    
    # Time-based data
    trend_start_date = Column(DateTime, nullable=True)
    trend_peak_date = Column(DateTime, nullable=True)
    trend_end_date = Column(DateTime, nullable=True)
    
    # Associated content
    related_memes = Column(JSON, default=list)  # List of meme IDs
    related_templates = Column(JSON, default=list)  # List of template IDs
    
    # Source information
    detected_sources = Column(JSON, default=list)  # Where trend was detected
    confidence_score = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<MemeTrendDB(trend_id='{self.trend_id}', topic='{self.topic}')>"
