"""
SQLAlchemy models for API key management and rate limiting.

This module defines database models for API key management,
rate limiting, usage tracking, and access control.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import uuid4
import secrets

from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.database.base import Base


class APIKey(Base):
    """API key management and authentication."""

    __tablename__ = "api_keys"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    
    # Key information
    key_id = Column(String(255), nullable=False, unique=True, index=True)  # Public identifier
    key_hash = Column(String(255), nullable=False, unique=True)  # Hashed secret key
    key_prefix = Column(String(20), nullable=False)  # First few chars for identification
    
    # Key metadata
    name = Column(String(255), nullable=False)  # User-friendly name
    description = Column(Text)
    
    # Permissions and scope
    scopes = Column(JSON, default=list)  # List of allowed scopes/permissions
    allowed_endpoints = Column(JSON, default=list)  # Specific endpoints allowed
    allowed_methods = Column(JSON, default=list)  # HTTP methods allowed
    
    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=1000)
    rate_limit_per_day = Column(Integer, default=10000)
    rate_limit_per_month = Column(Integer, default=100000)
    
    # Usage quotas
    quota_requests_daily = Column(Integer)
    quota_requests_monthly = Column(Integer)
    quota_tokens_daily = Column(Integer)
    quota_tokens_monthly = Column(Integer)
    quota_storage_mb = Column(Float)
    
    # IP and domain restrictions
    allowed_ips = Column(JSON, default=list)  # List of allowed IP addresses/ranges
    allowed_domains = Column(JSON, default=list)  # List of allowed domains
    allowed_referers = Column(JSON, default=list)  # List of allowed referers
    
    # Key lifecycle
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    last_used = Column(DateTime(timezone=True))
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    is_revoked = Column(Boolean, default=False, index=True)
    revoked_at = Column(DateTime(timezone=True))
    revoked_reason = Column(String(255))
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    
    # Security
    require_https = Column(Boolean, default=True)
    require_signature = Column(Boolean, default=False)
    webhook_secret = Column(String(255))  # For webhook signatures
    
    # Metadata
    key_metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("UserDB", back_populates="api_keys")
    usage_logs = relationship("APIKeyUsageLog", back_populates="api_key", cascade="all, delete-orphan")
    rate_limit_logs = relationship("RateLimitLog", back_populates="api_key", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class APIKeyUsageLog(Base):
    """API key usage tracking."""

    __tablename__ = "api_key_usage_logs"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    api_key_id = Column(UUID(as_uuid=True), ForeignKey('api_keys.id'), nullable=False, index=True)
    
    # Request details
    endpoint = Column(String(500), nullable=False, index=True)
    method = Column(String(10), nullable=False, index=True)
    status_code = Column(Integer, nullable=False, index=True)
    
    # Request/response metrics
    request_size_bytes = Column(Integer)
    response_size_bytes = Column(Integer)
    processing_time_ms = Column(Float)
    
    # Token usage
    tokens_used = Column(Integer, default=0)
    token_type = Column(String(50))  # input, output, embedding, etc.
    
    # Client information
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    referer = Column(String(500))
    
    # Geographic information
    country = Column(String(2))  # ISO country code
    region = Column(String(100))
    city = Column(String(100))
    
    # Error information
    error_code = Column(String(100))
    error_message = Column(Text)
    error_type = Column(String(100))
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Additional metadata
    request_metadata = Column(JSON, default=dict)
    
    # Relationships
    api_key = relationship("APIKey", back_populates="usage_logs")
    
    def __repr__(self):
        return f"<APIKeyUsageLog(api_key_id={self.api_key_id}, endpoint='{self.endpoint}', status={self.status_code})>"


class RateLimitLog(Base):
    """Rate limiting events and violations."""

    __tablename__ = "rate_limit_logs"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    api_key_id = Column(UUID(as_uuid=True), ForeignKey('api_keys.id'), nullable=False, index=True)
    
    # Rate limit details
    limit_type = Column(String(50), nullable=False, index=True)  # per_minute, per_hour, per_day, per_month
    limit_value = Column(Integer, nullable=False)
    current_usage = Column(Integer, nullable=False)
    
    # Violation details
    is_violation = Column(Boolean, nullable=False, index=True)
    violation_type = Column(String(50))  # rate_exceeded, quota_exceeded, ip_blocked
    
    # Request context
    endpoint = Column(String(500))
    method = Column(String(10))
    ip_address = Column(String(45))
    
    # Time window
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    
    # Actions taken
    action_taken = Column(String(100))  # blocked, throttled, warned, allowed
    retry_after_seconds = Column(Integer)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    rate_limit_metadata = Column(JSON, default=dict)
    
    # Relationships
    api_key = relationship("APIKey", back_populates="rate_limit_logs")
    
    def __repr__(self):
        return f"<RateLimitLog(api_key_id={self.api_key_id}, type='{self.limit_type}', violation={self.is_violation})>"


class APIQuotaUsage(Base):
    """API quota usage tracking and aggregation."""

    __tablename__ = "api_quota_usage"
    __table_args__ = (
        UniqueConstraint('api_key_id', 'usage_date', 'quota_type', name='unique_quota_usage'),
        {'extend_existing': True}
    )
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    api_key_id = Column(UUID(as_uuid=True), ForeignKey('api_keys.id'), nullable=False, index=True)
    
    # Quota tracking
    usage_date = Column(DateTime(timezone=True), nullable=False, index=True)
    quota_type = Column(String(50), nullable=False, index=True)  # daily, monthly, hourly
    
    # Usage metrics
    requests_count = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    data_transferred_mb = Column(Float, default=0.0)
    storage_used_mb = Column(Float, default=0.0)
    
    # Quota limits
    requests_limit = Column(Integer)
    tokens_limit = Column(Integer)
    data_limit_mb = Column(Float)
    storage_limit_mb = Column(Float)
    
    # Status
    is_exceeded = Column(Boolean, default=False, index=True)
    exceeded_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Metadata
    usage_metadata = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<APIQuotaUsage(api_key_id={self.api_key_id}, type='{self.quota_type}', date={self.usage_date})>"


class APIEndpointMetrics(Base):
    """Aggregated metrics for API endpoints."""

    __tablename__ = "api_endpoint_metrics"
    __table_args__ = (
        UniqueConstraint('endpoint', 'method', 'metric_date', 'metric_period', name='unique_endpoint_metrics'),
        {'extend_existing': True}
    )
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Endpoint identification
    endpoint = Column(String(500), nullable=False, index=True)
    method = Column(String(10), nullable=False, index=True)
    
    # Time period
    metric_date = Column(DateTime(timezone=True), nullable=False, index=True)
    metric_period = Column(String(20), nullable=False)  # hourly, daily, weekly, monthly
    
    # Request metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time_ms = Column(Float, default=0.0)
    min_response_time_ms = Column(Float, default=0.0)
    max_response_time_ms = Column(Float, default=0.0)
    p95_response_time_ms = Column(Float, default=0.0)
    
    # Data transfer
    total_request_bytes = Column(Integer, default=0)
    total_response_bytes = Column(Integer, default=0)
    avg_request_size_bytes = Column(Float, default=0.0)
    avg_response_size_bytes = Column(Float, default=0.0)
    
    # Error analysis
    error_rate = Column(Float, default=0.0)
    most_common_error = Column(String(100))
    error_distribution = Column(JSON, default=dict)
    
    # Usage patterns
    unique_api_keys = Column(Integer, default=0)
    unique_ips = Column(Integer, default=0)
    peak_requests_per_minute = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<APIEndpointMetrics(endpoint='{self.endpoint}', method='{self.method}', period='{self.metric_period}')>"
