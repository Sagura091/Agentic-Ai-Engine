"""
Enhanced SQLAlchemy models for user management and authentication.

This module defines comprehensive database models for user management,
authentication, sessions, roles, and permissions.
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


class UserDB(Base):
    """Enhanced user model with full authentication support."""

    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic information
    username = Column(String(255), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    full_name = Column(String(255))
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    password_salt = Column(String(255))
    password_reset_token = Column(String(255), unique=True)
    password_reset_expires = Column(DateTime(timezone=True))
    
    # Account status
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False, index=True)
    is_superuser = Column(Boolean, default=False, index=True)
    
    # Email verification
    email_verification_token = Column(String(255), unique=True)
    email_verified_at = Column(DateTime(timezone=True))
    
    # Profile information
    avatar_url = Column(String(500))
    bio = Column(Text)
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    
    # Activity tracking
    last_login = Column(DateTime(timezone=True))
    last_activity = Column(DateTime(timezone=True))
    login_count = Column(Integer, default=0)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    
    # Preferences and settings
    preferences = Column(JSON, default=dict)
    notification_settings = Column(JSON, default=dict)
    privacy_settings = Column(JSON, default=dict)
    
    # Subscription and limits
    subscription_tier = Column(String(50), default='free')  # free, pro, enterprise
    api_quota_daily = Column(Integer, default=1000)
    api_quota_monthly = Column(Integer, default=10000)
    storage_limit_mb = Column(Float, default=100.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True))  # Soft delete
    
    # Metadata
    user_metadata = Column(JSON, default=dict)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    role_assignments = relationship("UserRoleAssignment", back_populates="user", foreign_keys="UserRoleAssignment.user_id", cascade="all, delete-orphan")
    audit_logs = relationship("UserAuditLog", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<UserDB(id={self.id}, username='{self.username}', email='{self.email}')>"


class UserSession(Base):
    """User session management."""

    __tablename__ = "user_sessions"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    
    # Session details
    session_token = Column(String(255), nullable=False, unique=True, index=True)
    refresh_token = Column(String(255), unique=True, index=True)
    session_type = Column(String(50), default='web')  # web, mobile, api, service
    
    # Device and location information
    device_info = Column(JSON, default=dict)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    location = Column(JSON, default=dict)  # Country, city, etc.
    
    # Session lifecycle
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    revoked_at = Column(DateTime(timezone=True))
    revoked_reason = Column(String(255))
    
    # Security
    csrf_token = Column(String(255))
    security_flags = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("UserDB", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, type='{self.session_type}')>"


class Role(Base):
    """Role definitions for RBAC."""

    __tablename__ = "roles"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Role information
    name = Column(String(100), nullable=False, unique=True, index=True)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Role hierarchy
    parent_role_id = Column(UUID(as_uuid=True), ForeignKey('roles.id'))
    level = Column(Integer, default=0)  # Hierarchy level
    
    # Role properties
    is_system_role = Column(Boolean, default=False)  # System-defined roles
    is_active = Column(Boolean, default=True, index=True)
    
    # Permissions
    permissions = Column(JSON, default=list)  # List of permission strings
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Metadata
    role_metadata = Column(JSON, default=dict)
    
    # Relationships
    user_assignments = relationship("UserRoleAssignment", back_populates="role", cascade="all, delete-orphan")
    child_roles = relationship("Role", backref="parent_role", remote_side=[id])
    
    def __repr__(self):
        return f"<Role(id={self.id}, name='{self.name}')>"


class UserRoleAssignment(Base):
    """User role assignments."""

    __tablename__ = "user_role_assignments"
    __table_args__ = (
        UniqueConstraint('user_id', 'role_id', name='unique_user_role'),
        {'extend_existing': True}
    )
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey('roles.id'), nullable=False, index=True)
    
    # Assignment details
    assigned_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # Context and conditions
    context = Column(JSON, default=dict)  # Additional context for the assignment
    conditions = Column(JSON, default=dict)  # Conditional permissions
    
    # Relationships
    user = relationship("UserDB", back_populates="role_assignments", foreign_keys=[user_id])
    role = relationship("Role", back_populates="user_assignments")
    assigner = relationship("UserDB", foreign_keys=[assigned_by])
    
    def __repr__(self):
        return f"<UserRoleAssignment(user_id={self.user_id}, role_id={self.role_id})>"


class UserAuditLog(Base):
    """Audit log for user actions."""

    __tablename__ = "user_audit_logs"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    
    # Action details
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), index=True)
    resource_id = Column(String(255), index=True)
    
    # Context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    session_id = Column(UUID(as_uuid=True), ForeignKey('user_sessions.id'))
    
    # Results
    success = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text)
    
    # Additional data
    old_values = Column(JSON, default=dict)
    new_values = Column(JSON, default=dict)
    audit_metadata = Column(JSON, default=dict)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("UserDB", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<UserAuditLog(user_id={self.user_id}, action='{self.action}', success={self.success})>"
