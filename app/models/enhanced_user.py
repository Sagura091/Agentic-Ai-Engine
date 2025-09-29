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


# UserDB class removed - now imported from app.models.auth to avoid conflicts


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


# REMOVED: Role model (roles now integrated into users.user_group field)
# class Role(Base):
#     """Role definitions for RBAC."""
#     __tablename__ = "roles"
#     # ... (model definition removed for optimized schema)


# REMOVED: UserRoleAssignment model (roles now integrated into users.user_group field)
# class UserRoleAssignment(Base):
#     """User role assignments."""
#     __tablename__ = "user_role_assignments"
#     # ... (model definition removed for optimized schema)


# REMOVED: UserAuditLog model (audit logging not needed for core functionality)
# class UserAuditLog(Base):
#     """Audit log for user actions."""
#     __tablename__ = "user_audit_logs"
#     # ... (model definition removed for optimized schema)
