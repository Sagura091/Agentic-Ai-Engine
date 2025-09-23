"""
Admin Settings Database Models.

This module provides database models for storing and managing admin settings
that are configured through the enhanced admin panel.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import Column, String, Text, DateTime, Boolean, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from app.models.database.base import Base


class AdminSetting(Base):
    """
    Admin setting storage model.
    
    Stores individual admin settings with metadata, validation, and audit trail.
    """
    
    __tablename__ = "admin_settings"
    __table_args__ = (
        Index('idx_admin_settings_category_key', 'category', 'key', unique=True),
        Index('idx_admin_settings_category', 'category'),
        Index('idx_admin_settings_updated', 'updated_at'),
        {'extend_existing': True}
    )
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Setting identification
    category = Column(String(100), nullable=False, index=True)
    key = Column(String(200), nullable=False, index=True)
    
    # Setting value and metadata
    value = Column(JSON, nullable=False)
    default_value = Column(JSON, nullable=False)
    setting_type = Column(String(50), nullable=False)  # string, integer, boolean, etc.
    
    # Setting configuration
    description = Column(Text)
    security_level = Column(String(50), default="admin_only")
    requires_restart = Column(Boolean, default=False)
    validation_rules = Column(JSON, default=dict)
    enum_values = Column(JSON, nullable=True)
    
    # Audit trail
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    updated_by = Column(UUID(as_uuid=True), nullable=True)  # User ID who made the change
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    is_system_managed = Column(Boolean, default=False)  # True for system-critical settings
    
    def __repr__(self):
        return f"<AdminSetting(category='{self.category}', key='{self.key}', value='{self.value}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "default_value": self.default_value,
            "setting_type": self.setting_type,
            "description": self.description,
            "security_level": self.security_level,
            "requires_restart": self.requires_restart,
            "validation_rules": self.validation_rules,
            "enum_values": self.enum_values,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "updated_by": str(self.updated_by) if self.updated_by else None,
            "is_active": self.is_active,
            "is_system_managed": self.is_system_managed
        }


class AdminSettingHistory(Base):
    """
    Admin setting change history model.
    
    Maintains a complete audit trail of all setting changes for compliance and debugging.
    """
    
    __tablename__ = "admin_setting_history"
    __table_args__ = (
        Index('idx_admin_setting_history_setting_id', 'setting_id'),
        Index('idx_admin_setting_history_changed_at', 'changed_at'),
        Index('idx_admin_setting_history_changed_by', 'changed_by'),
        {'extend_existing': True}
    )
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Reference to the setting
    setting_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    category = Column(String(100), nullable=False)
    key = Column(String(200), nullable=False)
    
    # Change details
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=False)
    change_reason = Column(Text, nullable=True)
    
    # Audit information
    changed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    changed_by = Column(UUID(as_uuid=True), nullable=False)  # User ID who made the change
    
    # System information
    system_restart_required = Column(Boolean, default=False)
    applied_at = Column(DateTime(timezone=True), nullable=True)  # When the setting was actually applied
    
    def __repr__(self):
        return f"<AdminSettingHistory(category='{self.category}', key='{self.key}', changed_at='{self.changed_at}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "setting_id": str(self.setting_id),
            "category": self.category,
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_reason": self.change_reason,
            "changed_at": self.changed_at.isoformat() if self.changed_at else None,
            "changed_by": str(self.changed_by),
            "system_restart_required": self.system_restart_required,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None
        }


class SystemConfigurationCache(Base):
    """
    System configuration cache model.
    
    Caches the current active system configuration for fast access by running systems.
    This is rebuilt whenever settings change and provides the single source of truth
    for all system components.
    """
    
    __tablename__ = "system_configuration_cache"
    __table_args__ = (
        Index('idx_system_config_cache_component', 'component_name'),
        Index('idx_system_config_cache_updated', 'updated_at'),
        {'extend_existing': True}
    )
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Component identification
    component_name = Column(String(100), nullable=False, unique=True, index=True)  # rag_system, agent_manager, etc.
    
    # Configuration data
    configuration = Column(JSON, nullable=False)
    configuration_hash = Column(String(64), nullable=False)  # For change detection
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    requires_restart = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<SystemConfigurationCache(component='{self.component_name}', updated='{self.updated_at}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "component_name": self.component_name,
            "configuration": self.configuration,
            "configuration_hash": self.configuration_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "requires_restart": self.requires_restart
        }
