"""
Settings API endpoints.

This module provides system settings management functionality.
"""

from typing import Dict, Any, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config.settings import get_settings as get_app_settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/settings", tags=["Settings"])


class SystemSettings(BaseModel):
    """System settings response."""
    general: Dict[str, Any]
    agents: Dict[str, Any]
    models: Dict[str, Any]
    monitoring: Dict[str, Any]
    security: Dict[str, Any]
    integrations: Dict[str, Any]


class SettingsUpdateRequest(BaseModel):
    """Settings update request."""
    category: str = Field(..., description="Settings category")
    settings: Dict[str, Any] = Field(..., description="Settings to update")


@router.get("", response_model=SystemSettings)
async def get_settings() -> SystemSettings:
    """
    Get current system settings.

    Returns:
        Current system settings
    """
    try:
        settings = get_app_settings()
        
        # Build settings response
        system_settings = SystemSettings(
            general={
                "app_name": "Agentic AI Microservice",
                "version": "0.1.0",
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG,
                "log_level": settings.LOG_LEVEL,
                "timezone": "UTC",
                "language": "en"
            },
            agents={
                "default_model": settings.DEFAULT_AGENT_MODEL,
                "max_concurrent_agents": settings.MAX_CONCURRENT_AGENTS,
                "agent_timeout": 300,  # seconds
                "auto_cleanup": True,
                "enable_memory": True,
                "enable_tools": True
            },
            models={
                "ollama_base_url": settings.OLLAMA_BASE_URL,
                "default_temperature": 0.7,
                "max_tokens": 4096,
                "timeout": 30,
                "retry_attempts": 3
            },
            monitoring={
                "enable_metrics": True,
                "metrics_interval": 60,  # seconds
                "enable_health_checks": True,
                "health_check_interval": 30,  # seconds
                "log_retention_days": 30
            },
            security={
                "cors_origins": settings.CORS_ORIGINS,
                "enable_authentication": False,  # Not implemented yet
                "rate_limiting": True,
                "max_requests_per_minute": 100
            },
            integrations={
                "openwebui_enabled": settings.OPENWEBUI_ENABLED,
                "openwebui_url": settings.OPENWEBUI_URL if hasattr(settings, 'OPENWEBUI_URL') else None,
                "redis_enabled": bool(settings.REDIS_URL),
                "redis_url": settings.REDIS_URL if settings.REDIS_URL else None,
                "websocket_enabled": True
            }
        )
        
        logger.info("Settings retrieved")
        
        return system_settings
        
    except Exception as e:
        logger.error("Failed to get settings", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve settings")


@router.put("")
async def update_settings(request: SettingsUpdateRequest) -> Dict[str, Any]:
    """
    Update system settings.
    
    Args:
        request: Settings update request
        
    Returns:
        Update result
    """
    try:
        # Note: This is a read-only implementation for now
        # In a full implementation, you would validate and persist settings
        
        logger.warning(
            "Settings update attempted (read-only mode)",
            category=request.category,
            settings_keys=list(request.settings.keys())
        )
        
        return {
            "status": "success",
            "message": "Settings update received (read-only mode)",
            "category": request.category,
            "updated_settings": request.settings,
            "timestamp": datetime.now().isoformat(),
            "note": "Settings updates are not persisted in this version"
        }
        
    except Exception as e:
        logger.error("Failed to update settings", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update settings")


@router.get("/categories")
async def get_settings_categories() -> Dict[str, Any]:
    """
    Get available settings categories.
    
    Returns:
        Available settings categories and their descriptions
    """
    try:
        categories = {
            "general": {
                "name": "General Settings",
                "description": "Basic application configuration",
                "editable": False
            },
            "agents": {
                "name": "Agent Settings",
                "description": "Agent behavior and limits configuration",
                "editable": True
            },
            "models": {
                "name": "Model Settings",
                "description": "AI model configuration and parameters",
                "editable": True
            },
            "monitoring": {
                "name": "Monitoring Settings",
                "description": "Monitoring and metrics configuration",
                "editable": True
            },
            "security": {
                "name": "Security Settings",
                "description": "Security and access control settings",
                "editable": False
            },
            "integrations": {
                "name": "Integration Settings",
                "description": "External service integrations",
                "editable": True
            }
        }
        
        logger.info("Settings categories retrieved")
        
        return {
            "categories": categories,
            "total_count": len(categories)
        }
        
    except Exception as e:
        logger.error("Failed to get settings categories", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve settings categories")


@router.get("/schema")
async def get_settings_schema() -> Dict[str, Any]:
    """
    Get settings schema for validation.
    
    Returns:
        Settings schema definition
    """
    try:
        schema = {
            "general": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "readonly": True},
                    "version": {"type": "string", "readonly": True},
                    "environment": {"type": "string", "readonly": True},
                    "debug": {"type": "boolean", "readonly": True},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"], "readonly": True},
                    "timezone": {"type": "string", "default": "UTC"},
                    "language": {"type": "string", "default": "en"}
                }
            },
            "agents": {
                "type": "object",
                "properties": {
                    "default_model": {"type": "string", "default": "llama3.2:latest"},
                    "max_concurrent_agents": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                    "agent_timeout": {"type": "integer", "minimum": 30, "maximum": 3600, "default": 300},
                    "auto_cleanup": {"type": "boolean", "default": True},
                    "enable_memory": {"type": "boolean", "default": True},
                    "enable_tools": {"type": "boolean", "default": True}
                }
            },
            "models": {
                "type": "object",
                "properties": {
                    "ollama_base_url": {"type": "string", "format": "uri"},
                    "default_temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0, "default": 0.7},
                    "max_tokens": {"type": "integer", "minimum": 1, "maximum": 32768, "default": 4096},
                    "timeout": {"type": "integer", "minimum": 5, "maximum": 300, "default": 30},
                    "retry_attempts": {"type": "integer", "minimum": 0, "maximum": 10, "default": 3}
                }
            },
            "monitoring": {
                "type": "object",
                "properties": {
                    "enable_metrics": {"type": "boolean", "default": True},
                    "metrics_interval": {"type": "integer", "minimum": 10, "maximum": 3600, "default": 60},
                    "enable_health_checks": {"type": "boolean", "default": True},
                    "health_check_interval": {"type": "integer", "minimum": 5, "maximum": 300, "default": 30},
                    "log_retention_days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 30}
                }
            },
            "security": {
                "type": "object",
                "properties": {
                    "cors_origins": {"type": "array", "items": {"type": "string"}, "readonly": True},
                    "enable_authentication": {"type": "boolean", "default": False, "readonly": True},
                    "rate_limiting": {"type": "boolean", "default": True},
                    "max_requests_per_minute": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 100}
                }
            },
            "integrations": {
                "type": "object",
                "properties": {
                    "openwebui_enabled": {"type": "boolean", "readonly": True},
                    "openwebui_url": {"type": "string", "format": "uri", "nullable": True},
                    "redis_enabled": {"type": "boolean", "readonly": True},
                    "redis_url": {"type": "string", "nullable": True, "readonly": True},
                    "websocket_enabled": {"type": "boolean", "default": True}
                }
            }
        }
        
        logger.info("Settings schema retrieved")
        
        return {
            "schema": schema,
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error("Failed to get settings schema", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve settings schema")


@router.post("/reset")
async def reset_settings() -> Dict[str, Any]:
    """
    Reset settings to defaults.
    
    Returns:
        Reset operation result
    """
    try:
        logger.warning("Settings reset attempted (read-only mode)")
        
        return {
            "status": "success",
            "message": "Settings reset completed (read-only mode)",
            "timestamp": datetime.now().isoformat(),
            "note": "Settings are not persisted in this version"
        }
        
    except Exception as e:
        logger.error("Failed to reset settings", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to reset settings")
