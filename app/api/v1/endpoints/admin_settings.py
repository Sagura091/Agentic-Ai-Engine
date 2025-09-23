"""
Admin Settings API endpoints.

This module provides admin-only settings management functionality
for system administration, user management, and global configurations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import text, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user, get_current_active_user
from app.models.auth import UserDB
from app.models.database.base import get_database_session
from app.api.v1.responses import StandardAPIResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/settings", tags=["Admin Settings"])


class SystemStatsResponse(BaseModel):
    """System statistics response."""
    total_users: int
    active_users: int
    admin_users: int
    total_agents: int
    total_workflows: int
    system_uptime: str
    database_size: str


class UserManagementResponse(BaseModel):
    """User management response."""
    users: List[Dict[str, Any]]
    total_count: int
    active_count: int
    admin_count: int


class SystemSettingsResponse(BaseModel):
    """System settings response."""
    general: Dict[str, Any]
    security: Dict[str, Any]
    agents: Dict[str, Any]
    models: Dict[str, Any]
    monitoring: Dict[str, Any]


class SystemSettingsUpdateRequest(BaseModel):
    """System settings update request."""
    category: str = Field(..., description="Settings category")
    settings: Dict[str, Any] = Field(..., description="Settings to update")


def require_admin(current_user: UserDB = Depends(get_current_active_user)) -> UserDB:
    """Require admin privileges."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


@router.get("/stats", response_model=StandardAPIResponse)
async def get_system_stats(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Get system statistics (admin only)."""
    try:
        async for session in get_database_session():
            # Get user statistics
            user_stats = await session.execute(
                text("""
                    SELECT 
                        COUNT(*) as total_users,
                        COUNT(CASE WHEN is_active = true THEN 1 END) as active_users,
                        COUNT(CASE WHEN user_group = 'admin' THEN 1 END) as admin_users
                    FROM users
                """)
            )
            user_data = user_stats.fetchone()
            
            # TODO: Get agent and workflow counts when those tables exist
            # agent_count = await session.execute(text("SELECT COUNT(*) FROM agents"))
            # workflow_count = await session.execute(text("SELECT COUNT(*) FROM workflows"))
            
            stats = SystemStatsResponse(
                total_users=user_data.total_users or 0,
                active_users=user_data.active_users or 0,
                admin_users=user_data.admin_users or 0,
                total_agents=0,  # TODO: Implement when agents table exists
                total_workflows=0,  # TODO: Implement when workflows table exists
                system_uptime="N/A",  # TODO: Implement system uptime tracking
                database_size="N/A"  # TODO: Implement database size calculation
            )
            
            logger.info("System stats retrieved", admin_user=str(admin_user.id))
            
            return StandardAPIResponse(
                success=True,
                message="System statistics retrieved successfully",
                data=stats
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get system stats", admin_user=str(admin_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )


@router.get("/users", response_model=StandardAPIResponse)
async def get_user_management(
    admin_user: UserDB = Depends(require_admin),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search users by username or email")
) -> StandardAPIResponse:
    """Get user management data (admin only)."""
    try:
        async for session in get_database_session():
            # Build search condition
            search_condition = ""
            params = {"offset": (page - 1) * limit, "limit": limit}
            
            if search:
                search_condition = "WHERE username ILIKE :search OR email ILIKE :search"
                params["search"] = f"%{search}%"
            
            # Get users with pagination
            users_query = f"""
                SELECT id, username, email, name, user_group, is_active,
                       created_at, last_login, login_count, failed_login_attempts
                FROM users 
                {search_condition}
                ORDER BY created_at DESC
                OFFSET :offset LIMIT :limit
            """
            
            users_result = await session.execute(text(users_query), params)
            users_data = users_result.fetchall()
            
            # Get total count
            count_query = f"SELECT COUNT(*) as total FROM users {search_condition}"
            count_params = {"search": params.get("search")} if search else {}
            count_result = await session.execute(text(count_query), count_params)
            total_count = count_result.fetchone().total
            
            # Format users data
            users = []
            for user in users_data:
                users.append({
                    "id": str(user.id),
                    "username": user.username,
                    "email": user.email,
                    "name": user.name,
                    "user_group": user.user_group,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "login_count": user.login_count or 0,
                    "failed_login_attempts": user.failed_login_attempts or 0
                })
            
            # Get summary stats
            active_count = sum(1 for user in users if user["is_active"])
            admin_count = sum(1 for user in users if user["user_group"] == "admin")
            
            response_data = UserManagementResponse(
                users=users,
                total_count=total_count,
                active_count=active_count,
                admin_count=admin_count
            )
            
            logger.info("User management data retrieved", admin_user=str(admin_user.id), page=page, limit=limit)
            
            return StandardAPIResponse(
                success=True,
                message="User management data retrieved successfully",
                data=response_data
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user management data", admin_user=str(admin_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user management data"
        )


@router.get("/system", response_model=StandardAPIResponse)
async def get_system_settings(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Get system settings (admin only)."""
    try:
        # TODO: Implement actual system settings storage
        # For now, return default/example settings
        settings = SystemSettingsResponse(
            general={
                "app_name": "Agentic AI Platform",
                "version": "1.0.0",
                "environment": "production",
                "maintenance_mode": False,
                "max_users": 1000,
                "registration_enabled": True
            },
            security={
                "password_min_length": 8,
                "password_require_special": True,
                "session_timeout": 3600,
                "max_login_attempts": 5,
                "lockout_duration": 900,
                "two_factor_required": False
            },
            agents={
                "max_agents_per_user": 10,
                "max_concurrent_agents": 5,
                "agent_timeout": 300,
                "enable_agent_sharing": True,
                "default_model": "llama3.1:8b"
            },
            models={
                "ollama_base_url": "http://localhost:11434",
                "default_temperature": 0.7,
                "max_tokens": 4096,
                "timeout": 30,
                "retry_attempts": 3
            },
            monitoring={
                "enable_logging": True,
                "log_level": "INFO",
                "enable_metrics": True,
                "metrics_retention_days": 30,
                "enable_alerts": True
            }
        )
        
        logger.info("System settings retrieved", admin_user=str(admin_user.id))
        
        return StandardAPIResponse(
            success=True,
            message="System settings retrieved successfully",
            data=settings
        )
        
    except Exception as e:
        logger.error("Failed to get system settings", admin_user=str(admin_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system settings"
        )


@router.post("/system", response_model=StandardAPIResponse)
async def update_system_settings(
    request: SystemSettingsUpdateRequest,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Update system settings (admin only)."""
    try:
        # TODO: Implement actual system settings storage and validation
        # For now, just log the update
        logger.info(
            "System settings updated", 
            admin_user=str(admin_user.id), 
            category=request.category,
            settings=request.settings
        )
        
        return StandardAPIResponse(
            success=True,
            message=f"System settings for {request.category} updated successfully",
            data=request.settings
        )
        
    except Exception as e:
        logger.error("Failed to update system settings", admin_user=str(admin_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update system settings"
        )
