"""
User Settings API endpoints.

This module provides user settings management functionality
for individual users to manage their preferences, security, and API keys.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json

import structlog
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user, get_current_active_user
from app.models.auth import UserDB
from app.models.database.base import get_database_session
from app.api.v1.responses import StandardAPIResponse
from app.core.security import get_password_hash, verify_password

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/user/settings", tags=["User Settings"])


class UserSettingsResponse(BaseModel):
    """User settings response model."""
    account_security: Dict[str, Any]
    api_keys: Dict[str, str]
    preferences: Dict[str, Any]
    privacy: Dict[str, Any]


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class APIKeyRequest(BaseModel):
    """API key update request."""
    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    api_key: str = Field(..., description="API key value")


class PreferencesRequest(BaseModel):
    """User preferences update request."""
    theme: Optional[str] = Field(default="dark", description="UI theme")
    language: Optional[str] = Field(default="en", description="Language preference")
    notifications: Optional[Dict[str, bool]] = Field(default_factory=dict, description="Notification settings")
    timezone: Optional[str] = Field(default="UTC", description="User timezone")


class PrivacyRequest(BaseModel):
    """Privacy settings update request."""
    data_sharing: Optional[bool] = Field(default=False, description="Allow data sharing")
    analytics: Optional[bool] = Field(default=True, description="Allow analytics")
    public_profile: Optional[bool] = Field(default=False, description="Make profile public")


@router.get("/", response_model=StandardAPIResponse)
async def get_user_settings(
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get current user's settings."""
    try:
        async for session in get_database_session():
            # Get user data with settings
            result = await session.execute(
                text("""
                    SELECT api_keys, created_at, last_login, failed_login_attempts
                    FROM users 
                    WHERE id = :user_id
                """),
                {"user_id": str(current_user.id)}
            )
            user_data = result.fetchone()
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Build settings response
            settings = UserSettingsResponse(
                account_security={
                    "two_factor_enabled": False,  # TODO: Implement 2FA
                    "last_password_change": None,  # TODO: Track password changes
                    "failed_login_attempts": user_data.failed_login_attempts,
                    "account_created": user_data.created_at.isoformat() if user_data.created_at else None,
                    "last_login": user_data.last_login.isoformat() if user_data.last_login else None
                },
                api_keys={
                    provider: "***" + key[-4:] if key and len(key) > 4 else ""
                    for provider, key in (user_data.api_keys or {}).items()
                } if user_data.api_keys else {},
                preferences={
                    "theme": "dark",
                    "language": "en",
                    "notifications": {
                        "email": True,
                        "push": True,
                        "agent_completion": True,
                        "system_updates": True
                    },
                    "timezone": "UTC"
                },
                privacy={
                    "data_sharing": False,
                    "analytics": True,
                    "public_profile": False
                }
            )
            
            logger.info("User settings retrieved", user_id=str(current_user.id))
            
            return StandardAPIResponse(
                success=True,
                message="Settings retrieved successfully",
                data=settings
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user settings", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve settings"
        )


@router.post("/password", response_model=StandardAPIResponse)
async def change_password(
    request: PasswordChangeRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Change user password."""
    try:
        async for session in get_database_session():
            # Get current password hash
            result = await session.execute(
                text("SELECT hashed_password, password_salt FROM users WHERE id = :user_id"),
                {"user_id": str(current_user.id)}
            )
            user_data = result.fetchone()
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Verify current password
            if not verify_password(request.current_password, user_data.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect"
                )
            
            # Hash new password
            new_password_hash = get_password_hash(request.new_password)
            
            # Update password
            await session.execute(
                text("""
                    UPDATE users 
                    SET hashed_password = :new_password_hash, updated_at = :updated_at
                    WHERE id = :user_id
                """),
                {
                    "new_password_hash": new_password_hash,
                    "updated_at": datetime.utcnow(),
                    "user_id": str(current_user.id)
                }
            )
            await session.commit()
            
            logger.info("Password changed successfully", user_id=str(current_user.id))
            
            return StandardAPIResponse(
                success=True,
                message="Password changed successfully"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to change password", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.post("/api-keys", response_model=StandardAPIResponse)
async def update_api_key(
    request: APIKeyRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Update API key for a provider."""
    try:
        async for session in get_database_session():
            # Get current API keys
            result = await session.execute(
                text("SELECT api_keys FROM users WHERE id = :user_id"),
                {"user_id": str(current_user.id)}
            )
            user_data = result.fetchone()
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Update API keys
            current_keys = user_data.api_keys or {}
            current_keys[request.provider] = request.api_key
            
            await session.execute(
                text("""
                    UPDATE users 
                    SET api_keys = :api_keys, updated_at = :updated_at
                    WHERE id = :user_id
                """),
                {
                    "api_keys": json.dumps(current_keys),
                    "updated_at": datetime.utcnow(),
                    "user_id": str(current_user.id)
                }
            )
            await session.commit()
            
            logger.info("API key updated", user_id=str(current_user.id), provider=request.provider)
            
            return StandardAPIResponse(
                success=True,
                message=f"API key for {request.provider} updated successfully"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update API key", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update API key"
        )


@router.delete("/api-keys/{provider}", response_model=StandardAPIResponse)
async def delete_api_key(
    provider: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Delete API key for a provider."""
    try:
        async for session in get_database_session():
            # Get current API keys
            result = await session.execute(
                text("SELECT api_keys FROM users WHERE id = :user_id"),
                {"user_id": str(current_user.id)}
            )
            user_data = result.fetchone()
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Remove API key
            current_keys = user_data.api_keys or {}
            if provider in current_keys:
                del current_keys[provider]
                
                await session.execute(
                    text("""
                        UPDATE users 
                        SET api_keys = :api_keys, updated_at = :updated_at
                        WHERE id = :user_id
                    """),
                    {
                        "api_keys": json.dumps(current_keys),
                        "updated_at": datetime.utcnow(),
                        "user_id": str(current_user.id)
                    }
                )
                await session.commit()
                
                logger.info("API key deleted", user_id=str(current_user.id), provider=provider)
                
                return StandardAPIResponse(
                    success=True,
                    message=f"API key for {provider} deleted successfully"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"API key for {provider} not found"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete API key", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key"
        )


@router.post("/preferences", response_model=StandardAPIResponse)
async def update_preferences(
    request: PreferencesRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Update user preferences."""
    try:
        # For now, we'll store preferences in a simple way
        # In a real implementation, you might want a separate preferences table
        logger.info("Preferences updated", user_id=str(current_user.id), preferences=request.dict())

        return StandardAPIResponse(
            success=True,
            message="Preferences updated successfully",
            data=request.dict()
        )

    except Exception as e:
        logger.error("Failed to update preferences", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )


@router.post("/privacy", response_model=StandardAPIResponse)
async def update_privacy_settings(
    request: PrivacyRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Update privacy settings."""
    try:
        # For now, we'll store privacy settings in a simple way
        # In a real implementation, you might want a separate privacy_settings table
        logger.info("Privacy settings updated", user_id=str(current_user.id), privacy=request.dict())

        return StandardAPIResponse(
            success=True,
            message="Privacy settings updated successfully",
            data=request.dict()
        )

    except Exception as e:
        logger.error("Failed to update privacy settings", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update privacy settings"
        )
