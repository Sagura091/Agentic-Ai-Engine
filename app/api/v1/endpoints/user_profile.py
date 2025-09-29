"""
User Profile API endpoints.

This module provides user profile management functionality
for users to manage their public identity and profile information.
"""

from typing import Optional
from datetime import datetime
import json

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user, get_current_active_user
from app.models.auth import UserDB
from app.models.database.base import get_database_session
from app.api.v1.responses import StandardAPIResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/user/profile", tags=["User Profile"])


class UserProfileResponse(BaseModel):
    """User profile response model."""
    id: str
    username: str
    email: str
    name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    user_group: str
    is_active: bool
    created_at: str
    last_login: Optional[str]
    stats: dict


class ProfileUpdateRequest(BaseModel):
    """Profile update request."""
    name: Optional[str] = Field(None, max_length=255, description="Display name")
    bio: Optional[str] = Field(None, max_length=500, description="User bio")
    avatar_url: Optional[str] = Field(None, description="Avatar image URL")


class PublicProfileResponse(BaseModel):
    """Public profile response (limited info)."""
    username: str
    name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    user_group: str
    created_at: str
    stats: dict


@router.get("/", response_model=StandardAPIResponse)
async def get_user_profile(
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get current user's profile."""
    try:
        async for session in get_database_session():
            # Get user profile data
            result = await session.execute(
                text("""
                    SELECT id, username, email, name, user_group, is_active,
                           created_at, updated_at, last_login, login_count
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
            
            # Get user activity stats (placeholder - implement based on your needs)
            stats = {
                "total_logins": user_data.login_count or 0,
                "agents_created": 0,  # TODO: Count from agents table
                "workflows_created": 0,  # TODO: Count from workflows table
                "last_activity": user_data.last_login.isoformat() if user_data.last_login else None
            }
            
            profile = UserProfileResponse(
                id=str(user_data.id),
                username=user_data.username,
                email=user_data.email,
                name=user_data.name,
                bio=None,  # TODO: Add bio field to users table
                avatar_url=None,  # TODO: Add avatar_url field to users table
                user_group=user_data.user_group,
                is_active=user_data.is_active,
                created_at=user_data.created_at.isoformat() if user_data.created_at else "",
                last_login=user_data.last_login.isoformat() if user_data.last_login else None,
                stats=stats
            )
            
            logger.info("User profile retrieved", user_id=str(current_user.id))
            
            return StandardAPIResponse(
                success=True,
                message="Profile retrieved successfully",
                data=profile
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user profile", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )


@router.put("/", response_model=StandardAPIResponse)
async def update_user_profile(
    request: ProfileUpdateRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Update current user's profile."""
    try:
        async for session in get_database_session():
            # Build update query dynamically based on provided fields
            update_fields = []
            params = {"user_id": str(current_user.id), "updated_at": datetime.utcnow()}
            
            if request.name is not None:
                update_fields.append("name = :name")
                params["name"] = request.name
            
            # TODO: Add bio and avatar_url fields to users table
            # if request.bio is not None:
            #     update_fields.append("bio = :bio")
            #     params["bio"] = request.bio
            
            # if request.avatar_url is not None:
            #     update_fields.append("avatar_url = :avatar_url")
            #     params["avatar_url"] = request.avatar_url
            
            if not update_fields:
                return StandardAPIResponse(
                    success=True,
                    message="No changes to update"
                )
            
            # Update user profile
            query = f"""
                UPDATE users 
                SET {', '.join(update_fields)}, updated_at = :updated_at
                WHERE id = :user_id
            """
            
            await session.execute(text(query), params)
            await session.commit()
            
            logger.info("User profile updated", user_id=str(current_user.id), fields=list(params.keys()))
            
            return StandardAPIResponse(
                success=True,
                message="Profile updated successfully"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user profile", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.get("/{username}/public", response_model=StandardAPIResponse)
async def get_public_profile(
    username: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get public profile of a user by username."""
    try:
        async for session in get_database_session():
            # Get public profile data
            result = await session.execute(
                text("""
                    SELECT username, name, user_group, created_at, login_count
                    FROM users 
                    WHERE username = :username AND is_active = true
                """),
                {"username": username}
            )
            user_data = result.fetchone()
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Get public stats
            stats = {
                "member_since": user_data.created_at.isoformat() if user_data.created_at else "",
                "total_logins": user_data.login_count or 0,
                "public_agents": 0,  # TODO: Count public agents
                "public_workflows": 0  # TODO: Count public workflows
            }
            
            profile = PublicProfileResponse(
                username=user_data.username,
                name=user_data.name,
                bio=None,  # TODO: Add bio field
                avatar_url=None,  # TODO: Add avatar_url field
                user_group=user_data.user_group,
                created_at=user_data.created_at.isoformat() if user_data.created_at else "",
                stats=stats
            )
            
            logger.info("Public profile retrieved", username=username, viewer=str(current_user.id))
            
            return StandardAPIResponse(
                success=True,
                message="Public profile retrieved successfully",
                data=profile
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get public profile", username=username, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve public profile"
        )


@router.post("/avatar", response_model=StandardAPIResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Upload user avatar image."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Validate file size (max 5MB)
        if file.size and file.size > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size must be less than 5MB"
            )
        
        # TODO: Implement actual file upload to storage service
        # For now, return a placeholder response
        avatar_url = f"/avatars/{current_user.id}/{file.filename}"
        
        logger.info("Avatar uploaded", user_id=str(current_user.id), filename=file.filename)
        
        return StandardAPIResponse(
            success=True,
            message="Avatar uploaded successfully",
            data={"avatar_url": avatar_url}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload avatar", user_id=str(current_user.id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload avatar"
        )
