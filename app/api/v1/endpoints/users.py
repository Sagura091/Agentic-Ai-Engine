"""
User Management API endpoints.

This module provides user management functionality for administrators
including user CRUD operations, role management, and user analytics.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

import structlog
from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import text

from app.core.auth import get_current_user, get_current_active_user
from app.models.user import User
from app.models.database.base import get_database_session
from app.core.pagination import AdvancedQueryParams
from app.api.v1.responses import StandardAPIResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/users", tags=["User Management"])


class UserResponse(BaseModel):
    """User response model."""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    is_superuser: bool
    created_at: datetime
    last_login: Optional[datetime]
    subscription_tier: Optional[str] = "free"
    api_quota_daily: Optional[int] = 1000
    preferences: Optional[Dict[str, Any]] = None


class UserCreateRequest(BaseModel):
    """User creation request for admins."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    is_superuser: bool = False
    subscription_tier: str = "free"


class UserUpdateRequest(BaseModel):
    """User update request for admins."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_superuser: Optional[bool] = None
    subscription_tier: Optional[str] = None
    api_quota_daily: Optional[int] = None
    preferences: Optional[Dict[str, Any]] = None


class UserStatsResponse(BaseModel):
    """User statistics response."""
    total_users: int
    active_users: int
    verified_users: int
    superusers: int
    users_by_tier: Dict[str, int]
    recent_registrations: int
    recent_logins: int


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin privileges."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


@router.get("/", response_model=StandardAPIResponse)
async def list_users(
    query_params: AdvancedQueryParams = Depends(),
    current_user: User = Depends(require_admin)
) -> StandardAPIResponse:
    """
    List all users with pagination and filtering.
    
    Admin only endpoint for user management.
    """
    try:
        async for session in get_database_session():
            # Build base query
            base_query = """
                SELECT id, username, email, full_name, is_active, is_verified, 
                       is_superuser, created_at, last_login, subscription_tier,
                       api_quota_daily, preferences
                FROM users 
                WHERE deleted_at IS NULL
            """
            
            # Add search filter
            params = {}
            if query_params.search:
                base_query += " AND (username ILIKE :search OR email ILIKE :search OR full_name ILIKE :search)"
                params["search"] = f"%{query_params.search}%"
            
            # Add status filter
            if query_params.filters:
                if "is_active" in query_params.filters:
                    base_query += " AND is_active = :is_active"
                    params["is_active"] = query_params.filters["is_active"] == "true"
                
                if "is_verified" in query_params.filters:
                    base_query += " AND is_verified = :is_verified"
                    params["is_verified"] = query_params.filters["is_verified"] == "true"
                
                if "subscription_tier" in query_params.filters:
                    base_query += " AND subscription_tier = :subscription_tier"
                    params["subscription_tier"] = query_params.filters["subscription_tier"]
            
            # Get total count
            count_query = f"SELECT COUNT(*) as total FROM ({base_query}) as filtered_users"
            count_result = await session.execute(text(count_query), params)
            total_count = count_result.fetchone().total
            
            # Add pagination
            base_query += f" ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
            params["limit"] = query_params.page_size
            params["offset"] = (query_params.page - 1) * query_params.page_size
            
            # Execute query
            result = await session.execute(text(base_query), params)
            users = result.fetchall()
            
            # Format response
            users_data = []
            for user in users:
                users_data.append(UserResponse(
                    id=str(user.id),
                    username=user.username,
                    email=user.email,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    is_verified=user.is_verified,
                    is_superuser=user.is_superuser,
                    created_at=user.created_at,
                    last_login=user.last_login,
                    subscription_tier=user.subscription_tier,
                    api_quota_daily=user.api_quota_daily,
                    preferences=user.preferences
                ))
            
            logger.info("Users listed", count=len(users_data), admin_user=current_user.username)
            
            return StandardAPIResponse(
                success=True,
                message=f"Retrieved {len(users_data)} users",
                data=users_data,
                pagination={
                    "page": query_params.page,
                    "page_size": query_params.page_size,
                    "total_items": total_count,
                    "total_pages": (total_count + query_params.page_size - 1) // query_params.page_size
                }
            )
            
    except Exception as e:
        logger.error("Failed to list users", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/{user_id}", response_model=StandardAPIResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_admin)
) -> StandardAPIResponse:
    """Get specific user details."""
    try:
        async for session in get_database_session():
            result = await session.execute(
                text("""
                    SELECT id, username, email, full_name, is_active, is_verified,
                           is_superuser, created_at, last_login, subscription_tier,
                           api_quota_daily, preferences, updated_at
                    FROM users 
                    WHERE id = :user_id AND deleted_at IS NULL
                """),
                {"user_id": user_id}
            )
            user = result.fetchone()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            user_data = UserResponse(
                id=str(user.id),
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_verified=user.is_verified,
                is_superuser=user.is_superuser,
                created_at=user.created_at,
                last_login=user.last_login,
                subscription_tier=user.subscription_tier,
                api_quota_daily=user.api_quota_daily,
                preferences=user.preferences
            )
            
            logger.info("User retrieved", user_id=user_id, admin_user=current_user.username)
            
            return StandardAPIResponse(
                success=True,
                message="User retrieved successfully",
                data=user_data
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.post("/", response_model=StandardAPIResponse)
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_admin)
) -> StandardAPIResponse:
    """Create a new user (admin only)."""
    try:
        from app.core.security import get_password_hash
        
        async for session in get_database_session():
            # Check if username exists
            existing_username = await session.execute(
                text("SELECT id FROM users WHERE username = :username"),
                {"username": request.username}
            )
            if existing_username.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )
            
            # Check if email exists
            existing_email = await session.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": request.email}
            )
            if existing_email.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists"
                )
            
            # Create user
            user_id = str(uuid.uuid4())
            hashed_password = get_password_hash(request.password)
            
            await session.execute(
                text("""
                    INSERT INTO users (
                        id, username, email, full_name, hashed_password,
                        is_active, is_verified, is_superuser, subscription_tier,
                        api_quota_daily, created_at
                    ) VALUES (
                        :id, :username, :email, :full_name, :hashed_password,
                        :is_active, :is_verified, :is_superuser, :subscription_tier,
                        :api_quota_daily, :created_at
                    )
                """),
                {
                    "id": user_id,
                    "username": request.username,
                    "email": request.email,
                    "full_name": request.full_name,
                    "hashed_password": hashed_password,
                    "is_active": request.is_active,
                    "is_verified": request.is_verified,
                    "is_superuser": request.is_superuser,
                    "subscription_tier": request.subscription_tier,
                    "api_quota_daily": 1000,  # Default quota
                    "created_at": datetime.utcnow()
                }
            )
            await session.commit()
            
            logger.info("User created by admin", user_id=user_id, admin_user=current_user.username)
            
            return StandardAPIResponse(
                success=True,
                message="User created successfully",
                data={"user_id": user_id, "username": request.username}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.put("/{user_id}", response_model=StandardAPIResponse)
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: User = Depends(require_admin)
) -> StandardAPIResponse:
    """Update user (admin only)."""
    try:
        async for session in get_database_session():
            # Check if user exists
            existing_user = await session.execute(
                text("SELECT id FROM users WHERE id = :user_id AND deleted_at IS NULL"),
                {"user_id": user_id}
            )
            if not existing_user.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )

            # Build update query
            update_fields = []
            params = {"user_id": user_id, "updated_at": datetime.utcnow()}

            if request.username is not None:
                # Check if username is taken by another user
                existing_username = await session.execute(
                    text("SELECT id FROM users WHERE username = :username AND id != :user_id"),
                    {"username": request.username, "user_id": user_id}
                )
                if existing_username.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username already exists"
                    )
                update_fields.append("username = :username")
                params["username"] = request.username

            if request.email is not None:
                # Check if email is taken by another user
                existing_email = await session.execute(
                    text("SELECT id FROM users WHERE email = :email AND id != :user_id"),
                    {"email": request.email, "user_id": user_id}
                )
                if existing_email.fetchone():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already exists"
                    )
                update_fields.append("email = :email")
                params["email"] = request.email

            if request.full_name is not None:
                update_fields.append("full_name = :full_name")
                params["full_name"] = request.full_name

            if request.is_active is not None:
                update_fields.append("is_active = :is_active")
                params["is_active"] = request.is_active

            if request.is_verified is not None:
                update_fields.append("is_verified = :is_verified")
                params["is_verified"] = request.is_verified

            if request.is_superuser is not None:
                update_fields.append("is_superuser = :is_superuser")
                params["is_superuser"] = request.is_superuser

            if request.subscription_tier is not None:
                update_fields.append("subscription_tier = :subscription_tier")
                params["subscription_tier"] = request.subscription_tier

            if request.api_quota_daily is not None:
                update_fields.append("api_quota_daily = :api_quota_daily")
                params["api_quota_daily"] = request.api_quota_daily

            if request.preferences is not None:
                update_fields.append("preferences = :preferences")
                params["preferences"] = request.preferences

            if update_fields:
                update_fields.append("updated_at = :updated_at")
                query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = :user_id"
                await session.execute(text(query), params)
                await session.commit()

            logger.info("User updated by admin", user_id=user_id, admin_user=current_user.username)

            return StandardAPIResponse(
                success=True,
                message="User updated successfully"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/{user_id}", response_model=StandardAPIResponse)
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_admin)
) -> StandardAPIResponse:
    """Delete user (admin only) - soft delete."""
    try:
        # Prevent admin from deleting themselves
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )

        async for session in get_database_session():
            # Check if user exists
            existing_user = await session.execute(
                text("SELECT id, username FROM users WHERE id = :user_id AND deleted_at IS NULL"),
                {"user_id": user_id}
            )
            user_row = existing_user.fetchone()

            if not user_row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )

            # Soft delete user
            await session.execute(
                text("UPDATE users SET deleted_at = :deleted_at WHERE id = :user_id"),
                {"deleted_at": datetime.utcnow(), "user_id": user_id}
            )
            await session.commit()

            logger.info("User deleted by admin", user_id=user_id, username=user_row.username, admin_user=current_user.username)

            return StandardAPIResponse(
                success=True,
                message=f"User {user_row.username} deleted successfully"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete user", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get("/stats/overview", response_model=StandardAPIResponse)
async def get_user_stats(
    current_user: User = Depends(require_admin)
) -> StandardAPIResponse:
    """Get user statistics overview (admin only)."""
    try:
        async for session in get_database_session():
            # Get basic stats
            stats_query = text("""
                SELECT
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN is_active = true THEN 1 END) as active_users,
                    COUNT(CASE WHEN is_verified = true THEN 1 END) as verified_users,
                    COUNT(CASE WHEN is_superuser = true THEN 1 END) as superusers
                FROM users
                WHERE deleted_at IS NULL
            """)
            stats_result = await session.execute(stats_query)
            stats = stats_result.fetchone()

            # Get users by tier
            tier_query = text("""
                SELECT subscription_tier, COUNT(*) as count
                FROM users
                WHERE deleted_at IS NULL
                GROUP BY subscription_tier
            """)
            tier_result = await session.execute(tier_query)
            users_by_tier = {row.subscription_tier: row.count for row in tier_result.fetchall()}

            # Get recent registrations (last 7 days)
            recent_reg_query = text("""
                SELECT COUNT(*) as count
                FROM users
                WHERE created_at >= NOW() - INTERVAL '7 days' AND deleted_at IS NULL
            """)
            recent_reg_result = await session.execute(recent_reg_query)
            recent_registrations = recent_reg_result.fetchone().count

            # Get recent logins (last 7 days)
            recent_login_query = text("""
                SELECT COUNT(*) as count
                FROM users
                WHERE last_login >= NOW() - INTERVAL '7 days' AND deleted_at IS NULL
            """)
            recent_login_result = await session.execute(recent_login_query)
            recent_logins = recent_login_result.fetchone().count

            stats_data = UserStatsResponse(
                total_users=stats.total_users,
                active_users=stats.active_users,
                verified_users=stats.verified_users,
                superusers=stats.superusers,
                users_by_tier=users_by_tier,
                recent_registrations=recent_registrations,
                recent_logins=recent_logins
            )

            logger.info("User stats retrieved", admin_user=current_user.username)

            return StandardAPIResponse(
                success=True,
                message="User statistics retrieved successfully",
                data=stats_data
            )

    except Exception as e:
        logger.error("Failed to get user stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user statistics"
        )
