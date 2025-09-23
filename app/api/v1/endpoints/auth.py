"""
Authentication API Endpoints.

This module provides REST API endpoints for user authentication,
registration, session management, and user profile operations.
"""

from typing import Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app.services.auth_service import auth_service
from app.models.auth import UserCreate, UserLogin, TokenResponse, UserResponse
from app.backend_logging.backend_logger import get_logger, LogCategory

logger = structlog.get_logger(__name__)
security = HTTPBearer(auto_error=False)

router = APIRouter(tags=["Authentication"])


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """
    Dependency to get current authenticated user.

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        UserResponse: Current user information

    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await auth_service.get_current_user(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: str = Field(..., description="Email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str = Field(..., description="Refresh token")


class FirstTimeSetupResponse(BaseModel):
    """First-time setup status response."""
    is_first_time_setup: bool = Field(..., description="Whether this is first-time setup")
    message: str = Field(..., description="Status message")


@router.get("/setup/status", response_model=FirstTimeSetupResponse)
async def get_first_time_setup_status() -> FirstTimeSetupResponse:
    """
    Check if this is the first-time setup.

    Returns whether any users exist in the system. If no users exist,
    the next registration will create the first admin user.

    Returns:
        First-time setup status information
    """
    try:
        is_first_time = await auth_service.is_first_time_setup()

        if is_first_time:
            message = "No users found. The first registered user will become an administrator."
        else:
            message = "System has been initialized with users."

        get_logger().info(
            f"First-time setup status checked: {is_first_time}",
            LogCategory.USER_INTERACTION,
            "AuthAPI",
            data={"is_first_time_setup": is_first_time}
        )

        return FirstTimeSetupResponse(
            is_first_time_setup=is_first_time,
            message=message
        )

    except Exception as e:
        get_logger().error(
            f"Error checking first-time setup status: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AuthAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check setup status"
        )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    request: Request
) -> TokenResponse:
    """
    Register a new user account.

    Creates a new user account with proper validation, password hashing,
    and automatic login with session creation.

    **First-Time Setup**: If this is the first user in the system, they will
    automatically become an administrator with full system privileges.

    Args:
        user_data: User registration information
        request: HTTP request for IP tracking

    Returns:
        Authentication tokens and user information

    Raises:
        HTTPException: If registration fails or user already exists
    """
    try:
        # Get client IP for security tracking
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")

        # Register user
        user_response, token_response = await auth_service.register_user(user_data)

        get_logger().info(
            f"User registered successfully: {user_response.username}",
            LogCategory.USER_INTERACTION,
            "AuthAPI",
            data={
                "user_id": user_response.id,
                "username": user_response.username,
                "email": user_response.email,
                "ip_address": client_ip
            }
        )

        return token_response

    except ValueError as e:
        get_logger().warn(
            f"User registration failed: {str(e)}",
            LogCategory.SECURITY_EVENTS,
            "AuthAPI",
            data={"error": str(e), "username": user_data.username}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        get_logger().error(
            f"User registration error: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AuthAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLogin,
    request: Request
) -> TokenResponse:
    """
    Authenticate user and create session.

    Validates user credentials, creates authentication tokens,
    and establishes a secure session.

    Args:
        login_data: Login credentials
        request: HTTP request for IP and user agent tracking

    Returns:
        Authentication tokens and user information

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Get client information for security
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")

        # Authenticate user
        token_response = await auth_service.authenticate_user(
            login_data, client_ip, user_agent
        )

        get_logger().info(
            f"User logged in successfully: {token_response.user.username}",
            LogCategory.USER_INTERACTION,
            "AuthAPI",
            data={
                "user_id": token_response.user.id,
                "username": token_response.user.username,
                "ip_address": client_ip
            }
        )

        return token_response

    except ValueError as e:
        get_logger().warn(
            f"Login failed: {str(e)}",
            LogCategory.SECURITY_EVENTS,
            "AuthAPI",
            data={"error": str(e), "username_or_email": login_data.username_or_email}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        get_logger().error(
            f"Login error: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AuthAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    refresh_request: RefreshTokenRequest
) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Generates new authentication tokens using a valid refresh token,
    extending the user's session.

    Args:
        refresh_request: Refresh token request

    Returns:
        New authentication tokens

    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    try:
        token_response = await auth_service.refresh_token(refresh_request.refresh_token)

        get_logger().info(
            f"Token refreshed for user: {token_response.user.username}",
            LogCategory.USER_INTERACTION,
            "AuthAPI",
            data={"user_id": token_response.user.id}
        )

        return token_response

    except ValueError as e:
        get_logger().warn(
            f"Token refresh failed: {str(e)}",
            LogCategory.SECURITY_EVENTS,
            "AuthAPI",
            data={"error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        get_logger().error(
            f"Token refresh error: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AuthAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """
    Logout user and invalidate session.

    Deactivates the user's current session and invalidates tokens.

    Args:
        credentials: Bearer token credentials

    Returns:
        Logout confirmation
    """
    try:
        if credentials:
            success = await auth_service.logout_user(credentials.credentials)
            if success:
                get_logger().info(
                    "User logged out successfully",
                    LogCategory.USER_INTERACTION,
                    "AuthAPI"
                )

        return {"message": "Logged out successfully"}

    except Exception as e:
        get_logger().error(
            f"Logout error: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AuthAPI",
            error=e
        )
        # Don't fail logout even if there's an error
        return {"message": "Logged out"}


@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Get current user profile information.

    Returns the authenticated user's profile data.

    Args:
        current_user: Current authenticated user

    Returns:
        User profile information
    """
    return current_user


@router.get("/verify-token")
async def verify_token(
    current_user: UserResponse = Depends(get_current_user)
) -> dict:
    """
    Verify if the current token is valid.

    Checks token validity and returns user information.

    Args:
        current_user: Current authenticated user

    Returns:
        Token verification result
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "expires_at": datetime.utcnow().isoformat()
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserResponse:
    """
    Dependency to get current authenticated user.

    Args:
        credentials: Bearer token credentials

    Returns:
        Current user information

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await auth_service.get_current_user(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserResponse]:
    """
    Dependency to get current user if authenticated (optional).

    Args:
        credentials: Bearer token credentials

    Returns:
        Current user information or None
    """
    if not credentials:
        return None

    return await auth_service.get_current_user(credentials.credentials)






