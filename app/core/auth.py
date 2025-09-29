"""
Authentication and authorization utilities.

This module provides authentication and authorization functionality
for the agentic AI system.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models.auth import UserDB

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserDB:
    """
    Get the current authenticated user.
    
    For now, this is a simple implementation that returns a default user.
    In production, this would validate JWT tokens and return the actual user.
    """
    # For development/testing, return a default user
    # In production, this would validate the JWT token and return the actual user
    if credentials is None:
        # Allow unauthenticated access for development
        from uuid import uuid4
        return UserDB(
            id=uuid4(),
            username="default_user",
            email="default@example.com",
            name="Default User",
            hashed_password="mock_hash",
            is_active=True,
            user_group="admin"  # Give admin access for first-time setup
        )
    
    # In production, validate the token here
    token = credentials.credentials
    
    # For now, accept any token and return default user
    from uuid import uuid4
    return UserDB(
        id=uuid4(),
        username="authenticated_user",
        email="user@example.com",
        name="Authenticated User",
        hashed_password="mock_hash",
        is_active=True,
        user_group="admin"  # Give admin access for first-time setup
    )


async def get_current_active_user(current_user: UserDB = Depends(get_current_user)) -> UserDB:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def verify_token(token: str) -> Optional[dict]:
    """
    Verify a JWT token and return the payload.
    
    This is a placeholder implementation.
    In production, this would use proper JWT validation.
    """
    # Placeholder implementation
    return {"sub": "user", "username": "default_user"}


def create_access_token(data: dict) -> str:
    """
    Create a JWT access token.
    
    This is a placeholder implementation.
    In production, this would use proper JWT creation.
    """
    # Placeholder implementation
    return "fake_jwt_token"
