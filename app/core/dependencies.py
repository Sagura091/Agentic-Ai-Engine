"""
Dependency injection for the Agentic AI Microservice.

This module provides dependency injection functions for FastAPI endpoints,
including database sessions, authentication, and service dependencies.
"""

from typing import AsyncGenerator, Optional

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Security scheme for JWT authentication
security = HTTPBearer(auto_error=False)


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        Database session
    """
    # Import here to avoid circular imports
    from app.models.database.base import get_database_session as get_db_session
    
    async with get_db_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error("Database session error", error=str(e))
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings = Depends(get_settings)
) -> Optional[dict]:
    """
    Dependency to get current authenticated user.
    
    Args:
        credentials: JWT credentials from request
        settings: Application settings
        
    Returns:
        User information if authenticated, None otherwise
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        return None
    
    try:
        # Import here to avoid circular imports
        from app.core.security import decode_access_token
        
        payload = decode_access_token(credentials.credentials, settings.SECRET_KEY)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Return user info (in a real implementation, you'd fetch from database)
        return {
            "id": user_id,
            "username": payload.get("username"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
        }
        
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_authentication(
    current_user: Optional[dict] = Depends(get_current_user)
) -> dict:
    """
    Dependency that requires authentication.
    
    Args:
        current_user: Current user from authentication
        
    Returns:
        User information
        
    Raises:
        HTTPException: If not authenticated
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return current_user


async def require_admin(
    current_user: dict = Depends(require_authentication)
) -> dict:
    """
    Dependency that requires admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
        
    Raises:
        HTTPException: If not admin
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    
    return current_user


async def get_orchestrator():
    """
    Dependency to get the agent orchestrator.
    
    Returns:
        Agent orchestrator instance
    """
    from app.orchestration.orchestrator import orchestrator
    return orchestrator


async def get_websocket_manager():
    """
    Dependency to get the WebSocket manager.
    
    Returns:
        WebSocket manager instance
    """
    from app.api.websocket.manager import websocket_manager
    return websocket_manager


async def get_monitoring_service():
    """
    Dependency to get the monitoring service.
    
    Returns:
        Monitoring service instance
    """
    from app.services.monitoring_service import monitoring_service
    return monitoring_service
