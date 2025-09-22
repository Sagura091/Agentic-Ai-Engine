"""
User API Key Management Endpoints.

This module provides REST API endpoints for users to manage their own
API keys for external providers (OpenAI, Anthropic, Google, Microsoft).
"""

from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auth import UserAPIKeyCreate, UserAPIKeyResponse, UserAPIKeyUpdate, UserResponse
from app.models.database.base import get_database_session
from app.api.v1.endpoints.auth import get_current_user
from app.services.enhanced_auth_service import enhanced_auth_service
from app.backend_logging.backend_logger import get_logger, LogCategory

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/user/api-keys", tags=["User API Keys"])


@router.post("/", response_model=UserAPIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: UserAPIKeyCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> UserAPIKeyResponse:
    """
    Create a new API key for external provider.
    
    Allows users to securely store their API keys for external providers
    like OpenAI, Anthropic, Google, and Microsoft.
    
    Args:
        key_data: API key creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created API key information (without the actual key)
        
    Raises:
        HTTPException: If API key creation fails
    """
    try:
        api_key = await enhanced_auth_service.create_user_api_key(
            user_id=current_user.id,
            key_data=key_data,
            db=db
        )
        
        get_logger().info(
            f"API key created for provider: {key_data.provider}",
            LogCategory.USER_MANAGEMENT,
            "UserAPIKeyAPI",
            data={
                "user_id": current_user.id,
                "provider": key_data.provider,
                "key_name": key_data.key_name,
                "is_default": key_data.is_default
            }
        )
        
        return api_key
        
    except ValueError as e:
        get_logger().warning(
            f"API key creation failed: {str(e)}",
            LogCategory.SECURITY,
            "UserAPIKeyAPI",
            data={"error": str(e), "user_id": current_user.id, "provider": key_data.provider}
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        get_logger().error(
            f"API key creation error: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "UserAPIKeyAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/", response_model=List[UserAPIKeyResponse])
async def list_api_keys(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
    provider: Optional[str] = None
) -> List[UserAPIKeyResponse]:
    """
    List user's API keys.
    
    Returns all API keys owned by the current user, optionally filtered by provider.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        provider: Optional provider filter
        
    Returns:
        List of user's API keys (without actual key values)
    """
    try:
        api_keys = await enhanced_auth_service.get_user_api_keys(
            user_id=current_user.id,
            provider=provider,
            db=db
        )
        
        get_logger().info(
            f"Listed {len(api_keys)} API keys for user",
            LogCategory.USER_MANAGEMENT,
            "UserAPIKeyAPI",
            data={
                "user_id": current_user.id,
                "provider_filter": provider,
                "key_count": len(api_keys)
            }
        )
        
        return api_keys
        
    except Exception as e:
        get_logger().error(
            f"Failed to list API keys: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "UserAPIKeyAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )


@router.get("/{key_id}", response_model=UserAPIKeyResponse)
async def get_api_key(
    key_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> UserAPIKeyResponse:
    """
    Get specific API key details.
    
    Returns API key information if it belongs to the current user.
    
    Args:
        key_id: API key ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        API key information (without actual key value)
        
    Raises:
        HTTPException: If API key not found or access denied
    """
    try:
        api_key = await enhanced_auth_service.get_user_api_key_by_id(
            user_id=current_user.id,
            key_id=key_id,
            db=db
        )
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or access denied"
            )
        
        return api_key
        
    except HTTPException:
        raise
    except Exception as e:
        get_logger().error(
            f"Failed to get API key {key_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "UserAPIKeyAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get API key"
        )


@router.put("/{key_id}", response_model=UserAPIKeyResponse)
async def update_api_key(
    key_id: str,
    key_data: UserAPIKeyUpdate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> UserAPIKeyResponse:
    """
    Update API key information.
    
    Updates API key metadata, name, or default status.
    
    Args:
        key_id: API key ID
        key_data: Update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated API key information
        
    Raises:
        HTTPException: If API key not found or update fails
    """
    try:
        api_key = await enhanced_auth_service.update_user_api_key(
            user_id=current_user.id,
            key_id=key_id,
            key_data=key_data,
            db=db
        )
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or access denied"
            )
        
        get_logger().info(
            f"API key updated: {key_id}",
            LogCategory.USER_MANAGEMENT,
            "UserAPIKeyAPI",
            data={
                "user_id": current_user.id,
                "key_id": key_id,
                "updated_fields": list(key_data.dict(exclude_unset=True).keys())
            }
        )
        
        return api_key
        
    except HTTPException:
        raise
    except Exception as e:
        get_logger().error(
            f"Failed to update API key {key_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "UserAPIKeyAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update API key"
        )


@router.delete("/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> dict:
    """
    Delete API key.
    
    Permanently removes the API key from the user's account.
    
    Args:
        key_id: API key ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Deletion confirmation
        
    Raises:
        HTTPException: If API key not found or deletion fails
    """
    try:
        success = await enhanced_auth_service.delete_user_api_key(
            user_id=current_user.id,
            key_id=key_id,
            db=db
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or access denied"
            )
        
        get_logger().info(
            f"API key deleted: {key_id}",
            LogCategory.USER_MANAGEMENT,
            "UserAPIKeyAPI",
            data={
                "user_id": current_user.id,
                "key_id": key_id
            }
        )
        
        return {"message": "API key deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        get_logger().error(
            f"Failed to delete API key {key_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "UserAPIKeyAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key"
        )


@router.post("/{key_id}/test")
async def test_api_key(
    key_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> dict:
    """
    Test API key validity.
    
    Tests if the API key is valid by making a simple API call to the provider.
    
    Args:
        key_id: API key ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Test result
        
    Raises:
        HTTPException: If API key not found or test fails
    """
    try:
        result = await enhanced_auth_service.test_user_api_key(
            user_id=current_user.id,
            key_id=key_id,
            db=db
        )
        
        get_logger().info(
            f"API key tested: {key_id}",
            LogCategory.USER_MANAGEMENT,
            "UserAPIKeyAPI",
            data={
                "user_id": current_user.id,
                "key_id": key_id,
                "test_result": result.get("valid", False)
            }
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        get_logger().error(
            f"Failed to test API key {key_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "UserAPIKeyAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test API key"
        )
