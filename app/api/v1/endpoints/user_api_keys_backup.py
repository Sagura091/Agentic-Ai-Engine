"""
OPTIMIZED User API Key Management Endpoints.

This module provides REST API endpoints for users to manage their own
API keys for external providers (OpenAI, Anthropic, Google, Microsoft).

OPTIMIZED: API keys are stored directly in users.api_keys JSON field.
"""

from typing import Dict, Any
import structlog
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from app.models.auth import UserDB, APIKeyUpdate, APIKeyDelete, APIKeysResponse
from app.models.database.base import get_database_session
from app.api.v1.endpoints.auth import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api-keys", tags=["user-api-keys"])


@router.get("/", response_model=APIKeysResponse)
async def get_user_api_keys(
    current_user: UserDB = Depends(get_current_user),
    session: AsyncSession = Depends(get_database_session)
):
    """Get list of providers with stored API keys (without exposing actual keys)."""
    try:
        providers = current_user.get_available_providers()
        return APIKeysResponse(providers=providers)
    except Exception as e:
        logger.error("Failed to get user API keys", user_id=str(current_user.id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve API keys")


@router.put("/", response_model=Dict[str, str])
async def update_user_api_key(
    api_key_data: APIKeyUpdate,
    current_user: UserDB = Depends(get_current_user),
    session: AsyncSession = Depends(get_database_session)
):
    """Add or update an API key for a specific provider."""
    try:
        # Validate provider
        valid_providers = ['openai', 'anthropic', 'google', 'microsoft', 'gemini', 'azure']
        if api_key_data.provider.lower() not in valid_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider. Supported providers: {', '.join(valid_providers)}"
            )

        # Update API key
        current_user.set_api_key(api_key_data.provider.lower(), api_key_data.api_key)

        # Save to database
        await session.execute(
            update(UserDB)
            .where(UserDB.id == current_user.id)
            .values(api_keys=current_user.api_keys)
        )
        await session.commit()

        logger.info("API key updated", user_id=str(current_user.id), provider=api_key_data.provider)
        return {"message": f"API key for {api_key_data.provider} updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update API key", user_id=str(current_user.id), error=str(e))
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to update API key")


@router.delete("/", response_model=Dict[str, str])
async def delete_user_api_key(
    api_key_data: APIKeyDelete,
    current_user: UserDB = Depends(get_current_user),
    session: AsyncSession = Depends(get_database_session)
):
    """Remove an API key for a specific provider."""
    try:
        # Remove API key
        if not current_user.remove_api_key(api_key_data.provider.lower()):
            raise HTTPException(
                status_code=404,
                detail=f"No API key found for provider: {api_key_data.provider}"
            )

        # Save to database
        await session.execute(
            update(UserDB)
            .where(UserDB.id == current_user.id)
            .values(api_keys=current_user.api_keys)
        )
        await session.commit()

        logger.info("API key deleted", user_id=str(current_user.id), provider=api_key_data.provider)
        return {"message": f"API key for {api_key_data.provider} removed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete API key", user_id=str(current_user.id), error=str(e))
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete API key")


@router.get("/providers", response_model=Dict[str, Any])
async def get_supported_providers():
    """Get list of supported API providers."""
    providers = {
        "supported_providers": [
            {
                "name": "openai",
                "display_name": "OpenAI",
                "description": "GPT-4, GPT-3.5, DALL-E, Whisper",
                "key_format": "sk-..."
            },
            {
                "name": "anthropic",
                "display_name": "Anthropic",
                "description": "Claude 3.5 Sonnet, Claude 3 Haiku",
                "key_format": "sk-ant-..."
            },
            {
                "name": "google",
                "display_name": "Google AI",
                "description": "Gemini Pro, Gemini Flash",
                "key_format": "AIza..."
            },
            {
                "name": "microsoft",
                "display_name": "Microsoft Azure",
                "description": "Azure OpenAI Service",
                "key_format": "..."
            }
        ],
        "default_provider": {
            "name": "ollama",
            "display_name": "Ollama (Local)",
            "description": "Local LLM inference - no API key needed",
            "key_format": "none"
        }
    }
    return providers


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
