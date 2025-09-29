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
