"""
LLM Provider Management API Endpoints.

This module provides REST API endpoints for managing LLM providers,
including provider registration, model listing, and connection testing.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
import structlog

from app.services.llm_service import get_llm_service, initialize_llm_service
from app.core.dependencies import get_current_user
from typing import Optional

logger = structlog.get_logger(__name__)

router = APIRouter()


class ProviderCredentialsRequest(BaseModel):
    """Request model for provider credentials."""
    provider: str = Field(..., description="Provider name (ollama, openai, anthropic, google)")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Custom base URL")
    organization: Optional[str] = Field(None, description="Organization ID (OpenAI)")
    project: Optional[str] = Field(None, description="Project ID (OpenAI)")
    additional_headers: Optional[Dict[str, str]] = Field(None, description="Additional headers")


class ModelTestRequest(BaseModel):
    """Request model for testing a model."""
    provider: str = Field(..., description="Provider name")
    model_id: str = Field(..., description="Model ID to test")


class LLMConfigRequest(BaseModel):
    """Request model for LLM configuration."""
    provider: str = Field(..., description="Provider name")
    model_id: str = Field(..., description="Model ID")
    model_name: Optional[str] = Field(None, description="Human-readable model name")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(None, gt=0, description="Top-k sampling")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


@router.get("/providers", summary="Get available LLM providers")
async def get_providers(current_user: Optional[str] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get list of available LLM providers."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        providers = await llm_service.get_available_providers()
        provider_info = await llm_service.get_provider_info()
        
        return {
            "success": True,
            "providers": providers,
            "info": provider_info
        }
        
    except Exception as e:
        logger.error("Failed to get providers", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get providers: {str(e)}"
        )


@router.get("/models", summary="Get all available models")
async def get_all_models(current_user: Optional[str] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get all available models from all providers."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        models = await llm_service.get_all_models()
        
        return {
            "success": True,
            "models": models
        }
        
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )


@router.get("/models/{provider}", summary="Get models by provider")
async def get_models_by_provider(
    provider: str,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get models from a specific provider."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        models = await llm_service.get_models_by_provider(provider)
        
        return {
            "success": True,
            "provider": provider,
            "models": models
        }
        
    except Exception as e:
        logger.error("Failed to get models by provider", provider=provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models for provider {provider}: {str(e)}"
        )


@router.post("/providers/register", summary="Register provider credentials")
async def register_provider(
    request: ProviderCredentialsRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Register or update provider credentials."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        credentials = {
            "api_key": request.api_key,
            "base_url": request.base_url,
            "organization": request.organization,
            "project": request.project,
            "additional_headers": request.additional_headers
        }
        
        success = await llm_service.register_provider_credentials(request.provider, credentials)
        
        if success:
            return {
                "success": True,
                "message": f"Provider {request.provider} registered successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to register provider {request.provider}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register provider", provider=request.provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register provider: {str(e)}"
        )


@router.post("/test/model", summary="Test model availability")
async def test_model(
    request: ModelTestRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Test if a model is available from a provider."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        is_valid = await llm_service.validate_model_config(request.provider, request.model_id)
        
        return {
            "success": True,
            "provider": request.provider,
            "model_id": request.model_id,
            "is_available": is_valid
        }
        
    except Exception as e:
        logger.error("Failed to test model", provider=request.provider, model=request.model_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test model: {str(e)}"
        )


@router.get("/test/providers", summary="Test all provider connections")
async def test_all_providers(current_user: Optional[str] = Depends(get_current_user)) -> Dict[str, Any]:
    """Test connection to all registered providers."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        results = await llm_service.test_all_providers()
        
        return {
            "success": True,
            "test_results": results
        }
        
    except Exception as e:
        logger.error("Failed to test providers", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test providers: {str(e)}"
        )


@router.get("/test/providers/{provider}", summary="Test specific provider connection")
async def test_provider(
    provider: str,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Test connection to a specific provider."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        result = await llm_service.test_provider_connection(provider)
        
        return {
            "success": True,
            "test_result": result
        }
        
    except Exception as e:
        logger.error("Failed to test provider", provider=provider, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test provider {provider}: {str(e)}"
        )


@router.post("/test/config", summary="Test LLM configuration")
async def test_llm_config(
    request: LLMConfigRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Test an LLM configuration by creating an instance."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        config = request.dict()
        llm_instance = await llm_service.create_llm_instance(config)
        
        return {
            "success": True,
            "message": "LLM configuration is valid",
            "config": {
                "provider": request.provider,
                "model_id": request.model_id,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        }
        
    except Exception as e:
        request_dict = {}
        try:
            if hasattr(request, 'dict'):
                request_dict = request.dict()
            elif hasattr(request, '__dict__'):
                request_dict = request.__dict__
            else:
                request_dict = {"type": str(type(request))}
        except Exception:
            request_dict = {"error": "Could not serialize request"}

        logger.error("Failed to test LLM config", config=request_dict, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid LLM configuration: {str(e)}"
        )


@router.get("/default-config", summary="Get default LLM configuration")
async def get_default_config(current_user: Optional[str] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get the default LLM configuration."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()
        
        config = await llm_service.get_default_model_config()
        
        return {
            "success": True,
            "default_config": config
        }
        
    except Exception as e:
        logger.error("Failed to get default config", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get default config: {str(e)}"
        )
