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


@router.post("/ollama/pull", summary="Download Ollama model")
async def pull_ollama_model(
    model_name: str,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Download/pull a model from Ollama."""
    try:
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()

        # Get Ollama provider
        ollama_provider = await llm_service.get_provider("ollama")
        if not ollama_provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ollama provider not available"
            )

        # Pull the model
        success = await ollama_provider.pull_model(model_name)

        if success:
            return {
                "success": True,
                "message": f"Model {model_name} downloaded successfully",
                "model_name": model_name
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download model {model_name}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pull Ollama model", model=model_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download model: {str(e)}"
        )


@router.get("/ollama/available", summary="Get available Ollama models for download")
async def get_available_ollama_models(current_user: Optional[str] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get list of available Ollama models that can be downloaded."""
    try:
        # Popular Ollama models categorized by use case
        available_models = {
            "recommended": [
                {
                    "name": "llama3.2:latest",
                    "size": "2.0GB",
                    "description": "Latest Llama 3.2 model with excellent tool calling support",
                    "capabilities": ["text", "tools", "conversation"],
                    "recommended": True
                },
                {
                    "name": "llama3.1:8b",
                    "size": "4.7GB",
                    "description": "Llama 3.1 8B with superior tool calling capabilities",
                    "capabilities": ["text", "tools", "conversation"],
                    "recommended": True
                },
                {
                    "name": "qwen2.5:latest",
                    "size": "4.4GB",
                    "description": "Qwen 2.5 with strong reasoning and tool support",
                    "capabilities": ["text", "tools", "conversation", "reasoning"],
                    "recommended": True
                }
            ],
            "code": [
                {
                    "name": "codellama:latest",
                    "size": "3.8GB",
                    "description": "Code Llama for programming tasks",
                    "capabilities": ["code", "text"],
                    "recommended": False
                },
                {
                    "name": "deepseek-coder:latest",
                    "size": "3.7GB",
                    "description": "DeepSeek Coder for advanced programming",
                    "capabilities": ["code", "text"],
                    "recommended": False
                }
            ],
            "lightweight": [
                {
                    "name": "llama3.2:3b",
                    "size": "2.0GB",
                    "description": "Lightweight Llama 3.2 3B model",
                    "capabilities": ["text", "conversation"],
                    "recommended": False
                },
                {
                    "name": "phi3:latest",
                    "size": "2.3GB",
                    "description": "Microsoft Phi-3 lightweight model",
                    "capabilities": ["text", "conversation"],
                    "recommended": False
                }
            ],
            "specialized": [
                {
                    "name": "mistral:latest",
                    "size": "4.1GB",
                    "description": "Mistral 7B for general tasks",
                    "capabilities": ["text", "conversation"],
                    "recommended": False
                },
                {
                    "name": "gemma2:latest",
                    "size": "5.4GB",
                    "description": "Google Gemma 2 model",
                    "capabilities": ["text", "conversation"],
                    "recommended": False
                }
            ]
        }

        return {
            "success": True,
            "available_models": available_models
        }

    except Exception as e:
        logger.error("Failed to get available Ollama models", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available models: {str(e)}"
        )


@router.get("/provider-templates", summary="Get LLM provider configuration templates")
async def get_provider_templates(current_user: Optional[str] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get pre-configured templates for different LLM provider setups."""
    try:
        templates = {
            "local_development": {
                "name": "Local Development",
                "description": "Optimized for local development with Ollama",
                "settings": {
                    "enable_ollama": True,
                    "enable_openai": False,
                    "enable_anthropic": False,
                    "enable_google": False,
                    "ollama_base_url": "http://localhost:11434",
                    "ollama_timeout": 120,
                    "ollama_max_concurrent_requests": 5,
                    "default_provider": "ollama",
                    "default_model": "llama3.2:latest"
                }
            },
            "production_hybrid": {
                "name": "Production Hybrid",
                "description": "Balanced setup with local and cloud providers",
                "settings": {
                    "enable_ollama": True,
                    "enable_openai": True,
                    "enable_anthropic": False,
                    "enable_google": False,
                    "ollama_base_url": "http://localhost:11434",
                    "ollama_timeout": 60,
                    "ollama_max_concurrent_requests": 10,
                    "openai_timeout": 30,
                    "openai_max_retries": 3,
                    "default_provider": "openai",
                    "fallback_provider": "ollama"
                }
            },
            "cloud_only": {
                "name": "Cloud Only",
                "description": "Cloud-based providers for maximum performance",
                "settings": {
                    "enable_ollama": False,
                    "enable_openai": True,
                    "enable_anthropic": True,
                    "enable_google": True,
                    "openai_timeout": 30,
                    "anthropic_timeout": 30,
                    "google_timeout": 30,
                    "default_provider": "openai",
                    "fallback_provider": "anthropic"
                }
            },
            "high_performance": {
                "name": "High Performance",
                "description": "Optimized for high-throughput applications",
                "settings": {
                    "enable_ollama": True,
                    "enable_openai": True,
                    "ollama_max_concurrent_requests": 20,
                    "ollama_connection_pool_size": 10,
                    "openai_max_retries": 5,
                    "request_timeout": 120,
                    "enable_load_balancing": True,
                    "enable_failover": True
                }
            }
        }

        return {
            "success": True,
            "templates": templates
        }

    except Exception as e:
        logger.error("Failed to get provider templates", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider templates: {str(e)}"
        )
