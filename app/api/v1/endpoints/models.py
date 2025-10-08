"""
Models API endpoints.

This module provides model management functionality for the Agentic AI system.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    description: str
    provider: str
    model_type: str
    capabilities: List[str]
    parameters: Dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    """Model list response."""
    models: List[ModelInfo]
    total_count: int
    available_providers: List[str]


@router.get("", response_model=ModelListResponse)
async def list_models(
    provider: Optional[str] = Query(default=None, description="Filter by provider"),
    model_type: Optional[str] = Query(default=None, description="Filter by model type"),
    limit: int = Query(default=50, description="Maximum number of models to return")
) -> ModelListResponse:
    """
    List available models.
    
    Args:
        provider: Optional provider filter
        model_type: Optional model type filter
        limit: Maximum number of models to return
        
    Returns:
        List of available models
    """
    try:
        settings = get_settings()
        
        # Get models from Ollama if available
        ollama_models = []
        try:
            from app.http_client import HTTPClient, ClientConfig, ConnectionPoolConfig
            config = ClientConfig(
                timeout=10,
                verify_ssl=False,
                pool_config=ConnectionPoolConfig(max_per_host=3, keepalive_timeout=60)
            )
            async with HTTPClient(settings.OLLAMA_BASE_URL, config) as client:
                response = await client.get("/api/tags", stream=False)
                if response.status_code == 200:
                    ollama_data = response.json()
                    ollama_models = [
                        ModelInfo(
                            id=model["name"],
                            name=model["name"],
                            description=f"Ollama model: {model['name']}",
                            provider="ollama",
                            model_type="language_model",
                            capabilities=["text_generation", "conversation"],
                            parameters={
                                "size": model.get("size", 0),
                                "modified_at": model.get("modified_at", ""),
                                "digest": model.get("digest", "")
                            },
                            status="available",
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                        for model in ollama_data.get("models", [])
                    ]
        except Exception as e:
            logger.warning("Failed to fetch Ollama models", error=str(e))
        
        # Add default/built-in models
        default_models = [
            ModelInfo(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                description="OpenAI GPT-3.5 Turbo model",
                provider="openai",
                model_type="language_model",
                capabilities=["text_generation", "conversation", "function_calling"],
                parameters={
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "top_p": 1.0
                },
                status="available",
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ModelInfo(
                id="gpt-4",
                name="GPT-4",
                description="OpenAI GPT-4 model",
                provider="openai",
                model_type="language_model",
                capabilities=["text_generation", "conversation", "function_calling", "reasoning"],
                parameters={
                    "max_tokens": 8192,
                    "temperature": 0.7,
                    "top_p": 1.0
                },
                status="available",
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ModelInfo(
                id="claude-3-sonnet",
                name="Claude 3 Sonnet",
                description="Anthropic Claude 3 Sonnet model",
                provider="anthropic",
                model_type="language_model",
                capabilities=["text_generation", "conversation", "reasoning", "analysis"],
                parameters={
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                status="available",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        # Combine all models
        all_models = ollama_models + default_models
        
        # Apply filters
        if provider:
            all_models = [m for m in all_models if m.provider == provider]
        if model_type:
            all_models = [m for m in all_models if m.model_type == model_type]
        
        # Apply limit
        all_models = all_models[:limit]
        
        # Get available providers
        available_providers = list(set(m.provider for m in all_models))
        
        logger.info(
            "Models listed",
            total_count=len(all_models),
            provider=provider,
            model_type=model_type,
            available_providers=available_providers
        )
        
        return ModelListResponse(
            models=all_models,
            total_count=len(all_models),
            available_providers=available_providers
        )
        
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """
    Get specific model information.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model information
    """
    try:
        # Get all models and find the specific one
        models_response = await list_models()
        
        for model in models_response.models:
            if model.id == model_id:
                logger.info("Model info retrieved", model_id=model_id)
                return model
        
        logger.warning("Model not found", model_id=model_id)
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@router.get("/{model_id}/status")
async def get_model_status(model_id: str) -> Dict[str, Any]:
    """
    Get model status and availability.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model status information
    """
    try:
        # Check if model exists
        model = await get_model(model_id)
        
        # Check model availability based on provider
        status = "available"
        details = {}
        
        if model.provider == "ollama":
            try:
                settings = get_settings()
                from app.http_client import HTTPClient, ClientConfig, ConnectionPoolConfig
                config = ClientConfig(
                    timeout=10,
                    verify_ssl=False,
                    pool_config=ConnectionPoolConfig(max_per_host=3, keepalive_timeout=60)
                )
                async with HTTPClient(settings.OLLAMA_BASE_URL, config) as client:
                    response = await client.get("/api/tags", stream=False)
                    if response.status_code == 200:
                        ollama_models = response.json().get("models", [])
                        if not any(m["name"] == model_id for m in ollama_models):
                            status = "unavailable"
                            details["reason"] = "Model not found in Ollama"
                    else:
                        status = "unknown"
                        details["reason"] = "Cannot connect to Ollama"
            except Exception as e:
                status = "unknown"
                details["reason"] = f"Error checking Ollama: {str(e)}"
        
        logger.info("Model status checked", model_id=model_id, status=status)
        
        return {
            "model_id": model_id,
            "status": status,
            "provider": model.provider,
            "last_checked": datetime.now().isoformat(),
            "details": details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model status", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model status")


@router.post("/{model_id}/test")
async def test_model(model_id: str) -> Dict[str, Any]:
    """
    Test model availability and basic functionality.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Test results
    """
    try:
        # Get model info
        model = await get_model(model_id)
        
        # Perform basic test based on provider
        test_result = {
            "model_id": model_id,
            "provider": model.provider,
            "test_timestamp": datetime.now().isoformat(),
            "status": "success",
            "response_time": 0.0,
            "details": {}
        }
        
        if model.provider == "ollama":
            try:
                settings = get_settings()
                from app.http_client import HTTPClient, ClientConfig, ConnectionPoolConfig
                import time

                start_time = time.time()
                config = ClientConfig(
                    timeout=30,
                    verify_ssl=False,
                    pool_config=ConnectionPoolConfig(max_per_host=3, keepalive_timeout=60)
                )
                async with HTTPClient(settings.OLLAMA_BASE_URL, config) as client:
                    test_payload = {
                        "model": model_id,
                        "prompt": "Hello, this is a test. Please respond with 'Test successful'.",
                        "stream": False
                    }
                    response = await client.post(
                        "/api/generate",
                        body=test_payload
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        test_result["response_time"] = response_time
                        test_result["details"]["response"] = result.get("response", "")
                    else:
                        test_result["status"] = "failed"
                        test_result["details"]["error"] = f"HTTP {response.status_code}"
                        
            except Exception as e:
                test_result["status"] = "failed"
                test_result["details"]["error"] = str(e)
        else:
            # For non-Ollama models, just return basic availability
            test_result["details"]["note"] = "Basic availability check only"
        
        logger.info("Model tested", model_id=model_id, status=test_result["status"])
        
        return test_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to test model", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to test model")
