"""
Embedding Models API Endpoints.

This module provides REST API endpoints for managing embedding models:
- List available models
- Download models from Hugging Face
- Get download progress
- Test models
- Delete models
- Update configuration
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import structlog

from app.rag.core.embedding_model_manager import (
    embedding_model_manager,
    EmbeddingModelInfo,
    ModelDownloadProgress
)
from app.core.dependencies import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter()


class ModelDownloadRequest(BaseModel):
    """Request to download an embedding model."""
    model_id: str = Field(..., description="Model identifier")
    force_redownload: bool = Field(default=False, description="Force redownload if exists")


class ModelTestRequest(BaseModel):
    """Request to test an embedding model."""
    model_id: str = Field(..., description="Model identifier")
    test_text: str = Field(default="This is a test sentence.", description="Text to test with")


class EmbeddingConfigUpdateRequest(BaseModel):
    """Request to update embedding configuration."""
    current_model: str = Field(..., description="Current active model")
    batch_size: int = Field(default=32, ge=1, le=128, description="Batch size for processing")
    max_length: int = Field(default=512, ge=128, le=2048, description="Maximum sequence length")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    cache_embeddings: bool = Field(default=True, description="Whether to cache embeddings")
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""
    url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    key: str = Field(..., description="OpenAI API key")


class OllamaConfig(BaseModel):
    """Ollama configuration."""
    url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    key: str = Field(default="", description="Ollama API key (optional)")


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration."""
    url: str = Field(..., description="Azure OpenAI endpoint URL")
    key: str = Field(..., description="Azure OpenAI API key")
    version: str = Field(default="2023-05-15", description="Azure OpenAI API version")


class GlobalEmbeddingConfig(BaseModel):
    """Global embedding configuration."""
    embedding_engine: str = Field(default="", description="Embedding engine: '', 'ollama', 'openai', 'azure_openai'")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    embedding_batch_size: int = Field(default=32, ge=1, le=100, description="Embedding batch size")
    openai_config: Optional[OpenAIConfig] = Field(default=None, description="OpenAI configuration")
    ollama_config: Optional[OllamaConfig] = Field(default=None, description="Ollama configuration")
    azure_openai_config: Optional[AzureOpenAIConfig] = Field(default=None, description="Azure OpenAI configuration")


class EmbeddingTestRequest(BaseModel):
    """Request to test embedding connection."""
    embedding_engine: str = Field(..., description="Embedding engine to test")
    embedding_model: str = Field(..., description="Embedding model to test")
    openai_config: Optional[OpenAIConfig] = Field(default=None, description="OpenAI configuration")
    ollama_config: Optional[OllamaConfig] = Field(default=None, description="Ollama configuration")
    azure_openai_config: Optional[AzureOpenAIConfig] = Field(default=None, description="Azure OpenAI configuration")


@router.get("/models", summary="List available embedding models")
async def list_embedding_models(
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get list of all available embedding models.
    
    Returns:
        List of available embedding models with their information
    """
    try:
        models = embedding_model_manager.get_available_models()
        
        return {
            "success": True,
            "models": [model.dict() for model in models],
            "total_count": len(models),
            "downloaded_count": len([m for m in models if m.is_downloaded])
        }
        
    except Exception as e:
        logger.error("Failed to list embedding models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve embedding models")


@router.get("/models/downloaded", summary="List downloaded embedding models")
async def list_downloaded_models(
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get list of downloaded embedding models.
    
    Returns:
        List of downloaded embedding models
    """
    try:
        models = embedding_model_manager.get_downloaded_models()
        
        return {
            "success": True,
            "models": [model.dict() for model in models],
            "count": len(models)
        }
        
    except Exception as e:
        logger.error("Failed to list downloaded models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve downloaded models")


@router.get("/models/{model_id}/status", summary="Get model status")
async def get_model_status(
    model_id: str,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get status information for a specific model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model status and information
    """
    try:
        # URL decode model_id (replace underscores with slashes)
        decoded_model_id = model_id.replace("_", "/")
        
        model_info = embedding_model_manager.get_model_info(decoded_model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {decoded_model_id} not found")
        
        # Get download progress if available
        progress = embedding_model_manager.get_download_progress(decoded_model_id)
        
        return {
            "success": True,
            "model_info": model_info.dict(),
            "download_progress": progress.dict() if progress else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model status", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model status")


@router.post("/download", summary="Download embedding model")
async def download_embedding_model(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Download an embedding model from Hugging Face.
    
    Args:
        request: Download request with model ID
        background_tasks: FastAPI background tasks
        
    Returns:
        Download initiation response
    """
    try:
        model_info = embedding_model_manager.get_model_info(request.model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Check if already downloaded
        if model_info.is_downloaded and not request.force_redownload:
            return {
                "success": True,
                "message": f"Model {request.model_id} is already downloaded",
                "model_id": request.model_id,
                "status": "already_downloaded"
            }
        
        # Start download in background
        background_tasks.add_task(
            embedding_model_manager.download_model,
            request.model_id,
            request.force_redownload
        )
        
        logger.info("Model download initiated", model_id=request.model_id)
        
        return {
            "success": True,
            "message": f"Download started for model {request.model_id}",
            "model_id": request.model_id,
            "status": "download_started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to initiate model download", model_id=request.model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start model download")


@router.post("/test", summary="Test embedding model")
async def test_embedding_model(
    request: ModelTestRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Test an embedding model with sample text.
    
    Args:
        request: Test request with model ID and text
        
    Returns:
        Test results including performance metrics
    """
    try:
        result = await embedding_model_manager.test_model(request.model_id, request.test_text)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Model test failed"))
        
        logger.info("Model test completed", model_id=request.model_id)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model test failed", model_id=request.model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to test model")


@router.delete("/models/{model_id}", summary="Delete embedding model")
async def delete_embedding_model(
    model_id: str,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a downloaded embedding model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Deletion result
    """
    try:
        # URL decode model_id (replace underscores with slashes)
        decoded_model_id = model_id.replace("_", "/")
        
        success = embedding_model_manager.delete_model(decoded_model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {decoded_model_id} not found or not downloaded")
        
        logger.info("Model deleted", model_id=decoded_model_id)
        
        return {
            "success": True,
            "message": f"Model {decoded_model_id} deleted successfully",
            "model_id": decoded_model_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete model", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete model")


@router.get("/config", summary="Get embedding configuration")
async def get_embedding_config(
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current embedding configuration.
    
    Returns:
        Current embedding configuration
    """
    try:
        # Get current configuration from settings or default
        config = {
            "current_model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32,
            "max_length": 512,
            "normalize": True,
            "cache_embeddings": True,
            "device": "auto"
        }
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        logger.error("Failed to get embedding config", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve embedding configuration")


@router.post("/config", summary="Update embedding configuration")
async def update_embedding_config(
    request: EmbeddingConfigUpdateRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update embedding configuration.
    
    Args:
        request: Configuration update request
        
    Returns:
        Update result
    """
    try:
        # Validate that the model exists and is downloaded
        model_info = embedding_model_manager.get_model_info(request.current_model)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {request.current_model} not found")
        
        if not model_info.is_downloaded:
            raise HTTPException(status_code=400, detail=f"Model {request.current_model} is not downloaded")
        
        # Update configuration (in a real implementation, this would update settings)
        config = request.dict()
        
        logger.info("Embedding configuration updated", config=config)
        
        return {
            "success": True,
            "message": "Embedding configuration updated successfully",
            "config": config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update embedding config", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update embedding configuration")


@router.get("/stats", summary="Get embedding system statistics")
async def get_embedding_stats(
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get embedding system statistics.
    
    Returns:
        System statistics including model usage and performance
    """
    try:
        models = embedding_model_manager.get_available_models()
        downloaded_models = embedding_model_manager.get_downloaded_models()
        
        total_size_mb = sum(model.size_mb for model in downloaded_models)
        total_usage = sum(model.usage_count for model in downloaded_models)
        
        stats = {
            "total_models": len(models),
            "downloaded_models": len(downloaded_models),
            "total_size_mb": round(total_size_mb, 2),
            "total_usage_count": total_usage,
            "most_used_model": None,
            "recently_used_models": []
        }
        
        # Find most used model
        if downloaded_models:
            most_used = max(downloaded_models, key=lambda m: m.usage_count)
            if most_used.usage_count > 0:
                stats["most_used_model"] = {
                    "model_id": most_used.model_id,
                    "usage_count": most_used.usage_count
                }
            
            # Get recently used models
            recent_models = sorted(
                [m for m in downloaded_models if m.last_used],
                key=lambda m: m.last_used,
                reverse=True
            )[:5]
            
            stats["recently_used_models"] = [
                {
                    "model_id": model.model_id,
                    "last_used": model.last_used.isoformat() if model.last_used else None,
                    "usage_count": model.usage_count
                }
                for model in recent_models
            ]
        
        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        logger.error("Failed to get embedding stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve embedding statistics")


# ============================================================================
# GLOBAL EMBEDDING CONFIGURATION ENDPOINTS
# ============================================================================

@router.get("/config")
async def get_global_embedding_config(current_user: dict = Depends(get_current_user)):
    """
    Get global embedding configuration.

    Returns the current global embedding configuration that applies to all
    knowledge bases in the system.
    """
    try:
        # Get configuration from embedding model manager or config store
        config = embedding_model_manager.get_global_config()

        return {
            "success": True,
            "embedding_engine": config.get("embedding_engine", ""),
            "embedding_model": config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            "embedding_batch_size": config.get("embedding_batch_size", 32),
            "openai_config": {
                "url": config.get("openai_url", "https://api.openai.com/v1"),
                "key": config.get("openai_key", "")
            },
            "ollama_config": {
                "url": config.get("ollama_url", "http://localhost:11434"),
                "key": config.get("ollama_key", "")
            },
            "azure_openai_config": {
                "url": config.get("azure_url", ""),
                "key": config.get("azure_key", ""),
                "version": config.get("azure_version", "2023-05-15")
            }
        }

    except Exception as e:
        logger.error("Failed to get global embedding config", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve embedding configuration")


@router.post("/config")
async def update_global_embedding_config(
    config: GlobalEmbeddingConfig,
    current_user: dict = Depends(get_current_user)
):
    """
    Update global embedding configuration.

    Updates the global embedding configuration that applies to all
    knowledge bases in the system.
    """
    try:
        # Validate configuration
        if config.embedding_engine == "openai" and not config.openai_config:
            raise HTTPException(status_code=400, detail="OpenAI configuration required")

        if config.embedding_engine == "azure_openai" and not config.azure_openai_config:
            raise HTTPException(status_code=400, detail="Azure OpenAI configuration required")

        if config.embedding_engine == "ollama" and not config.ollama_config:
            raise HTTPException(status_code=400, detail="Ollama configuration required")

        # Update configuration
        config_dict = {
            "embedding_engine": config.embedding_engine,
            "embedding_model": config.embedding_model,
            "embedding_batch_size": config.embedding_batch_size
        }

        if config.openai_config:
            config_dict.update({
                "openai_url": config.openai_config.url,
                "openai_key": config.openai_config.key
            })

        if config.ollama_config:
            config_dict.update({
                "ollama_url": config.ollama_config.url,
                "ollama_key": config.ollama_config.key
            })

        if config.azure_openai_config:
            config_dict.update({
                "azure_url": config.azure_openai_config.url,
                "azure_key": config.azure_openai_config.key,
                "azure_version": config.azure_openai_config.version
            })

        # Save configuration
        embedding_model_manager.update_global_config(config_dict)

        logger.info("Global embedding configuration updated",
                   engine=config.embedding_engine,
                   model=config.embedding_model)

        return {
            "success": True,
            "message": "Global embedding configuration updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update global embedding config", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update embedding configuration")


@router.post("/test")
async def test_embedding_connection(
    test_config: EmbeddingTestRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Test embedding connection with the provided configuration.

    Tests the connection to the specified embedding provider without
    saving the configuration.
    """
    try:
        # Test the connection based on the engine
        test_text = "This is a test sentence for embedding."

        if test_config.embedding_engine == "openai":
            if not test_config.openai_config:
                raise HTTPException(status_code=400, detail="OpenAI configuration required")

            # Test OpenAI connection
            success = await embedding_model_manager.test_openai_connection(
                url=test_config.openai_config.url,
                key=test_config.openai_config.key,
                model=test_config.embedding_model,
                test_text=test_text
            )

        elif test_config.embedding_engine == "azure_openai":
            if not test_config.azure_openai_config:
                raise HTTPException(status_code=400, detail="Azure OpenAI configuration required")

            # Test Azure OpenAI connection
            success = await embedding_model_manager.test_azure_connection(
                url=test_config.azure_openai_config.url,
                key=test_config.azure_openai_config.key,
                version=test_config.azure_openai_config.version,
                model=test_config.embedding_model,
                test_text=test_text
            )

        elif test_config.embedding_engine == "ollama":
            if not test_config.ollama_config:
                raise HTTPException(status_code=400, detail="Ollama configuration required")

            # Test Ollama connection
            success = await embedding_model_manager.test_ollama_connection(
                url=test_config.ollama_config.url,
                key=test_config.ollama_config.key,
                model=test_config.embedding_model,
                test_text=test_text
            )

        else:
            # Test default sentence transformers
            success = await embedding_model_manager.test_default_connection(
                model=test_config.embedding_model,
                test_text=test_text
            )

        if success:
            return {
                "success": True,
                "message": "Connection test successful"
            }
        else:
            raise HTTPException(status_code=400, detail="Connection test failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to test embedding connection", error=str(e))
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")
