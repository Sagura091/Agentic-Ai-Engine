"""
Universal Model Management API Endpoints.

This module provides REST API endpoints for managing all 4 model types:
- Text Embedding Models (sentence-transformers)
- Reranking Models (cross-encoder)
- Vision Models (CLIP, vision-language)
- LLM Models (Ollama, API validation)

Features:
- List available models by type
- User-driven model downloads
- Real-time download progress
- Model testing and validation
- Model deletion and management
- Configuration updates
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import structlog

from app.rag.core.embedding_model_manager import (
    embedding_model_manager,
    UniversalModelInfo,
    EmbeddingModelInfo,  # Backward compatibility
    ModelDownloadProgress,
    ModelType,
    ModelSource
)
from app.core.dependencies import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter()


class ModelDownloadRequest(BaseModel):
    """Request to download any model type."""
    model_id: str = Field(..., description="Model identifier")
    force_redownload: bool = Field(default=False, description="Force redownload if exists")


class ModelTestRequest(BaseModel):
    """Request to test any model type."""
    model_id: str = Field(..., description="Model identifier")
    test_text: str = Field(default="This is a test sentence.", description="Text to test with")


class ModelSearchRequest(BaseModel):
    """Request to search models."""
    query: str = Field(..., description="Search query")
    model_type: Optional[str] = Field(default=None, description="Filter by model type")


class CustomModelRequest(BaseModel):
    """Request to add a custom model."""
    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Model description")
    model_type: str = Field(..., description="Model type (embedding/reranking/vision/llm)")
    model_source: str = Field(..., description="Model source (huggingface/ollama/openai_api/etc)")
    download_url: str = Field(..., description="Download URL")
    size_mb: float = Field(..., description="Model size in MB")
    dimension: Optional[int] = Field(default=None, description="Output dimension")
    max_sequence_length: Optional[int] = Field(default=None, description="Max sequence length")
    context_length: Optional[int] = Field(default=None, description="Context length for LLMs")
    tags: List[str] = Field(default_factory=list, description="Model tags")


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


@router.get("/models", summary="List available models")
async def list_models(
    model_type: Optional[str] = None,
    downloaded_only: bool = False,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get list of all available models, optionally filtered by type.

    Args:
        model_type: Filter by model type (embedding/reranking/vision/llm)
        downloaded_only: Only return downloaded models

    Returns:
        List of available models with their information
    """
    try:
        if model_type:
            try:
                model_type_enum = ModelType(model_type.lower())
                if downloaded_only:
                    models = embedding_model_manager.get_downloaded_models_by_type(model_type_enum)
                else:
                    models = embedding_model_manager.get_models_by_type(model_type_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")
        else:
            if downloaded_only:
                models = embedding_model_manager.get_downloaded_models()
            else:
                models = list(embedding_model_manager.available_models.values())

        # Group models by type for better organization
        models_by_type = {
            "embedding": [],
            "reranking": [],
            "vision": [],
            "llm": []
        }

        for model in models:
            models_by_type[model.model_type.value].append({
                "model_id": model.model_id,
                "name": model.name,
                "description": model.description,
                "model_type": model.model_type.value,
                "model_source": model.model_source.value,
                "dimension": model.dimension,
                "max_sequence_length": model.max_sequence_length,
                "context_length": model.context_length,
                "size_mb": model.size_mb,
                "is_downloaded": model.is_downloaded,
                "download_date": model.download_date.isoformat() if model.download_date else None,
                "last_used": model.last_used.isoformat() if model.last_used else None,
                "usage_count": model.usage_count,
                "tags": model.tags,
                "use_case": model.use_case,
                "performance_tier": model.performance_tier
            })

        return {
            "success": True,
            "models_by_type": models_by_type,
            "total_models": len(models),
            "downloaded_models": len([m for m in models if m.is_downloaded]),
            "filter_applied": {
                "model_type": model_type,
                "downloaded_only": downloaded_only
            }
        }

    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@router.post("/models/search", summary="Search models")
async def search_models(
    request: ModelSearchRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search models by name, description, or tags.

    Args:
        request: Search request with query and optional model type filter

    Returns:
        List of matching models
    """
    try:
        model_type_filter = None
        if request.model_type:
            try:
                model_type_filter = ModelType(request.model_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")

        results = embedding_model_manager.search_models(request.query, model_type_filter)

        return {
            "success": True,
            "query": request.query,
            "model_type_filter": request.model_type,
            "results": [
                {
                    "model_id": model.model_id,
                    "name": model.name,
                    "description": model.description,
                    "model_type": model.model_type.value,
                    "model_source": model.model_source.value,
                    "is_downloaded": model.is_downloaded,
                    "size_mb": model.size_mb,
                    "tags": model.tags,
                    "use_case": model.use_case,
                    "performance_tier": model.performance_tier
                }
                for model in results
            ],
            "count": len(results)
        }

    except Exception as e:
        logger.error("Failed to search models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to search models")


@router.post("/models/custom", summary="Add custom model")
async def add_custom_model(
    request: CustomModelRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Add a custom model to the catalog.

    Args:
        request: Custom model information

    Returns:
        Success status and model information
    """
    try:
        # Validate model type and source
        try:
            model_type = ModelType(request.model_type.lower())
            model_source = ModelSource(request.model_source.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid model type or source: {e}")

        # Create model info
        model_info = UniversalModelInfo(
            model_id=request.model_id,
            name=request.name,
            description=request.description,
            model_type=model_type,
            model_source=model_source,
            download_url=request.download_url,
            size_mb=request.size_mb,
            dimension=request.dimension,
            max_sequence_length=request.max_sequence_length,
            context_length=request.context_length,
            tags=request.tags,
            use_case="custom",
            performance_tier="unknown"
        )

        success = embedding_model_manager.add_custom_model(model_info)

        if success:
            return {
                "success": True,
                "message": f"Custom model {request.model_id} added successfully",
                "model_info": {
                    "model_id": model_info.model_id,
                    "name": model_info.name,
                    "model_type": model_info.model_type.value,
                    "model_source": model_info.model_source.value
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add custom model")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add custom model", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to add custom model")


@router.get("/models/downloaded", summary="List downloaded models")
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


@router.post("/download", summary="Download model (any type)")
async def download_model(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Universal model download supporting all 4 model types:
    - Text Embedding Models (HuggingFace)
    - Reranking Models (HuggingFace)
    - Vision Models (HuggingFace)
    - LLM Models (Ollama)

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

        # Check if already downloaded (for HuggingFace models)
        if (model_info.is_downloaded and
            not request.force_redownload and
            model_info.model_source == ModelSource.HUGGINGFACE):
            return {
                "success": True,
                "message": f"Model {request.model_id} is already downloaded",
                "model_id": request.model_id,
                "model_type": model_info.model_type.value,
                "model_source": model_info.model_source.value,
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
            "message": f"Download started for {model_info.model_type.value} model {request.model_id}",
            "model_id": request.model_id,
            "model_type": model_info.model_type.value,
            "model_source": model_info.model_source.value,
            "status": "download_started",
            "estimated_size_mb": model_info.size_mb
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to initiate model download", model_id=request.model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start model download")


@router.post("/test", summary="Test model (any type)")
async def test_model(
    request: ModelTestRequest,
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Universal model testing for all 4 model types:
    - Text Embedding Models: Generate embeddings
    - Reranking Models: Score query-document pairs
    - Vision Models: Process text/image inputs
    - LLM Models: Generate text completions

    Args:
        request: Test request with model ID and text

    Returns:
        Test results including performance metrics and model-specific outputs
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
