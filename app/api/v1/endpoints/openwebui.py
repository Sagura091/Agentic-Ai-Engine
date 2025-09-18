"""
OpenWebUI-compatible API endpoints.

This module provides OpenAI-compatible endpoints that integrate with OpenWebUI
through the Pipelines framework, exposing our LangGraph agents as models.
"""

import asyncio
import time
from typing import Dict, Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.integrations.openwebui.pipeline import (
    openwebui_pipeline,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse
)

# Import new backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogLevel, LogCategory, PerformanceMetrics, APIMetrics
from app.backend_logging.context import CorrelationContext
from app.core.dependencies import get_monitoring_service

logger = structlog.get_logger(__name__)
backend_logger = get_logger()

router = APIRouter(prefix="/v1", tags=["OpenWebUI"])


@router.on_event("startup")
async def initialize_openwebui_pipeline():
    """Initialize the OpenWebUI pipeline on startup."""
    try:
        await openwebui_pipeline.initialize()
        logger.info("OpenWebUI pipeline initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize OpenWebUI pipeline", error=str(e))


@router.get("/models")
async def get_models() -> Dict[str, Any]:
    """
    Get available models in OpenAI format.
    
    This endpoint is compatible with OpenWebUI and returns our agents
    as available models that can be selected in the UI.
    
    Returns:
        OpenAI-compatible models list
    """
    try:
        models = await openwebui_pipeline.get_models()
        
        logger.info(
            "Models list requested",
            models_count=len(models.get("data", []))
        )
        
        return models
        
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@router.post("/chat/completions")
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    monitoring_service = None  # Dependency injection will be added later
) -> Any:
    """
    OpenAI-compatible chat completions endpoint.
    
    This endpoint handles chat completion requests from OpenWebUI and routes
    them to our LangGraph agents or multi-agent workflows.
    
    Args:
        request: OpenAI-compatible chat completion request
        
    Returns:
        Chat completion response (streaming or non-streaming)
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Chat completion requested",
            model=request.model,
            messages_count=len(request.messages),
            stream=request.stream,
            user=request.user
        )
        
        # Execute the chat completion
        result = await openwebui_pipeline.chat_completion(request)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Handle streaming response
        if request.stream:
            logger.info(
                "Streaming chat completion started",
                model=request.model,
                execution_time=execution_time
            )
            
            return StreamingResponse(
                result,
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )
        
        # Handle non-streaming response
        logger.info(
            "Chat completion completed",
            model=request.model,
            execution_time=execution_time,
            response_length=len(result.choices[0]["message"]["content"]) if result.choices else 0
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "Chat completion failed",
            model=request.model,
            error=str(e),
            user=request.user
        )
        raise HTTPException(
            status_code=500,
            detail=f"Chat completion failed: {str(e)}"
        )


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> Dict[str, Any]:
    """
    Get specific model information.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model information
    """
    try:
        models = await openwebui_pipeline.get_models()
        
        for model in models.get("data", []):
            if model["id"] == model_id:
                logger.info("Model info requested", model_id=model_id)
                return model
        
        logger.warning("Model not found", model_id=model_id)
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@router.post("/embeddings")
async def create_embeddings(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI-compatible embeddings endpoint (placeholder).
    
    This is a placeholder for future embeddings support.
    Currently returns a not implemented response.
    
    Args:
        request: Embeddings request
        
    Returns:
        Not implemented response
    """
    logger.warning("Embeddings endpoint called but not implemented")
    raise HTTPException(
        status_code=501,
        detail="Embeddings not yet implemented in Agentic AI service"
    )


@router.get("/health/openwebui")
async def openwebui_health() -> Dict[str, Any]:
    """
    Health check endpoint specific to OpenWebUI integration.
    
    Returns:
        Health status of OpenWebUI integration
    """
    try:
        # Check if pipeline is initialized
        if not openwebui_pipeline.orchestrator:
            return {
                "status": "unhealthy",
                "reason": "Pipeline not initialized"
            }
        
        if not openwebui_pipeline.orchestrator.is_initialized:
            return {
                "status": "unhealthy",
                "reason": "Orchestrator not initialized"
            }
        
        # Check available models
        models = await openwebui_pipeline.get_models()
        models_count = len(models.get("data", []))
        
        if models_count == 0:
            return {
                "status": "unhealthy",
                "reason": "No models available"
            }
        
        return {
            "status": "healthy",
            "models_available": models_count,
            "pipeline_ready": True,
            "orchestrator_ready": True
        }
        
    except Exception as e:
        logger.error("OpenWebUI health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "reason": str(e)
        }


@router.post("/agents/register")
async def register_custom_agent(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register a custom agent as an available model.
    
    This endpoint allows dynamic registration of new agents that will
    appear as selectable models in OpenWebUI.
    
    Args:
        agent_config: Agent configuration
        
    Returns:
        Registration result
    """
    try:
        # Validate required fields
        required_fields = ["id", "name", "description", "agent_type"]
        for field in required_fields:
            if field not in agent_config:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Register the agent model
        openwebui_pipeline.available_models[agent_config["id"]] = agent_config
        
        logger.info(
            "Custom agent registered",
            agent_id=agent_config["id"],
            agent_name=agent_config["name"]
        )
        
        return {
            "status": "success",
            "message": f"Agent {agent_config['id']} registered successfully",
            "agent_id": agent_config["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register custom agent", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register agent: {str(e)}"
        )


@router.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str) -> Dict[str, Any]:
    """
    Unregister an agent model.
    
    Args:
        agent_id: Agent identifier to remove
        
    Returns:
        Unregistration result
    """
    try:
        if agent_id not in openwebui_pipeline.available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found"
            )
        
        # Remove the agent model
        del openwebui_pipeline.available_models[agent_id]
        
        logger.info("Agent unregistered", agent_id=agent_id)
        
        return {
            "status": "success",
            "message": f"Agent {agent_id} unregistered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to unregister agent", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unregister agent: {str(e)}"
        )
