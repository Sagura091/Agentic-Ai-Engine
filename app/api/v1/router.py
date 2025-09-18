"""
Main API router for version 1 of the Agentic AI Microservice API.

This module aggregates all API endpoints and provides a single router
for the FastAPI application to include.
"""

from fastapi import APIRouter, Depends

from app.api.v1.endpoints import (
    agents,
    health,
    workflows,
    admin,
    openwebui,
    standalone,
    autonomous_agents,
    enhanced_orchestration,
    monitoring,
    models,
    settings,
    frontend_logs,
    backend_logs,
    llm_providers,
    conversational_agents,
    rag,
    embedding_models,
    rag_upload
)

# Create the main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"],
)

api_router.include_router(
    agents.router,
    prefix="/agents",
    tags=["agents"],
)

api_router.include_router(
    workflows.router,
    prefix="/workflows",
    tags=["workflows"],
)

# Add explicit route for /agents without trailing slash to avoid 307 redirects
from app.api.v1.endpoints.agents import create_agent as agents_create_agent, list_agents as agents_list_agents, AgentCreateRequest, AgentResponse
from app.core.dependencies import get_orchestrator
from typing import List

@api_router.get("/agents", response_model=List[AgentResponse], tags=["agents"])
async def list_agents_no_slash(
    orchestrator = Depends(get_orchestrator)
):
    """List agents endpoint without trailing slash to avoid 307 redirects."""
    return await agents_list_agents(orchestrator)

@api_router.post("/agents", response_model=AgentResponse, tags=["agents"])
async def create_agent_no_slash(
    request: AgentCreateRequest,
    orchestrator = Depends(get_orchestrator)
):
    """Create agent endpoint without trailing slash to avoid 307 redirects."""
    return await agents_create_agent(request, orchestrator)

api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
)

# Standalone Agent API (Primary functionality)
api_router.include_router(
    standalone.router,
    tags=["standalone"],
)

# OpenWebUI-compatible endpoints (optional integration)
api_router.include_router(
    openwebui.router,
    tags=["openwebui"],
)

# Autonomous Agents API (Revolutionary agentic capabilities)
api_router.include_router(
    autonomous_agents.router,
    tags=["autonomous"],
)

# Enhanced Orchestration API (Unlimited agents and dynamic tools)
api_router.include_router(
    enhanced_orchestration.router,
    tags=["orchestration"],
)

# Monitoring API (System metrics and monitoring)
api_router.include_router(
    monitoring.router,
    tags=["monitoring"],
)

# Models API (Model management and information)
api_router.include_router(
    models.router,
    tags=["models"],
)

# Settings API (System configuration)
api_router.include_router(
    settings.router,
    tags=["settings"],
)

# Frontend Logs API (Frontend logging and debugging)
api_router.include_router(
    frontend_logs.router,
    prefix="/frontend",
    tags=["frontend-logs"],
)

# Backend Logs API (Backend logging and monitoring)
api_router.include_router(
    backend_logs.router,
    prefix="/backend",
    tags=["backend-logs"],
)

# LLM Providers API (Multi-provider LLM management)
api_router.include_router(
    llm_providers.router,
    prefix="/llm",
    tags=["llm-providers"],
)

api_router.include_router(
    conversational_agents.router,
    prefix="/conversational-agents",
    tags=["conversational-agents"],
)

# RAG API (Revolutionary knowledge management)
api_router.include_router(
    rag.router,
    tags=["rag"],
)

# Embedding Models API (Model download and management)
api_router.include_router(
    embedding_models.router,
    prefix="/rag/embeddings",
    tags=["embedding-models"],
)

# RAG Upload API (Revolutionary file upload and processing)
api_router.include_router(
    rag_upload.router,
    tags=["rag-upload"],
)


# ============================================================================
# AGENT BUILDER PLATFORM INTEGRATION
# ============================================================================

from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import Agent Builder Platform
from app.agent_builder_platform import get_agent_builder_platform, get_platform_sync
from app.agents.factory import AgentBuilderConfig, AgentType
from app.llm.models import LLMConfig, ProviderType


class PlatformStatusResponse(BaseModel):
    """Platform status response model."""
    status: str
    platform_metrics: Dict[str, Any]
    components: Dict[str, bool]
    registry_stats: Optional[Dict[str, Any]] = None
    llm_provider_health: Optional[Dict[str, Any]] = None


class QuickAgentRequest(BaseModel):
    """Quick agent creation request."""
    template_name: str = Field(..., description="Template name to use")
    agent_name: Optional[str] = Field(None, description="Custom agent name")
    customizations: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")
    owner: str = Field(default="api_user", description="Agent owner")


class QuickAgentResponse(BaseModel):
    """Quick agent creation response."""
    agent_id: str
    agent_name: str
    template_used: str
    status: str
    created_at: datetime


# Agent Builder Platform Status Endpoint
@api_router.get("/platform/status", response_model=PlatformStatusResponse, tags=["Agent Builder Platform"])
async def get_platform_status() -> PlatformStatusResponse:
    """
    Get comprehensive Agent Builder Platform status.

    This endpoint provides a complete overview of the platform including
    component health, metrics, and operational statistics.
    """
    try:
        platform = await get_agent_builder_platform()
        status_data = platform.get_platform_status()

        return PlatformStatusResponse(
            status=status_data.get("initialization_status", "unknown"),
            platform_metrics=status_data.get("platform_metrics", {}),
            components=status_data.get("components", {}),
            registry_stats=status_data.get("registry_stats"),
            llm_provider_health=status_data.get("llm_provider_health")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get platform status: {str(e)}")


# Quick Agent Creation Endpoint
@api_router.post("/platform/quick-agent", response_model=QuickAgentResponse, tags=["Agent Builder Platform"])
async def create_quick_agent(request: QuickAgentRequest) -> QuickAgentResponse:
    """
    Create an agent quickly from a template.

    This endpoint provides a simplified way to create agents using
    predefined templates with optional customizations.
    """
    try:
        platform = await get_agent_builder_platform()

        agent_id = await platform.create_agent_from_template(
            template_name=request.template_name,
            agent_name=request.agent_name,
            customizations=request.customizations,
            owner=request.owner
        )

        if not agent_id:
            raise HTTPException(status_code=400, detail="Failed to create agent")

        return QuickAgentResponse(
            agent_id=agent_id,
            agent_name=request.agent_name or f"Agent from {request.template_name}",
            template_used=request.template_name,
            status="created",
            created_at=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


# Platform Templates Endpoint
@api_router.get("/platform/templates", response_model=List[Dict[str, Any]], tags=["Agent Builder Platform"])
async def get_platform_templates() -> List[Dict[str, Any]]:
    """
    Get all available agent templates.

    Returns a list of all predefined agent templates that can be used
    for quick agent creation.
    """
    try:
        platform = await get_agent_builder_platform()
        return platform.list_available_templates()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


# Platform Agents List Endpoint
@api_router.get("/platform/agents", response_model=List[Dict[str, Any]], tags=["Agent Builder Platform"])
async def list_platform_agents(owner: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all agents created through the platform.

    Optionally filter by owner to see only specific user's agents.
    """
    try:
        platform = await get_agent_builder_platform()
        return platform.list_agents(owner=owner)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


# ============================================================================
# VISUAL BUILDER & COMPONENT MANAGEMENT ENDPOINTS
# ============================================================================

@api_router.get("/components/palette")
async def get_component_palette():
    """Get component palette for visual builder."""
    try:
        platform = await get_agent_builder_platform()
        palette = platform.template_library.get_component_palette()

        return {
            "status": "success",
            "palette": palette,
            "total_components": sum(len(components) for components in palette.values())
        }

    except Exception as e:
        logger.error("Failed to get component palette", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get component palette: {str(e)}")


@api_router.post("/components/custom")
async def save_custom_component(component_data: Dict[str, Any]):
    """Save a custom component to the library."""
    try:
        platform = await get_agent_builder_platform()

        # Create component from data
        from app.agents.templates import AgentComponent
        component = AgentComponent(
            component_id=component_data["id"],
            name=component_data["name"],
            component_type=component_data["type"],
            configuration=component_data["configuration"]
        )

        # Save to library
        success = platform.template_library.component_library.save_custom_component(component)

        if success:
            return {
                "status": "success",
                "message": "Custom component saved successfully",
                "component_id": component.component_id
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to save custom component")

    except Exception as e:
        logger.error("Failed to save custom component", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save custom component: {str(e)}")


@api_router.post("/agents/from-components")
async def create_agent_from_components(request_data: Dict[str, Any]):
    """Create an agent from visual components."""
    try:
        platform = await get_agent_builder_platform()

        components = request_data.get("components", [])
        base_config = request_data.get("config", {})

        # Create agent configuration from components
        config = platform.template_library.create_agent_from_components(components, base_config)

        # Create the agent
        agent_id = await platform.create_agent_from_config(config)

        return {
            "status": "success",
            "message": "Agent created from components successfully",
            "agent_id": agent_id,
            "components_used": components,
            "agent_type": config.agent_type.value
        }

    except Exception as e:
        logger.error("Failed to create agent from components", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create agent from components: {str(e)}")


@api_router.get("/models/available")
async def get_available_models():
    """Get all available models for manual selection."""
    try:
        platform = await get_agent_builder_platform()
        models = platform.llm_manager.get_available_models_by_provider()

        return {
            "status": "success",
            "models": models,
            "total_providers": len(models)
        }

    except Exception as e:
        logger.error("Failed to get available models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")


@api_router.get("/models/recommendations/{agent_type}")
async def get_model_recommendations(agent_type: str):
    """Get model recommendations for a specific agent type."""
    try:
        platform = await get_agent_builder_platform()
        recommendations = platform.llm_manager.get_model_recommendations(agent_type)

        return {
            "status": "success",
            "agent_type": agent_type,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error("Failed to get model recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model recommendations: {str(e)}")


@api_router.get("/platform/async-status")
async def get_async_status():
    """Get async processing status."""
    try:
        platform = await get_agent_builder_platform()
        status = platform.get_async_status()

        return {
            "status": "success",
            "async_processing": status
        }

    except Exception as e:
        logger.error("Failed to get async status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get async status: {str(e)}")


@api_router.get("/platform/distributed-status")
async def get_distributed_status():
    """Get distributed architecture status."""
    try:
        platform = await get_agent_builder_platform()
        status = platform.agent_registry.get_distributed_status()

        return {
            "status": "success",
            "distributed_architecture": status
        }

    except Exception as e:
        logger.error("Failed to get distributed status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get distributed status: {str(e)}")
