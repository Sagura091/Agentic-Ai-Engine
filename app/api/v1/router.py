"""
Main API router for version 1 of the Agentic AI Microservice API.

This module aggregates all API endpoints and provides a single router
for the FastAPI application to include.
"""

from fastapi import APIRouter, Depends, Response

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
    rag_upload,
    database_management,
    nodes,
    session_documents
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

api_router.include_router(
    nodes.router,
    prefix="/nodes",
    tags=["nodes"],
)

# Add explicit route for /agents without trailing slash to avoid 307 redirects
from app.api.v1.endpoints.agents import create_agent as agents_create_agent, list_agents as agents_list_agents, AgentCreateRequest, AgentResponse
from app.core.dependencies import get_orchestrator, get_current_user
from app.core.pagination import AdvancedQueryParams
from app.api.v1.responses import StandardAPIResponse
from typing import List, Optional

@api_router.get("/agents", response_model=StandardAPIResponse, tags=["agents"])
async def list_agents_no_slash(
    response: Response,
    query_params: AdvancedQueryParams = Depends(),
    current_user: Optional[str] = Depends(get_current_user)
):
    """List agents endpoint without trailing slash to avoid 307 redirects."""
    return await agents_list_agents(response=response, query_params=query_params, current_user=current_user)

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

# Database Management API (Migration and maintenance)
api_router.include_router(
    database_management.router,
    tags=["database-management"],
)

# Session Documents API (Revolutionary session-based document processing)
api_router.include_router(
    session_documents.router,
    tags=["session-documents"],
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


# ============================================================================
# REVOLUTIONARY COMPONENT WORKFLOW EXECUTION ENDPOINTS
# ============================================================================

class ComponentWorkflowRequest(BaseModel):
    """Component workflow execution request."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    components: List[Dict[str, Any]] = Field(..., description="List of workflow components")
    execution_mode: str = Field(default="sequential", description="Execution mode: sequential, parallel, autonomous")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution context")


class ComponentWorkflowResponse(BaseModel):
    """Component workflow execution response."""
    workflow_id: str
    status: str
    message: str
    total_steps: int
    execution_mode: str
    queued_at: datetime


class WorkflowStepStatusResponse(BaseModel):
    """Workflow step status response."""
    step_id: str
    status: str
    component_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = Field(default_factory=list)


class ComponentExecutionRequest(BaseModel):
    """Component execution request."""
    component_id: str = Field(..., description="Component identifier")
    component_type: str = Field(..., description="Component type")
    component_config: Dict[str, Any] = Field(..., description="Component configuration")
    execution_mode: str = Field(default="default", description="Execution mode")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution context")


class ComponentExecutionResponse(BaseModel):
    """Component execution response."""
    component_id: str
    status: str
    message: str
    execution_mode: str
    queued_at: datetime


@api_router.post("/workflows/execute-components", response_model=ComponentWorkflowResponse, tags=["Component Workflows"])
async def execute_component_workflow(request: ComponentWorkflowRequest) -> ComponentWorkflowResponse:
    """
    Execute a component-based workflow.

    This revolutionary endpoint allows execution of workflows created from
    visual components with support for sequential, parallel, and autonomous
    execution modes.
    """
    try:
        # Get the unified system orchestrator
        from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
        orchestrator = get_enhanced_system_orchestrator()

        if not orchestrator.is_initialized:
            await orchestrator.initialize()

        # Execute component workflow
        result = await orchestrator.execute_component_workflow(
            workflow_id=request.workflow_id,
            components=request.components,
            execution_mode=request.execution_mode,
            context=request.context
        )

        logger.info(
            "Component workflow execution started",
            workflow_id=request.workflow_id,
            num_components=len(request.components),
            execution_mode=request.execution_mode
        )

        return ComponentWorkflowResponse(
            workflow_id=result["workflow_id"],
            status=result["status"],
            message=result["message"],
            total_steps=result["total_steps"],
            execution_mode=request.execution_mode,
            queued_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error("Failed to execute component workflow", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to execute component workflow: {str(e)}")


@api_router.get("/workflows/steps/{step_id}/status", response_model=WorkflowStepStatusResponse, tags=["Component Workflows"])
async def get_workflow_step_status(step_id: str) -> WorkflowStepStatusResponse:
    """
    Get the status of a specific workflow step.

    This endpoint provides detailed information about the execution state
    of individual workflow steps, including timing, results, and events.
    """
    try:
        from app.agent_builder_platform import get_step_state_tracker
        step_tracker = get_step_state_tracker()

        step_state = step_tracker.get_step_state(step_id)

        if not step_state:
            raise HTTPException(status_code=404, detail=f"Step not found: {step_id}")

        return WorkflowStepStatusResponse(
            step_id=step_state["step_id"],
            status=step_state["status"],
            component_type=step_state.get("component_type"),
            start_time=step_state.get("start_time"),
            end_time=step_state.get("end_time"),
            execution_time=step_state.get("execution_time"),
            result=step_state.get("result"),
            events=step_state.get("events", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow step status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get step status: {str(e)}")


@api_router.post("/components/execute", response_model=ComponentExecutionResponse, tags=["Component Workflows"])
async def execute_component(request: ComponentExecutionRequest) -> ComponentExecutionResponse:
    """
    Execute a single component.

    This endpoint allows execution of individual components with support
    for autonomous, instruction-based, and default execution modes.
    """
    try:
        from app.agent_builder_platform import get_component_agent_manager
        component_manager = await get_component_agent_manager()

        # Create component agent
        component_agent = await component_manager.create_component_agent(
            agent_id=request.component_id,
            component_type=request.component_type,
            component_config=request.component_config
        )

        # Execute component agent
        result = await component_manager.execute_component_agent(
            agent_id=request.component_id,
            execution_context=request.context,
            execution_mode=request.execution_mode
        )

        logger.info(
            "Component execution started",
            component_id=request.component_id,
            component_type=request.component_type,
            execution_mode=request.execution_mode
        )

        return ComponentExecutionResponse(
            component_id=result["agent_id"],
            status=result["status"],
            message=result["message"],
            execution_mode=request.execution_mode,
            queued_at=datetime.utcnow()
        )

    except Exception as e:
        logger.error("Failed to execute component", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to execute component: {str(e)}")


@api_router.get("/workflows/{workflow_id}/steps", tags=["Component Workflows"])
async def get_workflow_steps(workflow_id: str) -> Dict[str, Any]:
    """
    Get all steps for a specific workflow.

    This endpoint returns information about all steps associated with
    a workflow, including their current status and execution details.
    """
    try:
        from app.agent_builder_platform import get_step_state_tracker
        step_tracker = get_step_state_tracker()

        step_ids = step_tracker.get_workflow_steps(workflow_id)

        steps = []
        for step_id in step_ids:
            step_state = step_tracker.get_step_state(step_id)
            if step_state:
                steps.append({
                    "step_id": step_id,
                    "status": step_state["status"],
                    "component_type": step_state.get("component_type"),
                    "start_time": step_state.get("start_time"),
                    "end_time": step_state.get("end_time"),
                    "execution_time": step_state.get("execution_time")
                })

        return {
            "workflow_id": workflow_id,
            "total_steps": len(steps),
            "steps": steps
        }

    except Exception as e:
        logger.error("Failed to get workflow steps", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get workflow steps: {str(e)}")


@api_router.get("/components/agents", tags=["Component Workflows"])
async def list_component_agents() -> Dict[str, Any]:
    """
    List all component agents.

    This endpoint returns information about all component agents
    currently managed by the system.
    """
    try:
        from app.agent_builder_platform import get_component_agent_manager
        component_manager = await get_component_agent_manager()

        agents = component_manager.list_component_agents()
        active_agents = component_manager.get_active_agents()

        return {
            "total_agents": len(agents),
            "active_agents": len(active_agents),
            "agents": agents,
            "active_agent_ids": active_agents
        }

    except Exception as e:
        logger.error("Failed to list component agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list component agents: {str(e)}")


@api_router.get("/workflows/active", tags=["Component Workflows"])
async def get_active_workflows() -> Dict[str, Any]:
    """
    Get all active workflows.

    This endpoint returns information about all currently active
    component workflows and their execution status.
    """
    try:
        from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
        orchestrator = get_enhanced_system_orchestrator()

        if orchestrator.component_workflow_executor:
            active_workflows = orchestrator.component_workflow_executor.list_active_workflows()

            workflow_details = []
            for workflow_id in active_workflows:
                workflow_status = orchestrator.component_workflow_executor.get_workflow_status(workflow_id)
                if workflow_status:
                    workflow_details.append({
                        "workflow_id": workflow_id,
                        "status": workflow_status["status"],
                        "execution_mode": workflow_status.get("execution_mode"),
                        "current_step": workflow_status.get("current_step", 0),
                        "total_steps": workflow_status.get("total_steps", 0),
                        "start_time": workflow_status.get("start_time")
                    })

            return {
                "total_active_workflows": len(active_workflows),
                "workflows": workflow_details
            }
        else:
            return {
                "total_active_workflows": 0,
                "workflows": [],
                "message": "Component workflow executor not initialized"
            }

    except Exception as e:
        logger.error("Failed to get active workflows", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get active workflows: {str(e)}")
