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
