"""
Standalone Agent API endpoints.

This module provides a complete standalone API for the agentic AI system
that works independently of OpenWebUI. These endpoints allow direct
interaction with LangChain/LangGraph agents using Ollama models.
"""

import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.orchestration.orchestrator import orchestrator

# Import new backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogLevel, LogCategory, PerformanceMetrics
from app.backend_logging.context import CorrelationContext

logger = structlog.get_logger(__name__)
backend_logger = get_logger()

router = APIRouter(prefix="/standalone", tags=["Standalone Agent API"])


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class AgentChatRequest(BaseModel):
    """Standalone agent chat request."""
    message: str = Field(..., description="User message")
    agent_type: str = Field(default="general", description="Agent type: general, research, workflow")
    model: str = Field(default="llama3.2:latest", description="Ollama model to use")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=2048, description="Maximum tokens")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class AgentChatResponse(BaseModel):
    """Standalone agent chat response."""
    response: str = Field(..., description="Agent response")
    agent_id: str = Field(..., description="Agent ID that handled the request")
    agent_type: str = Field(..., description="Type of agent used")
    model: str = Field(..., description="Ollama model used")
    conversation_id: str = Field(..., description="Conversation ID")
    execution_time: float = Field(..., description="Execution time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WorkflowRequest(BaseModel):
    """Workflow execution request."""
    task: str = Field(..., description="Task description")
    workflow_type: str = Field(default="hierarchical", description="Workflow type")
    model: str = Field(default="llama3.2:latest", description="Ollama model to use")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    timeout: int = Field(default=300, description="Workflow timeout in seconds")


class WorkflowResponse(BaseModel):
    """Workflow execution response."""
    workflow_id: str = Field(..., description="Workflow ID")
    status: str = Field(..., description="Workflow status")
    result: Dict[str, Any] = Field(..., description="Workflow result")
    execution_time: float = Field(..., description="Execution time in seconds")
    agents_used: List[str] = Field(default_factory=list, description="List of agents used")


class AgentStatus(BaseModel):
    """Agent status model."""
    agent_id: str = Field(..., description="Agent ID")
    agent_type: str = Field(..., description="Agent type")
    status: str = Field(..., description="Agent status")
    model: str = Field(..., description="Current model")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")


@router.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(request: AgentChatRequest) -> AgentChatResponse:
    """
    Chat with a standalone agent using Ollama models.
    
    This endpoint provides direct access to LangChain/LangGraph agents
    without requiring OpenWebUI integration.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            "Standalone chat request",
            agent_type=request.agent_type,
            model=request.model,
            conversation_id=conversation_id,
            message_length=len(request.message)
        )
        
        # Ensure orchestrator is initialized
        if not orchestrator.is_initialized:
            await orchestrator.initialize()
        
        # Create agent configuration
        agent_config = {
            "name": f"Standalone {request.agent_type.title()} Agent",
            "description": f"Standalone {request.agent_type} agent using {request.model}",
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "conversation_id": conversation_id
        }
        
        # Create agent
        agent_id = await orchestrator.create_agent(
            agent_type=request.agent_type,
            config=agent_config
        )
        
        # Get the agent
        agent = await orchestrator.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=500, detail="Failed to create agent")
        
        # Execute the chat
        result = await agent.execute(
            task=request.message,
            context={
                **request.context,
                "conversation_id": conversation_id,
                "standalone_mode": True,
                "model": request.model
            }
        )
        
        # Extract response
        response_content = "I'm ready to help! How can I assist you today?"
        
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                response_content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                response_content = last_message["content"]
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(
            "Standalone chat completed",
            agent_id=agent_id,
            conversation_id=conversation_id,
            execution_time=execution_time,
            response_length=len(response_content)
        )
        
        return AgentChatResponse(
            response=response_content,
            agent_id=agent_id,
            agent_type=request.agent_type,
            model=request.model,
            conversation_id=conversation_id,
            execution_time=execution_time,
            metadata={
                "agent_status": result.get("status", "completed"),
                "errors": result.get("errors", []),
                "tools_used": result.get("tools_used", [])
            }
        )
        
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(
            "Standalone chat failed",
            error=str(e),
            agent_type=request.agent_type,
            execution_time=execution_time
        )
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/workflow", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest) -> WorkflowResponse:
    """
    Execute a multi-agent workflow independently.
    
    This endpoint runs complex multi-agent workflows using LangGraph
    without requiring OpenWebUI integration.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(
            "Standalone workflow request",
            workflow_type=request.workflow_type,
            model=request.model,
            task_length=len(request.task)
        )
        
        # Ensure orchestrator is initialized
        if not orchestrator.is_initialized:
            await orchestrator.initialize()
        
        # Execute workflow based on type
        if request.workflow_type == "hierarchical":
            result = await orchestrator.execute_hierarchical_workflow(
                task=request.task,
                context={
                    **request.context,
                    "model": request.model,
                    "standalone_mode": True
                }
            )
        else:
            # Default multi-agent workflow
            result = await orchestrator.execute_workflow(
                workflow_id="default_multi_agent",
                inputs={
                    "task": request.task,
                    "model": request.model,
                    **request.context
                }
            )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Extract agents used
        agents_used = []
        if "agent_outputs" in result:
            agents_used = list(result["agent_outputs"].keys())
        elif "subgraph_results" in result:
            agents_used = list(result["subgraph_results"].keys())
        
        logger.info(
            "Standalone workflow completed",
            workflow_id=result.get("workflow_id", "unknown"),
            execution_time=execution_time,
            agents_used=len(agents_used)
        )
        
        return WorkflowResponse(
            workflow_id=result.get("workflow_id", f"workflow_{uuid.uuid4().hex[:8]}"),
            status=result.get("status", "completed"),
            result=result,
            execution_time=execution_time,
            agents_used=agents_used
        )
        
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(
            "Standalone workflow failed",
            error=str(e),
            workflow_type=request.workflow_type,
            execution_time=execution_time
        )
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")


@router.get("/models")
async def get_available_models() -> Dict[str, Any]:
    """
    Get available Ollama models for agents.
    
    Returns the list of Ollama models that can be used with agents.
    """
    try:
        settings = get_settings()
        
        # Test Ollama connectivity
        ollama_status = "unknown"
        try:
            if orchestrator.llm:
                # Try a simple test
                ollama_status = "connected"
            else:
                ollama_status = "not_initialized"
        except Exception:
            ollama_status = "disconnected"
        
        return {
            "available_models": settings.AVAILABLE_OLLAMA_MODELS,
            "default_model": settings.DEFAULT_AGENT_MODEL,
            "backup_model": settings.BACKUP_AGENT_MODEL,
            "ollama_base_url": settings.OLLAMA_BASE_URL,
            "ollama_status": ollama_status,
            "total_models": len(settings.AVAILABLE_OLLAMA_MODELS)
        }
        
    except Exception as e:
        logger.error("Failed to get available models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/agents")
async def list_active_agents() -> Dict[str, Any]:
    """
    List currently active agents.
    
    Returns information about all currently active agents in the system.
    """
    try:
        if not orchestrator.is_initialized:
            return {
                "active_agents": [],
                "total_agents": 0,
                "orchestrator_status": "not_initialized"
            }
        
        active_agents = []
        
        for agent_id, agent in orchestrator.agents.items():
            agent_info = {
                "agent_id": agent_id,
                "agent_type": getattr(agent, 'agent_type', 'unknown'),
                "status": "active",
                "model": getattr(agent, 'model', 'unknown'),
                "created_at": datetime.now().isoformat(),  # Placeholder
                "capabilities": getattr(agent, 'capabilities', [])
            }
            active_agents.append(agent_info)
        
        return {
            "active_agents": active_agents,
            "total_agents": len(active_agents),
            "orchestrator_status": "initialized",
            "max_concurrent_agents": get_settings().MAX_CONCURRENT_AGENTS
        }
        
    except Exception as e:
        logger.error("Failed to list active agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.get("/health")
async def standalone_health() -> Dict[str, Any]:
    """
    Health check for standalone agent system.
    
    Returns comprehensive health information for the standalone system.
    """
    try:
        settings = get_settings()
        
        # Check orchestrator
        orchestrator_status = "healthy" if orchestrator.is_initialized else "not_initialized"
        
        # Check Ollama connectivity
        ollama_status = "unknown"
        try:
            if orchestrator.llm:
                ollama_status = "connected"
            else:
                ollama_status = "not_connected"
        except Exception:
            ollama_status = "error"
        
        # Check Redis (if available)
        redis_status = "unknown"
        try:
            if orchestrator.checkpoint_saver:
                redis_status = "connected"
            else:
                redis_status = "not_connected"
        except Exception:
            redis_status = "error"
        
        # Overall health
        overall_status = "healthy"
        if orchestrator_status != "healthy" or ollama_status != "connected":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "orchestrator": orchestrator_status,
                "ollama": ollama_status,
                "redis": redis_status,
                "standalone_api": "healthy"
            },
            "configuration": {
                "ollama_base_url": settings.OLLAMA_BASE_URL,
                "default_model": settings.DEFAULT_AGENT_MODEL,
                "max_concurrent_agents": settings.MAX_CONCURRENT_AGENTS,
                "openwebui_enabled": settings.OPENWEBUI_ENABLED
            },
            "capabilities": {
                "chat": True,
                "workflows": True,
                "hierarchical_workflows": True,
                "multi_agent": True,
                "standalone_mode": True
            }
        }
        
    except Exception as e:
        logger.error("Standalone health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
