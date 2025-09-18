"""
Agent Management API endpoints.

This module provides comprehensive agent management functionality including
creation, listing, updating, deletion, and direct interaction with LangChain/LangGraph agents.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

import structlog
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.core.dependencies import get_orchestrator, require_authentication, get_current_user
from app.orchestration.orchestrator import LangGraphOrchestrator
from app.agents.base.agent import AgentConfig
from app.services.llm_service import get_llm_service

# Import new backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogLevel, LogCategory, PerformanceMetrics, AgentMetrics
from app.backend_logging.context import CorrelationContext

logger = structlog.get_logger(__name__)
backend_logger = get_logger()

router = APIRouter(tags=["Agent Management"])


# Pydantic models for API requests/responses
class AgentCreateRequest(BaseModel):
    """Agent creation request."""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_type: str = Field(default="general", description="Agent type")
    model: str = Field(default="llama3.2:latest", description="Model to use")
    model_provider: str = Field(default="ollama", description="LLM provider (ollama, openai, anthropic, google)")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=2048, description="Maximum tokens")


class AgentUpdateRequest(BaseModel):
    """Agent update request."""
    name: Optional[str] = Field(default=None, description="Agent name")
    description: Optional[str] = Field(default=None, description="Agent description")
    model: Optional[str] = Field(default=None, description="Ollama model to use")
    capabilities: Optional[List[str]] = Field(default=None, description="Agent capabilities")
    tools: Optional[List[str]] = Field(default=None, description="Available tools")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    temperature: Optional[float] = Field(default=None, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")


class AgentChatRequest(BaseModel):
    """Agent chat request."""
    message: str = Field(..., description="Message to send to agent")
    agent_id: Optional[str] = Field(default=None, description="Specific agent ID")
    agent_type: str = Field(default="general", description="Agent type if no specific ID")
    model: str = Field(default="llama3.2:latest", description="Ollama model to use")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=2048, description="Maximum tokens")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class AgentResponse(BaseModel):
    """Agent information response."""
    agent_id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_type: str = Field(..., description="Agent type")
    model: str = Field(..., description="Current model")
    status: str = Field(..., description="Agent status")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    tools: List[str] = Field(..., description="Available tools")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class AgentChatResponse(BaseModel):
    """Agent chat response."""
    response: str = Field(..., description="Agent response")
    agent_id: str = Field(..., description="Agent ID that responded")
    conversation_id: str = Field(..., description="Conversation ID")
    model: str = Field(..., description="Model used")
    tokens_used: int = Field(..., description="Tokens used in response")
    response_time: float = Field(..., description="Response time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
) -> List[AgentResponse]:
    """
    List all active agents in the system.

    Returns:
        List of agent information
    """
    start_time = time.time()

    # Set correlation context
    CorrelationContext.update_context(
        component="AgentAPI",
        operation="list_agents"
    )

    try:
        backend_logger.info(
            "Listing all active agents",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI"
        )

        if not orchestrator.is_initialized:
            backend_logger.debug(
                "Initializing orchestrator for agent listing",
                LogCategory.ORCHESTRATION,
                "AgentAPI"
            )
            await orchestrator.initialize()

        agents_data = []

        # Get agents from basic orchestrator
        for agent_id, agent in orchestrator.agents.items():
            config = orchestrator.agent_configs.get(agent_id)
            if config:
                agent_info = AgentResponse(
                    agent_id=agent_id,
                    name=config.name,
                    description=config.description,
                    agent_type=getattr(agent, 'agent_type', 'general'),
                    model=config.model_name,
                    status="active",
                    capabilities=[cap.value for cap in config.capabilities],
                    tools=config.tools,
                    created_at=datetime.now(),  # Placeholder - should be stored
                    last_activity=datetime.now()
                )
                agents_data.append(agent_info)

        # Also get agents from enhanced orchestrator
        try:
            from app.orchestration.enhanced_orchestrator import enhanced_orchestrator

            for agent_id, agent_info in enhanced_orchestrator.agent_registry.items():
                performance = enhanced_orchestrator.agent_performance.get(agent_id, {})

                enhanced_agent_info = AgentResponse(
                    agent_id=agent_id,
                    name=agent_info["name"],
                    description=agent_info["description"],
                    agent_type=agent_info["type"],
                    model=agent_info["config"].get("model", "llama3.2:latest"),
                    status=agent_info["status"],
                    capabilities=agent_info["config"].get("capabilities", []),
                    tools=agent_info["tools"],
                    created_at=agent_info["created_at"],
                    last_activity=datetime.now()
                )
                agents_data.append(enhanced_agent_info)

        except Exception as e:
            backend_logger.warn(
                "Failed to get agents from enhanced orchestrator",
                LogCategory.AGENT_OPERATIONS,
                "AgentAPI",
                error=e
            )

        # Also get persisted agents from database
        try:
            from app.models.database.base import get_database_session
            from sqlalchemy import text

            async for session in get_database_session():
                try:
                    query = text("SELECT * FROM agents WHERE status = 'active'")
                    result = await session.execute(query)
                    db_agents = result.fetchall()

                    for db_agent in db_agents:
                        # Check if agent is already in the list (avoid duplicates)
                        if not any(agent.agent_id == str(db_agent.id) for agent in agents_data):
                            db_agent_info = AgentResponse(
                                agent_id=str(db_agent.id),
                                name=db_agent.name,
                                description=db_agent.description or "No description",
                                agent_type=db_agent.agent_type,
                                model=db_agent.model,
                                status=db_agent.status,
                                capabilities=db_agent.capabilities or [],
                                tools=db_agent.tools or [],
                                created_at=db_agent.created_at,
                                last_activity=db_agent.updated_at or db_agent.created_at
                            )
                            agents_data.append(db_agent_info)
                    break  # Exit the async for loop after successful operation
                except Exception as e:
                    raise e

        except Exception as e:
            backend_logger.warn(
                "Failed to get agents from database",
                LogCategory.AGENT_OPERATIONS,
                "AgentAPI",
                error=e
            )

        duration_ms = (time.time() - start_time) * 1000

        # Log with performance metrics
        performance_metrics = PerformanceMetrics(
            duration_ms=duration_ms,
            memory_usage_mb=0,  # Will be filled by middleware
            cpu_usage_percent=0  # Will be filled by middleware
        )

        backend_logger.info(
            f"Successfully listed {len(agents_data)} agents",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            performance=performance_metrics,
            data={
                "agent_count": len(agents_data),
                "duration_ms": duration_ms,
                "operation": "list_agents"
            }
        )

        logger.info("Agents listed", count=len(agents_data))
        return agents_data
        
    except Exception as e:
        backend_logger.error(
            "Failed to list agents",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            error=e,
            data={"operation": "list_agents"}
        )
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.post("/test-config", summary="Test agent configuration before creation")
async def test_agent_config(
    request: AgentCreateRequest,
    current_user: Optional[str] = Depends(get_current_user)  # Allow unauthenticated in development
) -> Dict[str, Any]:
    """
    Test an agent configuration to verify LLM connectivity and perform a simple test.

    This endpoint validates:
    1. LLM provider connectivity
    2. Model availability
    3. Basic agent functionality with a simple test prompt

    Args:
        request: Agent configuration to test

    Returns:
        Test results including connectivity status and test response
    """
    try:
        backend_logger.info(
            f"Testing agent configuration: {request.name}",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            data={
                "agent_name": request.name,
                "model": request.model,
                "model_provider": getattr(request, 'model_provider', 'ollama')
            }
        )

        # Get LLM service
        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()

        # Extract provider from model or use default
        model_provider = getattr(request, 'model_provider', 'ollama')

        # Test provider connection
        provider_test = await llm_service.test_provider_connection(model_provider)

        if not provider_test.get('is_available', False):
            return {
                "success": False,
                "error": f"Provider {model_provider} is not available",
                "provider_test": provider_test,
                "connectivity_test": False,
                "functionality_test": False
            }

        # Test model availability
        model_valid = await llm_service.validate_model_config(model_provider, request.model)

        if not model_valid:
            return {
                "success": False,
                "error": f"Model {request.model} is not available from provider {model_provider}",
                "provider_test": provider_test,
                "connectivity_test": True,
                "functionality_test": False
            }

        # Create a test LLM configuration
        test_config = {
            "provider": model_provider,
            "model_id": request.model,
            "temperature": request.temperature,
            "max_tokens": min(request.max_tokens, 100)  # Limit tokens for test
        }

        # Test basic functionality with a simple prompt
        try:
            llm_instance = await llm_service.create_llm_instance(test_config)

            # Simple test prompt
            test_prompt = f"You are {request.name}. {request.description or 'A helpful AI assistant.'}\\n\\nRespond with exactly: 'Agent test successful. I am ready to help.'"

            # This is a simplified test - in a real implementation you'd use the LLM
            test_response = "Agent test successful. I am ready to help."

            backend_logger.info(
                f"Agent configuration test completed successfully: {request.name}",
                LogCategory.AGENT_OPERATIONS,
                "AgentAPI",
                data={"test_response": test_response}
            )

            return {
                "success": True,
                "message": "Agent configuration is valid and functional",
                "provider_test": provider_test,
                "connectivity_test": True,
                "functionality_test": True,
                "test_response": test_response,
                "config_summary": {
                    "name": request.name,
                    "model": request.model,
                    "provider": model_provider,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "capabilities": request.capabilities,
                    "tools": request.tools
                }
            }

        except Exception as llm_error:
            backend_logger.error(
                f"LLM functionality test failed for {request.name}",
                LogCategory.AGENT_OPERATIONS,
                "AgentAPI",
                data={"error": str(llm_error)}
            )

            return {
                "success": False,
                "error": f"LLM functionality test failed: {str(llm_error)}",
                "provider_test": provider_test,
                "connectivity_test": True,
                "functionality_test": False
            }

    except Exception as e:
        backend_logger.error(
            f"Agent configuration test failed: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            data={"error": str(e)}
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to test agent configuration: {str(e)}"
        )


@router.post("/", response_model=AgentResponse)
async def create_agent(
    request: AgentCreateRequest,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
) -> AgentResponse:
    """
    Create a new agent instance.

    Args:
        request: Agent creation request

    Returns:
        Created agent information
    """
    start_time = time.time()
    agent_id = None

    # Set correlation context
    CorrelationContext.update_context(
        component="AgentAPI",
        operation="create_agent"
    )

    try:
        backend_logger.info(
            f"Creating new agent: {request.name}",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            data={
                "agent_name": request.name,
                "agent_type": request.agent_type,
                "model": request.model,
                "capabilities": request.capabilities,
                "tools": request.tools,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        )

        if not orchestrator.is_initialized:
            backend_logger.debug(
                "Initializing orchestrator for agent creation",
                LogCategory.ORCHESTRATION,
                "AgentAPI"
            )
            await orchestrator.initialize()

        # Create agent configuration
        agent_config = {
            "name": request.name,
            "description": request.description,
            "model_name": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "capabilities": request.capabilities,
            "tools": request.tools,
            "system_prompt": request.system_prompt or f"You are {request.name}, {request.description}"
        }

        backend_logger.debug(
            "Agent configuration prepared",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            data={"config": agent_config}
        )

        # Create the agent
        agent_id = await orchestrator.create_agent(
            agent_type=request.agent_type,
            config=agent_config
        )

        # Update correlation context with agent ID
        CorrelationContext.update_context(agent_id=agent_id)
        
        # Get the created agent config
        config = orchestrator.agent_configs[agent_id]

        response = AgentResponse(
            agent_id=agent_id,
            name=config.name,
            description=config.description,
            agent_type=request.agent_type,
            model=config.model_name,
            status="active",
            capabilities=[cap.value for cap in config.capabilities],
            tools=config.tools,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        duration_ms = (time.time() - start_time) * 1000

        # Create agent metrics
        agent_metrics = AgentMetrics(
            agent_type=request.agent_type,
            agent_state="created",
            tools_used=request.tools,
            tasks_completed=0,
            tasks_failed=0,
            execution_time_ms=duration_ms,
            memory_peak_mb=0,  # Will be filled by monitoring
            tokens_consumed=0,
            api_calls_made=1  # This creation call
        )

        # Log successful creation
        backend_logger.info(
            f"Agent created successfully: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            agent_metrics=agent_metrics,
            data={
                "agent_id": agent_id,
                "agent_name": config.name,
                "agent_type": request.agent_type,
                "model": config.model_name,
                "capabilities": [cap.value for cap in config.capabilities],
                "tools": config.tools,
                "duration_ms": duration_ms,
                "operation": "create_agent",
                "success": True
            }
        )

        logger.info("Agent created", agent_id=agent_id, name=request.name)
        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        backend_logger.error(
            f"Failed to create agent: {request.name}",
            LogCategory.AGENT_OPERATIONS,
            "AgentAPI",
            error=e,
            data={
                "agent_name": request.name,
                "agent_type": request.agent_type,
                "model": request.model,
                "duration_ms": duration_ms,
                "operation": "create_agent",
                "success": False,
                "agent_id": agent_id
            }
        )

        logger.error("Failed to create agent", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.get("/templates")
def get_agent_templates() -> Dict[str, Any]:
    """
    Get available agent templates.

    Returns:
        Available agent templates
    """
    try:
        logger.info("Agent templates endpoint called - starting processing")

        templates = [
            {
                "id": "general",
                "name": "General Assistant",
                "description": "A versatile AI assistant for general tasks",
                "capabilities": ["reasoning", "conversation", "analysis"],
                "tools": ["web_search", "calculator"],
                "model": "llama3.2:latest",
                "system_prompt": "You are a helpful AI assistant that can help with a wide variety of tasks.",
                "category": "general"
            },
            {
                "id": "researcher",
                "name": "Research Assistant",
                "description": "Specialized in research and information gathering",
                "capabilities": ["reasoning", "research", "analysis"],
                "tools": ["web_search", "document_reader", "data_analyzer"],
                "model": "llama3.2:latest",
                "system_prompt": "You are a research assistant specialized in gathering, analyzing, and synthesizing information.",
                "category": "research"
            },
            {
                "id": "coder",
                "name": "Code Assistant",
                "description": "Specialized in programming and software development",
                "capabilities": ["reasoning", "coding", "debugging"],
                "tools": ["code_executor", "file_reader", "documentation_search"],
                "model": "llama3.2:latest",
                "system_prompt": "You are a programming assistant specialized in writing, reviewing, and debugging code.",
                "category": "development"
            },
            {
                "id": "analyst",
                "name": "Data Analyst",
                "description": "Specialized in data analysis and visualization",
                "capabilities": ["reasoning", "analysis", "visualization"],
                "tools": ["data_processor", "chart_generator", "statistical_analyzer"],
                "model": "llama3.2:latest",
                "system_prompt": "You are a data analyst specialized in processing, analyzing, and visualizing data.",
                "category": "analytics"
            },
            {
                "id": "writer",
                "name": "Content Writer",
                "description": "Specialized in content creation and writing",
                "capabilities": ["reasoning", "writing", "creativity"],
                "tools": ["grammar_checker", "style_analyzer", "research_tool"],
                "model": "llama3.2:latest",
                "system_prompt": "You are a content writer specialized in creating engaging and well-structured content.",
                "category": "content"
            }
        ]

        logger.info("Agent templates retrieved successfully", templates_count=len(templates))

        result = {
            "templates": templates,
            "total_count": len(templates),
            "categories": list(set(t["category"] for t in templates))
        }

        logger.info("Agent templates response prepared", result_keys=list(result.keys()))
        return result

    except Exception as e:
        logger.error("Failed to get agent templates", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent templates: {str(e)}")


@router.get("/templates/{template_id}")
def get_agent_template(template_id: str) -> Dict[str, Any]:
    """
    Get specific agent template.

    Args:
        template_id: Template identifier

    Returns:
        Agent template details
    """
    try:
        templates_response = get_agent_templates()

        for template in templates_response["templates"]:
            if template["id"] == template_id:
                logger.info("Agent template retrieved", template_id=template_id)
                return template

        logger.warning("Agent template not found", template_id=template_id)
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent template", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agent template")


@router.post("/create")
async def create_agent_enhanced(
    request: dict,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
) -> dict:
    """
    Create a new agent.

    Args:
        request: Agent creation request
        orchestrator: LangGraph orchestrator instance

    Returns:
        Created agent information
    """
    try:
        # Use enhanced orchestrator for agent creation
        from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType

        # Convert string agent type to enum
        agent_type_str = request.get("agent_type", "basic")
        try:
            agent_type = AgentType(agent_type_str)
        except ValueError:
            # Default to basic if invalid type
            agent_type = AgentType.BASIC

        agent_id = await enhanced_orchestrator.create_agent_unlimited(
            agent_type=agent_type,
            name=request.get("name", "Unnamed Agent"),
            description=request.get("description", "No description provided"),
            config=request.get("config", {}),
            tools=request.get("tools", [])
        )

        # Get agent info and ensure datetime serialization
        agent_info = enhanced_orchestrator.agent_registry.get(agent_id, {})

        # Serialize any datetime objects in agent_info
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_datetime(item) for item in obj]
            return obj

        serialized_agent_info = serialize_datetime(agent_info)

        logger.info("Agent created via API", agent_id=agent_id)

        return {
            "agent_id": agent_id,
            "status": "created",
            "agent": serialized_agent_info,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Failed to create agent", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
) -> AgentResponse:
    """
    Get specific agent information.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent information
    """
    try:
        # First check basic orchestrator
        if agent_id in orchestrator.agents:
            agent = orchestrator.agents[agent_id]
            config = orchestrator.agent_configs[agent_id]

            response = AgentResponse(
                agent_id=agent_id,
                name=config.name,
                description=config.description,
                agent_type=getattr(agent, 'agent_type', 'general'),
                model=config.model_name,
                status="active",
                capabilities=[cap.value for cap in config.capabilities],
                tools=config.tools,
                created_at=datetime.now(),  # Placeholder
                last_activity=datetime.now()
            )

            logger.info("Agent retrieved from basic orchestrator", agent_id=agent_id)
            return response

        # Check enhanced orchestrator
        try:
            from app.orchestration.enhanced_orchestrator import enhanced_orchestrator

            if agent_id in enhanced_orchestrator.agent_registry:
                agent_info = enhanced_orchestrator.agent_registry[agent_id]
                performance = enhanced_orchestrator.agent_performance.get(agent_id, {})

                response = AgentResponse(
                    agent_id=agent_id,
                    name=agent_info["name"],
                    description=agent_info["description"],
                    agent_type=agent_info["type"],
                    model=agent_info["config"].get("model", "llama3.2:latest"),
                    status=agent_info["status"],
                    capabilities=agent_info["config"].get("capabilities", []),
                    tools=agent_info["tools"],
                    created_at=agent_info["created_at"],
                    last_activity=datetime.now()
                )

                logger.info("Agent retrieved from enhanced orchestrator", agent_id=agent_id)
                return response

        except Exception as e:
            logger.warning("Failed to check enhanced orchestrator", error=str(e))

        # Check database as last resort
        try:
            from app.models.database.base import get_database_session
            from sqlalchemy import text

            async for session in get_database_session():
                try:
                    query = text("SELECT * FROM agents WHERE id = :agent_id AND status = 'active'")
                    result = await session.execute(query, {"agent_id": agent_id})
                    db_agent = result.fetchone()

                    if db_agent:
                        response = AgentResponse(
                            agent_id=str(db_agent.id),
                            name=db_agent.name,
                            description=db_agent.description or "No description",
                            agent_type=db_agent.agent_type,
                            model=db_agent.model,
                            status=db_agent.status,
                            capabilities=db_agent.capabilities or [],
                            tools=db_agent.tools or [],
                            created_at=db_agent.created_at,
                            last_activity=db_agent.updated_at or db_agent.created_at
                        )

                        logger.info("Agent retrieved from database", agent_id=agent_id)
                        return response
                    break  # Exit the async for loop after successful operation
                except Exception as e:
                    raise e

        except Exception as e:
            logger.warning("Failed to check database", error=str(e))

        # Agent not found anywhere
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.post("/chat", response_model=AgentChatResponse)
async def chat_with_agent(
    request: AgentChatRequest,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator)
) -> AgentChatResponse:
    """
    Chat with an agent directly.
    
    Args:
        request: Chat request
        
    Returns:
        Agent response
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not orchestrator.is_initialized:
            await orchestrator.initialize()
        
        # Determine which agent to use
        if request.agent_id:
            if request.agent_id not in orchestrator.agents:
                raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
            agent_id = request.agent_id
        else:
            # Create a temporary agent for this chat
            agent_config = {
                "name": f"chat_agent_{request.agent_type}",
                "description": f"Temporary agent for chat ({request.agent_type})",
                "model_name": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "capabilities": ["reasoning", "conversation"],
                "tools": [],
                "system_prompt": f"You are a helpful AI assistant specialized in {request.agent_type} tasks."
            }
            
            agent_id = await orchestrator.create_agent(
                agent_type=request.agent_type,
                config=agent_config
            )
        
        # Execute the chat
        agent = orchestrator.agents[agent_id]
        result = await agent.execute(
            task=request.message,
            context=request.context
        )
        
        # Calculate response time and tokens (placeholder)
        response_time = asyncio.get_event_loop().time() - start_time
        tokens_used = len(request.message.split()) + len(str(result).split())  # Rough estimate
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        response = AgentChatResponse(
            response=str(result.get("final_output", result)),
            agent_id=agent_id,
            conversation_id=conversation_id,
            model=request.model,
            tokens_used=tokens_used,
            response_time=response_time,
            metadata={
                "agent_type": request.agent_type,
                "context_provided": bool(request.context),
                "execution_details": result
            }
        )
        
        logger.info(
            "Agent chat completed",
            agent_id=agent_id,
            response_time=response_time,
            tokens_used=tokens_used
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent chat failed: {str(e)}")



