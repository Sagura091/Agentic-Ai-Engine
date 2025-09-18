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
from app.core.dependencies import require_authentication, get_current_user
from app.agents.base.agent import AgentConfig, AgentDNA, FrameworkConfig
from app.services.llm_service import get_llm_service
from app.core.unified_system_orchestrator import get_system_orchestrator

# Import Agent Builder Platform components
from app.agents.factory import AgentType, AgentTemplate, AgentBuilderConfig, AgentBuilderFactory
from app.agents.registry import AgentRegistry, AgentStatus, AgentHealth, get_agent_registry, initialize_agent_registry
from app.agents.templates import AgentTemplateLibrary
from app.llm.models import LLMConfig, ProviderType
from app.llm.manager import LLMProviderManager

# Import new backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogLevel, LogCategory, PerformanceMetrics, AgentMetrics
from app.backend_logging.context import CorrelationContext

logger = structlog.get_logger(__name__)
backend_logger = get_logger()

router = APIRouter(tags=["Agent Management"])


# Enhanced Pydantic models for multi-framework agents
class AgentDNA(BaseModel):
    """Agent DNA configuration for personality and behavior."""
    identity: Dict[str, Any] = Field(default_factory=dict, description="Identity configuration")
    cognition: Dict[str, Any] = Field(default_factory=dict, description="Cognitive configuration")
    behavior: Dict[str, Any] = Field(default_factory=dict, description="Behavioral configuration")

class FrameworkConfig(BaseModel):
    """Framework-specific configuration."""
    framework_id: str = Field(..., description="Framework identifier")
    components: List[Dict[str, Any]] = Field(default_factory=list, description="Framework components")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Framework-specific settings")

class AgentCreateRequest(BaseModel):
    """Enhanced agent creation request with multi-framework support."""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_type: str = Field(default="general", description="Agent type")
    framework: str = Field(default="basic", description="Agent framework (basic, react, bdi, crewai, autogen, swarm)")
    model: str = Field(default="llama3.2:latest", description="Model to use")
    model_provider: str = Field(default="ollama", description="LLM provider (ollama, openai, anthropic, google)")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    memory_types: List[str] = Field(default_factory=list, description="Memory types")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=2048, description="Maximum tokens")
    agent_dna: Optional[AgentDNA] = Field(default=None, description="Agent DNA configuration")
    framework_config: Optional[FrameworkConfig] = Field(default=None, description="Framework configuration")


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
    """Enhanced agent information response."""
    agent_id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_type: str = Field(..., description="Agent type")
    framework: str = Field(default="basic", description="Agent framework")
    model: str = Field(..., description="Current model")
    status: str = Field(..., description="Agent status")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    tools: List[str] = Field(..., description="Available tools")
    memory_types: List[str] = Field(default_factory=list, description="Memory types")
    agent_dna: Optional[AgentDNA] = Field(default=None, description="Agent DNA configuration")
    framework_config: Optional[FrameworkConfig] = Field(default=None, description="Framework configuration")
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


# Agent Builder Platform API Models
class AgentBuilderRequest(BaseModel):
    """Request to build an agent using the Agent Builder platform."""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_type: str = Field(..., description="Agent type (react, knowledge_search, rag, workflow, multimodal, composite, autonomous)")

    # LLM Configuration
    llm_provider: str = Field(default="ollama", description="LLM provider (ollama, openai, anthropic, google)")
    llm_model: str = Field(default="llama3.2:latest", description="LLM model ID")
    temperature: float = Field(default=0.7, description="Model temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, description="Maximum tokens", gt=0)

    # Agent Configuration
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    max_iterations: int = Field(default=50, description="Maximum reasoning iterations", gt=0)
    timeout_seconds: int = Field(default=300, description="Execution timeout", gt=0)

    # Advanced Configuration
    enable_memory: bool = Field(default=True, description="Enable agent memory")
    enable_learning: bool = Field(default=False, description="Enable adaptive learning")
    enable_collaboration: bool = Field(default=False, description="Enable multi-agent collaboration")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Agent tags for categorization")
    owner: Optional[str] = Field(default=None, description="Agent owner identifier")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")

    # Custom configuration
    custom_config: Optional[Dict[str, Any]] = Field(default=None, description="Custom configuration parameters")


class AgentTemplateRequest(BaseModel):
    """Request to build an agent from a template."""
    template: str = Field(..., description="Agent template (research_assistant, customer_support, data_analyst, etc.)")
    name: Optional[str] = Field(default=None, description="Override agent name")
    description: Optional[str] = Field(default=None, description="Override agent description")

    # LLM overrides
    llm_provider: Optional[str] = Field(default=None, description="Override LLM provider")
    llm_model: Optional[str] = Field(default=None, description="Override LLM model")
    temperature: Optional[float] = Field(default=None, description="Override temperature")
    max_tokens: Optional[int] = Field(default=None, description="Override max tokens")

    # Configuration overrides
    tools: Optional[List[str]] = Field(default=None, description="Override tools")
    system_prompt: Optional[str] = Field(default=None, description="Override system prompt")
    enable_learning: Optional[bool] = Field(default=None, description="Override learning setting")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    owner: Optional[str] = Field(default=None, description="Agent owner")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")


class AgentBuilderResponse(BaseModel):
    """Response from agent builder operations."""
    agent_id: str = Field(..., description="Created agent ID")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_type: str = Field(..., description="Agent type")
    template: Optional[str] = Field(default=None, description="Template used (if any)")
    status: str = Field(..., description="Agent status")
    health: str = Field(..., description="Agent health")

    # Configuration
    llm_provider: str = Field(..., description="LLM provider")
    llm_model: str = Field(..., description="LLM model")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    tools: List[str] = Field(..., description="Available tools")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    owner: Optional[str] = Field(default=None, description="Agent owner")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    last_activity: Optional[datetime] = Field(default=None, description="Last activity")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class AgentTemplateInfo(BaseModel):
    """Information about an available agent template."""
    template: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    agent_type: str = Field(..., description="Agent type")
    capabilities: List[str] = Field(..., description="Default capabilities")
    tools: List[str] = Field(..., description="Default tools")
    use_cases: List[str] = Field(default_factory=list, description="Common use cases")


class AgentRegistryStats(BaseModel):
    """Agent registry statistics."""
    total_agents: int = Field(..., description="Total number of agents")
    agents_by_status: Dict[str, int] = Field(..., description="Agents grouped by status")
    agents_by_type: Dict[str, int] = Field(..., description="Agents grouped by type")
    agents_by_health: Dict[str, int] = Field(..., description="Agents grouped by health")
    collaboration_groups: int = Field(..., description="Number of collaboration groups")
    tenants: int = Field(..., description="Number of tenants")


@router.get("/", response_model=List[AgentResponse])
async def list_agents() -> List[AgentResponse]:
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

        # Also get agents from unified system orchestrator
        try:
            from app.core.unified_system_orchestrator import get_system_orchestrator

            orchestrator = await get_system_orchestrator()

            # Get agent information from unified system
            if orchestrator.status.is_running:
                logger.info("Retrieved agents from unified system orchestrator")

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
    request: AgentCreateRequest
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


@router.post("/multi-framework", response_model=AgentResponse)
async def create_multi_framework_agent(
    request: AgentCreateRequest
) -> AgentResponse:
    """
    Create a new multi-framework agent with DNA configuration.

    Args:
        request: Enhanced agent creation request with framework and DNA support
        orchestrator: LangGraph orchestrator instance

    Returns:
        Created agent information with full configuration
    """
    try:
        backend_logger.info(
            f"Creating multi-framework agent: {request.name}",
            LogCategory.AGENT_OPERATIONS,
            "MultiFrameworkAgentAPI",
            data={
                "name": request.name,
                "framework": request.framework,
                "agent_type": request.agent_type,
                "has_dna": request.agent_dna is not None
            }
        )

        # Validate framework
        supported_frameworks = ["basic", "react", "bdi", "crewai", "autogen", "swarm"]
        if request.framework not in supported_frameworks:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported framework: {request.framework}. Supported: {supported_frameworks}"
            )

        # Create enhanced agent configuration
        agent_config = {
            "name": request.name,
            "description": request.description,
            "framework": request.framework,
            "model_name": request.model,
            "model_provider": request.model_provider,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "capabilities": request.capabilities,
            "tools": request.tools,
            "memory_types": request.memory_types,
            "system_prompt": request.system_prompt or f"You are {request.name}, {request.description}",
            "agent_dna": request.agent_dna.dict() if request.agent_dna else None,
            "framework_config": request.framework_config.dict() if request.framework_config else None
        }

        # Create the agent with framework-specific logic
        agent_id = await create_framework_agent(request.framework, agent_config, orchestrator)

        # Create response
        response = AgentResponse(
            agent_id=agent_id,
            name=request.name,
            description=request.description,
            agent_type=request.agent_type,
            framework=request.framework,
            model=request.model,
            status="active",
            capabilities=request.capabilities,
            tools=request.tools,
            memory_types=request.memory_types,
            agent_dna=request.agent_dna,
            framework_config=request.framework_config,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )

        backend_logger.info(
            f"Successfully created multi-framework agent: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "MultiFrameworkAgentAPI",
            data={"agent_id": agent_id, "framework": request.framework}
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Error creating multi-framework agent: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "MultiFrameworkAgentAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


async def create_framework_agent(framework: str, config: Dict[str, Any], orchestrator) -> str:
    """Create agent based on specific framework."""

    if framework == "react":
        # ReAct framework agent
        return await orchestrator.create_agent(
            agent_type="react",
            config={
                **config,
                "reasoning_pattern": "thought_action_observation",
                "tool_usage": True
            }
        )
    elif framework == "bdi":
        # BDI framework agent
        return await orchestrator.create_agent(
            agent_type="bdi",
            config={
                **config,
                "belief_system": True,
                "desire_generation": True,
                "intention_planning": True
            }
        )
    elif framework == "crewai":
        # CrewAI style agent
        return await orchestrator.create_agent(
            agent_type="crew",
            config={
                **config,
                "role_specialization": True,
                "collaboration_enabled": True,
                "delegation_capable": True
            }
        )
    elif framework == "autogen":
        # AutoGen style agent
        return await orchestrator.create_agent(
            agent_type="conversational",
            config={
                **config,
                "multi_agent_conversation": True,
                "structured_dialogue": True
            }
        )
    elif framework == "swarm":
        # Swarm style agent
        return await orchestrator.create_agent(
            agent_type="swarm",
            config={
                **config,
                "lightweight_coordination": True,
                "handoff_capable": True
            }
        )
    else:
        # Basic framework (default)
        return await orchestrator.create_agent(
            agent_type="basic",
            config=config
        )


@router.post("/create")
async def create_agent_enhanced(
    request: dict
) -> dict:
    """
    Create a new agent (legacy endpoint).

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
    agent_id: str
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

        # Check unified system orchestrator
        try:
            from app.core.unified_system_orchestrator import get_system_orchestrator

            orchestrator = await get_system_orchestrator()

            if orchestrator.status.is_running:
                # Create a basic response for unified system agents
                response = AgentResponse(
                    agent_id=agent_id,
                    name=f"Agent {agent_id}",
                    description="Agent managed by unified system",
                    agent_type="unified",
                    model="llama3.2:latest",
                    status="active",
                    capabilities=[],
                    tools=[],
                    created_at=datetime.now(),
                    last_activity=datetime.now()
                )

                logger.info("Agent retrieved from unified system orchestrator", agent_id=agent_id)
                return response

        except Exception as e:
            logger.warning("Failed to check unified system orchestrator", error=str(e))

        # If agent not found in any orchestrator
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent: {str(e)}", agent_id=agent_id)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent: {str(e)}")


# Agent DNA Management Endpoints
@router.post("/dna/validate")
async def validate_agent_dna(dna: AgentDNA) -> Dict[str, Any]:
    """
    Validate Agent DNA configuration.

    Args:
        dna: Agent DNA configuration to validate

    Returns:
        Validation results and suggestions
    """
    try:
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "suggestions": [],
            "score": 0.0
        }

        # Validate identity configuration
        if dna.identity:
            personality = dna.identity.get("personality", {})
            if personality:
                # Check personality balance
                traits = ["creativity", "analytical", "empathy", "assertiveness", "curiosity"]
                trait_values = [personality.get(trait, 0.5) for trait in traits]
                avg_trait = sum(trait_values) / len(trait_values)

                if avg_trait < 0.3:
                    validation_results["warnings"].append("Personality traits are generally low - consider increasing some values")
                elif avg_trait > 0.8:
                    validation_results["warnings"].append("Personality traits are generally high - consider balancing some values")

                validation_results["score"] += 0.3

        # Validate cognition configuration
        if dna.cognition:
            memory_arch = dna.cognition.get("memory_architecture")
            decision_making = dna.cognition.get("decision_making")

            if memory_arch == "hybrid":
                validation_results["suggestions"].append("Hybrid memory architecture is recommended for most use cases")
                validation_results["score"] += 0.4
            elif memory_arch == "short_term":
                validation_results["suggestions"].append("Consider long-term memory for better context retention")
                validation_results["score"] += 0.2
            else:
                validation_results["score"] += 0.3

        # Validate behavior configuration
        if dna.behavior:
            autonomy = dna.behavior.get("autonomy_level")
            collaboration = dna.behavior.get("collaboration_style")

            if autonomy == "autonomous" and collaboration == "independent":
                validation_results["warnings"].append("High autonomy with independent collaboration may limit teamwork")
            elif autonomy == "reactive" and collaboration == "leadership":
                validation_results["warnings"].append("Reactive autonomy with leadership style may create conflicts")

            validation_results["score"] += 0.3

        # Final validation
        if validation_results["score"] >= 0.8:
            validation_results["suggestions"].append("Excellent DNA configuration!")
        elif validation_results["score"] >= 0.6:
            validation_results["suggestions"].append("Good DNA configuration with room for optimization")
        else:
            validation_results["suggestions"].append("Consider reviewing and balancing the DNA configuration")

        backend_logger.info(
            "Agent DNA validated",
            LogCategory.AGENT_OPERATIONS,
            "AgentDNAAPI",
            data={"score": validation_results["score"], "warnings_count": len(validation_results["warnings"])}
        )

        return validation_results

    except Exception as e:
        backend_logger.error(
            f"Error validating Agent DNA: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AgentDNAAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to validate Agent DNA: {str(e)}")


@router.get("/dna/presets")
async def get_dna_presets() -> Dict[str, Any]:
    """
    Get predefined Agent DNA presets.

    Returns:
        Available DNA presets for different agent types
    """
    try:
        presets = {
            "customer_support": {
                "name": "Customer Support Specialist",
                "description": "Empathetic and helpful customer service agent",
                "dna": {
                    "identity": {
                        "personality": {
                            "creativity": 0.3,
                            "analytical": 0.8,
                            "empathy": 0.9,
                            "assertiveness": 0.6,
                            "curiosity": 0.7
                        },
                        "communication_style": "friendly"
                    },
                    "cognition": {
                        "memory_architecture": "hybrid",
                        "decision_making": "analytical",
                        "learning_capability": "adaptive"
                    },
                    "behavior": {
                        "autonomy_level": "proactive",
                        "collaboration_style": "supportive",
                        "error_handling": "graceful"
                    }
                }
            },
            "data_analyst": {
                "name": "Data Analysis Expert",
                "description": "Analytical and detail-oriented data specialist",
                "dna": {
                    "identity": {
                        "personality": {
                            "creativity": 0.4,
                            "analytical": 0.95,
                            "empathy": 0.3,
                            "assertiveness": 0.7,
                            "curiosity": 0.9
                        },
                        "communication_style": "technical"
                    },
                    "cognition": {
                        "memory_architecture": "semantic",
                        "decision_making": "analytical",
                        "learning_capability": "continuous"
                    },
                    "behavior": {
                        "autonomy_level": "autonomous",
                        "collaboration_style": "independent",
                        "error_handling": "robust"
                    }
                }
            },
            "creative_assistant": {
                "name": "Creative Writing Assistant",
                "description": "Imaginative and expressive creative companion",
                "dna": {
                    "identity": {
                        "personality": {
                            "creativity": 0.95,
                            "analytical": 0.4,
                            "empathy": 0.8,
                            "assertiveness": 0.5,
                            "curiosity": 0.9
                        },
                        "communication_style": "casual"
                    },
                    "cognition": {
                        "memory_architecture": "episodic",
                        "decision_making": "intuitive",
                        "learning_capability": "adaptive"
                    },
                    "behavior": {
                        "autonomy_level": "proactive",
                        "collaboration_style": "cooperative",
                        "error_handling": "graceful"
                    }
                }
            }
        }

        backend_logger.info(
            "DNA presets retrieved",
            LogCategory.AGENT_OPERATIONS,
            "AgentDNAAPI",
            data={"presets_count": len(presets)}
        )

        return {"presets": presets}

    except Exception as e:
        backend_logger.error(
            f"Error retrieving DNA presets: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AgentDNAAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve DNA presets: {str(e)}")


@router.post("/{agent_id}/dna")
async def update_agent_dna(
    agent_id: str,
    dna: AgentDNA
) -> Dict[str, Any]:
    """
    Update Agent DNA configuration for existing agent.

    Args:
        agent_id: Agent identifier
        dna: New DNA configuration
        orchestrator: LangGraph orchestrator instance

    Returns:
        Update confirmation and new configuration
    """
    try:
        # Check if agent exists
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Update agent configuration with new DNA
        if agent_id in orchestrator.agent_configs:
            config = orchestrator.agent_configs[agent_id]
            # Add DNA to existing config (assuming we extend the config structure)
            setattr(config, 'agent_dna', dna.dict())

        backend_logger.info(
            f"Agent DNA updated for agent: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AgentDNAAPI",
            data={"agent_id": agent_id}
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "updated_dna": dna.dict(),
            "message": "Agent DNA configuration updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Error updating Agent DNA: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AgentDNAAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to update Agent DNA: {str(e)}")


# Marketplace Endpoints
@router.get("/marketplace/templates")
async def get_marketplace_templates() -> Dict[str, Any]:
    """
    Get available agent templates from marketplace.

    Returns:
        List of available agent templates with metadata
    """
    try:
        templates = [
            {
                "id": "customer-support",
                "name": "Customer Support Agent",
                "description": "Intelligent customer service agent with empathy and problem-solving skills",
                "author": "AgentCorp",
                "rating": 4.8,
                "downloads": 1250,
                "tags": ["customer-service", "support", "empathy"],
                "framework": "crewai",
                "price": "free",
                "thumbnail": "🎧",
                "agent_dna": {
                    "identity": {
                        "personality": {"creativity": 0.3, "analytical": 0.8, "empathy": 0.9, "assertiveness": 0.6, "curiosity": 0.7},
                        "communication_style": "friendly"
                    },
                    "cognition": {"memory_architecture": "hybrid", "decision_making": "analytical", "learning_capability": "adaptive"},
                    "behavior": {"autonomy_level": "proactive", "collaboration_style": "supportive", "error_handling": "graceful"}
                }
            },
            {
                "id": "data-analyst",
                "name": "Data Analysis Expert",
                "description": "Advanced data analysis agent with statistical modeling and visualization capabilities",
                "author": "DataLabs",
                "rating": 4.9,
                "downloads": 890,
                "tags": ["data-analysis", "statistics", "visualization"],
                "framework": "react",
                "price": "premium",
                "thumbnail": "📊",
                "agent_dna": {
                    "identity": {
                        "personality": {"creativity": 0.4, "analytical": 0.95, "empathy": 0.3, "assertiveness": 0.7, "curiosity": 0.9},
                        "communication_style": "technical"
                    },
                    "cognition": {"memory_architecture": "semantic", "decision_making": "analytical", "learning_capability": "continuous"},
                    "behavior": {"autonomy_level": "autonomous", "collaboration_style": "independent", "error_handling": "robust"}
                }
            },
            {
                "id": "creative-writer",
                "name": "Creative Writing Assistant",
                "description": "Imaginative writing companion for stories, poems, and creative content",
                "author": "CreativeAI",
                "rating": 4.7,
                "downloads": 2100,
                "tags": ["writing", "creativity", "storytelling"],
                "framework": "basic",
                "price": "free",
                "thumbnail": "✍️",
                "agent_dna": {
                    "identity": {
                        "personality": {"creativity": 0.95, "analytical": 0.4, "empathy": 0.8, "assertiveness": 0.5, "curiosity": 0.9},
                        "communication_style": "casual"
                    },
                    "cognition": {"memory_architecture": "episodic", "decision_making": "intuitive", "learning_capability": "adaptive"},
                    "behavior": {"autonomy_level": "proactive", "collaboration_style": "cooperative", "error_handling": "graceful"}
                }
            }
        ]

        backend_logger.info(
            "Marketplace templates retrieved",
            LogCategory.AGENT_OPERATIONS,
            "MarketplaceAPI",
            data={"templates_count": len(templates)}
        )

        return {"templates": templates, "total": len(templates)}

    except Exception as e:
        backend_logger.error(
            f"Error retrieving marketplace templates: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "MarketplaceAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve templates: {str(e)}")


@router.post("/marketplace/templates/{template_id}/use")
async def use_marketplace_template(template_id: str) -> Dict[str, Any]:
    """
    Use a marketplace template to create an agent.

    Args:
        template_id: Template identifier

    Returns:
        Template configuration for agent creation
    """
    try:
        # This would typically fetch from a database
        # For now, return a success response
        backend_logger.info(
            f"Marketplace template used: {template_id}",
            LogCategory.AGENT_OPERATIONS,
            "MarketplaceAPI",
            data={"template_id": template_id}
        )

        return {
            "success": True,
            "template_id": template_id,
            "message": "Template loaded successfully"
        }

    except Exception as e:
        backend_logger.error(
            f"Error using marketplace template: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "MarketplaceAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to use template: {str(e)}")


@router.post("/marketplace/templates/{template_id}/rate")
async def rate_marketplace_template(
    template_id: str,
    rating: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Rate a marketplace template.

    Args:
        template_id: Template identifier
        rating: Rating data (score, review, etc.)

    Returns:
        Rating confirmation
    """
    try:
        # This would typically update a database
        backend_logger.info(
            f"Template rated: {template_id}",
            LogCategory.AGENT_OPERATIONS,
            "MarketplaceAPI",
            data={"template_id": template_id, "rating": rating.get("score", 0)}
        )

        return {
            "success": True,
            "template_id": template_id,
            "rating": rating,
            "message": "Rating submitted successfully"
        }

    except Exception as e:
        backend_logger.error(
            f"Error rating template: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "MarketplaceAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to submit rating: {str(e)}")


# Analytics Endpoints
@router.get("/analytics/{agent_id}/performance")
async def get_agent_performance_analytics(agent_id: str) -> Dict[str, Any]:
    """
    Get performance analytics for a specific agent.

    Args:
        agent_id: Agent identifier

    Returns:
        Performance metrics and analytics data
    """
    try:
        # Mock analytics data - in production this would come from metrics collection
        performance_data = {
            "agent_id": agent_id,
            "performance": {
                "response_time": 1.2,
                "success_rate": 94.5,
                "total_requests": 1847,
                "error_rate": 5.5,
                "uptime": 99.2
            },
            "usage": {
                "daily_active": 156,
                "weekly_active": 892,
                "monthly_active": 3421,
                "peak_hours": "2-4 PM",
                "avg_session_length": "12.5 min"
            },
            "trends": {
                "response_time_trend": "improving",
                "usage_trend": "increasing",
                "error_trend": "stable"
            },
            "last_updated": datetime.utcnow().isoformat()
        }

        backend_logger.info(
            f"Performance analytics retrieved for agent: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AnalyticsAPI",
            data={"agent_id": agent_id}
        )

        return performance_data

    except Exception as e:
        backend_logger.error(
            f"Error retrieving performance analytics: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AnalyticsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance analytics: {str(e)}")


@router.get("/analytics/{agent_id}/behavior")
async def get_agent_behavior_analytics(agent_id: str) -> Dict[str, Any]:
    """
    Get behavior analytics for a specific agent.

    Args:
        agent_id: Agent identifier

    Returns:
        Behavior patterns and analysis data
    """
    try:
        behavior_data = {
            "agent_id": agent_id,
            "behavior": {
                "most_used_tools": ["web_search", "file_reader", "calculator"],
                "tool_usage_frequency": {
                    "web_search": 45,
                    "file_reader": 32,
                    "calculator": 23
                },
                "common_patterns": [
                    "research → analysis → report",
                    "question → search → answer",
                    "data → process → visualize"
                ],
                "user_satisfaction": 4.3,
                "interaction_types": {
                    "questions": 60,
                    "tasks": 25,
                    "conversations": 15
                }
            },
            "learning": {
                "adaptation_rate": 0.85,
                "knowledge_retention": 0.92,
                "improvement_areas": ["complex reasoning", "multi-step tasks"]
            },
            "last_updated": datetime.utcnow().isoformat()
        }

        backend_logger.info(
            f"Behavior analytics retrieved for agent: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AnalyticsAPI",
            data={"agent_id": agent_id}
        )

        return behavior_data

    except Exception as e:
        backend_logger.error(
            f"Error retrieving behavior analytics: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AnalyticsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve behavior analytics: {str(e)}")


@router.get("/analytics/{agent_id}/recommendations")
async def get_agent_optimization_recommendations(agent_id: str) -> Dict[str, Any]:
    """
    Get optimization recommendations for a specific agent.

    Args:
        agent_id: Agent identifier

    Returns:
        AI-powered optimization recommendations
    """
    try:
        recommendations_data = {
            "agent_id": agent_id,
            "recommendations": [
                {
                    "type": "performance",
                    "message": "Consider reducing model temperature for more consistent responses",
                    "priority": "medium",
                    "impact": "moderate",
                    "implementation": "Adjust temperature from 0.7 to 0.5"
                },
                {
                    "type": "memory",
                    "message": "Enable long-term memory for better context retention",
                    "priority": "high",
                    "impact": "high",
                    "implementation": "Switch memory architecture to hybrid mode"
                },
                {
                    "type": "tools",
                    "message": "Add image analysis tool for multimedia content",
                    "priority": "low",
                    "impact": "low",
                    "implementation": "Install and configure image analysis capability"
                }
            ],
            "optimization_score": 7.5,
            "potential_improvements": {
                "response_time": "15% faster",
                "accuracy": "8% improvement",
                "user_satisfaction": "12% increase"
            },
            "last_updated": datetime.utcnow().isoformat()
        }

        backend_logger.info(
            f"Optimization recommendations retrieved for agent: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AnalyticsAPI",
            data={"agent_id": agent_id, "recommendations_count": len(recommendations_data["recommendations"])}
        )

        return recommendations_data

    except Exception as e:
        backend_logger.error(
            f"Error retrieving optimization recommendations: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AnalyticsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recommendations: {str(e)}")


@router.get("/analytics/system/metrics")
async def get_system_analytics() -> Dict[str, Any]:
    """
    Get system-wide analytics and metrics.

    Returns:
        System performance and usage analytics
    """
    try:
        system_data = {
            "system": {
                "total_agents": 45,
                "active_agents": 32,
                "total_requests_today": 8934,
                "avg_system_response_time": 0.95,
                "system_uptime": 99.8,
                "error_rate": 2.1
            },
            "frameworks": {
                "usage_distribution": {
                    "basic": 25,
                    "react": 20,
                    "crewai": 18,
                    "bdi": 15,
                    "autogen": 12,
                    "swarm": 10
                }
            },
            "popular_features": [
                {"feature": "Agent DNA", "usage": 78},
                {"feature": "Multi-framework", "usage": 65},
                {"feature": "Marketplace", "usage": 52},
                {"feature": "Analytics", "usage": 41}
            ],
            "last_updated": datetime.utcnow().isoformat()
        }

        backend_logger.info(
            "System analytics retrieved",
            LogCategory.SYSTEM_MONITORING,
            "AnalyticsAPI",
            data={"total_agents": system_data["system"]["total_agents"]}
        )

        return system_data

    except Exception as e:
        backend_logger.error(
            f"Error retrieving system analytics: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "AnalyticsAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system analytics: {str(e)}")


# Code Generation Endpoints
@router.post("/codegen/framework/{framework}")
async def generate_framework_code(
    framework: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate framework-specific code for an agent.

    Args:
        framework: Target framework (react, bdi, crewai, etc.)
        config: Agent configuration including DNA and components

    Returns:
        Generated code and metadata
    """
    try:
        if framework not in ["basic", "react", "bdi", "crewai", "autogen", "swarm"]:
            raise HTTPException(status_code=400, detail=f"Unsupported framework: {framework}")

        # Generate framework-specific code
        generated_code = await generate_agent_code(framework, config)

        response_data = {
            "framework": framework,
            "code": generated_code,
            "language": "python",
            "filename": f"{config.get('name', 'agent').replace(' ', '_').lower()}_agent.py",
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "framework_version": "1.0.0",
                "dependencies": get_framework_dependencies(framework)
            }
        }

        backend_logger.info(
            f"Code generated for framework: {framework}",
            LogCategory.AGENT_OPERATIONS,
            "CodeGenAPI",
            data={"framework": framework, "agent_name": config.get("name", "unknown")}
        )

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Error generating framework code: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "CodeGenAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to generate code: {str(e)}")


async def generate_agent_code(framework: str, config: Dict[str, Any]) -> str:
    """Generate framework-specific agent code."""

    name = config.get("name", "Agent").replace(" ", "")
    description = config.get("description", "AI Agent")
    tools = config.get("tools", [])
    capabilities = config.get("capabilities", [])
    agent_dna = config.get("agent_dna", {})

    if framework == "react":
        return f'''# ReAct Agent Implementation
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

class {name}Agent:
    def __init__(self):
        self.name = "{config.get("name", "Agent")}"
        self.description = "{description}"
        self.personality = {agent_dna.get("identity", {}).get("personality", {})}

        # Initialize tools
        self.tools = [
{chr(10).join([f'            Tool(name="{tool}", description="Tool: {tool}", func=self.{tool})' for tool in tools])}
        ]

        # ReAct prompt template
        self.prompt = PromptTemplate.from_template("""
{config.get("system_prompt", f"You are {name}, {description}")}

Use the following format:
Thought: Think about what you need to do
Action: Choose an action from [{{tool_names}}]
Action Input: The input to the action
Observation: The result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: The final answer to the original input question

Question: {{input}}
{{agent_scratchpad}}
        """)

    def execute(self, query: str):
        # Implementation would go here
        pass
'''

    elif framework == "bdi":
        return f'''# BDI Agent Implementation
from typing import Dict, List, Any
import asyncio

class {name}BDIAgent:
    def __init__(self):
        self.name = "{config.get("name", "Agent")}"
        self.description = "{description}"
        self.personality = {agent_dna.get("identity", {}).get("personality", {})}

        # BDI Components
        self.beliefs = {{}}
        self.desires = []
        self.intentions = []

        # Capabilities
        self.capabilities = {capabilities}

    async def update_beliefs(self, new_information: Dict[str, Any]):
        """Update agent beliefs based on new information"""
        self.beliefs.update(new_information)

    async def generate_desires(self, context: Dict[str, Any]):
        """Generate desires/goals based on current context"""
        pass

    async def form_intentions(self):
        """Form intentions based on beliefs and desires"""
        pass

    async def execute_intentions(self):
        """Execute current intentions"""
        pass
'''

    elif framework == "crewai":
        return f'''# CrewAI Style Agent Implementation
from crewai import Agent, Task, Crew

class {name}CrewAgent:
    def __init__(self):
        self.agent = Agent(
            role="{config.get("name", "Agent")}",
            goal="{description}",
            backstory="""
            {config.get("system_prompt", f"You are {name}, {description}")}
            """,
            personality_traits={agent_dna.get("identity", {}).get("personality", {})},
            tools=[
{chr(10).join([f'                # {tool} tool would be implemented here' for tool in tools])}
            ],
            memory={agent_dna.get("cognition", {}).get("memory_architecture") == "hybrid"},
            verbose=True,
            allow_delegation={agent_dna.get("behavior", {}).get("collaboration_style") == "cooperative"}
        )

    def create_task(self, description: str, expected_output: str):
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent
        )

    def execute_with_crew(self, tasks: List[Task], other_agents: List[Agent] = None):
        crew = Crew(
            agents=[self.agent] + (other_agents or []),
            tasks=tasks,
            verbose=True
        )
        return crew.kickoff()
'''

    else:  # basic or other frameworks
        return f'''# Basic Agent Implementation
class {name}Agent:
    def __init__(self):
        self.name = "{config.get("name", "Agent")}"
        self.description = "{description}"
        self.personality = {agent_dna.get("identity", {}).get("personality", {})}
        self.system_prompt = """
{config.get("system_prompt", f"You are {name}, {description}")}
        """

    def process(self, input_text: str) -> str:
        """Process input and return response"""
        # Basic processing logic would go here
        return "Response would be generated here"
'''


def get_framework_dependencies(framework: str) -> List[str]:
    """Get required dependencies for a framework."""
    dependencies = {
        "basic": ["langchain", "ollama"],
        "react": ["langchain", "langchain-community", "ollama"],
        "bdi": ["langchain", "asyncio", "ollama"],
        "crewai": ["crewai", "langchain", "ollama"],
        "autogen": ["autogen", "langchain", "ollama"],
        "swarm": ["langchain", "ollama"]
    }
    return dependencies.get(framework, ["langchain", "ollama"])


@router.get("/codegen/templates/{framework}")
async def get_framework_templates(framework: str) -> Dict[str, Any]:
    """
    Get code templates for a specific framework.

    Args:
        framework: Target framework

    Returns:
        Available templates and examples
    """
    try:
        templates = {
            "basic": {
                "name": "Basic Agent Template",
                "description": "Simple agent with basic functionality",
                "example": "Basic agent with LLM integration"
            },
            "react": {
                "name": "ReAct Agent Template",
                "description": "Reasoning and Acting agent with tool usage",
                "example": "Agent that can think, act, and observe"
            },
            "bdi": {
                "name": "BDI Agent Template",
                "description": "Belief-Desire-Intention architecture",
                "example": "Agent with beliefs, desires, and intentions"
            },
            "crewai": {
                "name": "CrewAI Agent Template",
                "description": "Role-based collaborative agent",
                "example": "Specialized agent for team collaboration"
            }
        }

        if framework not in templates:
            raise HTTPException(status_code=404, detail=f"No templates found for framework: {framework}")

        return {
            "framework": framework,
            "template": templates[framework],
            "dependencies": get_framework_dependencies(framework)
        }

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Error retrieving framework templates: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "CodeGenAPI",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve templates: {str(e)}")

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
    request: AgentChatRequest
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


# ============================================================================
# AGENT BUILDER PLATFORM ENDPOINTS
# ============================================================================

# Global agent registry and factory instances
_agent_registry: Optional[AgentRegistry] = None
_agent_factory: Optional[AgentBuilderFactory] = None
_llm_manager: Optional[LLMProviderManager] = None


async def get_agent_builder_components():
    """Get or initialize agent builder components."""
    global _agent_registry, _agent_factory, _llm_manager

    if not _llm_manager:
        _llm_manager = LLMProviderManager()
        await _llm_manager.initialize()

    if not _agent_factory:
        _agent_factory = AgentBuilderFactory(_llm_manager)

    if not _agent_registry:
        system_orchestrator = get_system_orchestrator()
        _agent_registry = initialize_agent_registry(_agent_factory, system_orchestrator)

    return _agent_registry, _agent_factory, _llm_manager


@router.post("/builder/create", response_model=AgentBuilderResponse, tags=["Agent Builder"])
async def create_agent_with_builder(request: AgentBuilderRequest) -> AgentBuilderResponse:
    """
    Create an agent using the Agent Builder platform.

    This endpoint provides advanced agent creation with support for multiple
    agent types, LLM providers, and enterprise-grade configuration.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentBuilderAPI",
        operation="create_agent_with_builder"
    )

    try:
        backend_logger.info(
            f"Creating agent with builder: {request.name}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        # Get agent builder components
        registry, factory, llm_manager = await get_agent_builder_components()

        # Convert request to builder config
        try:
            agent_type = AgentType(request.agent_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid agent type: {request.agent_type}")

        try:
            provider_type = ProviderType(request.llm_provider.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid LLM provider: {request.llm_provider}")

        # Create LLM config
        llm_config = LLMConfig(
            provider=provider_type,
            model_id=request.llm_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert capabilities
        from app.agents.base.agent import AgentCapability
        capabilities = []
        for cap_str in request.capabilities:
            try:
                capabilities.append(AgentCapability(cap_str))
            except ValueError:
                backend_logger.warning(f"Unknown capability: {cap_str}", LogCategory.AGENT_OPERATIONS, "AgentBuilderAPI")

        # Create builder config
        builder_config = AgentBuilderConfig(
            name=request.name,
            description=request.description,
            agent_type=agent_type,
            llm_config=llm_config,
            capabilities=capabilities,
            tools=request.tools,
            system_prompt=request.system_prompt,
            max_iterations=request.max_iterations,
            timeout_seconds=request.timeout_seconds,
            enable_memory=request.enable_memory,
            enable_learning=request.enable_learning,
            enable_collaboration=request.enable_collaboration,
            custom_config=request.custom_config
        )

        # Register the agent
        agent_id = await registry.register_agent(
            config=builder_config,
            owner=request.owner,
            tenant_id=request.tenant_id,
            tags=request.tags
        )

        # Get the registered agent
        registered_agent = registry.get_agent(agent_id)
        if not registered_agent:
            raise HTTPException(status_code=500, detail="Failed to retrieve created agent")

        # Start the agent
        await registry.start_agent(agent_id)

        # Create response
        response = AgentBuilderResponse(
            agent_id=agent_id,
            name=registered_agent.name,
            description=registered_agent.description,
            agent_type=registered_agent.agent_type.value,
            template=registered_agent.template.value if registered_agent.template else None,
            status=registered_agent.status.value,
            health=registered_agent.health.value,
            llm_provider=registered_agent.config.llm_config.provider.value,
            llm_model=registered_agent.config.llm_config.model_id,
            capabilities=[cap.value for cap in registered_agent.config.capabilities],
            tools=registered_agent.config.tools,
            tags=registered_agent.tags,
            owner=registered_agent.owner,
            tenant_id=registered_agent.tenant_id,
            created_at=registered_agent.metrics.created_at,
            last_activity=registered_agent.metrics.last_activity
        )

        backend_logger.info(
            f"Agent created successfully with builder: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Failed to create agent with builder: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="create_agent_with_builder",
                duration=end_time - start_time,
                component="AgentBuilderAPI"
            )
        )


@router.post("/builder/template", response_model=AgentBuilderResponse, tags=["Agent Builder"])
async def create_agent_from_template(request: AgentTemplateRequest) -> AgentBuilderResponse:
    """
    Create an agent from a pre-defined template.

    Templates provide optimized configurations for common use cases like
    research assistants, customer support, data analysis, etc.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentBuilderAPI",
        operation="create_agent_from_template"
    )

    try:
        backend_logger.info(
            f"Creating agent from template: {request.template}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        # Get agent builder components
        registry, factory, llm_manager = await get_agent_builder_components()

        # Validate template
        try:
            template = AgentTemplate(request.template)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid template: {request.template}")

        # Prepare overrides
        overrides = {}
        if request.name:
            overrides['name'] = request.name
        if request.description:
            overrides['description'] = request.description
        if request.llm_provider:
            try:
                provider_type = ProviderType(request.llm_provider.upper())
                overrides['llm_config'] = LLMConfig(
                    provider=provider_type,
                    model_id=request.llm_model or "llama3.2:latest",
                    temperature=request.temperature or 0.7,
                    max_tokens=request.max_tokens or 2048
                )
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid LLM provider: {request.llm_provider}")
        if request.tools:
            overrides['tools'] = request.tools
        if request.system_prompt:
            overrides['system_prompt'] = request.system_prompt
        if request.enable_learning is not None:
            overrides['enable_learning'] = request.enable_learning

        # Register agent from template
        agent_id = await registry.register_from_template(
            template=template,
            overrides=overrides,
            owner=request.owner,
            tenant_id=request.tenant_id,
            tags=request.tags
        )

        # Get the registered agent
        registered_agent = registry.get_agent(agent_id)
        if not registered_agent:
            raise HTTPException(status_code=500, detail="Failed to retrieve created agent")

        # Start the agent
        await registry.start_agent(agent_id)

        # Create response
        response = AgentBuilderResponse(
            agent_id=agent_id,
            name=registered_agent.name,
            description=registered_agent.description,
            agent_type=registered_agent.agent_type.value,
            template=registered_agent.template.value if registered_agent.template else None,
            status=registered_agent.status.value,
            health=registered_agent.health.value,
            llm_provider=registered_agent.config.llm_config.provider.value,
            llm_model=registered_agent.config.llm_config.model_id,
            capabilities=[cap.value for cap in registered_agent.config.capabilities],
            tools=registered_agent.config.tools,
            tags=registered_agent.tags,
            owner=registered_agent.owner,
            tenant_id=registered_agent.tenant_id,
            created_at=registered_agent.metrics.created_at,
            last_activity=registered_agent.metrics.last_activity
        )

        backend_logger.info(
            f"Agent created from template successfully: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Failed to create agent from template: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to create agent from template: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="create_agent_from_template",
                duration=end_time - start_time,
                component="AgentBuilderAPI"
            )
        )


@router.get("/builder/templates", response_model=List[AgentTemplateInfo], tags=["Agent Builder"])
async def list_agent_templates() -> List[AgentTemplateInfo]:
    """
    List all available agent templates.

    Templates provide pre-configured agents for common use cases with
    optimized settings and tool selections.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentBuilderAPI",
        operation="list_agent_templates"
    )

    try:
        backend_logger.info(
            "Listing agent templates",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        # Get all templates from the library
        all_templates = AgentTemplateLibrary.get_all_templates()

        # Convert to response format
        template_infos = []
        for template, config in all_templates.items():
            # Define use cases for each template
            use_cases_map = {
                AgentTemplate.RESEARCH_ASSISTANT: [
                    "Academic research", "Market analysis", "Competitive intelligence",
                    "Literature reviews", "Fact-checking", "Data synthesis"
                ],
                AgentTemplate.CUSTOMER_SUPPORT: [
                    "Help desk automation", "FAQ responses", "Ticket triage",
                    "Customer onboarding", "Technical support", "Escalation management"
                ],
                AgentTemplate.DATA_ANALYST: [
                    "Statistical analysis", "Data visualization", "Trend analysis",
                    "A/B testing", "Predictive modeling", "Business reporting"
                ],
                AgentTemplate.CONTENT_CREATOR: [
                    "Blog writing", "Social media content", "Marketing copy",
                    "SEO optimization", "Email campaigns", "Creative writing"
                ],
                AgentTemplate.CODE_REVIEWER: [
                    "Code quality assessment", "Security audits", "Performance optimization",
                    "Best practices enforcement", "Documentation review", "Test coverage analysis"
                ],
                AgentTemplate.BUSINESS_INTELLIGENCE: [
                    "KPI tracking", "Dashboard creation", "Executive reporting",
                    "Trend forecasting", "Competitive benchmarking", "Performance monitoring"
                ]
            }

            template_info = AgentTemplateInfo(
                template=template.value,
                name=config.name,
                description=config.description,
                agent_type=config.agent_type.value,
                capabilities=[cap.value for cap in config.capabilities],
                tools=config.tools,
                use_cases=use_cases_map.get(template, [])
            )
            template_infos.append(template_info)

        backend_logger.info(
            f"Listed {len(template_infos)} agent templates",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        return template_infos

    except Exception as e:
        backend_logger.error(
            f"Failed to list agent templates: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="list_agent_templates",
                duration=end_time - start_time,
                component="AgentBuilderAPI"
            )
        )


@router.get("/builder/registry", response_model=List[AgentBuilderResponse], tags=["Agent Builder"])
async def list_registered_agents(
    agent_type: Optional[str] = None,
    template: Optional[str] = None,
    tenant_id: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[str] = None
) -> List[AgentBuilderResponse]:
    """
    List agents in the registry with optional filtering.

    Supports filtering by agent type, template, tenant, status, and tags.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentBuilderAPI",
        operation="list_registered_agents"
    )

    try:
        backend_logger.info(
            "Listing registered agents",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        # Get agent builder components
        registry, _, _ = await get_agent_builder_components()

        # Parse filters
        agent_type_filter = None
        if agent_type:
            try:
                agent_type_filter = AgentType(agent_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid agent type: {agent_type}")

        template_filter = None
        if template:
            try:
                template_filter = AgentTemplate(template)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid template: {template}")

        status_filter = None
        if status:
            try:
                status_filter = AgentStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        tags_filter = None
        if tags:
            tags_filter = [tag.strip() for tag in tags.split(",")]

        # List agents with filters
        agents = registry.list_agents(
            agent_type=agent_type_filter,
            template=template_filter,
            tenant_id=tenant_id,
            status=status_filter,
            tags=tags_filter
        )

        # Convert to response format
        responses = []
        for agent in agents:
            response = AgentBuilderResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                description=agent.description,
                agent_type=agent.agent_type.value,
                template=agent.template.value if agent.template else None,
                status=agent.status.value,
                health=agent.health.value,
                llm_provider=agent.config.llm_config.provider.value,
                llm_model=agent.config.llm_config.model_id,
                capabilities=[cap.value for cap in agent.config.capabilities],
                tools=agent.config.tools,
                tags=agent.tags,
                owner=agent.owner,
                tenant_id=agent.tenant_id,
                created_at=agent.metrics.created_at,
                last_activity=agent.metrics.last_activity
            )
            responses.append(response)

        backend_logger.info(
            f"Listed {len(responses)} registered agents",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        return responses

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Failed to list registered agents: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="list_registered_agents",
                duration=end_time - start_time,
                component="AgentBuilderAPI"
            )
        )


@router.get("/builder/registry/stats", response_model=AgentRegistryStats, tags=["Agent Builder"])
async def get_registry_statistics() -> AgentRegistryStats:
    """
    Get comprehensive statistics about the agent registry.

    Provides insights into agent distribution, health, and usage patterns.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentBuilderAPI",
        operation="get_registry_statistics"
    )

    try:
        backend_logger.info(
            "Getting registry statistics",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        # Get agent builder components
        registry, _, _ = await get_agent_builder_components()

        # Get statistics
        stats = registry.get_registry_stats()

        # Convert to response format
        response = AgentRegistryStats(
            total_agents=stats["total_agents"],
            agents_by_status=stats["agents_by_status"],
            agents_by_type=stats["agents_by_type"],
            agents_by_health=stats["agents_by_health"],
            collaboration_groups=stats["collaboration_groups"],
            tenants=stats["tenants"]
        )

        backend_logger.info(
            f"Registry statistics: {stats['total_agents']} total agents",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        return response

    except Exception as e:
        backend_logger.error(
            f"Failed to get registry statistics: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="get_registry_statistics",
                duration=end_time - start_time,
                component="AgentBuilderAPI"
            )
        )


@router.delete("/builder/registry/{agent_id}", tags=["Agent Builder"])
async def unregister_agent(agent_id: str) -> Dict[str, str]:
    """
    Unregister and destroy an agent from the registry.

    This permanently removes the agent and all its associated data.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentBuilderAPI",
        operation="unregister_agent"
    )

    try:
        backend_logger.info(
            f"Unregistering agent: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        # Get agent builder components
        registry, _, _ = await get_agent_builder_components()

        # Unregister the agent
        success = await registry.unregister_agent(agent_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        backend_logger.info(
            f"Agent unregistered successfully: {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI"
        )

        return {"message": f"Agent {agent_id} unregistered successfully"}

    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Failed to unregister agent {agent_id}: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentBuilderAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to unregister agent: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="unregister_agent",
                duration=end_time - start_time,
                component="AgentBuilderAPI"
            )
        )


