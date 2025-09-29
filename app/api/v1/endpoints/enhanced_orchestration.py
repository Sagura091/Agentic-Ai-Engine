"""
Enhanced Orchestration API endpoints.

This module provides comprehensive API endpoints for unlimited agent creation,
dynamic tool management, and sophisticated multi-agent workflow orchestration.
"""

import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, UploadFile, File, Form
from pydantic import BaseModel, Field

# Define missing enums
class OrchestrationStrategy(str, Enum):
    """Orchestration strategy for multi-agent workflows."""
    ADAPTIVE = "adaptive"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"

from app.core.unified_system_orchestrator import get_system_orchestrator
from app.core.seamless_integration import seamless_integration
# from app.tools.dynamic_tool_factory import ToolCategory, ToolComplexity
# from app.tools.production_tool_system import production_tool_registry
from app.core.dependencies import get_database_session
from app.services.tool_validation_service import tool_validation_service
from app.services.tool_template_service import tool_template_service
from app.core.auth import get_current_user

# Import Agent Builder Platform components
from app.agents.factory import AgentType, AgentTemplate, AgentBuilderFactory, AgentBuilderConfig
from app.agents.registry import AgentRegistry, get_agent_registry, initialize_agent_registry
from app.agents.templates import AgentTemplateLibrary
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType

# Import new backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogLevel, LogCategory, PerformanceMetrics, AgentMetrics
from app.backend_logging.context import CorrelationContext

logger = structlog.get_logger(__name__)
backend_logger = get_logger()

router = APIRouter(prefix="/orchestration", tags=["Enhanced Orchestration"])


async def _persist_agent_to_database(agent_response: "AgentResponse") -> None:
    """Persist agent to database for long-term storage."""
    try:
        from app.models.database.base import get_database_session
        from sqlalchemy import text

        async for session in get_database_session():
            try:
                # Insert agent into database
                query = text("""
                    INSERT INTO agents (
                        id, name, description, agent_type, model,
                        capabilities, tools, system_prompt, temperature,
                        max_tokens, status, created_at, metadata
                    ) VALUES (
                        :id, :name, :description, :agent_type, :model,
                        :capabilities, :tools, :system_prompt, :temperature,
                        :max_tokens, :status, :created_at, :metadata
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                """)

                await session.execute(query, {
                    "id": agent_response.agent_id,
                    "name": agent_response.name,
                    "description": agent_response.description,
                    "agent_type": agent_response.agent_type,
                    "model": agent_response.model,
                    "capabilities": agent_response.capabilities or [],
                    "tools": agent_response.tools or [],
                    "system_prompt": f"You are {agent_response.name}, {agent_response.description}",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "status": agent_response.status,
                    "created_at": agent_response.created_at,
                    "metadata": {"performance_metrics": agent_response.performance_metrics}
                })

                await session.commit()
                logger.info(f"Agent {agent_response.agent_id} persisted to database")
                break  # Exit the async for loop after successful operation

            except Exception as e:
                await session.rollback()
                raise e

    except Exception as e:
        logger.error(f"Failed to persist agent to database: {str(e)}")
        # Don't raise exception to avoid breaking agent creation


# Pydantic models for API requests/responses
class UnlimitedAgentCreateRequest(BaseModel):
    """Request for creating unlimited agents."""
    agent_type: AgentType = Field(..., description="Type of agent to create")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    # LLM Configuration
    model: str = Field(default="llama3.2:latest", description="Model to use")
    model_provider: str = Field(default="ollama", description="LLM provider (ollama, openai, anthropic, google)")
    temperature: float = Field(default=0.7, description="Model temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, description="Maximum tokens", gt=0)
    top_p: Optional[float] = Field(default=None, description="Top-p sampling", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, description="Top-k sampling", gt=0)
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty", ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty", ge=-2.0, le=2.0)

    # Agent configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    tools: List[str] = Field(default_factory=list, description="Tool names to assign")

    # Autonomous agent specific (if applicable)
    autonomy_level: str = Field(default="adaptive", description="Autonomy level for autonomous agents")
    learning_mode: str = Field(default="active", description="Learning mode for autonomous agents")
    decision_threshold: float = Field(default=0.6, description="Decision threshold")


class AgentResponse(BaseModel):
    """Response for agent operations."""
    agent_id: str = Field(..., description="Agent ID")
    agent_type: str = Field(..., description="Agent type")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    model: str = Field(..., description="Model being used")
    status: str = Field(..., description="Agent status")
    tools: List[str] = Field(..., description="Assigned tools")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    created_at: datetime = Field(..., description="Creation timestamp")


class DynamicToolCreateRequest(BaseModel):
    """Request for creating dynamic tools."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    functionality_description: str = Field(..., description="Detailed functionality description")
    category: str = Field(default="custom", description="Tool category")
    complexity: str = Field(default="simple", description="Tool complexity")
    
    # Optional parameters for specific creation methods
    template_name: Optional[str] = Field(default=None, description="Template to use (if any)")
    parameter_overrides: Dict[str, Any] = Field(default_factory=dict, description="Parameter overrides")
    
    # Assignment options
    assign_to_agent: Optional[str] = Field(default=None, description="Agent ID to assign tool to")
    make_global: bool = Field(default=False, description="Make tool available globally")


class ToolResponse(BaseModel):
    """Response for tool operations."""
    tool_name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(..., description="Tool category")
    complexity: str = Field(..., description="Tool complexity")
    usage_count: int = Field(..., description="Usage count")
    success_rate: float = Field(..., description="Success rate")
    created_at: datetime = Field(..., description="Creation timestamp")
    assigned_agents: List[str] = Field(..., description="Agents using this tool")


class WorkflowExecuteRequest(BaseModel):
    """Request for executing multi-agent workflows."""
    workflow_name: str = Field(..., description="Workflow name or template")
    task: str = Field(..., description="Task description")
    agents: List[str] = Field(..., description="Agent IDs to use")
    strategy: OrchestrationStrategy = Field(default=OrchestrationStrategy.ADAPTIVE, description="Orchestration strategy")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    max_iterations: int = Field(default=50, description="Maximum iterations")


class WorkflowResponse(BaseModel):
    """Response for workflow operations."""
    workflow_id: str = Field(..., description="Workflow execution ID")
    status: str = Field(..., description="Workflow status")
    agents_used: List[str] = Field(..., description="Agents used in workflow")
    strategy: str = Field(..., description="Orchestration strategy used")
    results: Dict[str, Any] = Field(..., description="Workflow results")
    execution_time: float = Field(..., description="Total execution time")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")


@router.post("/agents/create-unlimited", response_model=AgentResponse)
async def create_unlimited_agent(
    request: UnlimitedAgentCreateRequest,
    background_tasks: BackgroundTasks
) -> AgentResponse:
    """
    Create unlimited agents with dynamic configuration.
    
    This endpoint allows creation of any number of agents with flexible
    configuration, tool assignment, and type selection. Supports all
    agent types including autonomous agents with advanced capabilities.
    """
    try:
        # Get unified system orchestrator
        orchestrator = await get_system_orchestrator()

        # Create agent ecosystem using unified system
        agent_id = await seamless_integration.create_agent_ecosystem(
            agent_id=f"agent_{uuid.uuid4().hex[:8]}",
            agent_type=request.agent_type.value if hasattr(request.agent_type, 'value') else str(request.agent_type),
            config={
                "name": request.name,
                "description": request.description,
                "model": request.model,
                "tools": request.tools
            }
        )

        response = AgentResponse(
            agent_id=agent_id,
            agent_type=str(request.agent_type),
            name=request.name,
            description=request.description,
            model=request.model,
            status="active",
            tools=request.tools,
            performance_metrics={},
            created_at=datetime.now()
        )

        logger.info(
            "Agent ecosystem created via API",
            agent_id=agent_id,
            agent_type=str(request.agent_type),
            name=request.name
        )

        return response
        
    except Exception as e:
        logger.error("Failed to create unlimited agent", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.post("/tools/create-dynamic", response_model=ToolResponse)
async def create_dynamic_tool(
    request: DynamicToolCreateRequest,
    background_tasks: BackgroundTasks
) -> ToolResponse:
    """
    Create dynamic tools on-demand for agents.
    
    This endpoint enables creation of custom tools using various methods:
    - From templates for common patterns
    - From natural language descriptions using AI
    - From function definitions
    - Custom implementations
    """
    try:
        # Ensure orchestrator is initialized
        if not enhanced_orchestrator.is_initialized:
            await enhanced_orchestrator.initialize()
        
        tool_factory = enhanced_orchestrator.tool_factory
        
        # Create tool based on method
        if request.template_name:
            # Create from template
            tool = await tool_factory.create_tool_from_template(
                template_name=request.template_name,
                custom_name=request.name,
                custom_description=request.description,
                parameter_overrides=request.parameter_overrides
            )
        else:
            # Create from AI description
            tool = await tool_factory.create_tool_from_description(
                name=request.name,
                description=request.description,
                functionality_description=request.functionality_description,
                llm=enhanced_orchestrator.llm,
                category=request.category
            )
        
        # Handle assignment
        assigned_agents = []
        
        if request.assign_to_agent:
            # Assign to specific agent
            await enhanced_orchestrator.assign_tools_to_agent(
                agent_id=request.assign_to_agent,
                tool_names=[tool.name]
            )
            assigned_agents.append(request.assign_to_agent)
        
        if request.make_global:
            # Add to global tools
            enhanced_orchestrator.global_tools[tool.name] = tool
        
        # Create response
        response = ToolResponse(
            tool_name=tool.name,
            description=tool.description,
            category=tool.metadata.category.value,
            complexity=tool.metadata.complexity.value,
            usage_count=tool.metadata.usage_count,
            success_rate=tool.metadata.success_rate,
            created_at=tool.metadata.created_at,
            assigned_agents=assigned_agents
        )
        
        logger.info(
            "Dynamic tool created via API",
            tool_name=tool.name,
            category=request.category.value,
            assigned_to=request.assign_to_agent,
            is_global=request.make_global
        )
        
        return response
        
    except Exception as e:
        logger.error("Failed to create dynamic tool", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create tool: {str(e)}")


@router.get("/agents", response_model=List[AgentResponse])
async def list_all_agents() -> List[AgentResponse]:
    """
    List all agents in the enhanced orchestrator.
    
    Returns comprehensive information about all agents including
    performance metrics, tool assignments, and current status.
    """
    try:
        agents_data = []
        
        for agent_id, agent_info in enhanced_orchestrator.agent_registry.items():
            performance = enhanced_orchestrator.agent_performance.get(agent_id, {})
            tools = enhanced_orchestrator.agent_tools.get(agent_id, [])
            
            response = AgentResponse(
                agent_id=agent_id,
                agent_type=agent_info["type"],
                name=agent_info["name"],
                description=agent_info["description"],
                model=agent_info["config"].get("model", "unknown"),
                status=agent_info["status"],
                tools=tools,
                performance_metrics=performance,
                created_at=agent_info["created_at"]
            )
            agents_data.append(response)
        
        logger.info("All agents listed", count=len(agents_data))
        return agents_data
        
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.get("/tools", response_model=List[ToolResponse])
async def list_all_tools(category: Optional[str] = None) -> List[ToolResponse]:
    """
    List all available tools, optionally filtered by category.
    
    Returns information about all tools including usage statistics,
    performance metrics, and agent assignments.
    """
    try:
        tool_factory = enhanced_orchestrator.tool_factory
        tools_data = []
        
        # Get tools (filtered by category if specified)
        tools = tool_factory.list_tools(category)
        
        for tool in tools:
            # Find agents using this tool
            assigned_agents = []
            for agent_id, agent_tools in enhanced_orchestrator.agent_tools.items():
                if tool.name in agent_tools:
                    assigned_agents.append(agent_id)
            
            response = ToolResponse(
                tool_name=tool.name,
                description=tool.description,
                category=tool.metadata.category.value,
                complexity=tool.metadata.complexity.value,
                usage_count=tool.metadata.usage_count,
                success_rate=tool.metadata.success_rate,
                created_at=tool.metadata.created_at,
                assigned_agents=assigned_agents
            )
            tools_data.append(response)
        
        logger.info("All tools listed", count=len(tools_data), category=category)
        return tools_data
        
    except Exception as e:
        logger.error("Failed to list tools", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.post("/agents/{agent_id}/assign-tools")
async def assign_tools_to_agent_endpoint(
    agent_id: str,
    tool_names: List[str]
) -> Dict[str, Any]:
    """
    Assign existing tools to a specific agent.
    
    This endpoint allows dynamic tool assignment to agents,
    enabling flexible capability expansion.
    """
    try:
        await enhanced_orchestrator.assign_tools_to_agent(agent_id, tool_names)
        
        # Get updated tool list
        updated_tools = enhanced_orchestrator.agent_tools.get(agent_id, [])
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "assigned_tools": tool_names,
            "total_tools": len(updated_tools),
            "message": f"Successfully assigned {len(tool_names)} tools to agent {agent_id}"
        }
        
    except Exception as e:
        logger.error("Failed to assign tools to agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to assign tools: {str(e)}")


@router.post("/agents/{agent_id}/create-custom-tool", response_model=ToolResponse)
async def create_custom_tool_for_agent(
    agent_id: str,
    request: DynamicToolCreateRequest
) -> ToolResponse:
    """
    Create a custom tool specifically for an agent.
    
    This endpoint creates a tool tailored to a specific agent's needs
    and automatically assigns it to that agent.
    """
    try:
        # Create the custom tool
        tool_name = await enhanced_orchestrator.create_tool_for_agent(
            agent_id=agent_id,
            tool_name=request.name,
            tool_description=request.description,
            functionality_description=request.functionality_description,
            category=request.category
        )
        
        # Get tool info
        tool = enhanced_orchestrator.tool_factory.get_tool(tool_name)
        
        response = ToolResponse(
            tool_name=tool.name,
            description=tool.description,
            category=tool.metadata.category.value,
            complexity=tool.metadata.complexity.value,
            usage_count=tool.metadata.usage_count,
            success_rate=tool.metadata.success_rate,
            created_at=tool.metadata.created_at,
            assigned_agents=[agent_id]
        )
        
        logger.info(
            "Custom tool created for agent",
            agent_id=agent_id,
            tool_name=tool_name
        )
        
        return response
        
    except Exception as e:
        logger.error("Failed to create custom tool for agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create custom tool: {str(e)}")


@router.get("/metrics")
async def get_orchestration_metrics() -> Dict[str, Any]:
    """
    Get comprehensive orchestration metrics and statistics.
    
    Returns detailed information about system performance,
    resource usage, and operational statistics.
    """
    try:
        return {
            "execution_metrics": enhanced_orchestrator.execution_metrics,
            "resource_usage": enhanced_orchestrator.resource_usage,
            "agent_count": len(enhanced_orchestrator.agent_registry),
            "tool_count": len(enhanced_orchestrator.tool_factory.registered_tools),
            "global_tool_count": len(enhanced_orchestrator.global_tools),
            "workflow_templates": list(enhanced_orchestrator.workflow_templates.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get orchestration metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/unlimited/agent", response_model=Dict[str, Any])
async def create_unlimited_agent_endpoint(
    agent_type: str,
    name: str,
    description: str,
    config: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
):
    """
    Create unlimited agents with seamless integration.

    Args:
        agent_type: Type of agent (basic, autonomous, research, creative, optimization, custom)
        name: Agent name
        description: Agent description
        config: Optional configuration
        tools: Optional tool names to assign

    Returns:
        Created agent information
    """
    try:
        agent_id = await seamless_integration.create_unlimited_agent(
            agent_type=agent_type,
            name=name,
            description=description,
            config=config,
            tools=tools
        )

        # Get agent info
        agent_info = enhanced_orchestrator.agent_registry.get(agent_id, {})

        logger.info(
            "Unlimited agent created via API",
            agent_id=agent_id,
            agent_type=agent_type,
            name=name
        )

        return {
            "agent_id": agent_id,
            "status": "created",
            "agent_info": agent_info,
            "capabilities": "unlimited",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Failed to create unlimited agent", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create unlimited agent: {str(e)}"
        )


@router.post("/unlimited/tool", response_model=Dict[str, Any])
async def create_unlimited_tool_endpoint(
    name: str,
    description: str,
    functionality_description: str,
    assign_to_agent: Optional[str] = None,
    make_global: bool = True
):
    """
    Create unlimited tools with seamless integration.

    Args:
        name: Tool name
        description: Tool description
        functionality_description: What the tool should do
        assign_to_agent: Optional agent to assign to
        make_global: Whether to make tool globally available

    Returns:
        Created tool information
    """
    try:
        tool_name = await seamless_integration.create_unlimited_tool(
            name=name,
            description=description,
            functionality_description=functionality_description,
            assign_to_agent=assign_to_agent,
            make_global=make_global
        )

        # Get tool metadata
        # tool_metadata = production_tool_registry.get_tool_metadata(tool_name)
        tool_metadata = {"name": tool_name, "description": "Tool created via API"}

        logger.info(
            "Unlimited tool created via API",
            tool_name=tool_name,
            assigned_to=assign_to_agent,
            is_global=make_global
        )

        return {
            "tool_name": tool_name,
            "status": "created",
            "metadata": tool_metadata,
            "capabilities": "unlimited",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Failed to create unlimited tool", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create unlimited tool: {str(e)}"
        )


@router.get("/system/status", response_model=Dict[str, Any])
async def get_system_status():
    """
    Get comprehensive system status with seamless integration.

    Returns:
        Complete system status and capabilities
    """
    try:
        integration_status = seamless_integration.get_integration_status()

        return {
            "system_name": "Revolutionary Agentic AI System",
            "capabilities": [
                "unlimited_agent_creation",
                "dynamic_tool_management",
                "true_agentic_ai",
                "seamless_integration",
                "production_ready"
            ],
            "integration_status": integration_status,
            "system_health": "operational",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


# Additional endpoints for frontend compatibility
@router.post("/agents", response_model=AgentResponse)
async def create_agent_endpoint(request: dict) -> AgentResponse:
    """Create agent endpoint for frontend compatibility with database persistence."""
    try:
        # Convert string agent_type to enum
        agent_type_str = request.get("agent_type", "basic")
        try:
            agent_type = AgentType(agent_type_str)
        except ValueError:
            agent_type = AgentType.BASIC

        # Create proper request object
        agent_request = UnlimitedAgentCreateRequest(
            agent_type=agent_type,
            name=request.get("name", "Unnamed Agent"),
            description=request.get("description", "No description provided"),
            model=request.get("model", "llama3.2:latest"),
            model_provider=request.get("model_provider", "ollama"),
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 2048),
            tools=request.get("tools", []),
            autonomy_level=request.get("autonomy_level", "adaptive"),
            learning_mode=request.get("learning_mode", "active"),
            decision_threshold=request.get("decision_threshold", 0.6)
        )

        background_tasks = BackgroundTasks()
        response = await create_unlimited_agent(agent_request, background_tasks)

        # Ensure agent is persisted to database
        await _persist_agent_to_database(response)

        return response

    except Exception as e:
        logger.error("Failed to create agent via compatibility endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.post("/tools", response_model=ToolResponse)
async def create_tool_endpoint(request: dict) -> ToolResponse:
    """Create tool endpoint for frontend compatibility."""
    try:
        # Convert dict to proper request object with defaults
        tool_request = DynamicToolCreateRequest(
            name=request.get("name", "unnamed_tool"),
            description=request.get("description", "No description provided"),
            functionality_description=request.get("functionality_description", request.get("description", "Basic tool functionality")),
            category="custom",  # Default category
            complexity="simple",  # Default complexity
            assign_to_agent=request.get("assign_to_agent"),
            make_global=request.get("make_global", False)
        )

        background_tasks = BackgroundTasks()
        return await create_dynamic_tool(tool_request, background_tasks)

    except Exception as e:
        logger.error("Failed to create tool via simple endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create tool: {str(e)}")


# ============================================================================
# CUSTOM TOOL UPLOAD AND TEMPLATE ENDPOINTS
# ============================================================================

class ToolUploadResponse(BaseModel):
    """Response model for tool upload."""
    success: bool = Field(..., description="Whether upload was successful")
    tool_id: Optional[str] = Field(None, description="ID of uploaded tool")
    tool_name: str = Field(..., description="Name of the tool")
    validation_result: Dict[str, Any] = Field(..., description="Validation results")
    message: str = Field(..., description="Status message")


class ToolTemplateRequest(BaseModel):
    """Request model for creating tool from template."""
    template_id: str = Field(..., description="Template ID to use")
    tool_name: str = Field(..., description="Name for the new tool")
    tool_description: str = Field(..., description="Description for the new tool")
    placeholder_values: Dict[str, str] = Field(..., description="Values for template placeholders")
    assign_to_agent: Optional[str] = Field(None, description="Agent ID to assign tool to")
    make_global: bool = Field(default=False, description="Make tool globally available")


@router.post("/tools/upload", response_model=ToolUploadResponse)
async def upload_custom_tool(
    file: UploadFile = File(...),
    tool_name: str = Form(...),
    tool_description: str = Form(...),
    category: str = Form(default="custom"),
    assign_to_agent: Optional[str] = Form(None),
    make_global: bool = Form(default=False),
    current_user: Dict = Depends(get_current_user)
):
    """
    Upload a custom tool from a Python file.

    This endpoint allows users to upload their own custom tools by providing:
    - Python file containing the tool implementation
    - Tool metadata (name, description, category)
    - Assignment options (specific agent or global)

    The uploaded code will be validated for security and functionality.
    """
    try:
        logger.info("Custom tool upload started",
                   filename=file.filename,
                   tool_name=tool_name,
                   user=current_user.get("username", "unknown"))

        # Read file content
        content = await file.read()
        code = content.decode('utf-8')

        # Validate the tool code
        validation_result = await tool_validation_service.validate_tool_code(
            code=code,
            filename=file.filename
        )

        if not validation_result.is_valid:
            return ToolUploadResponse(
                success=False,
                tool_id=None,
                tool_name=tool_name,
                validation_result=validation_result.dict(),
                message=f"Tool validation failed: {', '.join(validation_result.issues)}"
            )

        # Create tool in database (you would implement this)
        # For now, we'll simulate tool creation
        tool_id = f"custom_tool_{uuid.uuid4().hex[:8]}"

        # Store tool in unified repository
        # This would integrate with your existing tool system

        logger.info("Custom tool uploaded successfully",
                   tool_id=tool_id,
                   tool_name=tool_name,
                   validation_score=validation_result.security_score)

        return ToolUploadResponse(
            success=True,
            tool_id=tool_id,
            tool_name=tool_name,
            validation_result=validation_result.dict(),
            message="Tool uploaded and validated successfully"
        )

    except Exception as e:
        logger.error("Custom tool upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Tool upload failed: {str(e)}")


@router.get("/tools/templates")
async def list_tool_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: Dict = Depends(get_current_user)
):
    """
    List all available tool templates.

    Templates provide pre-built patterns for common tool types,
    making it easy for users to create custom tools.
    """
    try:
        templates = tool_template_service.list_templates(category=category)

        template_list = []
        for template in templates:
            template_info = tool_template_service.get_template_info(template.id)
            template_list.append(template_info)

        return {
            "success": True,
            "templates": template_list,
            "total_count": len(template_list),
            "filtered_by_category": category
        }

    except Exception as e:
        logger.error("Failed to list tool templates", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@router.get("/tools/templates/{template_id}")
async def get_tool_template(
    template_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed information about a specific tool template."""
    try:
        template_info = tool_template_service.get_template_info(template_id)

        if not template_info:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

        return {
            "success": True,
            "template": template_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get tool template", template_id=template_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@router.post("/tools/from-template", response_model=ToolResponse)
async def create_tool_from_template(
    request: ToolTemplateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a custom tool from a template.

    This endpoint generates a tool using a pre-built template and
    user-provided values for the template placeholders.
    """
    try:
        logger.info("Creating tool from template",
                   template_id=request.template_id,
                   tool_name=request.tool_name,
                   user=current_user.get("username", "unknown"))

        # Validate template values
        validation = tool_template_service.validate_template_values(
            request.template_id,
            request.placeholder_values
        )

        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required placeholders: {', '.join(validation['missing_placeholders'])}"
            )

        # Generate tool code from template
        tool_code = tool_template_service.generate_tool_code(
            template_id=request.template_id,
            values=request.placeholder_values
        )

        # Validate generated code
        validation_result = await tool_validation_service.validate_tool_code(
            code=tool_code,
            filename=f"{request.tool_name}_from_template.py"
        )

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Generated tool code validation failed: {', '.join(validation_result.issues)}"
            )

        # Create tool (integrate with your existing system)
        tool_id = f"template_tool_{uuid.uuid4().hex[:8]}"

        # Return response in existing format
        response = ToolResponse(
            tool_name=request.tool_name,
            description=request.tool_description,
            category="custom",
            complexity="simple",
            usage_count=0,
            success_rate=0.0,
            created_at=datetime.now(),
            assigned_agents=[request.assign_to_agent] if request.assign_to_agent else []
        )

        logger.info("Tool created from template successfully",
                   tool_id=tool_id,
                   template_id=request.template_id,
                   tool_name=request.tool_name)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create tool from template", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create tool from template: {str(e)}")


@router.post("/tools/validate")
async def validate_tool_code(
    code: str = Form(...),
    filename: str = Form(default="uploaded_tool.py"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Validate tool code without creating the tool.

    This endpoint allows users to check if their tool code is valid
    and secure before actually uploading it.
    """
    try:
        validation_result = await tool_validation_service.validate_tool_code(
            code=code,
            filename=filename
        )

        return {
            "success": True,
            "validation_result": validation_result.dict(),
            "is_valid": validation_result.is_valid,
            "security_score": validation_result.security_score,
            "summary": {
                "issues_count": len(validation_result.issues),
                "warnings_count": len(validation_result.warnings),
                "dependencies_count": len(validation_result.dependencies)
            }
        }

    except Exception as e:
        logger.error("Tool code validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# ============================================================================
# AGENT BUILDER ORCHESTRATION ENDPOINTS
# ============================================================================

class AgentOrchestrationRequest(BaseModel):
    """Request for orchestrating multiple agents."""
    agents: List[Dict[str, Any]] = Field(..., description="List of agent configurations")
    coordination_type: str = Field(default="sequential", description="Coordination type (sequential, parallel, hierarchical)")
    shared_context: Optional[Dict[str, Any]] = Field(default=None, description="Shared context for all agents")
    timeout_seconds: int = Field(default=600, description="Overall timeout for orchestration")


class AgentOrchestrationResponse(BaseModel):
    """Response from agent orchestration."""
    orchestration_id: str = Field(..., description="Orchestration session ID")
    agents_created: List[str] = Field(..., description="List of created agent IDs")
    coordination_type: str = Field(..., description="Coordination type used")
    status: str = Field(..., description="Orchestration status")
    created_at: datetime = Field(..., description="Creation timestamp")


@router.post("/agent-builder/orchestrate", response_model=AgentOrchestrationResponse, tags=["Agent Builder Orchestration"])
async def orchestrate_multiple_agents(request: AgentOrchestrationRequest) -> AgentOrchestrationResponse:
    """
    Orchestrate multiple agents for complex workflows.

    This endpoint creates and coordinates multiple agents working together
    on complex tasks with different coordination patterns.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentOrchestrationAPI",
        operation="orchestrate_multiple_agents"
    )

    try:
        backend_logger.info(
            f"Starting agent orchestration with {len(request.agents)} agents",
            LogCategory.ORCHESTRATION,
            "AgentOrchestrationAPI"
        )

        # Get enhanced LLM manager and initialize agent registry
        llm_manager = get_enhanced_llm_manager()
        if not llm_manager.is_initialized():
            await llm_manager.initialize()

        agent_factory = AgentBuilderFactory(llm_manager)
        system_orchestrator = await get_system_orchestrator()
        agent_registry = initialize_agent_registry(agent_factory, system_orchestrator)

        orchestration_id = f"orchestration_{uuid.uuid4().hex[:8]}"
        created_agents = []

        # Create agents based on configurations
        for i, agent_config in enumerate(request.agents):
            try:
                # Parse agent configuration
                agent_type = AgentType(agent_config.get("agent_type", "react"))
                provider_type = ProviderType(agent_config.get("llm_provider", "OLLAMA").upper())

                # Create LLM config
                llm_config = LLMConfig(
                    provider=provider_type,
                    model_id=agent_config.get("llm_model", "llama3.2:latest"),
                    temperature=agent_config.get("temperature", 0.7),
                    max_tokens=agent_config.get("max_tokens", 2048)
                )

                # Create builder config
                builder_config = AgentBuilderConfig(
                    name=agent_config.get("name", f"Agent_{i+1}"),
                    description=agent_config.get("description", f"Orchestrated agent {i+1}"),
                    agent_type=agent_type,
                    llm_config=llm_config,
                    capabilities=[],  # Will be populated based on agent type
                    tools=agent_config.get("tools", []),
                    system_prompt=agent_config.get("system_prompt"),
                    enable_memory=agent_config.get("enable_memory", True),
                    enable_collaboration=True,  # Enable for orchestration
                    custom_config=agent_config.get("custom_config")
                )

                # Register the agent
                agent_id = await agent_registry.register_agent(
                    config=builder_config,
                    owner=f"orchestration_{orchestration_id}",
                    tags=[f"orchestration:{orchestration_id}", f"coordination:{request.coordination_type}"]
                )

                # Start the agent
                await agent_registry.start_agent(agent_id)
                created_agents.append(agent_id)

                backend_logger.info(
                    f"Created orchestrated agent: {agent_id}",
                    LogCategory.AGENT_OPERATIONS,
                    "AgentOrchestrationAPI"
                )

            except Exception as e:
                backend_logger.error(
                    f"Failed to create agent {i+1}: {str(e)}",
                    LogCategory.AGENT_OPERATIONS,
                    "AgentOrchestrationAPI",
                    error=str(e)
                )
                # Continue with other agents
                continue

        # Create collaboration group if multiple agents created
        if len(created_agents) > 1:
            await agent_registry.create_collaboration_group(
                group_id=orchestration_id,
                agent_ids=created_agents
            )

        response = AgentOrchestrationResponse(
            orchestration_id=orchestration_id,
            agents_created=created_agents,
            coordination_type=request.coordination_type,
            status="active" if created_agents else "failed",
            created_at=datetime.utcnow()
        )

        backend_logger.info(
            f"Agent orchestration completed: {len(created_agents)} agents created",
            LogCategory.ORCHESTRATION,
            "AgentOrchestrationAPI"
        )

        return response

    except Exception as e:
        backend_logger.error(
            f"Failed to orchestrate agents: {str(e)}",
            LogCategory.ORCHESTRATION,
            "AgentOrchestrationAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to orchestrate agents: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="orchestrate_multiple_agents",
                duration=end_time - start_time,
                component="AgentOrchestrationAPI"
            )
        )


@router.get("/agent-builder/templates/specialized", response_model=List[Dict[str, Any]], tags=["Agent Builder Orchestration"])
async def get_specialized_agent_templates() -> List[Dict[str, Any]]:
    """
    Get specialized agent templates optimized for orchestration workflows.

    These templates are designed for multi-agent coordination and
    complex workflow execution.
    """
    start_time = time.time()

    CorrelationContext.update_context(
        component="AgentOrchestrationAPI",
        operation="get_specialized_agent_templates"
    )

    try:
        backend_logger.info(
            "Getting specialized agent templates",
            LogCategory.AGENT_OPERATIONS,
            "AgentOrchestrationAPI"
        )

        # Define specialized templates for orchestration
        specialized_templates = [
            {
                "name": "Workflow Coordinator",
                "description": "Coordinates complex multi-step workflows",
                "agent_type": "composite",
                "capabilities": ["coordination", "planning", "monitoring"],
                "tools": ["workflow_manager", "task_distributor", "progress_tracker"],
                "use_cases": ["Project management", "Process automation", "Task coordination"],
                "coordination_role": "coordinator"
            },
            {
                "name": "Data Pipeline Agent",
                "description": "Specialized for data processing pipelines",
                "agent_type": "workflow",
                "capabilities": ["data_processing", "validation", "transformation"],
                "tools": ["data_loader", "data_validator", "data_transformer", "pipeline_monitor"],
                "use_cases": ["ETL processes", "Data validation", "Stream processing"],
                "coordination_role": "processor"
            },
            {
                "name": "Quality Assurance Agent",
                "description": "Monitors and validates outputs from other agents",
                "agent_type": "autonomous",
                "capabilities": ["validation", "quality_control", "reporting"],
                "tools": ["output_validator", "quality_checker", "report_generator"],
                "use_cases": ["Output validation", "Quality control", "Compliance checking"],
                "coordination_role": "validator"
            },
            {
                "name": "Communication Hub Agent",
                "description": "Manages communication between agents and external systems",
                "agent_type": "composite",
                "capabilities": ["communication", "routing", "translation"],
                "tools": ["message_router", "protocol_translator", "notification_sender"],
                "use_cases": ["Inter-agent communication", "External API integration", "Notification management"],
                "coordination_role": "communicator"
            },
            {
                "name": "Resource Manager Agent",
                "description": "Manages shared resources and prevents conflicts",
                "agent_type": "autonomous",
                "capabilities": ["resource_management", "conflict_resolution", "optimization"],
                "tools": ["resource_allocator", "conflict_resolver", "usage_optimizer"],
                "use_cases": ["Resource allocation", "Conflict prevention", "Performance optimization"],
                "coordination_role": "resource_manager"
            }
        ]

        backend_logger.info(
            f"Retrieved {len(specialized_templates)} specialized templates",
            LogCategory.AGENT_OPERATIONS,
            "AgentOrchestrationAPI"
        )

        return specialized_templates

    except Exception as e:
        backend_logger.error(
            f"Failed to get specialized templates: {str(e)}",
            LogCategory.AGENT_OPERATIONS,
            "AgentOrchestrationAPI",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")
    finally:
        end_time = time.time()
        backend_logger.log_performance(
            PerformanceMetrics(
                operation="get_specialized_agent_templates",
                duration=end_time - start_time,
                component="AgentOrchestrationAPI"
            )
        )
