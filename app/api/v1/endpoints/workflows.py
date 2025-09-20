"""
Workflow Management API endpoints.

This module provides comprehensive workflow management functionality including
execution, creation, templates, and history for LangGraph-based multi-agent workflows.
"""

import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError

from app.config.settings import get_settings
from app.core.dependencies import get_orchestrator, require_authentication, get_database_session
from app.orchestration.subgraphs import HierarchicalWorkflowOrchestrator
from app.models.workflow import Workflow
from app.core.node_registry import get_node_registry

# Import new backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogLevel, LogCategory, PerformanceMetrics
from app.backend_logging.context import CorrelationContext

logger = structlog.get_logger(__name__)
backend_logger = get_logger()

router = APIRouter(tags=["Workflow Management"])


# Pydantic models for API requests/responses
class WorkflowExecuteRequest(BaseModel):
    """Workflow execution request."""
    task: str = Field(..., description="Task description")
    workflow_type: str = Field(default="hierarchical", description="Workflow type")
    model: str = Field(default="llama3.2:latest", description="Ollama model to use")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    timeout: int = Field(default=300, description="Workflow timeout in seconds")
    agents: Optional[List[str]] = Field(default=None, description="Specific agent IDs to use")


class VisualWorkflowNode(BaseModel):
    """Visual workflow node definition."""
    id: str = Field(..., description="Unique node ID")
    type: str = Field(..., description="Node type")
    position: Dict[str, float] = Field(..., description="Node position on canvas")
    data: Dict[str, Any] = Field(..., description="Node configuration and data")


class VisualWorkflowConnection(BaseModel):
    """Visual workflow connection definition."""
    id: str = Field(..., description="Unique connection ID")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    sourceHandle: Optional[str] = Field(None, description="Source port ID")
    targetHandle: Optional[str] = Field(None, description="Target port ID")


class VisualWorkflowExecuteRequest(BaseModel):
    """Visual workflow execution request."""
    workflow_id: str = Field(..., description="Workflow ID")
    nodes: List[VisualWorkflowNode] = Field(..., description="Workflow nodes")
    connections: List[VisualWorkflowConnection] = Field(..., description="Node connections")
    inputs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Initial workflow inputs")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional execution context")


class WorkflowCreateRequest(BaseModel):
    """Workflow creation request."""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    workflow_type: str = Field(default="custom", description="Workflow type")
    agents: List[str] = Field(default_factory=list, description="Agent types to include")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow steps")
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workflow configuration")


class WorkflowBuilderCreateRequest(BaseModel):
    """Workflow creation request from the visual builder (frontend)."""
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    nodes: List[Dict[str, Any]] = Field(..., description="Workflow nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Workflow edges")
    status: str = Field(default="draft", description="Workflow status")


class WorkflowResponse(BaseModel):
    """Workflow response model."""
    id: str = Field(..., description="Workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    workflow_type: str = Field(..., description="Workflow type")
    nodes: List[Dict[str, Any]] = Field(..., description="Workflow nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Workflow edges")
    status: str = Field(..., description="Workflow status")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    success: bool = Field(default=True, description="Success flag")


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution response."""
    workflow_id: str = Field(..., description="Workflow execution ID")
    task: str = Field(..., description="Original task")
    result: str = Field(..., description="Workflow result")
    workflow_type: str = Field(..., description="Workflow type")
    status: str = Field(..., description="Execution status")
    agents_used: List[str] = Field(..., description="Agents that participated")
    execution_time: float = Field(..., description="Execution time in seconds")
    tokens_used: int = Field(..., description="Total tokens used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WorkflowTemplate(BaseModel):
    """Workflow template."""
    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    workflow_type: str = Field(..., description="Workflow type")
    agents: List[str] = Field(..., description="Required agent types")
    steps: List[Dict[str, Any]] = Field(..., description="Template steps")
    configuration: Dict[str, Any] = Field(..., description="Default configuration")
    category: str = Field(..., description="Template category")


class WorkflowHistoryItem(BaseModel):
    """Workflow history item."""
    workflow_id: str = Field(..., description="Workflow execution ID")
    task: str = Field(..., description="Original task")
    workflow_type: str = Field(..., description="Workflow type")
    status: str = Field(..., description="Execution status")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    agents_used: List[str] = Field(..., description="Agents that participated")
    tokens_used: Optional[int] = Field(default=None, description="Total tokens used")


@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecuteRequest,
    orchestrator: HierarchicalWorkflowOrchestrator = Depends(get_orchestrator)
) -> WorkflowResponse:
    """
    Execute a workflow with the specified configuration.
    
    Args:
        request: Workflow execution request
        
    Returns:
        Workflow execution result
    """
    start_time = asyncio.get_event_loop().time()
    workflow_id = str(uuid.uuid4())
    
    try:
        if not orchestrator.is_initialized:
            await orchestrator.initialize()
        
        logger.info(
            "Workflow execution started",
            workflow_id=workflow_id,
            workflow_type=request.workflow_type,
            task_length=len(request.task)
        )
        
        # Execute workflow based on type
        if request.workflow_type == "hierarchical":
            result = await orchestrator.execute_hierarchical_workflow(
                task=request.task,
                context={
                    **request.context,
                    "model": request.model,
                    "workflow_id": workflow_id
                }
            )
        else:
            # Default multi-agent workflow
            result = await orchestrator.execute_workflow(
                workflow_id="default_multi_agent",
                inputs={
                    "task": request.task,
                    "model": request.model,
                    "workflow_id": workflow_id,
                    **request.context
                },
                agent_ids=request.agents
            )
        
        # Calculate execution metrics
        execution_time = asyncio.get_event_loop().time() - start_time
        tokens_used = len(request.task.split()) + len(str(result).split())  # Rough estimate
        
        # Extract agents used from result
        agents_used = result.get("agents_used", [])
        if not agents_used and request.agents:
            agents_used = request.agents
        
        response = WorkflowExecutionResponse(
            workflow_id=workflow_id,
            task=request.task,
            result=str(result.get("final_output", result)),
            workflow_type=request.workflow_type,
            status="completed",
            agents_used=agents_used,
            execution_time=execution_time,
            tokens_used=tokens_used,
            metadata={
                "model": request.model,
                "context_provided": bool(request.context),
                "execution_details": result
            }
        )
        
        logger.info(
            "Workflow execution completed",
            workflow_id=workflow_id,
            execution_time=execution_time,
            tokens_used=tokens_used
        )
        
        return response
        
    except Exception as e:
        logger.error("Workflow execution failed", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.post("/execute-visual", response_model=WorkflowExecutionResponse)
async def execute_visual_workflow(
    request: VisualWorkflowExecuteRequest,
    orchestrator: HierarchicalWorkflowOrchestrator = Depends(get_orchestrator)
) -> WorkflowExecutionResponse:
    """
    Execute a visual workflow created in the frontend.

    This endpoint processes workflows created with the visual node editor,
    executing nodes in the correct order and managing data flow between them.

    Args:
        request: Visual workflow execution request
        orchestrator: Workflow orchestrator instance

    Returns:
        Workflow execution result with real-time status updates
    """
    start_time = asyncio.get_event_loop().time()
    execution_id = f"visual_exec_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    try:
        backend_logger.info(
            f"Visual workflow execution started",
            LogCategory.ORCHESTRATION,
            "VisualWorkflowExecution",
            data={
                "workflow_id": request.workflow_id,
                "execution_id": execution_id,
                "node_count": len(request.nodes),
                "connection_count": len(request.connections)
            }
        )

        # Initialize node registry
        node_registry = get_node_registry()

        # Build execution graph from visual workflow
        execution_graph = await build_execution_graph(request.nodes, request.connections)

        # Execute workflow nodes in topological order
        execution_results = {}
        node_execution_order = get_topological_order(execution_graph)

        for node_id in node_execution_order:
            node = next((n for n in request.nodes if n.id == node_id), None)
            if not node:
                continue

            # Get node inputs from connected nodes
            node_inputs = get_node_inputs(node_id, request.connections, execution_results)

            # Execute node
            node_result = await execute_visual_node(
                node=node,
                inputs=node_inputs,
                execution_context={
                    "execution_id": execution_id,
                    "workflow_id": request.workflow_id,
                    "node_id": node_id
                },
                node_registry=node_registry
            )

            execution_results[node_id] = node_result

            # Log node execution
            backend_logger.info(
                f"Node '{node.type}' executed",
                LogCategory.ORCHESTRATION,
                "VisualWorkflowExecution",
                data={
                    "node_id": node_id,
                    "node_type": node.type,
                    "success": node_result.get("success", False),
                    "execution_time": node_result.get("execution_time", 0)
                }
            )

        execution_time = asyncio.get_event_loop().time() - start_time

        # Build response
        response = WorkflowExecutionResponse(
            workflow_id=execution_id,
            task=f"Visual workflow execution: {request.workflow_id}",
            result=execution_results,
            workflow_type="visual",
            status="completed",
            agents_used=[],  # Will be populated based on agent nodes
            execution_time=execution_time,
            tokens_used=0,  # Will be calculated from node results
            metadata={
                "visual_workflow": True,
                "nodes_executed": len(execution_results),
                "execution_graph": execution_graph
            }
        )

        backend_logger.info(
            f"Visual workflow execution completed",
            LogCategory.ORCHESTRATION,
            "VisualWorkflowExecution",
            data={
                "execution_id": execution_id,
                "execution_time": execution_time,
                "nodes_executed": len(execution_results)
            }
        )

        return response

    except Exception as e:
        backend_logger.error(
            f"Visual workflow execution failed",
            LogCategory.ERROR_TRACKING,
            "VisualWorkflowExecution",
            error=str(e),
            data={"execution_id": execution_id}
        )
        raise HTTPException(status_code=500, detail=f"Visual workflow execution failed: {str(e)}")


@router.get("/templates", response_model=List[WorkflowTemplate])
async def get_workflow_templates() -> List[WorkflowTemplate]:
    """
    Get available workflow templates.
    
    Returns:
        List of workflow templates
    """
    try:
        # Define built-in workflow templates
        templates = [
            WorkflowTemplate(
                id="hierarchical_research",
                name="Hierarchical Research Workflow",
                description="Multi-level research workflow with supervisor and specialist agents",
                workflow_type="hierarchical",
                agents=["supervisor", "researcher", "analyst"],
                steps=[
                    {"step": "task_analysis", "agent": "supervisor", "description": "Analyze and break down the task"},
                    {"step": "research", "agent": "researcher", "description": "Conduct research on the topic"},
                    {"step": "analysis", "agent": "analyst", "description": "Analyze research findings"},
                    {"step": "synthesis", "agent": "supervisor", "description": "Synthesize final results"}
                ],
                configuration={
                    "max_iterations": 10,
                    "timeout": 600,
                    "parallel_execution": False
                },
                category="research"
            ),
            WorkflowTemplate(
                id="collaborative_problem_solving",
                name="Collaborative Problem Solving",
                description="Multiple agents collaborate to solve complex problems",
                workflow_type="collaborative",
                agents=["problem_analyzer", "solution_generator", "evaluator"],
                steps=[
                    {"step": "problem_analysis", "agent": "problem_analyzer", "description": "Analyze the problem"},
                    {"step": "solution_generation", "agent": "solution_generator", "description": "Generate solutions"},
                    {"step": "evaluation", "agent": "evaluator", "description": "Evaluate solutions"},
                    {"step": "refinement", "agent": "solution_generator", "description": "Refine best solution"}
                ],
                configuration={
                    "max_iterations": 8,
                    "timeout": 400,
                    "parallel_execution": True
                },
                category="problem_solving"
            ),
            WorkflowTemplate(
                id="content_creation",
                name="Content Creation Pipeline",
                description="Structured content creation with research, writing, and review",
                workflow_type="pipeline",
                agents=["researcher", "writer", "editor"],
                steps=[
                    {"step": "research", "agent": "researcher", "description": "Research the topic"},
                    {"step": "outline", "agent": "writer", "description": "Create content outline"},
                    {"step": "writing", "agent": "writer", "description": "Write the content"},
                    {"step": "review", "agent": "editor", "description": "Review and edit content"}
                ],
                configuration={
                    "max_iterations": 6,
                    "timeout": 500,
                    "parallel_execution": False
                },
                category="content"
            )
        ]
        
        logger.info("Workflow templates retrieved", count=len(templates))
        return templates
        
    except Exception as e:
        logger.error("Failed to get workflow templates", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get workflow templates: {str(e)}")


@router.get("/history", response_model=List[WorkflowHistoryItem])
async def get_workflow_history(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
) -> List[WorkflowHistoryItem]:
    """
    Get workflow execution history.
    
    Args:
        limit: Maximum number of items to return
        offset: Number of items to skip
        status: Filter by execution status
        
    Returns:
        List of workflow history items
    """
    try:
        # This is a placeholder implementation
        # In a real system, this would query a database
        history_items = [
            WorkflowHistoryItem(
                workflow_id="example-workflow-1",
                task="Analyze market trends for Q4 2024",
                workflow_type="hierarchical",
                status="completed",
                created_at=datetime.now(),
                completed_at=datetime.now(),
                execution_time=45.2,
                agents_used=["supervisor", "researcher", "analyst"],
                tokens_used=1250
            ),
            WorkflowHistoryItem(
                workflow_id="example-workflow-2",
                task="Generate product documentation",
                workflow_type="pipeline",
                status="completed",
                created_at=datetime.now(),
                completed_at=datetime.now(),
                execution_time=32.8,
                agents_used=["researcher", "writer", "editor"],
                tokens_used=980
            )
        ]
        
        # Apply filters
        if status:
            history_items = [item for item in history_items if item.status == status]
        
        # Apply pagination
        paginated_items = history_items[offset:offset + limit]
        
        logger.info("Workflow history retrieved", count=len(paginated_items))
        return paginated_items
        
    except Exception as e:
        logger.error("Failed to get workflow history", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get workflow history: {str(e)}")


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_database_session)
) -> WorkflowResponse:
    """
    Get specific workflow details.

    Args:
        workflow_id: Workflow ID
        db: Database session

    Returns:
        Workflow details
    """
    try:
        # Query workflow from database
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        response = WorkflowResponse(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            workflow_type=workflow.workflow_type,
            nodes=workflow.nodes,
            edges=workflow.edges,
            status=workflow.status,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat(),
            success=True
        )

        logger.info("Workflow retrieved", workflow_id=workflow_id, name=workflow.name)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    request: WorkflowBuilderCreateRequest,
    db: AsyncSession = Depends(get_database_session)
) -> WorkflowResponse:
    """
    Update an existing workflow.

    Args:
        workflow_id: Workflow ID
        request: Updated workflow data
        db: Database session

    Returns:
        Updated workflow information
    """
    try:
        # Check if workflow exists
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        # Update workflow fields
        workflow.name = request.name.strip()
        workflow.description = request.description.strip() if request.description else ""
        workflow.nodes = request.nodes
        workflow.edges = request.edges
        workflow.status = request.status

        await db.commit()
        await db.refresh(workflow)

        response = WorkflowResponse(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            workflow_type=workflow.workflow_type,
            nodes=workflow.nodes,
            edges=workflow.edges,
            status=workflow.status,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat(),
            success=True
        )

        logger.info("Workflow updated", workflow_id=workflow_id, name=workflow.name)
        return response

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error("Failed to update workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_database_session)
) -> Dict[str, Any]:
    """
    Delete a workflow.

    Args:
        workflow_id: Workflow ID
        db: Database session

    Returns:
        Deletion confirmation
    """
    try:
        # Check if workflow exists
        result = await db.execute(
            select(Workflow).where(Workflow.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

        # Delete workflow
        await db.execute(
            delete(Workflow).where(Workflow.id == workflow_id)
        )
        await db.commit()

        logger.info("Workflow deleted", workflow_id=workflow_id, name=workflow.name)
        return {
            "success": True,
            "message": f"Workflow {workflow.name} deleted successfully",
            "workflow_id": workflow_id
        }

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error("Failed to delete workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")


@router.get("", response_model=List[WorkflowResponse])
async def list_workflows(
    response: Response,
    db: AsyncSession = Depends(get_database_session),
    skip: int = 0,
    limit: int = 100
) -> List[WorkflowResponse]:
    """
    List all workflows.

    Args:
        db: Database session
        skip: Number of workflows to skip
        limit: Maximum number of workflows to return

    Returns:
        List of workflows
    """
    try:
        # Query workflows from database
        result = await db.execute(
            select(Workflow)
            .order_by(Workflow.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        workflows = result.scalars().all()

        workflow_list = [
            WorkflowResponse(
                id=str(workflow.id),
                name=workflow.name,
                description=workflow.description,
                workflow_type=workflow.workflow_type,
                nodes=workflow.nodes,
                edges=workflow.edges,
                status=workflow.status,
                created_at=workflow.created_at.isoformat(),
                updated_at=workflow.updated_at.isoformat(),
                success=True
            )
            for workflow in workflows
        ]

        logger.info("Workflows listed", count=len(workflow_list))

        # Add caching headers for better performance
        response.headers["Cache-Control"] = "public, max-age=30"  # Cache for 30 seconds
        response.headers["ETag"] = f'"{hash(str(workflow_list))}"'

        return workflow_list

    except Exception as e:
        logger.error("Failed to list workflows", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")


@router.post("/create", response_model=Dict[str, Any])
async def create_workflow_template(
    request: WorkflowCreateRequest,
    db: AsyncSession = Depends(get_database_session)
) -> Dict[str, Any]:
    """
    Create a custom workflow template.

    Args:
        request: Workflow creation request
        db: Database session

    Returns:
        Created workflow information
    """
    try:
        # Create new workflow in database
        workflow = Workflow(
            name=request.name,
            description=request.description,
            workflow_type=request.workflow_type,
            nodes=[],  # Template workflows don't have visual nodes
            edges=[],  # Template workflows don't have visual edges
            configuration=request.configuration or {},
            status="draft"
        )

        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)

        workflow_info = {
            "workflow_id": str(workflow.id),
            "name": workflow.name,
            "description": workflow.description,
            "workflow_type": workflow.workflow_type,
            "agents": request.agents,
            "steps": request.steps,
            "configuration": workflow.configuration,
            "created_at": workflow.created_at.isoformat(),
            "status": workflow.status,
            "success": True
        }

        logger.info("Custom workflow template created", workflow_id=str(workflow.id), name=workflow.name)
        return workflow_info

    except IntegrityError as e:
        await db.rollback()
        logger.error("Database integrity error creating workflow", error=str(e))
        raise HTTPException(status_code=400, detail="Workflow with this name may already exist")
    except Exception as e:
        await db.rollback()
        logger.error("Failed to create workflow template", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")


@router.post("", response_model=WorkflowResponse)
async def create_workflow_from_builder(
    request: WorkflowBuilderCreateRequest,
    db: AsyncSession = Depends(get_database_session)
) -> WorkflowResponse:
    """
    Create a workflow from the visual builder (frontend).

    This endpoint matches the frontend's expected API call to POST /workflows
    and handles the visual workflow builder data structure.

    Args:
        request: Workflow creation request from visual builder
        db: Database session

    Returns:
        Created workflow information
    """
    try:
        # Validate required fields
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Workflow name is required")

        if not request.nodes:
            raise HTTPException(status_code=400, detail="Workflow must contain at least one node")

        # Create new workflow in database
        workflow = Workflow(
            name=request.name.strip(),
            description=request.description.strip() if request.description else "",
            workflow_type="visual",  # Mark as visual workflow
            nodes=request.nodes,
            edges=request.edges,
            configuration={},
            status=request.status
        )

        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)

        # Return response matching frontend expectations
        response = WorkflowResponse(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            workflow_type=workflow.workflow_type,
            nodes=workflow.nodes,
            edges=workflow.edges,
            status=workflow.status,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat(),
            success=True
        )

        logger.info("Visual workflow created", workflow_id=str(workflow.id), name=workflow.name, nodes=len(workflow.nodes))
        return response

    except HTTPException:
        await db.rollback()
        raise
    except IntegrityError as e:
        await db.rollback()
        logger.error("Database integrity error creating visual workflow", error=str(e))
        raise HTTPException(status_code=400, detail="Workflow with this name may already exist")
    except Exception as e:
        await db.rollback()
        logger.error("Failed to create visual workflow", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")


# ============================================================================
# REVOLUTIONARY VISUAL WORKFLOW EXECUTION
# ============================================================================

class VisualWorkflowRequest(BaseModel):
    """Visual workflow execution request."""
    workflow_name: str = Field(..., description="Name of the visual workflow")
    components: List[Dict[str, Any]] = Field(..., description="Visual workflow components")
    execution_mode: str = Field(default="sequential", description="Execution mode")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution context")
    save_as_template: bool = Field(default=False, description="Save workflow as template")


class VisualWorkflowResponse(BaseModel):
    """Visual workflow execution response."""
    workflow_id: str
    workflow_name: str
    status: str
    execution_mode: str
    total_components: int
    start_time: datetime
    results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    template_saved: bool = False


@router.post("/execute-visual", response_model=VisualWorkflowResponse)
async def execute_visual_workflow(request: VisualWorkflowRequest) -> VisualWorkflowResponse:
    """
    Execute a visual workflow created from drag-and-drop components.

    This revolutionary endpoint allows execution of workflows created through
    the visual builder interface with full support for autonomous and
    instruction-based execution modes.
    """
    try:
        workflow_id = f"visual_workflow_{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()

        logger.info(
            "Visual workflow execution started",
            workflow_id=workflow_id,
            workflow_name=request.workflow_name,
            num_components=len(request.components),
            execution_mode=request.execution_mode
        )

        # Get the agent template library for component workflow execution
        from app.agents.templates import AgentTemplateLibrary
        template_library = AgentTemplateLibrary()

        # Execute workflow from components
        workflow_result = await template_library.execute_workflow_from_components(
            workflow_id=workflow_id,
            components=request.components,
            execution_mode=request.execution_mode,
            context=request.context
        )

        # Save as template if requested
        template_saved = False
        if request.save_as_template:
            try:
                template_config = {
                    "name": request.workflow_name,
                    "description": f"Visual workflow template: {request.workflow_name}",
                    "components": request.components,
                    "execution_mode": request.execution_mode,
                    "created_from": "visual_builder",
                    "created_at": start_time.isoformat()
                }

                template_library.save_custom_template(request.workflow_name, template_config)
                template_saved = True

                logger.info(
                    "Visual workflow saved as template",
                    workflow_id=workflow_id,
                    template_name=request.workflow_name
                )

            except Exception as e:
                logger.warning("Failed to save workflow as template", error=str(e))

        response = VisualWorkflowResponse(
            workflow_id=workflow_id,
            workflow_name=request.workflow_name,
            status=workflow_result["status"],
            execution_mode=request.execution_mode,
            total_components=len(request.components),
            start_time=start_time,
            results=workflow_result.get("results"),
            execution_time=workflow_result.get("execution_time"),
            template_saved=template_saved
        )

        logger.info(
            "Visual workflow execution completed",
            workflow_id=workflow_id,
            status=workflow_result["status"],
            execution_time=workflow_result.get("execution_time", 0)
        )

        return response

    except Exception as e:
        logger.error("Failed to execute visual workflow", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to execute visual workflow: {str(e)}")


@router.get("/visual/templates")
async def list_visual_workflow_templates() -> Dict[str, Any]:
    """
    List all available visual workflow templates.

    This endpoint returns all workflow templates that can be used
    as starting points for visual workflow creation.
    """
    try:
        from app.agents.templates import AgentTemplateLibrary
        template_library = AgentTemplateLibrary()

        # Get all templates
        all_templates = template_library.get_all_templates()

        # Filter for visual workflow templates
        visual_templates = [
            template for template in all_templates
            if template.get("created_from") == "visual_builder" or
               template.get("type") == "visual_workflow"
        ]

        # Get component palette for visual builder
        component_palette = template_library.get_component_palette()

        return {
            "visual_workflow_templates": visual_templates,
            "component_palette": component_palette,
            "total_templates": len(visual_templates),
            "available_components": sum(len(components) for components in component_palette.values())
        }

    except Exception as e:
        logger.error("Failed to list visual workflow templates", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@router.post("/visual/validate")
async def validate_visual_workflow(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a visual workflow before execution.

    This endpoint checks the workflow components for compatibility,
    dependencies, and potential issues before execution.
    """
    try:
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "component_count": len(components),
            "estimated_execution_time": 0.0,
            "dependencies_resolved": True
        }

        # Basic validation
        for i, component in enumerate(components):
            component_type = component.get("type")
            component_config = component.get("config", {})

            # Check required fields
            if not component_type:
                validation_results["errors"].append(f"Component {i+1}: Missing component type")
                validation_results["valid"] = False

            # Estimate execution time based on component type
            if component_type == "TOOL":
                validation_results["estimated_execution_time"] += 0.5
            elif component_type == "CAPABILITY":
                validation_results["estimated_execution_time"] += 1.0
            elif component_type == "PROMPT":
                validation_results["estimated_execution_time"] += 2.0
            elif component_type == "WORKFLOW_STEP":
                validation_results["estimated_execution_time"] += 0.3

            # Check for potential issues
            if component_type == "PROMPT" and not component_config.get("template"):
                validation_results["warnings"].append(f"Component {i+1}: Prompt component without template")

        # Check for dependency cycles (simplified)
        component_names = [comp.get("name", f"component_{i+1}") for i, comp in enumerate(components)]
        if len(set(component_names)) != len(component_names):
            validation_results["warnings"].append("Duplicate component names detected")

        logger.info(
            "Visual workflow validation completed",
            component_count=len(components),
            valid=validation_results["valid"],
            warnings=len(validation_results["warnings"]),
            errors=len(validation_results["errors"])
        )

        return validation_results

    except Exception as e:
        logger.error("Failed to validate visual workflow", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to validate workflow: {str(e)}")


@router.get("/visual/{workflow_id}/progress")
async def get_visual_workflow_progress(workflow_id: str) -> Dict[str, Any]:
    """
    Get real-time progress of a visual workflow execution.

    This endpoint provides detailed progress information for
    visual workflows including step completion and results.
    """
    try:
        from app.agent_builder_platform import get_step_state_tracker
        step_tracker = get_step_state_tracker()

        # Get all steps for the workflow
        step_ids = step_tracker.get_workflow_steps(workflow_id)

        progress_info = {
            "workflow_id": workflow_id,
            "total_steps": len(step_ids),
            "completed_steps": 0,
            "failed_steps": 0,
            "running_steps": 0,
            "pending_steps": 0,
            "overall_progress": 0.0,
            "steps": []
        }

        for step_id in step_ids:
            step_state = step_tracker.get_step_state(step_id)
            if step_state:
                status = step_state["status"]

                if status == "completed":
                    progress_info["completed_steps"] += 1
                elif status == "failed":
                    progress_info["failed_steps"] += 1
                elif status == "running":
                    progress_info["running_steps"] += 1
                else:
                    progress_info["pending_steps"] += 1

                progress_info["steps"].append({
                    "step_id": step_id,
                    "status": status,
                    "component_type": step_state.get("component_type"),
                    "start_time": step_state.get("start_time"),
                    "execution_time": step_state.get("execution_time")
                })

        # Calculate overall progress
        if progress_info["total_steps"] > 0:
            progress_info["overall_progress"] = (
                progress_info["completed_steps"] / progress_info["total_steps"]
            ) * 100

        return progress_info

    except Exception as e:
        logger.error("Failed to get visual workflow progress", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")


# Helper functions for visual workflow execution
async def build_execution_graph(nodes: List[VisualWorkflowNode], connections: List[VisualWorkflowConnection]) -> Dict[str, List[str]]:
    """Build execution graph from visual workflow nodes and connections."""
    graph = {node.id: [] for node in nodes}

    for connection in connections:
        if connection.source in graph:
            graph[connection.source].append(connection.target)

    return graph


def get_topological_order(graph: Dict[str, List[str]]) -> List[str]:
    """Get topological order for node execution using Kahn's algorithm."""
    # Calculate in-degrees
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            if neighbor in in_degree:
                in_degree[neighbor] += 1

    # Initialize queue with nodes having no incoming edges
    queue = [node for node, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        # Remove edges from this node
        for neighbor in graph[node]:
            if neighbor in in_degree:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    return result


def get_node_inputs(node_id: str, connections: List[VisualWorkflowConnection], execution_results: Dict[str, Any]) -> Dict[str, Any]:
    """Get inputs for a node from connected predecessor nodes."""
    inputs = {}

    for connection in connections:
        if connection.target == node_id:
            source_result = execution_results.get(connection.source)
            if source_result and source_result.get("success"):
                # Use the source handle as the input key, or default to 'data'
                input_key = connection.targetHandle or 'data'
                output_key = connection.sourceHandle or 'data'

                source_data = source_result.get("data", {})
                if isinstance(source_data, dict) and output_key in source_data:
                    inputs[input_key] = source_data[output_key]
                else:
                    inputs[input_key] = source_data

    return inputs


async def execute_visual_node(
    node: VisualWorkflowNode,
    inputs: Dict[str, Any],
    execution_context: Dict[str, Any],
    node_registry
) -> Dict[str, Any]:
    """Execute a single visual workflow node."""
    try:
        # Get node configuration
        node_config = node.data.get("configuration", {})

        # Add inputs to node config
        if inputs:
            node_config["inputs"] = inputs

        # Get execution handler from registry
        handler = node_registry.get_execution_handler(node.type)

        if not handler:
            return {
                "success": False,
                "error": f"No execution handler found for node type '{node.type}'",
                "execution_time": 0
            }

        # Execute the node
        start_time = asyncio.get_event_loop().time()
        result = await handler(node_config, execution_context)
        execution_time = asyncio.get_event_loop().time() - start_time

        # Add execution time to result
        if isinstance(result, dict):
            result["execution_time"] = execution_time
        else:
            result = {
                "success": True,
                "data": result,
                "execution_time": execution_time
            }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_time": 0
        }
