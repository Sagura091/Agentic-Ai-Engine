"""
Node Management API Endpoints

This module provides API endpoints for managing workflow nodes,
including listing available nodes, getting node definitions, and executing nodes.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.node_registry import get_node_registry, NodeRegistry, PortType, NodeCategory
from app.core.node_bootstrap import get_node_bootstrap, NodeBootstrap
from app.core.connection_validator import get_connection_validator, ConnectionValidator, ConnectionValidationRequest
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()

router = APIRouter(tags=["nodes"])


# Pydantic models for API requests/responses
class NodePortResponse(BaseModel):
    """Response model for node ports."""
    id: str
    name: str
    type: str
    required: bool = True
    description: str = ""
    default_value: Any = None


class NodeDefinitionResponse(BaseModel):
    """Response model for node definitions."""
    node_type: str
    name: str
    description: str
    category: str
    input_ports: List[NodePortResponse]
    output_ports: List[NodePortResponse]
    configuration_schema: Dict[str, Any]
    default_configuration: Dict[str, Any]
    icon: str = "🔧"
    color: str = "bg-gray-500"
    is_experimental: bool = False
    execution_timeout: int = 300


class NodeExecutionRequest(BaseModel):
    """Request model for node execution."""
    node_type: str = Field(..., description="Type of node to execute")
    node_config: Dict[str, Any] = Field(default_factory=dict, description="Node configuration")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")


class NodeExecutionResponse(BaseModel):
    """Response model for node execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConnectionValidationResponse(BaseModel):
    """Response model for connection validation."""
    is_valid: bool
    result: str
    message: str
    suggestions: Optional[List[str]] = None


@router.get("/", response_model=List[NodeDefinitionResponse])
async def list_nodes(
    category: Optional[str] = None,
    registry: NodeRegistry = Depends(get_node_registry)
) -> List[NodeDefinitionResponse]:
    """
    List all available workflow nodes.
    
    Args:
        category: Optional category filter
        registry: Node registry dependency
        
    Returns:
        List of node definitions
    """
    try:
        all_nodes = registry.get_all_nodes()
        
        # Filter by category if specified
        if category:
            try:
                category_enum = NodeCategory(category)
                filtered_nodes = {
                    k: v for k, v in all_nodes.items() 
                    if v.category == category_enum
                }
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category: {category}"
                )
        else:
            filtered_nodes = all_nodes
        
        # Convert to response models
        response_nodes = []
        for node in filtered_nodes.values():
            response_nodes.append(NodeDefinitionResponse(
                node_type=node.node_type,
                name=node.name,
                description=node.description,
                category=node.category.value,
                input_ports=[
                    NodePortResponse(
                        id=port.id,
                        name=port.name,
                        type=port.type.value,
                        required=port.required,
                        description=port.description,
                        default_value=port.default_value
                    )
                    for port in node.input_ports
                ],
                output_ports=[
                    NodePortResponse(
                        id=port.id,
                        name=port.name,
                        type=port.type.value,
                        required=port.required,
                        description=port.description,
                        default_value=port.default_value
                    )
                    for port in node.output_ports
                ],
                configuration_schema=node.configuration_schema,
                default_configuration=node.default_configuration,
                icon=node.icon,
                color=node.color,
                is_experimental=node.is_experimental,
                execution_timeout=node.execution_timeout
            ))
        
        backend_logger.info(
            f"Listed {len(response_nodes)} nodes",
            LogCategory.API_LAYER,
            "NodesAPI",
            data={"category_filter": category, "node_count": len(response_nodes)}
        )
        
        return response_nodes
        
    except Exception as e:
        backend_logger.error(
            "Error listing nodes",
            LogCategory.API_LAYER,
            "NodesAPI",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing nodes: {str(e)}"
        )


@router.get("/{node_type}", response_model=NodeDefinitionResponse)
async def get_node_definition(
    node_type: str,
    registry: NodeRegistry = Depends(get_node_registry)
) -> NodeDefinitionResponse:
    """
    Get a specific node definition.
    
    Args:
        node_type: Type of node to retrieve
        registry: Node registry dependency
        
    Returns:
        Node definition
    """
    try:
        node = registry.get_node(node_type)
        
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node type '{node_type}' not found"
            )
        
        return NodeDefinitionResponse(
            node_type=node.node_type,
            name=node.name,
            description=node.description,
            category=node.category.value,
            input_ports=[
                NodePortResponse(
                    id=port.id,
                    name=port.name,
                    type=port.type.value,
                    required=port.required,
                    description=port.description,
                    default_value=port.default_value
                )
                for port in node.input_ports
            ],
            output_ports=[
                NodePortResponse(
                    id=port.id,
                    name=port.name,
                    type=port.type.value,
                    required=port.required,
                    description=port.description,
                    default_value=port.default_value
                )
                for port in node.output_ports
            ],
            configuration_schema=node.configuration_schema,
            default_configuration=node.default_configuration,
            icon=node.icon,
            color=node.color,
            is_experimental=node.is_experimental,
            execution_timeout=node.execution_timeout
        )
        
    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Error getting node definition for '{node_type}'",
            LogCategory.API_LAYER,
            "NodesAPI",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting node definition: {str(e)}"
        )


@router.post("/execute", response_model=NodeExecutionResponse)
async def execute_node(
    request: NodeExecutionRequest,
    registry: NodeRegistry = Depends(get_node_registry)
) -> NodeExecutionResponse:
    """
    Execute a workflow node.
    
    Args:
        request: Node execution request
        registry: Node registry dependency
        
    Returns:
        Node execution result
    """
    try:
        # Get the execution handler
        handler = registry.get_execution_handler(request.node_type)
        
        if not handler:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No execution handler found for node type '{request.node_type}'"
            )
        
        backend_logger.info(
            f"Executing node '{request.node_type}'",
            LogCategory.ORCHESTRATION,
            "NodesAPI",
            data={
                "node_type": request.node_type,
                "execution_context": request.execution_context
            }
        )
        
        # Execute the node
        result = await handler(request.node_config, request.execution_context)
        
        return NodeExecutionResponse(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error"),
            metadata=result.get("metadata")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        backend_logger.error(
            f"Error executing node '{request.node_type}'",
            LogCategory.ORCHESTRATION,
            "NodesAPI",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing node: {str(e)}"
        )


@router.post("/validate-connection", response_model=ConnectionValidationResponse)
async def validate_connection(
    request: ConnectionValidationRequest,
    validator: ConnectionValidator = Depends(get_connection_validator)
) -> ConnectionValidationResponse:
    """
    Validate a connection between two nodes.
    
    Args:
        request: Connection validation request
        validator: Connection validator dependency
        
    Returns:
        Connection validation result
    """
    try:
        result = await validator.validate_connection(request)
        
        return ConnectionValidationResponse(
            is_valid=result.is_valid,
            result=result.result.value,
            message=result.message,
            suggestions=result.suggestions
        )
        
    except Exception as e:
        backend_logger.error(
            "Error validating connection",
            LogCategory.API_LAYER,
            "NodesAPI",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating connection: {str(e)}"
        )


@router.get("/categories/", response_model=List[str])
async def list_node_categories() -> List[str]:
    """
    List all available node categories.
    
    Returns:
        List of node category names
    """
    return [category.value for category in NodeCategory]


@router.get("/port-types/", response_model=List[str])
async def list_port_types() -> List[str]:
    """
    List all available port types.
    
    Returns:
        List of port type names
    """
    return [port_type.value for port_type in PortType]


@router.get("/bootstrap/status")
async def get_bootstrap_status(
    bootstrap: NodeBootstrap = Depends(get_node_bootstrap)
) -> Dict[str, Any]:
    """
    Get the status of the node bootstrap system.
    
    Args:
        bootstrap: Node bootstrap dependency
        
    Returns:
        Bootstrap status information
    """
    registered_nodes = bootstrap.get_registered_nodes()
    
    return {
        "initialized": bootstrap.is_initialized(),
        "total_nodes": len(registered_nodes),
        "registered_nodes": [
            {
                "node_type": node.node_type,
                "name": node.name,
                "category": node.category.value
            }
            for node in registered_nodes
        ]
    }
