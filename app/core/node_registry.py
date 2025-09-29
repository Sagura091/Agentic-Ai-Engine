"""
Node Registry System for Advanced Workflow Nodes

This module provides a centralized registry for managing node types,
their definitions, execution handlers, and validation rules.
"""

import asyncio
import inspect
from typing import Dict, List, Any, Optional, Callable, Type
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from app.models.workflow import NodeDefinition
from app.models.database.base import get_database_session
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()


class PortType(str, Enum):
    """Port types for node connections."""
    DATA = "data"
    CONTROL = "control"
    TEXT = "text"
    JSON = "json"
    VECTOR = "vector"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    MEMORY = "memory"
    AGENT = "agent"
    NUMBER = "number"
    BOOLEAN = "boolean"


class NodeCategory(str, Enum):
    """Node categories for organization."""
    AGENT_FRAMEWORKS = "agent_frameworks"
    TOOLS = "tools"
    MEMORY = "memory"
    DOCUMENT_PROCESSING = "document_processing"
    EMBEDDINGS = "embeddings"
    INTEGRATIONS = "integrations"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    SECURITY = "security"
    DATA_OPERATIONS = "data_operations"
    WORKFLOW = "workflow"
    SPECIALIZED = "specialized"
    CONTROL_FLOW = "control_flow"


@dataclass
class NodePort:
    """Definition of a node port."""
    id: str
    name: str
    type: PortType
    required: bool = True
    description: str = ""
    default_value: Any = None


@dataclass
class NodeConnectionRule:
    """Rules for node connections."""
    allowed_input_types: List[PortType]
    allowed_output_types: List[PortType]
    max_input_connections: int = -1  # -1 for unlimited
    max_output_connections: int = -1  # -1 for unlimited
    semantic_validation: Optional[Callable] = None


@dataclass
class RegisteredNode:
    """A registered node type with all its metadata."""
    node_type: str
    name: str
    description: str
    category: NodeCategory
    input_ports: List[NodePort]
    output_ports: List[NodePort]
    configuration_schema: Dict[str, Any]
    default_configuration: Dict[str, Any]
    execution_handler: Callable
    connection_rules: NodeConnectionRule
    icon: str = "🔧"
    color: str = "bg-gray-500"
    is_experimental: bool = False
    execution_timeout: int = 300


class NodeRegistry:
    """Central registry for all node types."""
    
    def __init__(self):
        self._nodes: Dict[str, RegisteredNode] = {}
        self._execution_handlers: Dict[str, Callable] = {}
        self._port_compatibility_matrix: Dict[PortType, List[PortType]] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the node registry."""
        if self._initialized:
            return
            
        backend_logger.info(
            "Initializing Node Registry System",
            LogCategory.SYSTEM_HEALTH,
            "NodeRegistry"
        )
        
        # Initialize port compatibility matrix
        self._setup_port_compatibility()
        
        # Load existing node definitions from database
        await self._load_node_definitions()
        
        self._initialized = True
        
        backend_logger.info(
            "Node Registry System initialized",
            LogCategory.SYSTEM_HEALTH,
            "NodeRegistry",
            data={"registered_nodes": len(self._nodes)}
        )
    
    def _setup_port_compatibility(self):
        """Setup the port type compatibility matrix."""
        self._port_compatibility_matrix = {
            PortType.DATA: [PortType.DATA, PortType.TEXT, PortType.JSON, PortType.NUMBER],
            PortType.TEXT: [PortType.TEXT, PortType.JSON, PortType.DATA],
            PortType.JSON: [PortType.JSON, PortType.TEXT, PortType.DATA],
            PortType.VECTOR: [PortType.VECTOR, PortType.MEMORY],
            PortType.MEMORY: [PortType.MEMORY, PortType.VECTOR],
            PortType.AGENT: [PortType.AGENT],
            PortType.CONTROL: [PortType.CONTROL],
            PortType.FILE: [PortType.FILE, PortType.DATA],
            PortType.IMAGE: [PortType.IMAGE, PortType.FILE],
            PortType.AUDIO: [PortType.AUDIO, PortType.FILE],
            PortType.NUMBER: [PortType.NUMBER, PortType.DATA, PortType.TEXT],
            PortType.BOOLEAN: [PortType.BOOLEAN, PortType.DATA]
        }
    
    async def _load_node_definitions(self):
        """Load existing node definitions from database."""
        try:
            async for session in get_database_session():
                result = await session.execute(select(NodeDefinition))
                definitions = result.scalars().all()
                
                for definition in definitions:
                    # Convert database model to RegisteredNode
                    # Note: execution_handler will need to be resolved separately
                    pass
                    
        except Exception as e:
            backend_logger.error(
                "Failed to load node definitions from database",
                LogCategory.SYSTEM_HEALTH,
                "NodeRegistry",
                error=str(e)
            )
    
    def register_node(self, node: RegisteredNode) -> bool:
        """Register a new node type."""
        try:
            if node.node_type in self._nodes:
                backend_logger.warning(
                    f"Node type '{node.node_type}' already registered, overwriting",
                    LogCategory.SYSTEM_HEALTH,
                    "NodeRegistry"
                )
            
            # Validate node definition
            self._validate_node_definition(node)
            
            # Register the node
            self._nodes[node.node_type] = node
            self._execution_handlers[node.node_type] = node.execution_handler
            
            backend_logger.info(
                f"Node type '{node.node_type}' registered successfully",
                LogCategory.SYSTEM_HEALTH,
                "NodeRegistry",
                data={"node_name": node.name, "category": node.category.value}
            )
            
            return True
            
        except Exception as e:
            backend_logger.error(
                f"Failed to register node type '{node.node_type}'",
                LogCategory.SYSTEM_HEALTH,
                "NodeRegistry",
                error=str(e)
            )
            return False
    
    def _validate_node_definition(self, node: RegisteredNode):
        """Validate a node definition."""
        # Validate node type
        if not node.node_type or not isinstance(node.node_type, str):
            raise ValueError("Node type must be a non-empty string")
        
        # Validate ports
        for port in node.input_ports + node.output_ports:
            if not isinstance(port.type, PortType):
                raise ValueError(f"Invalid port type: {port.type}")
        
        # Validate execution handler
        if not callable(node.execution_handler):
            raise ValueError("Execution handler must be callable")
        
        # Validate handler signature
        sig = inspect.signature(node.execution_handler)
        if len(sig.parameters) < 2:
            raise ValueError("Execution handler must accept at least 2 parameters (node_config, execution_context)")
    
    def get_node(self, node_type: str) -> Optional[RegisteredNode]:
        """Get a registered node by type."""
        return self._nodes.get(node_type)
    
    def get_all_nodes(self) -> Dict[str, RegisteredNode]:
        """Get all registered nodes."""
        return self._nodes.copy()
    
    def get_nodes_by_category(self, category: NodeCategory) -> List[RegisteredNode]:
        """Get all nodes in a specific category."""
        return [node for node in self._nodes.values() if node.category == category]
    
    def get_execution_handler(self, node_type: str) -> Optional[Callable]:
        """Get the execution handler for a node type."""
        return self._execution_handlers.get(node_type)
    
    def is_connection_valid(self, source_port_type: PortType, target_port_type: PortType) -> bool:
        """Check if a connection between two port types is valid."""
        compatible_types = self._port_compatibility_matrix.get(source_port_type, [])
        return target_port_type in compatible_types
    
    async def persist_node_definition(self, node: RegisteredNode):
        """Persist a node definition to the database."""
        try:
            async for session in get_database_session():
                # Check if definition already exists
                result = await session.execute(
                    select(NodeDefinition).where(NodeDefinition.node_type == node.node_type)
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Update existing definition
                    existing.name = node.name
                    existing.description = node.description
                    existing.category = node.category.value
                    existing.input_ports = [
                        {
                            "id": port.id,
                            "name": port.name,
                            "type": port.type.value,
                            "required": port.required,
                            "description": port.description,
                            "default_value": port.default_value
                        }
                        for port in node.input_ports
                    ]
                    existing.output_ports = [
                        {
                            "id": port.id,
                            "name": port.name,
                            "type": port.type.value,
                            "required": port.required,
                            "description": port.description,
                            "default_value": port.default_value
                        }
                        for port in node.output_ports
                    ]
                    existing.configuration_schema = node.configuration_schema
                    existing.default_configuration = node.default_configuration
                    existing.execution_handler = node.execution_handler.__name__
                    existing.execution_timeout = node.execution_timeout
                    existing.icon = node.icon
                    existing.color = node.color
                    existing.is_experimental = node.is_experimental
                else:
                    # Create new definition
                    definition = NodeDefinition(
                        node_type=node.node_type,
                        name=node.name,
                        description=node.description,
                        category=node.category.value,
                        input_ports=[
                            {
                                "id": port.id,
                                "name": port.name,
                                "type": port.type.value,
                                "required": port.required,
                                "description": port.description,
                                "default_value": port.default_value
                            }
                            for port in node.input_ports
                        ],
                        output_ports=[
                            {
                                "id": port.id,
                                "name": port.name,
                                "type": port.type.value,
                                "required": port.required,
                                "description": port.description,
                                "default_value": port.default_value
                            }
                            for port in node.output_ports
                        ],
                        configuration_schema=node.configuration_schema,
                        default_configuration=node.default_configuration,
                        execution_handler=node.execution_handler.__name__,
                        execution_timeout=node.execution_timeout,
                        icon=node.icon,
                        color=node.color,
                        is_experimental=node.is_experimental
                    )
                    session.add(definition)
                
                await session.commit()
                
                backend_logger.info(
                    f"Node definition '{node.node_type}' persisted to database",
                    LogCategory.SYSTEM_HEALTH,
                    "NodeRegistry"
                )
                
        except Exception as e:
            backend_logger.error(
                f"Failed to persist node definition '{node.node_type}'",
                LogCategory.SYSTEM_HEALTH,
                "NodeRegistry",
                error=str(e)
            )


# Global node registry instance
node_registry = NodeRegistry()


async def get_node_registry() -> NodeRegistry:
    """Get the global node registry instance."""
    if not node_registry._initialized:
        await node_registry.initialize()
    return node_registry
