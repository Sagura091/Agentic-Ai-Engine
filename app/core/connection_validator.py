"""
Connection Validation System for Advanced Workflow Nodes

This module provides comprehensive validation for node connections,
including port type compatibility, semantic validation, and connection rules.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.core.node_registry import NodeRegistry, PortType, RegisteredNode, get_node_registry
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()


class ValidationResult(str, Enum):
    """Validation result types."""
    VALID = "valid"
    INVALID_PORT_TYPE = "invalid_port_type"
    INVALID_SAME_NODE = "invalid_same_node"
    INVALID_MAX_CONNECTIONS = "invalid_max_connections"
    INVALID_SEMANTIC = "invalid_semantic"
    INVALID_NODE_TYPE = "invalid_node_type"
    INVALID_PORT_NOT_FOUND = "invalid_port_not_found"


@dataclass
class ConnectionValidationRequest:
    """Request for connection validation."""
    source_node_type: str
    source_port_id: str
    target_node_type: str
    target_port_id: str
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    existing_connections: Optional[List[Dict[str, Any]]] = None


@dataclass
class ConnectionValidationResponse:
    """Response from connection validation."""
    is_valid: bool
    result: ValidationResult
    message: str
    suggestions: List[str] = None


class ConnectionValidator:
    """Validates connections between workflow nodes."""
    
    def __init__(self):
        self._node_registry: Optional[NodeRegistry] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connection validator."""
        if self._initialized:
            return
            
        self._node_registry = await get_node_registry()
        self._initialized = True
        
        backend_logger.info(
            "Connection Validator initialized",
            LogCategory.SYSTEM_OPERATIONS,
            "ConnectionValidator"
        )
    
    async def validate_connection(self, request: ConnectionValidationRequest) -> ConnectionValidationResponse:
        """Validate a connection between two nodes."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get source and target nodes
            source_node = self._node_registry.get_node(request.source_node_type)
            target_node = self._node_registry.get_node(request.target_node_type)
            
            if not source_node:
                return ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_NODE_TYPE,
                    message=f"Source node type '{request.source_node_type}' not found"
                )
            
            if not target_node:
                return ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_NODE_TYPE,
                    message=f"Target node type '{request.target_node_type}' not found"
                )
            
            # Check if connecting to same node
            if request.source_node_id and request.target_node_id:
                if request.source_node_id == request.target_node_id:
                    return ConnectionValidationResponse(
                        is_valid=False,
                        result=ValidationResult.INVALID_SAME_NODE,
                        message="Cannot connect a node to itself"
                    )
            
            # Find source and target ports
            source_port = self._find_port(source_node.output_ports, request.source_port_id)
            target_port = self._find_port(target_node.input_ports, request.target_port_id)
            
            if not source_port:
                return ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_PORT_NOT_FOUND,
                    message=f"Source port '{request.source_port_id}' not found in node '{request.source_node_type}'"
                )
            
            if not target_port:
                return ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_PORT_NOT_FOUND,
                    message=f"Target port '{request.target_port_id}' not found in node '{request.target_node_type}'"
                )
            
            # Check port type compatibility
            if not self._node_registry.is_connection_valid(source_port.type, target_port.type):
                compatible_types = self._get_compatible_types(source_port.type)
                return ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_PORT_TYPE,
                    message=f"Port types incompatible: {source_port.type.value} -> {target_port.type.value}",
                    suggestions=[f"Compatible target types: {', '.join([t.value for t in compatible_types])}"]
                )
            
            # Check connection limits
            if request.existing_connections:
                connection_limit_result = self._check_connection_limits(
                    source_node, target_node, request, request.existing_connections
                )
                if not connection_limit_result.is_valid:
                    return connection_limit_result
            
            # Perform semantic validation
            semantic_result = await self._perform_semantic_validation(
                source_node, target_node, request
            )
            if not semantic_result.is_valid:
                return semantic_result
            
            return ConnectionValidationResponse(
                is_valid=True,
                result=ValidationResult.VALID,
                message="Connection is valid"
            )
            
        except Exception as e:
            backend_logger.error(
                "Error during connection validation",
                LogCategory.SYSTEM_OPERATIONS,
                "ConnectionValidator",
                error=str(e)
            )
            return ConnectionValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_SEMANTIC,
                message=f"Validation error: {str(e)}"
            )
    
    def _find_port(self, ports: List, port_id: str):
        """Find a port by ID in a list of ports."""
        for port in ports:
            if port.id == port_id:
                return port
        return None
    
    def _get_compatible_types(self, port_type: PortType) -> List[PortType]:
        """Get compatible port types for a given port type."""
        return self._node_registry._port_compatibility_matrix.get(port_type, [])
    
    def _check_connection_limits(
        self, 
        source_node: RegisteredNode, 
        target_node: RegisteredNode,
        request: ConnectionValidationRequest,
        existing_connections: List[Dict[str, Any]]
    ) -> ConnectionValidationResponse:
        """Check if connection limits are exceeded."""
        
        # Count existing connections for source node
        source_output_connections = len([
            conn for conn in existing_connections
            if conn.get('source_node_id') == request.source_node_id
        ])
        
        # Count existing connections for target node
        target_input_connections = len([
            conn for conn in existing_connections
            if conn.get('target_node_id') == request.target_node_id
        ])
        
        # Check source node output limit
        if (source_node.connection_rules.max_output_connections != -1 and 
            source_output_connections >= source_node.connection_rules.max_output_connections):
            return ConnectionValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_MAX_CONNECTIONS,
                message=f"Source node '{source_node.name}' has reached maximum output connections ({source_node.connection_rules.max_output_connections})"
            )
        
        # Check target node input limit
        if (target_node.connection_rules.max_input_connections != -1 and 
            target_input_connections >= target_node.connection_rules.max_input_connections):
            return ConnectionValidationResponse(
                is_valid=False,
                result=ValidationResult.INVALID_MAX_CONNECTIONS,
                message=f"Target node '{target_node.name}' has reached maximum input connections ({target_node.connection_rules.max_input_connections})"
            )
        
        return ConnectionValidationResponse(
            is_valid=True,
            result=ValidationResult.VALID,
            message="Connection limits OK"
        )
    
    async def _perform_semantic_validation(
        self,
        source_node: RegisteredNode,
        target_node: RegisteredNode,
        request: ConnectionValidationRequest
    ) -> ConnectionValidationResponse:
        """Perform semantic validation based on node-specific rules."""
        
        # Check if source node has semantic validation rules
        if source_node.connection_rules.semantic_validation:
            try:
                is_valid = await source_node.connection_rules.semantic_validation(
                    source_node, target_node, request
                )
                if not is_valid:
                    return ConnectionValidationResponse(
                        is_valid=False,
                        result=ValidationResult.INVALID_SEMANTIC,
                        message=f"Semantic validation failed for source node '{source_node.name}'"
                    )
            except Exception as e:
                return ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_SEMANTIC,
                    message=f"Semantic validation error: {str(e)}"
                )
        
        # Check if target node has semantic validation rules
        if target_node.connection_rules.semantic_validation:
            try:
                is_valid = await target_node.connection_rules.semantic_validation(
                    source_node, target_node, request
                )
                if not is_valid:
                    return ConnectionValidationResponse(
                        is_valid=False,
                        result=ValidationResult.INVALID_SEMANTIC,
                        message=f"Semantic validation failed for target node '{target_node.name}'"
                    )
            except Exception as e:
                return ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_SEMANTIC,
                    message=f"Semantic validation error: {str(e)}"
                )
        
        return ConnectionValidationResponse(
            is_valid=True,
            result=ValidationResult.VALID,
            message="Semantic validation passed"
        )
    
    async def validate_workflow_connections(self, workflow_data: Dict[str, Any]) -> List[ConnectionValidationResponse]:
        """Validate all connections in a workflow."""
        results = []
        
        nodes = workflow_data.get('nodes', [])
        connections = workflow_data.get('connections', [])
        
        for connection in connections:
            # Find source and target nodes
            source_node_data = next((n for n in nodes if n['id'] == connection['source']), None)
            target_node_data = next((n for n in nodes if n['id'] == connection['target']), None)
            
            if not source_node_data or not target_node_data:
                results.append(ConnectionValidationResponse(
                    is_valid=False,
                    result=ValidationResult.INVALID_NODE_TYPE,
                    message=f"Connection references non-existent nodes: {connection['id']}"
                ))
                continue
            
            request = ConnectionValidationRequest(
                source_node_type=source_node_data['type'],
                source_port_id=connection.get('sourceHandle', 'output'),
                target_node_type=target_node_data['type'],
                target_port_id=connection.get('targetHandle', 'input'),
                source_node_id=connection['source'],
                target_node_id=connection['target'],
                existing_connections=connections
            )
            
            result = await self.validate_connection(request)
            results.append(result)
        
        return results


# Global connection validator instance
connection_validator = ConnectionValidator()


async def get_connection_validator() -> ConnectionValidator:
    """Get the global connection validator instance."""
    if not connection_validator._initialized:
        await connection_validator.initialize()
    return connection_validator
