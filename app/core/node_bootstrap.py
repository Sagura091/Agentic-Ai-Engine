"""
Node Bootstrap System

This module handles the registration and initialization of all workflow nodes.
It automatically registers all available nodes with the node registry.
"""

import asyncio
from typing import List

from app.core.node_registry import get_node_registry, RegisteredNode
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()

# Import all node definitions
from app.nodes.timer_node import create_timer_node_definition
from app.nodes.sequential_execution_node import create_sequential_execution_node_definition
from app.nodes.agent_message_node import create_agent_message_node_definition


class NodeBootstrap:
    """Bootstrap system for registering all workflow nodes."""
    
    def __init__(self):
        self._registered_nodes: List[RegisteredNode] = []
        self._initialized = False
    
    async def initialize_all_nodes(self):
        """Initialize and register all available nodes."""
        if self._initialized:
            return
        
        backend_logger.info(
            "Starting node bootstrap initialization",
            LogCategory.SYSTEM_HEALTH,
            "NodeBootstrap"
        )
        
        # Get the node registry
        registry = await get_node_registry()
        
        # Define all available node creation functions
        node_creators = [
            create_timer_node_definition,
            create_sequential_execution_node_definition,
            create_agent_message_node_definition,
        ]
        
        # Register each node
        for creator in node_creators:
            try:
                node_definition = creator()
                success = registry.register_node(node_definition)
                
                if success:
                    self._registered_nodes.append(node_definition)
                    
                    # Persist to database
                    await registry.persist_node_definition(node_definition)
                    
                    backend_logger.info(
                        f"Node '{node_definition.node_type}' registered successfully",
                        LogCategory.SYSTEM_HEALTH,
                        "NodeBootstrap",
                        data={
                            "node_type": node_definition.node_type,
                            "node_name": node_definition.name,
                            "category": node_definition.category.value
                        }
                    )
                else:
                    backend_logger.error(
                        f"Failed to register node '{node_definition.node_type}'",
                        LogCategory.SYSTEM_HEALTH,
                        "NodeBootstrap"
                    )
                    
            except Exception as e:
                backend_logger.error(
                    f"Error creating node definition with {creator.__name__}",
                    LogCategory.SYSTEM_HEALTH,
                    "NodeBootstrap",
                    error=str(e)
                )
        
        self._initialized = True
        
        backend_logger.info(
            "Node bootstrap initialization completed",
            LogCategory.SYSTEM_HEALTH,
            "NodeBootstrap",
            data={
                "total_nodes_registered": len(self._registered_nodes),
                "registered_nodes": [node.node_type for node in self._registered_nodes]
            }
        )
    
    def get_registered_nodes(self) -> List[RegisteredNode]:
        """Get all registered nodes."""
        return self._registered_nodes.copy()
    
    def is_initialized(self) -> bool:
        """Check if the bootstrap system is initialized."""
        return self._initialized


# Global bootstrap instance
node_bootstrap = NodeBootstrap()


async def initialize_node_system():
    """Initialize the entire node system."""
    await node_bootstrap.initialize_all_nodes()


async def get_node_bootstrap() -> NodeBootstrap:
    """Get the global node bootstrap instance."""
    if not node_bootstrap.is_initialized():
        await node_bootstrap.initialize_all_nodes()
    return node_bootstrap
