"""
Workflow Node Implementations

This package contains all workflow node type implementations for the
Agentic AI Engine's workflow orchestration system.

Available Nodes:
- AGENT_MESSAGE: Inter-agent communication for sending and receiving messages
- SEQUENTIAL_EXECUTION: Execute multiple inputs in a defined sequential order
- TIMER: Time-based triggers and delays for workflow control

Each node provides:
- Node definition with metadata (name, description, category, icon, color)
- Input/output port definitions
- Configuration schema
- Execution handler
- Connection rules

Usage:
    from app.nodes import (
        create_agent_message_node_definition,
        create_sequential_execution_node_definition,
        create_timer_node_definition
    )
    
    # Create node definitions
    timer_node = create_timer_node_definition()
    sequential_node = create_sequential_execution_node_definition()
    message_node = create_agent_message_node_definition()
"""

from .agent_message_node import (
    create_agent_message_node_definition,
    AgentMessageNodeExecutor,
    AgentMessageQueue,
    message_queue
)
from .sequential_execution_node import (
    create_sequential_execution_node_definition,
    SequentialExecutionNodeExecutor
)
from .timer_node import (
    create_timer_node_definition,
    TimerNodeExecutor
)

__all__ = [
    # Node definition creators
    "create_agent_message_node_definition",
    "create_sequential_execution_node_definition",
    "create_timer_node_definition",
    
    # Node executors (for advanced usage)
    "AgentMessageNodeExecutor",
    "SequentialExecutionNodeExecutor",
    "TimerNodeExecutor",
    
    # Utilities
    "AgentMessageQueue",
    "message_queue",
]

