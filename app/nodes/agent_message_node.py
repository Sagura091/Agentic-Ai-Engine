"""
AGENT_MESSAGE Node Implementation

This module implements the AGENT_MESSAGE node for inter-agent communication.
The AGENT_MESSAGE node enables agents to send messages to each other within workflows.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from uuid import uuid4

from app.core.node_registry import (
    RegisteredNode, NodePort, NodeConnectionRule, PortType, NodeCategory
)
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()


@dataclass
class AgentMessageContext:
    """Execution context for agent message operations."""
    node_id: str
    workflow_id: str
    execution_id: str
    configuration: Dict[str, Any]
    input_data: Dict[str, Any]
    start_time: datetime


class AgentMessageQueue:
    """Simple in-memory message queue for agent communication."""
    
    def __init__(self):
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        self._message_history: List[Dict[str, Any]] = []
    
    async def send_message(self, sender_id: str, recipient_id: str, message: Dict[str, Any]) -> str:
        """Send a message from one agent to another."""
        message_id = str(uuid4())
        
        message_envelope = {
            "message_id": message_id,
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status": "sent"
        }
        
        # Add to recipient's queue
        if recipient_id not in self._messages:
            self._messages[recipient_id] = []
        
        self._messages[recipient_id].append(message_envelope)
        
        # Add to history
        self._message_history.append(message_envelope)
        
        backend_logger.debug(
            f"Message sent from {sender_id} to {recipient_id}",
            LogCategory.ORCHESTRATION,
            "AgentMessageNode",
            data={
                "message_id": message_id,
                "sender": sender_id,
                "recipient": recipient_id
            }
        )
        
        return message_id
    
    async def receive_messages(self, agent_id: str, mark_as_read: bool = True) -> List[Dict[str, Any]]:
        """Receive messages for an agent."""
        messages = self._messages.get(agent_id, [])
        
        if mark_as_read:
            # Mark messages as read
            for msg in messages:
                msg["status"] = "read"
                msg["read_at"] = datetime.now().isoformat()
            
            # Clear the queue
            self._messages[agent_id] = []
        
        return messages
    
    async def get_message_history(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get message history for an agent or all messages."""
        if agent_id:
            return [
                msg for msg in self._message_history
                if msg["sender_id"] == agent_id or msg["recipient_id"] == agent_id
            ]
        return self._message_history.copy()


# Global message queue instance
message_queue = AgentMessageQueue()


class AgentMessageNodeExecutor:
    """Executor for AGENT_MESSAGE nodes."""
    
    @staticmethod
    async def execute_agent_message_node(node_config: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an AGENT_MESSAGE node.
        
        Args:
            node_config: Node configuration including message settings
            execution_context: Execution context with workflow information
            
        Returns:
            Dict containing execution results
        """
        try:
            context = AgentMessageContext(
                node_id=execution_context.get('node_id', ''),
                workflow_id=execution_context.get('workflow_id', ''),
                execution_id=execution_context.get('execution_id', ''),
                configuration=node_config,
                input_data=execution_context.get('input_data', {}),
                start_time=datetime.now()
            )
            
            backend_logger.info(
                "Starting AGENT_MESSAGE node execution",
                LogCategory.ORCHESTRATION,
                "AgentMessageNode",
                data={
                    "node_id": context.node_id,
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id
                }
            )
            
            # Get message configuration
            message_type = node_config.get('message_type', 'send')  # send, receive, broadcast
            sender_id = node_config.get('sender_id', context.node_id)
            recipient_id = node_config.get('recipient_id', '')
            message_content = node_config.get('message_content', '')
            
            # Override with input data if provided
            input_data = context.input_data
            if isinstance(input_data, dict):
                sender_id = input_data.get('sender_id', sender_id)
                recipient_id = input_data.get('recipient_id', recipient_id)
                message_content = input_data.get('message', message_content)
            
            result = {}
            
            if message_type == 'send':
                result = await AgentMessageNodeExecutor._send_message(
                    sender_id, recipient_id, message_content, context
                )
            elif message_type == 'receive':
                result = await AgentMessageNodeExecutor._receive_messages(
                    sender_id, context  # sender_id acts as receiver in this case
                )
            elif message_type == 'broadcast':
                result = await AgentMessageNodeExecutor._broadcast_message(
                    sender_id, message_content, node_config, context
                )
            else:
                raise ValueError(f"Unknown message type: {message_type}")
            
            execution_time = (datetime.now() - context.start_time).total_seconds()
            
            backend_logger.info(
                "AGENT_MESSAGE node execution completed",
                LogCategory.ORCHESTRATION,
                "AgentMessageNode",
                data={
                    "node_id": context.node_id,
                    "execution_time_seconds": execution_time,
                    "message_type": message_type
                }
            )
            
            return {
                "success": True,
                "data": {
                    "message_type": message_type,
                    "execution_time_seconds": execution_time,
                    "result": result,
                    "completed_at": datetime.now().isoformat()
                },
                "metadata": {
                    "node_type": "AGENT_MESSAGE",
                    "execution_context": {
                        "node_id": context.node_id,
                        "workflow_id": context.workflow_id,
                        "execution_id": context.execution_id
                    }
                }
            }
            
        except Exception as e:
            backend_logger.error(
                "AGENT_MESSAGE node execution failed",
                LogCategory.ORCHESTRATION,
                "AgentMessageNode",
                error=str(e),
                data={
                    "node_id": execution_context.get('node_id', ''),
                    "workflow_id": execution_context.get('workflow_id', '')
                }
            )
            
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "node_type": "AGENT_MESSAGE",
                    "execution_context": execution_context
                }
            }
    
    @staticmethod
    async def _send_message(sender_id: str, recipient_id: str, message_content: Any, context: AgentMessageContext) -> Dict[str, Any]:
        """Send a message to another agent."""
        if not recipient_id:
            raise ValueError("Recipient ID is required for sending messages")
        
        message = {
            "content": message_content,
            "sender_context": {
                "node_id": context.node_id,
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id
            },
            "message_metadata": {
                "sent_at": datetime.now().isoformat(),
                "message_type": "direct"
            }
        }
        
        message_id = await message_queue.send_message(sender_id, recipient_id, message)
        
        return {
            "action": "send",
            "message_id": message_id,
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "message_content": message_content,
            "status": "sent"
        }
    
    @staticmethod
    async def _receive_messages(agent_id: str, context: AgentMessageContext) -> Dict[str, Any]:
        """Receive messages for an agent."""
        messages = await message_queue.receive_messages(agent_id, mark_as_read=True)
        
        return {
            "action": "receive",
            "agent_id": agent_id,
            "message_count": len(messages),
            "messages": messages,
            "received_at": datetime.now().isoformat()
        }
    
    @staticmethod
    async def _broadcast_message(sender_id: str, message_content: Any, node_config: Dict[str, Any], context: AgentMessageContext) -> Dict[str, Any]:
        """Broadcast a message to multiple agents."""
        recipient_ids = node_config.get('broadcast_recipients', [])
        
        if not recipient_ids:
            raise ValueError("Broadcast recipients list is required for broadcasting")
        
        message = {
            "content": message_content,
            "sender_context": {
                "node_id": context.node_id,
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id
            },
            "message_metadata": {
                "sent_at": datetime.now().isoformat(),
                "message_type": "broadcast"
            }
        }
        
        sent_messages = []
        
        for recipient_id in recipient_ids:
            try:
                message_id = await message_queue.send_message(sender_id, recipient_id, message)
                sent_messages.append({
                    "recipient_id": recipient_id,
                    "message_id": message_id,
                    "status": "sent"
                })
            except Exception as e:
                sent_messages.append({
                    "recipient_id": recipient_id,
                    "message_id": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        successful_sends = len([msg for msg in sent_messages if msg["status"] == "sent"])
        
        return {
            "action": "broadcast",
            "sender_id": sender_id,
            "total_recipients": len(recipient_ids),
            "successful_sends": successful_sends,
            "failed_sends": len(recipient_ids) - successful_sends,
            "message_content": message_content,
            "sent_messages": sent_messages
        }


def create_agent_message_node_definition() -> RegisteredNode:
    """Create the AGENT_MESSAGE node definition."""
    return RegisteredNode(
        node_type="AGENT_MESSAGE",
        name="Agent Message",
        description="Inter-agent communication for sending and receiving messages",
        category=NodeCategory.COMMUNICATION,
        input_ports=[
            NodePort(
                id="message_data",
                name="Message Data",
                type=PortType.JSON,
                required=False,
                description="Message data including sender, recipient, and content"
            ),
            NodePort(
                id="agent_input",
                name="Agent Input",
                type=PortType.AGENT,
                required=False,
                description="Agent context for message operations"
            )
        ],
        output_ports=[
            NodePort(
                id="message_result",
                name="Message Result",
                type=PortType.JSON,
                required=True,
                description="Result of message operation"
            ),
            NodePort(
                id="agent_output",
                name="Agent Output",
                type=PortType.AGENT,
                required=False,
                description="Agent context after message operation"
            )
        ],
        configuration_schema={
            "type": "object",
            "properties": {
                "message_type": {
                    "type": "string",
                    "enum": ["send", "receive", "broadcast"],
                    "default": "send",
                    "description": "Type of message operation"
                },
                "sender_id": {
                    "type": "string",
                    "default": "",
                    "description": "ID of the sending agent"
                },
                "recipient_id": {
                    "type": "string",
                    "default": "",
                    "description": "ID of the receiving agent (for send operation)"
                },
                "message_content": {
                    "type": "string",
                    "default": "",
                    "description": "Content of the message to send"
                },
                "broadcast_recipients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "List of recipient IDs for broadcast operation"
                }
            },
            "required": ["message_type"]
        },
        default_configuration={
            "message_type": "send",
            "sender_id": "",
            "recipient_id": "",
            "message_content": "",
            "broadcast_recipients": []
        },
        execution_handler=AgentMessageNodeExecutor.execute_agent_message_node,
        connection_rules=NodeConnectionRule(
            allowed_input_types=[PortType.JSON, PortType.AGENT],
            allowed_output_types=[PortType.JSON, PortType.AGENT],
            max_input_connections=-1,
            max_output_connections=-1
        ),
        icon="💬",
        color="bg-green-500",
        execution_timeout=60  # 1 minute max
    )
