"""
WebSocket manager for real-time communication with agents.

This module provides WebSocket connection management for real-time
agent communication and status updates.
"""

import asyncio
import json
from typing import Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

_backend_logger = get_backend_logger()


class WebSocketManager:
    """
    Manager for WebSocket connections and real-time communication.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self.is_initialized = False

        _backend_logger.info(
            "WebSocket manager created",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.manager"
        )

    async def initialize(self) -> None:
        """Initialize the WebSocket manager."""
        if self.is_initialized:
            return

        self.is_initialized = True
        _backend_logger.info(
            "WebSocket manager initialized",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.manager"
        )

    async def shutdown(self) -> None:
        """Shutdown the WebSocket manager."""
        if not self.is_initialized:
            return

        # Close all active connections
        for connection_id in list(self.active_connections.keys()):
            await self.disconnect(connection_id)

        self.is_initialized = False
        _backend_logger.info(
            "WebSocket manager shut down",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.manager"
        )
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            connection_id: Unique connection identifier
            metadata: Optional connection metadata
        """
        await websocket.accept()
        
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = metadata or {}

        _backend_logger.info(
            "WebSocket connection established",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.manager",
            data={
                "connection_id": connection_id,
                "total_connections": len(self.active_connections)
            }
        )
        
        # Send welcome message
        await self.send_personal_message(
            connection_id,
            {
                "type": "connection_established",
                "connection_id": connection_id,
                "message": "Connected to Agentic AI Microservice"
            }
        )
    
    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect a WebSocket connection.
        
        Args:
            connection_id: Connection to disconnect
        """
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.close()
            except Exception as e:
                _backend_logger.warn(
                    "Error closing WebSocket",
                    LogCategory.API_OPERATIONS,
                    "app.api.websocket.manager",
                    data={
                        "connection_id": connection_id,
                        "error": str(e)
                    }
                )

            del self.active_connections[connection_id]
            del self.connection_metadata[connection_id]

            _backend_logger.info(
                "WebSocket connection closed",
                LogCategory.API_OPERATIONS,
                "app.api.websocket.manager",
                data={
                    "connection_id": connection_id,
                    "total_connections": len(self.active_connections)
                }
            )
    
    async def send_personal_message(
        self,
        connection_id: str,
        message: Dict
    ) -> None:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: Target connection
            message: Message to send
        """
        if connection_id not in self.active_connections:
            _backend_logger.warn(
                "Attempted to send message to non-existent connection",
                LogCategory.API_OPERATIONS,
                "app.api.websocket.manager",
                data={"connection_id": connection_id}
            )
            return

        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(message))

        except WebSocketDisconnect:
            _backend_logger.info(
                "WebSocket disconnected during message send",
                LogCategory.API_OPERATIONS,
                "app.api.websocket.manager",
                data={"connection_id": connection_id}
            )
            await self.disconnect(connection_id)

        except Exception as e:
            _backend_logger.error(
                "Error sending WebSocket message",
                LogCategory.API_OPERATIONS,
                "app.api.websocket.manager",
                data={
                    "connection_id": connection_id,
                    "error": str(e)
                }
            )
            await self.disconnect(connection_id)
    
    async def broadcast(self, message: Dict) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        if not self.active_connections:
            return
        
        # Send to all connections concurrently
        tasks = [
            self.send_personal_message(connection_id, message)
            for connection_id in list(self.active_connections.keys())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)

        _backend_logger.info(
            "Message broadcasted",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.manager",
            data={
                "message_type": message.get("type"),
                "connections": len(self.active_connections)
            }
        )
    
    async def send_agent_update(
        self,
        agent_id: str,
        status: str,
        data: Optional[Dict] = None
    ) -> None:
        """
        Send agent status update to all connections.
        
        Args:
            agent_id: Agent identifier
            status: Agent status
            data: Additional data
        """
        message = {
            "type": "agent_update",
            "agent_id": agent_id,
            "status": status,
            "data": data or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.broadcast(message)
    
    async def send_workflow_update(
        self,
        workflow_id: str,
        status: str,
        data: Optional[Dict] = None
    ) -> None:
        """
        Send workflow status update to all connections.
        
        Args:
            workflow_id: Workflow identifier
            status: Workflow status
            data: Additional data
        """
        message = {
            "type": "workflow_update",
            "workflow_id": workflow_id,
            "status": status,
            "data": data or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.broadcast(message)
    
    def get_connection_count(self) -> int:
        """
        Get the number of active connections.
        
        Returns:
            Number of active connections
        """
        return len(self.active_connections)
    
    def get_connections(self) -> List[str]:
        """
        Get list of active connection IDs.
        
        Returns:
            List of connection IDs
        """
        return list(self.active_connections.keys())


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
