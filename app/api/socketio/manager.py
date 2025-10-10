"""
Socket.IO manager for real-time communication.

This module provides Socket.IO support for the frontend to connect
using the socket.io-client library.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, Optional

import socketio
from fastapi import FastAPI

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

_backend_logger = get_backend_logger()


class SocketIOManager:
    """
    Manager for Socket.IO connections and real-time communication.
    """
    
    def __init__(self):
        """Initialize the Socket.IO manager."""
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",
            async_mode='asgi',
            logger=False,
            engineio_logger=False,
            allow_upgrades=True,
            ping_timeout=60,
            ping_interval=25,
            always_connect=True,  # Always allow connections
            cookie=None  # Disable cookie-based authentication
        )
        self.active_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.connection_metadata: Dict[str, Dict] = {}
        self.is_initialized = False
        
        # Setup event handlers
        self._setup_event_handlers()

        _backend_logger.info(
            "Socket.IO manager created",
            LogCategory.API_OPERATIONS,
            "app.api.socketio.manager"
        )
    
    def _setup_event_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle client connection."""
            _backend_logger.info(
                "Socket.IO connection attempt",
                LogCategory.API_OPERATIONS,
                "app.api.socketio.manager",
                data={"session_id": sid, "auth": auth, "environ_keys": list(environ.keys()) if environ else None}
            )

            # Always allow connection - no authentication required
            connection_id = str(uuid.uuid4())
            self.active_connections[sid] = connection_id
            self.connection_metadata[connection_id] = {
                'session_id': sid,
                'connected_at': asyncio.get_event_loop().time(),
                'environ': environ
            }

            _backend_logger.info(
                "Socket.IO client connected",
                LogCategory.API_OPERATIONS,
                "app.api.socketio.manager",
                data={
                    "session_id": sid,
                    "connection_id": connection_id,
                    "total_connections": len(self.active_connections)
                }
            )

            # Send welcome message
            await self.sio.emit('connection_established', {
                'connection_id': connection_id,
                'message': 'Connected to Agentic AI Service',
                'timestamp': asyncio.get_event_loop().time()
            }, room=sid)

            # Send initial system status
            await self._send_system_status(sid)

            return True  # Explicitly allow connection
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection."""
            connection_id = self.active_connections.get(sid)

            if connection_id:
                del self.active_connections[sid]
                if connection_id in self.connection_metadata:
                    del self.connection_metadata[connection_id]

            _backend_logger.info(
                "Socket.IO client disconnected",
                LogCategory.API_OPERATIONS,
                "app.api.socketio.manager",
                data={
                    "session_id": sid,
                    "connection_id": connection_id,
                    "total_connections": len(self.active_connections)
                }
            )
        
        @self.sio.event
        async def execute_agent(sid, data):
            """Handle agent execution request."""
            try:
                connection_id = self.active_connections.get(sid)
                if not connection_id:
                    await self.sio.emit('error', {'message': 'Invalid connection'}, room=sid)
                    return
                
                agent_id = data.get('agent_id')
                task = data.get('task')
                context = data.get('context', {})
                
                if not agent_id or not task:
                    await self.sio.emit('error', {'message': 'Missing agent_id or task'}, room=sid)
                    return
                
                # Send execution started notification
                await self.sio.emit('agent_execution_started', {
                    'agent_id': agent_id,
                    'task': task,
                    'timestamp': asyncio.get_event_loop().time()
                }, room=sid)
                
                # Execute agent task (mock for now)
                await asyncio.sleep(1)  # Simulate processing
                
                # Send result
                await self.sio.emit('agent_execution_completed', {
                    'agent_id': agent_id,
                    'task': task,
                    'result': f'Mock result for task: {task}',
                    'timestamp': asyncio.get_event_loop().time()
                }, room=sid)

                _backend_logger.info(
                    "Agent execution completed",
                    LogCategory.API_OPERATIONS,
                    "app.api.socketio.manager",
                    data={"agent_id": agent_id, "task": task}
                )

            except Exception as e:
                _backend_logger.error(
                    "Error executing agent",
                    LogCategory.API_OPERATIONS,
                    "app.api.socketio.manager",
                    data={"error": str(e)}
                )
                await self.sio.emit('error', {'message': f'Agent execution failed: {str(e)}'}, room=sid)
        
        @self.sio.event
        async def get_agents(sid, data):
            """Handle get agents request."""
            try:
                connection_id = self.active_connections.get(sid)
                if not connection_id:
                    await self.sio.emit('error', {'message': 'Invalid connection'}, room=sid)
                    return
                
                # Get agents from orchestrator
                from app.core.unified_system_orchestrator import get_orchestrator_with_compatibility
                enhanced_orchestrator = get_orchestrator_with_compatibility()

                agents_data = []
                if enhanced_orchestrator.is_initialized:
                    for agent_id, agent in enhanced_orchestrator.agents.items():
                        agents_data.append({
                            'id': agent_id,
                            'name': getattr(agent.config, 'name', f'Agent-{agent_id}') if hasattr(agent, 'config') else f'Agent-{agent_id}',
                            'status': 'active',
                            'type': 'general'
                        })
                
                await self.sio.emit('agents_list', {
                    'agents': agents_data,
                    'total_count': len(agents_data)
                }, room=sid)

                _backend_logger.info(
                    "Agents list sent",
                    LogCategory.API_OPERATIONS,
                    "app.api.socketio.manager",
                    data={"agents_count": len(agents_data)}
                )

            except Exception as e:
                _backend_logger.error(
                    "Error getting agents",
                    LogCategory.API_OPERATIONS,
                    "app.api.socketio.manager",
                    data={"error": str(e)}
                )
                await self.sio.emit('error', {'message': f'Failed to get agents: {str(e)}'}, room=sid)
        
        @self.sio.event
        async def create_agent(sid, data):
            """Handle agent creation request."""
            try:
                connection_id = self.active_connections.get(sid)
                if not connection_id:
                    await self.sio.emit('error', {'message': 'Invalid connection'}, room=sid)
                    return

                # Extract agent data
                agent_data = data.get('data', {})
                agent_type = agent_data.get('agent_type', 'basic')
                name = agent_data.get('name')
                description = agent_data.get('description')
                config = agent_data.get('config', {})
                tools = agent_data.get('tools', [])

                if not name or not description:
                    await self.sio.emit('error', {'message': 'Missing name or description'}, room=sid)
                    return

                # Create agent using enhanced orchestrator
                from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
                from app.agents.factory import AgentType
                enhanced_orchestrator = get_enhanced_system_orchestrator()

                # Convert string agent type to enum
                try:
                    agent_type_enum = AgentType(agent_type)
                except ValueError:
                    agent_type_enum = AgentType.BASIC

                agent_id = await enhanced_orchestrator.create_agent_unlimited(
                    agent_type=agent_type_enum,
                    name=name,
                    description=description,
                    config=config,
                    tools=tools
                )

                # Get agent info and ensure datetime serialization
                agent_info = enhanced_orchestrator.agent_registry.get(agent_id, {})

                # Serialize any datetime objects in agent_info
                def serialize_datetime(obj):
                    from datetime import datetime
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: serialize_datetime(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_datetime(item) for item in obj]
                    return obj

                serialized_agent_info = serialize_datetime(agent_info)

                # Send creation result
                await self.sio.emit('agent_created', {
                    'agent_id': agent_id,
                    'agent_info': {
                        'name': name,
                        'description': description,
                        'agent_type': agent_type,
                        'status': 'active',
                        **serialized_agent_info
                    },
                    'timestamp': asyncio.get_event_loop().time()
                }, room=sid)

                _backend_logger.info(
                    "Agent created via Socket.IO",
                    LogCategory.API_OPERATIONS,
                    "app.api.socketio.manager",
                    data={"agent_id": agent_id, "name": name}
                )

            except Exception as e:
                _backend_logger.error(
                    "Error creating agent via Socket.IO",
                    LogCategory.API_OPERATIONS,
                    "app.api.socketio.manager",
                    data={"error": str(e)}
                )
                await self.sio.emit('error', {'message': f'Agent creation failed: {str(e)}'}, room=sid)

        @self.sio.event
        async def ping(sid, data):
            """Handle ping request."""
            await self.sio.emit('pong', {
                'timestamp': asyncio.get_event_loop().time(),
                'message': 'pong'
            }, room=sid)
    
    async def _send_system_status(self, sid: str):
        """Send system status to a specific client."""
        try:
            from app.core.unified_system_orchestrator import get_orchestrator_with_compatibility
            enhanced_orchestrator = get_orchestrator_with_compatibility()

            status = {
                'system_status': 'healthy',
                'orchestrator_initialized': enhanced_orchestrator.status.is_initialized,
                'active_agents': len(enhanced_orchestrator.agents) if enhanced_orchestrator.status.is_initialized else 0,
                'active_workflows': len(enhanced_orchestrator.workflows) if enhanced_orchestrator.status.is_initialized else 0,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            await self.sio.emit('system_status', status, room=sid)

        except Exception as e:
            _backend_logger.error(
                "Error sending system status",
                LogCategory.API_OPERATIONS,
                "app.api.socketio.manager",
                data={"error": str(e)}
            )
    
    async def initialize(self) -> None:
        """Initialize the Socket.IO manager."""
        if self.is_initialized:
            return

        self.is_initialized = True
        _backend_logger.info(
            "Socket.IO manager initialized",
            LogCategory.API_OPERATIONS,
            "app.api.socketio.manager"
        )
    
    def mount_to_app(self, app: FastAPI) -> None:
        """Mount Socket.IO to FastAPI app."""
        # Create ASGI app for Socket.IO
        sio_asgi_app = socketio.ASGIApp(self.sio, other_asgi_app=app)

        # Replace the app's __call__ method to handle Socket.IO
        app.mount("/socket.io", sio_asgi_app)

        _backend_logger.info(
            "Socket.IO mounted to FastAPI app",
            LogCategory.API_OPERATIONS,
            "app.api.socketio.manager"
        )
    
    async def broadcast(self, event: str, data: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            event: Event name
            data: Data to broadcast
        """
        if not self.active_connections:
            return
        
        await self.sio.emit(event, data)

        _backend_logger.info(
            "Message broadcasted",
            LogCategory.API_OPERATIONS,
            "app.api.socketio.manager",
            data={
                "event": event,
                "connections": len(self.active_connections)
            }
        )
    
    async def send_to_connection(self, connection_id: str, event: str, data: Dict[str, Any]) -> None:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: Target connection ID
            event: Event name
            data: Data to send
        """
        # Find session ID for connection ID
        session_id = None
        for sid, cid in self.active_connections.items():
            if cid == connection_id:
                session_id = sid
                break
        
        if session_id:
            await self.sio.emit(event, data, room=session_id)
        else:
            _backend_logger.warn(
                "Connection not found",
                LogCategory.API_OPERATIONS,
                "app.api.socketio.manager",
                data={"connection_id": connection_id}
            )
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)


# Global Socket.IO manager instance
socketio_manager = SocketIOManager()
