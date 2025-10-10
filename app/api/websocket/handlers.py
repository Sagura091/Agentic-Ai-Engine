"""
WebSocket handlers for real-time agent communication and collaboration.

This module provides WebSocket handlers for real-time communication
between the frontend and the agentic AI backend, including collaborative editing.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

from fastapi import WebSocket, WebSocketDisconnect

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

from app.api.websocket.manager import websocket_manager
from app.core.unified_system_orchestrator import get_orchestrator_with_compatibility
from app.core.seamless_integration import seamless_integration

_backend_logger = get_backend_logger()


async def handle_websocket_connection(websocket: WebSocket) -> None:
    """
    Handle WebSocket connection for real-time agent communication.
    
    Args:
        websocket: WebSocket connection
    """
    connection_id = str(uuid.uuid4())
    
    try:
        # Accept the connection
        await websocket_manager.connect(websocket, connection_id)

        _backend_logger.info(
            "WebSocket connection established",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id}
        )

        # Send initial system status
        await send_system_status(connection_id)

        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Route message to appropriate handler
                await route_message(connection_id, message)

            except WebSocketDisconnect:
                _backend_logger.info(
                    "WebSocket client disconnected",
                    LogCategory.API_OPERATIONS,
                    "app.api.websocket.handlers",
                    data={"connection_id": connection_id}
                )
                break
            except json.JSONDecodeError as e:
                _backend_logger.error(
                    "Invalid JSON received",
                    LogCategory.API_OPERATIONS,
                    "app.api.websocket.handlers",
                    data={"connection_id": connection_id, "error": str(e)}
                )
                await send_error(connection_id, "Invalid JSON format")
            except Exception as e:
                _backend_logger.error(
                    "Error handling WebSocket message",
                    LogCategory.API_OPERATIONS,
                    "app.api.websocket.handlers",
                    data={"connection_id": connection_id, "error": str(e)}
                )
                await send_error(connection_id, f"Error processing message: {str(e)}")

    except Exception as e:
        _backend_logger.error(
            "WebSocket connection error",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
    
    finally:
        # Clean up connection
        await websocket_manager.disconnect(connection_id)


async def route_message(connection_id: str, message: Dict[str, Any]) -> None:
    """
    Route incoming WebSocket message to appropriate handler.
    
    Args:
        connection_id: WebSocket connection ID
        message: Incoming message
    """
    message_type = message.get("type")
    
    if message_type == "execute_agent":
        await handle_execute_agent(connection_id, message)
    elif message_type == "create_agent":
        await handle_create_agent(connection_id, message)
    elif message_type == "create_tool":
        await handle_create_tool(connection_id, message)
    elif message_type == "execute_workflow":
        await handle_execute_workflow(connection_id, message)
    elif message_type == "execute_visual_workflow":
        await handle_execute_visual_workflow(connection_id, message)
    elif message_type == "get_system_status":
        await send_system_status(connection_id)
    elif message_type == "get_agents":
        await handle_get_agents(connection_id, message)
    elif message_type == "get_tools":
        await handle_get_tools(connection_id, message)
    elif message_type == "ping":
        await send_pong(connection_id)
    else:
        await send_error(connection_id, f"Unknown message type: {message_type}")


async def handle_execute_agent(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle agent execution request."""
    try:
        data = message.get("data", {})
        agent_id = data.get("agent_id")
        task = data.get("task")
        context = data.get("context", {})
        
        if not agent_id or not task:
            await send_error(connection_id, "Missing agent_id or task")
            return
        
        # Send execution started notification
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "agent_execution_started",
                "agent_id": agent_id,
                "task": task,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
        # Execute agent task
        orchestrator = get_orchestrator_with_compatibility()
        result = await orchestrator.execute_agent_task(
            agent_id=agent_id,
            task=task,
            context=context
        )
        
        # Send execution result
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "agent_execution_completed",
                "agent_id": agent_id,
                "task": task,
                "result": result,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        _backend_logger.error(
            "Error executing agent",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Agent execution failed: {str(e)}")


async def handle_create_agent(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle agent creation request."""
    try:
        data = message.get("data", {})
        agent_type = data.get("agent_type", "basic")
        name = data.get("name")
        description = data.get("description")
        config = data.get("config", {})
        tools = data.get("tools", [])
        
        if not name or not description:
            await send_error(connection_id, "Missing name or description")
            return
        
        # Create agent
        agent_id = await seamless_integration.create_unlimited_agent(
            agent_type=agent_type,
            name=name,
            description=description,
            config=config,
            tools=tools
        )
        
        # Get agent info and ensure datetime serialization
        orchestrator = get_orchestrator_with_compatibility()
        agent_info = orchestrator.agents.get(agent_id, {})

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
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "agent_created",
                "agent_id": agent_id,
                "agent_info": serialized_agent_info,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        _backend_logger.error(
            "Error creating agent",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Agent creation failed: {str(e)}")


async def handle_create_tool(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle tool creation request."""
    try:
        data = message.get("data", {})
        name = data.get("name")
        description = data.get("description")
        functionality_description = data.get("functionality_description")
        assign_to_agent = data.get("assign_to_agent")
        make_global = data.get("make_global", False)
        
        if not name or not description or not functionality_description:
            await send_error(connection_id, "Missing required tool parameters")
            return
        
        # Create tool
        tool_id = await seamless_integration.create_unlimited_tool(
            name=name,
            description=description,
            functionality_description=functionality_description,
            assign_to_agent=assign_to_agent,
            make_global=make_global
        )
        
        # Send creation result
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "tool_created",
                "tool_id": tool_id,
                "name": name,
                "description": description,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        _backend_logger.error(
            "Error creating tool",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Tool creation failed: {str(e)}")


async def handle_execute_workflow(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle workflow execution request."""
    try:
        data = message.get("data", {})
        workflow_id = data.get("workflow_id")
        task = data.get("task")
        workflow_type = data.get("workflow_type", "multi_agent")
        context = data.get("context", {})
        
        if not workflow_id or not task:
            await send_error(connection_id, "Missing workflow_id or task")
            return
        
        # Send execution started notification
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "workflow_execution_started",
                "workflow_id": workflow_id,
                "task": task,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
        # Execute workflow
        orchestrator = get_orchestrator_with_compatibility()
        if workflow_type == "hierarchical":
            result = await orchestrator.execute_hierarchical_workflow(
                task=task,
                context=context
            )
        else:
            result = await orchestrator.execute_workflow(
                workflow_id=workflow_id,
                inputs={"task": task, **context}
            )
        
        # Send execution result
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "workflow_execution_completed",
                "workflow_id": workflow_id,
                "task": task,
                "result": result,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        _backend_logger.error(
            "Error executing workflow",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Workflow execution failed: {str(e)}")


async def handle_execute_visual_workflow(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle visual workflow execution request with real-time updates."""
    try:
        data = message.get("data", {})
        workflow_id = data.get("workflow_id")
        nodes = data.get("nodes", [])
        connections = data.get("connections", [])
        inputs = data.get("inputs", {})
        context = data.get("context", {})

        if not workflow_id or not nodes:
            await send_error(connection_id, "Missing workflow_id or nodes")
            return

        execution_id = f"visual_exec_{int(asyncio.get_event_loop().time())}_{uuid.uuid4().hex[:8]}"

        # Send execution started notification
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "visual_workflow_execution_started",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "node_count": len(nodes),
                "connection_count": len(connections),
                "timestamp": asyncio.get_event_loop().time()
            }
        )

        # Execute visual workflow with real-time updates
        asyncio.create_task(
            _execute_visual_workflow_with_updates(
                connection_id, workflow_id, execution_id, nodes, connections, inputs, context
            )
        )

    except Exception as e:
        _backend_logger.error(
            "Error starting visual workflow execution",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Visual workflow execution failed: {str(e)}")


async def handle_get_agents(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle get agents request."""
    try:
        orchestrator = get_orchestrator_with_compatibility()
        agents = list(orchestrator.agents.values())
        
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "agents_list",
                "agents": agents,
                "count": len(agents),
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        _backend_logger.error(
            "Error getting agents",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Failed to get agents: {str(e)}")


async def handle_get_tools(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle get tools request."""
    try:
        orchestrator = get_orchestrator_with_compatibility()
        tools = list(orchestrator.enhanced_orchestrator.tool_registry.get_all_tools().keys()) if hasattr(orchestrator.enhanced_orchestrator, 'tool_registry') else []
        
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "tools_list",
                "tools": tools,
                "count": len(tools),
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        _backend_logger.error(
            "Error getting tools",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Failed to get tools: {str(e)}")


async def send_system_status(connection_id: str) -> None:
    """Send current system status to client."""
    try:
        orchestrator = get_orchestrator_with_compatibility()
        status = {
            "agents_count": len(orchestrator.agents),
            "tools_count": len(orchestrator.enhanced_orchestrator.tool_registry.get_all_tools()) if hasattr(orchestrator.enhanced_orchestrator, 'tool_registry') else 0,
            "system_health": "healthy",
            "capabilities": [
                "unlimited_agents",
                "dynamic_tools",
                "autonomous_intelligence",
                "multi_agent_coordination"
            ]
        }
        
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "system_status",
                "status": status,
                "timestamp": asyncio.get_event_loop().time()
            }
        )
        
    except Exception as e:
        _backend_logger.error(
            "Error sending system status",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )


async def send_pong(connection_id: str) -> None:
    """Send pong response to ping."""
    await websocket_manager.send_personal_message(
        connection_id,
        {
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time()
        }
    )


async def send_error(connection_id: str, error_message: str) -> None:
    """Send error message to client."""
    await websocket_manager.send_personal_message(
        connection_id,
        {
            "type": "error",
            "error": error_message,
            "timestamp": asyncio.get_event_loop().time()
        }
    )


# Collaboration WebSocket Handlers
collaboration_workspaces: Dict[str, Dict[str, Any]] = {}


async def handle_collaboration_connection(websocket: WebSocket, workspace_id: str) -> None:
    """
    Handle WebSocket connection for real-time collaboration.

    Args:
        websocket: WebSocket connection
        workspace_id: Workspace identifier for collaboration
    """
    connection_id = str(uuid.uuid4())
    user_id = f"user_{connection_id[:8]}"

    try:
        # Accept the connection
        await websocket.accept()

        # Initialize workspace if it doesn't exist
        if workspace_id not in collaboration_workspaces:
            collaboration_workspaces[workspace_id] = {
                "users": {},
                "document_state": {},
                "comments": {},
                "last_activity": asyncio.get_event_loop().time()
            }

        workspace = collaboration_workspaces[workspace_id]

        # Add user to workspace
        workspace["users"][user_id] = {
            "connection_id": connection_id,
            "websocket": websocket,
            "name": f"User-{connection_id[:4]}",
            "color": f"hsl({hash(user_id) % 360}, 70%, 50%)",
            "cursor": None,
            "selection": None,
            "joined_at": asyncio.get_event_loop().time()
        }

        _backend_logger.info(
            "Collaboration connection established",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={
                "workspace_id": workspace_id,
                "user_id": user_id,
                "total_users": len(workspace["users"])
            }
        )

        # Send initial workspace state
        await send_collaboration_state(websocket, workspace_id, user_id)

        # Notify other users about new user
        await broadcast_user_joined(workspace_id, user_id)

        # Handle incoming collaboration messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                await handle_collaboration_message(workspace_id, user_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                _backend_logger.error(
                    "Error handling collaboration message",
                    LogCategory.API_OPERATIONS,
                    "app.api.websocket.handlers",
                    data={"error": str(e)}
                )
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Message handling error: {str(e)}"
                }))

    except Exception as e:
        _backend_logger.error(
            "Collaboration connection error",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"error": str(e)}
        )
    finally:
        # Clean up user from workspace
        if workspace_id in collaboration_workspaces:
            workspace = collaboration_workspaces[workspace_id]
            if user_id in workspace["users"]:
                del workspace["users"][user_id]

                # Notify other users about user leaving
                await broadcast_user_left(workspace_id, user_id)

                # Clean up empty workspaces
                if not workspace["users"]:
                    del collaboration_workspaces[workspace_id]

        _backend_logger.info(
            "Collaboration connection closed",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"workspace_id": workspace_id, "user_id": user_id}
        )


async def handle_collaboration_message(workspace_id: str, user_id: str, message: Dict[str, Any]) -> None:
    """Handle collaboration message from user."""

    message_type = message.get("type")
    workspace = collaboration_workspaces.get(workspace_id)

    if not workspace or user_id not in workspace["users"]:
        return

    user = workspace["users"][user_id]

    if message_type == "cursor_update":
        # Update user cursor position
        user["cursor"] = message.get("cursor")
        await broadcast_cursor_update(workspace_id, user_id, message.get("cursor"))

    elif message_type == "selection_update":
        # Update user selection
        user["selection"] = message.get("selection")
        await broadcast_selection_update(workspace_id, user_id, message.get("selection"))

    elif message_type == "document_change":
        # Handle document changes (Yjs integration would go here)
        changes = message.get("changes", [])
        workspace["document_state"] = message.get("document_state", {})
        await broadcast_document_changes(workspace_id, user_id, changes)

    elif message_type == "comment_add":
        # Add comment
        comment_id = str(uuid.uuid4())
        comment = {
            "id": comment_id,
            "user_id": user_id,
            "user_name": user["name"],
            "content": message.get("content"),
            "position": message.get("position"),
            "timestamp": asyncio.get_event_loop().time()
        }
        workspace["comments"][comment_id] = comment
        await broadcast_comment_added(workspace_id, comment)

    elif message_type == "ping":
        # Respond to ping
        await user["websocket"].send_text(json.dumps({
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time()
        }))


async def send_collaboration_state(websocket: WebSocket, workspace_id: str, user_id: str) -> None:
    """Send initial collaboration state to user."""
    workspace = collaboration_workspaces.get(workspace_id)
    if not workspace:
        return

    # Prepare user list (excluding current user)
    users = {}
    for uid, user_data in workspace["users"].items():
        if uid != user_id:
            users[uid] = {
                "name": user_data["name"],
                "color": user_data["color"],
                "cursor": user_data["cursor"],
                "selection": user_data["selection"]
            }

    state = {
        "type": "collaboration_state",
        "workspace_id": workspace_id,
        "user_id": user_id,
        "users": users,
        "document_state": workspace["document_state"],
        "comments": workspace["comments"]
    }

    await websocket.send_text(json.dumps(state))


async def broadcast_user_joined(workspace_id: str, user_id: str) -> None:
    """Broadcast user joined event to all other users."""
    workspace = collaboration_workspaces.get(workspace_id)
    if not workspace:
        return

    user = workspace["users"][user_id]
    message = {
        "type": "user_joined",
        "user_id": user_id,
        "user": {
            "name": user["name"],
            "color": user["color"]
        }
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_user_left(workspace_id: str, user_id: str) -> None:
    """Broadcast user left event to all other users."""
    message = {
        "type": "user_left",
        "user_id": user_id
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_cursor_update(workspace_id: str, user_id: str, cursor: Dict[str, Any]) -> None:
    """Broadcast cursor update to all other users."""
    message = {
        "type": "cursor_update",
        "user_id": user_id,
        "cursor": cursor
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_selection_update(workspace_id: str, user_id: str, selection: Dict[str, Any]) -> None:
    """Broadcast selection update to all other users."""
    message = {
        "type": "selection_update",
        "user_id": user_id,
        "selection": selection
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_document_changes(workspace_id: str, user_id: str, changes: list) -> None:
    """Broadcast document changes to all other users."""
    message = {
        "type": "document_changes",
        "user_id": user_id,
        "changes": changes
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_comment_added(workspace_id: str, comment: Dict[str, Any]) -> None:
    """Broadcast new comment to all users."""
    message = {
        "type": "comment_added",
        "comment": comment
    }

    await broadcast_to_workspace(workspace_id, message)


async def broadcast_to_workspace(workspace_id: str, message: Dict[str, Any], exclude_user: str = None) -> None:
    """Broadcast message to all users in workspace."""
    workspace = collaboration_workspaces.get(workspace_id)
    if not workspace:
        return

    message_text = json.dumps(message)
    disconnected_users = []

    for user_id, user_data in workspace["users"].items():
        if exclude_user and user_id == exclude_user:
            continue

        try:
            await user_data["websocket"].send_text(message_text)
        except Exception as e:
            _backend_logger.warn(
                f"Failed to send message to user {user_id}: {str(e)}",
                LogCategory.API_OPERATIONS,
                "app.api.websocket.handlers"
            )
            disconnected_users.append(user_id)

    # Clean up disconnected users
    for user_id in disconnected_users:
        if user_id in workspace["users"]:
            del workspace["users"][user_id]


# Collaboration WebSocket Handlers
collaboration_workspaces: Dict[str, Dict[str, Any]] = {}


async def handle_collaboration_connection(websocket: WebSocket, workspace_id: str) -> None:
    """
    Handle WebSocket connection for real-time collaboration.

    Args:
        websocket: WebSocket connection
        workspace_id: Workspace identifier for collaboration
    """
    connection_id = str(uuid.uuid4())
    user_id = f"user_{connection_id[:8]}"

    try:
        # Accept the connection
        await websocket.accept()

        # Initialize workspace if it doesn't exist
        if workspace_id not in collaboration_workspaces:
            collaboration_workspaces[workspace_id] = {
                "users": {},
                "document_state": {},
                "comments": {},
                "last_activity": asyncio.get_event_loop().time()
            }

        workspace = collaboration_workspaces[workspace_id]

        # Add user to workspace
        workspace["users"][user_id] = {
            "connection_id": connection_id,
            "websocket": websocket,
            "name": f"User-{connection_id[:4]}",
            "color": f"hsl({hash(user_id) % 360}, 70%, 50%)",
            "cursor": None,
            "selection": None,
            "joined_at": asyncio.get_event_loop().time()
        }

        _backend_logger.info(
            "Collaboration connection established",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={
                "workspace_id": workspace_id,
                "user_id": user_id,
                "total_users": len(workspace["users"])
            }
        )

        # Send initial workspace state
        await send_collaboration_state(websocket, workspace_id, user_id)

        # Notify other users about new user
        await broadcast_user_joined(workspace_id, user_id)

        # Handle incoming collaboration messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                await handle_collaboration_message(workspace_id, user_id, message)

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                _backend_logger.error(
                    "Error handling collaboration message",
                    LogCategory.API_OPERATIONS,
                    "app.api.websocket.handlers",
                    data={"error": str(e)}
                )
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Message handling error: {str(e)}"
                }))

    except Exception as e:
        _backend_logger.error(
            "Collaboration connection error",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"error": str(e)}
        )
    finally:
        # Clean up user from workspace
        if workspace_id in collaboration_workspaces:
            workspace = collaboration_workspaces[workspace_id]
            if user_id in workspace["users"]:
                del workspace["users"][user_id]

                # Notify other users about user leaving
                await broadcast_user_left(workspace_id, user_id)

                # Clean up empty workspaces
                if not workspace["users"]:
                    del collaboration_workspaces[workspace_id]

        _backend_logger.info(
            "Collaboration connection closed",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"workspace_id": workspace_id, "user_id": user_id}
        )


async def handle_collaboration_message(workspace_id: str, user_id: str, message: Dict[str, Any]) -> None:
    """Handle collaboration message from user."""

    message_type = message.get("type")
    workspace = collaboration_workspaces.get(workspace_id)

    if not workspace or user_id not in workspace["users"]:
        return

    user = workspace["users"][user_id]

    if message_type == "cursor_update":
        # Update user cursor position
        user["cursor"] = message.get("cursor")
        await broadcast_cursor_update(workspace_id, user_id, message.get("cursor"))

    elif message_type == "selection_update":
        # Update user selection
        user["selection"] = message.get("selection")
        await broadcast_selection_update(workspace_id, user_id, message.get("selection"))

    elif message_type == "document_change":
        # Handle document changes (Yjs integration would go here)
        changes = message.get("changes", [])
        workspace["document_state"] = message.get("document_state", {})
        await broadcast_document_changes(workspace_id, user_id, changes)

    elif message_type == "comment_add":
        # Add comment
        comment_id = str(uuid.uuid4())
        comment = {
            "id": comment_id,
            "user_id": user_id,
            "user_name": user["name"],
            "content": message.get("content"),
            "position": message.get("position"),
            "timestamp": asyncio.get_event_loop().time()
        }
        workspace["comments"][comment_id] = comment
        await broadcast_comment_added(workspace_id, comment)

    elif message_type == "ping":
        # Respond to ping
        await user["websocket"].send_text(json.dumps({
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time()
        }))


async def send_collaboration_state(websocket: WebSocket, workspace_id: str, user_id: str) -> None:
    """Send initial collaboration state to user."""
    workspace = collaboration_workspaces.get(workspace_id)
    if not workspace:
        return

    # Prepare user list (excluding current user)
    users = {}
    for uid, user_data in workspace["users"].items():
        if uid != user_id:
            users[uid] = {
                "name": user_data["name"],
                "color": user_data["color"],
                "cursor": user_data["cursor"],
                "selection": user_data["selection"]
            }

    state = {
        "type": "collaboration_state",
        "workspace_id": workspace_id,
        "user_id": user_id,
        "users": users,
        "document_state": workspace["document_state"],
        "comments": workspace["comments"]
    }

    await websocket.send_text(json.dumps(state))


async def broadcast_user_joined(workspace_id: str, user_id: str) -> None:
    """Broadcast user joined event to all other users."""
    workspace = collaboration_workspaces.get(workspace_id)
    if not workspace:
        return

    user = workspace["users"][user_id]
    message = {
        "type": "user_joined",
        "user_id": user_id,
        "user": {
            "name": user["name"],
            "color": user["color"]
        }
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_user_left(workspace_id: str, user_id: str) -> None:
    """Broadcast user left event to all other users."""
    message = {
        "type": "user_left",
        "user_id": user_id
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_cursor_update(workspace_id: str, user_id: str, cursor: Dict[str, Any]) -> None:
    """Broadcast cursor update to all other users."""
    message = {
        "type": "cursor_update",
        "user_id": user_id,
        "cursor": cursor
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_selection_update(workspace_id: str, user_id: str, selection: Dict[str, Any]) -> None:
    """Broadcast selection update to all other users."""
    message = {
        "type": "selection_update",
        "user_id": user_id,
        "selection": selection
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_document_changes(workspace_id: str, user_id: str, changes: list) -> None:
    """Broadcast document changes to all other users."""
    message = {
        "type": "document_changes",
        "user_id": user_id,
        "changes": changes
    }

    await broadcast_to_workspace(workspace_id, message, exclude_user=user_id)


async def broadcast_comment_added(workspace_id: str, comment: Dict[str, Any]) -> None:
    """Broadcast new comment to all users."""
    message = {
        "type": "comment_added",
        "comment": comment
    }

    await broadcast_to_workspace(workspace_id, message)


async def broadcast_to_workspace(workspace_id: str, message: Dict[str, Any], exclude_user: str = None) -> None:
    """Broadcast message to all users in workspace."""
    workspace = collaboration_workspaces.get(workspace_id)
    if not workspace:
        return

    message_text = json.dumps(message)
    disconnected_users = []

    for user_id, user_data in workspace["users"].items():
        if exclude_user and user_id == exclude_user:
            continue

        try:
            await user_data["websocket"].send_text(message_text)
        except Exception as e:
            _backend_logger.warn(
                f"Failed to send message to user {user_id}: {str(e)}",
                LogCategory.API_OPERATIONS,
                "app.api.websocket.handlers"
            )
            disconnected_users.append(user_id)

    # Clean up disconnected users
    for user_id in disconnected_users:
        if user_id in workspace["users"]:
            del workspace["users"][user_id]


# ============================================================================
# REVOLUTIONARY COMPONENT WORKFLOW EXECUTION HANDLERS
# ============================================================================

async def handle_component_workflow_execution(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle component workflow execution request with real-time updates."""
    try:
        data = message.get("data", {})
        workflow_id = data.get("workflow_id")
        components = data.get("components", [])
        execution_mode = data.get("execution_mode", "sequential")
        context = data.get("context", {})

        if not workflow_id or not components:
            await send_error(connection_id, "Missing workflow_id or components")
            return

        # Send execution started notification
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "component_workflow_started",
                "workflow_id": workflow_id,
                "total_components": len(components),
                "execution_mode": execution_mode,
                "timestamp": asyncio.get_event_loop().time()
            }
        )

        # Get the unified system orchestrator
        from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
        orchestrator = get_enhanced_system_orchestrator()

        if not orchestrator.is_initialized:
            await orchestrator.initialize()

        # Start component workflow execution with real-time updates
        asyncio.create_task(
            _execute_component_workflow_with_updates(
                connection_id, workflow_id, components, execution_mode, context, orchestrator
            )
        )

    except Exception as e:
        _backend_logger.error(
            "Error handling component workflow execution",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Component workflow execution failed: {str(e)}")


async def _execute_component_workflow_with_updates(
    connection_id: str,
    workflow_id: str,
    components: List[Dict[str, Any]],
    execution_mode: str,
    context: Dict[str, Any],
    orchestrator
) -> None:
    """Execute component workflow with real-time progress updates."""
    try:
        # Execute component workflow
        result = await orchestrator.execute_component_workflow(
            workflow_id=workflow_id,
            components=components,
            execution_mode=execution_mode,
            context=context
        )

        # Monitor workflow progress and send updates
        await _monitor_workflow_progress(connection_id, workflow_id, orchestrator)

    except Exception as e:
        _backend_logger.error(
            "Error in component workflow execution",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"error": str(e)}
        )
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "component_workflow_error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
        )


async def _monitor_workflow_progress(connection_id: str, workflow_id: str, orchestrator) -> None:
    """Monitor workflow progress and send real-time updates."""
    try:
        from app.agent_builder_platform import get_step_state_tracker
        step_tracker = get_step_state_tracker()

        # Monitor for up to 5 minutes
        max_monitoring_time = 300  # 5 minutes
        start_time = asyncio.get_event_loop().time()
        last_update_time = start_time

        while (asyncio.get_event_loop().time() - start_time) < max_monitoring_time:
            current_time = asyncio.get_event_loop().time()

            # Get workflow status
            workflow_status = None
            if orchestrator.component_workflow_executor:
                workflow_status = orchestrator.component_workflow_executor.get_workflow_status(workflow_id)

            if workflow_status:
                # Send progress update every 2 seconds or when status changes
                if (current_time - last_update_time) >= 2.0:
                    step_ids = step_tracker.get_workflow_steps(workflow_id)

                    progress_info = {
                        "workflow_id": workflow_id,
                        "status": workflow_status["status"],
                        "current_step": workflow_status.get("current_step", 0),
                        "total_steps": workflow_status.get("total_steps", 0),
                        "completed_steps": 0,
                        "failed_steps": 0,
                        "running_steps": 0,
                        "step_updates": []
                    }

                    # Get step details
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

                            progress_info["step_updates"].append({
                                "step_id": step_id,
                                "status": status,
                                "component_type": step_state.get("component_type"),
                                "execution_time": step_state.get("execution_time")
                            })

                    await websocket_manager.send_personal_message(
                        connection_id,
                        {
                            "type": "component_workflow_progress",
                            **progress_info,
                            "timestamp": current_time
                        }
                    )

                    last_update_time = current_time

                # Check if workflow is completed
                if workflow_status["status"] in ["completed", "failed"]:
                    await websocket_manager.send_personal_message(
                        connection_id,
                        {
                            "type": "component_workflow_completed",
                            "workflow_id": workflow_id,
                            "status": workflow_status["status"],
                            "results": workflow_status.get("results", {}),
                            "execution_time": (
                                workflow_status.get("end_time", datetime.utcnow()) -
                                workflow_status.get("start_time", datetime.utcnow())
                            ).total_seconds() if workflow_status.get("end_time") else None,
                            "timestamp": current_time
                        }
                    )
                    break

            # Wait before next check
            await asyncio.sleep(1.0)

    except Exception as e:
        _backend_logger.error(
            "Error monitoring workflow progress",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"error": str(e)}
        )


async def handle_component_agent_status(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle component agent status request."""
    try:
        data = message.get("data", {})
        agent_id = data.get("agent_id")

        if not agent_id:
            await send_error(connection_id, "Missing agent_id")
            return

        from app.agent_builder_platform import get_component_agent_manager
        component_manager = await get_component_agent_manager()

        agent_status = component_manager.get_component_agent(agent_id)

        if agent_status:
            await websocket_manager.send_personal_message(
                connection_id,
                {
                    "type": "component_agent_status",
                    "agent_id": agent_id,
                    "status": agent_status,
                    "timestamp": asyncio.get_event_loop().time()
                }
            )
        else:
            await send_error(connection_id, f"Component agent not found: {agent_id}")

    except Exception as e:
        _backend_logger.error(
            "Error getting component agent status",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Failed to get component agent status: {str(e)}")


async def handle_workflow_step_updates(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle workflow step updates subscription."""
    try:
        data = message.get("data", {})
        workflow_id = data.get("workflow_id")

        if not workflow_id:
            await send_error(connection_id, "Missing workflow_id")
            return

        from app.agent_builder_platform import get_step_state_tracker
        step_tracker = get_step_state_tracker()

        # Get current step states
        step_ids = step_tracker.get_workflow_steps(workflow_id)

        step_updates = []
        for step_id in step_ids:
            step_state = step_tracker.get_step_state(step_id)
            if step_state:
                step_updates.append({
                    "step_id": step_id,
                    "status": step_state["status"],
                    "component_type": step_state.get("component_type"),
                    "start_time": step_state.get("start_time"),
                    "end_time": step_state.get("end_time"),
                    "execution_time": step_state.get("execution_time"),
                    "events": step_state.get("events", [])
                })

        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "workflow_step_updates",
                "workflow_id": workflow_id,
                "total_steps": len(step_ids),
                "step_updates": step_updates,
                "timestamp": asyncio.get_event_loop().time()
            }
        )

    except Exception as e:
        _backend_logger.error(
            "Error handling workflow step updates",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Failed to get workflow step updates: {str(e)}")


async def handle_component_palette_request(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle component palette request for visual builder."""
    try:
        from app.agents.templates import AgentTemplateLibrary
        template_library = AgentTemplateLibrary()

        # Get component palette
        component_palette = template_library.get_component_palette()

        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "component_palette",
                "palette": component_palette,
                "total_components": sum(len(components) for components in component_palette.values()),
                "timestamp": asyncio.get_event_loop().time()
            }
        )

    except Exception as e:
        _backend_logger.error(
            "Error getting component palette",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await send_error(connection_id, f"Failed to get component palette: {str(e)}")


async def _execute_visual_workflow_with_updates(
    connection_id: str,
    workflow_id: str,
    execution_id: str,
    nodes: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    inputs: Dict[str, Any],
    context: Dict[str, Any]
) -> None:
    """Execute visual workflow with real-time progress updates."""
    try:
        from app.core.node_registry import get_node_registry

        # Initialize node registry
        node_registry = get_node_registry()

        # Build execution graph
        execution_graph = {}
        for node in nodes:
            execution_graph[node["id"]] = []

        for connection in connections:
            if connection["source"] in execution_graph:
                execution_graph[connection["source"]].append(connection["target"])

        # Get topological order
        def get_topological_order(graph):
            in_degree = {node: 0 for node in graph}
            for node in graph:
                for neighbor in graph[node]:
                    if neighbor in in_degree:
                        in_degree[neighbor] += 1

            queue = [node for node, degree in in_degree.items() if degree == 0]
            result = []

            while queue:
                node = queue.pop(0)
                result.append(node)

                for neighbor in graph[node]:
                    if neighbor in in_degree:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            queue.append(neighbor)

            return result

        node_execution_order = get_topological_order(execution_graph)
        execution_results = {}

        # Send progress update
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "visual_workflow_progress",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "executing",
                "total_nodes": len(nodes),
                "completed_nodes": 0,
                "current_node": node_execution_order[0] if node_execution_order else None,
                "execution_order": node_execution_order,
                "timestamp": asyncio.get_event_loop().time()
            }
        )

        # Execute nodes in order
        for i, node_id in enumerate(node_execution_order):
            node = next((n for n in nodes if n["id"] == node_id), None)
            if not node:
                continue

            # Send node execution start
            await websocket_manager.send_personal_message(
                connection_id,
                {
                    "type": "visual_node_execution_start",
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "node_id": node_id,
                    "node_type": node["type"],
                    "timestamp": asyncio.get_event_loop().time()
                }
            )

            # Get node inputs from connected nodes
            node_inputs = {}
            for connection in connections:
                if connection["target"] == node_id:
                    source_result = execution_results.get(connection["source"])
                    if source_result and source_result.get("success"):
                        input_key = connection.get("targetHandle", "data")
                        output_key = connection.get("sourceHandle", "data")

                        source_data = source_result.get("data", {})
                        if isinstance(source_data, dict) and output_key in source_data:
                            node_inputs[input_key] = source_data[output_key]
                        else:
                            node_inputs[input_key] = source_data

            # Execute node
            try:
                node_config = node.get("data", {}).get("configuration", {})
                if node_inputs:
                    node_config["inputs"] = node_inputs

                handler = node_registry.get_execution_handler(node["type"])
                if handler:
                    start_time = asyncio.get_event_loop().time()
                    result = await handler(node_config, {
                        "execution_id": execution_id,
                        "workflow_id": workflow_id,
                        "node_id": node_id
                    })
                    execution_time = asyncio.get_event_loop().time() - start_time

                    if isinstance(result, dict):
                        result["execution_time"] = execution_time
                    else:
                        result = {
                            "success": True,
                            "data": result,
                            "execution_time": execution_time
                        }
                else:
                    result = {
                        "success": False,
                        "error": f"No execution handler found for node type '{node['type']}'",
                        "execution_time": 0
                    }

                execution_results[node_id] = result

                # Send node execution complete
                await websocket_manager.send_personal_message(
                    connection_id,
                    {
                        "type": "visual_node_execution_complete",
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "node_id": node_id,
                        "node_type": node["type"],
                        "success": result.get("success", False),
                        "result": result,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                )

                # Send progress update
                await websocket_manager.send_personal_message(
                    connection_id,
                    {
                        "type": "visual_workflow_progress",
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "status": "executing",
                        "total_nodes": len(nodes),
                        "completed_nodes": i + 1,
                        "current_node": node_execution_order[i + 1] if i + 1 < len(node_execution_order) else None,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                )

            except Exception as node_error:
                error_result = {
                    "success": False,
                    "error": str(node_error),
                    "execution_time": 0
                }
                execution_results[node_id] = error_result

                # Send node execution error
                await websocket_manager.send_personal_message(
                    connection_id,
                    {
                        "type": "visual_node_execution_error",
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "node_id": node_id,
                        "node_type": node["type"],
                        "error": str(node_error),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                )

        # Send final completion
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "visual_workflow_execution_completed",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "completed",
                "results": execution_results,
                "total_nodes": len(nodes),
                "completed_nodes": len(execution_results),
                "timestamp": asyncio.get_event_loop().time()
            }
        )

    except Exception as e:
        _backend_logger.error(
            "Error executing visual workflow",
            LogCategory.API_OPERATIONS,
            "app.api.websocket.handlers",
            data={"connection_id": connection_id, "error": str(e)}
        )
        await websocket_manager.send_personal_message(
            connection_id,
            {
                "type": "visual_workflow_execution_error",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
        )
