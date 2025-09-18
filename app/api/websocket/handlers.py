"""
WebSocket handlers for real-time agent communication and collaboration.

This module provides WebSocket handlers for real-time communication
between the frontend and the agentic AI backend, including collaborative editing.
"""

import asyncio
import json
import uuid
from typing import Dict, Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from app.api.websocket.manager import websocket_manager
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator
from app.core.seamless_integration import seamless_integration

logger = structlog.get_logger(__name__)


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
        
        logger.info("WebSocket connection established", connection_id=connection_id)
        
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
                logger.info("WebSocket client disconnected", connection_id=connection_id)
                break
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON received", connection_id=connection_id, error=str(e))
                await send_error(connection_id, "Invalid JSON format")
            except Exception as e:
                logger.error("Error handling WebSocket message", connection_id=connection_id, error=str(e))
                await send_error(connection_id, f"Error processing message: {str(e)}")
    
    except Exception as e:
        logger.error("WebSocket connection error", connection_id=connection_id, error=str(e))
    
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
        result = await enhanced_orchestrator.execute_agent_task(
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
        logger.error("Error executing agent", connection_id=connection_id, error=str(e))
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
        logger.error("Error creating agent", connection_id=connection_id, error=str(e))
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
        logger.error("Error creating tool", connection_id=connection_id, error=str(e))
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
        if workflow_type == "hierarchical":
            result = await enhanced_orchestrator.execute_hierarchical_workflow(
                task=task,
                context=context
            )
        else:
            result = await enhanced_orchestrator.execute_workflow(
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
        logger.error("Error executing workflow", connection_id=connection_id, error=str(e))
        await send_error(connection_id, f"Workflow execution failed: {str(e)}")


async def handle_get_agents(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle get agents request."""
    try:
        agents = await enhanced_orchestrator.list_agents()
        
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
        logger.error("Error getting agents", connection_id=connection_id, error=str(e))
        await send_error(connection_id, f"Failed to get agents: {str(e)}")


async def handle_get_tools(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle get tools request."""
    try:
        tools = list(enhanced_orchestrator.tool_registry.get_all_tools().keys())
        
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
        logger.error("Error getting tools", connection_id=connection_id, error=str(e))
        await send_error(connection_id, f"Failed to get tools: {str(e)}")


async def send_system_status(connection_id: str) -> None:
    """Send current system status to client."""
    try:
        status = {
            "agents_count": len(enhanced_orchestrator.agents),
            "tools_count": len(enhanced_orchestrator.tool_registry.get_all_tools()),
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
        logger.error("Error sending system status", connection_id=connection_id, error=str(e))


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

        logger.info(
            "Collaboration connection established",
            workspace_id=workspace_id,
            user_id=user_id,
            total_users=len(workspace["users"])
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
                logger.error("Error handling collaboration message", error=str(e))
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Message handling error: {str(e)}"
                }))

    except Exception as e:
        logger.error("Collaboration connection error", error=str(e))
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

        logger.info("Collaboration connection closed", workspace_id=workspace_id, user_id=user_id)


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
            logger.warning(f"Failed to send message to user {user_id}: {str(e)}")
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

        logger.info(
            "Collaboration connection established",
            workspace_id=workspace_id,
            user_id=user_id,
            total_users=len(workspace["users"])
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
                logger.error("Error handling collaboration message", error=str(e))
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Message handling error: {str(e)}"
                }))

    except Exception as e:
        logger.error("Collaboration connection error", error=str(e))
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

        logger.info("Collaboration connection closed", workspace_id=workspace_id, user_id=user_id)


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
            logger.warning(f"Failed to send message to user {user_id}: {str(e)}")
            disconnected_users.append(user_id)

    # Clean up disconnected users
    for user_id in disconnected_users:
        if user_id in workspace["users"]:
            del workspace["users"][user_id]
