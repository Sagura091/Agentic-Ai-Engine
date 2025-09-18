"""
WebSocket handlers for real-time agent communication.

This module provides WebSocket handlers for real-time communication
between the frontend and the agentic AI backend.
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
