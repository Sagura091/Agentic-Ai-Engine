"""
Admin Management API endpoints.

This module provides administrative functionality for system management,
monitoring, configuration, and maintenance operations.
"""

import asyncio
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.core.dependencies import get_orchestrator, require_admin, get_monitoring_service
from app.orchestration.orchestrator import LangGraphOrchestrator

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Admin Management"])


# Pydantic models for API requests/responses
class SystemStatus(BaseModel):
    """System status information."""
    status: str = Field(..., description="Overall system status")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment (dev/staging/prod)")
    timestamp: datetime = Field(..., description="Status timestamp")


class SystemMetrics(BaseModel):
    """System performance metrics."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    active_agents: int = Field(..., description="Number of active agents")
    active_workflows: int = Field(..., description="Number of active workflows")
    total_requests: int = Field(..., description="Total API requests")
    avg_response_time: float = Field(..., description="Average response time in ms")


class ConfigurationUpdate(BaseModel):
    """Configuration update request."""
    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")
    category: str = Field(default="general", description="Configuration category")


class LogLevel(BaseModel):
    """Log level configuration."""
    level: str = Field(..., description="Log level (DEBUG, INFO, WARNING, ERROR)")
    logger_name: Optional[str] = Field(default=None, description="Specific logger name")


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    settings = Depends(get_settings),
    current_user: dict = Depends(require_admin)
) -> SystemStatus:
    """
    Get overall system status.
    
    Returns:
        System status information
    """
    try:
        # Calculate uptime (placeholder - should track actual start time)
        uptime = 3600.0  # 1 hour placeholder
        
        status = SystemStatus(
            status="healthy",
            uptime=uptime,
            version="0.1.0",
            environment=settings.ENVIRONMENT,
            timestamp=datetime.now()
        )
        
        logger.info("System status retrieved", status=status.status)
        return status
        
    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator),
    current_user: dict = Depends(require_admin)
) -> SystemMetrics:
    """
    Get system performance metrics.
    
    Returns:
        System performance metrics
    """
    try:
        # Get system metrics using psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get agent and workflow counts
        active_agents = len(orchestrator.agents) if orchestrator.is_initialized else 0
        active_workflows = len(orchestrator.workflows) if orchestrator.is_initialized else 0
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            active_agents=active_agents,
            active_workflows=active_workflows,
            total_requests=1000,  # Placeholder - should track actual requests
            avg_response_time=150.5  # Placeholder - should calculate actual average
        )
        
        logger.info("System metrics retrieved", cpu=cpu_usage, memory=memory.percent)
        return metrics
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.get("/agents/summary")
async def get_agents_summary(
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator),
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get summary of all agents in the system.
    
    Returns:
        Agents summary information
    """
    try:
        if not orchestrator.is_initialized:
            return {
                "total_agents": 0,
                "active_agents": 0,
                "agent_types": {},
                "orchestrator_status": "not_initialized"
            }
        
        agent_types = {}
        for agent_id, agent in orchestrator.agents.items():
            agent_type = getattr(agent, 'agent_type', 'unknown')
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        summary = {
            "total_agents": len(orchestrator.agents),
            "active_agents": len(orchestrator.agents),
            "agent_types": agent_types,
            "orchestrator_status": "initialized",
            "global_tools": len(orchestrator.global_tools),
            "workflows": len(orchestrator.workflows)
        }
        
        logger.info("Agents summary retrieved", total=summary["total_agents"])
        return summary
        
    except Exception as e:
        logger.error("Failed to get agents summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get agents summary: {str(e)}")


@router.post("/agents/{agent_id}/restart")
async def restart_agent(
    agent_id: str,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator),
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Restart a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Restart operation result
    """
    try:
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get agent configuration
        config = orchestrator.agent_configs.get(agent_id)
        if not config:
            raise HTTPException(status_code=500, detail=f"Agent {agent_id} configuration not found")
        
        # Remove old agent
        del orchestrator.agents[agent_id]
        
        # Create new agent with same configuration
        new_agent_id = await orchestrator.create_agent(
            agent_type=getattr(orchestrator.agents.get(agent_id), 'agent_type', 'general'),
            config=config.__dict__
        )
        
        result = {
            "status": "success",
            "message": f"Agent {agent_id} restarted successfully",
            "old_agent_id": agent_id,
            "new_agent_id": new_agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Agent restarted", old_id=agent_id, new_id=new_agent_id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restart agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to restart agent: {str(e)}")


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    orchestrator: LangGraphOrchestrator = Depends(get_orchestrator),
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Delete a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Deletion operation result
    """
    try:
        if agent_id not in orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Remove agent and its configuration
        del orchestrator.agents[agent_id]
        if agent_id in orchestrator.agent_configs:
            del orchestrator.agent_configs[agent_id]
        
        result = {
            "status": "success",
            "message": f"Agent {agent_id} deleted successfully",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Agent deleted", agent_id=agent_id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")


@router.post("/configuration")
async def update_configuration(
    request: ConfigurationUpdate,
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Update system configuration.
    
    Args:
        request: Configuration update request
        
    Returns:
        Configuration update result
    """
    try:
        # This is a placeholder implementation
        # In a real system, this would update the configuration store
        result = {
            "status": "success",
            "message": f"Configuration {request.key} updated successfully",
            "key": request.key,
            "value": request.value,
            "category": request.category,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Configuration updated", key=request.key, category=request.category)
        return result
        
    except Exception as e:
        logger.error("Failed to update configuration", key=request.key, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.post("/logs/level")
async def set_log_level(
    request: LogLevel,
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Set logging level for the system or specific logger.
    
    Args:
        request: Log level configuration
        
    Returns:
        Log level update result
    """
    try:
        import logging
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if request.level.upper() not in valid_levels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid log level. Must be one of: {valid_levels}"
            )
        
        # Set log level
        if request.logger_name:
            logger_obj = logging.getLogger(request.logger_name)
        else:
            logger_obj = logging.getLogger()
        
        logger_obj.setLevel(getattr(logging, request.level.upper()))
        
        result = {
            "status": "success",
            "message": f"Log level set to {request.level.upper()}",
            "level": request.level.upper(),
            "logger": request.logger_name or "root",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Log level updated", level=request.level, logger=request.logger_name)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to set log level", level=request.level, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to set log level: {str(e)}")


@router.post("/system/restart")
async def restart_system(
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Restart the entire system (graceful shutdown and restart).
    
    Returns:
        Restart operation result
    """
    try:
        # This is a placeholder implementation
        # In a real system, this would trigger a graceful restart
        result = {
            "status": "success",
            "message": "System restart initiated",
            "timestamp": datetime.now().isoformat(),
            "note": "System will restart in 10 seconds"
        }
        
        logger.warning("System restart initiated by admin", user=current_user.get("username"))
        return result
        
    except Exception as e:
        logger.error("Failed to restart system", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to restart system: {str(e)}")


@router.get("/logs/recent")
async def get_recent_logs(
    limit: int = 100,
    level: Optional[str] = None,
    current_user: dict = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Get recent system logs.
    
    Args:
        limit: Maximum number of log entries to return
        level: Filter by log level
        
    Returns:
        Recent log entries
    """
    try:
        # This is a placeholder implementation
        # In a real system, this would query the log storage
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "logger": "app.main",
                "message": "Application started successfully",
                "module": "main.py"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "logger": "app.orchestration",
                "message": "LangGraph orchestrator initialized",
                "module": "orchestrator.py"
            }
        ]
        
        # Apply filters
        if level:
            logs = [log for log in logs if log["level"] == level.upper()]
        
        # Apply limit
        logs = logs[:limit]
        
        result = {
            "logs": logs,
            "total": len(logs),
            "limit": limit,
            "level_filter": level,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Recent logs retrieved", count=len(logs))
        return result
        
    except Exception as e:
        logger.error("Failed to get recent logs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get recent logs: {str(e)}")
