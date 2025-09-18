"""
Monitoring API endpoints.

This module provides monitoring and metrics endpoints for the Agentic AI system.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.services.monitoring_service import monitoring_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


class MonitoringResponse(BaseModel):
    """Base monitoring response."""
    timestamp: datetime = Field(default_factory=datetime.now)
    timeframe: str
    data: Dict[str, Any]


class AgentActivityResponse(MonitoringResponse):
    """Agent activity monitoring response."""
    active_agents: int
    total_requests: int
    average_response_time: float
    success_rate: float


class WorkflowActivityResponse(MonitoringResponse):
    """Workflow activity monitoring response."""
    active_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_execution_time: float


class SystemMetricsResponse(MonitoringResponse):
    """System metrics monitoring response."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]


@router.get("/agents", response_model=AgentActivityResponse)
async def get_agent_activity(
    timeframe: str = Query(default="24h", description="Time frame for metrics (1h, 24h, 7d, 30d)"),
    agent_type: Optional[str] = Query(default=None, description="Filter by agent type")
) -> AgentActivityResponse:
    """
    Get agent activity metrics.
    
    Args:
        timeframe: Time frame for metrics
        agent_type: Optional agent type filter
        
    Returns:
        Agent activity metrics
    """
    try:
        # Get orchestrator for agent metrics
        from app.orchestration.orchestrator import orchestrator
        
        # Calculate metrics based on current state
        active_agents = len(orchestrator.agents) if orchestrator.is_initialized else 0
        
        # Mock data for now - in production this would come from metrics storage
        total_requests = active_agents * 10  # Simulate requests
        average_response_time = 1.2  # seconds
        success_rate = 0.95  # 95%
        
        logger.info(
            "Agent activity requested",
            timeframe=timeframe,
            agent_type=agent_type,
            active_agents=active_agents
        )
        
        return AgentActivityResponse(
            timeframe=timeframe,
            data={
                "agents": [
                    {
                        "id": agent_id,
                        "name": agent.config.name if hasattr(agent, 'config') else f"Agent-{agent_id}",
                        "type": agent_type or "general",
                        "status": "active",
                        "last_activity": datetime.now().isoformat()
                    }
                    for agent_id, agent in orchestrator.agents.items()
                ] if orchestrator.is_initialized else []
            },
            active_agents=active_agents,
            total_requests=total_requests,
            average_response_time=average_response_time,
            success_rate=success_rate
        )
        
    except Exception as e:
        logger.error("Failed to get agent activity", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agent activity")


@router.get("/workflows", response_model=WorkflowActivityResponse)
async def get_workflow_activity(
    timeframe: str = Query(default="24h", description="Time frame for metrics"),
    workflow_type: Optional[str] = Query(default=None, description="Filter by workflow type")
) -> WorkflowActivityResponse:
    """
    Get workflow activity metrics.
    
    Args:
        timeframe: Time frame for metrics
        workflow_type: Optional workflow type filter
        
    Returns:
        Workflow activity metrics
    """
    try:
        # Get orchestrator for workflow metrics
        from app.orchestration.orchestrator import orchestrator
        
        # Calculate metrics based on current state
        active_workflows = len(orchestrator.workflows) if orchestrator.is_initialized else 0
        
        # Mock data for now
        completed_workflows = active_workflows * 5
        failed_workflows = int(completed_workflows * 0.05)  # 5% failure rate
        average_execution_time = 30.5  # seconds
        
        logger.info(
            "Workflow activity requested",
            timeframe=timeframe,
            workflow_type=workflow_type,
            active_workflows=active_workflows
        )
        
        return WorkflowActivityResponse(
            timeframe=timeframe,
            data={
                "workflows": [
                    {
                        "id": workflow_id,
                        "name": f"Workflow-{workflow_id}",
                        "type": workflow_type or "general",
                        "status": "active",
                        "started_at": datetime.now().isoformat()
                    }
                    for workflow_id in orchestrator.workflows.keys()
                ] if orchestrator.is_initialized else []
            },
            active_workflows=active_workflows,
            completed_workflows=completed_workflows,
            failed_workflows=failed_workflows,
            average_execution_time=average_execution_time
        )
        
    except Exception as e:
        logger.error("Failed to get workflow activity", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve workflow activity")


@router.get("/system", response_model=SystemMetricsResponse)
async def get_system_metrics(
    timeframe: str = Query(default="24h", description="Time frame for metrics"),
    metric_type: Optional[str] = Query(default=None, description="Filter by metric type")
) -> SystemMetricsResponse:
    """
    Get system performance metrics.
    
    Args:
        timeframe: Time frame for metrics
        metric_type: Optional metric type filter
        
    Returns:
        System performance metrics
    """
    try:
        import psutil
        
        # Get real system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        logger.info(
            "System metrics requested",
            timeframe=timeframe,
            metric_type=metric_type,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent
        )
        
        return SystemMetricsResponse(
            timeframe=timeframe,
            data={
                "cpu": {
                    "usage_percent": cpu_usage,
                    "cores": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            },
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            network_io={
                "bytes_sent": float(network.bytes_sent),
                "bytes_recv": float(network.bytes_recv)
            }
        )
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.get("/errors")
async def get_error_logs(
    limit: int = Query(default=100, description="Maximum number of errors to return"),
    severity: Optional[str] = Query(default=None, description="Filter by severity level"),
    component: Optional[str] = Query(default=None, description="Filter by component")
) -> Dict[str, Any]:
    """
    Get recent error logs.
    
    Args:
        limit: Maximum number of errors to return
        severity: Optional severity filter
        component: Optional component filter
        
    Returns:
        Recent error logs
    """
    try:
        # Mock error logs for now - in production this would come from log aggregation
        errors = [
            {
                "id": f"error-{i}",
                "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                "severity": "warning" if i % 3 == 0 else "error",
                "component": "orchestrator" if i % 2 == 0 else "agent",
                "message": f"Sample error message {i}",
                "details": f"Error details for incident {i}"
            }
            for i in range(min(limit, 10))  # Limit to 10 mock errors
        ]
        
        # Apply filters
        if severity:
            errors = [e for e in errors if e["severity"] == severity]
        if component:
            errors = [e for e in errors if e["component"] == component]
        
        logger.info(
            "Error logs requested",
            limit=limit,
            severity=severity,
            component=component,
            errors_count=len(errors)
        )
        
        return {
            "errors": errors,
            "total_count": len(errors),
            "filters": {
                "severity": severity,
                "component": component,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error("Failed to get error logs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve error logs")


@router.get("/health")
async def get_monitoring_health() -> Dict[str, Any]:
    """
    Get monitoring system health status.
    
    Returns:
        Monitoring system health information
    """
    try:
        health_checks = await monitoring_service.get_health_status()
        
        return {
            "status": "healthy" if all(check.get("healthy", False) for check in health_checks.values()) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": health_checks,
            "monitoring_service": {
                "initialized": monitoring_service.is_initialized,
                "metrics_collection": monitoring_service.metrics_collection_task is not None
            }
        }
        
    except Exception as e:
        logger.error("Failed to get monitoring health", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring health")
