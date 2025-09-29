"""
Monitoring API endpoints.

This module provides monitoring and metrics endpoints for the Agentic AI system.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import structlog
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.config.settings import get_settings
# from app.services.monitoring_service import monitoring_service

# Import Agent Builder Platform components for monitoring
from app.agents.registry import get_agent_registry
from app.llm.manager import get_enhanced_llm_manager
from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator

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
        from app.core.unified_system_orchestrator import get_orchestrator_with_compatibility
        enhanced_orchestrator = get_orchestrator_with_compatibility()

        # Calculate metrics based on current state
        active_workflows = len(enhanced_orchestrator.workflows) if enhanced_orchestrator.status.is_initialized else 0
        
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
    response: Response,
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

        # Get real system metrics - use non-blocking CPU measurement
        cpu_usage = psutil.cpu_percent(interval=None)  # Non-blocking, uses cached value
        memory = psutil.virtual_memory()

        # Get disk usage - handle Windows vs Unix paths
        import os
        if os.name == 'nt':  # Windows
            # Use shutil.disk_usage instead of psutil on Windows
            import shutil
            total, used, free = shutil.disk_usage('.')
            # Create a mock disk object with the same interface as psutil
            class DiskUsage:
                def __init__(self, total, used, free):
                    self.total = total
                    self.used = used
                    self.free = free
                    self.percent = (used / total) * 100 if total > 0 else 0
            disk = DiskUsage(total, used, free)
        else:  # Unix/Linux
            disk = psutil.disk_usage('/')

        network = psutil.net_io_counters()
        
        logger.info(
            "System metrics requested",
            timeframe=timeframe,
            metric_type=metric_type,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent
        )

        # Add caching headers for system metrics (cache for 10 seconds)
        response.headers["Cache-Control"] = "public, max-age=10"
        response.headers["ETag"] = f'"{hash((cpu_usage, memory.percent, disk.percent))}"'

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
        # health_checks = await monitoring_service.get_health_status()
        health_checks = {"system": {"healthy": True, "status": "operational"}}

        return {
            "status": "healthy" if all(check.get("healthy", False) for check in health_checks.values()) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": health_checks,
            "monitoring_service": {
                "initialized": True,  # monitoring_service.is_initialized,
                "metrics_collection": False  # monitoring_service.metrics_collection_task is not None
            }
        }
        
    except Exception as e:
        logger.error("Failed to get monitoring health", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve monitoring health")


# ============================================================================
# AGENT BUILDER PLATFORM MONITORING ENDPOINTS
# ============================================================================

class AgentBuilderMetricsResponse(BaseModel):
    """Agent Builder platform metrics response."""
    timestamp: datetime = Field(default_factory=datetime.now)
    registry_stats: Dict[str, Any]
    llm_provider_health: Dict[str, Any]
    agent_performance: Dict[str, Any]
    system_integration_status: Dict[str, Any]


class AgentRegistryStatsResponse(BaseModel):
    """Agent registry statistics response."""
    total_agents: int
    agents_by_status: Dict[str, int]
    agents_by_type: Dict[str, int]
    agents_by_health: Dict[str, int]
    collaboration_groups: int
    tenants: int
    top_performing_agents: List[Dict[str, Any]]


class LLMProviderHealthResponse(BaseModel):
    """LLM provider health monitoring response."""
    providers: Dict[str, Dict[str, Any]]
    total_requests: int
    total_errors: int
    average_response_time: float
    model_usage_distribution: Dict[str, int]


@router.get("/agent-builder/metrics", response_model=AgentBuilderMetricsResponse, tags=["Agent Builder Monitoring"])
async def get_agent_builder_metrics() -> AgentBuilderMetricsResponse:
    """
    Get comprehensive Agent Builder platform metrics.

    Provides detailed insights into agent registry, LLM providers,
    performance metrics, and system integration status.
    """
    try:
        logger.info("Getting Agent Builder platform metrics")

        # Get agent registry stats
        registry_stats = {}
        agent_registry = get_agent_registry()
        if agent_registry:
            registry_stats = agent_registry.get_registry_stats()

        # Get LLM provider health
        llm_provider_health = {}
        llm_manager = get_enhanced_llm_manager()
        if llm_manager and llm_manager.is_initialized():
            llm_provider_health = llm_manager.get_provider_health_status()

        # Get agent performance metrics
        agent_performance = {}
        if llm_manager:
            usage_stats = llm_manager.get_model_usage_stats()
            agent_performance = {
                "model_usage_stats": usage_stats,
                "total_model_requests": sum(stats.get("total_requests", 0) for stats in usage_stats.values()),
                "total_tokens_processed": sum(stats.get("total_tokens", 0) for stats in usage_stats.values()),
                "average_response_time": sum(stats.get("avg_response_time", 0) for stats in usage_stats.values()) / len(usage_stats) if usage_stats else 0
            }

        # Get system integration status
        system_integration_status = {}
        enhanced_orchestrator = get_enhanced_system_orchestrator()
        if enhanced_orchestrator:
            system_integration_status = enhanced_orchestrator.get_system_status()

        response = AgentBuilderMetricsResponse(
            registry_stats=registry_stats,
            llm_provider_health=llm_provider_health,
            agent_performance=agent_performance,
            system_integration_status=system_integration_status
        )

        logger.info("Agent Builder metrics retrieved successfully")
        return response

    except Exception as e:
        logger.error(f"Failed to get Agent Builder metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@router.get("/agent-builder/registry/stats", response_model=AgentRegistryStatsResponse, tags=["Agent Builder Monitoring"])
async def get_agent_registry_statistics() -> AgentRegistryStatsResponse:
    """
    Get detailed agent registry statistics.

    Provides comprehensive insights into agent distribution,
    performance, and operational metrics.
    """
    try:
        logger.info("Getting agent registry statistics")

        agent_registry = get_agent_registry()
        if not agent_registry:
            raise HTTPException(status_code=503, detail="Agent registry not available")

        # Get basic registry stats
        stats = agent_registry.get_registry_stats()

        # Get top performing agents
        agents = agent_registry.list_agents()
        top_performing_agents = []

        for agent in agents[:10]:  # Top 10 agents
            agent_metrics = agent_registry.get_agent_metrics(agent.agent_id)
            if agent_metrics:
                performance_data = {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "agent_type": agent.agent_type.value,
                    "status": agent.status.value,
                    "health": agent.health.value,
                    "total_executions": agent_metrics.total_executions,
                    "success_rate": (agent_metrics.successful_executions / agent_metrics.total_executions * 100) if agent_metrics.total_executions > 0 else 0,
                    "average_execution_time": agent_metrics.average_execution_time,
                    "last_activity": agent_metrics.last_activity.isoformat() if agent_metrics.last_activity else None
                }
                top_performing_agents.append(performance_data)

        # Sort by success rate and execution count
        top_performing_agents.sort(key=lambda x: (x["success_rate"], x["total_executions"]), reverse=True)

        response = AgentRegistryStatsResponse(
            total_agents=stats["total_agents"],
            agents_by_status=stats["agents_by_status"],
            agents_by_type=stats["agents_by_type"],
            agents_by_health=stats["agents_by_health"],
            collaboration_groups=stats["collaboration_groups"],
            tenants=stats["tenants"],
            top_performing_agents=top_performing_agents[:5]  # Top 5
        )

        logger.info(f"Agent registry statistics retrieved: {stats['total_agents']} total agents")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent registry statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@router.get("/agent-builder/llm-providers/health", response_model=LLMProviderHealthResponse, tags=["Agent Builder Monitoring"])
async def get_llm_provider_health() -> LLMProviderHealthResponse:
    """
    Get LLM provider health and performance metrics.

    Monitors the health and performance of all configured
    LLM providers including usage statistics and error rates.
    """
    try:
        logger.info("Getting LLM provider health metrics")

        llm_manager = get_enhanced_llm_manager()
        if not llm_manager or not llm_manager.is_initialized():
            raise HTTPException(status_code=503, detail="LLM manager not available")

        # Get provider health status
        provider_health = llm_manager.get_provider_health_status()

        # Get usage statistics
        usage_stats = llm_manager.get_model_usage_stats()

        # Calculate aggregate metrics
        total_requests = sum(stats.get("total_requests", 0) for stats in usage_stats.values())
        total_errors = sum(stats.get("error_count", 0) for stats in usage_stats.values())

        # Calculate average response time
        response_times = [stats.get("avg_response_time", 0) for stats in usage_stats.values() if stats.get("avg_response_time", 0) > 0]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0

        # Model usage distribution
        model_usage_distribution = {}
        for model_key, stats in usage_stats.items():
            model_usage_distribution[model_key] = stats.get("total_requests", 0)

        response = LLMProviderHealthResponse(
            providers=provider_health,
            total_requests=total_requests,
            total_errors=total_errors,
            average_response_time=average_response_time,
            model_usage_distribution=model_usage_distribution
        )

        logger.info(f"LLM provider health retrieved: {len(provider_health)} providers")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get LLM provider health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve provider health: {str(e)}")


@router.get("/agent-builder/performance/dashboard", response_model=Dict[str, Any], tags=["Agent Builder Monitoring"])
async def get_agent_builder_dashboard() -> Dict[str, Any]:
    """
    Get comprehensive Agent Builder platform dashboard data.

    Provides all key metrics and insights needed for a
    monitoring dashboard in a single endpoint.
    """
    try:
        logger.info("Getting Agent Builder dashboard data")

        # Get all monitoring data
        agent_metrics = await get_agent_builder_metrics()
        registry_stats = await get_agent_registry_statistics()
        provider_health = await get_llm_provider_health()

        # Calculate key performance indicators
        kpis = {
            "total_agents": registry_stats.total_agents,
            "healthy_agents": registry_stats.agents_by_health.get("healthy", 0),
            "active_agents": registry_stats.agents_by_status.get("running", 0),
            "total_llm_requests": provider_health.total_requests,
            "llm_error_rate": (provider_health.total_errors / provider_health.total_requests * 100) if provider_health.total_requests > 0 else 0,
            "average_response_time": provider_health.average_response_time,
            "collaboration_groups": registry_stats.collaboration_groups,
            "system_health": "healthy" if agent_metrics.system_integration_status.get("status") == "running" else "degraded"
        }

        # Recent activity summary
        recent_activity = {
            "new_agents_24h": 0,  # Would be calculated from actual data
            "completed_tasks_24h": 0,  # Would be calculated from actual data
            "errors_24h": provider_health.total_errors,
            "peak_concurrent_agents": registry_stats.agents_by_status.get("running", 0)
        }

        # Alerts and recommendations
        alerts = []
        recommendations = []

        # Generate alerts based on metrics
        if kpis["llm_error_rate"] > 5:
            alerts.append({
                "level": "warning",
                "message": f"High LLM error rate: {kpis['llm_error_rate']:.1f}%",
                "timestamp": datetime.now().isoformat()
            })

        if kpis["average_response_time"] > 5:
            alerts.append({
                "level": "warning",
                "message": f"High average response time: {kpis['average_response_time']:.2f}s",
                "timestamp": datetime.now().isoformat()
            })

        # Generate recommendations
        if registry_stats.total_agents > 0:
            healthy_ratio = kpis["healthy_agents"] / registry_stats.total_agents
            if healthy_ratio < 0.9:
                recommendations.append({
                    "priority": "high",
                    "message": "Consider investigating unhealthy agents",
                    "action": "Review agent health status and logs"
                })

        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "kpis": kpis,
            "registry_overview": {
                "total_agents": registry_stats.total_agents,
                "agents_by_type": registry_stats.agents_by_type,
                "agents_by_status": registry_stats.agents_by_status,
                "agents_by_health": registry_stats.agents_by_health,
                "top_performers": registry_stats.top_performing_agents[:3]
            },
            "llm_overview": {
                "providers": len(provider_health.providers),
                "total_requests": provider_health.total_requests,
                "error_rate": kpis["llm_error_rate"],
                "avg_response_time": provider_health.average_response_time,
                "model_distribution": provider_health.model_usage_distribution
            },
            "recent_activity": recent_activity,
            "alerts": alerts,
            "recommendations": recommendations,
            "system_status": agent_metrics.system_integration_status
        }

        logger.info("Agent Builder dashboard data retrieved successfully")
        return dashboard_data

    except Exception as e:
        logger.error(f"Failed to get Agent Builder dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")
