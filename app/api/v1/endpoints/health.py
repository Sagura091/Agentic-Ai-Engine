"""
Health check endpoints for the Agentic AI Microservice.

This module provides comprehensive health checking capabilities including
service status, dependency checks, and system metrics.
"""

import asyncio
import time
from typing import Dict, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config.settings import get_settings
from app.core.exceptions import ExternalServiceError

logger = structlog.get_logger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: float
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]


class ServiceCheck(BaseModel):
    """Individual service check result."""
    status: str
    response_time_ms: float
    error: str = None


# Track application start time
_start_time = time.time()


@router.get("/", response_model=HealthStatus)
async def health_check(settings = Depends(get_settings)):
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    current_time = time.time()
    uptime = current_time - _start_time
    
    # Perform basic checks
    checks = await _perform_health_checks(settings)
    
    # Determine overall status
    overall_status = "healthy"
    for check_name, check_result in checks.items():
        if check_result.get("status") != "healthy":
            overall_status = "unhealthy"
            break
    
    return HealthStatus(
        status=overall_status,
        timestamp=current_time,
        version=settings.VERSION,
        uptime_seconds=uptime,
        checks=checks,
    )


@router.get("/ready")
async def readiness_check(settings = Depends(get_settings)):
    """
    Readiness check for Kubernetes/container orchestration.
    
    Returns:
        Simple ready/not ready status
    """
    try:
        checks = await _perform_health_checks(settings)
        
        # Check if all critical services are healthy
        critical_services = ["database", "redis"]
        for service in critical_services:
            if service in checks and checks[service].get("status") != "healthy":
                raise HTTPException(
                    status_code=503,
                    detail=f"Service not ready: {service} is {checks[service].get('status')}"
                )
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/container orchestration.
    
    Returns:
        Simple alive/dead status
    """
    return {"status": "alive"}


async def _perform_health_checks(settings) -> Dict[str, Dict[str, Any]]:
    """
    Perform health checks for all dependencies.
    
    Args:
        settings: Application settings
        
    Returns:
        Dictionary of health check results
    """
    checks = {}
    
    # Run all checks concurrently
    check_tasks = [
        _check_database(settings),
        _check_redis(settings),
        _check_ollama(settings),
        _check_openwebui(settings),
        _check_disk_space(settings),
        _check_memory_usage(),
    ]
    
    check_names = [
        "database",
        "redis", 
        "ollama",
        "openwebui",
        "disk_space",
        "memory",
    ]
    
    try:
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        for name, result in zip(check_names, results):
            if isinstance(result, Exception):
                checks[name] = {
                    "status": "unhealthy",
                    "error": str(result),
                    "response_time_ms": 0,
                }
            else:
                checks[name] = result
                
    except Exception as e:
        logger.error("Health checks failed", error=str(e))
        checks["error"] = {"status": "unhealthy", "error": str(e)}
    
    return checks


async def _check_database(settings) -> Dict[str, Any]:
    """Check database connectivity."""
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        from app.models.database.base import get_database_session
        
        async for session in get_database_session():
            # Simple query to test connection
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
            break  # Only need one iteration for health check
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
        }
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": round(response_time, 2),
        }


async def _check_redis(settings) -> Dict[str, Any]:
    """Check Redis connectivity."""
    start_time = time.time()
    
    try:
        import aioredis
        
        redis = aioredis.from_url(settings.REDIS_URL)
        await redis.ping()
        await redis.close()
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
        }
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": round(response_time, 2),
        }


async def _check_ollama(settings) -> Dict[str, Any]:
    """Check Ollama service connectivity using connection pooling."""
    start_time = time.time()

    try:
        from app.http_client import HTTPClient, ClientConfig, ConnectionPoolConfig

        # Use HTTPClient with connection pooling for faster health checks
        config = ClientConfig(
            timeout=5,
            verify_ssl=False,
            pool_config=ConnectionPoolConfig(max_per_host=2, keepalive_timeout=30)
        )
        async with HTTPClient(settings.OLLAMA_BASE_URL, config) as client:
            response = await client.get("/api/tags", stream=False)
            response.raise_for_status()

        response_time = (time.time() - start_time) * 1000

        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
        }

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": round(response_time, 2),
        }


async def _check_openwebui(settings) -> Dict[str, Any]:
    """Check OpenWebUI service connectivity using connection pooling."""
    start_time = time.time()

    try:
        from app.http_client import HTTPClient, ClientConfig, ConnectionPoolConfig

        # Use HTTPClient with connection pooling for faster health checks
        config = ClientConfig(
            timeout=5,
            verify_ssl=False,
            pool_config=ConnectionPoolConfig(max_per_host=2, keepalive_timeout=30)
        )
        async with HTTPClient(settings.OPENWEBUI_BASE_URL, config) as client:
            response = await client.get("/api/config", stream=False)
            response.raise_for_status()

        response_time = (time.time() - start_time) * 1000

        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
        }
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": round(response_time, 2),
        }


async def _check_disk_space(settings) -> Dict[str, Any]:
    """Check available disk space."""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage(settings.DATA_DIR)
        free_percent = (free / total) * 100
        
        status = "healthy" if free_percent > 10 else "unhealthy"
        
        return {
            "status": status,
            "free_space_gb": round(free / (1024**3), 2),
            "free_percent": round(free_percent, 2),
            "total_space_gb": round(total / (1024**3), 2),
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def _check_memory_usage() -> Dict[str, Any]:
    """Check memory usage."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        available_percent = memory.available / memory.total * 100
        
        status = "healthy" if available_percent > 10 else "unhealthy"
        
        return {
            "status": status,
            "available_percent": round(available_percent, 2),
            "used_percent": round(memory.percent, 2),
            "total_gb": round(memory.total / (1024**3), 2),
        }
        
    except ImportError:
        # psutil not available, skip memory check
        return {
            "status": "unknown",
            "error": "psutil not available",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/status")
async def detailed_status_check():
    """
    Detailed status check endpoint.

    Returns:
        Detailed system status information
    """
    from datetime import datetime

    return {
        "status": "healthy",
        "service": "agentic-ai-microservice",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - _start_time,
        "capabilities": [
            "unlimited_agents",
            "dynamic_tools",
            "autonomous_intelligence",
            "multi_agent_coordination"
        ]
    }
