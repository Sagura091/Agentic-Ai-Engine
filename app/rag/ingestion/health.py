"""
Health check system for RAG ingestion pipeline.

This module provides comprehensive health checks:
- Component health status
- Dependency checks
- Resource availability
- Performance metrics
- Readiness and liveness probes
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
            "duration_ms": self.duration_ms
        }


class HealthCheck:
    """
    Health check for a component.
    
    Provides:
    - Async health check execution
    - Timeout handling
    - Error recovery
    - Result caching
    """
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult]],
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: int = 30
    ):
        """
        Initialize health check.
        
        Args:
            name: Check name
            check_func: Async function that performs the check
            timeout_seconds: Timeout for check
            cache_ttl_seconds: Cache TTL
        """
        self.name = name
        self.check_func = check_func
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Cache
        self._cached_result: Optional[HealthCheckResult] = None
        self._cache_expires_at: Optional[datetime] = None
    
    async def execute(self, use_cache: bool = True) -> HealthCheckResult:
        """
        Execute health check.
        
        Args:
            use_cache: Use cached result if available
            
        Returns:
            HealthCheckResult
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            return self._cached_result
        
        # Execute check with timeout
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(
                self.check_func(),
                timeout=self.timeout_seconds
            )
            
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            result.duration_ms = duration_ms
            
        except asyncio.TimeoutError:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            result = HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            result = HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=duration_ms
            )
        
        # Update cache
        self._cached_result = result
        self._cache_expires_at = datetime.utcnow() + timedelta(seconds=self.cache_ttl_seconds)
        
        return result
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is valid."""
        if self._cached_result is None or self._cache_expires_at is None:
            return False
        
        return datetime.utcnow() < self._cache_expires_at


class HealthCheckRegistry:
    """
    Registry for health checks.
    
    Provides:
    - Component health checks
    - Aggregate health status
    - Readiness and liveness probes
    - Dependency tracking
    """
    
    def __init__(self):
        """Initialize health check registry."""
        self._checks: Dict[str, HealthCheck] = {}
        self._lock = asyncio.Lock()
        
        logger.info("HealthCheckRegistry initialized")
    
    def register(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult]],
        timeout_seconds: float = 5.0,
        cache_ttl_seconds: int = 30
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Check function
            timeout_seconds: Timeout
            cache_ttl_seconds: Cache TTL
        """
        check = HealthCheck(
            name=name,
            check_func=check_func,
            timeout_seconds=timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds
        )
        
        self._checks[name] = check
        
        logger.info("Health check registered", name=name)
    
    async def check(self, name: str, use_cache: bool = True) -> Optional[HealthCheckResult]:
        """
        Execute a specific health check.
        
        Args:
            name: Check name
            use_cache: Use cache
            
        Returns:
            HealthCheckResult if check exists
        """
        check = self._checks.get(name)
        
        if not check:
            return None
        
        return await check.execute(use_cache=use_cache)
    
    async def check_all(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """
        Execute all health checks.
        
        Args:
            use_cache: Use cache
            
        Returns:
            Dictionary of check_name -> HealthCheckResult
        """
        results = {}
        
        # Execute all checks in parallel
        tasks = {
            name: check.execute(use_cache=use_cache)
            for name, check in self._checks.items()
        }
        
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to execute check: {str(e)}"
                )
        
        return results
    
    async def get_aggregate_status(self, use_cache: bool = True) -> HealthStatus:
        """
        Get aggregate health status.
        
        Args:
            use_cache: Use cache
            
        Returns:
            Aggregate HealthStatus
        """
        results = await self.check_all(use_cache=use_cache)
        
        if not results:
            return HealthStatus.UNKNOWN
        
        # If any check is unhealthy, overall is unhealthy
        if any(r.status == HealthStatus.UNHEALTHY for r in results.values()):
            return HealthStatus.UNHEALTHY
        
        # If any check is degraded, overall is degraded
        if any(r.status == HealthStatus.DEGRADED for r in results.values()):
            return HealthStatus.DEGRADED
        
        # All checks healthy
        return HealthStatus.HEALTHY
    
    async def liveness_probe(self) -> bool:
        """
        Liveness probe - is the service alive?
        
        Returns:
            True if alive
        """
        # For liveness, we just check if critical components are not completely dead
        critical_checks = ["pipeline", "processor_registry"]
        
        for check_name in critical_checks:
            if check_name in self._checks:
                result = await self.check(check_name, use_cache=True)
                if result and result.status == HealthStatus.UNHEALTHY:
                    return False
        
        return True
    
    async def readiness_probe(self) -> bool:
        """
        Readiness probe - is the service ready to accept traffic?
        
        Returns:
            True if ready
        """
        # For readiness, all checks must be healthy or degraded
        status = await self.get_aggregate_status(use_cache=True)
        return status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    async def get_health_report(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive health report.
        
        Args:
            use_cache: Use cache
            
        Returns:
            Health report dictionary
        """
        results = await self.check_all(use_cache=use_cache)
        aggregate_status = await self.get_aggregate_status(use_cache=use_cache)
        
        return {
            "status": aggregate_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                name: result.to_dict()
                for name, result in results.items()
            },
            "liveness": await self.liveness_probe(),
            "readiness": await self.readiness_probe()
        }


# Standard health check functions

async def check_pipeline_health(pipeline) -> HealthCheckResult:
    """Check pipeline health."""
    try:
        # Check if pipeline is initialized
        if not hasattr(pipeline, 'processor_registry') or pipeline.processor_registry is None:
            return HealthCheckResult(
                component="pipeline",
                status=HealthStatus.UNHEALTHY,
                message="Pipeline not initialized"
            )
        
        # Check queue depth
        queue_size = pipeline.processing_queue.qsize() if hasattr(pipeline, 'processing_queue') else 0
        
        # Check if shutdown
        is_shutdown = pipeline._shutdown_event.is_set() if hasattr(pipeline, '_shutdown_event') else False
        
        if is_shutdown:
            return HealthCheckResult(
                component="pipeline",
                status=HealthStatus.UNHEALTHY,
                message="Pipeline is shutting down"
            )
        
        # Determine status based on queue depth
        if queue_size > 1000:
            status = HealthStatus.DEGRADED
            message = f"Queue depth high: {queue_size}"
        else:
            status = HealthStatus.HEALTHY
            message = "Pipeline operational"
        
        return HealthCheckResult(
            component="pipeline",
            status=status,
            message=message,
            details={
                "queue_size": queue_size,
                "is_shutdown": is_shutdown
            }
        )
        
    except Exception as e:
        return HealthCheckResult(
            component="pipeline",
            status=HealthStatus.UNHEALTHY,
            message=f"Pipeline check failed: {str(e)}"
        )


async def check_knowledge_base_health(kb_manager) -> HealthCheckResult:
    """Check knowledge base health."""
    try:
        # Try to get collections
        collections = await kb_manager.list_collections()
        
        return HealthCheckResult(
            component="knowledge_base",
            status=HealthStatus.HEALTHY,
            message="Knowledge base operational",
            details={
                "collections": len(collections)
            }
        )
        
    except Exception as e:
        return HealthCheckResult(
            component="knowledge_base",
            status=HealthStatus.UNHEALTHY,
            message=f"Knowledge base check failed: {str(e)}"
        )


async def check_embedding_cache_health(cache) -> HealthCheckResult:
    """Check embedding cache health."""
    try:
        stats = cache.get_stats()
        
        utilization = stats.get("utilization", 0.0)
        hit_rate = stats.get("hit_rate", 0.0)
        
        # Determine status
        if utilization > 0.95:
            status = HealthStatus.DEGRADED
            message = f"Cache nearly full: {utilization:.1%}"
        elif hit_rate < 0.1 and stats.get("total_requests", 0) > 100:
            status = HealthStatus.DEGRADED
            message = f"Low cache hit rate: {hit_rate:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = "Cache operational"
        
        return HealthCheckResult(
            component="embedding_cache",
            status=status,
            message=message,
            details=stats
        )
        
    except Exception as e:
        return HealthCheckResult(
            component="embedding_cache",
            status=HealthStatus.UNHEALTHY,
            message=f"Cache check failed: {str(e)}"
        )


# Global health check registry
_health_registry: Optional[HealthCheckRegistry] = None
_health_lock = asyncio.Lock()


async def get_health_registry() -> HealthCheckRegistry:
    """
    Get global health check registry.
    
    Returns:
        HealthCheckRegistry instance
    """
    global _health_registry
    
    if _health_registry is None:
        async with _health_lock:
            if _health_registry is None:
                _health_registry = HealthCheckRegistry()
    
    return _health_registry

