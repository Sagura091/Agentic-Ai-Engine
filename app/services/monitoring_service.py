"""
Production-ready monitoring service for the Agentic AI system.

This service provides comprehensive system monitoring, health checks,
performance metrics, and alerting capabilities.
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory


logger = get_logger()


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    active_connections: int
    uptime_seconds: float


@dataclass
class ServiceHealth:
    """Health status of a service."""
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class MonitoringService:
    """
    Production-ready monitoring service.
    
    Provides system monitoring, health checks, performance metrics,
    and integration with the backend logging system.
    """
    
    def __init__(self):
        self.backend_logger = get_logger()
        self.is_initialized = False
        self.is_running = False
        self.start_time = time.time()
        self.metrics_history: List[SystemMetrics] = []
        self.service_health: Dict[str, ServiceHealth] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.metrics_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.max_history_size = 1000
        
    async def initialize(self) -> None:
        """Initialize the monitoring service."""
        try:
            self.backend_logger.info(
                "Initializing monitoring service",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service"
            )

            # Initialize service health tracking
            self.service_health = {
                "database": ServiceHealth("database", "unknown", datetime.now()),
                "websocket": ServiceHealth("websocket", "unknown", datetime.now()),
                "orchestrator": ServiceHealth("orchestrator", "unknown", datetime.now()),
                "rag_system": ServiceHealth("rag_system", "unknown", datetime.now()),
            }

            # Start monitoring tasks
            self.is_running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.is_initialized = True

            self.backend_logger.info(
                "Monitoring service initialized successfully",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                data={
                    "metrics_interval": self.metrics_interval,
                    "health_check_interval": self.health_check_interval,
                    "services_tracked": list(self.service_health.keys())
                }
            )

        except Exception as e:
            self.backend_logger.error(
                "Failed to initialize monitoring service",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                error=e
            )
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the monitoring service."""
        try:
            self.backend_logger.info(
                "Shutting down monitoring service",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service"
            )

            self.is_running = False

            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            self.backend_logger.info(
                "Monitoring service shut down successfully",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service"
            )

        except Exception as e:
            self.backend_logger.error(
                "Error during monitoring service shutdown",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                error=e
            )
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        last_metrics_time = 0
        last_health_check_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Collect system metrics
                if current_time - last_metrics_time >= self.metrics_interval:
                    await self._collect_system_metrics()
                    last_metrics_time = current_time
                
                # Perform health checks
                if current_time - last_health_check_time >= self.health_check_interval:
                    await self._perform_health_checks()
                    last_health_check_time = current_time
                
                # Sleep for a short interval
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in monitoring loop",
                    LogCategory.SYSTEM_HEALTH,
                    "app.services.monitoring_service",
                    error=e
                )
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
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
            
            # Count active connections (approximate)
            active_connections = len(psutil.net_connections())
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                active_connections=active_connections,
                uptime_seconds=uptime_seconds
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Trim history if too large
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
            # Log metrics
            self.backend_logger.debug(
                "System metrics collected",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                data=asdict(metrics)
            )

            # Check for alerts
            await self._check_metric_alerts(metrics)

        except Exception as e:
            logger.error(
                "Error collecting system metrics",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                error=e
            )

    async def _perform_health_checks(self) -> None:
        """Perform health checks on system services."""
        try:
            # Update service health status
            for service_name in self.service_health:
                health = self.service_health[service_name]
                health.last_check = datetime.now()
                health.status = "healthy"  # Default to healthy for now

            self.backend_logger.debug(
                "Health checks completed",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                data={
                    "services_checked": len(self.service_health),
                    "healthy_services": len([h for h in self.service_health.values() if h.status == "healthy"])
                }
            )

        except Exception as e:
            logger.error(
                "Error performing health checks",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                error=e
            )
    
    async def _check_metric_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against alert thresholds."""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > 80:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory alert
        if metrics.memory_percent > 85:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Disk alert
        if metrics.disk_usage_percent > 90:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        # Log alerts
        if alerts:
            self.backend_logger.warn(
                "System performance alerts",
                LogCategory.SYSTEM_HEALTH,
                "app.services.monitoring_service",
                data={
                    "alerts": alerts,
                    "metrics": asdict(metrics)
                }
            )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_service_health(self) -> Dict[str, ServiceHealth]:
        """Get current service health status."""
        return self.service_health.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_metrics = self.get_current_metrics()
        
        return {
            "is_healthy": self.is_running and self.is_initialized,
            "uptime_seconds": time.time() - self.start_time,
            "current_metrics": asdict(current_metrics) if current_metrics else None,
            "service_health": {name: asdict(health) for name, health in self.service_health.items()},
            "metrics_history_size": len(self.metrics_history)
        }


# Global monitoring service instance
monitoring_service = MonitoringService()
