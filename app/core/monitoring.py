"""
Comprehensive monitoring and logging system for the Agentic AI platform.

This module provides structured logging, metrics collection, and monitoring
capabilities for production environments.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog
import psutil
import json
from contextlib import asynccontextmanager
from functools import wraps

logger = structlog.get_logger(__name__)


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a request or operation."""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime
    success: bool
    error_code: Optional[str] = None
    request_id: Optional[str] = None


class MetricsCollector:
    """Collects and stores metrics for monitoring."""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def record_metric(self, metric: Metric):
        """Record a metric."""
        async with self.lock:
            self.metrics.append(metric)
            
            # Update counters
            counter_key = f"{metric.name}_count"
            self.counters[counter_key] += 1
            
            # Update gauges
            self.gauges[metric.name] = metric.value
            
            # Update histograms
            self.histograms[metric.name].append(metric.value)
            if len(self.histograms[metric.name]) > 1000:  # Keep last 1000 values
                self.histograms[metric.name] = self.histograms[metric.name][-1000:]
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        async with self.lock:
            return {
                "total_metrics": len(self.metrics),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "avg": sum(values) / len(values) if values else 0,
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
                    for name, values in self.histograms.items()
                }
            }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_total_gb = disk.total / (1024 * 1024 * 1024)
            disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            process_cpu_percent = process.cpu_percent()
            
            # Uptime
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime_seconds,
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "used_mb": memory_used_mb,
                    "total_mb": memory_total_mb,
                    "percent": memory_percent
                },
                "disk": {
                    "used_gb": disk_used_gb,
                    "total_gb": disk_total_gb,
                    "percent": disk_percent
                },
                "network": {
                    "bytes_sent": network_bytes_sent,
                    "bytes_recv": network_bytes_recv
                },
                "process": {
                    "memory_mb": process_memory_mb,
                    "cpu_percent": process_cpu_percent
                },
                "requests": {
                    "total": self.request_count,
                    "errors": self.error_count,
                    "error_rate": self.error_count / max(self.request_count, 1)
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    async def record_request(self, success: bool, duration_ms: float, operation: str):
        """Record a request metric."""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        # Record performance metric
        metric = Metric(
            name="request_duration",
            value=duration_ms,
            timestamp=datetime.utcnow(),
            tags={"operation": operation, "success": str(success)},
            unit="ms"
        )
        await self.metrics_collector.record_metric(metric)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        system_metrics = await self.get_system_metrics()
        metrics_summary = await self.metrics_collector.get_metrics_summary()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        # Check CPU usage
        if system_metrics.get("cpu", {}).get("percent", 0) > 80:
            health_status = "degraded"
            issues.append("High CPU usage")
        
        # Check memory usage
        if system_metrics.get("memory", {}).get("percent", 0) > 85:
            health_status = "degraded"
            issues.append("High memory usage")
        
        # Check disk usage
        if system_metrics.get("disk", {}).get("percent", 0) > 90:
            health_status = "critical"
            issues.append("High disk usage")
        
        # Check error rate
        error_rate = system_metrics.get("requests", {}).get("error_rate", 0)
        if error_rate > 0.1:  # 10% error rate
            health_status = "degraded"
            issues.append(f"High error rate: {error_rate:.2%}")
        
        return {
            "status": health_status,
            "issues": issues,
            "system_metrics": system_metrics,
            "metrics_summary": metrics_summary,
            "timestamp": datetime.utcnow().isoformat()
        }


class PerformanceMonitor:
    """Monitors performance of operations and requests."""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.active_operations: Dict[str, datetime] = {}
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, request_id: Optional[str] = None):
        """Context manager for monitoring operations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            duration_ms = (end_time - start_time) * 1000
            memory_usage_mb = end_memory - start_memory
            
            # Record metrics
            await self.system_monitor.record_request(
                success=success,
                duration_ms=duration_ms,
                operation=operation_name
            )
            
            # Store operation time
            self.operation_times[operation_name].append(duration_ms)
            if len(self.operation_times[operation_name]) > 1000:
                self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for an operation."""
        times = self.operation_times.get(operation_name, [])
        if not times:
            return {"count": 0}
        
        return {
            "count": len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "p50_ms": self._percentile(times, 50),
            "p95_ms": self._percentile(times, 95),
            "p99_ms": self._percentile(times, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable] = []
    
    def add_alert_rule(self, name: str, condition: Callable, severity: str = "warning"):
        """Add an alert rule."""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity
        })
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules."""
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    alert = {
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "timestamp": datetime.utcnow().isoformat(),
                        "metrics": metrics
                    }
                    self.alerts.append(alert)
                    
                    # Notify handlers
                    for handler in self.alert_handlers:
                        try:
                            await handler(alert)
                        except Exception as e:
                            logger.error(f"Alert handler failed: {e}")
            except Exception as e:
                logger.error(f"Alert rule {rule['name']} failed: {e}")
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return self.alerts[-100:]  # Last 100 alerts


class MonitoringService:
    """Main monitoring service that coordinates all monitoring components."""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.monitoring_enabled = True
        self.metrics_interval = 60  # seconds
        self.health_check_interval = 30  # seconds
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            lambda m: m.get("cpu", {}).get("percent", 0) > 80,
            "warning"
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            lambda m: m.get("memory", {}).get("percent", 0) > 85,
            "warning"
        )
        
        # High disk usage
        self.alert_manager.add_alert_rule(
            "high_disk_usage",
            lambda m: m.get("disk", {}).get("percent", 0) > 90,
            "critical"
        )
        
        # High error rate
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            lambda m: m.get("requests", {}).get("error_rate", 0) > 0.1,
            "warning"
        )
    
    async def start_monitoring(self):
        """Start the monitoring service."""
        if not self.monitoring_enabled:
            return
        
        logger.info("Starting monitoring service")
        
        # Start background tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._health_check_loop())
    
    async def _metrics_collection_loop(self):
        """Background task for collecting metrics."""
        while self.monitoring_enabled:
            try:
                system_metrics = await self.performance_monitor.system_monitor.get_system_metrics()
                await self.alert_manager.check_alerts(system_metrics)
                await asyncio.sleep(self.metrics_interval)
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _health_check_loop(self):
        """Background task for health checks."""
        while self.monitoring_enabled:
            try:
                health_status = await self.performance_monitor.system_monitor.get_health_status()
                if health_status["status"] != "healthy":
                    logger.warning(f"Health check failed: {health_status}")
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        system_metrics = await self.performance_monitor.system_monitor.get_health_status()
        metrics_summary = await self.performance_monitor.system_monitor.metrics_collector.get_metrics_summary()
        
        # Get operation statistics
        operation_stats = {}
        for operation in self.performance_monitor.operation_times.keys():
            operation_stats[operation] = self.performance_monitor.get_operation_stats(operation)
        
        return {
            "system_health": system_metrics,
            "metrics_summary": metrics_summary,
            "operation_stats": operation_stats,
            "active_alerts": self.alert_manager.get_active_alerts(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def monitor_function(self, operation_name: str):
        """Decorator for monitoring functions."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with self.performance_monitor.monitor_operation(operation_name):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator


# Global monitoring service instance
monitoring_service = MonitoringService()


# Convenience functions
async def start_monitoring():
    """Start the global monitoring service."""
    await monitoring_service.start_monitoring()


async def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    return await monitoring_service.performance_monitor.system_monitor.get_health_status()


async def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get monitoring dashboard data."""
    return await monitoring_service.get_monitoring_dashboard()


def monitor_operation(operation_name: str):
    """Decorator for monitoring operations."""
    return monitoring_service.monitor_function(operation_name)


# Export all components
__all__ = [
    "Metric", "PerformanceMetrics", "MetricsCollector", "SystemMonitor",
    "PerformanceMonitor", "AlertManager", "MonitoringService",
    "monitoring_service", "start_monitoring", "get_health_status",
    "get_monitoring_dashboard", "monitor_operation"
]


