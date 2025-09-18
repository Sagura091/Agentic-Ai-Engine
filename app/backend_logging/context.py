"""
Logging Context Management

Provides context management for maintaining correlation IDs, session information,
and other contextual data throughout the request lifecycle.
"""

import asyncio
import threading
from contextvars import ContextVar
from typing import Optional, Dict, Any
from uuid import uuid4
import time
import os
import psutil

from .models import LogContext


# Context variables for async operations
_correlation_context: ContextVar[Optional[LogContext]] = ContextVar('correlation_context', default=None)
_request_start_time: ContextVar[Optional[float]] = ContextVar('request_start_time', default=None)


class CorrelationContext:
    """Manages correlation context for request tracing"""
    
    @staticmethod
    def get_context() -> LogContext:
        """Get the current correlation context"""
        context = _correlation_context.get()
        if context is None:
            context = LogContext()
            _correlation_context.set(context)
        return context
    
    @staticmethod
    def set_context(context: LogContext) -> None:
        """Set the correlation context"""
        _correlation_context.set(context)
    
    @staticmethod
    def update_context(**kwargs) -> LogContext:
        """Update the current context with new values"""
        context = CorrelationContext.get_context()
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
        _correlation_context.set(context)
        return context
    
    @staticmethod
    def clear_context() -> None:
        """Clear the correlation context"""
        _correlation_context.set(None)
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID"""
        return str(uuid4())
    
    @staticmethod
    def start_request_timer() -> None:
        """Start timing a request"""
        _request_start_time.set(time.time())
    
    @staticmethod
    def get_request_duration() -> Optional[float]:
        """Get the duration of the current request in milliseconds"""
        start_time = _request_start_time.get()
        if start_time is not None:
            return (time.time() - start_time) * 1000
        return None


class SystemContext:
    """Provides system-level context information"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get current system information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent()
            
            return {
                "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown',
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "cpu_usage_percent": cpu_percent,
                "thread_count": threading.active_count(),
            }
        except Exception:
            return {
                "hostname": "unknown",
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
            }
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Get current CPU usage percentage"""
        try:
            process = psutil.Process()
            return process.cpu_percent()
        except Exception:
            return 0.0


class AgentContext:
    """Manages agent-specific context information"""
    
    def __init__(self, agent_id: str, agent_type: str = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.start_time = time.time()
        self.operations = []
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tools_used": set(),
            "api_calls_made": 0,
            "tokens_consumed": 0,
        }
    
    def add_operation(self, operation: str, duration_ms: float = None):
        """Add an operation to the agent's history"""
        self.operations.append({
            "operation": operation,
            "timestamp": time.time(),
            "duration_ms": duration_ms
        })
    
    def increment_metric(self, metric: str, value: int = 1):
        """Increment a metric counter"""
        if metric in self.metrics:
            if isinstance(self.metrics[metric], set):
                # For sets like tools_used, add the value
                self.metrics[metric].add(value)
            else:
                # For counters, increment
                self.metrics[metric] += value
    
    def get_execution_time(self) -> float:
        """Get total execution time in milliseconds"""
        return (time.time() - self.start_time) * 1000
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary"""
        metrics = self.metrics.copy()
        # Convert sets to lists for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, set):
                metrics[key] = list(value)
        
        metrics["execution_time_ms"] = self.get_execution_time()
        metrics["operations_count"] = len(self.operations)
        return metrics


class RequestContext:
    """Manages HTTP request-specific context"""
    
    def __init__(self, method: str, url: str, headers: Dict[str, str] = None):
        self.method = method
        self.url = url
        self.headers = headers or {}
        self.start_time = time.time()
        self.request_size = 0
        self.response_size = 0
        self.status_code = None
        self.user_agent = self.headers.get("user-agent", "unknown")
        self.client_ip = self.headers.get("x-forwarded-for", "unknown")
    
    def set_request_size(self, size: int):
        """Set the request body size"""
        self.request_size = size
    
    def set_response_info(self, status_code: int, response_size: int = 0):
        """Set response information"""
        self.status_code = status_code
        self.response_size = response_size
    
    def get_duration(self) -> float:
        """Get request duration in milliseconds"""
        return (time.time() - self.start_time) * 1000
    
    def get_api_metrics(self) -> Dict[str, Any]:
        """Get API metrics for this request"""
        return {
            "method": self.method,
            "endpoint": self.url,
            "status_code": self.status_code,
            "response_time_ms": self.get_duration(),
            "request_size_bytes": self.request_size,
            "response_size_bytes": self.response_size,
            "user_agent": self.user_agent,
            "client_ip": self.client_ip,
        }


class DatabaseContext:
    """Manages database operation context"""
    
    def __init__(self, operation_type: str, table_name: str = None):
        self.operation_type = operation_type
        self.table_name = table_name
        self.start_time = time.time()
        self.query_count = 0
        self.rows_affected = 0
        self.connection_info = {}
    
    def add_query(self, rows_affected: int = 0):
        """Add a query to the context"""
        self.query_count += 1
        self.rows_affected += rows_affected
    
    def set_connection_info(self, pool_size: int, active_connections: int):
        """Set connection pool information"""
        self.connection_info = {
            "pool_size": pool_size,
            "active_connections": active_connections
        }
    
    def get_duration(self) -> float:
        """Get operation duration in milliseconds"""
        return (time.time() - self.start_time) * 1000
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database metrics for this operation"""
        metrics = {
            "query_type": self.operation_type,
            "execution_time_ms": self.get_duration(),
            "query_count": self.query_count,
            "rows_affected": self.rows_affected,
        }
        
        if self.table_name:
            metrics["table_name"] = self.table_name
        
        metrics.update(self.connection_info)
        return metrics
