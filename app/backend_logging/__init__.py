"""
Backend Logging System for Agentic AI Microservice

This module provides comprehensive logging capabilities for the FastAPI backend,
including agent lifecycle monitoring, performance tracking, error handling,
and operational visibility.
"""

# Use lazy imports to avoid circular dependencies
from .formatters import StructuredFormatter, JSONFormatter
from .handlers import AsyncFileHandler, RotatingFileHandler

# Lazy imports for items that might cause circular dependencies
def get_logger(name: str = None):
    """Get a logger instance with lazy import to avoid circular dependencies."""
    from .backend_logger import get_logger as _get_logger
    return _get_logger()

def get_backend_logger():
    """Get the main backend logger with lazy import."""
    from .backend_logger import BackendLogger
    return BackendLogger

def get_log_context():
    """Get log context with lazy import."""
    from .context import LogContext, CorrelationContext
    return LogContext, CorrelationContext

def get_log_models():
    """Get log models with lazy import."""
    from .models import LogEntry, LogLevel, LogCategory
    return LogEntry, LogLevel, LogCategory

def get_logging_middleware():
    """Get logging middleware with lazy import."""
    from .middleware import LoggingMiddleware
    return LoggingMiddleware

__all__ = [
    'get_logger',
    'get_backend_logger',
    'get_log_context',
    'get_log_models',
    'get_logging_middleware',
    'StructuredFormatter',
    'JSONFormatter',
    'AsyncFileHandler',
    'RotatingFileHandler'
]
