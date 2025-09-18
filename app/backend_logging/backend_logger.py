"""
Backend Logger - Core Logging Orchestrator

Provides the main logging interface for the agentic AI microservice backend.
Handles structured logging, context management, and integration with various
backend components.
"""

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import threading
import queue
import time

from .models import (
    LogEntry, LogLevel, LogCategory, LogContext, 
    PerformanceMetrics, ErrorDetails, AgentMetrics,
    APIMetrics, DatabaseMetrics, LogConfiguration
)
from .context import CorrelationContext, SystemContext
from .formatters import JSONFormatter, StructuredFormatter
from .handlers import AsyncFileHandler


class BackendLogger:
    """
    Main backend logger class that provides comprehensive logging capabilities
    for the agentic AI microservice.
    """
    
    def __init__(self, config: LogConfiguration = None):
        self.config = config or LogConfiguration()
        self.log_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.is_running = True
        self.worker_thread = None
        self.file_handlers = {}
        self.console_handler = None
        
        # Initialize logging infrastructure
        self._setup_logging()
        self._start_worker_thread()
        
        # Log system initialization
        self.info(
            "Backend logging system initialized",
            category=LogCategory.SYSTEM_HEALTH,
            component="BackendLogger",
            data={"config": self.config.dict()}
        )
    
    def _setup_logging(self):
        """Setup logging handlers and formatters"""
        # Create logs directory
        logs_dir = Path("logs/backend")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file handlers for different categories
        if self.config.enable_file_output:
            for category in LogCategory:
                log_file = logs_dir / f"{category.value}_{datetime.now().strftime('%Y%m%d')}.log"
                handler = AsyncFileHandler(
                    filename=str(log_file),
                    max_bytes=self.config.max_log_file_size_mb * 1024 * 1024,
                    backup_count=self.config.max_log_files
                )
                
                if self.config.enable_json_format:
                    handler.setFormatter(JSONFormatter())
                else:
                    handler.setFormatter(StructuredFormatter())
                
                self.file_handlers[category] = handler
        
        # Setup console handler
        if self.config.enable_console_output:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setFormatter(StructuredFormatter())
    
    def _start_worker_thread(self):
        """Start the background worker thread for async logging"""
        if self.config.enable_async_logging:
            self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.worker_thread.start()
    
    def _log_worker(self):
        """Background worker that processes log entries"""
        while self.is_running:
            try:
                # Process logs in batches
                logs_to_process = []
                
                # Collect logs with timeout
                try:
                    log_entry = self.log_queue.get(timeout=self.config.flush_interval_seconds)
                    logs_to_process.append(log_entry)
                    
                    # Collect additional logs without blocking
                    while len(logs_to_process) < self.config.buffer_size:
                        try:
                            log_entry = self.log_queue.get_nowait()
                            logs_to_process.append(log_entry)
                        except queue.Empty:
                            break
                
                except queue.Empty:
                    continue
                
                # Process the batch
                for log_entry in logs_to_process:
                    self._write_log_entry(log_entry)
                    
            except Exception as e:
                # Don't let logging errors crash the worker
                print(f"Logging worker error: {e}", file=sys.stderr)
    
    def _write_log_entry(self, log_entry: LogEntry):
        """Write a log entry to appropriate handlers"""
        try:
            # Write to file handler for the category
            if log_entry.category in self.file_handlers:
                handler = self.file_handlers[log_entry.category]
                record = self._create_log_record(log_entry)
                handler.emit(record)
            
            # Write to console if enabled and appropriate level
            if (self.console_handler and 
                self._should_log_to_console(log_entry.level)):
                record = self._create_log_record(log_entry)
                self.console_handler.emit(record)
                
        except Exception as e:
            print(f"Error writing log entry: {e}", file=sys.stderr)
    
    def _create_log_record(self, log_entry: LogEntry) -> logging.LogRecord:
        """Create a Python logging record from our log entry"""
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.FATAL: logging.CRITICAL
        }
        
        record = logging.LogRecord(
            name=log_entry.component,
            level=level_mapping.get(log_entry.level, logging.INFO),
            pathname="",
            lineno=0,
            msg=log_entry.message,
            args=(),
            exc_info=None
        )
        
        # Add our custom fields
        record.log_entry = log_entry
        return record
    
    def _should_log_to_console(self, level: LogLevel) -> bool:
        """Determine if a log level should be written to console"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARN: 2,
            LogLevel.ERROR: 3,
            LogLevel.FATAL: 4
        }
        
        return level_order.get(level, 1) >= level_order.get(self.config.log_level, 1)
    
    def _should_exclude_message(self, message: str) -> bool:
        """Check if message should be excluded based on patterns"""
        return any(pattern in message.lower() for pattern in self.config.exclude_patterns)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory,
        component: str,
        data: Optional[Dict[str, Any]] = None,
        context: Optional[LogContext] = None,
        performance: Optional[PerformanceMetrics] = None,
        agent_metrics: Optional[AgentMetrics] = None,
        api_metrics: Optional[APIMetrics] = None,
        database_metrics: Optional[DatabaseMetrics] = None,
        error_details: Optional[ErrorDetails] = None,
        **kwargs
    ):
        """
        Main logging method that creates and queues log entries
        """
        # Skip if message should be excluded
        if self._should_exclude_message(message):
            return
        
        # Get or create context
        if context is None:
            context = CorrelationContext.get_context()
        
        # Add system information
        system_info = SystemContext.get_system_info()
        
        # Create log entry
        log_entry = LogEntry(
            level=level,
            category=category,
            message=message,
            component=component,
            context=context,
            data=data,
            performance=performance,
            agent_metrics=agent_metrics,
            api_metrics=api_metrics,
            database_metrics=database_metrics,
            error_details=error_details,
            hostname=system_info.get("hostname"),
            process_id=system_info.get("process_id"),
            thread_id=str(system_info.get("thread_id")),
            environment=kwargs.get("environment", "development"),
            version=kwargs.get("version", "1.0.0")
        )
        
        # Add to queue for async processing
        if self.config.enable_async_logging:
            try:
                self.log_queue.put_nowait(log_entry)
            except queue.Full:
                # If queue is full, write directly (blocking)
                self._write_log_entry(log_entry)
        else:
            # Synchronous logging
            self._write_log_entry(log_entry)
    
    # Convenience methods for different log levels
    def debug(self, message: str, category: LogCategory, component: str, **kwargs):
        """Log a debug message"""
        self.log(LogLevel.DEBUG, message, category, component, **kwargs)
    
    def info(self, message: str, category: LogCategory, component: str, **kwargs):
        """Log an info message"""
        self.log(LogLevel.INFO, message, category, component, **kwargs)
    
    def warn(self, message: str, category: LogCategory, component: str, **kwargs):
        """Log a warning message"""
        self.log(LogLevel.WARN, message, category, component, **kwargs)
    
    def error(self, message: str, category: LogCategory, component: str, 
              error: Exception = None, **kwargs):
        """Log an error message with optional exception details"""
        error_details = None
        if error:
            error_details = ErrorDetails(
                error_type=type(error).__name__,
                error_code=getattr(error, 'code', None),
                stack_trace=traceback.format_exc() if self.config.include_stack_trace else None,
                severity="high" if isinstance(error, (SystemError, MemoryError)) else "medium"
            )
        
        self.log(LogLevel.ERROR, message, category, component, 
                error_details=error_details, **kwargs)
    
    def fatal(self, message: str, category: LogCategory, component: str, 
             error: Exception = None, **kwargs):
        """Log a fatal error message"""
        error_details = None
        if error:
            error_details = ErrorDetails(
                error_type=type(error).__name__,
                error_code=getattr(error, 'code', None),
                stack_trace=traceback.format_exc() if self.config.include_stack_trace else None,
                severity="critical",
                user_impact="service_unavailable"
            )
        
        self.log(LogLevel.FATAL, message, category, component, 
                error_details=error_details, **kwargs)
    
    def get_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        levels: Optional[List[LogLevel]] = None,
        categories: Optional[List[LogCategory]] = None,
        components: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Retrieve logs based on filters (placeholder for file-based implementation)
        In a production system, this would read from log files or a database
        """
        # This is a simplified implementation
        # In practice, you'd read from log files or a log database
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "queue_size": self.log_queue.qsize() if hasattr(self.log_queue, 'qsize') else 0,
            "worker_running": self.is_running,
            "handlers_count": len(self.file_handlers),
            "config": self.config.dict()
        }

    def shutdown(self):
        """Shutdown the logging system gracefully"""
        self.info(
            "Backend logging system shutting down",
            category=LogCategory.SYSTEM_HEALTH,
            component="BackendLogger"
        )

        self.is_running = False

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

        # Close all handlers
        for handler in self.file_handlers.values():
            handler.close()

        if self.console_handler:
            self.console_handler.close()


# Global logger instance
_global_logger: Optional[BackendLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> BackendLogger:
    """Get the global backend logger instance"""
    global _global_logger
    
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = BackendLogger()
    
    return _global_logger


def configure_logger(config: LogConfiguration) -> BackendLogger:
    """Configure the global logger with custom settings"""
    global _global_logger
    
    with _logger_lock:
        if _global_logger:
            _global_logger.shutdown()
        _global_logger = BackendLogger(config)
    
    return _global_logger
