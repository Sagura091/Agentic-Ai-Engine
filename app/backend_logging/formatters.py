"""
Logging Formatters

Provides various formatters for log output including JSON and structured text formats.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

from .models import LogEntry


class JSONFormatter(logging.Formatter):
    """
    JSON formatter that outputs log entries as structured JSON
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON"""
        try:
            # Get our custom log entry if available
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                log_dict = self._log_entry_to_dict(log_entry)
            else:
                # Fallback for standard logging records
                log_dict = self._standard_record_to_dict(record)
            
            return json.dumps(log_dict, separators=(',', ':'), default=str)
            
        except Exception as e:
            # Fallback to basic format if JSON serialization fails
            return f'{{"timestamp": "{datetime.utcnow().isoformat()}", "level": "ERROR", "message": "JSON formatting error: {str(e)}", "original_message": "{record.getMessage()}"}}'
    
    def _log_entry_to_dict(self, log_entry: LogEntry) -> Dict[str, Any]:
        """Convert LogEntry to dictionary"""
        log_dict = {
            "timestamp": log_entry.timestamp.isoformat(),
            "level": log_entry.level.value,
            "category": log_entry.category.value,
            "component": log_entry.component,
            "message": log_entry.message,
        }
        
        # Add context information
        if log_entry.context:
            context_dict = log_entry.context.dict(exclude_none=True)
            log_dict["context"] = context_dict
        
        # Add additional data
        if log_entry.data:
            log_dict["data"] = log_entry.data
        
        # Add metrics
        if log_entry.performance:
            log_dict["performance"] = log_entry.performance.dict(exclude_none=True)
        
        if log_entry.agent_metrics:
            log_dict["agent_metrics"] = log_entry.agent_metrics.dict(exclude_none=True)
        
        if log_entry.api_metrics:
            log_dict["api_metrics"] = log_entry.api_metrics.dict(exclude_none=True)
        
        if log_entry.database_metrics:
            log_dict["database_metrics"] = log_entry.database_metrics.dict(exclude_none=True)
        
        # Add error details
        if log_entry.error_details:
            log_dict["error_details"] = log_entry.error_details.dict(exclude_none=True)
        
        # Add system information
        if log_entry.hostname:
            log_dict["hostname"] = log_entry.hostname
        if log_entry.process_id:
            log_dict["process_id"] = log_entry.process_id
        if log_entry.thread_id:
            log_dict["thread_id"] = log_entry.thread_id
        if log_entry.environment:
            log_dict["environment"] = log_entry.environment
        if log_entry.version:
            log_dict["version"] = log_entry.version
        
        return log_dict
    
    def _standard_record_to_dict(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Convert standard logging record to dictionary"""
        log_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "category": "system_health",  # Default category
        }
        
        # Add exception information if present
        if record.exc_info:
            log_dict["error_details"] = {
                "stack_trace": self.formatException(record.exc_info)
            }
        
        return log_dict


class StructuredFormatter(logging.Formatter):
    """
    Human-readable structured formatter for console output
    """
    
    def __init__(self, include_context: bool = True, include_metrics: bool = False):
        super().__init__()
        self.include_context = include_context
        self.include_metrics = include_metrics
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as structured text"""
        try:
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                return self._format_log_entry(log_entry)
            else:
                return self._format_standard_record(record)
                
        except Exception as e:
            # Fallback format
            return f"[{datetime.utcnow().isoformat()}] [ERROR] [Formatter] Formatting error: {str(e)} | Original: {record.getMessage()}"
    
    def _format_log_entry(self, log_entry: LogEntry) -> str:
        """Format a LogEntry as structured text"""
        # Basic log line
        timestamp = log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [
            f"[{timestamp}]",
            f"[{log_entry.level.value}]",
            f"[{log_entry.category.value}]",
            f"[{log_entry.component}]",
            log_entry.message
        ]
        
        base_line = " ".join(parts)
        
        # Add context information
        context_parts = []
        if self.include_context and log_entry.context:
            if log_entry.context.correlation_id:
                context_parts.append(f"correlation_id={log_entry.context.correlation_id}")
            if log_entry.context.session_id:
                context_parts.append(f"session_id={log_entry.context.session_id}")
            if log_entry.context.agent_id:
                context_parts.append(f"agent_id={log_entry.context.agent_id}")
            if log_entry.context.request_id:
                context_parts.append(f"request_id={log_entry.context.request_id}")
        
        if context_parts:
            base_line += f" | Context: {', '.join(context_parts)}"
        
        # Add performance metrics
        if self.include_metrics and log_entry.performance:
            metrics_parts = []
            if log_entry.performance.duration_ms:
                metrics_parts.append(f"duration={log_entry.performance.duration_ms:.2f}ms")
            if log_entry.performance.memory_usage_mb:
                metrics_parts.append(f"memory={log_entry.performance.memory_usage_mb:.2f}MB")
            if log_entry.performance.cpu_usage_percent:
                metrics_parts.append(f"cpu={log_entry.performance.cpu_usage_percent:.1f}%")
            
            if metrics_parts:
                base_line += f" | Performance: {', '.join(metrics_parts)}"
        
        # Add data if present
        if log_entry.data:
            try:
                data_str = json.dumps(log_entry.data, separators=(',', ':'), default=str)
                if len(data_str) > 200:
                    data_str = data_str[:200] + "..."
                base_line += f" | Data: {data_str}"
            except Exception:
                base_line += f" | Data: <serialization_error>"
        
        # Add error details
        if log_entry.error_details:
            error_parts = []
            if log_entry.error_details.error_type:
                error_parts.append(f"type={log_entry.error_details.error_type}")
            if log_entry.error_details.error_code:
                error_parts.append(f"code={log_entry.error_details.error_code}")
            if log_entry.error_details.severity:
                error_parts.append(f"severity={log_entry.error_details.severity}")
            
            if error_parts:
                base_line += f" | Error: {', '.join(error_parts)}"
            
            # Add stack trace on new line if present
            if log_entry.error_details.stack_trace:
                base_line += f"\nStack Trace:\n{log_entry.error_details.stack_trace}"
        
        return base_line
    
    def _format_standard_record(self, record: logging.LogRecord) -> str:
        """Format a standard logging record"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        parts = [
            f"[{timestamp}]",
            f"[{record.levelname}]",
            f"[system_health]",
            f"[{record.name}]",
            record.getMessage()
        ]
        
        base_line = " ".join(parts)
        
        # Add exception information if present
        if record.exc_info:
            base_line += f"\nException:\n{self.formatException(record.exc_info)}"
        
        return base_line


class CompactFormatter(logging.Formatter):
    """
    Compact formatter for high-volume logging scenarios
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record in compact format"""
        try:
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                timestamp = log_entry.timestamp.strftime("%H:%M:%S.%f")[:-3]
                
                # Compact format: TIME LEVEL COMPONENT MESSAGE
                parts = [
                    timestamp,
                    log_entry.level.value[0],  # First letter of level
                    log_entry.component[:10],  # Truncated component name
                    log_entry.message[:100]    # Truncated message
                ]
                
                line = " ".join(parts)
                
                # Add correlation ID if present
                if log_entry.context and log_entry.context.correlation_id:
                    line += f" [{log_entry.context.correlation_id[:8]}]"
                
                return line
            else:
                timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
                return f"{timestamp} {record.levelname[0]} {record.name[:10]} {record.getMessage()[:100]}"
                
        except Exception:
            return f"{datetime.utcnow().strftime('%H:%M:%S')} E Formatter <error>"


class MetricsFormatter(logging.Formatter):
    """
    Specialized formatter for metrics and performance data
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format metrics-focused log entries"""
        try:
            if hasattr(record, 'log_entry'):
                log_entry: LogEntry = record.log_entry
                
                # Focus on metrics data
                metrics_data = {}
                
                if log_entry.performance:
                    metrics_data.update(log_entry.performance.dict(exclude_none=True))
                
                if log_entry.agent_metrics:
                    metrics_data.update({f"agent_{k}": v for k, v in log_entry.agent_metrics.dict(exclude_none=True).items()})
                
                if log_entry.api_metrics:
                    metrics_data.update({f"api_{k}": v for k, v in log_entry.api_metrics.dict(exclude_none=True).items()})
                
                if log_entry.database_metrics:
                    metrics_data.update({f"db_{k}": v for k, v in log_entry.database_metrics.dict(exclude_none=True).items()})
                
                # Create metrics line
                timestamp = log_entry.timestamp.isoformat()
                component = log_entry.component
                
                metrics_str = " ".join([f"{k}={v}" for k, v in metrics_data.items()])
                
                return f"{timestamp} {component} {metrics_str}"
            else:
                return record.getMessage()
                
        except Exception:
            return f"{datetime.utcnow().isoformat()} metrics_formatter_error"
