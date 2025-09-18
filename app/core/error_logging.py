"""
Comprehensive Error Logging System

Provides detailed error tracking, categorization, and actionable debugging information
for the AI agent system with location tracking and context preservation.
"""

import asyncio
import json
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
import structlog
import uuid

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    AGENT_EXECUTION = "agent_execution"
    RAG_SYSTEM = "rag_system"
    INTEGRATION = "integration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error tracking."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorDetails:
    """Detailed error information."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    location: str
    stack_trace: Optional[str] = None
    context: Optional[ErrorContext] = None
    resolution_steps: List[str] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorLogger:
    """Comprehensive error logging system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.error_storage: List[ErrorDetails] = []
        self.max_stored_errors = 1000
        self.log_file_path = Path("logs/error_details.jsonl")
        self.log_file_path.parent.mkdir(exist_ok=True)
    
    async def log_error(
        self,
        error: Union[Exception, str],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        location: str = "unknown",
        context: Optional[ErrorContext] = None,
        resolution_steps: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an error with comprehensive details.
        
        Args:
            error: Exception or error message
            severity: Error severity level
            category: Error category
            location: Location where error occurred
            context: Additional context information
            resolution_steps: Suggested resolution steps
            metadata: Additional metadata
            
        Returns:
            Error ID for tracking
        """
        error_id = str(uuid.uuid4())
        
        # Extract error message and stack trace
        if isinstance(error, Exception):
            message = str(error)
            stack_trace = traceback.format_exc()
        else:
            message = str(error)
            stack_trace = None
        
        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            category=category,
            message=message,
            location=location,
            stack_trace=stack_trace,
            context=context,
            resolution_steps=resolution_steps or self._get_default_resolution_steps(category),
            metadata=metadata or {}
        )
        
        # Store error
        await self._store_error(error_details)
        
        # Log to structured logger
        await self._log_to_structlog(error_details)
        
        # Send alerts for critical errors
        if severity == ErrorSeverity.CRITICAL:
            await self._send_critical_alert(error_details)
        
        return error_id
    
    async def _store_error(self, error_details: ErrorDetails) -> None:
        """Store error details in memory and file."""
        # Add to memory storage
        self.error_storage.append(error_details)
        
        # Maintain max storage limit
        if len(self.error_storage) > self.max_stored_errors:
            self.error_storage = self.error_storage[-self.max_stored_errors:]
        
        # Write to file
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                error_dict = asdict(error_details)
                # Convert datetime to ISO string
                error_dict['timestamp'] = error_details.timestamp.isoformat()
                f.write(json.dumps(error_dict, default=str) + '\n')
        except Exception as e:
            logger.error("Failed to write error to file", error=str(e))
    
    async def _log_to_structlog(self, error_details: ErrorDetails) -> None:
        """Log error to structured logger."""
        log_data = {
            "error_id": error_details.error_id,
            "severity": error_details.severity.value,
            "category": error_details.category.value,
            "location": error_details.location,
            "message": error_details.message,
        }
        
        if error_details.context:
            log_data.update({
                "request_id": error_details.context.request_id,
                "user_id": error_details.context.user_id,
                "agent_id": error_details.context.agent_id,
                "endpoint": error_details.context.endpoint,
            })
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **log_data)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **log_data)
        else:
            logger.info("Low severity error occurred", **log_data)
    
    async def _send_critical_alert(self, error_details: ErrorDetails) -> None:
        """Send alert for critical errors."""
        # This would integrate with alerting systems like Slack, email, etc.
        logger.critical(
            "CRITICAL ERROR ALERT",
            error_id=error_details.error_id,
            message=error_details.message,
            location=error_details.location,
            resolution_steps=error_details.resolution_steps
        )
    
    def _get_default_resolution_steps(self, category: ErrorCategory) -> List[str]:
        """Get default resolution steps for error category."""
        resolution_map = {
            ErrorCategory.NETWORK: [
                "Check network connectivity",
                "Verify service endpoints are accessible",
                "Check proxy settings and authentication",
                "Review firewall and security group settings"
            ],
            ErrorCategory.DATABASE: [
                "Check database connection settings",
                "Verify database service is running",
                "Check database credentials and permissions",
                "Review connection pool settings"
            ],
            ErrorCategory.VALIDATION: [
                "Review input data format and types",
                "Check required fields are present",
                "Validate data against schema requirements",
                "Review API documentation for correct format"
            ],
            ErrorCategory.AUTHENTICATION: [
                "Verify authentication credentials",
                "Check token expiration and refresh",
                "Review authentication configuration",
                "Validate user permissions"
            ],
            ErrorCategory.AGENT_EXECUTION: [
                "Check agent configuration and tools",
                "Review agent memory and resource usage",
                "Verify agent dependencies are available",
                "Check agent execution logs for details"
            ],
            ErrorCategory.RAG_SYSTEM: [
                "Check vector database connectivity",
                "Verify embedding model availability",
                "Review document ingestion status",
                "Check search index integrity"
            ],
            ErrorCategory.INTEGRATION: [
                "Verify external service availability",
                "Check API keys and authentication",
                "Review integration configuration",
                "Test connectivity to external services"
            ],
            ErrorCategory.SYSTEM: [
                "Check system resource availability",
                "Review system logs for related errors",
                "Verify service dependencies",
                "Check system configuration"
            ]
        }
        
        return resolution_map.get(category, [
            "Review error details and stack trace",
            "Check system logs for related issues",
            "Verify configuration settings",
            "Contact system administrator if issue persists"
        ])
    
    async def get_errors(
        self,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        limit: int = 100
    ) -> List[ErrorDetails]:
        """Get stored errors with optional filtering."""
        errors = self.error_storage
        
        if severity:
            errors = [e for e in errors if e.severity == severity]
        
        if category:
            errors = [e for e in errors if e.category == category]
        
        # Sort by timestamp (newest first) and limit
        errors.sort(key=lambda x: x.timestamp, reverse=True)
        return errors[:limit]
    
    async def get_error_by_id(self, error_id: str) -> Optional[ErrorDetails]:
        """Get specific error by ID."""
        for error in self.error_storage:
            if error.error_id == error_id:
                return error
        return None
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = len(self.error_storage)
        
        if total_errors == 0:
            return {"total_errors": 0}
        
        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = len([
                e for e in self.error_storage if e.severity == severity
            ])
        
        # Count by category
        category_counts = {}
        for category in ErrorCategory:
            category_counts[category.value] = len([
                e for e in self.error_storage if e.category == category
            ])
        
        # Recent errors (last hour)
        recent_errors = len([
            e for e in self.error_storage 
            if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
        ])
        
        return {
            "total_errors": total_errors,
            "recent_errors_last_hour": recent_errors,
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "error_rate_per_hour": recent_errors
        }


# Global error logger instance
error_logger = ErrorLogger()


@asynccontextmanager
async def error_context(
    location: str,
    context: Optional[ErrorContext] = None,
    category: ErrorCategory = ErrorCategory.UNKNOWN
):
    """Context manager for automatic error logging."""
    try:
        yield
    except Exception as e:
        await error_logger.log_error(
            error=e,
            severity=ErrorSeverity.HIGH,
            category=category,
            location=location,
            context=context
        )
        raise


# Convenience functions
async def log_error(
    error: Union[Exception, str],
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    location: str = "unknown",
    **kwargs
) -> str:
    """Convenience function for logging errors."""
    return await error_logger.log_error(
        error=error,
        severity=severity,
        category=category,
        location=location,
        **kwargs
    )


async def log_critical_error(
    error: Union[Exception, str],
    location: str,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    **kwargs
) -> str:
    """Convenience function for logging critical errors."""
    return await error_logger.log_error(
        error=error,
        severity=ErrorSeverity.CRITICAL,
        category=category,
        location=location,
        **kwargs
    )
