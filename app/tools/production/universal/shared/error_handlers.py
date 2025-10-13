"""
Error Handlers for Revolutionary Universal Tools

Comprehensive error handling system for all Universal Tools.
NO SHORTCUTS - Complete error coverage.
"""

from typing import Optional, Any, Dict
from enum import Enum

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory, LogLevel

logger = get_logger()


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    FILE_OPERATION = "file_operation"
    VALIDATION = "validation"
    CONVERSION = "conversion"
    PERMISSION = "permission"
    NETWORK = "network"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    RUNTIME = "runtime"
    DATA = "data"
    SECURITY = "security"


class UniversalToolError(Exception):
    """
    Base exception for all Universal Tool errors.
    
    Provides comprehensive error information including:
    - Error message
    - Error category
    - Severity level
    - Context information
    - Recovery suggestions
    - Original exception (if any)
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.RUNTIME,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize Universal Tool Error.
        
        Args:
            message: Human-readable error message
            category: Error category for classification
            severity: Error severity level
            context: Additional context information
            recovery_suggestion: Suggestion for how to recover from error
            original_exception: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.original_exception = original_exception
        
        # Log the error
        self._log_error()
    
    def _log_error(self) -> None:
        """Log the error with full context."""
        # Map severity to log level
        log_level_map = {
            ErrorSeverity.LOW: LogLevel.DEBUG,
            ErrorSeverity.MEDIUM: LogLevel.WARNING,
            ErrorSeverity.HIGH: LogLevel.ERROR,
            ErrorSeverity.CRITICAL: LogLevel.CRITICAL,
        }

        log_level = log_level_map.get(self.severity, LogLevel.ERROR)

        # Prepare log data
        log_data = {
            "error_message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "recovery_suggestion": self.recovery_suggestion,
        }

        # Log based on severity
        if self.severity == ErrorSeverity.LOW:
            logger.debug(
                "Universal Tool Error",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.error_handlers",
                data=log_data,
                error=self.original_exception
            )
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(
                "Universal Tool Error",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.error_handlers",
                data=log_data,
                error=self.original_exception
            )
        else:  # HIGH or CRITICAL
            logger.error(
                "Universal Tool Error",
                LogCategory.TOOL_OPERATIONS,
                "app.tools.production.universal.shared.error_handlers",
                data=log_data,
                error=self.original_exception
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "recovery_suggestion": self.recovery_suggestion,
            "original_exception": str(self.original_exception) if self.original_exception else None,
        }
    
    def __str__(self) -> str:
        """String representation of error."""
        parts = [f"{self.__class__.__name__}: {self.message}"]
        
        if self.category:
            parts.append(f"Category: {self.category.value}")
        
        if self.severity:
            parts.append(f"Severity: {self.severity.value}")
        
        if self.recovery_suggestion:
            parts.append(f"Recovery: {self.recovery_suggestion}")
        
        if self.context:
            parts.append(f"Context: {self.context}")
        
        return " | ".join(parts)


class FileOperationError(UniversalToolError):
    """Error during file operations (read, write, delete, etc.)."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if file_path:
            context["file_path"] = file_path
        if operation:
            context["operation"] = operation
        
        super().__init__(
            message=message,
            category=ErrorCategory.FILE_OPERATION,
            context=context,
            **kwargs
        )


class ValidationError(UniversalToolError):
    """Error during input validation."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if field_name:
            context["field_name"] = field_name
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
        if expected_type:
            context["expected_type"] = expected_type
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            context=context,
            **kwargs
        )


class ConversionError(UniversalToolError):
    """Error during format conversion."""
    
    def __init__(
        self,
        message: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if source_format:
            context["source_format"] = source_format
        if target_format:
            context["target_format"] = target_format
        
        super().__init__(
            message=message,
            category=ErrorCategory.CONVERSION,
            context=context,
            **kwargs
        )


class PermissionError(UniversalToolError):
    """Error due to insufficient permissions."""
    
    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if required_permission:
            context["required_permission"] = required_permission
        if resource:
            context["resource"] = resource
        
        super().__init__(
            message=message,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class NetworkError(UniversalToolError):
    """Error during network operations."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if url:
            context["url"] = url
        if status_code:
            context["status_code"] = status_code
        
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            context=context,
            **kwargs
        )


class DependencyError(UniversalToolError):
    """Error due to missing or incompatible dependencies."""
    
    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if dependency_name:
            context["dependency_name"] = dependency_name
        if required_version:
            context["required_version"] = required_version
        
        super().__init__(
            message=message,
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            **kwargs
        )


class SecurityError(UniversalToolError):
    """Error due to security violations."""
    
    def __init__(
        self,
        message: str,
        violation_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop("context", {})
        if violation_type:
            context["violation_type"] = violation_type
        
        super().__init__(
            message=message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            **kwargs
        )

