"""
Custom exceptions for the Agentic AI Microservice.

This module defines custom exception classes for better error handling
and more informative error messages throughout the application.
"""

from typing import Any, Dict, Optional

# Import new backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory
from app.backend_logging.context import CorrelationContext

backend_logger = get_logger()


class AgentException(Exception):
    """
    Base exception class for all agent-related errors.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        status_code: HTTP status code
        detail: Additional error details
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "agent_error",
        status_code: int = 500,
        detail: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.detail = detail or {}

        # Log the exception creation
        backend_logger.error(
            f"AgentException created: {message}",
            LogCategory.ERROR_TRACKING,
            "AgentException",
            data={
                "error_code": error_code,
                "status_code": status_code,
                "detail": self.detail,
                "exception_type": self.__class__.__name__
            }
        )

        super().__init__(self.message)


class AgentNotFoundError(AgentException):
    """Raised when a requested agent is not found."""
    
    def __init__(self, agent_id: str):
        super().__init__(
            message=f"Agent with ID '{agent_id}' not found",
            error_code="agent_not_found",
            status_code=404,
            detail={"agent_id": agent_id},
        )


class AgentCreationError(AgentException):
    """Raised when agent creation fails."""
    
    def __init__(self, reason: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Failed to create agent: {reason}",
            error_code="agent_creation_failed",
            status_code=400,
            detail=detail or {},
        )


class AgentExecutionError(AgentException):
    """Raised when agent execution fails."""
    
    def __init__(self, agent_id: str, reason: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Agent '{agent_id}' execution failed: {reason}",
            error_code="agent_execution_failed",
            status_code=500,
            detail={"agent_id": agent_id, **(detail or {})},
        )


class AgentTimeoutError(AgentException):
    """Raised when agent execution times out."""
    
    def __init__(self, agent_id: str, timeout_seconds: int):
        super().__init__(
            message=f"Agent '{agent_id}' execution timed out after {timeout_seconds} seconds",
            error_code="agent_timeout",
            status_code=408,
            detail={"agent_id": agent_id, "timeout_seconds": timeout_seconds},
        )


class WorkflowError(AgentException):
    """Raised when workflow execution fails."""
    
    def __init__(self, workflow_id: str, reason: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Workflow '{workflow_id}' failed: {reason}",
            error_code="workflow_failed",
            status_code=500,
            detail={"workflow_id": workflow_id, **(detail or {})},
        )


class WorkflowNotFoundError(AgentException):
    """Raised when a requested workflow is not found."""
    
    def __init__(self, workflow_id: str):
        super().__init__(
            message=f"Workflow with ID '{workflow_id}' not found",
            error_code="workflow_not_found",
            status_code=404,
            detail={"workflow_id": workflow_id},
        )


class LLMIntegrationError(AgentException):
    """Raised when LLM integration fails."""
    
    def __init__(self, provider: str, reason: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"LLM integration with '{provider}' failed: {reason}",
            error_code="llm_integration_failed",
            status_code=502,
            detail={"provider": provider, **(detail or {})},
        )


class StateManagementError(AgentException):
    """Raised when state management operations fail."""
    
    def __init__(self, operation: str, reason: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"State management operation '{operation}' failed: {reason}",
            error_code="state_management_failed",
            status_code=500,
            detail={"operation": operation, **(detail or {})},
        )


class ConfigurationError(AgentException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, setting: str, reason: str):
        super().__init__(
            message=f"Configuration error for '{setting}': {reason}",
            error_code="configuration_error",
            status_code=500,
            detail={"setting": setting},
        )


class ValidationError(AgentException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, reason: str, value: Any = None):
        super().__init__(
            message=f"Validation error for field '{field}': {reason}",
            error_code="validation_error",
            status_code=422,
            detail={"field": field, "value": value},
        )


class ResourceLimitError(AgentException):
    """Raised when resource limits are exceeded."""
    
    def __init__(self, resource: str, limit: Any, current: Any):
        super().__init__(
            message=f"Resource limit exceeded for '{resource}': {current} > {limit}",
            error_code="resource_limit_exceeded",
            status_code=429,
            detail={"resource": resource, "limit": limit, "current": current},
        )


class AuthenticationError(AgentException):
    """Raised when authentication fails."""
    
    def __init__(self, reason: str = "Invalid credentials"):
        super().__init__(
            message=f"Authentication failed: {reason}",
            error_code="authentication_failed",
            status_code=401,
        )


class AuthorizationError(AgentException):
    """Raised when authorization fails."""
    
    def __init__(self, resource: str, action: str):
        super().__init__(
            message=f"Access denied: insufficient permissions for '{action}' on '{resource}'",
            error_code="authorization_failed",
            status_code=403,
            detail={"resource": resource, "action": action},
        )


class ExternalServiceError(AgentException):
    """Raised when external service integration fails."""
    
    def __init__(self, service: str, reason: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"External service '{service}' error: {reason}",
            error_code="external_service_error",
            status_code=502,
            detail={"service": service, **(detail or {})},
        )
