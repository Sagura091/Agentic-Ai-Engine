"""
Comprehensive exception handling system for the Agentic AI platform.

This module provides structured exception handling with proper error codes,
logging, and user-friendly error messages.
"""

import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes for the platform."""
    
    # System Errors (1000-1999)
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    REDIS_ERROR = "REDIS_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INITIALIZATION_ERROR = "INITIALIZATION_ERROR"
    
    # Validation Errors (2000-2999)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    VALUE_TOO_LARGE = "VALUE_TOO_LARGE"
    VALUE_TOO_SMALL = "VALUE_TOO_SMALL"
    
    # Authentication & Authorization (3000-3999)
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    
    # Agent Errors (4000-4999)
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_ALREADY_EXISTS = "AGENT_ALREADY_EXISTS"
    AGENT_CREATION_FAILED = "AGENT_CREATION_FAILED"
    AGENT_UPDATE_FAILED = "AGENT_UPDATE_FAILED"
    AGENT_DELETION_FAILED = "AGENT_DELETION_FAILED"
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_QUOTA_EXCEEDED = "AGENT_QUOTA_EXCEEDED"
    
    # Tool Errors (5000-5999)
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_ACCESS_DENIED = "TOOL_ACCESS_DENIED"
    TOOL_DEPENDENCY_MISSING = "TOOL_DEPENDENCY_MISSING"
    
    # RAG Errors (6000-6999)
    RAG_SYSTEM_ERROR = "RAG_SYSTEM_ERROR"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_PROCESSING_FAILED = "DOCUMENT_PROCESSING_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    VECTOR_SEARCH_FAILED = "VECTOR_SEARCH_FAILED"
    COLLECTION_NOT_FOUND = "COLLECTION_NOT_FOUND"
    
    # Workflow Errors (7000-7999)
    WORKFLOW_NOT_FOUND = "WORKFLOW_NOT_FOUND"
    WORKFLOW_EXECUTION_FAILED = "WORKFLOW_EXECUTION_FAILED"
    WORKFLOW_STEP_FAILED = "WORKFLOW_STEP_FAILED"
    WORKFLOW_TIMEOUT = "WORKFLOW_TIMEOUT"
    WORKFLOW_CYCLE_DETECTED = "WORKFLOW_CYCLE_DETECTED"
    
    # External Service Errors (8000-8999)
    LLM_SERVICE_ERROR = "LLM_SERVICE_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_QUOTA_EXCEEDED = "LLM_QUOTA_EXCEEDED"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"
    
    # Rate Limiting (9000-9999)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    CONCURRENT_REQUEST_LIMIT = "CONCURRENT_REQUEST_LIMIT"


class BaseAgenticException(Exception):
    """Base exception class for all Agentic AI platform exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.user_message = user_message or message
        self.request_id = request_id
        self.timestamp = datetime.utcnow()
        
        # Log the exception
        self._log_exception()
        
        super().__init__(self.message)
    
    def _log_exception(self):
        """Log the exception with appropriate level."""
        log_data = {
            "error_code": self.error_code.value,
            "status_code": self.status_code,
            "details": self.details,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.status_code >= 500:
            logger.error(
                f"System error: {self.message}",
                **log_data,
                exc_info=True
            )
        elif self.status_code >= 400:
            logger.warning(
                f"Client error: {self.message}",
                **log_data
            )
        else:
            logger.info(
                f"Application error: {self.message}",
                **log_data
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code.value,
            "message": self.user_message,
            "details": self.details,
            "status_code": self.status_code,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


# ============================================================================
# SYSTEM EXCEPTIONS
# ============================================================================

class SystemException(BaseAgenticException):
    """System-level exceptions."""
    pass


class DatabaseException(SystemException):
    """Database-related exceptions."""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=500,
            details=details,
            user_message="A database error occurred. Please try again later.",
            request_id=request_id
        )


class RedisException(SystemException):
    """Redis-related exceptions."""
    
    def __init__(
        self,
        message: str = "Redis operation failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.REDIS_ERROR,
            status_code=500,
            details=details,
            user_message="A caching error occurred. Please try again later.",
            request_id=request_id
        )


class ConfigurationException(SystemException):
    """Configuration-related exceptions."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            status_code=500,
            details=details,
            user_message="A configuration error occurred. Please contact support.",
            request_id=request_id
        )


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationException(BaseAgenticException):
    """Input validation exceptions."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        validation_details = details or {}
        if field:
            validation_details["field"] = field
        if value is not None:
            validation_details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=validation_details,
            user_message=f"Invalid input: {message}",
            request_id=request_id
        )


class InvalidInputException(ValidationException):
    """Invalid input exceptions."""
    
    def __init__(
        self,
        field: str,
        value: Any,
        expected_format: str,
        request_id: Optional[str] = None
    ):
        message = f"Invalid {field}: '{value}'. Expected format: {expected_format}"
        super().__init__(
            message=message,
            field=field,
            value=value,
            details={"expected_format": expected_format},
            request_id=request_id
        )


class MissingFieldException(ValidationException):
    """Missing required field exceptions."""
    
    def __init__(
        self,
        field: str,
        request_id: Optional[str] = None
    ):
        message = f"Required field '{field}' is missing"
        super().__init__(
            message=message,
            field=field,
            request_id=request_id
        )


# ============================================================================
# AUTHENTICATION & AUTHORIZATION EXCEPTIONS
# ============================================================================

class AuthenticationException(BaseAgenticException):
    """Authentication-related exceptions."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            status_code=401,
            details=details,
            user_message="Authentication failed. Please check your credentials.",
            request_id=request_id
        )


class AuthorizationException(BaseAgenticException):
    """Authorization-related exceptions."""
    
    def __init__(
        self,
        message: str = "Access denied",
        resource: Optional[str] = None,
        action: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        details = {}
        if resource:
            details["resource"] = resource
        if action:
            details["action"] = action
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_ERROR,
            status_code=403,
            details=details,
            user_message="Access denied. You don't have permission to perform this action.",
            request_id=request_id
        )


class TokenException(BaseAgenticException):
    """Token-related exceptions."""
    
    def __init__(
        self,
        message: str = "Invalid token",
        token_type: str = "access",
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_TOKEN,
            status_code=401,
            details={"token_type": token_type},
            user_message="Invalid or expired token. Please authenticate again.",
            request_id=request_id
        )


# ============================================================================
# AGENT EXCEPTIONS
# ============================================================================

class AgentException(BaseAgenticException):
    """Agent-related exceptions."""
    pass


class AgentNotFoundException(AgentException):
    """Agent not found exceptions."""
    
    def __init__(
        self,
        agent_id: str,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=f"Agent not found: {agent_id}",
            error_code=ErrorCode.AGENT_NOT_FOUND,
            status_code=404,
            details={"agent_id": agent_id},
            user_message="The requested agent was not found.",
            request_id=request_id
        )


class AgentCreationException(AgentException):
    """Agent creation exceptions."""
    
    def __init__(
        self,
        message: str = "Failed to create agent",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_CREATION_FAILED,
            status_code=400,
            details=details,
            user_message="Failed to create agent. Please check your input and try again.",
            request_id=request_id
        )


class AgentExecutionException(AgentException):
    """Agent execution exceptions."""
    
    def __init__(
        self,
        agent_id: str,
        message: str = "Agent execution failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        execution_details = details or {}
        execution_details["agent_id"] = agent_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_EXECUTION_FAILED,
            status_code=500,
            details=execution_details,
            user_message="Agent execution failed. Please try again or contact support.",
            request_id=request_id
        )


class AgentTimeoutException(AgentException):
    """Agent timeout exceptions."""
    
    def __init__(
        self,
        agent_id: str,
        timeout_seconds: int,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=f"Agent execution timed out after {timeout_seconds} seconds",
            error_code=ErrorCode.AGENT_TIMEOUT,
            status_code=408,
            details={
                "agent_id": agent_id,
                "timeout_seconds": timeout_seconds
            },
            user_message="Agent execution timed out. Please try again with a simpler request.",
            request_id=request_id
        )


# Alias for backward compatibility
AgentExecutionError = AgentExecutionException
AgentTimeoutError = AgentTimeoutException


# ============================================================================
# TOOL EXCEPTIONS
# ============================================================================

class ToolException(BaseAgenticException):
    """Tool-related exceptions."""
    pass


class ToolNotFoundException(ToolException):
    """Tool not found exceptions."""
    
    def __init__(
        self,
        tool_id: str,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=f"Tool not found: {tool_id}",
            error_code=ErrorCode.TOOL_NOT_FOUND,
            status_code=404,
            details={"tool_id": tool_id},
            user_message="The requested tool was not found.",
            request_id=request_id
        )


class ToolExecutionException(ToolException):
    """Tool execution exceptions."""
    
    def __init__(
        self,
        tool_id: str,
        message: str = "Tool execution failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        execution_details = details or {}
        execution_details["tool_id"] = tool_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            status_code=500,
            details=execution_details,
            user_message="Tool execution failed. Please try again or contact support.",
            request_id=request_id
        )


class ToolAccessDeniedException(ToolException):
    """Tool access denied exceptions."""
    
    def __init__(
        self,
        tool_id: str,
        agent_id: str,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=f"Access denied to tool {tool_id} for agent {agent_id}",
            error_code=ErrorCode.TOOL_ACCESS_DENIED,
            status_code=403,
            details={"tool_id": tool_id, "agent_id": agent_id},
            user_message="Access denied to the requested tool.",
            request_id=request_id
        )


# ============================================================================
# RAG EXCEPTIONS
# ============================================================================

class RAGException(BaseAgenticException):
    """RAG-related exceptions."""
    pass


class DocumentProcessingException(RAGException):
    """Document processing exceptions."""
    
    def __init__(
        self,
        document_id: str,
        message: str = "Document processing failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        processing_details = details or {}
        processing_details["document_id"] = document_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
            status_code=500,
            details=processing_details,
            user_message="Document processing failed. Please try again or contact support.",
            request_id=request_id
        )


class EmbeddingException(RAGException):
    """Embedding generation exceptions."""
    
    def __init__(
        self,
        message: str = "Embedding generation failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.EMBEDDING_FAILED,
            status_code=500,
            details=details,
            user_message="Failed to generate embeddings. Please try again later.",
            request_id=request_id
        )


class VectorSearchException(RAGException):
    """Vector search exceptions."""
    
    def __init__(
        self,
        query: str,
        message: str = "Vector search failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        search_details = details or {}
        search_details["query"] = query
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VECTOR_SEARCH_FAILED,
            status_code=500,
            details=search_details,
            user_message="Search failed. Please try again with a different query.",
            request_id=request_id
        )


# ============================================================================
# WORKFLOW EXCEPTIONS
# ============================================================================

class WorkflowException(BaseAgenticException):
    """Workflow-related exceptions."""
    pass


class WorkflowNotFoundException(WorkflowException):
    """Workflow not found exceptions."""
    
    def __init__(
        self,
        workflow_id: str,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=f"Workflow not found: {workflow_id}",
            error_code=ErrorCode.WORKFLOW_NOT_FOUND,
            status_code=404,
            details={"workflow_id": workflow_id},
            user_message="The requested workflow was not found.",
            request_id=request_id
        )


class WorkflowExecutionException(WorkflowException):
    """Workflow execution exceptions."""
    
    def __init__(
        self,
        workflow_id: str,
        step_id: Optional[str] = None,
        message: str = "Workflow execution failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        execution_details = details or {}
        execution_details["workflow_id"] = workflow_id
        if step_id:
            execution_details["step_id"] = step_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.WORKFLOW_EXECUTION_FAILED,
            status_code=500,
            details=execution_details,
            user_message="Workflow execution failed. Please try again or contact support.",
            request_id=request_id
        )


# ============================================================================
# EXTERNAL SERVICE EXCEPTIONS
# ============================================================================

class LLMException(BaseAgenticException):
    """LLM service exceptions."""
    pass


class LLMServiceException(LLMException):
    """LLM service errors."""
    
    def __init__(
        self,
        provider: str,
        message: str = "LLM service error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        service_details = details or {}
        service_details["provider"] = provider
        
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            status_code=502,
            details=service_details,
            user_message="Language model service is temporarily unavailable. Please try again later.",
            request_id=request_id
        )


class LLMTimeoutException(LLMException):
    """LLM timeout exceptions."""
    
    def __init__(
        self,
        provider: str,
        timeout_seconds: int,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=f"LLM request timed out after {timeout_seconds} seconds",
            error_code=ErrorCode.LLM_TIMEOUT,
            status_code=408,
            details={
                "provider": provider,
                "timeout_seconds": timeout_seconds
            },
            user_message="Language model request timed out. Please try again with a shorter request.",
            request_id=request_id
        )


class ExternalAPIException(BaseAgenticException):
    """External API exceptions."""
    
    def __init__(
        self,
        service: str,
        message: str = "External API error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        api_details = details or {}
        api_details["service"] = service
        
        super().__init__(
            message=message,
            error_code=ErrorCode.EXTERNAL_API_ERROR,
            status_code=502,
            details=api_details,
            user_message="External service is temporarily unavailable. Please try again later.",
            request_id=request_id
        )


# ============================================================================
# RATE LIMITING EXCEPTIONS
# ============================================================================

class RateLimitException(BaseAgenticException):
    """Rate limiting exceptions."""

    def __init__(
        self,
        limit_type: str,
        limit_value: int,
        window_seconds: int,
        request_id: Optional[str] = None
    ):
        super().__init__(
            message=f"Rate limit exceeded: {limit_type}",
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=429,
            details={
                "limit_type": limit_type,
                "limit_value": limit_value,
                "window_seconds": window_seconds
            },
            user_message=f"Rate limit exceeded. Please wait {window_seconds} seconds before trying again.",
            request_id=request_id
        )


# ============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# ============================================================================

# Alias for ExternalAPIException (used in health.py)
ExternalServiceError = ExternalAPIException


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

def handle_exception(exc: Exception, request_id: Optional[str] = None) -> BaseAgenticException:
    """Convert any exception to a BaseAgenticException."""
    if isinstance(exc, BaseAgenticException):
        return exc
    
    # Log the unexpected exception
    logger.error(
        f"Unexpected exception: {str(exc)}",
        request_id=request_id,
        exc_info=True
    )
    
    # Convert to system exception
    return SystemException(
        message=f"Unexpected error: {str(exc)}",
        details={"original_exception": type(exc).__name__},
        request_id=request_id
    )


def get_error_response(exc: BaseAgenticException) -> Dict[str, Any]:
    """Get standardized error response from exception."""
    return {
        "success": False,
        "error": exc.error_code.value,
        "message": exc.user_message,
        "details": exc.details,
        "status_code": exc.status_code,
        "request_id": exc.request_id,
        "timestamp": exc.timestamp.isoformat()
    }


# Export all exceptions
__all__ = [
    "BaseAgenticException", "SystemException", "DatabaseException", "RedisException",
    "ConfigurationException", "ValidationException", "InvalidInputException",
    "MissingFieldException", "AuthenticationException", "AuthorizationException",
    "TokenException", "AgentException", "AgentNotFoundException", "AgentCreationException",
    "AgentExecutionException", "AgentTimeoutException", "ToolException",
    "ToolNotFoundException", "ToolExecutionException", "ToolAccessDeniedException",
    "RAGException", "DocumentProcessingException", "EmbeddingException",
    "VectorSearchException", "WorkflowException", "WorkflowNotFoundException",
    "WorkflowExecutionException", "LLMException", "LLMServiceException",
    "LLMTimeoutException", "ExternalAPIException", "ExternalServiceError", "RateLimitException",
    "handle_exception", "get_error_response"
]