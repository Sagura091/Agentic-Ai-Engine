"""
Revolutionary Unified API Response System.

This module provides standardized response formats for all API endpoints,
ensuring consistency, performance tracking, and comprehensive error handling.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

_backend_logger = get_backend_logger()


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL_ERROR = "internal_error"
    AGENT_ERROR = "agent_error"
    RAG_ERROR = "rag_error"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PaginationInfo(BaseModel):
    """Pagination information for list responses."""
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Items per page")
    total: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class ResponsePerformance(BaseModel):
    """Performance metrics for API responses."""
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    cache_hit: bool = Field(default=False, description="Whether response was served from cache")
    database_queries: int = Field(default=0, description="Number of database queries executed")
    ai_operations: int = Field(default=0, description="Number of AI operations performed")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")


class ErrorDetails(BaseModel):
    """Detailed error information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    suggestions: List[str] = Field(default_factory=list, description="Suggested solutions")
    retry_after: Optional[int] = Field(default=None, description="Retry after seconds")
    documentation_url: Optional[str] = Field(default=None, description="Link to documentation")


class StandardAPIResponse(BaseModel):
    """Revolutionary unified API response format."""
    success: bool = Field(default=True, description="Whether the request was successful")
    data: Any = Field(..., description="Response data")
    message: str = Field(default="Success", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request ID")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    pagination: Optional[PaginationInfo] = Field(default=None, description="Pagination information")
    performance: Optional[ResponsePerformance] = Field(default=None, description="Performance metrics")


class StandardErrorResponse(BaseModel):
    """Revolutionary unified error response format."""
    success: bool = Field(default=False, description="Always false for errors")
    error: ErrorDetails = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: str = Field(..., description="Unique request ID")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for debugging")


class APIResponseWrapper:
    """Revolutionary API response wrapper with performance tracking."""
    
    @staticmethod
    def success(
        data: Any,
        message: str = "Success",
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pagination: Optional[PaginationInfo] = None,
        performance: Optional[ResponsePerformance] = None
    ) -> StandardAPIResponse:
        """Wrap successful responses with standardized format."""
        try:
            return StandardAPIResponse(
                success=True,
                data=data,
                message=message,
                request_id=request_id or str(uuid.uuid4()),
                metadata=metadata,
                pagination=pagination,
                performance=performance
            )
        except Exception as e:
            _backend_logger.error(
                "Failed to create success response",
                LogCategory.API_OPERATIONS,
                "app.api.v1.responses",
                data={"error": str(e)}
            )
            # Fallback to basic response
            return StandardAPIResponse(
                success=True,
                data=data,
                message=message,
                request_id=request_id or str(uuid.uuid4())
            )
    
    @staticmethod
    def error(
        error_code: str,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        retry_after: Optional[int] = None,
        documentation_url: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> StandardErrorResponse:
        """Wrap error responses with standardized format."""
        try:
            error_details = ErrorDetails(
                code=error_code,
                message=message,
                category=category,
                severity=severity,
                details=details or {},
                suggestions=suggestions or [],
                retry_after=retry_after,
                documentation_url=documentation_url
            )
            
            return StandardErrorResponse(
                error=error_details,
                request_id=request_id or str(uuid.uuid4()),
                trace_id=trace_id
            )
        except Exception as e:
            _backend_logger.error(
                "Failed to create error response",
                LogCategory.API_OPERATIONS,
                "app.api.v1.responses",
                data={"error": str(e)}
            )
            # Fallback to basic error response
            return StandardErrorResponse(
                error=ErrorDetails(
                    code="RESPONSE_CREATION_ERROR",
                    message="Failed to create error response",
                    category=ErrorCategory.INTERNAL_ERROR,
                    severity=ErrorSeverity.HIGH
                ),
                request_id=request_id or str(uuid.uuid4())
            )
    
    @staticmethod
    def from_exception(
        exception: Exception,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> StandardErrorResponse:
        """Create error response from exception."""
        try:
            # Map common exceptions to error categories
            if isinstance(exception, ValueError):
                category = ErrorCategory.VALIDATION
                severity = ErrorSeverity.LOW
            elif isinstance(exception, PermissionError):
                category = ErrorCategory.AUTHORIZATION
                severity = ErrorSeverity.MEDIUM
            elif isinstance(exception, FileNotFoundError):
                category = ErrorCategory.NOT_FOUND
                severity = ErrorSeverity.LOW
            elif isinstance(exception, ConnectionError):
                category = ErrorCategory.EXTERNAL_SERVICE
                severity = ErrorSeverity.HIGH
            else:
                category = ErrorCategory.INTERNAL_ERROR
                severity = ErrorSeverity.HIGH
            
            return APIResponseWrapper.error(
                error_code=type(exception).__name__.upper(),
                message=str(exception),
                category=category,
                severity=severity,
                request_id=request_id,
                trace_id=trace_id,
                details={"exception_type": type(exception).__name__}
            )
        except Exception as e:
            _backend_logger.error(
                "Failed to create exception response",
                LogCategory.API_OPERATIONS,
                "app.api.v1.responses",
                data={"error": str(e)}
            )
            # Ultimate fallback
            return StandardErrorResponse(
                error=ErrorDetails(
                    code="UNKNOWN_ERROR",
                    message="An unknown error occurred",
                    category=ErrorCategory.INTERNAL_ERROR,
                    severity=ErrorSeverity.CRITICAL
                ),
                request_id=request_id or str(uuid.uuid4())
            )


# Response type aliases for convenience
SuccessResponse = StandardAPIResponse
ErrorResponse = StandardErrorResponse
