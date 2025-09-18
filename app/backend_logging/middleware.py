"""
Logging Middleware for FastAPI

Provides middleware for automatic request/response logging, context management,
and performance monitoring.
"""

import time
import json
import asyncio
from typing import Callable, Optional, Dict, Any
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .backend_logger import get_logger
from .models import LogLevel, LogCategory, APIMetrics, PerformanceMetrics
from .context import CorrelationContext, RequestContext, SystemContext


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that automatically logs all HTTP requests and responses
    with performance metrics and context management.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list] = None,
        include_request_body: bool = False,
        include_response_body: bool = False,
        max_body_size: int = 1024,
        log_level: LogLevel = LogLevel.INFO
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
        self.max_body_size = max_body_size
        self.log_level = log_level
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process HTTP request and response with logging"""
        
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Generate correlation ID if not present
        correlation_id = request.headers.get("x-correlation-id") or str(uuid4())
        
        # Set up correlation context
        context = CorrelationContext.get_context()
        context.correlation_id = correlation_id
        context.request_id = str(uuid4())
        context.component = "API"
        context.operation = f"{request.method} {request.url.path}"
        
        CorrelationContext.set_context(context)
        CorrelationContext.start_request_timer()
        
        # Create request context
        request_context = RequestContext(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers)
        )
        
        # Log request
        await self._log_request(request, request_context, correlation_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Set response info
            request_context.set_response_info(
                status_code=response.status_code,
                response_size=self._get_response_size(response)
            )
            
            # Log successful response
            await self._log_response(request, response, request_context, duration_ms, correlation_id)
            
            # Add correlation ID to response headers
            response.headers["x-correlation-id"] = correlation_id
            
            return response
            
        except Exception as e:
            # Calculate duration for error case
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            await self._log_error(request, e, request_context, duration_ms, correlation_id)
            
            # Re-raise the exception
            raise
        
        finally:
            # Clear correlation context
            CorrelationContext.clear_context()
    
    async def _log_request(self, request: Request, request_context: RequestContext, correlation_id: str):
        """Log incoming HTTP request"""
        
        # Get request body if enabled
        request_body = None
        if self.include_request_body:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    request_body = body.decode('utf-8', errors='ignore')
                else:
                    request_body = f"<body_too_large:{len(body)}_bytes>"
                
                # Try to parse as JSON for better logging
                if request.headers.get("content-type", "").startswith("application/json"):
                    try:
                        request_body = json.loads(request_body)
                    except json.JSONDecodeError:
                        pass
                        
            except Exception:
                request_body = "<body_read_error>"
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Create log data
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }
        
        if request_body is not None:
            log_data["request_body"] = request_body
        
        # Set request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_context.set_request_size(int(content_length))
            except ValueError:
                pass
        
        # Log the request
        self.logger.log(
            level=self.log_level,
            message=f"HTTP Request: {request.method} {request.url.path}",
            category=LogCategory.API_LAYER,
            component="HTTPMiddleware",
            data=log_data
        )
    
    async def _log_response(self, request: Request, response: Response, 
                          request_context: RequestContext, duration_ms: float, correlation_id: str):
        """Log HTTP response"""
        
        # Get response body if enabled
        response_body = None
        if self.include_response_body and hasattr(response, 'body'):
            try:
                if isinstance(response, StreamingResponse):
                    response_body = "<streaming_response>"
                else:
                    body = getattr(response, 'body', b'')
                    if isinstance(body, bytes) and len(body) <= self.max_body_size:
                        response_body = body.decode('utf-8', errors='ignore')
                        
                        # Try to parse as JSON
                        if response.headers.get("content-type", "").startswith("application/json"):
                            try:
                                response_body = json.loads(response_body)
                            except json.JSONDecodeError:
                                pass
                    elif len(body) > self.max_body_size:
                        response_body = f"<body_too_large:{len(body)}_bytes>"
                        
            except Exception:
                response_body = "<body_read_error>"
        
        # Create API metrics
        api_metrics = APIMetrics(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            response_time_ms=duration_ms,
            request_size_bytes=request_context.request_size,
            response_size_bytes=request_context.response_size,
            authentication_method=self._get_auth_method(request)
        )
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            duration_ms=duration_ms,
            memory_usage_mb=SystemContext.get_memory_usage(),
            cpu_usage_percent=SystemContext.get_cpu_usage()
        )
        
        # Create log data
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "response_headers": dict(response.headers),
        }
        
        if response_body is not None:
            log_data["response_body"] = response_body
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = LogLevel.ERROR
        elif response.status_code >= 400:
            log_level = LogLevel.WARN
        else:
            log_level = self.log_level
        
        # Log the response
        self.logger.log(
            level=log_level,
            message=f"HTTP Response: {response.status_code} {request.method} {request.url.path} ({duration_ms:.2f}ms)",
            category=LogCategory.API_LAYER,
            component="HTTPMiddleware",
            data=log_data,
            api_metrics=api_metrics,
            performance=performance_metrics
        )
    
    async def _log_error(self, request: Request, error: Exception, 
                        request_context: RequestContext, duration_ms: float, correlation_id: str):
        """Log HTTP request error"""
        
        # Create API metrics for error case
        api_metrics = APIMetrics(
            method=request.method,
            endpoint=request.url.path,
            status_code=500,  # Assume 500 for unhandled exceptions
            response_time_ms=duration_ms,
            request_size_bytes=request_context.request_size
        )
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            duration_ms=duration_ms,
            memory_usage_mb=SystemContext.get_memory_usage(),
            cpu_usage_percent=SystemContext.get_cpu_usage()
        )
        
        # Create log data
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration_ms": duration_ms,
        }
        
        # Log the error
        self.logger.error(
            message=f"HTTP Error: {request.method} {request.url.path} - {type(error).__name__}: {str(error)}",
            category=LogCategory.API_LAYER,
            component="HTTPMiddleware",
            error=error,
            data=log_data,
            api_metrics=api_metrics,
            performance=performance_metrics
        )
    
    def _get_response_size(self, response: Response) -> int:
        """Get response body size"""
        try:
            if isinstance(response, StreamingResponse):
                return 0  # Can't determine size for streaming responses
            
            body = getattr(response, 'body', b'')
            if isinstance(body, bytes):
                return len(body)
            elif isinstance(body, str):
                return len(body.encode('utf-8'))
            else:
                return 0
        except Exception:
            return 0
    
    def _get_auth_method(self, request: Request) -> Optional[str]:
        """Determine authentication method used"""
        auth_header = request.headers.get("authorization", "")
        
        if auth_header.startswith("Bearer "):
            return "bearer_token"
        elif auth_header.startswith("Basic "):
            return "basic_auth"
        elif "api-key" in request.headers:
            return "api_key"
        elif "x-api-key" in request.headers:
            return "api_key"
        else:
            return None


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Specialized middleware for performance monitoring and alerting
    """
    
    def __init__(
        self,
        app: ASGIApp,
        slow_request_threshold_ms: float = 1000,
        memory_threshold_mb: float = 500,
        cpu_threshold_percent: float = 80
    ):
        super().__init__(app)
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance"""
        
        start_time = time.time()
        start_memory = SystemContext.get_memory_usage()
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            end_memory = SystemContext.get_memory_usage()
            memory_delta = end_memory - start_memory
            cpu_usage = SystemContext.get_cpu_usage()
            
            # Check for performance issues
            await self._check_performance_thresholds(
                request, duration_ms, end_memory, memory_delta, cpu_usage
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Log performance data even for errors
            self.logger.warn(
                message=f"Request failed with performance impact: {request.method} {request.url.path}",
                category=LogCategory.PERFORMANCE,
                component="PerformanceMonitor",
                data={
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "memory_usage_mb": SystemContext.get_memory_usage()
                }
            )
            
            raise
    
    async def _check_performance_thresholds(
        self, request: Request, duration_ms: float, memory_mb: float, 
        memory_delta_mb: float, cpu_percent: float
    ):
        """Check if performance thresholds are exceeded"""
        
        issues = []
        
        # Check slow request
        if duration_ms > self.slow_request_threshold_ms:
            issues.append(f"slow_request:{duration_ms:.2f}ms")
        
        # Check high memory usage
        if memory_mb > self.memory_threshold_mb:
            issues.append(f"high_memory:{memory_mb:.2f}MB")
        
        # Check high CPU usage
        if cpu_percent > self.cpu_threshold_percent:
            issues.append(f"high_cpu:{cpu_percent:.1f}%")
        
        # Check significant memory increase
        if memory_delta_mb > 50:  # More than 50MB increase
            issues.append(f"memory_spike:{memory_delta_mb:.2f}MB")
        
        if issues:
            self.logger.warn(
                message=f"Performance threshold exceeded: {request.method} {request.url.path} - {', '.join(issues)}",
                category=LogCategory.PERFORMANCE,
                component="PerformanceMonitor",
                data={
                    "duration_ms": duration_ms,
                    "memory_usage_mb": memory_mb,
                    "memory_delta_mb": memory_delta_mb,
                    "cpu_usage_percent": cpu_percent,
                    "issues": issues,
                    "url": str(request.url),
                    "method": request.method
                },
                performance=PerformanceMetrics(
                    duration_ms=duration_ms,
                    memory_usage_mb=memory_mb,
                    cpu_usage_percent=cpu_percent
                )
            )
