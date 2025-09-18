"""
Custom middleware for the Agentic AI Microservice.

This module provides custom middleware for logging, metrics, security,
and other cross-cutting concerns.
"""

import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

AGENT_OPERATIONS = Counter(
    "agent_operations_total",
    "Total agent operations",
    ["operation", "status"]
)

ACTIVE_AGENTS = Counter(
    "active_agents_total",
    "Number of active agents"
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured logging of HTTP requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process HTTP request with structured logging.
        
        Args:
            request: HTTP request
            call_next: Next middleware or endpoint
            
        Returns:
            HTTP response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            "HTTP request started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                "HTTP request completed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_seconds=duration,
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "HTTP request failed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                error=str(e),
                error_type=type(e).__name__,
                duration_seconds=duration,
            )
            
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting Prometheus metrics.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process HTTP request with metrics collection.
        
        Args:
            request: HTTP request
            call_next: Next middleware or endpoint
            
        Returns:
            HTTP response
        """
        # Start timing
        start_time = time.time()
        
        # Get endpoint path template
        endpoint = self._get_endpoint_path(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Record error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            raise
    
    def _get_endpoint_path(self, request: Request) -> str:
        """
        Get the endpoint path template for metrics.
        
        Args:
            request: HTTP request
            
        Returns:
            Endpoint path template
        """
        # Try to get route pattern from FastAPI
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope["route"]
            if hasattr(route, "path"):
                return route.path
        
        # Fallback to actual path
        return request.url.path


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for security headers and basic security measures.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process HTTP request with security enhancements.
        
        Args:
            request: HTTP request
            call_next: Next middleware or endpoint
            
        Returns:
            HTTP response with security headers
        """
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' localhost:*; "
            "script-src 'self' 'unsafe-inline' localhost:*; "
            "style-src 'self' 'unsafe-inline' localhost:*; "
            "img-src 'self' data: https: localhost:*; "
            "font-src 'self' data: localhost:*; "
            "connect-src 'self' ws: wss: localhost:* ws://localhost:* wss://localhost:*; "
            "frame-ancestors 'none';"
        )
        
        # Remove server header
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Basic rate limiting middleware.
    
    Note: For production use, consider using Redis-based rate limiting
    or a dedicated rate limiting service.
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: ASGI application
            calls: Number of calls allowed per period
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process HTTP request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware or endpoint
            
        Returns:
            HTTP response or rate limit error
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        current_time = time.time()
        
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        
        # Clean old requests
        self.clients[client_ip] = [
            req_time for req_time in self.clients[client_ip]
            if current_time - req_time < self.period
        ]
        
        # Check if rate limit exceeded
        if len(self.clients[client_ip]) >= self.calls:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                calls=len(self.clients[client_ip]),
                limit=self.calls,
                period=self.period,
            )
            
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Retry-After": str(self.period),
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + self.period)),
                }
            )
        
        # Add current request
        self.clients[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.calls - len(self.clients[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.period))
        
        return response
