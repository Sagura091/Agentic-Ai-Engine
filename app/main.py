"""
Main FastAPI application entry point for the Agentic AI Microservice.

This module sets up the FastAPI application with all necessary middleware,
routers, and integrations for the agentic AI system.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.config.settings import get_settings
from app.core.exceptions import AgentException
from app.core.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    SecurityMiddleware,
)
from app.api.v1.router import api_router
from app.api.websocket.manager import websocket_manager
from app.orchestration.orchestrator import orchestrator
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator
from app.core.seamless_integration import seamless_integration
from app.services.monitoring_service import monitoring_service

# Import backend logging system
from app.backend_logging.backend_logger import get_logger, configure_logger
from app.backend_logging.middleware import LoggingMiddleware as BackendLoggingMiddleware, PerformanceMonitoringMiddleware
from app.backend_logging.models import LogConfiguration, LogLevel, LogCategory


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    settings = get_settings()

    # Initialize backend logging system first
    backend_logger = configure_logger(LogConfiguration(
        log_level=LogLevel.INFO if settings.ENVIRONMENT == "production" else LogLevel.DEBUG,
        enable_console_output=True,
        enable_file_output=True,
        enable_json_format=True,
        enable_async_logging=True,
        enable_performance_logging=True,
        enable_agent_metrics=True,
        enable_api_metrics=True,
        enable_database_metrics=True,
        log_retention_days=30,
        max_log_file_size_mb=100,
        max_log_files=10
    ))

    # Startup
    logger.info("Starting Revolutionary Agentic AI Microservice", version=app.version)
    backend_logger.info(
        "Backend logging system initialized and FastAPI application starting",
        category=LogCategory.SYSTEM_HEALTH,
        component="FastAPI",
        data={"version": app.version, "environment": settings.ENVIRONMENT}
    )

    try:
        # Initialize seamless integration system (replaces orchestrator)
        backend_logger.info(
            "Initializing seamless integration system",
            category=LogCategory.ORCHESTRATION,
            component="SeamlessIntegration"
        )
        await seamless_integration.initialize_complete_system()

        # Initialize supporting services
        backend_logger.info(
            "Initializing supporting services",
            category=LogCategory.SYSTEM_HEALTH,
            component="ServiceInitialization"
        )
        await websocket_manager.initialize()
        await monitoring_service.initialize()

        logger.info("All services initialized successfully with seamless integration")
        logger.info("System capabilities: Unlimited agents, Dynamic tools, True agentic AI")

        backend_logger.info(
            "All services initialized successfully - System ready",
            category=LogCategory.SYSTEM_HEALTH,
            component="FastAPI",
            data={
                "capabilities": ["unlimited_agents", "dynamic_tools", "true_agentic_ai"],
                "services": ["seamless_integration", "websocket_manager", "monitoring_service"]
            }
        )

        yield

    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        backend_logger.fatal(
            "Failed to initialize services - System startup failed",
            category=LogCategory.SYSTEM_HEALTH,
            component="FastAPI",
            error=e,
            data={"startup_phase": "service_initialization"}
        )
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Agentic AI Microservice")
        backend_logger.info(
            "Shutting down Agentic AI Microservice",
            category=LogCategory.SYSTEM_HEALTH,
            component="FastAPI"
        )

        try:
            await monitoring_service.shutdown()
            await websocket_manager.shutdown()
            await orchestrator.shutdown()

            logger.info("All services shut down successfully")
            backend_logger.info(
                "All services shut down successfully",
                category=LogCategory.SYSTEM_HEALTH,
                component="FastAPI"
            )

            # Shutdown backend logging system last
            backend_logger.shutdown()

        except Exception as e:
            logger.error("Error during shutdown", error=str(e))
            backend_logger.error(
                "Error during shutdown",
                category=LogCategory.SYSTEM_HEALTH,
                component="FastAPI",
                error=e
            )


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Agentic AI Microservice",
        description="Revolutionary agentic AI microservice with LangChain/LangGraph integration and OpenWebUI compatibility",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add routers
    setup_routers(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI, settings) -> None:
    """
    Configure application middleware.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Backend logging middleware (add first for comprehensive logging)
    app.add_middleware(
        BackendLoggingMiddleware,
        exclude_paths=["/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico"],
        include_request_body=settings.ENVIRONMENT != "production",
        include_response_body=settings.ENVIRONMENT != "production",
        max_body_size=1024,
        log_level=LogLevel.INFO if settings.ENVIRONMENT == "production" else LogLevel.DEBUG
    )

    # Performance monitoring middleware
    app.add_middleware(
        PerformanceMonitoringMiddleware,
        slow_request_threshold_ms=1000,
        memory_threshold_mb=500,
        cpu_threshold_percent=80
    )

    # Custom middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)


def setup_routers(app: FastAPI) -> None:
    """
    Configure application routers.

    Args:
        app: FastAPI application instance
    """
    # API routes
    app.include_router(api_router, prefix="/api/v1")

    # Socket.IO support for frontend compatibility
    # Note: Socket.IO is handled separately in the ASGI app wrapper

    # WebSocket endpoint for real-time communication (native WebSocket)
    @app.websocket("/ws")
    async def websocket_endpoint(websocket):
        """WebSocket endpoint for real-time agent communication."""
        from app.api.websocket.handlers import handle_websocket_connection
        await handle_websocket_connection(websocket)

    # Collaboration WebSocket endpoint
    @app.websocket("/collaboration/{workspace_id}")
    async def collaboration_websocket(websocket, workspace_id: str):
        """WebSocket endpoint for real-time collaboration."""
        from app.api.websocket.handlers import handle_collaboration_connection
        await handle_collaboration_connection(websocket, workspace_id)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for container orchestration."""
        from datetime import datetime
        return {
            "status": "healthy",
            "service": "agentic-ai-microservice",
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
        }

    # Metrics endpoint for Prometheus
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Configure global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    @app.exception_handler(AgentException)
    async def agent_exception_handler(request: Request, exc: AgentException):
        """Handle custom agent exceptions."""
        # Log with both structured and backend logging
        logger.error(
            "Agent exception occurred",
            error=str(exc),
            error_code=exc.error_code,
            path=request.url.path,
        )

        # Backend logging with detailed context
        backend_logger = get_logger()
        backend_logger.error(
            f"Agent exception: {exc.error_code} - {str(exc)}",
            category=LogCategory.AGENT_OPERATIONS,
            component="ExceptionHandler",
            error=exc,
            data={
                "error_code": exc.error_code,
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else "unknown"
            }
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": str(exc),
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        # Log with both structured and backend logging
        logger.error(
            "Unhandled exception occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
        )

        # Backend logging with detailed context
        backend_logger = get_logger()
        backend_logger.fatal(
            f"Unhandled exception: {type(exc).__name__} - {str(exc)}",
            category=LogCategory.ERROR_TRACKING,
            component="ExceptionHandler",
            error=exc,
            data={
                "path": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else "unknown",
                "query_params": dict(request.query_params),
                "headers": dict(request.headers)
            }
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An internal server error occurred",
            },
        )


# Create the application instance
app = create_app()


def main() -> None:
    """
    Main entry point for running the application.
    """
    settings = get_settings()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not settings.DEBUG else logging.DEBUG,
        format="%(message)s",
        stream=sys.stdout,
    )
    
    # Run the application with Socket.IO support
    uvicorn.run(
        "app.main:socketio_app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True,
        loop="uvloop" if sys.platform != "win32" else "asyncio",
    )


# Create the FastAPI application
app = create_app()

# Create Socket.IO ASGI app for frontend compatibility
def create_socketio_app():
    """Create Socket.IO ASGI application that wraps FastAPI."""
    from app.api.socketio.manager import socketio_manager
    import socketio

    # Create Socket.IO ASGI app that wraps FastAPI
    return socketio.ASGIApp(socketio_manager.sio, other_asgi_app=app)

# Export Socket.IO app for uvicorn
socketio_app = create_socketio_app()


if __name__ == "__main__":
    main()
