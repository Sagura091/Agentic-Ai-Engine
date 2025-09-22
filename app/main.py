"""
Main FastAPI application entry point for the Agentic AI Microservice.

This module sets up the FastAPI application with all necessary middleware,
routers, and integrations for the agentic AI system.
"""

import asyncio
import json
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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
# from app.orchestration.orchestrator import orchestrator
# from app.orchestration.enhanced_orchestrator import enhanced_orchestrator
from app.core.seamless_integration import seamless_integration
from app.services.monitoring_service import monitoring_service

# Import backend logging system
from app.backend_logging.backend_logger import get_logger, configure_logger
from app.backend_logging.middleware import LoggingMiddleware as BackendLoggingMiddleware, PerformanceMonitoringMiddleware
from app.backend_logging.models import LogConfiguration, LogLevel, LogCategory


# Configure clean, production-ready logging
def setup_clean_logging():
    """Setup clean logging with minimal console output and detailed file logging."""

    # Set root logger to WARNING to reduce spam
    logging.getLogger().setLevel(logging.WARNING)

    # Set specific loggers to appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.ERROR)

    # Configure structlog for clean output
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Clean console output - only essential info
            structlog.dev.ConsoleRenderer(colors=True, pad_event=25) if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Setup clean logging
setup_clean_logging()

logger = structlog.get_logger(__name__)


def setup_signal_handlers():
    """Setup graceful signal handlers for hot reload compatibility."""
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Allow the lifespan context manager to handle cleanup
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # On Windows, also handle SIGBREAK
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal_handler)


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

    # Initialize backend logging system with minimal console spam
    backend_logger = configure_logger(LogConfiguration(
        log_level=LogLevel.ERROR,  # Only errors to console
        enable_console_output=False,  # Disable console output to reduce spam
        enable_file_output=True,      # Keep detailed file logging
        enable_json_format=True,      # JSON format for file logs
        enable_async_logging=True,
        enable_performance_logging=True,
        enable_agent_metrics=True,
        enable_api_metrics=True,
        enable_database_metrics=True,
        log_retention_days=30,
        max_log_file_size_mb=100,
        max_log_files=10
    ))

    # Clean startup message
    print(f"🚀 Starting Agentic AI Microservice v{app.version}")
    logger.info("System initializing...", version=app.version)

    try:
        # Initialize seamless integration system (replaces orchestrator)
        print("⚙️  Initializing core systems...")
        await seamless_integration.initialize_complete_system()

        # Initialize system orchestrator
        from app.core.unified_system_orchestrator import get_system_orchestrator
        orchestrator = await get_system_orchestrator()
        await orchestrator.initialize()

        await websocket_manager.initialize()
        await monitoring_service.initialize()

        # Initialize advanced node system
        backend_logger.info(
            "Initializing advanced node system",
            category=LogCategory.SYSTEM_HEALTH,
            component="NodeSystem"
        )
        from app.core.node_bootstrap import initialize_node_system
        await initialize_node_system()

        print("✅ All services initialized successfully")
        print("🎯 System ready: Unlimited agents, Dynamic tools, True agentic AI")
        logger.info("System startup complete")

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
            # Graceful shutdown with timeout
            await asyncio.wait_for(monitoring_service.shutdown(), timeout=5.0)
            await asyncio.wait_for(websocket_manager.shutdown(), timeout=5.0)
            # await orchestrator.shutdown()

            logger.info("All services shut down successfully")
            backend_logger.info(
                "All services shut down successfully",
                category=LogCategory.SYSTEM_HEALTH,
                component="FastAPI"
            )

            # Shutdown backend logging system last
            backend_logger.shutdown()

            # Force cleanup of scientific computing libraries
            try:
                import gc
                gc.collect()
                # Give time for Intel Fortran runtime cleanup
                await asyncio.sleep(0.1)
            except Exception:
                pass  # Ignore cleanup errors

        except asyncio.TimeoutError:
            logger.warning("Shutdown timeout reached, forcing exit")
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
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

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
    # Optimize for development performance - reduce logging overhead
    app.add_middleware(
        BackendLoggingMiddleware,
        exclude_paths=["/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico", "/api/v1/monitoring/system"],
        include_request_body=False,  # Disable request body logging for better performance
        include_response_body=False,  # Disable response body logging for better performance
        max_body_size=512,  # Reduce max body size
        log_level=LogLevel.INFO  # Use INFO level for better performance
    )

    # Performance monitoring middleware - optimized thresholds
    app.add_middleware(
        PerformanceMonitoringMiddleware,
        slow_request_threshold_ms=2000,  # Increase threshold to reduce noise
        memory_threshold_mb=1000,  # Increase memory threshold
        cpu_threshold_percent=90  # Increase CPU threshold
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
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time agent communication."""
        try:
            # Accept the connection immediately to test
            await websocket.accept()
            logger.info("✅ WebSocket connection accepted successfully")

            # Send a test message
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "message": "Connected to Agentic AI Backend",
                "timestamp": datetime.now().isoformat()
            }))

            # Keep connection alive and handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    logger.info("📨 Received WebSocket message", message_type=message.get("type"))

                    # Echo back a response
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "original_message": message,
                        "timestamp": datetime.now().isoformat()
                    }))

                except WebSocketDisconnect:
                    logger.info("🔌 WebSocket client disconnected")
                    break
                except Exception as e:
                    logger.error("❌ WebSocket message error", error=str(e))
                    break

        except Exception as e:
            logger.error("❌ WebSocket connection error", error=str(e))
            try:
                await websocket.close(code=1000)
            except:
                pass

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
    
    # Configure clean console logging - only warnings and errors
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors on console
        format="%(levelname)s: %(message)s",  # Simple format
        stream=sys.stdout,
    )
    
    # Run the application with Socket.IO support and clean logging
    uvicorn.run(
        "app.main:socketio_app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="warning",  # Only warnings and errors from uvicorn
        access_log=False,  # Disable access logs for cleaner output
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
