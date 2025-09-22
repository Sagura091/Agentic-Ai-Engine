"""
Database configuration and session management for the Agentic AI Microservice.

This module provides database session management, connection pooling,
and SQLAlchemy configuration for PostgreSQL.
"""

from typing import AsyncGenerator
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

# SQLAlchemy declarative base
Base = declarative_base()

# Global variables for engine and session factory
_engine = None
_async_session_factory = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        
        # Create async engine with connection pooling
        _engine = create_async_engine(
            settings.database_url_async,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_POOL_MAX_OVERFLOW,
            pool_timeout=settings.DATABASE_POOL_TIMEOUT,
            pool_recycle=settings.DATABASE_POOL_RECYCLE,
            pool_pre_ping=True,
            echo=settings.DEBUG,  # Log SQL queries in debug mode
            poolclass=NullPool if settings.ENVIRONMENT == "test" else None,
        )
        
        logger.info(
            "Database engine created",
            database_url=settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else settings.DATABASE_URL,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_POOL_MAX_OVERFLOW
        )
    
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        logger.info("Database session factory created")
    
    return _async_session_factory


async def get_session() -> AsyncSession:
    """
    Get a database session for direct use.

    Returns:
        AsyncSession: Database session
    """
    session_factory = get_session_factory()
    return session_factory()


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for dependency injection.

    Yields:
        AsyncSession: Database session
    """
    session_factory = get_session_factory()

    async with session_factory() as session:
        try:
            logger.debug("Database session created")
            yield session
        except Exception as e:
            logger.error("Database session error", error=str(e))
            await session.rollback()
            raise
        finally:
            await session.close()
            logger.debug("Database session closed")


async def close_database():
    """Close database connections and cleanup resources."""
    global _engine, _async_session_factory
    
    if _engine:
        await _engine.dispose()
        logger.info("Database engine disposed")
        _engine = None
        _async_session_factory = None


async def init_database():
    """Initialize database and create tables if needed."""
    try:
        # Import models to ensure they are registered with Base
        from app.models.agent import Agent, TaskExecution
        from app.models.workflow import Workflow, WorkflowExecution, WorkflowStepExecution, WorkflowTemplate
        from app.models.tool import Tool, AgentTool, ToolExecution, ToolCategory, ToolTemplate

        # Import new authentication and user management models
        from app.models.auth import (
            UserDB, UserSessionDB, ProjectDB, ProjectMemberDB,
            ConversationDB, MessageDB, NotificationDB
        )

        engine = get_engine()

        # Create tables (in production, use Alembic migrations instead)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified")

    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


async def check_database_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        async for session in get_database_session():
            # Simple query to test connection
            result = await session.execute("SELECT 1")
            result.fetchone()
            logger.info("Database connection check successful")
            return True
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False
