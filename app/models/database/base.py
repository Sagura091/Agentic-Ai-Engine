"""
Database base configuration and session management.

This module provides the SQLAlchemy declarative base, engine creation,
and session factory for async database operations.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.orm import declarative_base
import structlog

logger = structlog.get_logger(__name__)

# SQLAlchemy declarative base for all models
Base = declarative_base()

# Global engine and session factory instances
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the database engine.
    
    Returns:
        AsyncEngine: The SQLAlchemy async engine instance
    """
    global _engine
    
    if _engine is None:
        from app.config.settings import get_settings
        settings = get_settings()
        
        # Create async engine with connection pooling
        _engine = create_async_engine(
            settings.database_url_async,
            pool_size=settings.DATABASE_POOL_SIZE,        # 50 connections
            max_overflow=settings.DATABASE_POOL_MAX_OVERFLOW,  # 20 overflow
            pool_timeout=settings.DATABASE_POOL_TIMEOUT,  # 30 seconds
            pool_recycle=settings.DATABASE_POOL_RECYCLE,  # 1 hour
            pool_pre_ping=True,                           # Health checks
            echo=settings.DEBUG,                          # SQL logging
        )
        
        logger.info(
            "Database engine created",
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_POOL_MAX_OVERFLOW
        )
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the session factory.
    
    Returns:
        async_sessionmaker: The SQLAlchemy async session factory
    """
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


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for dependency injection.
    
    This is an async generator that yields a database session and ensures
    proper cleanup and error handling.
    
    Yields:
        AsyncSession: Database session
        
    Example:
        async for session in get_database_session():
            # Use session
            result = await session.execute(query)
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


async def init_database() -> None:
    """
    Initialize the database by creating all tables.
    
    This should be called during application startup to ensure
    all tables are created.
    """
    engine = get_engine()
    
    async with engine.begin() as conn:
        # Import all models to ensure they are registered with Base
        from app.models import (
            Agent, TaskExecution,
            Conversation, Message,
            Workflow, WorkflowExecution, WorkflowStepExecution, WorkflowTemplate,
            NodeDefinition, WorkflowNode, WorkflowConnection, NodeExecutionState,
            Tool, AgentTool, ToolExecution, ToolCategory, ToolTemplate,
            User, UserDB, UserSession,
            AutonomousAgentState, AutonomousGoalDB, AutonomousDecisionDB,
            AgentMemoryDB, LearningExperienceDB,
            DocumentDB, DocumentChunkDB,
            KnowledgeBase, KnowledgeBaseAccess
        )
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def close_database() -> None:
    """
    Close database connections and dispose of the engine.
    
    This should be called during application shutdown to ensure
    proper cleanup of database resources.
    """
    global _engine, _async_session_factory
    
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database engine disposed")
        _engine = None
        _async_session_factory = None


__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "get_database_session",
    "init_database",
    "close_database"
]

