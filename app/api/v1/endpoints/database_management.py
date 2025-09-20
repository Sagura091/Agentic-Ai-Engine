"""
Database Management API Endpoints.

This module provides API endpoints for managing database migrations,
monitoring database health, and performing database maintenance tasks.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.auth import get_current_user
from app.models.user import User
from app.models.database.migrations.run_all_migrations import MasterMigrationRunner
from app.services.knowledge_base_migration_service import knowledge_base_migration_service

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/database", tags=["Database Management"])


# Request/Response Models
class MigrationStatusResponse(BaseModel):
    """Response model for migration status."""
    autonomous_tables: Dict[str, Any] = Field(..., description="Autonomous tables migration status")
    enhanced_tables: Dict[str, Any] = Field(..., description="Enhanced tables migration status")
    knowledge_base_data: Dict[str, Any] = Field(..., description="Knowledge base data migration status")
    overall_status: str = Field(..., description="Overall migration status")


class MigrationRunResponse(BaseModel):
    """Response model for migration run."""
    success: bool = Field(..., description="Whether migrations succeeded")
    total_migrations: int = Field(..., description="Total number of migrations")
    completed_migrations: int = Field(..., description="Number of completed migrations")
    failed_migrations: int = Field(..., description="Number of failed migrations")
    migration_results: Dict[str, Any] = Field(..., description="Detailed migration results")
    errors: list = Field(default_factory=list, description="List of errors")


class KnowledgeBaseMigrationResponse(BaseModel):
    """Response model for knowledge base migration."""
    success: bool = Field(..., description="Whether migration succeeded")
    message: str = Field(..., description="Migration message")
    migrated_count: int = Field(..., description="Number of migrated knowledge bases")
    skipped_count: int = Field(..., description="Number of skipped knowledge bases")
    errors: list = Field(default_factory=list, description="List of errors")


class DatabaseHealthResponse(BaseModel):
    """Response model for database health check."""
    healthy: bool = Field(..., description="Whether database is healthy")
    connection_status: str = Field(..., description="Database connection status")
    total_tables: int = Field(..., description="Total number of tables")
    migration_status: str = Field(..., description="Migration status")
    last_check: datetime = Field(..., description="Last health check timestamp")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")


@router.get("/health", response_model=DatabaseHealthResponse, summary="Database health check")
async def get_database_health(
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive database health information.
    
    Provides information about:
    - Database connectivity
    - Table counts
    - Migration status
    - Overall health metrics
    """
    try:
        logger.info("Database health check requested", user=current_user.username)
        
        # Get migration status
        runner = MasterMigrationRunner()
        migration_status = await runner.get_migration_status()
        
        # Check database connectivity
        from app.models.database.base import get_session_factory
        session_factory = get_session_factory()
        
        connection_healthy = True
        table_count = 0
        
        try:
            async with session_factory() as session:
                # Test connection with a simple query
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                result.scalar()
                
                # Count tables
                result = await session.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                """))
                table_count = result.scalar()
                
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            connection_healthy = False
        
        overall_healthy = (
            connection_healthy and 
            migration_status.get("overall_status") == "completed"
        )
        
        return DatabaseHealthResponse(
            healthy=overall_healthy,
            connection_status="connected" if connection_healthy else "disconnected",
            total_tables=table_count,
            migration_status=migration_status.get("overall_status", "unknown"),
            last_check=datetime.utcnow(),
            details={
                "migration_details": migration_status,
                "connection_healthy": connection_healthy
            }
        )
        
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/migrations/status", response_model=MigrationStatusResponse, summary="Get migration status")
async def get_migration_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get the status of all database migrations.
    
    Returns detailed information about:
    - Autonomous tables migration
    - Enhanced tables migration  
    - Knowledge base data migration
    - Overall migration status
    """
    try:
        logger.info("Migration status requested", user=current_user.username)
        
        runner = MasterMigrationRunner()
        status = await runner.get_migration_status()
        
        return MigrationStatusResponse(**status)
        
    except Exception as e:
        logger.error("Failed to get migration status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get migration status: {str(e)}")


@router.post("/migrations/run", response_model=MigrationRunResponse, summary="Run all migrations")
async def run_all_migrations(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Run all database migrations.
    
    This endpoint will:
    1. Create autonomous agent tables
    2. Create enhanced platform tables
    3. Migrate knowledge base data from JSON to database
    
    **Warning**: This operation may take several minutes to complete.
    """
    try:
        logger.info("Migration run requested", user=current_user.username)
        
        # Check if user has admin privileges (you may want to implement proper RBAC)
        if not getattr(current_user, 'is_superuser', False):
            raise HTTPException(status_code=403, detail="Admin privileges required for database migrations")
        
        runner = MasterMigrationRunner()
        results = await runner.run_all_migrations()
        
        logger.info(
            "Migration run completed",
            user=current_user.username,
            success=results["success"],
            completed=results["completed_migrations"],
            failed=results["failed_migrations"]
        )
        
        return MigrationRunResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Migration run failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Migration run failed: {str(e)}")


@router.post("/migrations/knowledge-bases", response_model=KnowledgeBaseMigrationResponse, summary="Migrate knowledge bases")
async def migrate_knowledge_bases(
    current_user: User = Depends(get_current_user)
):
    """
    Migrate knowledge base metadata from JSON files to database.
    
    This endpoint specifically handles migrating knowledge base data
    from the JSON file storage to PostgreSQL database tables.
    """
    try:
        logger.info("Knowledge base migration requested", user=current_user.username)
        
        result = await knowledge_base_migration_service.migrate_from_json()
        
        logger.info(
            "Knowledge base migration completed",
            user=current_user.username,
            success=result["success"],
            migrated=result["migrated_count"],
            skipped=result["skipped_count"]
        )
        
        return KnowledgeBaseMigrationResponse(**result)
        
    except Exception as e:
        logger.error("Knowledge base migration failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Knowledge base migration failed: {str(e)}")


@router.get("/migrations/knowledge-bases/status", summary="Get knowledge base migration status")
async def get_knowledge_base_migration_status(
    current_user: User = Depends(get_current_user)
):
    """Get the status of knowledge base migration from JSON to database."""
    try:
        logger.info("Knowledge base migration status requested", user=current_user.username)
        
        status = await knowledge_base_migration_service.get_migration_status()
        
        return status
        
    except Exception as e:
        logger.error("Failed to get knowledge base migration status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get migration status: {str(e)}")


@router.post("/migrations/knowledge-bases/export", summary="Export knowledge bases to JSON")
async def export_knowledge_bases_to_json(
    output_path: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Export knowledge bases from database back to JSON format.
    
    Useful for:
    - Creating backups
    - Data portability
    - Debugging migration issues
    """
    try:
        logger.info("Knowledge base export requested", user=current_user.username, output_path=output_path)
        
        result = await knowledge_base_migration_service.export_to_json(output_path)
        
        logger.info(
            "Knowledge base export completed",
            user=current_user.username,
            success=result["success"],
            count=result["count"],
            output_path=result["output_path"]
        )
        
        return result
        
    except Exception as e:
        logger.error("Knowledge base export failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Knowledge base export failed: {str(e)}")


@router.get("/tables", summary="List database tables")
async def list_database_tables(
    current_user: User = Depends(get_current_user)
):
    """
    List all database tables with basic information.
    
    Provides information about:
    - Table names
    - Schemas
    - Row counts (approximate)
    - Table sizes
    """
    try:
        logger.info("Database tables list requested", user=current_user.username)
        
        from app.models.database.base import get_session_factory
        from sqlalchemy import text
        
        session_factory = get_session_factory()
        
        async with session_factory() as session:
            # Get table information
            result = await session.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_stat_get_tuples_returned(c.oid) as row_count_estimate
                FROM pg_tables pt
                JOIN pg_class c ON c.relname = pt.tablename
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                ORDER BY schemaname, tablename
            """))
            
            tables = []
            for row in result:
                tables.append({
                    "schema": row[0],
                    "name": row[1],
                    "size": row[2],
                    "estimated_rows": row[3] or 0
                })
        
        return {
            "success": True,
            "total_tables": len(tables),
            "tables": tables
        }
        
    except Exception as e:
        logger.error("Failed to list database tables", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")
