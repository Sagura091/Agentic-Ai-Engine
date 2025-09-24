"""
🚀 Revolutionary Agent Migration API Endpoints

Comprehensive agent migration system with:
✅ Single agent model switching
✅ Bulk agent updates with batching
✅ Real-time progress tracking
✅ Performance comparison
✅ Rollback capabilities
✅ Migration history
✅ Compatibility checking

MIGRATION FEATURES:
- Zero-downtime agent migrations
- Automatic rollback on failures
- Performance validation
- User notification integration
- Comprehensive audit trail
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user, get_current_active_user
from app.models.auth import UserDB
from app.models.database.base import get_database_session
from app.api.v1.responses import StandardAPIResponse
from app.services.agent_migration_service import agent_migration_service, MigrationJobType
from app.services.model_performance_comparator import model_performance_comparator

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/agent-migration", tags=["Agent Migration"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SingleAgentMigrationRequest(BaseModel):
    """Request to migrate a single agent."""
    agent_id: str = Field(..., description="ID of the agent to migrate")
    target_model: str = Field(..., description="Target model to migrate to")
    validate_compatibility: bool = Field(default=True, description="Whether to validate model compatibility")
    rollback_enabled: bool = Field(default=True, description="Whether to enable rollback on failure")


class BulkAgentMigrationRequest(BaseModel):
    """Request to migrate multiple agents."""
    agent_ids: List[str] = Field(..., description="List of agent IDs to migrate")
    target_model: str = Field(..., description="Target model to migrate to")
    validate_compatibility: bool = Field(default=True, description="Whether to validate model compatibility")
    rollback_enabled: bool = Field(default=True, description="Whether to enable rollback on failure")


class ModelComparisonRequest(BaseModel):
    """Request to compare two models."""
    model_a: str = Field(..., description="First model to compare")
    model_b: str = Field(..., description="Second model to compare")
    include_live_benchmarks: bool = Field(default=False, description="Whether to run live benchmarks")


# ============================================================================
# SINGLE AGENT MIGRATION
# ============================================================================

@router.post("/migrate-agent")
async def migrate_single_agent(
    request: SingleAgentMigrationRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Migrate a single agent to a new model."""
    try:
        logger.info(f"🔄 User {current_user.email} requesting single agent migration: {request.agent_id} -> {request.target_model}")
        
        # Start the migration
        result = await agent_migration_service.migrate_single_agent(
            agent_id=request.agent_id,
            target_model=request.target_model,
            user_id=str(current_user.id),
            validate_compatibility=request.validate_compatibility,
            rollback_enabled=request.rollback_enabled
        )
        
        if result["success"]:
            return StandardAPIResponse(
                success=True,
                message=result["message"],
                data={
                    "job_id": result["job_id"],
                    "agent_id": request.agent_id,
                    "target_model": request.target_model,
                    "estimated_duration": result["estimated_duration"],
                    "status": "migration_started"
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start single agent migration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start agent migration: {str(e)}"
        )


# ============================================================================
# BULK AGENT MIGRATION
# ============================================================================

@router.post("/migrate-agents-bulk")
async def migrate_agents_bulk(
    request: BulkAgentMigrationRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Migrate multiple agents to a new model."""
    try:
        logger.info(f"🔄 User {current_user.email} requesting bulk agent migration: {len(request.agent_ids)} agents -> {request.target_model}")
        
        # Validate request
        if len(request.agent_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No agent IDs provided for migration"
            )
        
        if len(request.agent_ids) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot migrate more than 100 agents at once"
            )
        
        # Start the bulk migration
        result = await agent_migration_service.migrate_bulk_agents(
            agent_ids=request.agent_ids,
            target_model=request.target_model,
            user_id=str(current_user.id),
            validate_compatibility=request.validate_compatibility,
            rollback_enabled=request.rollback_enabled
        )
        
        if result["success"]:
            return StandardAPIResponse(
                success=True,
                message=result["message"],
                data={
                    "job_id": result["job_id"],
                    "agent_count": len(request.agent_ids),
                    "target_model": request.target_model,
                    "estimated_duration": result["estimated_duration"],
                    "status": "bulk_migration_started"
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start bulk agent migration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk migration: {str(e)}"
        )


# ============================================================================
# MIGRATION PROGRESS AND MANAGEMENT
# ============================================================================

@router.get("/migration-progress/{job_id}")
async def get_migration_progress(
    job_id: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get migration job progress."""
    try:
        logger.info(f"📊 User {current_user.email} checking migration progress: {job_id}")
        
        # Get progress from migration service
        progress = await agent_migration_service.get_migration_progress(job_id)
        
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Migration job {job_id} not found"
            )
        
        # Verify user owns this job
        if progress["user_id"] != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view your own migration jobs"
            )
        
        return StandardAPIResponse(
            success=True,
            message="Migration progress retrieved successfully",
            data=progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get migration progress: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get migration progress: {str(e)}"
        )


@router.post("/cancel-migration/{job_id}")
async def cancel_migration(
    job_id: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Cancel an active migration job."""
    try:
        logger.info(f"🛑 User {current_user.email} cancelling migration: {job_id}")
        
        # Cancel the migration
        result = await agent_migration_service.cancel_migration(job_id, str(current_user.id))
        
        if result["success"]:
            return StandardAPIResponse(
                success=True,
                message=result["message"],
                data={"job_id": job_id, "status": "cancelled"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to cancel migration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel migration: {str(e)}"
        )


@router.post("/rollback-migration/{job_id}")
async def rollback_migration(
    job_id: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Rollback a completed migration."""
    try:
        logger.info(f"🔄 User {current_user.email} rolling back migration: {job_id}")
        
        # Rollback the migration
        result = await agent_migration_service.rollback_migration(job_id, str(current_user.id))
        
        if result["success"]:
            return StandardAPIResponse(
                success=True,
                message=result["message"],
                data={"job_id": job_id, "status": "rolled_back"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to rollback migration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback migration: {str(e)}"
        )


# ============================================================================
# MIGRATION HISTORY
# ============================================================================

@router.get("/migration-history")
async def get_migration_history(
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of jobs to return"),
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get migration history for the current user."""
    try:
        logger.info(f"📋 User {current_user.email} requesting migration history (limit: {limit})")
        
        # Get migration history
        history = await agent_migration_service.get_user_migration_history(str(current_user.id), limit)
        
        return StandardAPIResponse(
            success=True,
            message=f"Retrieved {len(history)} migration jobs",
            data={
                "migration_jobs": history,
                "total_count": len(history),
                "user_id": str(current_user.id)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get migration history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get migration history: {str(e)}"
        )


# ============================================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================================

@router.post("/compare-models")
async def compare_models(
    request: ModelComparisonRequest,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Compare performance between two models."""
    try:
        logger.info(f"🔍 User {current_user.email} comparing models: {request.model_a} vs {request.model_b}")

        # Run model comparison
        comparison = await model_performance_comparator.compare_models(
            model_a=request.model_a,
            model_b=request.model_b,
            include_live_benchmarks=request.include_live_benchmarks
        )

        # Convert comparison to serializable format
        comparison_data = {
            "model_a": comparison.model_a,
            "model_b": comparison.model_b,
            "overall_winner": comparison.overall_winner,
            "confidence": comparison.confidence,
            "summary": comparison.summary,
            "recommendations": comparison.recommendations,
            "timestamp": comparison.timestamp.isoformat(),
            "performance_scores": {
                metric.value: {
                    "model_a_score": score_a.score,
                    "model_b_score": score_b.score,
                    "winner": comparison.model_a if score_a.score > score_b.score else comparison.model_b,
                    "difference": abs(score_a.score - score_b.score)
                }
                for metric, (score_a, score_b) in comparison.performance_scores.items()
            },
            "benchmark_results": comparison.benchmark_results
        }

        return StandardAPIResponse(
            success=True,
            message=f"Model comparison completed: {comparison.overall_winner} wins",
            data=comparison_data
        )

    except Exception as e:
        logger.error(f"❌ Failed to compare models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare models: {str(e)}"
        )


@router.get("/model-performance/{model_name}")
async def get_model_performance(
    model_name: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get comprehensive performance profile for a model."""
    try:
        logger.info(f"📊 User {current_user.email} requesting performance profile: {model_name}")

        # Get model performance profile
        profile = await model_performance_comparator.get_model_performance_profile(model_name)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Performance profile not found for model: {model_name}"
            )

        return StandardAPIResponse(
            success=True,
            message=f"Performance profile retrieved for {model_name}",
            data=profile
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get model performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model performance: {str(e)}"
        )


@router.get("/model-trends/{model_name}")
async def get_model_performance_trends(
    model_name: str,
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get performance trends for a model over time."""
    try:
        logger.info(f"📈 User {current_user.email} requesting performance trends: {model_name} ({days} days)")

        # Get performance trends
        trends = await model_performance_comparator.get_performance_trends(model_name, days)

        if not trends:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Performance trends not found for model: {model_name}"
            )

        return StandardAPIResponse(
            success=True,
            message=f"Performance trends retrieved for {model_name}",
            data=trends
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get performance trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance trends: {str(e)}"
        )


# ============================================================================
# AGENT UPGRADE SUGGESTIONS
# ============================================================================

@router.get("/upgrade-suggestions")
async def get_agent_upgrade_suggestions(
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get upgrade suggestions for user's agents."""
    try:
        logger.info(f"💡 User {current_user.email} requesting agent upgrade suggestions")

        # This would typically analyze user's agents and suggest better models
        # For now, return a placeholder response
        suggestions = [
            {
                "agent_id": "example-agent-1",
                "agent_name": "Research Assistant",
                "current_model": "llama3.2:latest",
                "suggested_model": "llama3.1:latest",
                "improvement_reason": "Better reasoning capabilities",
                "performance_gain": 15.5,
                "migration_complexity": "low"
            }
        ]

        return StandardAPIResponse(
            success=True,
            message=f"Found {len(suggestions)} upgrade suggestions",
            data={
                "suggestions": suggestions,
                "total_count": len(suggestions),
                "user_id": str(current_user.id)
            }
        )

    except Exception as e:
        logger.error(f"❌ Failed to get upgrade suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get upgrade suggestions: {str(e)}"
        )
