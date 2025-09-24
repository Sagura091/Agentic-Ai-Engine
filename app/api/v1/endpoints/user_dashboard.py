"""
🚀 Revolutionary User Dashboard API

Provides users with real-time information about:
- Available AI models and their capabilities
- Agent upgrade suggestions
- System updates and notifications
- Performance metrics and recommendations

FEATURES:
✅ Real-time model availability
✅ Intelligent agent upgrade suggestions
✅ Personalized recommendations
✅ Performance analytics
✅ User preference management
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.auth import get_current_active_user
from ....core.admin_model_manager import admin_model_manager
from ....core.configuration_broadcaster import configuration_broadcaster, NotificationType
from ....models.auth import UserDB
from ....models.agent import Agent
from ....models.database.base import get_database_session
from ..responses import StandardAPIResponse
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/user/dashboard", tags=["User Dashboard"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AvailableModel(BaseModel):
    """Available AI model information."""
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Model provider (ollama, openai, etc.)")
    description: Optional[str] = Field(None, description="Model description")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    performance_tier: str = Field(..., description="Performance tier (basic, standard, premium)")
    size_gb: Optional[float] = Field(None, description="Model size in GB")
    context_length: Optional[int] = Field(None, description="Maximum context length")
    is_multimodal: bool = Field(default=False, description="Supports vision/images")
    recommended_use_cases: List[str] = Field(default_factory=list, description="Recommended use cases")


class AgentUpgradeSuggestion(BaseModel):
    """Agent upgrade suggestion."""
    agent_id: UUID = Field(..., description="Agent ID")
    agent_name: str = Field(..., description="Agent name")
    current_model: str = Field(..., description="Current model")
    suggested_model: str = Field(..., description="Suggested better model")
    improvement_type: str = Field(..., description="Type of improvement (speed, quality, efficiency)")
    performance_gain: float = Field(..., description="Expected performance improvement percentage")
    reason: str = Field(..., description="Reason for suggestion")
    migration_complexity: str = Field(..., description="Migration complexity (easy, medium, complex)")


class UserDashboardData(BaseModel):
    """Complete user dashboard data."""
    available_models: List[AvailableModel] = Field(default_factory=list)
    agent_suggestions: List[AgentUpgradeSuggestion] = Field(default_factory=list)
    user_agents_count: int = Field(..., description="Total number of user agents")
    active_agents_count: int = Field(..., description="Number of active agents")
    recent_notifications: List[Dict[str, Any]] = Field(default_factory=list)
    system_status: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class NotificationPreferencesUpdate(BaseModel):
    """User notification preferences update."""
    model_updates: bool = Field(default=True, description="Notify about new models")
    system_updates: bool = Field(default=True, description="Notify about system updates")
    agent_suggestions: bool = Field(default=True, description="Notify about agent upgrade suggestions")
    performance_alerts: bool = Field(default=True, description="Notify about performance issues")
    security_updates: bool = Field(default=True, description="Notify about security updates")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/", response_model=StandardAPIResponse)
async def get_dashboard_data(
    current_user: UserDB = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_database_session)
) -> StandardAPIResponse:
    """Get complete user dashboard data."""
    try:
        logger.info(f"📊 Loading dashboard data for user {current_user.username}")
        
        # Get available models
        available_models = await _get_available_models()
        
        # Get user's agents
        user_agents = await _get_user_agents(current_user.id, session)
        
        # Get agent upgrade suggestions
        agent_suggestions = await _get_agent_upgrade_suggestions(user_agents, available_models)
        
        # Get recent notifications (placeholder for now)
        recent_notifications = []
        
        # Get system status
        system_status = await _get_system_status()
        
        # Get performance metrics
        performance_metrics = await _get_performance_metrics(user_agents)
        
        dashboard_data = UserDashboardData(
            available_models=available_models,
            agent_suggestions=agent_suggestions,
            user_agents_count=len(user_agents),
            active_agents_count=len([a for a in user_agents if a.status == "active"]),
            recent_notifications=recent_notifications,
            system_status=system_status,
            performance_metrics=performance_metrics
        )
        
        logger.info(f"✅ Dashboard data loaded: {len(available_models)} models, {len(agent_suggestions)} suggestions")
        
        return StandardAPIResponse(
            success=True,
            message="Dashboard data retrieved successfully",
            data=dashboard_data.dict()
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to load dashboard data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dashboard data: {str(e)}"
        )


@router.get("/available-models", response_model=StandardAPIResponse)
async def get_available_models(
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get all available AI models for the user."""
    try:
        logger.info(f"🤖 Loading available models for user {current_user.username}")
        
        available_models = await _get_available_models()
        
        return StandardAPIResponse(
            success=True,
            message=f"Found {len(available_models)} available models",
            data={"models": [model.dict() for model in available_models]}
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to load available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load available models: {str(e)}"
        )


@router.get("/agent-suggestions", response_model=StandardAPIResponse)
async def get_agent_upgrade_suggestions(
    current_user: UserDB = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_database_session)
) -> StandardAPIResponse:
    """Get agent upgrade suggestions for the user."""
    try:
        logger.info(f"💡 Loading agent upgrade suggestions for user {current_user.username}")
        
        # Get user's agents
        user_agents = await _get_user_agents(current_user.id, session)
        
        # Get available models
        available_models = await _get_available_models()
        
        # Generate suggestions
        suggestions = await _get_agent_upgrade_suggestions(user_agents, available_models)
        
        return StandardAPIResponse(
            success=True,
            message=f"Found {len(suggestions)} upgrade suggestions",
            data={"suggestions": [suggestion.dict() for suggestion in suggestions]}
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to load agent suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load agent suggestions: {str(e)}"
        )


@router.post("/notification-preferences", response_model=StandardAPIResponse)
async def update_notification_preferences(
    preferences: NotificationPreferencesUpdate,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Update user notification preferences."""
    try:
        logger.info(f"🔔 Updating notification preferences for user {current_user.username}")
        
        # Convert to broadcaster format
        broadcaster_prefs = {
            NotificationType.MODEL_UPDATES: preferences.model_updates,
            NotificationType.SYSTEM_UPDATES: preferences.system_updates,
            NotificationType.AGENT_SUGGESTIONS: preferences.agent_suggestions,
            NotificationType.PERFORMANCE_ALERTS: preferences.performance_alerts,
            NotificationType.SECURITY_UPDATES: preferences.security_updates,
        }
        
        # Update preferences in broadcaster
        success = await configuration_broadcaster.update_user_preferences(
            str(current_user.id), 
            broadcaster_prefs
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update notification preferences"
            )
        
        return StandardAPIResponse(
            success=True,
            message="Notification preferences updated successfully",
            data=preferences.dict()
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to update notification preferences: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update notification preferences: {str(e)}"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _get_available_models() -> List[AvailableModel]:
    """Get all available AI models from admin model manager."""
    try:
        # Get models from admin model manager
        model_registry = await admin_model_manager.get_model_registry()
        
        available_models = []
        for model_name, model_info in model_registry.items():
            available_models.append(AvailableModel(
                name=model_name,
                provider=model_info.get("provider", "ollama"),
                description=model_info.get("description", ""),
                capabilities=model_info.get("capabilities", []),
                performance_tier=model_info.get("performance_tier", "standard"),
                size_gb=model_info.get("size_gb"),
                context_length=model_info.get("context_length"),
                is_multimodal=model_info.get("is_multimodal", False),
                recommended_use_cases=model_info.get("recommended_use_cases", [])
            ))
        
        return available_models
        
    except Exception as e:
        logger.error(f"❌ Failed to get available models: {str(e)}")
        return []


async def _get_user_agents(user_id: UUID, session: AsyncSession) -> List[Agent]:
    """Get all agents for a user."""
    try:
        # For now, get all agents (in real implementation, filter by user)
        result = await session.execute(
            select(Agent).where(Agent.status == "active")
        )
        return result.scalars().all()

    except Exception as e:
        logger.error(f"❌ Failed to get user agents: {str(e)}")
        return []


async def _get_agent_upgrade_suggestions(
    user_agents: List[Agent],
    available_models: List[AvailableModel]
) -> List[AgentUpgradeSuggestion]:
    """Generate intelligent agent upgrade suggestions."""
    try:
        suggestions = []

        # Create model performance mapping
        model_performance = {
            "llama3.2:latest": {"tier": "basic", "score": 60},
            "llama3.1:latest": {"tier": "standard", "score": 75},
            "llama3:latest": {"tier": "standard", "score": 70},
            "codellama:latest": {"tier": "standard", "score": 80},
            "mistral:latest": {"tier": "premium", "score": 85},
            "mixtral:latest": {"tier": "premium", "score": 90},
        }

        for agent in user_agents:
            current_model = agent.model
            current_performance = model_performance.get(current_model, {"tier": "basic", "score": 50})

            # Find better models
            for available_model in available_models:
                model_perf = model_performance.get(available_model.name, {"tier": "basic", "score": 50})

                if model_perf["score"] > current_performance["score"] + 10:  # Significant improvement
                    improvement_percentage = ((model_perf["score"] - current_performance["score"]) / current_performance["score"]) * 100

                    suggestions.append(AgentUpgradeSuggestion(
                        agent_id=agent.id,
                        agent_name=agent.name,
                        current_model=current_model,
                        suggested_model=available_model.name,
                        improvement_type="performance" if model_perf["score"] > current_performance["score"] + 20 else "efficiency",
                        performance_gain=round(improvement_percentage, 1),
                        reason=f"Upgrade to {available_model.performance_tier} tier model for better performance",
                        migration_complexity="easy" if available_model.provider == "ollama" else "medium"
                    ))

        return suggestions[:10]  # Limit to top 10 suggestions

    except Exception as e:
        logger.error(f"❌ Failed to generate agent suggestions: {str(e)}")
        return []


async def _get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    try:
        # Check Ollama connection
        ollama_status = await admin_model_manager.check_ollama_connection()

        return {
            "ollama_connected": ollama_status.get("connected", False),
            "available_models_count": len(ollama_status.get("models", [])),
            "system_health": "healthy" if ollama_status.get("connected", False) else "degraded",
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to get system status: {str(e)}")
        return {
            "ollama_connected": False,
            "available_models_count": 0,
            "system_health": "unknown",
            "last_updated": datetime.utcnow().isoformat()
        }


async def _get_performance_metrics(user_agents: List[Agent]) -> Dict[str, Any]:
    """Get performance metrics for user agents."""
    try:
        if not user_agents:
            return {
                "total_agents": 0,
                "active_agents": 0,
                "average_response_time": 0.0,
                "success_rate": 0.0
            }

        total_agents = len(user_agents)
        active_agents = len([a for a in user_agents if a.status == "active"])

        # Calculate averages
        total_tasks = sum(a.total_tasks_completed + a.total_tasks_failed for a in user_agents)
        successful_tasks = sum(a.total_tasks_completed for a in user_agents)
        avg_response_time = sum(a.average_response_time for a in user_agents) / total_agents if total_agents > 0 else 0.0
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "average_response_time": round(avg_response_time, 2),
            "success_rate": round(success_rate, 1),
            "total_tasks_completed": successful_tasks,
            "total_tasks": total_tasks
        }

    except Exception as e:
        logger.error(f"❌ Failed to get performance metrics: {str(e)}")
        return {
            "total_agents": 0,
            "active_agents": 0,
            "average_response_time": 0.0,
            "success_rate": 0.0
        }
