"""
Autonomous Goal Management System for Agentic AI.

This module implements sophisticated goal-setting, planning, and pursuit
capabilities for autonomous agents, enabling self-directed behavior and
adaptive goal management.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from app.agents.autonomous.autonomous_agent import AutonomousAgentConfig
from app.agents.autonomous.persistent_memory import PersistentMemorySystem, MemoryType, MemoryImportance

logger = structlog.get_logger(__name__)


class GoalType(str, Enum):
    """Types of autonomous goals."""
    ACHIEVEMENT = "achievement"              # Achieve specific outcomes
    TASK_COMPLETION = "task_completion"      # Complete specific tasks
    LEARNING = "learning"                    # Acquire new knowledge/skills
    OPTIMIZATION = "optimization"            # Improve performance
    EXPLORATION = "exploration"              # Explore new possibilities
    COLLABORATION = "collaboration"          # Work with other agents
    MAINTENANCE = "maintenance"              # Maintain system state
    CREATIVE = "creative"                    # Generate novel solutions


class GoalPriority(str, Enum):
    """Goal priority levels."""
    CRITICAL = "critical"    # Must be completed immediately
    HIGH = "high"           # Important, should be completed soon
    MEDIUM = "medium"       # Normal priority
    LOW = "low"            # Can be deferred
    BACKGROUND = "background"  # Ongoing, low-priority goals


class GoalStatus(str, Enum):
    """Goal execution status."""
    PENDING = "pending"         # Not yet started
    ACTIVE = "active"          # Currently being pursued
    PAUSED = "paused"          # Temporarily suspended
    COMPLETED = "completed"     # Successfully completed
    FAILED = "failed"          # Failed to complete
    CANCELLED = "cancelled"     # Cancelled before completion


class GoalMetrics(BaseModel):
    """Metrics for goal tracking and evaluation."""
    model_config = {"use_enum_values": True}  # Serialize enums as their values

    progress: float = Field(default=0.0, description="Progress from 0.0 to 1.0")
    effort_invested: float = Field(default=0.0, description="Resource units invested")
    time_elapsed: float = Field(default=0.0, description="Time spent in seconds")
    success_probability: float = Field(default=0.5, description="Estimated success probability")
    value_score: float = Field(default=0.5, description="Value/importance score")
    difficulty_score: float = Field(default=0.5, description="Difficulty assessment")


class AutonomousGoal(BaseModel):
    """Model for autonomous agent goals."""
    model_config = {"use_enum_values": True}  # Serialize enums as their values

    goal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Goal definition
    title: str = Field(..., description="Goal title")
    description: str = Field(..., description="Detailed goal description")
    goal_type: GoalType = Field(..., description="Type of goal")
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM, description="Goal priority")
    status: GoalStatus = Field(default=GoalStatus.PENDING, description="Current status")
    
    # Goal parameters
    target_outcome: Dict[str, Any] = Field(..., description="Desired outcome")
    success_criteria: List[str] = Field(..., description="Criteria for success")
    constraints: List[str] = Field(default_factory=list, description="Goal constraints")
    resources_required: Dict[str, Any] = Field(default_factory=dict, description="Required resources")
    
    # Timing
    deadline: Optional[datetime] = Field(default=None, description="Goal deadline")
    estimated_duration: Optional[timedelta] = Field(default=None, description="Estimated completion time")
    
    # Tracking
    metrics: GoalMetrics = Field(default_factory=GoalMetrics, description="Goal metrics")
    sub_goals: List[str] = Field(default_factory=list, description="Sub-goal IDs")
    parent_goal: Optional[str] = Field(default=None, description="Parent goal ID")
    
    # Context
    context: Dict[str, Any] = Field(default_factory=dict, description="Goal context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __hash__(self):
        """Make AutonomousGoal hashable using goal_id."""
        return hash(self.goal_id)

    def __eq__(self, other):
        """Define equality based on goal_id."""
        if isinstance(other, AutonomousGoal):
            return self.goal_id == other.goal_id
        return False


class GoalPlan(BaseModel):
    """Plan for achieving a goal."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = Field(..., description="Associated goal ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Plan structure
    steps: List[Dict[str, Any]] = Field(..., description="Planned steps")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Step dependencies")
    resources_allocation: Dict[str, Any] = Field(default_factory=dict, description="Resource allocation")
    
    # Execution strategy
    execution_strategy: str = Field(default="sequential", description="How to execute steps")
    contingency_plans: List[Dict[str, Any]] = Field(default_factory=list, description="Backup plans")
    
    # Metrics
    estimated_success_rate: float = Field(default=0.5, description="Estimated success probability")
    estimated_effort: float = Field(default=1.0, description="Estimated effort required")


class AutonomousGoalManager:
    """
    Advanced goal management system for autonomous agents.
    
    Provides capabilities for:
    - Autonomous goal generation and prioritization
    - Hierarchical goal decomposition
    - Dynamic goal adaptation and re-prioritization
    - Goal conflict resolution
    - Progress tracking and success evaluation
    """
    
    def __init__(self, config: AutonomousAgentConfig, memory_system: Optional[PersistentMemorySystem] = None):
        """
        Initialize the goal management system.

        Args:
            config: Autonomous agent configuration
            memory_system: Optional memory system for learning from past goal outcomes
        """
        self.config = config
        self.memory_system = memory_system
        self.goals: Dict[str, AutonomousGoal] = {}
        self.goal_plans: Dict[str, GoalPlan] = {}
        self.active_goals: List[str] = []
        self.goal_history: List[Dict[str, Any]] = []

        # Goal management parameters
        self.max_active_goals = 5
        self.goal_review_interval = timedelta(minutes=30)
        self.last_goal_review = datetime.utcnow()

        # Goal generation strategies
        self.goal_generation_enabled = config.enable_goal_setting
        self.proactive_goal_generation = config.enable_proactive_behavior
        
        logger.info(
            "Autonomous goal manager initialized",
            max_active_goals=self.max_active_goals,
            goal_generation_enabled=self.goal_generation_enabled
        )

    async def add_goal(
        self,
        title: str,
        description: str,
        goal_type: GoalType = GoalType.ACHIEVEMENT,
        priority: GoalPriority = GoalPriority.MEDIUM,
        target_outcome: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[List[str]] = None,
        deadline: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new goal to the goal manager."""
        goal_id = str(uuid.uuid4())

        goal = AutonomousGoal(
            goal_id=goal_id,
            title=title,
            description=description,
            goal_type=goal_type,
            priority=priority,
            target_outcome=target_outcome or {},
            success_criteria=success_criteria or [],
            deadline=deadline,
            context=context or {}
        )

        self.goals[goal_id] = goal
        self.active_goals.append(goal_id)

        logger.info("Goal added", goal_id=goal_id, title=title, priority=priority)
        return goal_id

    async def get_goal(self, goal_id: str) -> Optional[AutonomousGoal]:
        """Get a goal by ID."""
        return self.goals.get(goal_id)

    async def update_goal_status(self, goal_id: str, status: GoalStatus) -> bool:
        """
        Update the status of a goal.

        PRODUCTION IMPLEMENTATION: Stores goal outcomes in memory for learning.
        """
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            old_status = goal.status
            goal.status = status
            goal.updated_at = datetime.utcnow()

            # CRITICAL: Store goal outcome in memory when completed or failed
            if status in [GoalStatus.COMPLETED, GoalStatus.FAILED] and self.memory_system:
                await self._store_goal_outcome_in_memory(goal, old_status, status)

            logger.info("Goal status updated", goal_id=goal_id, status=status)
            return True
        return False

    async def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal from the manager."""
        if goal_id in self.goals:
            del self.goals[goal_id]
            if goal_id in self.active_goals:
                self.active_goals.remove(goal_id)
            logger.info("Goal removed", goal_id=goal_id)
            return True
        return False

    async def update_goal_stack(self, current_goals: List[str], context: Dict[str, Any]) -> List[str]:
        """Update the active goal stack based on current goals and context."""
        try:
            # Update active goals with current goals
            self.active_goals = current_goals

            # Optionally generate new goals based on context
            if context and len(current_goals) < self.max_active_goals:
                # Generate autonomous goals if we have capacity
                await self.generate_autonomous_goals(context)

            logger.info("Goal stack updated", active_goals=len(self.active_goals))
            return self.active_goals
        except Exception as e:
            logger.error("Failed to update goal stack", error=str(e))
            return current_goals

    async def generate_autonomous_goals(self, context: Dict[str, Any]) -> None:
        """Generate autonomous goals based on context - public interface method."""
        try:
            await self._generate_autonomous_goals(context)
        except Exception as e:
            logger.error("Failed to generate autonomous goals", error=str(e))

    async def create_autonomous_plan(
        self,
        context: Dict[str, Any],
        autonomy_level: str
    ) -> Dict[str, Any]:
        """
        Create an autonomous plan based on current context and goals.
        
        Args:
            context: Current agent context
            autonomy_level: Level of autonomy for planning
            
        Returns:
            Dictionary containing the autonomous plan
        """
        try:
            # Review and update existing goals
            await self._review_goals(context)
            
            # Generate new goals if needed
            if self.goal_generation_enabled and autonomy_level in ["high", "autonomous", "emergent"]:
                await self._generate_autonomous_goals(context)
            
            # Prioritize and select active goals
            active_goals = await self._select_active_goals(context)
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(active_goals, context)
            
            # Prepare plan summary with safe goal serialization
            goals_data = []
            for goal in active_goals:  # active_goals contains AutonomousGoal objects, not IDs
                try:
                    goal_dict = goal.dict()

                    # Convert timedelta to string for JSON serialization
                    if goal_dict.get("estimated_duration"):
                        goal_dict["estimated_duration"] = str(goal_dict["estimated_duration"])

                    # Convert any remaining enum objects to strings
                    for key, value in goal_dict.items():
                        if hasattr(value, 'value'):  # Check if it's an enum
                            goal_dict[key] = value.value

                    goals_data.append(goal_dict)
                except Exception as e:
                    # If goal serialization fails, create a safe fallback representation
                    logger.warning(f"Goal serialization failed for {goal.goal_id}, using fallback", error=str(e))
                    fallback_goal = {
                        "goal_id": goal.goal_id,
                        "title": getattr(goal, 'title', 'Unknown Goal'),
                        "description": getattr(goal, 'description', 'Goal serialization failed'),
                        "goal_type": getattr(goal.goal_type, 'value', str(goal.goal_type)) if hasattr(goal, 'goal_type') else 'unknown',
                        "priority": getattr(goal.priority, 'value', str(goal.priority)) if hasattr(goal, 'priority') else 'medium',
                        "status": getattr(goal.status, 'value', str(goal.status)) if hasattr(goal, 'status') else 'pending'
                    }
                    goals_data.append(fallback_goal)

            plan = {
                "plan_id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "autonomy_level": autonomy_level,
                "goals": goals_data,
                "execution_plan": execution_plan,
                "summary": f"Autonomous plan with {len(active_goals)} active goals",
                "complexity": self._assess_plan_complexity(active_goals),
                "estimated_duration": self._estimate_plan_duration(active_goals),
                "success_probability": self._estimate_plan_success_rate(active_goals)
            }
            
            logger.info(
                "Autonomous plan created",
                goals_count=len(active_goals),
                complexity=plan["complexity"],
                success_probability=plan["success_probability"]
            )
            
            return plan
            
        except Exception as e:
            logger.error("Autonomous planning failed", error=str(e))
            return {
                "plan_id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "error": str(e),
                "goals": [],
                "summary": "Planning failed, using fallback plan",
                "complexity": "low"
            }

    def _assess_plan_complexity(self, active_goals: List[AutonomousGoal]) -> str:
        """Assess the complexity of the execution plan based on active goals."""
        if not active_goals:
            return "low"

        goal_count = len(active_goals)
        high_priority_count = sum(1 for goal in active_goals if goal.priority == GoalPriority.HIGH)

        if goal_count <= 2 and high_priority_count <= 1:
            return "low"
        elif goal_count <= 4 and high_priority_count <= 2:
            return "medium"
        else:
            return "high"

    def _estimate_plan_duration(self, active_goals: List[AutonomousGoal]) -> int:
        """Estimate the duration of the execution plan in minutes."""
        if not active_goals:
            return 5

        base_duration = 10  # Base duration per goal in minutes
        total_duration = len(active_goals) * base_duration

        # Add complexity factor
        high_priority_goals = sum(1 for goal in active_goals if goal.priority == GoalPriority.HIGH)
        complexity_factor = 1 + (high_priority_goals * 0.5)

        return int(total_duration * complexity_factor)

    def _estimate_plan_success_rate(self, active_goals: List[AutonomousGoal]) -> float:
        """Estimate the success probability of the execution plan."""
        if not active_goals:
            return 0.8

        # Base success rate
        base_rate = 0.7

        # Factor in goal metrics
        avg_success_prob = sum(goal.metrics.success_probability for goal in active_goals) / len(active_goals)

        # Combine base rate with goal-specific probabilities
        combined_rate = (base_rate + avg_success_prob) / 2

        # Adjust for plan complexity
        complexity_penalty = len(active_goals) * 0.05  # 5% penalty per additional goal
        final_rate = max(0.1, combined_rate - complexity_penalty)

        return round(final_rate, 2)
    
    async def pursue_goal(
        self,
        goal: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pursue a specific goal autonomously.
        
        Args:
            goal: Goal to pursue
            state: Current agent state
            
        Returns:
            Result of goal pursuit
        """
        try:
            goal_id = goal.get("goal_id")
            if not goal_id or goal_id not in self.goals:
                return {"status": "error", "message": "Invalid goal"}
            
            goal_obj = self.goals[goal_id]
            
            # Update goal status
            goal_obj.status = GoalStatus.ACTIVE
            goal_obj.updated_at = datetime.utcnow()
            
            # Execute goal plan
            if goal_id in self.goal_plans:
                plan = self.goal_plans[goal_id]
                result = await self._execute_goal_plan(goal_obj, plan, state)
            else:
                # Create ad-hoc plan
                plan = await self._create_goal_plan(goal_obj, state)
                self.goal_plans[goal_id] = plan
                result = await self._execute_goal_plan(goal_obj, plan, state)
            
            # Update goal metrics
            await self._update_goal_metrics(goal_obj, result)
            
            logger.info(
                "Goal pursuit completed",
                goal_id=goal_id,
                goal_type=goal_obj.goal_type,
                status=result.get("status", "unknown")
            )
            
            return result
            
        except Exception as e:
            logger.error("Goal pursuit failed", goal_id=goal.get("goal_id"), error=str(e))
            return {"status": "error", "message": str(e)}
    
    async def _review_goals(self, context: Dict[str, Any]) -> None:
        """Review and update existing goals based on current context."""
        current_time = datetime.utcnow()
        
        # Check if review is needed
        if current_time - self.last_goal_review < self.goal_review_interval:
            return
        
        goals_to_update = []
        goals_to_remove = []
        
        for goal_id, goal in self.goals.items():
            # Check for expired goals
            if goal.deadline and current_time > goal.deadline:
                if goal.status not in [GoalStatus.COMPLETED, GoalStatus.FAILED]:
                    goal.status = GoalStatus.FAILED
                    goals_to_update.append(goal_id)
            
            # Check for completed goals
            if goal.status == GoalStatus.ACTIVE:
                completion_check = await self._check_goal_completion(goal, context)
                if completion_check["completed"]:
                    goal.status = GoalStatus.COMPLETED
                    goal.metrics.progress = 1.0
                    goals_to_update.append(goal_id)
            
            # Remove old completed/failed goals
            if goal.status in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED]:
                age = current_time - goal.updated_at
                if age > timedelta(hours=24):  # Keep for 24 hours
                    goals_to_remove.append(goal_id)
        
        # Update goal history and remove old goals
        for goal_id in goals_to_remove:
            self.goal_history.append({
                "goal_id": goal_id,
                "goal": self.goals[goal_id].dict(),
                "removed_at": current_time.isoformat()
            })
            del self.goals[goal_id]
            if goal_id in self.goal_plans:
                del self.goal_plans[goal_id]
            if goal_id in self.active_goals:
                self.active_goals.remove(goal_id)
        
        self.last_goal_review = current_time
        
        if goals_to_update or goals_to_remove:
            logger.debug(
                "Goal review completed",
                updated_goals=len(goals_to_update),
                removed_goals=len(goals_to_remove),
                active_goals=len(self.active_goals)
            )
    
    async def _generate_autonomous_goals(self, context: Dict[str, Any]) -> None:
        """Generate new autonomous goals based on context."""
        if len(self.goals) >= self.max_active_goals * 2:  # Limit total goals
            return
        
        # Analyze context for goal opportunities
        goal_opportunities = await self._identify_goal_opportunities(context)
        
        for opportunity in goal_opportunities:
            # Create goal from opportunity
            goal = AutonomousGoal(
                title=opportunity["title"],
                description=opportunity["description"],
                goal_type=GoalType(opportunity["type"]),
                priority=GoalPriority(opportunity.get("priority", "medium")),
                target_outcome=opportunity["target_outcome"],
                success_criteria=opportunity["success_criteria"],
                context=opportunity.get("context", {}),
                metadata={"autonomous_generation": True, "opportunity_score": opportunity.get("score", 0.5)}
            )
            
            self.goals[goal.goal_id] = goal
            
            logger.info(
                "Autonomous goal generated",
                goal_id=goal.goal_id,
                goal_type=goal.goal_type,
                priority=goal.priority
            )

    async def generate_proactive_goals(self, performance_data: Dict[str, Any], learning_insights: List[Any]) -> List[str]:
        """Generate proactive goals based on performance analysis and learning insights."""
        new_goal_ids = []

        # Goal 1: Performance improvement goals
        if performance_data.get("success_rate", 1.0) < 0.8:
            goal_id = await self.add_goal(
                title="Improve Task Success Rate",
                description=f"Increase success rate from {performance_data.get('success_rate', 0):.2f} to 0.9+",
                goal_type=GoalType.OPTIMIZATION,
                priority=GoalPriority.HIGH,
                target_outcome={"success_rate": 0.9, "improvement_target": True},
                success_criteria=["Achieve 90%+ success rate", "Maintain for 10+ tasks"],
                context={"proactive_generation": True, "trigger": "low_success_rate"}
            )
            new_goal_ids.append(goal_id)

        # Goal 2: Learning-based goals
        for insight in learning_insights:
            if hasattr(insight, 'actionable_recommendations'):
                for recommendation in insight.actionable_recommendations[:1]:  # Limit to 1 per insight
                    goal_id = await self.add_goal(
                        title=f"Learning Improvement: {insight.insight_type}",
                        description=f"Implement learning insight: {recommendation}",
                        goal_type=GoalType.LEARNING,
                        priority=GoalPriority.MEDIUM,
                        target_outcome={"learning_improvement": True, "insight_applied": True},
                        success_criteria=[f"Successfully apply: {recommendation}"],
                        context={"proactive_generation": True, "trigger": "learning_insight"}
                    )
                    new_goal_ids.append(goal_id)

        # Goal 3: Exploration goals
        if len(self.goals) < 3:  # Encourage exploration when few goals exist
            goal_id = await self.add_goal(
                title="Explore New Capabilities",
                description="Discover and test new capabilities or optimization opportunities",
                goal_type=GoalType.EXPLORATION,
                priority=GoalPriority.LOW,
                target_outcome={"new_capabilities_discovered": True},
                success_criteria=["Identify 2+ new capabilities", "Test at least 1 capability"],
                context={"proactive_generation": True, "trigger": "exploration_needed"}
            )
            new_goal_ids.append(goal_id)

        logger.info("Proactive goals generated", count=len(new_goal_ids), goal_ids=new_goal_ids)
        return new_goal_ids
    
    async def _identify_goal_opportunities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for new autonomous goals."""
        opportunities = []
        
        # Learning opportunities
        if context.get("performance_metrics", {}).get("learning_rate", 0) < 0.5:
            opportunities.append({
                "title": "Improve Learning Efficiency",
                "description": "Enhance learning capabilities and knowledge acquisition",
                "type": "learning",
                "priority": "high",
                "target_outcome": {"learning_rate": 0.8, "knowledge_expansion": True},
                "success_criteria": ["Achieve learning rate > 0.7", "Demonstrate new capabilities"],
                "score": 0.8
            })
        
        # Optimization opportunities
        error_rate = context.get("performance_metrics", {}).get("errors_encountered", 0)
        if error_rate > 0.2:
            opportunities.append({
                "title": "Reduce Error Rate",
                "description": "Optimize performance to reduce errors and improve reliability",
                "type": "optimization",
                "priority": "high",
                "target_outcome": {"error_rate": 0.1, "reliability_improvement": True},
                "success_criteria": ["Achieve error rate < 0.1", "Maintain performance for 1 hour"],
                "score": 0.9
            })
        
        # Exploration opportunities
        if self.proactive_goal_generation and len(self.active_goals) < self.max_active_goals:
            opportunities.append({
                "title": "Explore New Capabilities",
                "description": "Investigate new tools and approaches for problem-solving",
                "type": "exploration",
                "priority": "medium",
                "target_outcome": {"new_capabilities": True, "expanded_toolkit": True},
                "success_criteria": ["Discover new useful tool", "Successfully apply new approach"],
                "score": 0.6
            })
        
        return opportunities
    
    async def update_strategies(self, strategy_updates: Dict[str, Any]) -> None:
        """Update goal management strategies based on learning."""
        try:
            if "goal_prioritization" in strategy_updates:
                # Update goal prioritization logic
                prioritization_strategy = strategy_updates["goal_prioritization"]
                logger.info("Updated goal prioritization strategy", strategy=prioritization_strategy)
            
            if "goal_generation_frequency" in strategy_updates:
                # Update goal generation frequency
                frequency = strategy_updates["goal_generation_frequency"]
                self.goal_review_interval = timedelta(minutes=frequency)
                logger.info("Updated goal generation frequency", frequency_minutes=frequency)
            
            if "max_active_goals" in strategy_updates:
                # Update maximum active goals
                max_goals = strategy_updates["max_active_goals"]
                self.max_active_goals = max(1, min(10, max_goals))  # Clamp between 1-10
                logger.info("Updated max active goals", max_goals=self.max_active_goals)
            
        except Exception as e:
            logger.error("Strategy update failed", error=str(e))

    async def _select_active_goals(self, context: Dict[str, Any]) -> List[AutonomousGoal]:
        """Select and prioritize active goals based on context and constraints."""
        try:
            # Ensure active_goals contains only strings (goal IDs), not dictionaries
            goal_ids = []
            for item in self.active_goals:
                if isinstance(item, str):
                    goal_ids.append(item)
                elif isinstance(item, dict) and 'goal_id' in item:
                    goal_ids.append(item['goal_id'])
                elif hasattr(item, 'goal_id'):
                    goal_ids.append(item.goal_id)
                else:
                    logger.warning(f"Invalid goal item in active_goals: {type(item)}")

            # Get all available goals from the goals dict using active goal IDs
            available_goals = [self.goals[goal_id] for goal_id in goal_ids if goal_id in self.goals]

            if not available_goals:
                # If no active goals, try to get all goals and select the best ones
                all_goals = list(self.goals.values())
                if not all_goals:
                    return []
                available_goals = all_goals

            # Sort goals by priority and relevance
            def goal_score(goal: AutonomousGoal) -> float:
                # Convert enum values to strings for dictionary lookup
                priority_str = goal.priority.value if hasattr(goal.priority, 'value') else str(goal.priority)
                status_str = goal.status.value if hasattr(goal.status, 'value') else str(goal.status)
                goal_type_str = goal.goal_type.value if hasattr(goal.goal_type, 'value') else str(goal.goal_type)

                priority_weight = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(priority_str, 1.0)
                status_weight = {"active": 1.0, "paused": 0.5, "completed": 0.0, "pending": 0.8}.get(status_str, 0.0)

                # Consider goal type relevance to current context
                type_relevance = 1.0
                if "task" in context:
                    task_lower = context["task"].lower()
                    if goal_type_str == "learning" and ("learn" in task_lower or "adapt" in task_lower):
                        type_relevance = 1.5
                    elif goal_type_str == "optimization" and ("improve" in task_lower or "optimize" in task_lower):
                        type_relevance = 1.5
                    elif goal_type_str == "exploration" and ("explore" in task_lower or "discover" in task_lower):
                        type_relevance = 1.5

                return priority_weight * status_weight * type_relevance

            # Sort and select top goals
            sorted_goals = sorted(available_goals, key=goal_score, reverse=True)
            selected_goals = sorted_goals[:self.max_active_goals]

            logger.debug(
                "Selected active goals",
                total_available=len(available_goals),
                selected_count=len(selected_goals),
                max_allowed=self.max_active_goals
            )

            return selected_goals

        except Exception as e:
            logger.error("Goal selection failed", error=str(e))
            return []

    async def _create_execution_plan(self, active_goals: List[AutonomousGoal], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan for active goals."""
        try:
            plan_steps = []

            for goal in active_goals:
                # Create steps for each goal
                goal_steps = {
                    "goal_id": goal.goal_id,
                    "goal_type": goal.goal_type,
                    "priority": goal.priority,
                    "steps": [
                        f"Analyze requirements for {goal.goal_type} goal",
                        f"Execute {goal.goal_type} actions",
                        f"Monitor progress and adapt"
                    ],
                    "estimated_duration": "5-10 minutes",
                    "resources_needed": ["tools", "reasoning", "analysis"]
                }
                plan_steps.append(goal_steps)

            execution_plan = {
                "plan_id": str(uuid.uuid4()),
                "total_goals": len(active_goals),
                "steps": plan_steps,
                "estimated_total_time": f"{len(active_goals) * 5}-{len(active_goals) * 10} minutes",
                "complexity": "medium" if len(active_goals) > 2 else "low",
                "created_at": datetime.utcnow().isoformat()
            }

            logger.debug(
                "Execution plan created",
                plan_id=execution_plan["plan_id"],
                total_goals=len(active_goals),
                complexity=execution_plan["complexity"]
            )

            return execution_plan

        except Exception as e:
            logger.error("Execution plan creation failed", error=str(e))
            return {
                "plan_id": str(uuid.uuid4()),
                "total_goals": 0,
                "steps": [],
                "estimated_total_time": "0 minutes",
                "complexity": "low",
                "error": str(e)
            }

    async def _store_goal_outcome_in_memory(
        self,
        goal: AutonomousGoal,
        old_status: GoalStatus,
        new_status: GoalStatus
    ) -> None:
        """
        Store goal outcome in memory for future learning.

        PRODUCTION IMPLEMENTATION: Stores goal completion/failure with rich metadata
        to enable learning from past goal pursuit patterns.

        Args:
            goal: The goal that was updated
            old_status: Previous goal status
            new_status: New goal status
        """
        if not self.memory_system:
            return

        try:
            # Build memory content
            success = new_status == GoalStatus.COMPLETED
            content_parts = [
                f"Goal: {goal.title}",
                f"Type: {goal.goal_type.value}",
                f"Priority: {goal.priority.value}",
                f"Outcome: {'SUCCESS' if success else 'FAILED'}",
                f"Progress: {goal.metrics.progress:.1%}"
            ]

            if goal.description:
                content_parts.append(f"Description: {goal.description[:100]}")

            content = " | ".join(content_parts)

            # Determine importance based on priority and outcome
            if goal.priority == GoalPriority.CRITICAL:
                importance = MemoryImportance.CRITICAL
            elif goal.priority == GoalPriority.HIGH or success:
                importance = MemoryImportance.HIGH
            elif goal.priority == GoalPriority.MEDIUM:
                importance = MemoryImportance.MEDIUM
            else:
                importance = MemoryImportance.LOW

            # Build metadata
            duration = (goal.updated_at - goal.created_at).total_seconds() if goal.updated_at else 0
            metadata = {
                "goal_id": goal.goal_id,
                "goal_type": goal.goal_type.value,
                "priority": goal.priority.value,
                "success": success,
                "progress": goal.metrics.progress,
                "attempts": goal.metrics.attempts,
                "duration_seconds": duration,
                "target_outcome": goal.target_outcome,
                "success_criteria_met": len(goal.success_criteria) if success else 0,
                "total_success_criteria": len(goal.success_criteria),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Determine emotional valence
            if success:
                emotional_valence = 0.7  # Positive for success
            else:
                emotional_valence = -0.3  # Negative for failure

            # Store as episodic memory (goal pursuit experience)
            await self.memory_system.store_memory(
                content=content,
                memory_type=MemoryType.EPISODIC,
                importance=importance,
                metadata=metadata,
                tags={"goal", goal.goal_type.value, "success" if success else "failure"},
                emotional_valence=emotional_valence
            )

            logger.debug(
                "Stored goal outcome in memory",
                goal_id=goal.goal_id,
                success=success,
                importance=importance.value,
                memory_type="episodic"
            )

        except Exception as e:
            logger.warning(f"Failed to store goal outcome in memory: {e}")
