"""
Proactive Behavior System for Autonomous Agents.

This module implements proactive behavior capabilities that enable agents to:
- Monitor context for action triggers
- Initiate self-directed tasks and actions
- Generate autonomous goals based on observations
- Adapt behavior based on environmental changes
- Learn from proactive action outcomes
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable
from enum import Enum
from dataclasses import dataclass, field

import structlog
from langchain_core.language_models import BaseLanguageModel

from app.agents.autonomous.goal_manager import AutonomousGoal, GoalType, GoalPriority
from app.agents.autonomous.autonomous_agent import AutonomousDecision

logger = structlog.get_logger(__name__)


class TriggerType(str, Enum):
    """Types of proactive behavior triggers."""
    CONTEXT_CHANGE = "context_change"       # Environment or context changes
    IDLE_STATE = "idle_state"               # Agent has no active tasks
    PERFORMANCE_DROP = "performance_drop"   # Performance metrics decline
    OPPORTUNITY = "opportunity"             # New opportunity detected
    PATTERN_MATCH = "pattern_match"         # Learned pattern recognition
    TIME_BASED = "time_based"              # Scheduled or periodic triggers
    GOAL_COMPLETION = "goal_completion"     # Goal completed, need new goals
    ERROR_PATTERN = "error_pattern"         # Recurring error patterns


class ActionType(str, Enum):
    """Types of proactive actions."""
    GOAL_GENERATION = "goal_generation"     # Generate new goals
    TASK_INITIATION = "task_initiation"     # Start new tasks
    OPTIMIZATION = "optimization"           # Optimize current processes
    EXPLORATION = "exploration"             # Explore new capabilities
    LEARNING = "learning"                   # Initiate learning activities
    COLLABORATION = "collaboration"         # Reach out to other agents
    MAINTENANCE = "maintenance"             # Perform maintenance tasks
    ADAPTATION = "adaptation"               # Adapt strategies or behavior


@dataclass
class ProactiveTrigger:
    """Represents a trigger for proactive behavior."""
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_type: TriggerType = TriggerType.CONTEXT_CHANGE
    condition: str = ""
    threshold: float = 0.5
    action_type: ActionType = ActionType.GOAL_GENERATION
    priority: float = 0.5
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    success_rate: float = 0.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProactiveAction:
    """Represents a proactive action taken by the agent."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_id: str = ""
    action_type: ActionType = ActionType.GOAL_GENERATION
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    success: Optional[bool] = None
    outcome: Dict[str, Any] = field(default_factory=dict)
    learning_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProactiveBehaviorSystem:
    """
    System for enabling proactive behavior in autonomous agents.
    
    Monitors context and triggers autonomous actions based on learned
    patterns, environmental changes, and strategic opportunities.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseLanguageModel,
        max_triggers: int = 50,
        max_actions_per_cycle: int = 3
    ):
        """
        Initialize the proactive behavior system.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: Language model for reasoning
            max_triggers: Maximum number of triggers to maintain
            max_actions_per_cycle: Maximum actions per monitoring cycle
        """
        self.agent_id = agent_id
        self.llm = llm
        self.max_triggers = max_triggers
        self.max_actions_per_cycle = max_actions_per_cycle
        
        # Trigger and action storage
        self.triggers: Dict[str, ProactiveTrigger] = {}
        self.action_history: List[ProactiveAction] = []
        
        # Context monitoring
        self.context_history: List[Dict[str, Any]] = []
        self.last_context: Dict[str, Any] = {}
        self.monitoring_enabled = True
        
        # Performance tracking
        self.behavior_stats = {
            "triggers_activated": 0,
            "actions_initiated": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "goals_generated": 0,
            "tasks_initiated": 0
        }
        
        # Initialize default triggers
        self._initialize_default_triggers()
        
        logger.info(
            "Proactive Behavior System initialized",
            agent_id=agent_id,
            max_triggers=max_triggers,
            default_triggers=len(self.triggers)
        )
    
    def _initialize_default_triggers(self):
        """Initialize default proactive triggers."""
        
        # Idle state trigger
        idle_trigger = ProactiveTrigger(
            trigger_type=TriggerType.IDLE_STATE,
            condition="no_active_tasks_for_minutes > 5",
            threshold=0.7,
            action_type=ActionType.GOAL_GENERATION,
            priority=0.6,
            cooldown_minutes=15,
            metadata={"description": "Generate goals when idle"}
        )
        self.triggers[idle_trigger.trigger_id] = idle_trigger
        
        # Performance drop trigger
        performance_trigger = ProactiveTrigger(
            trigger_type=TriggerType.PERFORMANCE_DROP,
            condition="success_rate < 0.5",
            threshold=0.8,
            action_type=ActionType.OPTIMIZATION,
            priority=0.8,
            cooldown_minutes=60,
            metadata={"description": "Optimize when performance drops"}
        )
        self.triggers[performance_trigger.trigger_id] = performance_trigger
        
        # Error pattern trigger
        error_trigger = ProactiveTrigger(
            trigger_type=TriggerType.ERROR_PATTERN,
            condition="error_count > 3",
            threshold=0.9,
            action_type=ActionType.LEARNING,
            priority=0.9,
            cooldown_minutes=30,
            metadata={"description": "Learn from error patterns"}
        )
        self.triggers[error_trigger.trigger_id] = error_trigger
        
        # Goal completion trigger
        completion_trigger = ProactiveTrigger(
            trigger_type=TriggerType.GOAL_COMPLETION,
            condition="goals_completed_recently > 0",
            threshold=0.6,
            action_type=ActionType.GOAL_GENERATION,
            priority=0.7,
            cooldown_minutes=10,
            metadata={"description": "Generate new goals after completion"}
        )
        self.triggers[completion_trigger.trigger_id] = completion_trigger
    
    async def monitor_and_act(
        self,
        current_context: Dict[str, Any],
        force_evaluation: bool = False
    ) -> Dict[str, Any]:
        """
        Monitor context and trigger proactive actions.
        
        Args:
            current_context: Current agent context
            force_evaluation: Force evaluation even if recently done
            
        Returns:
            Monitoring and action results
        """
        try:
            if not self.monitoring_enabled:
                return {"status": "disabled", "message": "Monitoring is disabled"}
            
            # Update context history
            self.context_history.append(current_context)
            if len(self.context_history) > 100:  # Keep last 100 contexts
                self.context_history.pop(0)
            
            # Evaluate triggers
            triggered_actions = await self._evaluate_triggers(current_context, force_evaluation)
            
            # Execute proactive actions
            action_results = await self._execute_proactive_actions(triggered_actions, current_context)
            
            # Update context
            self.last_context = current_context
            
            results = {
                "status": "completed",
                "triggers_evaluated": len(self.triggers),
                "actions_triggered": len(triggered_actions),
                "actions_executed": len(action_results),
                "action_results": action_results,
                "behavior_stats": self.behavior_stats.copy()
            }
            
            logger.debug(
                "Proactive monitoring completed",
                agent_id=self.agent_id,
                triggers_evaluated=results["triggers_evaluated"],
                actions_triggered=results["actions_triggered"]
            )
            
            return results
            
        except Exception as e:
            logger.error("Proactive monitoring failed", agent_id=self.agent_id, error=str(e))
            return {"status": "failed", "error": str(e)}
    
    async def _evaluate_triggers(
        self,
        context: Dict[str, Any],
        force_evaluation: bool = False
    ) -> List[ProactiveTrigger]:
        """
        Evaluate all triggers against current context.
        
        Args:
            context: Current context
            force_evaluation: Force evaluation ignoring cooldowns
            
        Returns:
            List of triggered triggers
        """
        triggered = []
        current_time = datetime.utcnow()
        
        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue
            
            # Check cooldown
            if not force_evaluation and trigger.last_triggered:
                cooldown_end = trigger.last_triggered + timedelta(minutes=trigger.cooldown_minutes)
                if current_time < cooldown_end:
                    continue
            
            # Evaluate trigger condition
            if await self._evaluate_trigger_condition(trigger, context):
                triggered.append(trigger)
                trigger.last_triggered = current_time
                trigger.trigger_count += 1
                self.behavior_stats["triggers_activated"] += 1
        
        # Sort by priority
        triggered.sort(key=lambda t: t.priority, reverse=True)
        
        # Limit to max actions per cycle
        return triggered[:self.max_actions_per_cycle]
    
    async def _evaluate_trigger_condition(
        self,
        trigger: ProactiveTrigger,
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if a trigger condition is met.
        
        Args:
            trigger: Trigger to evaluate
            context: Current context
            
        Returns:
            True if trigger condition is met
        """
        try:
            # Simple condition evaluation based on trigger type
            if trigger.trigger_type == TriggerType.IDLE_STATE:
                # Check if agent has been idle
                active_tasks = context.get("active_tasks", [])
                last_activity = context.get("last_activity_minutes", 0)
                return len(active_tasks) == 0 and last_activity > 5
            
            elif trigger.trigger_type == TriggerType.PERFORMANCE_DROP:
                # Check performance metrics
                performance = context.get("performance_metrics", {})
                success_rate = performance.get("success_rate", 1.0)
                return success_rate < 0.5
            
            elif trigger.trigger_type == TriggerType.ERROR_PATTERN:
                # Check for error patterns
                errors = context.get("errors", [])
                recent_errors = context.get("recent_error_count", 0)
                return len(errors) > 0 or recent_errors > 3
            
            elif trigger.trigger_type == TriggerType.GOAL_COMPLETION:
                # Check for recently completed goals
                completed_goals = context.get("completed_goals_recent", 0)
                return completed_goals > 0
            
            elif trigger.trigger_type == TriggerType.CONTEXT_CHANGE:
                # Check for significant context changes
                if not self.last_context:
                    return False
                
                # Simple change detection
                current_keys = set(context.keys())
                last_keys = set(self.last_context.keys())
                key_changes = len(current_keys.symmetric_difference(last_keys))
                
                return key_changes > 2  # Significant change threshold
            
            elif trigger.trigger_type == TriggerType.OPPORTUNITY:
                # Check for new opportunities
                opportunities = context.get("opportunities", [])
                return len(opportunities) > 0
            
            return False
            
        except Exception as e:
            logger.error(
                "Failed to evaluate trigger condition",
                agent_id=self.agent_id,
                trigger_id=trigger.trigger_id,
                error=str(e)
            )
            return False
    
    async def _execute_proactive_actions(
        self,
        triggered_triggers: List[ProactiveTrigger],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute proactive actions for triggered triggers.
        
        Args:
            triggered_triggers: List of triggered triggers
            context: Current context
            
        Returns:
            List of action execution results
        """
        action_results = []
        
        for trigger in triggered_triggers:
            try:
                action = ProactiveAction(
                    trigger_id=trigger.trigger_id,
                    action_type=trigger.action_type,
                    description=f"Proactive {trigger.action_type.value} triggered by {trigger.trigger_type.value}",
                    context=context.copy()
                )
                
                # Execute the action based on type
                result = await self._execute_action(action, context)
                
                action.completed_at = datetime.utcnow()
                action.success = result.get("success", False)
                action.outcome = result
                
                # Update statistics
                self.behavior_stats["actions_initiated"] += 1
                if action.success:
                    self.behavior_stats["successful_actions"] += 1
                    trigger.success_rate = (trigger.success_rate * (trigger.trigger_count - 1) + 1.0) / trigger.trigger_count
                else:
                    self.behavior_stats["failed_actions"] += 1
                    trigger.success_rate = (trigger.success_rate * (trigger.trigger_count - 1) + 0.0) / trigger.trigger_count
                
                # Store action
                self.action_history.append(action)
                if len(self.action_history) > 1000:  # Keep last 1000 actions
                    self.action_history.pop(0)
                
                action_results.append({
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "trigger_type": trigger.trigger_type.value,
                    "success": action.success,
                    "outcome": action.outcome
                })
                
            except Exception as e:
                logger.error(
                    "Failed to execute proactive action",
                    agent_id=self.agent_id,
                    trigger_id=trigger.trigger_id,
                    action_type=trigger.action_type.value,
                    error=str(e)
                )
                
                action_results.append({
                    "action_id": "failed",
                    "action_type": trigger.action_type.value,
                    "trigger_type": trigger.trigger_type.value,
                    "success": False,
                    "error": str(e)
                })
        
        return action_results

    async def _execute_action(self, action: ProactiveAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a proactive action based on its type."""
        try:
            result = {"success": False, "message": "", "data": {}}

            if action.action_type == ActionType.GOAL_GENERATION:
                result = await self._execute_goal_generation(action, context)
            elif action.action_type == ActionType.LEARNING:
                result = await self._execute_learning_action(action, context)
            elif action.action_type == ActionType.OPTIMIZATION:
                result = await self._execute_optimization_action(action, context)
            elif action.action_type == ActionType.COLLABORATION:
                result = await self._execute_communication_action(action, context)
            elif action.action_type == ActionType.MAINTENANCE:
                result = await self._execute_resource_management(action, context)
            else:
                result = {"success": False, "message": f"Unknown action type: {action.action_type}"}

            logger.debug("Proactive action executed",
                        action_id=action.action_id,
                        action_type=action.action_type.value,
                        success=result.get("success", False))

            return result

        except Exception as e:
            logger.error("Failed to execute proactive action",
                        action_id=action.action_id,
                        action_type=action.action_type.value,
                        error=str(e))
            return {"success": False, "message": str(e)}

    async def _execute_goal_generation(self, action: ProactiveAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute goal generation action."""
        try:
            # Analyze context for potential goals
            potential_goals = []

            # Look for incomplete tasks or opportunities
            if "incomplete_tasks" in context:
                for task in context["incomplete_tasks"]:
                    potential_goals.append({
                        "goal": f"Complete task: {task}",
                        "priority": 0.7,
                        "context": {"task": task, "source": "proactive_analysis"}
                    })

            # Look for optimization opportunities
            if "performance_metrics" in context:
                metrics = context["performance_metrics"]
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)) and value < 0.8:  # Below 80% performance
                        potential_goals.append({
                            "goal": f"Improve {metric} performance",
                            "priority": 0.6,
                            "context": {"metric": metric, "current_value": value, "source": "performance_analysis"}
                        })

            return {
                "success": True,
                "message": f"Generated {len(potential_goals)} potential goals",
                "data": {"goals": potential_goals}
            }

        except Exception as e:
            return {"success": False, "message": f"Goal generation failed: {str(e)}"}

    async def _execute_learning_action(self, action: ProactiveAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute learning action."""
        try:
            learning_insights = []

            # Analyze recent experiences for patterns
            if "recent_actions" in context:
                actions = context["recent_actions"]
                success_rate = sum(1 for a in actions if a.get("success", False)) / len(actions) if actions else 0

                learning_insights.append({
                    "insight": f"Recent action success rate: {success_rate:.2%}",
                    "type": "performance_pattern",
                    "confidence": 0.8
                })

            # Identify improvement opportunities
            if "errors" in context:
                error_patterns = {}
                for error in context["errors"]:
                    error_type = error.get("type", "unknown")
                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

                for error_type, count in error_patterns.items():
                    learning_insights.append({
                        "insight": f"Frequent error pattern: {error_type} ({count} occurrences)",
                        "type": "error_pattern",
                        "confidence": min(0.9, count * 0.1)
                    })

            return {
                "success": True,
                "message": f"Generated {len(learning_insights)} learning insights",
                "data": {"insights": learning_insights}
            }

        except Exception as e:
            return {"success": False, "message": f"Learning action failed: {str(e)}"}

    async def _execute_optimization_action(self, action: ProactiveAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization action."""
        try:
            optimizations = []

            # Analyze resource usage
            if "resource_usage" in context:
                usage = context["resource_usage"]
                for resource, utilization in usage.items():
                    if utilization > 0.9:  # High utilization
                        optimizations.append({
                            "type": "resource_optimization",
                            "resource": resource,
                            "current_utilization": utilization,
                            "recommendation": "Consider scaling or load balancing"
                        })

            # Analyze performance bottlenecks
            if "performance_data" in context:
                perf_data = context["performance_data"]
                for metric, value in perf_data.items():
                    if isinstance(value, dict) and "response_time" in value:
                        if value["response_time"] > 1000:  # Slow response
                            optimizations.append({
                                "type": "performance_optimization",
                                "metric": metric,
                                "response_time": value["response_time"],
                                "recommendation": "Optimize processing pipeline"
                            })

            return {
                "success": True,
                "message": f"Identified {len(optimizations)} optimization opportunities",
                "data": {"optimizations": optimizations}
            }

        except Exception as e:
            return {"success": False, "message": f"Optimization action failed: {str(e)}"}

    async def _execute_communication_action(self, action: ProactiveAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute communication action."""
        try:
            communications = []

            # Check for status updates needed
            if "status_updates" in context:
                for update in context["status_updates"]:
                    communications.append({
                        "type": "status_update",
                        "recipient": update.get("recipient", "system"),
                        "message": update.get("message", "Status update available"),
                        "priority": update.get("priority", "normal")
                    })

            # Check for alerts or notifications
            if "alerts" in context:
                for alert in context["alerts"]:
                    communications.append({
                        "type": "alert",
                        "severity": alert.get("severity", "info"),
                        "message": alert.get("message", "Alert triggered"),
                        "requires_action": alert.get("requires_action", False)
                    })

            return {
                "success": True,
                "message": f"Prepared {len(communications)} communications",
                "data": {"communications": communications}
            }

        except Exception as e:
            return {"success": False, "message": f"Communication action failed: {str(e)}"}

    async def _execute_resource_management(self, action: ProactiveAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource management action."""
        try:
            management_actions = []

            # Check resource availability
            if "resource_status" in context:
                status = context["resource_status"]
                for resource, info in status.items():
                    if info.get("available", True) == False:
                        management_actions.append({
                            "type": "resource_allocation",
                            "resource": resource,
                            "action": "reallocate",
                            "reason": "Resource unavailable"
                        })
                    elif info.get("utilization", 0) < 0.3:  # Underutilized
                        management_actions.append({
                            "type": "resource_optimization",
                            "resource": resource,
                            "action": "scale_down",
                            "reason": "Low utilization"
                        })

            return {
                "success": True,
                "message": f"Identified {len(management_actions)} resource management actions",
                "data": {"actions": management_actions}
            }

        except Exception as e:
            return {"success": False, "message": f"Resource management failed: {str(e)}"}
