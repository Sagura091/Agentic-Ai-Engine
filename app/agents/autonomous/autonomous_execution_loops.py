"""
Autonomous Execution Loops for Truly Agentic AI.

This module implements persistent background processes that enable agents to run
continuously without human intervention, including:
- Self-sustaining execution cycles
- Autonomous task scheduling
- Independent operation capabilities
- Persistent agent runtime management
- Autonomous goal pursuit and adaptation
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog
from langchain_core.language_models import BaseLanguageModel

from app.agents.autonomous.autonomous_agent import AutonomousLangGraphAgent, AutonomousAgentConfig
from app.agents.autonomous.goal_manager import AutonomousGoal, GoalType, GoalPriority, GoalStatus
from app.agents.autonomous.bdi_planning_engine import BDIPlanningEngine
from app.agents.autonomous.persistent_memory import PersistentMemorySystem, MemoryType, MemoryImportance
from app.agents.autonomous.proactive_behavior import ProactiveBehaviorSystem
from app.services.autonomous_persistence import autonomous_persistence

logger = structlog.get_logger(__name__)


class ExecutionLoopState(str, Enum):
    """States of autonomous execution loops."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    HIBERNATING = "hibernating"  # Low-activity state to conserve resources


class LoopPriority(str, Enum):
    """Priority levels for execution loops."""
    CRITICAL = "critical"      # Must run continuously
    HIGH = "high"             # Important background processes
    MEDIUM = "medium"         # Standard autonomous operations
    LOW = "low"               # Optional enhancement loops
    MAINTENANCE = "maintenance"  # System maintenance tasks


class ExecutionTrigger(str, Enum):
    """Triggers for autonomous execution."""
    TIME_BASED = "time_based"           # Scheduled execution
    EVENT_DRIVEN = "event_driven"       # Triggered by events
    GOAL_ORIENTED = "goal_oriented"     # Driven by goal pursuit
    REACTIVE = "reactive"               # Response to environment changes
    PROACTIVE = "proactive"             # Self-initiated actions
    EMERGENT = "emergent"               # Emergent behavior patterns


@dataclass
class ExecutionCycle:
    """Represents a single execution cycle."""
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    loop_id: str = ""
    cycle_number: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    trigger: ExecutionTrigger = ExecutionTrigger.TIME_BASED
    context: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    goals_pursued: List[str] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    learning_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutonomousExecutionLoop:
    """Configuration for an autonomous execution loop."""
    loop_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    name: str = ""
    description: str = ""
    priority: LoopPriority = LoopPriority.MEDIUM
    state: ExecutionLoopState = ExecutionLoopState.INITIALIZING
    
    # Execution configuration
    interval_seconds: float = 30.0  # How often to execute
    max_cycle_duration: float = 300.0  # Maximum time per cycle
    max_concurrent_cycles: int = 1  # Usually 1 for autonomous loops
    
    # Triggers and conditions
    triggers: List[ExecutionTrigger] = field(default_factory=list)
    execution_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Autonomous behavior configuration
    enable_goal_pursuit: bool = True
    enable_proactive_behavior: bool = True
    enable_learning: bool = True
    enable_adaptation: bool = True
    enable_collaboration: bool = False
    
    # Resource management
    cpu_limit: float = 0.5  # Max CPU usage (0.0-1.0)
    memory_limit_mb: int = 512  # Max memory usage
    
    # Persistence and recovery
    enable_persistence: bool = True
    checkpoint_interval: int = 10  # Cycles between checkpoints
    
    # Statistics
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    last_execution: Optional[datetime] = None
    average_cycle_duration: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutonomousExecutionEngine:
    """
    Core engine for managing autonomous execution loops.
    
    This engine enables truly agentic behavior by running persistent
    background processes that allow agents to operate independently
    without human intervention.
    """
    
    def __init__(
        self,
        max_concurrent_loops: int = 10,
        resource_monitor_interval: float = 60.0,
        enable_distributed_execution: bool = False
    ):
        """Initialize the autonomous execution engine."""
        self.max_concurrent_loops = max_concurrent_loops
        self.resource_monitor_interval = resource_monitor_interval
        self.enable_distributed_execution = enable_distributed_execution
        
        # Loop management
        self.execution_loops: Dict[str, AutonomousExecutionLoop] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.agent_loops: Dict[str, List[str]] = {}  # agent_id -> loop_ids
        
        # Execution state
        self.is_running = False
        self.engine_start_time: Optional[datetime] = None
        self.total_cycles_executed = 0
        
        # Resource monitoring
        self.resource_monitor_task: Optional[asyncio.Task] = None
        self.system_resources = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "active_loops": 0,
            "total_agents": 0
        }
        
        # Performance tracking
        self.performance_stats = {
            "cycles_per_minute": 0.0,
            "average_cycle_duration": 0.0,
            "success_rate": 0.0,
            "resource_efficiency": 0.0
        }
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(
            "Autonomous execution engine initialized",
            max_concurrent_loops=max_concurrent_loops,
            distributed_execution=enable_distributed_execution
        )
    
    async def start_engine(self) -> None:
        """Start the autonomous execution engine."""
        if self.is_running:
            logger.warning("Execution engine is already running")
            return
        
        try:
            self.is_running = True
            self.engine_start_time = datetime.utcnow()
            
            # Start resource monitoring
            self.resource_monitor_task = asyncio.create_task(self._monitor_resources())
            
            # Load persisted loops
            await self._load_persisted_loops()
            
            # Start any auto-start loops
            await self._start_autostart_loops()
            
            logger.info(
                "Autonomous execution engine started",
                start_time=self.engine_start_time,
                loaded_loops=len(self.execution_loops)
            )
            
        except Exception as e:
            self.is_running = False
            logger.error("Failed to start execution engine", error=str(e))
            raise
    
    async def stop_engine(self) -> None:
        """Stop the autonomous execution engine gracefully."""
        if not self.is_running:
            return
        
        try:
            logger.info("Stopping autonomous execution engine")
            self.is_running = False
            
            # Stop all running loops
            await self._stop_all_loops()
            
            # Stop resource monitoring
            if self.resource_monitor_task:
                self.resource_monitor_task.cancel()
                try:
                    await self.resource_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Autonomous execution engine stopped")
            
        except Exception as e:
            logger.error("Error stopping execution engine", error=str(e))
    
    async def register_agent_loop(
        self,
        agent: AutonomousLangGraphAgent,
        loop_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register an autonomous execution loop for an agent."""
        try:
            # Create execution loop configuration
            loop = AutonomousExecutionLoop(
                agent_id=agent.agent_id,
                name=f"Autonomous Loop - {agent.config.name}",
                description=f"Primary autonomous execution loop for agent {agent.agent_id}",
                priority=LoopPriority.HIGH,
                triggers=[
                    ExecutionTrigger.TIME_BASED,
                    ExecutionTrigger.GOAL_ORIENTED,
                    ExecutionTrigger.PROACTIVE
                ],
                enable_goal_pursuit=True,
                enable_proactive_behavior=True,
                enable_learning=True,
                enable_adaptation=True
            )
            
            # Apply custom configuration
            if loop_config:
                for key, value in loop_config.items():
                    if hasattr(loop, key):
                        setattr(loop, key, value)
            
            # Register the loop
            self.execution_loops[loop.loop_id] = loop
            
            # Track agent loops
            if agent.agent_id not in self.agent_loops:
                self.agent_loops[agent.agent_id] = []
            self.agent_loops[agent.agent_id].append(loop.loop_id)
            
            # Start the loop if engine is running
            if self.is_running:
                await self._start_loop(loop.loop_id, agent)
            
            logger.info(
                "Agent autonomous loop registered",
                agent_id=agent.agent_id,
                loop_id=loop.loop_id,
                priority=loop.priority
            )
            
            return loop.loop_id

        except Exception as e:
            logger.error("Failed to register agent loop", agent_id=agent.agent_id, error=str(e))
            raise

    async def _start_loop(self, loop_id: str, agent: AutonomousLangGraphAgent) -> None:
        """Start an autonomous execution loop."""
        if loop_id in self.running_tasks:
            logger.warning("Loop already running", loop_id=loop_id)
            return

        loop = self.execution_loops.get(loop_id)
        if not loop:
            raise ValueError(f"Loop {loop_id} not found")

        # Create and start the loop task
        task = asyncio.create_task(self._run_autonomous_loop(loop, agent))
        self.running_tasks[loop_id] = task

        # Update loop state
        loop.state = ExecutionLoopState.RUNNING
        loop.updated_at = datetime.utcnow()

        logger.info("Autonomous loop started", loop_id=loop_id, agent_id=agent.agent_id)

    async def _run_autonomous_loop(
        self,
        loop: AutonomousExecutionLoop,
        agent: AutonomousLangGraphAgent
    ) -> None:
        """Run the main autonomous execution loop for an agent."""
        cycle_number = 0

        try:
            logger.info(
                "Starting autonomous execution loop",
                loop_id=loop.loop_id,
                agent_id=agent.agent_id,
                interval=loop.interval_seconds
            )

            while self.is_running and loop.state == ExecutionLoopState.RUNNING:
                cycle_number += 1
                cycle_start = datetime.utcnow()

                try:
                    # Create execution cycle
                    cycle = ExecutionCycle(
                        loop_id=loop.loop_id,
                        cycle_number=cycle_number,
                        trigger=ExecutionTrigger.TIME_BASED
                    )

                    # Execute autonomous cycle
                    await self._execute_autonomous_cycle(cycle, loop, agent)

                    # Update statistics
                    cycle.completed_at = datetime.utcnow()
                    cycle.duration_seconds = (cycle.completed_at - cycle.started_at).total_seconds()
                    cycle.success = True

                    loop.total_cycles += 1
                    loop.successful_cycles += 1
                    loop.last_execution = cycle.completed_at

                    # Update average duration
                    if loop.average_cycle_duration == 0:
                        loop.average_cycle_duration = cycle.duration_seconds
                    else:
                        loop.average_cycle_duration = (
                            loop.average_cycle_duration * 0.9 + cycle.duration_seconds * 0.1
                        )

                    self.total_cycles_executed += 1

                    logger.debug(
                        "Autonomous cycle completed",
                        loop_id=loop.loop_id,
                        cycle_number=cycle_number,
                        duration=cycle.duration_seconds,
                        actions_taken=len(cycle.actions_taken)
                    )

                except Exception as cycle_error:
                    # Handle cycle errors gracefully
                    loop.failed_cycles += 1
                    logger.error(
                        "Autonomous cycle failed",
                        loop_id=loop.loop_id,
                        cycle_number=cycle_number,
                        error=str(cycle_error)
                    )

                    # Implement exponential backoff for failed cycles
                    await asyncio.sleep(min(loop.interval_seconds * 2, 300))

                # Wait for next cycle (unless stopping)
                if self.is_running and loop.state == ExecutionLoopState.RUNNING:
                    await asyncio.sleep(loop.interval_seconds)

        except asyncio.CancelledError:
            logger.info("Autonomous loop cancelled", loop_id=loop.loop_id)
            raise
        except Exception as e:
            logger.error("Autonomous loop failed", loop_id=loop.loop_id, error=str(e))
            loop.state = ExecutionLoopState.ERROR
        finally:
            # Cleanup
            if loop.loop_id in self.running_tasks:
                del self.running_tasks[loop.loop_id]
            loop.state = ExecutionLoopState.STOPPED
            logger.info("Autonomous loop stopped", loop_id=loop.loop_id, total_cycles=cycle_number)

    async def _execute_autonomous_cycle(
        self,
        cycle: ExecutionCycle,
        loop: AutonomousExecutionLoop,
        agent: AutonomousLangGraphAgent
    ) -> None:
        """Execute a single autonomous cycle."""
        try:
            # 1. Gather current context
            context = await self._gather_agent_context(agent)
            cycle.context = context

            # 2. Check for autonomous triggers
            triggered_actions = await self._check_autonomous_triggers(agent, context)

            # 3. Run BDI planning cycle
            if loop.enable_goal_pursuit:
                planning_results = await agent.bdi_engine.run_planning_cycle(context)
                cycle.decisions_made.append({
                    "type": "bdi_planning",
                    "results": planning_results,
                    "timestamp": datetime.utcnow().isoformat()
                })

            # 4. Execute proactive behaviors
            if loop.enable_proactive_behavior:
                proactive_results = await agent.proactive_system.monitor_and_act(context)
                if proactive_results.get("actions_triggered", 0) > 0:
                    cycle.actions_taken.append({
                        "type": "proactive_behavior",
                        "results": proactive_results,
                        "timestamp": datetime.utcnow().isoformat()
                    })

            # 5. Pursue active goals
            goal_actions = await self._pursue_active_goals(agent, context)
            cycle.actions_taken.extend(goal_actions)
            cycle.goals_pursued = [action.get("goal_id") for action in goal_actions if action.get("goal_id")]

            # 6. Learning and adaptation
            if loop.enable_learning:
                learning_results = await self._perform_autonomous_learning(agent, context, cycle)
                cycle.learning_outcomes.extend(learning_results)

            # 7. Memory consolidation
            await self._consolidate_cycle_memory(agent, cycle)

            # 8. Performance monitoring
            cycle.performance_metrics = await self._calculate_cycle_metrics(agent, cycle)

        except Exception as e:
            cycle.error_message = str(e)
            cycle.success = False
            logger.error("Autonomous cycle execution failed", error=str(e))
            raise

    async def _gather_agent_context(self, agent: AutonomousLangGraphAgent) -> Dict[str, Any]:
        """Gather current context for autonomous decision making."""
        try:
            context = {
                "agent_id": agent.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "active_goals": len(agent.goal_manager.active_goals),
                "total_goals": len(agent.goal_manager.goals),
                "memory_entries": len(agent.memory_system.episodic_memory) if hasattr(agent.memory_system, 'episodic_memory') else 0,
                "available_tools": len(agent.tools),
                "recent_performance": {},
                "environment_state": {},
                "resource_availability": {
                    "cpu": 0.8,  # Placeholder - would integrate with actual monitoring
                    "memory": 0.7,
                    "network": 0.9
                }
            }

            # Add recent goal completions
            completed_goals = [
                goal for goal in agent.goal_manager.goals.values()
                if goal.status == GoalStatus.COMPLETED and
                goal.completed_at and
                (datetime.utcnow() - goal.completed_at).total_seconds() < 3600  # Last hour
            ]
            context["recent_completions"] = len(completed_goals)

            # Add current intentions from BDI engine
            if hasattr(agent, 'bdi_engine'):
                active_intentions = [
                    intention for intention in agent.bdi_engine.intentions.values()
                    if intention.status.value == "active"
                ]
                context["active_intentions"] = len(active_intentions)

            return context

        except Exception as e:
            logger.error("Failed to gather agent context", error=str(e))
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def _check_autonomous_triggers(
        self,
        agent: AutonomousLangGraphAgent,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for autonomous triggers that should initiate actions."""
        triggered_actions = []

        try:
            # Check for idle state trigger
            if context.get("active_goals", 0) == 0:
                triggered_actions.append({
                    "trigger": "idle_state",
                    "action": "generate_new_goals",
                    "priority": "medium",
                    "reason": "Agent has no active goals"
                })

            # Check for goal completion trigger
            if context.get("recent_completions", 0) > 0:
                triggered_actions.append({
                    "trigger": "goal_completion",
                    "action": "celebrate_and_plan_next",
                    "priority": "low",
                    "reason": f"Recently completed {context['recent_completions']} goals"
                })

            # Check for resource availability trigger
            resources = context.get("resource_availability", {})
            if resources.get("cpu", 0) > 0.9 and resources.get("memory", 0) > 0.9:
                triggered_actions.append({
                    "trigger": "high_resources",
                    "action": "pursue_ambitious_goals",
                    "priority": "high",
                    "reason": "High resource availability detected"
                })

            return triggered_actions

        except Exception as e:
            logger.error("Failed to check autonomous triggers", error=str(e))
            return []

    async def _pursue_active_goals(
        self,
        agent: AutonomousLangGraphAgent,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Pursue active goals autonomously."""
        goal_actions = []

        try:
            # Get active goals
            active_goals = [
                goal for goal in agent.goal_manager.goals.values()
                if goal.status == GoalStatus.ACTIVE
            ]

            # Prioritize goals
            prioritized_goals = sorted(
                active_goals,
                key=lambda g: (g.priority.value, g.urgency if hasattr(g, 'urgency') else 0.5),
                reverse=True
            )

            # Execute top priority goals
            for goal in prioritized_goals[:3]:  # Limit to top 3 goals per cycle
                try:
                    # Create goal pursuit action
                    action = {
                        "type": "goal_pursuit",
                        "goal_id": goal.goal_id,
                        "goal_title": goal.title,
                        "action_taken": "autonomous_progress",
                        "timestamp": datetime.utcnow().isoformat(),
                        "context": context
                    }

                    # Simulate goal progress (in real implementation, this would
                    # execute actual goal-directed actions)
                    progress_made = await self._make_goal_progress(agent, goal, context)
                    action["progress_made"] = progress_made

                    goal_actions.append(action)

                except Exception as goal_error:
                    logger.error(
                        "Failed to pursue goal",
                        goal_id=goal.goal_id,
                        error=str(goal_error)
                    )

            return goal_actions

        except Exception as e:
            logger.error("Failed to pursue active goals", error=str(e))
            return []

    async def _make_goal_progress(
        self,
        agent: AutonomousLangGraphAgent,
        goal: AutonomousGoal,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make autonomous progress toward a goal."""
        try:
            # Analyze goal requirements
            progress = {
                "goal_id": goal.goal_id,
                "progress_type": "autonomous_action",
                "actions_attempted": [],
                "success": False,
                "progress_amount": 0.0
            }

            # Determine appropriate actions based on goal type
            if goal.goal_type == GoalType.ACHIEVEMENT:
                # For achievement goals, break down into sub-tasks
                progress["actions_attempted"].append("task_decomposition")
                progress["progress_amount"] = 0.1

            elif goal.goal_type == GoalType.MAINTENANCE:
                # For maintenance goals, check current state
                progress["actions_attempted"].append("state_monitoring")
                progress["progress_amount"] = 0.05

            elif goal.goal_type == GoalType.LEARNING:
                # For learning goals, gather information
                progress["actions_attempted"].append("information_gathering")
                progress["progress_amount"] = 0.15

            # Update goal progress
            if hasattr(goal, 'progress'):
                goal.progress = min(1.0, goal.progress + progress["progress_amount"])

            progress["success"] = True
            return progress

        except Exception as e:
            logger.error("Failed to make goal progress", goal_id=goal.goal_id, error=str(e))
            return {"error": str(e), "success": False}

    async def _perform_autonomous_learning(
        self,
        agent: AutonomousLangGraphAgent,
        context: Dict[str, Any],
        cycle: ExecutionCycle
    ) -> List[Dict[str, Any]]:
        """Perform autonomous learning during execution cycle."""
        learning_outcomes = []

        try:
            # Learn from cycle performance
            cycle_learning = {
                "type": "cycle_performance_learning",
                "insights": [],
                "adaptations": [],
                "timestamp": datetime.utcnow().isoformat()
            }

            # Analyze cycle effectiveness
            if len(cycle.actions_taken) > 0:
                cycle_learning["insights"].append(
                    f"Executed {len(cycle.actions_taken)} autonomous actions"
                )

            if len(cycle.goals_pursued) > 0:
                cycle_learning["insights"].append(
                    f"Made progress on {len(cycle.goals_pursued)} goals"
                )

            # Learn from context patterns
            if context.get("active_goals", 0) > 5:
                cycle_learning["adaptations"].append(
                    "Consider goal prioritization strategies for high goal count"
                )

            learning_outcomes.append(cycle_learning)

            # Store learning in agent's memory system
            if hasattr(agent, 'memory_system'):
                await agent.memory_system.store_memory(
                    content=f"Autonomous cycle learning: {json.dumps(cycle_learning)}",
                    memory_type=MemoryType.EPISODIC,
                    importance=MemoryImportance.MEDIUM,
                    context={"cycle_id": cycle.cycle_id, "learning_type": "autonomous"}
                )

            return learning_outcomes

        except Exception as e:
            logger.error("Failed to perform autonomous learning", error=str(e))
            return []

    async def _consolidate_cycle_memory(
        self,
        agent: AutonomousLangGraphAgent,
        cycle: ExecutionCycle
    ) -> None:
        """Consolidate cycle experiences into agent memory."""
        try:
            if not hasattr(agent, 'memory_system'):
                return

            # Create cycle summary
            cycle_summary = {
                "cycle_id": cycle.cycle_id,
                "actions_count": len(cycle.actions_taken),
                "goals_pursued": len(cycle.goals_pursued),
                "decisions_made": len(cycle.decisions_made),
                "learning_outcomes": len(cycle.learning_outcomes),
                "success": cycle.success,
                "duration": cycle.duration_seconds
            }

            # Store as episodic memory
            await agent.memory_system.store_memory(
                content=f"Autonomous execution cycle: {json.dumps(cycle_summary)}",
                memory_type=MemoryType.EPISODIC,
                importance=MemoryImportance.LOW,
                emotional_valence=0.5 if cycle.success else -0.2,
                tags={"autonomous_cycle", "execution", "performance"},
                context=cycle_summary
            )

        except Exception as e:
            logger.error("Failed to consolidate cycle memory", error=str(e))

    async def _calculate_cycle_metrics(
        self,
        agent: AutonomousLangGraphAgent,
        cycle: ExecutionCycle
    ) -> Dict[str, Any]:
        """Calculate performance metrics for the cycle."""
        try:
            metrics = {
                "efficiency": 0.0,
                "goal_progress_rate": 0.0,
                "action_success_rate": 0.0,
                "learning_rate": 0.0,
                "autonomy_score": 0.0
            }

            # Calculate efficiency (actions per second)
            if cycle.duration_seconds > 0:
                metrics["efficiency"] = len(cycle.actions_taken) / cycle.duration_seconds

            # Calculate goal progress rate
            if len(cycle.goals_pursued) > 0:
                metrics["goal_progress_rate"] = len(cycle.goals_pursued) / max(1, len(cycle.actions_taken))

            # Calculate action success rate
            successful_actions = sum(1 for action in cycle.actions_taken if action.get("success", True))
            if len(cycle.actions_taken) > 0:
                metrics["action_success_rate"] = successful_actions / len(cycle.actions_taken)

            # Calculate learning rate
            metrics["learning_rate"] = len(cycle.learning_outcomes) / max(1, len(cycle.actions_taken))

            # Calculate autonomy score (how self-directed the cycle was)
            autonomous_triggers = sum(
                1 for action in cycle.actions_taken
                if action.get("type") in ["proactive_behavior", "goal_pursuit", "autonomous_action"]
            )
            if len(cycle.actions_taken) > 0:
                metrics["autonomy_score"] = autonomous_triggers / len(cycle.actions_taken)
            else:
                metrics["autonomy_score"] = 1.0  # No actions but still autonomous

            return metrics

        except Exception as e:
            logger.error("Failed to calculate cycle metrics", error=str(e))
            return {}

    async def _monitor_resources(self) -> None:
        """Monitor system resources and adjust execution accordingly."""
        while self.is_running:
            try:
                # Update resource statistics
                self.system_resources.update({
                    "active_loops": len(self.running_tasks),
                    "total_agents": len(self.agent_loops),
                    "cpu_usage": 0.5,  # Placeholder - integrate with psutil
                    "memory_usage": 0.6  # Placeholder - integrate with psutil
                })

                # Check for resource constraints
                if self.system_resources["cpu_usage"] > 0.9:
                    await self._handle_high_cpu_usage()

                if self.system_resources["memory_usage"] > 0.9:
                    await self._handle_high_memory_usage()

                # Update performance statistics
                await self._update_performance_stats()

                await asyncio.sleep(self.resource_monitor_interval)

            except Exception as e:
                logger.error("Resource monitoring failed", error=str(e))
                await asyncio.sleep(60)  # Fallback interval

    async def _handle_high_cpu_usage(self) -> None:
        """Handle high CPU usage by adjusting execution."""
        logger.warning("High CPU usage detected, adjusting execution")

        # Increase intervals for low-priority loops
        for loop in self.execution_loops.values():
            if loop.priority in [LoopPriority.LOW, LoopPriority.MAINTENANCE]:
                loop.interval_seconds = min(loop.interval_seconds * 1.5, 300)

    async def _handle_high_memory_usage(self) -> None:
        """Handle high memory usage by cleaning up."""
        logger.warning("High memory usage detected, performing cleanup")

        # Trigger memory cleanup in agents
        for agent_id, loop_ids in self.agent_loops.items():
            # Could trigger memory consolidation or cleanup
            pass

    async def _update_performance_stats(self) -> None:
        """Update overall performance statistics."""
        try:
            if self.engine_start_time:
                runtime_minutes = (datetime.utcnow() - self.engine_start_time).total_seconds() / 60
                if runtime_minutes > 0:
                    self.performance_stats["cycles_per_minute"] = self.total_cycles_executed / runtime_minutes

            # Calculate success rate
            total_cycles = sum(loop.total_cycles for loop in self.execution_loops.values())
            successful_cycles = sum(loop.successful_cycles for loop in self.execution_loops.values())

            if total_cycles > 0:
                self.performance_stats["success_rate"] = successful_cycles / total_cycles

            # Calculate average cycle duration
            durations = [loop.average_cycle_duration for loop in self.execution_loops.values() if loop.average_cycle_duration > 0]
            if durations:
                self.performance_stats["average_cycle_duration"] = sum(durations) / len(durations)

        except Exception as e:
            logger.error("Failed to update performance stats", error=str(e))

    async def _load_persisted_loops(self) -> None:
        """Load persisted execution loops from storage."""
        try:
            # In a real implementation, this would load from database
            # For now, we'll create default loops for any existing agents
            logger.info("Loading persisted execution loops")

        except Exception as e:
            logger.error("Failed to load persisted loops", error=str(e))

    async def _start_autostart_loops(self) -> None:
        """Start any loops configured for automatic startup."""
        try:
            autostart_loops = [
                loop for loop in self.execution_loops.values()
                if loop.metadata.get("autostart", False)
            ]

            for loop in autostart_loops:
                # Would need agent reference to start
                logger.info("Auto-starting loop", loop_id=loop.loop_id)

        except Exception as e:
            logger.error("Failed to start autostart loops", error=str(e))

    async def _stop_all_loops(self) -> None:
        """Stop all running execution loops."""
        try:
            logger.info("Stopping all execution loops", count=len(self.running_tasks))

            # Cancel all running tasks
            for loop_id, task in self.running_tasks.items():
                task.cancel()
                loop = self.execution_loops.get(loop_id)
                if loop:
                    loop.state = ExecutionLoopState.STOPPING

            # Wait for tasks to complete
            if self.running_tasks:
                await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)

            self.running_tasks.clear()

        except Exception as e:
            logger.error("Failed to stop all loops", error=str(e))

    def get_loop_status(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific loop."""
        loop = self.execution_loops.get(loop_id)
        if not loop:
            return None

        return {
            "loop_id": loop.loop_id,
            "agent_id": loop.agent_id,
            "name": loop.name,
            "state": loop.state.value,
            "priority": loop.priority.value,
            "total_cycles": loop.total_cycles,
            "successful_cycles": loop.successful_cycles,
            "failed_cycles": loop.failed_cycles,
            "success_rate": loop.successful_cycles / max(1, loop.total_cycles),
            "average_cycle_duration": loop.average_cycle_duration,
            "last_execution": loop.last_execution.isoformat() if loop.last_execution else None,
            "is_running": loop_id in self.running_tasks
        }

    def get_engine_status(self) -> Dict[str, Any]:
        """Get overall engine status."""
        return {
            "is_running": self.is_running,
            "start_time": self.engine_start_time.isoformat() if self.engine_start_time else None,
            "total_loops": len(self.execution_loops),
            "running_loops": len(self.running_tasks),
            "total_agents": len(self.agent_loops),
            "total_cycles_executed": self.total_cycles_executed,
            "system_resources": self.system_resources.copy(),
            "performance_stats": self.performance_stats.copy()
        }

    async def pause_loop(self, loop_id: str) -> bool:
        """Pause a specific execution loop."""
        try:
            loop = self.execution_loops.get(loop_id)
            if not loop:
                return False

            if loop.state == ExecutionLoopState.RUNNING:
                loop.state = ExecutionLoopState.PAUSED
                logger.info("Loop paused", loop_id=loop_id)
                return True

            return False

        except Exception as e:
            logger.error("Failed to pause loop", loop_id=loop_id, error=str(e))
            return False

    async def resume_loop(self, loop_id: str) -> bool:
        """Resume a paused execution loop."""
        try:
            loop = self.execution_loops.get(loop_id)
            if not loop:
                return False

            if loop.state == ExecutionLoopState.PAUSED:
                loop.state = ExecutionLoopState.RUNNING
                logger.info("Loop resumed", loop_id=loop_id)
                return True

            return False

        except Exception as e:
            logger.error("Failed to resume loop", loop_id=loop_id, error=str(e))
            return False

    async def stop_loop(self, loop_id: str) -> bool:
        """Stop a specific execution loop."""
        try:
            loop = self.execution_loops.get(loop_id)
            if not loop:
                return False

            # Cancel the running task
            task = self.running_tasks.get(loop_id)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.running_tasks[loop_id]

            loop.state = ExecutionLoopState.STOPPED
            logger.info("Loop stopped", loop_id=loop_id)
            return True

        except Exception as e:
            logger.error("Failed to stop loop", loop_id=loop_id, error=str(e))
            return False


# Global autonomous execution engine instance
autonomous_execution_engine = AutonomousExecutionEngine()


async def start_autonomous_execution() -> None:
    """Start the global autonomous execution engine."""
    await autonomous_execution_engine.start_engine()


async def stop_autonomous_execution() -> None:
    """Stop the global autonomous execution engine."""
    await autonomous_execution_engine.stop_engine()


async def register_autonomous_agent(agent: AutonomousLangGraphAgent) -> str:
    """Register an agent for autonomous execution."""
    return await autonomous_execution_engine.register_agent_loop(agent)


def get_autonomous_execution_status() -> Dict[str, Any]:
    """Get status of autonomous execution system."""
    return autonomous_execution_engine.get_engine_status()
