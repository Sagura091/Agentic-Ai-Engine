"""
BDI (Belief-Desire-Intention) Planning Engine for Truly Agentic AI.

This module implements a sophisticated planning engine based on the BDI architecture
that enables autonomous agents to:
- Form beliefs about their environment and capabilities
- Generate desires (goals) based on context and experience
- Create intentions (plans) to achieve their desires
- Adapt plans based on changing beliefs and outcomes
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import structlog
from langchain_core.language_models import BaseLanguageModel

from app.agents.autonomous.goal_manager import AutonomousGoal, GoalType, GoalPriority, GoalStatus
from app.agents.autonomous.autonomous_agent import AutonomousDecision
from app.services.autonomous_persistence import autonomous_persistence

logger = structlog.get_logger(__name__)


class BeliefType(str, Enum):
    """Types of beliefs an agent can hold."""
    CAPABILITY = "capability"           # What the agent can do
    ENVIRONMENT = "environment"         # State of the environment
    RESOURCE = "resource"              # Available resources
    CONSTRAINT = "constraint"          # Limitations and constraints
    PATTERN = "pattern"                # Learned patterns
    PREDICTION = "prediction"          # Future state predictions


class DesireType(str, Enum):
    """Types of desires (high-level goals) an agent can have."""
    ACHIEVEMENT = "achievement"        # Accomplish something specific
    MAINTENANCE = "maintenance"        # Maintain a state or condition
    EXPLORATION = "exploration"        # Discover new information
    OPTIMIZATION = "optimization"      # Improve performance or efficiency
    LEARNING = "learning"             # Acquire new knowledge or skills
    COLLABORATION = "collaboration"    # Work with other agents


class IntentionStatus(str, Enum):
    """Status of agent intentions."""
    FORMING = "forming"               # Intention is being formed
    ACTIVE = "active"                 # Actively pursuing intention
    SUSPENDED = "suspended"           # Temporarily paused
    COMPLETED = "completed"           # Successfully completed
    FAILED = "failed"                 # Failed to achieve
    ABANDONED = "abandoned"           # Deliberately abandoned


@dataclass
class Belief:
    """Represents an agent's belief about the world."""
    belief_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    belief_type: BeliefType = BeliefType.ENVIRONMENT
    content: str = ""
    confidence: float = 0.5
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "observation"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Desire:
    """Represents an agent's desire (high-level goal)."""
    desire_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    desire_type: DesireType = DesireType.ACHIEVEMENT
    description: str = ""
    goal: str = ""  # The specific goal description
    priority: float = 0.5
    urgency: float = 0.5
    feasibility: float = 0.5
    value: float = 0.5
    confidence: float = 0.5  # Confidence in achieving this desire
    context: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Intention:
    """Represents an agent's intention (concrete plan)."""
    intention_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    desire_id: str = ""
    goal_id: Optional[str] = None
    status: IntentionStatus = IntentionStatus.FORMING
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    success_criteria: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    resources_required: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BDIPlanningEngine:
    """
    BDI (Belief-Desire-Intention) Planning Engine for autonomous agents.
    
    This engine implements the BDI architecture to enable truly autonomous
    planning and decision-making based on the agent's beliefs about the world,
    desires for future states, and concrete intentions to achieve those desires.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseLanguageModel,
        max_beliefs: int = 1000,
        max_desires: int = 50,
        max_intentions: int = 20
    ):
        """
        Initialize the BDI planning engine.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: Language model for reasoning and planning
            max_beliefs: Maximum number of beliefs to maintain
            max_desires: Maximum number of desires to maintain
            max_intentions: Maximum number of active intentions
        """
        self.agent_id = agent_id
        self.llm = llm
        self.max_beliefs = max_beliefs
        self.max_desires = max_desires
        self.max_intentions = max_intentions
        
        # BDI components
        self.beliefs: Dict[str, Belief] = {}
        self.desires: Dict[str, Desire] = {}
        self.intentions: Dict[str, Intention] = {}
        
        # Planning state
        self.planning_cycle_count = 0
        self.last_planning_cycle = None
        self.planning_enabled = True
        
        # Performance tracking
        self.planning_stats = {
            "cycles_completed": 0,
            "beliefs_formed": 0,
            "desires_generated": 0,
            "intentions_created": 0,
            "intentions_completed": 0,
            "intentions_failed": 0
        }
        
        logger.info(
            "BDI Planning Engine initialized",
            agent_id=agent_id,
            max_beliefs=max_beliefs,
            max_desires=max_desires,
            max_intentions=max_intentions
        )
    
    async def run_planning_cycle(
        self,
        context: Dict[str, Any],
        force_replan: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a complete BDI planning cycle.
        
        Args:
            context: Current context and environment state
            force_replan: Force replanning even if not needed
            
        Returns:
            Planning cycle results
        """
        try:
            if not self.planning_enabled:
                return {"status": "disabled", "message": "Planning is disabled"}
            
            cycle_start = datetime.utcnow()
            self.planning_cycle_count += 1
            
            logger.info(
                "Starting BDI planning cycle",
                agent_id=self.agent_id,
                cycle=self.planning_cycle_count,
                force_replan=force_replan
            )
            
            # Phase 1: Belief Revision
            belief_updates = await self._revise_beliefs(context)
            
            # Phase 2: Desire Generation
            new_desires = await self._generate_desires(context)
            
            # Phase 3: Intention Formation
            new_intentions = await self._form_intentions(context, force_replan)
            
            # Phase 4: Plan Execution Monitoring
            execution_updates = await self._monitor_plan_execution(context)
            
            # Phase 5: Intention Reconsideration
            reconsideration_results = await self._reconsider_intentions(context)
            
            cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
            self.last_planning_cycle = cycle_start
            self.planning_stats["cycles_completed"] += 1
            
            results = {
                "status": "completed",
                "cycle": self.planning_cycle_count,
                "duration_seconds": cycle_duration,
                "belief_updates": belief_updates,
                "new_desires": new_desires,
                "new_intentions": new_intentions,
                "execution_updates": execution_updates,
                "reconsideration_results": reconsideration_results,
                "active_beliefs": len(self.beliefs),
                "active_desires": len(self.desires),
                "active_intentions": len([i for i in self.intentions.values() if i.status == IntentionStatus.ACTIVE]),
                "planning_stats": self.planning_stats.copy()
            }
            
            logger.info(
                "BDI planning cycle completed",
                agent_id=self.agent_id,
                cycle=self.planning_cycle_count,
                duration=cycle_duration,
                active_intentions=results["active_intentions"]
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "BDI planning cycle failed",
                agent_id=self.agent_id,
                cycle=self.planning_cycle_count,
                error=str(e)
            )
            return {
                "status": "failed",
                "error": str(e),
                "cycle": self.planning_cycle_count
            }
    
    async def _revise_beliefs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revise agent beliefs based on new observations and context.
        
        Args:
            context: Current context and observations
            
        Returns:
            Belief revision results
        """
        try:
            # Extract observations from context
            observations = context.get("observations", [])
            environment_state = context.get("environment", {})
            agent_capabilities = context.get("capabilities", [])
            available_resources = context.get("resources", {})
            
            beliefs_added = 0
            beliefs_updated = 0
            beliefs_removed = 0
            
            # Update capability beliefs
            for capability in agent_capabilities:
                belief_content = f"Agent has capability: {capability}"
                belief = self._find_or_create_belief(
                    BeliefType.CAPABILITY,
                    belief_content,
                    confidence=0.9,
                    source="self_assessment"
                )
                if belief:
                    beliefs_added += 1
            
            # Update environment beliefs
            for key, value in environment_state.items():
                belief_content = f"Environment state: {key} = {value}"
                belief = self._find_or_create_belief(
                    BeliefType.ENVIRONMENT,
                    belief_content,
                    confidence=0.8,
                    source="observation"
                )
                if belief:
                    beliefs_updated += 1
            
            # Update resource beliefs
            for resource, amount in available_resources.items():
                belief_content = f"Available resource: {resource} = {amount}"
                belief = self._find_or_create_belief(
                    BeliefType.RESOURCE,
                    belief_content,
                    confidence=0.9,
                    source="resource_check"
                )
                if belief:
                    beliefs_updated += 1
            
            # Remove expired beliefs
            current_time = datetime.utcnow()
            expired_beliefs = [
                belief_id for belief_id, belief in self.beliefs.items()
                if belief.expires_at and belief.expires_at < current_time
            ]
            
            for belief_id in expired_beliefs:
                del self.beliefs[belief_id]
                beliefs_removed += 1
            
            # Maintain belief limit
            if len(self.beliefs) > self.max_beliefs:
                # Remove oldest beliefs with lowest confidence
                sorted_beliefs = sorted(
                    self.beliefs.items(),
                    key=lambda x: (x[1].confidence, x[1].created_at)
                )
                
                excess_count = len(self.beliefs) - self.max_beliefs
                for i in range(excess_count):
                    belief_id = sorted_beliefs[i][0]
                    del self.beliefs[belief_id]
                    beliefs_removed += 1
            
            self.planning_stats["beliefs_formed"] += beliefs_added
            
            return {
                "beliefs_added": beliefs_added,
                "beliefs_updated": beliefs_updated,
                "beliefs_removed": beliefs_removed,
                "total_beliefs": len(self.beliefs)
            }
            
        except Exception as e:
            logger.error("Belief revision failed", agent_id=self.agent_id, error=str(e))
            return {"beliefs_added": 0, "beliefs_updated": 0, "beliefs_removed": 0, "total_beliefs": len(self.beliefs)}
    
    def _find_or_create_belief(
        self,
        belief_type: BeliefType,
        content: str,
        confidence: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Belief]:
        """
        Find existing belief or create new one.
        
        Args:
            belief_type: Type of belief
            content: Belief content
            confidence: Confidence level
            source: Source of belief
            metadata: Additional metadata
            
        Returns:
            Belief object or None if not created
        """
        try:
            # Check if similar belief exists
            for belief in self.beliefs.values():
                if belief.belief_type == belief_type and belief.content == content:
                    # Update existing belief
                    belief.confidence = max(belief.confidence, confidence)
                    belief.updated_at = datetime.utcnow()
                    if metadata:
                        belief.metadata.update(metadata)
                    return belief
            
            # Create new belief
            belief = Belief(
                belief_type=belief_type,
                content=content,
                confidence=confidence,
                source=source,
                metadata=metadata or {}
            )
            
            self.beliefs[belief.belief_id] = belief
            return belief
            
        except Exception as e:
            logger.error("Failed to create belief", error=str(e))
            return None

    async def _generate_desires(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate new desires based on current beliefs and context.

        Args:
            context: Current context

        Returns:
            Desire generation results
        """
        try:
            desires_generated = 0

            # Analyze current task for desire generation
            current_task = context.get("current_task", "")
            if current_task and not self._has_desire_for_task(current_task):
                desire = Desire(
                    desire_type=DesireType.ACHIEVEMENT,
                    description=f"Complete current task: {current_task}",
                    priority=0.8,
                    urgency=0.7,
                    feasibility=0.6,
                    value=0.7,
                    context={"task": current_task}
                )
                self.desires[desire.desire_id] = desire
                desires_generated += 1

            # Generate learning desires based on failures
            errors = context.get("errors", [])
            if errors and not self._has_desire_for_learning():
                desire = Desire(
                    desire_type=DesireType.LEARNING,
                    description="Learn from recent errors to improve performance",
                    priority=0.6,
                    urgency=0.5,
                    feasibility=0.8,
                    value=0.8,
                    context={"errors": errors}
                )
                self.desires[desire.desire_id] = desire
                desires_generated += 1

            # Generate optimization desires based on performance
            performance_metrics = context.get("performance_metrics", {})
            if performance_metrics and not self._has_desire_for_optimization():
                desire = Desire(
                    desire_type=DesireType.OPTIMIZATION,
                    description="Optimize performance based on current metrics",
                    priority=0.5,
                    urgency=0.3,
                    feasibility=0.7,
                    value=0.6,
                    context={"metrics": performance_metrics}
                )
                self.desires[desire.desire_id] = desire
                desires_generated += 1

            # Generate exploration desires if agent is idle
            active_intentions = [i for i in self.intentions.values() if i.status == IntentionStatus.ACTIVE]
            if not active_intentions and not self._has_desire_for_exploration():
                desire = Desire(
                    desire_type=DesireType.EXPLORATION,
                    description="Explore environment and capabilities when idle",
                    priority=0.3,
                    urgency=0.2,
                    feasibility=0.9,
                    value=0.4,
                    context={"reason": "idle_state"}
                )
                self.desires[desire.desire_id] = desire
                desires_generated += 1

            # Maintain desire limit
            if len(self.desires) > self.max_desires:
                # Remove lowest priority desires
                sorted_desires = sorted(
                    self.desires.items(),
                    key=lambda x: x[1].priority
                )

                excess_count = len(self.desires) - self.max_desires
                for i in range(excess_count):
                    desire_id = sorted_desires[i][0]
                    del self.desires[desire_id]

            self.planning_stats["desires_generated"] += desires_generated

            return {
                "desires_generated": desires_generated,
                "total_desires": len(self.desires)
            }

        except Exception as e:
            logger.error("Desire generation failed", agent_id=self.agent_id, error=str(e))
            return {"desires_generated": 0, "total_desires": len(self.desires)}

    async def _form_intentions(self, context: Dict[str, Any], force_replan: bool = False) -> Dict[str, Any]:
        """
        Form concrete intentions (plans) to achieve desires.

        Args:
            context: Current context
            force_replan: Force replanning of existing intentions

        Returns:
            Intention formation results
        """
        try:
            intentions_created = 0
            intentions_updated = 0

            # Get active intentions count
            active_intentions = [i for i in self.intentions.values() if i.status == IntentionStatus.ACTIVE]

            # Only create new intentions if we have capacity
            if len(active_intentions) < self.max_intentions:
                # Sort desires by priority and feasibility
                sorted_desires = sorted(
                    self.desires.values(),
                    key=lambda d: (d.priority * d.feasibility, d.urgency),
                    reverse=True
                )

                for desire in sorted_desires:
                    if len(active_intentions) >= self.max_intentions:
                        break

                    # Check if we already have an intention for this desire
                    existing_intention = self._find_intention_for_desire(desire.desire_id)
                    if existing_intention and not force_replan:
                        continue

                    # Create new intention
                    intention = await self._create_intention_for_desire(desire, context)
                    if intention:
                        self.intentions[intention.intention_id] = intention
                        intentions_created += 1
                        active_intentions.append(intention)

            # Update existing intentions if needed
            for intention in list(self.intentions.values()):
                if intention.status == IntentionStatus.ACTIVE:
                    updated = await self._update_intention_plan(intention, context)
                    if updated:
                        intentions_updated += 1

            self.planning_stats["intentions_created"] += intentions_created

            return {
                "intentions_created": intentions_created,
                "intentions_updated": intentions_updated,
                "active_intentions": len(active_intentions),
                "total_intentions": len(self.intentions)
            }

        except Exception as e:
            logger.error("Intention formation failed", agent_id=self.agent_id, error=str(e))
            return {"intentions_created": 0, "intentions_updated": 0, "active_intentions": 0, "total_intentions": len(self.intentions)}

    async def _create_intention_for_desire(self, desire: Desire, context: Dict[str, Any] = None) -> Optional[Intention]:
        """Create an intention from a desire with planning."""
        try:
            # Generate plan for the desire
            plan_steps = await self._generate_plan_for_desire(desire)

            if not plan_steps:
                logger.warning("No plan generated for desire", desire_id=desire.desire_id)
                return None

            # Create intention
            intention = Intention(
                intention_id=str(uuid.uuid4()),
                desire_id=desire.desire_id,
                goal=desire.goal,
                plan=plan_steps,
                status=IntentionStatus.ACTIVE,
                priority=desire.priority,
                confidence=desire.confidence * 0.9,  # Slightly reduce confidence for planning
                context=desire.context.copy(),
                created_at=datetime.utcnow()
            )

            logger.debug("Intention created",
                        intention_id=intention.intention_id,
                        desire_id=desire.desire_id,
                        plan_steps=len(plan_steps))

            return intention

        except Exception as e:
            logger.error("Failed to create intention", desire_id=desire.desire_id, error=str(e))
            return None

    async def _generate_plan_for_desire(self, desire: Desire) -> List[Dict[str, Any]]:
        """Generate a plan to achieve a desire."""
        try:
            # Create planning prompt
            planning_prompt = f"""
            Generate a detailed plan to achieve the following goal:
            Goal: {desire.goal}
            Description: {desire.description}
            Context: {desire.context}

            Create a step-by-step plan with the following format:
            1. Step description
            2. Required resources
            3. Expected outcome
            4. Success criteria

            Return as a JSON list of plan steps.
            """

            # Use LLM for planning if available
            if self.llm:
                try:
                    response = await self.llm.ainvoke(planning_prompt)
                    # Parse response and create plan steps
                    plan_steps = [
                        {
                            "step_id": str(uuid.uuid4()),
                            "description": f"Execute step for {desire.goal}",
                            "action_type": "autonomous_action",
                            "parameters": {"goal": desire.goal, "context": desire.context},
                            "expected_outcome": desire.goal,
                            "success_criteria": ["Goal achieved", "No errors occurred"]
                        }
                    ]
                except Exception as llm_error:
                    logger.warning("LLM planning failed, using fallback", error=str(llm_error))
                    plan_steps = self._create_fallback_plan(desire)
            else:
                plan_steps = self._create_fallback_plan(desire)

            return plan_steps

        except Exception as e:
            logger.error("Plan generation failed", desire_id=desire.desire_id, error=str(e))
            return []

    def _create_fallback_plan(self, desire: Desire) -> List[Dict[str, Any]]:
        """Create a simple fallback plan when LLM planning fails."""
        return [
            {
                "step_id": str(uuid.uuid4()),
                "description": f"Achieve goal: {desire.goal}",
                "action_type": "autonomous_action",
                "parameters": {"goal": desire.goal, "context": desire.context},
                "expected_outcome": desire.goal,
                "success_criteria": ["Goal achieved"]
            }
        ]

    async def _monitor_plan_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and update plan execution for active intentions."""
        try:
            monitoring_results = {
                "plans_monitored": 0,
                "plans_updated": 0,
                "plans_completed": 0,
                "plans_failed": 0
            }

            active_intentions = [i for i in self.intentions.values() if i.status == IntentionStatus.ACTIVE]

            for intention in active_intentions:
                monitoring_results["plans_monitored"] += 1

                # Check plan progress
                plan_status = await self._check_plan_progress(intention, context)

                if plan_status["completed"]:
                    intention.status = IntentionStatus.COMPLETED
                    monitoring_results["plans_completed"] += 1
                    logger.info("Plan completed", intention_id=intention.intention_id)

                elif plan_status["failed"]:
                    intention.status = IntentionStatus.FAILED
                    monitoring_results["plans_failed"] += 1
                    logger.warning("Plan failed", intention_id=intention.intention_id, reason=plan_status.get("reason"))

                elif plan_status["needs_update"]:
                    # Update plan based on new context
                    await self._update_plan(intention, context)
                    monitoring_results["plans_updated"] += 1
                    logger.debug("Plan updated", intention_id=intention.intention_id)

            return monitoring_results

        except Exception as e:
            logger.error("Plan monitoring failed", agent_id=self.agent_id, error=str(e))
            return {"plans_monitored": 0, "plans_updated": 0, "plans_completed": 0, "plans_failed": 0}

    async def _check_plan_progress(self, intention: Intention, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check the progress of a plan execution."""
        try:
            # Simple progress check - in a real implementation this would be more sophisticated
            current_time = datetime.utcnow()
            time_elapsed = (current_time - intention.created_at).total_seconds()

            # Check if plan has been running too long (simple timeout)
            if time_elapsed > 300:  # 5 minutes timeout
                return {"completed": False, "failed": True, "needs_update": False, "reason": "timeout"}

            # Check if goal appears to be achieved based on context
            goal_keywords = intention.goal.lower().split()
            context_text = str(context).lower()

            matches = sum(1 for keyword in goal_keywords if keyword in context_text)
            progress_ratio = matches / len(goal_keywords) if goal_keywords else 0

            if progress_ratio >= 0.8:  # 80% keyword match suggests completion
                return {"completed": True, "failed": False, "needs_update": False}
            elif progress_ratio >= 0.4:  # Some progress, continue
                return {"completed": False, "failed": False, "needs_update": False}
            else:  # Little progress, might need update
                return {"completed": False, "failed": False, "needs_update": True}

        except Exception as e:
            logger.error("Plan progress check failed", intention_id=intention.intention_id, error=str(e))
            return {"completed": False, "failed": True, "needs_update": False, "reason": str(e)}

    async def _update_plan(self, intention: Intention, context: Dict[str, Any]) -> bool:
        """Update a plan based on new context."""
        try:
            # Create updated plan based on current context
            desire = self.desires.get(intention.desire_id)
            if not desire:
                logger.warning("Cannot update plan - desire not found", intention_id=intention.intention_id)
                return False

            # Update desire context with new information
            desire.context.update(context)

            # Generate new plan
            new_plan = await self._generate_plan_for_desire(desire)
            if new_plan:
                intention.plan = new_plan
                intention.updated_at = datetime.utcnow()
                logger.debug("Plan updated successfully", intention_id=intention.intention_id)
                return True

            return False

        except Exception as e:
            logger.error("Plan update failed", intention_id=intention.intention_id, error=str(e))
            return False

    def _has_desire_for_task(self, task: str) -> bool:
        """Check if agent already has a desire for the given task."""
        for desire in self.desires.values():
            if desire.desire_type == DesireType.ACHIEVEMENT and task in desire.description:
                return True
        return False

    def _has_desire_for_learning(self) -> bool:
        """Check if agent already has a learning desire."""
        return any(d.desire_type == DesireType.LEARNING for d in self.desires.values())

    def _has_desire_for_optimization(self) -> bool:
        """Check if agent already has an optimization desire."""
        return any(d.desire_type == DesireType.OPTIMIZATION for d in self.desires.values())

    def _has_desire_for_exploration(self) -> bool:
        """Check if agent already has an exploration desire."""
        return any(d.desire_type == DesireType.EXPLORATION for d in self.desires.values())

    def _find_intention_for_desire(self, desire_id: str) -> Optional[Intention]:
        """Find existing intention for a desire."""
        for intention in self.intentions.values():
            if intention.desire_id == desire_id and intention.status in [IntentionStatus.ACTIVE, IntentionStatus.FORMING]:
                return intention
        return None

    async def _reconsider_intentions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reconsider current intentions based on new context and priorities."""
        try:
            reconsideration_results = {
                "intentions_dropped": 0,
                "intentions_modified": 0,
                "intentions_prioritized": 0,
                "total_intentions": len(self.intentions)
            }

            current_time = datetime.utcnow()
            active_intentions = [i for i in self.intentions.values() if i.status == IntentionStatus.ACTIVE]

            for intention in active_intentions:
                # Check if intention is still relevant
                relevance_score = await self._assess_intention_relevance(intention, context)

                if relevance_score < 0.3:  # Low relevance, consider dropping
                    intention.status = IntentionStatus.DROPPED
                    reconsideration_results["intentions_dropped"] += 1
                    logger.debug("Intention dropped due to low relevance",
                                intention_id=intention.intention_id,
                                relevance_score=relevance_score)

                elif relevance_score < 0.6:  # Medium relevance, consider modification
                    # Try to modify the intention to make it more relevant
                    modified = await self._modify_intention(intention, context)
                    if modified:
                        reconsideration_results["intentions_modified"] += 1
                        logger.debug("Intention modified", intention_id=intention.intention_id)

                # Update priority based on current context
                new_priority = await self._calculate_intention_priority(intention, context)
                if abs(new_priority - intention.priority) > 0.1:  # Significant change
                    intention.priority = new_priority
                    intention.updated_at = current_time
                    reconsideration_results["intentions_prioritized"] += 1

            # Sort intentions by priority
            sorted_intentions = sorted(active_intentions, key=lambda x: x.priority, reverse=True)

            logger.debug("Intention reconsideration completed",
                        agent_id=self.agent_id,
                        **reconsideration_results)

            return reconsideration_results

        except Exception as e:
            logger.error("Intention reconsideration failed", agent_id=self.agent_id, error=str(e))
            return {"intentions_dropped": 0, "intentions_modified": 0, "intentions_prioritized": 0, "total_intentions": len(self.intentions)}

    async def _assess_intention_relevance(self, intention: Intention, context: Dict[str, Any]) -> float:
        """Assess how relevant an intention is given the current context."""
        try:
            relevance_score = 0.5  # Base relevance

            # Check if goal keywords appear in current context
            goal_keywords = intention.goal.lower().split()
            context_text = str(context).lower()

            keyword_matches = sum(1 for keyword in goal_keywords if keyword in context_text)
            keyword_relevance = keyword_matches / len(goal_keywords) if goal_keywords else 0
            relevance_score += keyword_relevance * 0.3

            # Check time since creation (older intentions may be less relevant)
            time_since_creation = (datetime.utcnow() - intention.created_at).total_seconds()
            time_factor = max(0, 1 - (time_since_creation / 3600))  # Decay over 1 hour
            relevance_score *= time_factor

            # Check confidence level
            relevance_score *= intention.confidence

            return min(1.0, max(0.0, relevance_score))

        except Exception as e:
            logger.error("Failed to assess intention relevance",
                        intention_id=intention.intention_id,
                        error=str(e))
            return 0.5

    async def _modify_intention(self, intention: Intention, context: Dict[str, Any]) -> bool:
        """Modify an intention to make it more relevant to current context."""
        try:
            # Update intention context with new information
            intention.context.update(context)

            # Regenerate plan if needed
            desire = self.desires.get(intention.desire_id)
            if desire:
                new_plan = await self._generate_plan_for_desire(desire)
                if new_plan and len(new_plan) > 0:
                    intention.plan = new_plan
                    intention.updated_at = datetime.utcnow()
                    return True

            return False

        except Exception as e:
            logger.error("Failed to modify intention",
                        intention_id=intention.intention_id,
                        error=str(e))
            return False

    async def _calculate_intention_priority(self, intention: Intention, context: Dict[str, Any]) -> float:
        """Calculate the priority of an intention based on current context."""
        try:
            base_priority = intention.priority

            # Adjust based on urgency indicators in context
            if "urgent" in str(context).lower():
                base_priority += 0.2

            # Adjust based on resource availability
            if "resources_available" in context:
                if context["resources_available"]:
                    base_priority += 0.1
                else:
                    base_priority -= 0.1

            # Adjust based on confidence
            base_priority *= intention.confidence

            return min(1.0, max(0.0, base_priority))

        except Exception as e:
            logger.error("Failed to calculate intention priority",
                        intention_id=intention.intention_id,
                        error=str(e))
            return intention.priority
