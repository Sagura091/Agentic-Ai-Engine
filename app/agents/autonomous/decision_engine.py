"""
Autonomous Decision Engine for Agentic AI Systems.

This module implements sophisticated decision-making capabilities for autonomous
agents, including confidence-based reasoning, multi-criteria evaluation, and
adaptive decision strategies.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agents.autonomous.autonomous_agent import AutonomousDecision, AutonomousAgentConfig
from app.agents.autonomous.persistent_memory import PersistentMemorySystem, MemoryType, MemoryImportance

logger = structlog.get_logger(__name__)


class DecisionCriteria(BaseModel):
    """Criteria for autonomous decision evaluation."""
    name: str = Field(..., description="Criteria name")
    weight: float = Field(..., ge=0.0, le=1.0, description="Criteria weight")
    description: str = Field(..., description="Criteria description")
    evaluation_method: str = Field(..., description="How to evaluate this criteria")


class DecisionOption(BaseModel):
    """Option for autonomous decision-making."""
    option_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action: str = Field(..., description="Action to take")
    type: str = Field(..., description="Type of action")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    expected_outcome: Dict[str, Any] = Field(default_factory=dict, description="Expected results")
    risk_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk assessment")
    resource_cost: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    confidence_estimate: float = Field(default=0.5, ge=0.0, le=1.0, description="Success probability")


class DecisionResult(BaseModel):
    """Result of a decision-making process."""
    selected_option: DecisionOption = Field(..., description="The chosen decision option")
    all_options: List[DecisionOption] = Field(default_factory=list, description="All options considered")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the decision")
    reasoning: List[str] = Field(default_factory=list, description="Reasoning chain for the decision")
    expected_outcome: Dict[str, Any] = Field(default_factory=dict, description="Expected results")
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique decision ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the decision was made")


class AutonomousDecisionEngine:
    """
    Advanced decision engine for autonomous agents.
    
    Implements sophisticated decision-making algorithms including:
    - Multi-criteria decision analysis
    - Confidence-based reasoning
    - Risk assessment and mitigation
    - Learning from decision outcomes
    """
    
    def __init__(self, config: AutonomousAgentConfig, llm: BaseLanguageModel, memory_system: Optional[PersistentMemorySystem] = None):
        """
        Initialize the decision engine.

        Args:
            config: Autonomous agent configuration
            llm: Language model for decision reasoning
            memory_system: Optional memory system for learning from past decisions
        """
        self.config = config
        self.llm = llm
        self.memory_system = memory_system
        self.decision_history: List[AutonomousDecision] = []
        self.decision_patterns: Dict[str, Any] = {}
        self.confidence_calibration: Dict[str, float] = {}

        # Default decision criteria
        self.default_criteria = [
            DecisionCriteria(
                name="task_completion_necessity",
                weight=0.5,  # INCREASED: Prioritize task completion
                description="How necessary this action is for task completion",
                evaluation_method="task_completion_analysis"
            ),
            DecisionCriteria(
                name="success_probability",
                weight=0.3,  # INCREASED: Tool confidence matters more
                description="Likelihood of successful execution",
                evaluation_method="historical_success_rate"
            ),
            DecisionCriteria(
                name="goal_alignment",
                weight=0.1,  # REDUCED: Less important when no goals
                description="How well the option aligns with current goals",
                evaluation_method="goal_similarity"
            ),
            DecisionCriteria(
                name="resource_efficiency",
                weight=0.05,  # REDUCED: Less important
                description="Efficient use of available resources",
                evaluation_method="cost_benefit_analysis"
            ),
            DecisionCriteria(
                name="risk_level",
                weight=0.03,  # REDUCED: Less important
                description="Risk associated with the action",
                evaluation_method="risk_assessment"
            ),
            DecisionCriteria(
                name="learning_value",
                weight=0.02,  # REDUCED: Less important
                description="Potential learning and improvement value",
                evaluation_method="learning_potential"
            )
        ]

        logger.info("Autonomous decision engine initialized",
                   criteria_count=len(self.default_criteria),
                   memory_enabled=memory_system is not None)
    
    async def make_autonomous_decision(
        self,
        context: Dict[str, Any],
        confidence_threshold: float = 0.6,
        custom_criteria: Optional[List[DecisionCriteria]] = None
    ) -> AutonomousDecision:
        """
        Make an autonomous decision based on context and criteria.

        PRODUCTION IMPLEMENTATION: Retrieves relevant past decisions from memory
        to inform current decision-making with learned patterns and outcomes.

        Args:
            context: Decision context including goals, constraints, and available actions
            confidence_threshold: Minimum confidence required for decision
            custom_criteria: Custom decision criteria (optional)

        Returns:
            AutonomousDecision with chosen option and reasoning
        """
        try:
            # CRITICAL: Retrieve relevant past decisions from memory
            past_decisions = await self._retrieve_relevant_past_decisions(context)

            # Enrich context with learned patterns from past decisions
            if past_decisions:
                context["past_decision_insights"] = self._extract_decision_insights(past_decisions)
                logger.debug(
                    "Retrieved past decisions for context",
                    past_decisions_count=len(past_decisions),
                    insights_extracted=len(context.get("past_decision_insights", []))
                )

            # Generate decision options
            options = await self._generate_decision_options(context)

            if not options:
                # Create default "no action" option
                options = [DecisionOption(
                    action="no_action",
                    type="passive",
                    expected_outcome={"status": "maintain_current_state"},
                    confidence_estimate=0.8
                )]

            # Use custom or default criteria
            criteria = custom_criteria or self.default_criteria

            # Evaluate each option (now with past decision insights)
            evaluated_options = []
            for option in options:
                evaluation = await self._evaluate_option(option, context, criteria)
                evaluated_options.append((option, evaluation))

            # Select best option
            best_option, best_evaluation = self._select_best_option(evaluated_options)

            # Calculate overall confidence
            confidence = self._calculate_decision_confidence(best_evaluation, context)

            # Generate reasoning chain
            reasoning = await self._generate_reasoning_chain(
                best_option, best_evaluation, context, criteria
            )

            # Create decision record
            decision = AutonomousDecision(
                decision_type="autonomous_action",
                context=context,
                options_considered=[opt.dict() for opt, _ in evaluated_options],
                chosen_option=best_option.dict(),
                confidence=confidence,
                reasoning=reasoning,
                expected_outcome=best_option.expected_outcome
            )

            # Record decision for learning
            self.decision_history.append(decision)
            self._update_decision_patterns(decision)

            # CRITICAL: Store this decision in memory for future learning
            await self._store_decision_in_memory(decision, context, past_decisions)
            
            logger.info(
                "Autonomous decision made",
                decision_id=decision.decision_id,
                chosen_action=best_option.action,
                confidence=confidence,
                options_evaluated=len(options)
            )
            
            return decision
            
        except Exception as e:
            logger.error("Decision making failed", error=str(e))
            # Return safe default decision
            return AutonomousDecision(
                decision_type="error_fallback",
                context=context,
                options_considered=[],
                chosen_option={"action": "no_action", "type": "error_recovery"},
                confidence=0.1,
                reasoning=[f"Decision making failed: {str(e)}", "Defaulting to no action for safety"],
                expected_outcome={"status": "error_recovery"}
            )
    
    async def _generate_decision_options(self, context: Dict[str, Any]) -> List[DecisionOption]:
        """Generate possible decision options based on context."""
        options = []
        
        # Extract available actions from context
        available_tools = context.get("available_actions", {}).get("tools", [])
        current_goals = context.get("current_goals", [])
        constraints = context.get("constraints", [])
        
        # Generate tool-based options
        for tool in available_tools:
            if isinstance(tool, dict):
                tool_name = tool.get("name", "unknown")
                tool_params = tool.get("suggested_params", {})

                # Higher confidence for tools when task requires execution
                current_task = context.get("current_task", "").lower()
                execution_keywords = ["generate", "create", "build", "make", "produce", "develop"]
                output_keywords = ["excel", "spreadsheet", "document", "report", "file"]
                creative_keywords = ["meme", "roast", "screen", "capture", "chaos", "remix", "viral", "music", "lyric", "social", "unexpected", "creative"]
                tool_action_keywords = ["use", "execute", "run", "perform", "do", "action"]

                base_confidence = tool.get("confidence", 0.5)

                # Check for creative chaos tasks that require tool execution
                is_creative_task = any(keyword in current_task for keyword in creative_keywords)
                is_execution_task = any(keyword in current_task for keyword in execution_keywords)
                is_output_task = any(keyword in current_task for keyword in output_keywords)
                is_tool_action_task = any(keyword in current_task for keyword in tool_action_keywords)

                if is_creative_task or is_execution_task or is_output_task or is_tool_action_task:
                    # Boost tool confidence significantly for tasks that require tool usage
                    tool_confidence = min(0.95, base_confidence + 0.4)
                else:
                    tool_confidence = base_confidence

                option = DecisionOption(
                    action=f"use_tool_{tool_name}",
                    type="tool_use",
                    parameters={"tool": tool_name, "args": tool_params},
                    expected_outcome={"tool_result": f"result_from_{tool_name}"},
                    confidence_estimate=tool_confidence
                )
                options.append(option)
        
        # Generate goal-based options
        for goal in current_goals:
            if isinstance(goal, dict):
                goal_action = goal.get("action", "pursue_goal")
                
                option = DecisionOption(
                    action=goal_action,
                    type="goal_pursuit",
                    parameters={"goal": goal},
                    expected_outcome={"goal_progress": goal.get("expected_progress", 0.1)},
                    confidence_estimate=goal.get("feasibility", 0.5)
                )
                options.append(option)
        
        # Generate reasoning-based options - with reduced confidence after multiple iterations
        if context.get("requires_reasoning", True):
            # Check for excessive reasoning - reduce confidence if too much reasoning already done
            autonomous_reasoning = context.get("autonomous_reasoning", "")
            reasoning_iterations = autonomous_reasoning.lower().count("reasoning") + autonomous_reasoning.lower().count("analyze")

            # Check if task requires execution from the start
            current_task = context.get("current_task", "").lower()
            execution_keywords = ["generate", "create", "build", "make", "produce", "develop"]
            output_keywords = ["excel", "spreadsheet", "document", "report", "file"]
            task_requires_execution = any(keyword in current_task for keyword in execution_keywords)
            task_requires_output = any(keyword in current_task for keyword in output_keywords)

            # Start with very low reasoning confidence if task requires execution/output
            if task_requires_execution or task_requires_output:
                base_reasoning_confidence = 0.2  # Much lower for execution tasks
            else:
                base_reasoning_confidence = 0.7  # Normal for analysis tasks

            # Reduce reasoning confidence based on iteration count
            confidence_reduction = min(0.5, reasoning_iterations * 0.15)
            reasoning_confidence = max(0.1, base_reasoning_confidence - confidence_reduction)

            # Force extremely low confidence if task requires execution and we've reasoned at all
            if (task_requires_execution or task_requires_output) and reasoning_iterations > 0:
                reasoning_confidence = 0.05  # Extremely low confidence to force tool usage

            reasoning_option = DecisionOption(
                action="autonomous_reasoning",
                type="reasoning",
                parameters={"prompt": "Analyze current situation and determine best course of action"},
                expected_outcome={"reasoning_result": "analysis_and_recommendations"},
                confidence_estimate=reasoning_confidence
            )
            options.append(reasoning_option)
        
        # Generate exploration options (for learning)
        if self.config.learning_mode != "disabled":
            exploration_option = DecisionOption(
                action="explore_new_approach",
                type="exploration",
                parameters={"exploration_type": "random_action"},
                expected_outcome={"learning_value": "new_experience"},
                confidence_estimate=0.3,
                risk_level=0.7
            )
            options.append(exploration_option)
        
        logger.debug(f"Generated {len(options)} decision options", options_count=len(options))
        return options
    
    async def _evaluate_option(
        self,
        option: DecisionOption,
        context: Dict[str, Any],
        criteria: List[DecisionCriteria]
    ) -> Dict[str, float]:
        """Evaluate a decision option against criteria."""
        evaluation = {}
        
        for criterion in criteria:
            score = await self._evaluate_criterion(option, context, criterion)
            evaluation[criterion.name] = score
        
        # Calculate weighted score
        weighted_score = sum(
            evaluation[criterion.name] * criterion.weight
            for criterion in criteria
        )
        evaluation["weighted_score"] = weighted_score
        
        return evaluation
    
    async def _evaluate_criterion(
        self,
        option: DecisionOption,
        context: Dict[str, Any],
        criterion: DecisionCriteria
    ) -> float:
        """Evaluate a single criterion for an option."""
        try:
            if criterion.evaluation_method == "task_completion_analysis":
                return self._evaluate_task_completion_necessity(option, context)
            elif criterion.evaluation_method == "goal_similarity":
                return self._evaluate_goal_alignment(option, context)
            elif criterion.evaluation_method == "historical_success_rate":
                return self._evaluate_success_probability(option)
            elif criterion.evaluation_method == "cost_benefit_analysis":
                return self._evaluate_resource_efficiency(option, context)
            elif criterion.evaluation_method == "risk_assessment":
                return 1.0 - option.risk_level  # Lower risk = higher score
            elif criterion.evaluation_method == "learning_potential":
                return self._evaluate_learning_value(option, context)
            else:
                # Default evaluation
                return option.confidence_estimate
                
        except Exception as e:
            logger.warning(f"Criterion evaluation failed for {criterion.name}", error=str(e))
            return 0.5  # Neutral score on error

    def _evaluate_task_completion_necessity(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """Evaluate how necessary this action is for completing the current task."""
        try:
            # Get current task context
            current_task = context.get("current_task", "").lower()
            autonomous_reasoning = context.get("autonomous_reasoning", "").lower()

            # High necessity for tool usage when task requires generation/creation
            execution_keywords = ["generate", "create", "build", "make", "produce", "develop", "write", "export"]
            task_requires_execution = any(keyword in current_task for keyword in execution_keywords)

            if option.type == "tool_use":
                # Tool usage is highly necessary for execution tasks
                if task_requires_execution:
                    return 0.95

                # Check if task mentions specific outputs (Excel, spreadsheet, etc.)
                output_keywords = ["excel", "spreadsheet", "document", "report", "file", "analysis"]
                if any(keyword in current_task for keyword in output_keywords):
                    return 0.9

                # Medium necessity for analysis tasks
                analysis_keywords = ["analyze", "review", "examine", "study", "investigate"]
                if any(keyword in current_task for keyword in analysis_keywords):
                    return 0.7

                return 0.6  # Default tool necessity

            elif option.type == "reasoning":
                # Reasoning is less necessary if we've already reasoned extensively
                reasoning_count = autonomous_reasoning.count("reasoning") + autonomous_reasoning.count("analyze")

                # High necessity for initial reasoning
                if reasoning_count == 0:
                    return 0.8

                # Decreasing necessity for repeated reasoning
                necessity = max(0.2, 0.8 - (reasoning_count * 0.2))

                # Very low necessity if task clearly requires execution
                if task_requires_execution and reasoning_count > 1:
                    return 0.1

                return necessity

            else:
                # Other action types have medium necessity
                return 0.5

        except Exception as e:
            logger.warning(f"Task completion necessity evaluation failed", error=str(e))
            return 0.5

    def _evaluate_goal_alignment(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """Evaluate how well an option aligns with current goals."""
        current_goals = context.get("current_goals", [])
        if not current_goals:
            return 0.5
        
        # Simple alignment scoring based on action type and goal types
        option_type = option.type
        goal_types = [goal.get("type", "unknown") for goal in current_goals if isinstance(goal, dict)]
        
        alignment_score = 0.0
        for goal_type in goal_types:
            if option_type == goal_type:
                alignment_score += 1.0
            elif option_type in ["tool_use", "reasoning"] and goal_type in ["problem_solving", "analysis"]:
                alignment_score += 0.8
            elif option_type == "exploration" and goal_type == "learning":
                alignment_score += 0.9
            else:
                alignment_score += 0.3  # Some alignment for any action
        
        return min(1.0, alignment_score / len(goal_types))
    
    def _evaluate_success_probability(self, option: DecisionOption) -> float:
        """Evaluate success probability based on historical data."""
        # Use option's confidence estimate and historical patterns
        base_confidence = option.confidence_estimate
        
        # Adjust based on historical success for similar actions
        action_pattern = self.decision_patterns.get(option.action, {})
        historical_success = action_pattern.get("success_rate", 0.5)
        
        # Weighted combination
        adjusted_confidence = (base_confidence * 0.7) + (historical_success * 0.3)
        return min(1.0, adjusted_confidence)
    
    def _evaluate_resource_efficiency(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """Evaluate resource efficiency of an option."""
        resource_cost = option.resource_cost
        available_resources = context.get("available_resources", {})
        
        if not resource_cost:
            return 0.8  # No cost is efficient
        
        # Simple efficiency calculation
        efficiency_score = 1.0
        for resource, cost in resource_cost.items():
            available = available_resources.get(resource, 1.0)
            if available > 0:
                utilization = cost / available
                efficiency_score *= max(0.0, 1.0 - utilization)
        
        return efficiency_score
    
    def _evaluate_learning_value(self, option: DecisionOption, context: Dict[str, Any]) -> float:
        """Evaluate potential learning value of an option."""
        if option.type == "exploration":
            return 0.9  # High learning value for exploration
        elif option.type == "tool_use":
            # Learning value based on tool novelty
            tool_name = option.parameters.get("tool", "")
            usage_count = self.decision_patterns.get(f"tool_{tool_name}", {}).get("usage_count", 0)
            return max(0.1, 1.0 - (usage_count * 0.1))  # Diminishing learning value
        elif option.type == "reasoning":
            return 0.6  # Moderate learning value for reasoning
        else:
            return 0.3  # Low learning value for routine actions

    def _select_best_option(self, evaluated_options: List[Tuple[DecisionOption, Dict[str, float]]]) -> Tuple[DecisionOption, Dict[str, float]]:
        """Select the best option from evaluated options."""
        if not evaluated_options:
            # Create a default no-action option
            default_option = DecisionOption(
                action="no_action",
                type="safety",
                parameters={},
                confidence_estimate=0.1,
                expected_outcome={"status": "no_action_taken"}
            )
            default_evaluation = {"weighted_score": 0.1}
            return default_option, default_evaluation

        # Sort by weighted score (highest first)
        sorted_options = sorted(evaluated_options, key=lambda x: x[1].get("weighted_score", 0.0), reverse=True)

        # Return the best option
        best_option, best_evaluation = sorted_options[0]

        logger.debug(f"Selected best option: {best_option.action} with score {best_evaluation.get('weighted_score', 0.0)}")
        return best_option, best_evaluation

    def _calculate_decision_confidence(self, evaluation: Dict[str, float], context: Dict[str, Any]) -> float:
        """Calculate overall confidence in the decision."""
        base_confidence = evaluation.get("weighted_score", 0.5)

        # Adjust confidence based on context factors
        context_factors = {
            "goal_clarity": len(context.get("current_goals", [])) > 0,
            "resource_availability": len(context.get("available_resources", {})) > 0,
            "historical_success": len(self.decision_history) > 0
        }

        # Boost confidence if context is favorable
        confidence_boost = sum(0.1 for factor in context_factors.values() if factor)
        adjusted_confidence = min(1.0, base_confidence + confidence_boost)

        return adjusted_confidence

    async def _generate_reasoning_chain(
        self,
        option: DecisionOption,
        evaluation: Dict[str, float],
        context: Dict[str, Any],
        criteria: List[DecisionCriteria]
    ) -> List[str]:
        """Generate reasoning chain for the decision."""
        reasoning = []

        # Add context analysis
        reasoning.append(f"Analyzed {len(context.get('current_goals', []))} current goals")
        reasoning.append(f"Evaluated {len(criteria)} decision criteria")

        # Add option analysis
        reasoning.append(f"Selected action: {option.action} (type: {option.type})")
        reasoning.append(f"Confidence estimate: {option.confidence_estimate:.2f}")
        reasoning.append(f"Weighted evaluation score: {evaluation.get('weighted_score', 0.0):.2f}")

        # Add specific reasoning based on option type
        if option.type == "tool_use":
            tool_name = option.parameters.get("tool", "unknown")
            reasoning.append(f"Tool usage decision: {tool_name}")
            reasoning.append("Expected to advance current goals through tool execution")
        elif option.type == "exploration":
            reasoning.append("Exploration decision to gather new information")
            reasoning.append("High learning value for future decision making")
        elif option.type == "reasoning":
            reasoning.append("Reasoning-focused decision to analyze current situation")
            reasoning.append("Will improve understanding before taking action")

        return reasoning

    async def _retrieve_relevant_past_decisions(self, context: Dict[str, Any]) -> List[Any]:
        """
        Retrieve relevant past decisions from memory to inform current decision.

        PRODUCTION IMPLEMENTATION: Queries memory system for similar past decisions
        based on context, goals, and decision type.

        Args:
            context: Current decision context

        Returns:
            List of relevant past decision memories
        """
        if not self.memory_system:
            return []

        try:
            # Build query from context
            query_parts = []

            if "current_task" in context:
                query_parts.append(f"decision about {context['current_task']}")

            if "active_goals" in context and context["active_goals"]:
                goals_str = ", ".join(str(g) for g in context["active_goals"][:3])
                query_parts.append(f"goals: {goals_str}")

            if "available_actions" in context:
                actions_str = ", ".join(context["available_actions"][:5])
                query_parts.append(f"actions: {actions_str}")

            query = " | ".join(query_parts) if query_parts else "autonomous decision"

            # Retrieve relevant decision memories
            past_decisions = await self.memory_system.retrieve_relevant_memories(
                query=query,
                memory_types=[MemoryType.EPISODIC, MemoryType.PROCEDURAL],
                limit=5
            )

            logger.debug(
                "Retrieved past decisions from memory",
                query=query[:100],
                memories_found=len(past_decisions)
            )

            return past_decisions

        except Exception as e:
            logger.warning(f"Failed to retrieve past decisions from memory: {e}")
            return []

    def _extract_decision_insights(self, past_decisions: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract actionable insights from past decisions.

        PRODUCTION IMPLEMENTATION: Analyzes past decisions to identify patterns,
        successful strategies, and lessons learned.

        Args:
            past_decisions: List of past decision memories

        Returns:
            List of insights extracted from past decisions
        """
        insights = []

        for memory in past_decisions:
            try:
                content = memory.content if hasattr(memory, 'content') else str(memory)
                metadata = memory.metadata if hasattr(memory, 'metadata') else {}

                insight = {
                    "content": content[:200],
                    "success": metadata.get("success", None),
                    "confidence": metadata.get("confidence", 0.5),
                    "outcome": metadata.get("outcome", "unknown"),
                    "timestamp": memory.created_at if hasattr(memory, 'created_at') else None
                }

                # Extract specific lessons
                if metadata.get("success") is True:
                    insight["lesson"] = "successful_pattern"
                elif metadata.get("success") is False:
                    insight["lesson"] = "avoid_pattern"
                else:
                    insight["lesson"] = "informational"

                insights.append(insight)

            except Exception as e:
                logger.debug(f"Failed to extract insight from memory: {e}")
                continue

        return insights

    async def _store_decision_in_memory(
        self,
        decision: AutonomousDecision,
        context: Dict[str, Any],
        past_decisions: List[Any]
    ) -> None:
        """
        Store decision in memory for future learning.

        PRODUCTION IMPLEMENTATION: Stores decision with rich metadata including
        context, reasoning, and links to past decisions for pattern learning.

        Args:
            decision: The decision that was made
            context: Decision context
            past_decisions: Past decisions that informed this decision
        """
        if not self.memory_system:
            return

        try:
            # Build memory content
            content_parts = [
                f"Decision: {decision.chosen_option.get('action', 'unknown')}",
                f"Type: {decision.chosen_option.get('type', 'unknown')}",
                f"Confidence: {decision.confidence:.2f}"
            ]

            if decision.reasoning:
                reasoning_summary = " | ".join(decision.reasoning[:2])
                content_parts.append(f"Reasoning: {reasoning_summary}")

            content = " | ".join(content_parts)

            # Determine importance based on confidence and context
            if decision.confidence >= 0.8:
                importance = MemoryImportance.HIGH
            elif decision.confidence >= 0.6:
                importance = MemoryImportance.MEDIUM
            else:
                importance = MemoryImportance.LOW

            # Build metadata
            metadata = {
                "decision_id": decision.decision_id,
                "decision_type": decision.decision_type,
                "confidence": decision.confidence,
                "options_considered": len(decision.options_considered),
                "chosen_action": decision.chosen_option.get("action"),
                "chosen_type": decision.chosen_option.get("type"),
                "expected_outcome": decision.expected_outcome,
                "context_summary": {
                    "has_goals": "active_goals" in context,
                    "has_actions": "available_actions" in context,
                    "has_task": "current_task" in context
                },
                "informed_by_past": len(past_decisions) > 0,
                "past_decisions_count": len(past_decisions),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Store as procedural memory (decision-making skill)
            await self.memory_system.store_memory(
                content=content,
                memory_type=MemoryType.PROCEDURAL,
                importance=importance,
                metadata=metadata,
                tags={"decision", "autonomous", decision.chosen_option.get("type", "unknown")},
                emotional_valence=0.3  # Slightly positive for taking action
            )

            logger.debug(
                "Stored decision in memory",
                decision_id=decision.decision_id,
                importance=importance.value,
                memory_type="procedural"
            )

        except Exception as e:
            logger.warning(f"Failed to store decision in memory: {e}")

    def _update_decision_patterns(self, decision: AutonomousDecision) -> None:
        """Update decision patterns based on the made decision."""
        try:
            # Extract pattern key from decision
            pattern_key = f"{decision.chosen_option.get('type', 'unknown')}_{decision.chosen_option.get('action', 'unknown')}"

            # Initialize pattern if not exists
            if pattern_key not in self.decision_patterns:
                self.decision_patterns[pattern_key] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "total_confidence": 0.0,
                    "last_used": None
                }

            # Update pattern statistics
            pattern = self.decision_patterns[pattern_key]
            pattern["usage_count"] += 1
            pattern["total_confidence"] += decision.confidence
            pattern["last_used"] = datetime.utcnow().isoformat()

            # Update tool-specific patterns if it's a tool use decision
            if decision.chosen_option.get("type") == "tool_use":
                tool_name = decision.chosen_option.get("parameters", {}).get("tool", "")
                if tool_name:
                    tool_pattern_key = f"tool_{tool_name}"
                    if tool_pattern_key not in self.decision_patterns:
                        self.decision_patterns[tool_pattern_key] = {
                            "usage_count": 0,
                            "success_count": 0,
                            "average_confidence": 0.0
                        }

                    tool_pattern = self.decision_patterns[tool_pattern_key]
                    tool_pattern["usage_count"] += 1
                    tool_pattern["average_confidence"] = (
                        (tool_pattern["average_confidence"] * (tool_pattern["usage_count"] - 1) + decision.confidence)
                        / tool_pattern["usage_count"]
                    )

            logger.debug(f"Updated decision patterns for {pattern_key}", usage_count=pattern["usage_count"])

        except Exception as e:
            logger.warning(f"Failed to update decision patterns: {str(e)}")
            # Don't raise exception to avoid breaking the decision flow
