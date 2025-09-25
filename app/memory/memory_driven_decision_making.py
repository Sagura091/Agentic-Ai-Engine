"""
Revolutionary Memory-Driven Decision Making for Agentic AI.

Implements memory-informed planning and decision systems that leverage
past experiences, learned patterns, and contextual memory for intelligent decision making.

Key Features:
- Memory-informed decision making
- Experience-based planning
- Context-aware decision trees
- Memory-driven strategy selection
- Outcome prediction from memory
- Decision confidence scoring
- Adaptive decision strategies
"""

import asyncio
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class DecisionType(str, Enum):
    """Types of decisions."""
    STRATEGIC = "strategic"          # High-level strategic decisions
    TACTICAL = "tactical"           # Mid-level tactical decisions
    OPERATIONAL = "operational"     # Low-level operational decisions
    REACTIVE = "reactive"           # Reactive decisions to events
    PREDICTIVE = "predictive"       # Predictive decisions based on forecasts


class DecisionContext(str, Enum):
    """Context for decision making."""
    PLANNING = "planning"           # Planning phase decisions
    EXECUTION = "execution"         # Execution phase decisions
    PROBLEM_SOLVING = "problem_solving"  # Problem-solving decisions
    LEARNING = "learning"           # Learning-related decisions
    ADAPTATION = "adaptation"       # Adaptation decisions
    EMERGENCY = "emergency"         # Emergency response decisions


class ConfidenceLevel(str, Enum):
    """Confidence levels for decisions."""
    VERY_LOW = "very_low"          # 0.0 - 0.2
    LOW = "low"                    # 0.2 - 0.4
    MEDIUM = "medium"              # 0.4 - 0.6
    HIGH = "high"                  # 0.6 - 0.8
    VERY_HIGH = "very_high"        # 0.8 - 1.0


@dataclass
class DecisionOption:
    """A decision option with memory-based evaluation."""
    option_id: str
    description: str
    option_type: str = "action"
    
    # Memory-based evaluation
    supporting_memories: List[str] = field(default_factory=list)
    conflicting_memories: List[str] = field(default_factory=list)
    similar_past_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scoring
    memory_support_score: float = 0.0
    predicted_outcome_score: float = 0.0
    confidence_score: float = 0.0
    risk_score: float = 0.0
    
    # Context
    prerequisites: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    resources_required: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal factors
    urgency: float = 0.5
    time_sensitivity: float = 0.5
    execution_time_estimate: float = 0.0


@dataclass
class DecisionContext:
    """Context for a decision-making scenario."""
    context_id: str
    decision_type: DecisionType
    context_type: DecisionContext
    description: str
    
    # Current situation
    current_state: Dict[str, Any] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    available_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal factors
    deadline: Optional[datetime] = None
    time_pressure: float = 0.0
    
    # Stakeholders and impact
    stakeholders: List[str] = field(default_factory=list)
    impact_scope: str = "local"  # local, regional, global
    
    # Memory relevance
    relevant_memory_types: List[str] = field(default_factory=list)
    memory_time_window: Optional[Tuple[datetime, datetime]] = None


@dataclass
class DecisionRecord:
    """Record of a decision made and its outcome."""
    decision_id: str
    agent_id: str
    decision_context: DecisionContext
    chosen_option: DecisionOption
    alternative_options: List[DecisionOption] = field(default_factory=list)
    
    # Decision process
    decision_strategy: str = "memory_driven"
    reasoning: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    # Outcome tracking
    actual_outcome: Optional[Dict[str, Any]] = None
    outcome_quality: Optional[float] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    # Temporal information
    decided_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Memory involvement
    memories_consulted: List[str] = field(default_factory=list)
    new_memories_created: List[str] = field(default_factory=list)


class MemoryDrivenDecisionMaking:
    """
    Revolutionary Memory-Driven Decision Making System.
    
    Leverages past experiences, learned patterns, and contextual memory
    to make intelligent, informed decisions.
    """
    
    def __init__(self, agent_id: str, memory_collection: Any, knowledge_graph: Optional[Any] = None):
        """Initialize memory-driven decision making system."""
        self.agent_id = agent_id
        self.memory_collection = memory_collection
        self.knowledge_graph = knowledge_graph
        
        # Decision storage
        self.decision_records: Dict[str, DecisionRecord] = {}
        self.decision_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Decision strategies
        self.decision_strategies = {
            "memory_driven": self._memory_driven_strategy,
            "experience_based": self._experience_based_strategy,
            "pattern_matching": self._pattern_matching_strategy,
            "outcome_prediction": self._outcome_prediction_strategy,
            "hybrid": self._hybrid_strategy
        }
        
        # Configuration
        self.config = {
            "memory_relevance_threshold": 0.6,
            "confidence_threshold": 0.7,
            "max_options_to_evaluate": 10,
            "memory_time_decay_days": 30,
            "pattern_matching_threshold": 0.8,
            "outcome_prediction_weight": 0.4,
            "memory_support_weight": 0.3,
            "risk_aversion_factor": 0.2,
            "enable_learning_from_outcomes": True
        }
        
        # Statistics
        self.stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "decision_accuracy": 0.0,
            "avg_confidence": 0.0,
            "strategy_usage": defaultdict(int),
            "decision_type_distribution": defaultdict(int),
            "memory_consultation_rate": 0.0
        }
        
        logger.info(f"Memory-Driven Decision Making initialized for agent {agent_id}")
    
    async def make_decision(
        self,
        decision_context: DecisionContext,
        options: List[DecisionOption],
        strategy: str = "hybrid"
    ) -> Tuple[DecisionOption, DecisionRecord]:
        """Make a memory-driven decision."""
        try:
            decision_id = f"decision_{int(time.time())}_{self.agent_id}"
            
            # Evaluate options using memory
            evaluated_options = await self._evaluate_options_with_memory(
                decision_context, options
            )
            
            # Apply decision strategy
            if strategy not in self.decision_strategies:
                strategy = "hybrid"
            
            chosen_option, reasoning, confidence = await self.decision_strategies[strategy](
                decision_context, evaluated_options
            )
            
            # Create decision record
            decision_record = DecisionRecord(
                decision_id=decision_id,
                agent_id=self.agent_id,
                decision_context=decision_context,
                chosen_option=chosen_option,
                alternative_options=[opt for opt in evaluated_options if opt.option_id != chosen_option.option_id],
                decision_strategy=strategy,
                reasoning=reasoning,
                confidence=self._score_to_confidence_level(confidence)
            )
            
            # Store decision record
            self.decision_records[decision_id] = decision_record
            
            # Update statistics
            self._update_decision_stats(decision_record)
            
            # Learn from decision context
            await self._learn_decision_patterns(decision_context, chosen_option)
            
            logger.info(
                "Memory-driven decision made",
                agent_id=self.agent_id,
                decision_id=decision_id,
                strategy=strategy,
                confidence=confidence,
                chosen_option=chosen_option.description
            )
            
            return chosen_option, decision_record
            
        except Exception as e:
            logger.error(f"Failed to make memory-driven decision: {e}")
            # Return first option as fallback
            fallback_option = options[0] if options else DecisionOption("fallback", "No action")
            fallback_record = DecisionRecord(
                decision_id="fallback",
                agent_id=self.agent_id,
                decision_context=decision_context,
                chosen_option=fallback_option,
                confidence=ConfidenceLevel.VERY_LOW
            )
            return fallback_option, fallback_record
    
    async def _evaluate_options_with_memory(
        self,
        context: DecisionContext,
        options: List[DecisionOption]
    ) -> List[DecisionOption]:
        """Evaluate decision options using memory."""
        try:
            evaluated_options = []
            
            for option in options:
                # Find relevant memories for this option
                relevant_memories = await self._find_relevant_memories(context, option)
                
                # Analyze memory support
                memory_support = await self._analyze_memory_support(option, relevant_memories)
                
                # Predict outcomes based on similar past experiences
                outcome_prediction = await self._predict_outcomes(option, relevant_memories)
                
                # Calculate confidence based on memory evidence
                confidence = await self._calculate_memory_confidence(option, relevant_memories)
                
                # Assess risks based on past failures
                risk_assessment = await self._assess_risks(option, relevant_memories)
                
                # Update option with memory-based evaluation
                option.supporting_memories = memory_support["supporting"]
                option.conflicting_memories = memory_support["conflicting"]
                option.similar_past_outcomes = outcome_prediction["similar_outcomes"]
                option.memory_support_score = memory_support["support_score"]
                option.predicted_outcome_score = outcome_prediction["outcome_score"]
                option.confidence_score = confidence
                option.risk_score = risk_assessment
                
                evaluated_options.append(option)
            
            return evaluated_options
            
        except Exception as e:
            logger.error(f"Failed to evaluate options with memory: {e}")
            return options
    
    async def _find_relevant_memories(
        self,
        context: DecisionContext,
        option: DecisionOption
    ) -> List[Any]:
        """Find memories relevant to the decision context and option."""
        try:
            relevant_memories = []
            
            # Search for memories related to similar decisions
            if hasattr(self.memory_collection, 'search_memories'):
                # Search by context keywords
                context_query = f"{context.description} {option.description}"
                context_memories = await self.memory_collection.search_memories(
                    context_query, limit=20
                )
                relevant_memories.extend(context_memories)
            
            # Search for memories with similar outcomes
            if hasattr(self.memory_collection, 'episodic_memories'):
                for memory in self.memory_collection.episodic_memories.values():
                    if self._is_memory_relevant(memory, context, option):
                        relevant_memories.append(memory)
            
            # Remove duplicates
            unique_memories = []
            seen_ids = set()
            for memory in relevant_memories:
                memory_id = getattr(memory, 'id', str(memory))
                if memory_id not in seen_ids:
                    unique_memories.append(memory)
                    seen_ids.add(memory_id)
            
            return unique_memories[:50]  # Limit to top 50 relevant memories
            
        except Exception as e:
            logger.error(f"Failed to find relevant memories: {e}")
            return []
    
    def _is_memory_relevant(self, memory: Any, context: DecisionContext, option: DecisionOption) -> bool:
        """Check if a memory is relevant to the decision context."""
        try:
            # Check memory content for relevant keywords
            memory_content = getattr(memory, 'content', '').lower()
            context_keywords = context.description.lower().split()
            option_keywords = option.description.lower().split()
            
            # Simple keyword matching
            relevant_keywords = context_keywords + option_keywords
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword in memory_content)
            
            # Consider memory relevant if it has multiple keyword matches
            return keyword_matches >= 2
            
        except Exception as e:
            logger.error(f"Failed to check memory relevance: {e}")
            return False
    
    async def _analyze_memory_support(
        self,
        option: DecisionOption,
        relevant_memories: List[Any]
    ) -> Dict[str, Any]:
        """Analyze how memories support or conflict with an option."""
        try:
            supporting_memories = []
            conflicting_memories = []
            support_score = 0.0
            
            for memory in relevant_memories:
                # Analyze memory sentiment/outcome towards similar actions
                memory_outcome = self._extract_memory_outcome(memory)
                
                if memory_outcome > 0.6:  # Positive outcome
                    supporting_memories.append(getattr(memory, 'id', str(memory)))
                    support_score += memory_outcome
                elif memory_outcome < 0.4:  # Negative outcome
                    conflicting_memories.append(getattr(memory, 'id', str(memory)))
                    support_score -= (1.0 - memory_outcome)
            
            # Normalize support score
            if relevant_memories:
                support_score = max(0.0, min(1.0, support_score / len(relevant_memories)))
            
            return {
                "supporting": supporting_memories,
                "conflicting": conflicting_memories,
                "support_score": support_score
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze memory support: {e}")
            return {"supporting": [], "conflicting": [], "support_score": 0.5}
    
    def _extract_memory_outcome(self, memory: Any) -> float:
        """Extract outcome quality from a memory."""
        try:
            # Look for outcome indicators in memory content
            content = getattr(memory, 'content', '').lower()
            
            # Simple sentiment analysis based on keywords
            positive_keywords = ['success', 'good', 'effective', 'worked', 'achieved', 'accomplished']
            negative_keywords = ['failed', 'bad', 'ineffective', 'problem', 'error', 'mistake']
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in content)
            negative_count = sum(1 for keyword in negative_keywords if keyword in content)
            
            # Calculate outcome score
            if positive_count + negative_count == 0:
                return 0.5  # Neutral
            
            outcome_score = positive_count / (positive_count + negative_count)
            return outcome_score
            
        except Exception as e:
            logger.error(f"Failed to extract memory outcome: {e}")
            return 0.5
    
    async def _predict_outcomes(
        self,
        option: DecisionOption,
        relevant_memories: List[Any]
    ) -> Dict[str, Any]:
        """Predict outcomes based on similar past experiences."""
        try:
            similar_outcomes = []
            outcome_scores = []
            
            for memory in relevant_memories:
                # Extract outcome from memory
                outcome_score = self._extract_memory_outcome(memory)
                outcome_scores.append(outcome_score)
                
                # Create outcome record
                similar_outcomes.append({
                    "memory_id": getattr(memory, 'id', str(memory)),
                    "outcome_score": outcome_score,
                    "context": getattr(memory, 'content', '')[:100]
                })
            
            # Calculate predicted outcome score
            if outcome_scores:
                predicted_score = np.mean(outcome_scores)
                
                # Weight recent memories more heavily
                if hasattr(relevant_memories[0], 'created_at'):
                    weighted_scores = []
                    current_time = datetime.now()
                    
                    for i, memory in enumerate(relevant_memories):
                        if hasattr(memory, 'created_at'):
                            age_days = (current_time - memory.created_at).total_seconds() / 86400
                            weight = math.exp(-age_days / self.config["memory_time_decay_days"])
                            weighted_scores.append(outcome_scores[i] * weight)
                        else:
                            weighted_scores.append(outcome_scores[i])
                    
                    if weighted_scores:
                        predicted_score = np.mean(weighted_scores)
            else:
                predicted_score = 0.5  # Neutral prediction
            
            return {
                "similar_outcomes": similar_outcomes,
                "outcome_score": predicted_score
            }
            
        except Exception as e:
            logger.error(f"Failed to predict outcomes: {e}")
            return {"similar_outcomes": [], "outcome_score": 0.5}
    
    async def _calculate_memory_confidence(
        self,
        option: DecisionOption,
        relevant_memories: List[Any]
    ) -> float:
        """Calculate confidence based on memory evidence."""
        try:
            if not relevant_memories:
                return 0.3  # Low confidence without memory evidence
            
            # Base confidence on number of relevant memories
            memory_count_factor = min(1.0, len(relevant_memories) / 10.0)
            
            # Factor in memory quality/importance
            memory_qualities = []
            for memory in relevant_memories:
                importance = getattr(memory, 'importance', 'medium')
                if hasattr(importance, 'value'):
                    importance = importance.value
                
                quality_map = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4, "temporary": 0.2}
                quality = quality_map.get(importance, 0.6)
                memory_qualities.append(quality)
            
            avg_quality = np.mean(memory_qualities) if memory_qualities else 0.6
            
            # Factor in consistency of memory evidence
            outcome_scores = [self._extract_memory_outcome(memory) for memory in relevant_memories]
            consistency = 1.0 - np.std(outcome_scores) if len(outcome_scores) > 1 else 1.0
            
            # Calculate overall confidence
            confidence = (memory_count_factor * 0.4 + avg_quality * 0.4 + consistency * 0.2)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate memory confidence: {e}")
            return 0.5
    
    async def _assess_risks(
        self,
        option: DecisionOption,
        relevant_memories: List[Any]
    ) -> float:
        """Assess risks based on past failures."""
        try:
            risk_indicators = 0
            total_memories = len(relevant_memories)
            
            if total_memories == 0:
                return 0.5  # Medium risk without evidence
            
            for memory in relevant_memories:
                outcome_score = self._extract_memory_outcome(memory)
                if outcome_score < 0.4:  # Poor outcome
                    risk_indicators += 1
            
            # Calculate risk score
            risk_score = risk_indicators / total_memories
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Failed to assess risks: {e}")
            return 0.5
    
    async def _memory_driven_strategy(
        self,
        context: DecisionContext,
        options: List[DecisionOption]
    ) -> Tuple[DecisionOption, str, float]:
        """Memory-driven decision strategy."""
        # Choose option with highest memory support
        best_option = max(options, key=lambda opt: opt.memory_support_score)
        
        reasoning = f"Selected based on memory support score of {best_option.memory_support_score:.2f}"
        confidence = best_option.confidence_score
        
        return best_option, reasoning, confidence
    
    async def _experience_based_strategy(
        self,
        context: DecisionContext,
        options: List[DecisionOption]
    ) -> Tuple[DecisionOption, str, float]:
        """Experience-based decision strategy."""
        # Choose option with best predicted outcomes
        best_option = max(options, key=lambda opt: opt.predicted_outcome_score)
        
        reasoning = f"Selected based on predicted outcome score of {best_option.predicted_outcome_score:.2f}"
        confidence = best_option.confidence_score
        
        return best_option, reasoning, confidence
    
    async def _pattern_matching_strategy(
        self,
        context: DecisionContext,
        options: List[DecisionOption]
    ) -> Tuple[DecisionOption, str, float]:
        """Pattern matching decision strategy."""
        # Find patterns in past successful decisions
        successful_patterns = await self._find_successful_patterns(context)
        
        # Score options based on pattern matching
        best_option = options[0]
        best_score = 0.0
        
        for option in options:
            pattern_score = self._calculate_pattern_match(option, successful_patterns)
            if pattern_score > best_score:
                best_score = pattern_score
                best_option = option
        
        reasoning = f"Selected based on pattern matching score of {best_score:.2f}"
        confidence = best_score
        
        return best_option, reasoning, confidence
    
    async def _outcome_prediction_strategy(
        self,
        context: DecisionContext,
        options: List[DecisionOption]
    ) -> Tuple[DecisionOption, str, float]:
        """Outcome prediction decision strategy."""
        # Choose option with best risk-adjusted predicted outcome
        best_option = None
        best_score = -1.0
        
        for option in options:
            # Risk-adjusted score
            risk_adjusted_score = option.predicted_outcome_score * (1.0 - option.risk_score * self.config["risk_aversion_factor"])
            
            if risk_adjusted_score > best_score:
                best_score = risk_adjusted_score
                best_option = option
        
        if best_option is None:
            best_option = options[0]
        
        reasoning = f"Selected based on risk-adjusted outcome score of {best_score:.2f}"
        confidence = best_option.confidence_score
        
        return best_option, reasoning, confidence
    
    async def _hybrid_strategy(
        self,
        context: DecisionContext,
        options: List[DecisionOption]
    ) -> Tuple[DecisionOption, str, float]:
        """Hybrid decision strategy combining multiple factors."""
        best_option = None
        best_score = -1.0
        
        for option in options:
            # Weighted combination of factors
            hybrid_score = (
                option.memory_support_score * self.config["memory_support_weight"] +
                option.predicted_outcome_score * self.config["outcome_prediction_weight"] +
                option.confidence_score * 0.2 +
                (1.0 - option.risk_score) * self.config["risk_aversion_factor"]
            )
            
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_option = option
        
        if best_option is None:
            best_option = options[0]
        
        reasoning = f"Selected using hybrid strategy with combined score of {best_score:.2f}"
        confidence = best_option.confidence_score
        
        return best_option, reasoning, confidence

    async def _find_successful_patterns(self, context: DecisionContext) -> List[Dict[str, Any]]:
        """Find patterns from past successful decisions."""
        try:
            successful_patterns = []

            # Analyze past successful decisions
            for decision_record in self.decision_records.values():
                if (decision_record.outcome_quality and
                    decision_record.outcome_quality > 0.7 and
                    decision_record.decision_context.decision_type == context.decision_type):

                    pattern = {
                        "context_type": decision_record.decision_context.context_type,
                        "chosen_option_type": decision_record.chosen_option.option_type,
                        "strategy": decision_record.decision_strategy,
                        "outcome_quality": decision_record.outcome_quality,
                        "memories_used": decision_record.memories_consulted
                    }
                    successful_patterns.append(pattern)

            return successful_patterns

        except Exception as e:
            logger.error(f"Failed to find successful patterns: {e}")
            return []

    def _calculate_pattern_match(self, option: DecisionOption, patterns: List[Dict[str, Any]]) -> float:
        """Calculate how well an option matches successful patterns."""
        try:
            if not patterns:
                return 0.5

            match_scores = []

            for pattern in patterns:
                score = 0.0

                # Match option type
                if option.option_type == pattern.get("chosen_option_type"):
                    score += 0.5

                # Match memory usage patterns
                option_memories = set(option.supporting_memories)
                pattern_memories = set(pattern.get("memories_used", []))

                if option_memories and pattern_memories:
                    memory_overlap = len(option_memories.intersection(pattern_memories))
                    memory_union = len(option_memories.union(pattern_memories))
                    if memory_union > 0:
                        score += 0.3 * (memory_overlap / memory_union)

                # Weight by pattern success
                score *= pattern.get("outcome_quality", 0.5)
                match_scores.append(score)

            return np.mean(match_scores) if match_scores else 0.5

        except Exception as e:
            logger.error(f"Failed to calculate pattern match: {e}")
            return 0.5

    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    async def _learn_decision_patterns(self, context: DecisionContext, chosen_option: DecisionOption):
        """Learn patterns from decision context and chosen option."""
        try:
            pattern_key = f"{context.decision_type.value}_{context.context_type.value}"

            if pattern_key not in self.decision_patterns:
                self.decision_patterns[pattern_key] = {
                    "option_types": defaultdict(int),
                    "success_rates": defaultdict(list),
                    "memory_patterns": defaultdict(int),
                    "total_decisions": 0
                }

            pattern = self.decision_patterns[pattern_key]
            pattern["option_types"][chosen_option.option_type] += 1
            pattern["total_decisions"] += 1

            # Track memory usage patterns
            for memory_id in chosen_option.supporting_memories:
                pattern["memory_patterns"][memory_id] += 1

        except Exception as e:
            logger.error(f"Failed to learn decision patterns: {e}")

    async def record_decision_outcome(
        self,
        decision_id: str,
        actual_outcome: Dict[str, Any],
        outcome_quality: float,
        lessons_learned: Optional[List[str]] = None
    ):
        """Record the actual outcome of a decision."""
        try:
            if decision_id not in self.decision_records:
                logger.warning(f"Decision record not found: {decision_id}")
                return

            decision_record = self.decision_records[decision_id]
            decision_record.actual_outcome = actual_outcome
            decision_record.outcome_quality = outcome_quality
            decision_record.lessons_learned = lessons_learned or []
            decision_record.completed_at = datetime.now()

            # Update success statistics
            if outcome_quality > 0.6:
                self.stats["successful_decisions"] += 1

            # Update decision accuracy
            total_completed = sum(
                1 for record in self.decision_records.values()
                if record.outcome_quality is not None
            )

            if total_completed > 0:
                successful_completed = sum(
                    1 for record in self.decision_records.values()
                    if record.outcome_quality and record.outcome_quality > 0.6
                )
                self.stats["decision_accuracy"] = successful_completed / total_completed

            # Learn from outcome if enabled
            if self.config["enable_learning_from_outcomes"]:
                await self._learn_from_outcome(decision_record)

            logger.info(
                "Decision outcome recorded",
                agent_id=self.agent_id,
                decision_id=decision_id,
                outcome_quality=outcome_quality
            )

        except Exception as e:
            logger.error(f"Failed to record decision outcome: {e}")

    async def _learn_from_outcome(self, decision_record: DecisionRecord):
        """Learn from decision outcome to improve future decisions."""
        try:
            pattern_key = f"{decision_record.decision_context.decision_type.value}_{decision_record.decision_context.context_type.value}"

            if pattern_key in self.decision_patterns:
                pattern = self.decision_patterns[pattern_key]

                # Update success rates for this option type
                option_type = decision_record.chosen_option.option_type
                pattern["success_rates"][option_type].append(decision_record.outcome_quality)

                # Keep only recent success rates (last 20 decisions)
                if len(pattern["success_rates"][option_type]) > 20:
                    pattern["success_rates"][option_type] = pattern["success_rates"][option_type][-20:]

            # Create memory from lessons learned
            if decision_record.lessons_learned and hasattr(self.memory_collection, 'add_memory'):
                lesson_content = f"Decision lessons: {'; '.join(decision_record.lessons_learned)}"
                await self.memory_collection.add_memory(
                    content=lesson_content,
                    memory_type="procedural",
                    importance="high",
                    context={
                        "decision_id": decision_record.decision_id,
                        "decision_type": decision_record.decision_context.decision_type.value,
                        "outcome_quality": decision_record.outcome_quality
                    }
                )

        except Exception as e:
            logger.error(f"Failed to learn from outcome: {e}")

    def _update_decision_stats(self, decision_record: DecisionRecord):
        """Update decision-making statistics."""
        self.stats["total_decisions"] += 1
        self.stats["strategy_usage"][decision_record.decision_strategy] += 1
        self.stats["decision_type_distribution"][decision_record.decision_context.decision_type.value] += 1

        # Update average confidence
        confidence_values = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9
        }

        confidence_score = confidence_values.get(decision_record.confidence, 0.5)
        alpha = 0.1
        self.stats["avg_confidence"] = (
            alpha * confidence_score + (1 - alpha) * self.stats["avg_confidence"]
        )

        # Update memory consultation rate
        if decision_record.memories_consulted:
            consultation_rate = 1.0
        else:
            consultation_rate = 0.0

        self.stats["memory_consultation_rate"] = (
            alpha * consultation_rate + (1 - alpha) * self.stats["memory_consultation_rate"]
        )

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get comprehensive decision-making statistics."""
        # Calculate additional metrics
        pattern_stats = {}
        for pattern_key, pattern_data in self.decision_patterns.items():
            avg_success_rates = {}
            for option_type, success_rates in pattern_data["success_rates"].items():
                if success_rates:
                    avg_success_rates[option_type] = np.mean(success_rates)

            pattern_stats[pattern_key] = {
                "total_decisions": pattern_data["total_decisions"],
                "option_type_distribution": dict(pattern_data["option_types"]),
                "avg_success_rates": avg_success_rates
            }

        return {
            **self.stats,
            "total_decision_records": len(self.decision_records),
            "total_patterns": len(self.decision_patterns),
            "pattern_breakdown": pattern_stats,
            "config": self.config
        }
