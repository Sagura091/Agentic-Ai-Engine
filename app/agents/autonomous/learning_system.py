"""
Adaptive Learning System for Autonomous Agents.

This module implements sophisticated learning capabilities for autonomous agents,
including experience-based adaptation, pattern recognition, and behavioral evolution.
"""

import asyncio
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

from app.agents.autonomous.autonomous_agent import AutonomousDecision, AutonomousAgentConfig

logger = structlog.get_logger(__name__)


@dataclass
class LearningInsight:
    """Represents a learning insight from experience analysis."""
    insight_type: str
    description: str
    confidence: float
    actionable_recommendations: List[str]
    supporting_evidence: Dict[str, Any]


class PerformancePattern(BaseModel):
    """Pattern in agent performance data."""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: str = Field(..., description="Type of pattern detected")
    frequency: int = Field(..., description="How often this pattern occurs")
    success_rate: float = Field(..., description="Success rate for this pattern")
    context_conditions: Dict[str, Any] = Field(..., description="Conditions when pattern occurs")
    recommended_actions: List[str] = Field(..., description="Recommended actions for this pattern")


class AdaptationSuggestion(BaseModel):
    """Suggestion for behavioral adaptation."""
    adaptation_id: str = Field(..., description="Unique adaptation identifier")
    type: str = Field(..., description="Type of adaptation")
    description: str = Field(..., description="Description of the adaptation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this adaptation")
    expected_improvement: float = Field(..., description="Expected performance improvement")
    risk_level: float = Field(..., ge=0.0, le=1.0, description="Risk of applying this adaptation")
    implementation_details: Dict[str, Any] = Field(..., description="How to implement the adaptation")


class AdaptiveLearningSystem:
    """
    Advanced learning system for autonomous agents.
    
    Implements multiple learning strategies:
    - Experience-based learning from decision outcomes
    - Pattern recognition in performance data
    - Behavioral adaptation based on success/failure patterns
    - Meta-learning for learning strategy optimization
    """
    
    def __init__(self, config: AutonomousAgentConfig):
        """Initialize the adaptive learning system."""
        self.config = config
        self.learning_enabled = config.learning_mode != "disabled"
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        
        # Learning data storage
        self.experience_buffer = deque(maxlen=1000)  # Recent experiences
        self.performance_history = deque(maxlen=500)  # Performance metrics
        self.pattern_database: Dict[str, PerformancePattern] = {}
        self.adaptation_history: List[AdaptationSuggestion] = []
        self.learning_data: Dict[str, Any] = {}  # General learning data storage
        
        # Learning statistics
        self.learning_stats = {
            "total_experiences": 0,
            "patterns_discovered": 0,
            "adaptations_applied": 0,
            "learning_rate_adjustments": 0,
            "last_learning_session": None
        }
        
        # Pattern detection parameters
        self.min_pattern_frequency = 3
        self.pattern_confidence_threshold = 0.6
        
        logger.info(
            "Adaptive learning system initialized",
            learning_mode=config.learning_mode,
            learning_rate=self.learning_rate,
            buffer_size=self.experience_buffer.maxlen
        )
    
    async def analyze_and_learn(
        self,
        performance_data: Dict[str, Any],
        decision_history: List[AutonomousDecision],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze recent performance and generate learning insights.
        
        Args:
            performance_data: Recent performance metrics
            decision_history: Recent decision history
            context: Current context and state
            
        Returns:
            Dictionary containing learning insights and adaptation suggestions
        """
        if not self.learning_enabled:
            return {"status": "learning_disabled", "insights": [], "adaptations": []}
        
        try:
            # Store experience data
            experience = {
                "timestamp": datetime.utcnow(),
                "performance_data": performance_data,
                "decisions": [d.dict() for d in decision_history],
                "context": context
            }
            self.experience_buffer.append(experience)
            self.performance_history.append(performance_data)
            self.learning_stats["total_experiences"] += 1
            
            # Analyze patterns in recent data
            patterns = await self._detect_performance_patterns()
            
            # Generate learning insights
            insights = await self._generate_learning_insights(
                performance_data, decision_history, patterns
            )
            
            # Create adaptation suggestions
            adaptations = await self._suggest_adaptations(insights, patterns)
            
            # Update learning statistics
            self.learning_stats["last_learning_session"] = datetime.utcnow()
            
            # Prepare learning results
            learning_results = {
                "status": "learning_completed",
                "insights": [insight.__dict__ for insight in insights],
                "adaptations": [adapt.dict() for adapt in adaptations],
                "patterns_detected": len(patterns),
                "learning_rate": self.learning_rate,
                "adaptation_score": self._calculate_adaptation_score(adaptations),
                "summary": self._create_learning_summary(insights, adaptations)
            }
            
            logger.info(
                "Learning analysis completed",
                insights_count=len(insights),
                adaptations_count=len(adaptations),
                patterns_count=len(patterns)
            )
            
            return learning_results
            
        except Exception as e:
            logger.error("Learning analysis failed", error=str(e))
            return {
                "status": "learning_failed",
                "error": str(e),
                "insights": [],
                "adaptations": []
            }
    
    async def _detect_performance_patterns(self) -> List[PerformancePattern]:
        """Detect patterns in performance data."""
        if len(self.performance_history) < self.min_pattern_frequency:
            return []
        
        patterns = []
        
        # Convert performance history to analyzable format
        performance_data = list(self.performance_history)
        
        # Pattern 1: Success rate trends
        success_pattern = self._analyze_success_trends(performance_data)
        if success_pattern:
            patterns.append(success_pattern)
        
        # Pattern 2: Tool usage effectiveness
        tool_patterns = self._analyze_tool_usage_patterns(performance_data)
        patterns.extend(tool_patterns)
        
        # Pattern 3: Decision confidence correlation
        confidence_pattern = self._analyze_confidence_patterns(performance_data)
        if confidence_pattern:
            patterns.append(confidence_pattern)
        
        # Pattern 4: Error occurrence patterns
        error_patterns = self._analyze_error_patterns(performance_data)
        patterns.extend(error_patterns)
        
        # Store discovered patterns
        for pattern in patterns:
            self.pattern_database[pattern.pattern_id] = pattern
        
        self.learning_stats["patterns_discovered"] += len(patterns)
        
        return patterns
    
    def _analyze_success_trends(self, performance_data: List[Dict[str, Any]]) -> Optional[PerformancePattern]:
        """Analyze trends in success rates."""
        try:
            # Extract success indicators
            success_rates = []
            for data in performance_data[-10:]:  # Last 10 data points
                errors = data.get("errors_encountered", 0)
                outputs = data.get("outputs_generated", 0)
                success_rate = max(0.0, 1.0 - (errors / max(1, outputs + errors)))
                success_rates.append(success_rate)
            
            if len(success_rates) < 3:
                return None
            
            # Calculate trend
            avg_success = np.mean(success_rates)
            trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
            
            pattern_type = "improving" if trend > 0.05 else "declining" if trend < -0.05 else "stable"
            
            return PerformancePattern(
                pattern_id=f"success_trend_{pattern_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                pattern_type=f"success_trend_{pattern_type}",
                frequency=len(success_rates),
                success_rate=avg_success,
                context_conditions={"trend_slope": trend, "average_success": avg_success},
                recommended_actions=self._get_success_trend_recommendations(pattern_type, avg_success)
            )
            
        except Exception as e:
            logger.warning("Success trend analysis failed", error=str(e))
            return None
    
    def _analyze_tool_usage_patterns(self, performance_data: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """Analyze patterns in tool usage effectiveness."""
        patterns = []
        
        try:
            # Aggregate tool usage data
            tool_stats = defaultdict(lambda: {"uses": 0, "successes": 0})
            
            for data in performance_data:
                tools_used = data.get("tools_used", 0)
                errors = data.get("errors_encountered", 0)
                
                # Simple heuristic: if tools were used and no errors, consider successful
                if tools_used > 0:
                    success = errors == 0
                    for i in range(tools_used):  # Simplified tool tracking
                        tool_key = f"tool_{i}"
                        tool_stats[tool_key]["uses"] += 1
                        if success:
                            tool_stats[tool_key]["successes"] += 1
            
            # Create patterns for tools with sufficient data
            for tool_key, stats in tool_stats.items():
                if stats["uses"] >= self.min_pattern_frequency:
                    success_rate = stats["successes"] / stats["uses"]
                    
                    pattern = PerformancePattern(
                        pattern_id=f"tool_usage_{tool_key}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                        pattern_type="tool_effectiveness",
                        frequency=stats["uses"],
                        success_rate=success_rate,
                        context_conditions={"tool": tool_key, "usage_count": stats["uses"]},
                        recommended_actions=self._get_tool_usage_recommendations(tool_key, success_rate)
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            logger.warning("Tool usage pattern analysis failed", error=str(e))
        
        return patterns
    
    def _analyze_confidence_patterns(self, performance_data: List[Dict[str, Any]]) -> Optional[PerformancePattern]:
        """Analyze correlation between confidence levels and outcomes."""
        try:
            confidence_data = []
            outcome_data = []
            
            for data in performance_data:
                confidence_levels = data.get("confidence_levels", [])
                errors = data.get("errors_encountered", 0)
                outputs = data.get("outputs_generated", 0)
                
                if confidence_levels:
                    avg_confidence = np.mean(confidence_levels)
                    success_rate = max(0.0, 1.0 - (errors / max(1, outputs + errors)))
                    
                    confidence_data.append(avg_confidence)
                    outcome_data.append(success_rate)
            
            if len(confidence_data) < 3:
                return None
            
            # Calculate correlation
            correlation = np.corrcoef(confidence_data, outcome_data)[0, 1]
            
            if abs(correlation) > 0.3:  # Significant correlation
                pattern_type = "positive_correlation" if correlation > 0 else "negative_correlation"
                
                return PerformancePattern(
                    pattern_id=f"confidence_correlation_{pattern_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                    pattern_type="confidence_outcome_correlation",
                    frequency=len(confidence_data),
                    success_rate=np.mean(outcome_data),
                    context_conditions={"correlation": correlation, "pattern_type": pattern_type},
                    recommended_actions=self._get_confidence_correlation_recommendations(correlation)
                )
            
        except Exception as e:
            logger.warning("Confidence pattern analysis failed", error=str(e))
        
        return None
    
    def _analyze_error_patterns(self, performance_data: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """Analyze patterns in error occurrence."""
        patterns = []
        
        try:
            error_frequencies = []
            for data in performance_data:
                errors = data.get("errors_encountered", 0)
                error_frequencies.append(errors)
            
            if not error_frequencies:
                return patterns
            
            avg_errors = np.mean(error_frequencies)
            error_trend = np.polyfit(range(len(error_frequencies)), error_frequencies, 1)[0]
            
            if avg_errors > 0.5:  # Significant error rate
                pattern_type = "high_error_rate"
                if error_trend > 0.1:
                    pattern_type = "increasing_errors"
                elif error_trend < -0.1:
                    pattern_type = "decreasing_errors"
                
                pattern = PerformancePattern(
                    pattern_id=f"error_pattern_{pattern_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                    pattern_type=pattern_type,
                    frequency=len(error_frequencies),
                    success_rate=max(0.0, 1.0 - avg_errors),
                    context_conditions={"avg_errors": avg_errors, "error_trend": error_trend},
                    recommended_actions=self._get_error_pattern_recommendations(pattern_type, avg_errors)
                )
                patterns.append(pattern)
            
        except Exception as e:
            logger.warning("Error pattern analysis failed", error=str(e))
        
        return patterns
    
    async def _generate_learning_insights(
        self,
        performance_data: Dict[str, Any],
        decision_history: List[AutonomousDecision],
        patterns: List[PerformancePattern]
    ) -> List[LearningInsight]:
        """Generate actionable learning insights from analysis."""
        insights = []
        
        # Insight 1: Decision quality assessment
        if decision_history:
            decision_insight = self._analyze_decision_quality(decision_history)
            if decision_insight:
                insights.append(decision_insight)
        
        # Insight 2: Performance trend analysis
        performance_insight = self._analyze_performance_trends(performance_data)
        if performance_insight:
            insights.append(performance_insight)
        
        # Insight 3: Pattern-based insights
        for pattern in patterns:
            pattern_insight = self._create_pattern_insight(pattern)
            if pattern_insight:
                insights.append(pattern_insight)
        
        # Insight 4: Learning rate optimization
        learning_insight = self._analyze_learning_effectiveness()
        if learning_insight:
            insights.append(learning_insight)
        
        return insights
    
    def _analyze_decision_quality(self, decision_history: List[AutonomousDecision]) -> Optional[LearningInsight]:
        """Analyze the quality of recent decisions."""
        if not decision_history:
            return None
        
        try:
            # Analyze confidence vs. outcome correlation
            confidences = [d.confidence for d in decision_history]
            avg_confidence = np.mean(confidences)
            confidence_variance = np.var(confidences)
            
            # Determine insight based on confidence patterns
            if avg_confidence < 0.5:
                insight_type = "low_decision_confidence"
                description = "Recent decisions show low confidence levels, indicating uncertainty in decision-making"
                recommendations = [
                    "Consider gathering more information before making decisions",
                    "Adjust decision criteria to improve confidence",
                    "Implement additional validation steps"
                ]
            elif confidence_variance > 0.2:
                insight_type = "inconsistent_decision_confidence"
                description = "Decision confidence varies significantly, suggesting inconsistent decision-making"
                recommendations = [
                    "Standardize decision-making criteria",
                    "Implement confidence calibration mechanisms",
                    "Review decision patterns for consistency"
                ]
            else:
                insight_type = "stable_decision_confidence"
                description = "Decision confidence is stable and appropriate"
                recommendations = [
                    "Maintain current decision-making approach",
                    "Continue monitoring decision outcomes"
                ]
            
            return LearningInsight(
                insight_type=insight_type,
                description=description,
                confidence=0.8,
                actionable_recommendations=recommendations,
                supporting_evidence={
                    "avg_confidence": avg_confidence,
                    "confidence_variance": confidence_variance,
                    "decision_count": len(decision_history)
                }
            )
            
        except Exception as e:
            logger.warning("Decision quality analysis failed", error=str(e))
            return None

    def _analyze_performance_trends(self, performance_data: Dict[str, Any]) -> Optional[LearningInsight]:
        """Analyze performance trends to identify improvement opportunities."""
        try:
            if not performance_data:
                return None

            # Extract performance metrics
            success_rate = performance_data.get("success_rate", 0.0)
            response_time = performance_data.get("avg_response_time", 0.0)
            error_rate = performance_data.get("error_rate", 0.0)

            # Analyze trends
            insights = []
            if success_rate < 0.7:
                insights.append("Success rate is below optimal threshold")
            if response_time > 5.0:
                insights.append("Response time is slower than expected")
            if error_rate > 0.2:
                insights.append("Error rate is higher than acceptable")

            if not insights:
                insights.append("Performance metrics are within acceptable ranges")

            return LearningInsight(
                insight_type="performance_trend",
                description="Performance trend analysis",
                confidence=0.8,
                actionable_recommendations=[
                    "Monitor performance metrics regularly",
                    "Optimize response time if needed",
                    "Reduce error rate through better error handling"
                ],
                supporting_evidence={
                    "success_rate": success_rate,
                    "response_time": response_time,
                    "error_rate": error_rate,
                    "insights": insights
                }
            )

        except Exception as e:
            logger.warning("Performance trend analysis failed", error=str(e))
            return None

    def analyze_learning_effectiveness(self) -> Optional[LearningInsight]:
        """Analyze the effectiveness of the learning system."""
        try:
            # Calculate learning metrics
            total_experiences = len(self.experience_buffer)
            patterns_discovered = len(self.pattern_database)
            adaptations_made = len(self.adaptation_history)

            # Calculate learning rate
            if total_experiences > 0:
                learning_rate = patterns_discovered / total_experiences
            else:
                learning_rate = 0.0

            # Analyze effectiveness
            effectiveness_score = min(1.0, learning_rate * 2.0)  # Scale to 0-1

            suggestions = []
            if effectiveness_score < 0.3:
                suggestions.append("Increase learning rate to improve adaptation")
            elif effectiveness_score < 0.6:
                suggestions.append("Moderate learning effectiveness - continue current approach")
            else:
                suggestions.append("High learning effectiveness - maintain current strategy")

            return LearningInsight(
                insight_type="learning_effectiveness",
                description="Analysis of learning system effectiveness",
                confidence=0.8,
                data={
                    "total_experiences": total_experiences,
                    "patterns_discovered": patterns_discovered,
                    "adaptations_made": adaptations_made,
                    "learning_rate": learning_rate,
                    "effectiveness_score": effectiveness_score
                },
                suggestions=suggestions
            )

        except Exception as e:
            logger.warning("Learning effectiveness analysis failed", error=str(e))
            return None

    def _analyze_learning_effectiveness(self) -> Optional[LearningInsight]:
        """Analyze the effectiveness of the learning system."""
        try:
            # Get recent learning metrics
            total_experiences = len(self.experience_buffer)
            patterns_discovered = len(self.pattern_database)
            adaptations_made = len([exp for exp in self.experience_buffer if exp.outcome == "success"])

            # Calculate learning rate
            if total_experiences > 0:
                learning_rate = patterns_discovered / total_experiences
            else:
                learning_rate = 0.0

            # Analyze effectiveness
            effectiveness_score = min(1.0, learning_rate * 2.0)  # Scale to 0-1

            suggestions = []
            if effectiveness_score < 0.3:
                suggestions.append("Increase learning rate to improve adaptation")
            elif effectiveness_score < 0.6:
                suggestions.append("Moderate learning effectiveness - continue current approach")
            else:
                suggestions.append("High learning effectiveness - maintain current strategy")

            return LearningInsight(
                insight_type="learning_effectiveness",
                description="Analysis of learning system effectiveness",
                confidence=0.8,
                data={
                    "total_experiences": total_experiences,
                    "patterns_discovered": patterns_discovered,
                    "adaptations_made": adaptations_made,
                    "learning_rate": learning_rate,
                    "effectiveness_score": effectiveness_score
                },
                suggestions=suggestions
            )

        except Exception as e:
            logger.warning("Learning effectiveness analysis failed", error=str(e))
            return None

    def _calculate_adaptation_score(self, adaptations: List[Any]) -> float:
        """Calculate an adaptation score based on the quality and quantity of adaptations."""
        try:
            if not adaptations:
                return 0.0

            # Calculate score based on number and quality of adaptations
            base_score = min(1.0, len(adaptations) / 5.0)  # Scale based on quantity

            # Add quality bonus if adaptations have confidence scores
            quality_bonus = 0.0
            for adaptation in adaptations:
                if hasattr(adaptation, 'confidence'):
                    quality_bonus += adaptation.confidence
                elif isinstance(adaptation, dict) and 'confidence' in adaptation:
                    quality_bonus += adaptation['confidence']
                else:
                    quality_bonus += 0.5  # Default confidence

            if adaptations:
                quality_bonus = quality_bonus / len(adaptations)

            # Combine base score and quality bonus
            final_score = (base_score * 0.7) + (quality_bonus * 0.3)
            return min(1.0, final_score)

        except Exception as e:
            logger.warning("Failed to calculate adaptation score", error=str(e))
            return 0.5  # Default score

    async def run_continuous_learning_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a continuous learning cycle to adapt behavior based on experience."""
        try:
            cycle_results = {
                "cycle_timestamp": datetime.utcnow().isoformat(),
                "insights_generated": 0,
                "patterns_discovered": 0,
                "adaptations_suggested": 0,
                "learning_improvements": []
            }

            # Step 1: Analyze recent experiences
            recent_experiences = list(self.experience_buffer)[-10:]  # Last 10 experiences
            if recent_experiences:
                # Extract patterns from recent experiences
                new_patterns = await self._discover_patterns_from_experiences(recent_experiences)
                cycle_results["patterns_discovered"] = len(new_patterns)

                # Update pattern database
                for pattern in new_patterns:
                    self.pattern_database[pattern.pattern_id] = pattern

            # Step 2: Generate learning insights
            insights = self.generate_insights()
            cycle_results["insights_generated"] = len(insights)

            # Step 3: Suggest behavioral adaptations
            adaptations = await self._suggest_behavioral_adaptations(context, insights)
            cycle_results["adaptations_suggested"] = len(adaptations)

            # Step 4: Apply high-confidence adaptations
            applied_adaptations = await self._apply_adaptations(adaptations)
            cycle_results["learning_improvements"] = applied_adaptations

            # Step 5: Update learning parameters
            await self._update_learning_parameters(cycle_results)

            logger.info("Continuous learning cycle completed", **cycle_results)
            return cycle_results

        except Exception as e:
            logger.error("Continuous learning cycle failed", error=str(e))
            return {"status": "error", "message": str(e)}

    async def _discover_patterns_from_experiences(self, experiences: List[Any]) -> List[PerformancePattern]:
        """Discover new patterns from recent experiences."""
        patterns = []

        # Group experiences by context similarity
        context_groups = defaultdict(list)
        for exp in experiences:
            if hasattr(exp, 'context'):
                context_key = str(sorted(exp.context.items()))
                context_groups[context_key].append(exp)

        # Analyze each group for patterns
        for context_key, group_experiences in context_groups.items():
            if len(group_experiences) >= 3:  # Need minimum experiences for pattern
                success_rate = sum(1 for exp in group_experiences if getattr(exp, 'success', False)) / len(group_experiences)

                if success_rate > 0.8 or success_rate < 0.3:  # High success or failure patterns
                    pattern = PerformancePattern(
                        pattern_id=f"pattern_{len(self.pattern_database)}_{int(datetime.utcnow().timestamp())}",
                        pattern_type="context_performance",
                        frequency=len(group_experiences),
                        success_rate=success_rate,
                        context_conditions=group_experiences[0].context if hasattr(group_experiences[0], 'context') else {},
                        recommended_actions=[
                            "Replicate successful context" if success_rate > 0.8 else "Avoid or modify context",
                            "Analyze decision factors",
                            "Update decision criteria"
                        ]
                    )
                    patterns.append(pattern)

        return patterns

    async def _suggest_behavioral_adaptations(self, context: Dict[str, Any], insights: List[LearningInsight]) -> List[AdaptationSuggestion]:
        """Suggest behavioral adaptations based on insights and context."""
        adaptations = []

        for insight in insights:
            if insight.confidence > 0.7:  # High confidence insights
                adaptation = AdaptationSuggestion(
                    adaptation_id=f"adapt_{len(adaptations)}_{int(datetime.utcnow().timestamp())}",
                    type=f"behavioral_{insight.insight_type}",
                    description=f"Adapt behavior based on {insight.insight_type}: {insight.description}",
                    confidence=insight.confidence,
                    expected_improvement=min(insight.confidence * 0.3, 0.2),  # Conservative estimate
                    risk_level=1.0 - insight.confidence,
                    implementation_details={
                        "insight_type": insight.insight_type,
                        "recommendations": insight.actionable_recommendations,
                        "evidence": insight.supporting_evidence
                    }
                )
                adaptations.append(adaptation)

        return adaptations

    async def _apply_adaptations(self, adaptations: List[AdaptationSuggestion]) -> List[str]:
        """Apply high-confidence adaptations to improve behavior."""
        applied = []

        for adaptation in adaptations:
            if adaptation.confidence > 0.8 and adaptation.risk_level < 0.3:
                # Apply the adaptation (in a real system, this would modify behavior)
                self.learning_data[f"adaptation_{adaptation.adaptation_id}"] = {
                    "applied_at": datetime.utcnow().isoformat(),
                    "type": adaptation.type,
                    "description": adaptation.description,
                    "expected_improvement": adaptation.expected_improvement
                }
                applied.append(adaptation.description)

                logger.info("Behavioral adaptation applied",
                           adaptation_id=adaptation.adaptation_id,
                           type=adaptation.type,
                           confidence=adaptation.confidence)

        return applied

    async def _update_learning_parameters(self, cycle_results: Dict[str, Any]) -> None:
        """Update learning parameters based on cycle results."""
        # Adjust learning rate based on effectiveness
        if cycle_results.get("insights_generated", 0) > 3:
            self.learning_rate = min(self.learning_rate * 1.1, 1.0)  # Increase learning rate
        elif cycle_results.get("insights_generated", 0) == 0:
            self.learning_rate = max(self.learning_rate * 0.9, 0.1)  # Decrease learning rate

        # Update experience buffer size based on pattern discovery
        if cycle_results.get("patterns_discovered", 0) > 2:
            self.max_experience_buffer = min(self.max_experience_buffer + 10, 1000)

        logger.debug("Learning parameters updated",
                    learning_rate=self.learning_rate,
                    buffer_size=self.max_experience_buffer)

    async def learn_from_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a specific experience and update behavior."""
        try:
            # Store the experience
            self.experience_buffer.append(experience)
            self.learning_stats["total_experiences"] += 1

            # Extract learning insights from the experience
            insights = []

            # Analyze success/failure patterns
            if experience.get("success", False):
                insight = LearningInsight(
                    insight_type="success_pattern",
                    description=f"Successful action: {experience.get('action', 'unknown')}",
                    confidence=0.8,
                    actionable_recommendations=[
                        f"Repeat similar actions in context: {experience.get('context', {})}"
                    ],
                    supporting_evidence=experience
                )
                insights.append(insight)
            else:
                insight = LearningInsight(
                    insight_type="failure_pattern",
                    description=f"Failed action: {experience.get('action', 'unknown')}",
                    confidence=0.7,
                    actionable_recommendations=[
                        f"Avoid similar actions in context: {experience.get('context', {})}"
                    ],
                    supporting_evidence=experience
                )
                insights.append(insight)

            # Update learning data
            experience_key = f"experience_{len(self.experience_buffer)}"
            self.learning_data[experience_key] = {
                "experience": experience,
                "insights": [insight.__dict__ for insight in insights],
                "learned_at": datetime.utcnow().isoformat()
            }

            # Run pattern discovery if we have enough experiences
            if len(self.experience_buffer) >= 5:
                new_patterns = await self._discover_patterns_from_experiences(list(self.experience_buffer)[-5:])
                for pattern in new_patterns:
                    self.pattern_database[pattern.pattern_id] = pattern
                    self.learning_stats["patterns_discovered"] += 1

            learning_result = {
                "insights_generated": len(insights),
                "patterns_discovered": len(new_patterns) if len(self.experience_buffer) >= 5 else 0,
                "total_experiences": self.learning_stats["total_experiences"],
                "learning_effectiveness": min(len(insights) * 0.2, 1.0)
            }

            logger.info("Learning from experience completed", **learning_result)
            return learning_result

        except Exception as e:
            logger.error("Learning from experience failed", error=str(e))
            return {"status": "error", "message": str(e)}

    async def _suggest_adaptations(self, insights: List[LearningInsight], patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adaptation suggestions based on learning insights and patterns."""
        try:
            adaptations = []

            for insight in insights:
                # Create adaptations based on insight type
                if insight.insight_type == "performance":
                    adaptations.extend([
                        {
                            "type": "performance_optimization",
                            "description": "Optimize decision-making speed based on performance patterns",
                            "priority": "high",
                            "implementation": "Adjust reasoning timeout and iteration limits",
                            "expected_impact": "15-25% faster execution"
                        },
                        {
                            "type": "accuracy_improvement",
                            "description": "Improve accuracy based on historical success patterns",
                            "priority": "medium",
                            "implementation": "Weight successful decision patterns more heavily",
                            "expected_impact": "10-15% better accuracy"
                        }
                    ])
                elif insight.insight_type == "strategy":
                    adaptations.extend([
                        {
                            "type": "strategy_refinement",
                            "description": "Refine strategy selection based on context patterns",
                            "priority": "medium",
                            "implementation": "Update strategy selection algorithms",
                            "expected_impact": "Better context-appropriate decisions"
                        }
                    ])
                elif insight.insight_type == "efficiency":
                    adaptations.extend([
                        {
                            "type": "efficiency_enhancement",
                            "description": "Enhance efficiency based on resource usage patterns",
                            "priority": "low",
                            "implementation": "Optimize resource allocation algorithms",
                            "expected_impact": "5-10% resource savings"
                        }
                    ])

            # Add pattern-based adaptations
            if patterns.get("success_rate", 0) < 0.7:
                adaptations.append({
                    "type": "success_rate_improvement",
                    "description": "Improve overall success rate through better planning",
                    "priority": "high",
                    "implementation": "Enhance planning algorithms and validation",
                    "expected_impact": "Increase success rate to 80%+"
                })

            if patterns.get("average_execution_time", 0) > 30:
                adaptations.append({
                    "type": "execution_speed_optimization",
                    "description": "Reduce execution time through process optimization",
                    "priority": "medium",
                    "implementation": "Streamline decision processes and reduce redundancy",
                    "expected_impact": "20-30% faster execution"
                })

            logger.debug(
                "Adaptation suggestions generated",
                total_adaptations=len(adaptations),
                high_priority=len([a for a in adaptations if a["priority"] == "high"]),
                medium_priority=len([a for a in adaptations if a["priority"] == "medium"]),
                low_priority=len([a for a in adaptations if a["priority"] == "low"])
            )

            return adaptations

        except Exception as e:
            logger.error("Adaptation suggestion failed", error=str(e))
            return []
