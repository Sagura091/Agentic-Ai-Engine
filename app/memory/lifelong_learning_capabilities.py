"""
Revolutionary Lifelong Learning Capabilities for Agentic AI Memory.

Implements cross-task memory transfer, performance-driven weighting, and
continuous learning from experience based on state-of-the-art research.

Key Features:
- Cross-task memory transfer
- Performance-driven memory weighting
- Continuous learning from experience
- Meta-learning capabilities
- Skill acquisition and retention
- Knowledge distillation
- Adaptive memory strategies
"""

import asyncio
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class LearningType(str, Enum):
    """Types of learning."""
    TASK_SPECIFIC = "task_specific"      # Learning for specific tasks
    CROSS_TASK = "cross_task"           # Learning across tasks
    META_LEARNING = "meta_learning"      # Learning how to learn
    SKILL_ACQUISITION = "skill_acquisition"  # Acquiring new skills
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"  # Distilling knowledge
    ADAPTIVE_STRATEGY = "adaptive_strategy"  # Adapting strategies


class PerformanceMetric(str, Enum):
    """Performance metrics for learning."""
    SUCCESS_RATE = "success_rate"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    SPEED = "speed"
    QUALITY = "quality"
    ADAPTABILITY = "adaptability"


@dataclass
class LearningExperience:
    """A learning experience record."""
    experience_id: str
    agent_id: str
    task_type: str
    task_context: Dict[str, Any]
    
    # Performance data
    performance_metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    success: bool = True
    duration_seconds: float = 0.0
    
    # Memory involvement
    memories_used: List[str] = field(default_factory=list)
    memories_created: List[str] = field(default_factory=list)
    memory_effectiveness: Dict[str, float] = field(default_factory=dict)
    
    # Learning outcomes
    skills_acquired: List[str] = field(default_factory=list)
    knowledge_gained: Dict[str, Any] = field(default_factory=dict)
    strategies_learned: List[str] = field(default_factory=list)
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    learning_type: LearningType = LearningType.TASK_SPECIFIC


@dataclass
class SkillProfile:
    """Profile of an acquired skill."""
    skill_id: str
    skill_name: str
    skill_type: str
    description: str = ""
    
    # Skill metrics
    proficiency_level: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    # Learning history
    acquisition_date: datetime = field(default_factory=datetime.now)
    learning_experiences: List[str] = field(default_factory=list)
    improvement_rate: float = 0.0
    
    # Transfer potential
    transferable_to: List[str] = field(default_factory=list)
    prerequisite_skills: List[str] = field(default_factory=list)
    
    # Memory associations
    associated_memories: Set[str] = field(default_factory=set)
    key_knowledge: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaLearningStrategy:
    """A meta-learning strategy."""
    strategy_id: str
    strategy_name: str
    strategy_type: str
    description: str = ""
    
    # Strategy effectiveness
    success_rate: float = 0.0
    avg_performance_improvement: float = 0.0
    applications: int = 0
    
    # Strategy parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Learning context
    applicable_tasks: List[str] = field(default_factory=list)
    learned_from: List[str] = field(default_factory=list)  # Experience IDs


class LifelongLearningCapabilities:
    """
    Revolutionary Lifelong Learning Capabilities.
    
    Enables continuous learning from experience with cross-task transfer,
    performance-driven weighting, and meta-learning capabilities.
    """
    
    def __init__(self, agent_id: str, memory_collection: Any):
        """Initialize lifelong learning capabilities."""
        self.agent_id = agent_id
        self.memory_collection = memory_collection
        
        # Learning components
        self.learning_experiences: Dict[str, LearningExperience] = {}
        self.skill_profiles: Dict[str, SkillProfile] = {}
        self.meta_strategies: Dict[str, MetaLearningStrategy] = {}
        
        # Performance tracking
        self.task_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.memory_effectiveness_scores: Dict[str, float] = {}
        
        # Learning configuration
        self.config = {
            "learning_rate": 0.1,
            "transfer_threshold": 0.7,
            "skill_decay_rate": 0.01,
            "meta_learning_enabled": True,
            "cross_task_transfer_enabled": True,
            "performance_weighting_enabled": True,
            "min_experiences_for_transfer": 5,
            "skill_proficiency_threshold": 0.8,
            "strategy_success_threshold": 0.6
        }
        
        # Statistics
        self.stats = {
            "total_experiences": 0,
            "total_skills_acquired": 0,
            "total_strategies_learned": 0,
            "successful_transfers": 0,
            "avg_performance_improvement": 0.0,
            "learning_efficiency": 0.0,
            "skill_retention_rate": 0.0
        }
        
        logger.info(f"Lifelong Learning Capabilities initialized for agent {agent_id}")
    
    async def record_learning_experience(
        self,
        task_type: str,
        task_context: Dict[str, Any],
        performance_metrics: Dict[PerformanceMetric, float],
        memories_used: List[str],
        memories_created: List[str],
        success: bool = True,
        duration_seconds: float = 0.0
    ) -> str:
        """Record a learning experience."""
        try:
            experience_id = f"exp_{int(time.time())}_{self.agent_id}"
            
            # Calculate memory effectiveness
            memory_effectiveness = await self._calculate_memory_effectiveness(
                memories_used, performance_metrics, success
            )
            
            # Create learning experience
            experience = LearningExperience(
                experience_id=experience_id,
                agent_id=self.agent_id,
                task_type=task_type,
                task_context=task_context,
                performance_metrics=performance_metrics,
                success=success,
                duration_seconds=duration_seconds,
                memories_used=memories_used,
                memories_created=memories_created,
                memory_effectiveness=memory_effectiveness
            )
            
            self.learning_experiences[experience_id] = experience
            
            # Update performance history
            overall_performance = np.mean(list(performance_metrics.values()))
            self.task_performance_history[task_type].append(overall_performance)
            
            # Update memory effectiveness scores
            await self._update_memory_effectiveness_scores(memory_effectiveness)
            
            # Trigger learning processes
            await self._process_learning_experience(experience)
            
            # Update statistics
            self.stats["total_experiences"] += 1
            
            logger.info(
                "Learning experience recorded",
                agent_id=self.agent_id,
                experience_id=experience_id,
                task_type=task_type,
                success=success,
                performance=overall_performance
            )
            
            return experience_id
            
        except Exception as e:
            logger.error(f"Failed to record learning experience: {e}")
            return ""
    
    async def _calculate_memory_effectiveness(
        self,
        memories_used: List[str],
        performance_metrics: Dict[PerformanceMetric, float],
        success: bool
    ) -> Dict[str, float]:
        """Calculate effectiveness of memories used in the task."""
        effectiveness = {}
        
        if not memories_used:
            return effectiveness
        
        # Base effectiveness from performance
        base_effectiveness = np.mean(list(performance_metrics.values())) if performance_metrics else 0.5
        
        # Adjust for success/failure
        if not success:
            base_effectiveness *= 0.5
        
        # Distribute effectiveness among memories (could be more sophisticated)
        for memory_id in memories_used:
            effectiveness[memory_id] = base_effectiveness
        
        return effectiveness
    
    async def _update_memory_effectiveness_scores(self, memory_effectiveness: Dict[str, float]):
        """Update global memory effectiveness scores."""
        learning_rate = self.config["learning_rate"]
        
        for memory_id, effectiveness in memory_effectiveness.items():
            current_score = self.memory_effectiveness_scores.get(memory_id, 0.5)
            
            # Update with exponential moving average
            new_score = (1 - learning_rate) * current_score + learning_rate * effectiveness
            self.memory_effectiveness_scores[memory_id] = new_score
    
    async def _process_learning_experience(self, experience: LearningExperience):
        """Process a learning experience for skill acquisition and strategy learning."""
        try:
            # 1. Skill acquisition
            await self._identify_and_acquire_skills(experience)
            
            # 2. Strategy learning
            if self.config["meta_learning_enabled"]:
                await self._learn_meta_strategies(experience)
            
            # 3. Cross-task transfer
            if self.config["cross_task_transfer_enabled"]:
                await self._identify_transfer_opportunities(experience)
            
            # 4. Performance-driven weighting
            if self.config["performance_weighting_enabled"]:
                await self._update_performance_weights(experience)
            
        except Exception as e:
            logger.error(f"Failed to process learning experience: {e}")
    
    async def _identify_and_acquire_skills(self, experience: LearningExperience):
        """Identify and acquire new skills from the experience."""
        try:
            # Simple skill identification based on task type and success
            if experience.success and experience.performance_metrics:
                avg_performance = np.mean(list(experience.performance_metrics.values()))
                
                if avg_performance > self.config["skill_proficiency_threshold"]:
                    skill_id = f"skill_{experience.task_type}_{len(self.skill_profiles)}"
                    skill_name = f"{experience.task_type.replace('_', ' ').title()} Skill"
                    
                    # Check if skill already exists
                    existing_skill = None
                    for skill in self.skill_profiles.values():
                        if skill.skill_type == experience.task_type:
                            existing_skill = skill
                            break
                    
                    if existing_skill:
                        # Update existing skill
                        existing_skill.usage_count += 1
                        existing_skill.last_used = datetime.now()
                        existing_skill.learning_experiences.append(experience.experience_id)
                        
                        # Update proficiency with learning rate
                        learning_rate = self.config["learning_rate"]
                        existing_skill.proficiency_level = (
                            (1 - learning_rate) * existing_skill.proficiency_level +
                            learning_rate * avg_performance
                        )
                        
                        # Update improvement rate
                        if len(existing_skill.learning_experiences) > 1:
                            recent_experiences = existing_skill.learning_experiences[-5:]  # Last 5 experiences
                            improvement = self._calculate_skill_improvement(existing_skill.skill_id, recent_experiences)
                            existing_skill.improvement_rate = improvement
                    else:
                        # Create new skill
                        new_skill = SkillProfile(
                            skill_id=skill_id,
                            skill_name=skill_name,
                            skill_type=experience.task_type,
                            description=f"Skill acquired from {experience.task_type} tasks",
                            proficiency_level=avg_performance,
                            confidence=avg_performance,
                            last_used=datetime.now(),
                            usage_count=1,
                            learning_experiences=[experience.experience_id],
                            associated_memories=set(experience.memories_used)
                        )
                        
                        self.skill_profiles[skill_id] = new_skill
                        self.stats["total_skills_acquired"] += 1
                        
                        experience.skills_acquired.append(skill_id)
            
        except Exception as e:
            logger.error(f"Failed to identify and acquire skills: {e}")
    
    def _calculate_skill_improvement(self, skill_id: str, recent_experiences: List[str]) -> float:
        """Calculate skill improvement rate from recent experiences."""
        try:
            if len(recent_experiences) < 2:
                return 0.0
            
            performances = []
            for exp_id in recent_experiences:
                if exp_id in self.learning_experiences:
                    exp = self.learning_experiences[exp_id]
                    if exp.performance_metrics:
                        avg_perf = np.mean(list(exp.performance_metrics.values()))
                        performances.append(avg_perf)
            
            if len(performances) < 2:
                return 0.0
            
            # Calculate linear regression slope as improvement rate
            x = np.arange(len(performances))
            y = np.array(performances)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                return float(slope)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate skill improvement: {e}")
            return 0.0
    
    async def _learn_meta_strategies(self, experience: LearningExperience):
        """Learn meta-strategies from successful experiences."""
        try:
            if not experience.success:
                return
            
            # Identify patterns in successful experiences
            task_type = experience.task_type
            similar_experiences = [
                exp for exp in self.learning_experiences.values()
                if exp.task_type == task_type and exp.success
            ]
            
            if len(similar_experiences) >= self.config["min_experiences_for_transfer"]:
                # Analyze common patterns
                common_memories = self._find_common_memory_patterns(similar_experiences)
                common_strategies = self._identify_successful_strategies(similar_experiences)
                
                if common_strategies:
                    strategy_id = f"strategy_{task_type}_{len(self.meta_strategies)}"
                    strategy_name = f"Successful {task_type.replace('_', ' ').title()} Strategy"
                    
                    # Calculate strategy effectiveness
                    success_rate = len(similar_experiences) / max(len([
                        exp for exp in self.learning_experiences.values()
                        if exp.task_type == task_type
                    ]), 1)
                    
                    avg_improvement = np.mean([
                        np.mean(list(exp.performance_metrics.values()))
                        for exp in similar_experiences
                        if exp.performance_metrics
                    ])
                    
                    if success_rate > self.config["strategy_success_threshold"]:
                        strategy = MetaLearningStrategy(
                            strategy_id=strategy_id,
                            strategy_name=strategy_name,
                            strategy_type=task_type,
                            description=f"Meta-strategy learned from {len(similar_experiences)} successful experiences",
                            success_rate=success_rate,
                            avg_performance_improvement=avg_improvement,
                            applications=len(similar_experiences),
                            parameters={"common_memories": common_memories},
                            applicable_tasks=[task_type],
                            learned_from=[exp.experience_id for exp in similar_experiences]
                        )
                        
                        self.meta_strategies[strategy_id] = strategy
                        self.stats["total_strategies_learned"] += 1
                        
                        experience.strategies_learned.append(strategy_id)
            
        except Exception as e:
            logger.error(f"Failed to learn meta-strategies: {e}")
    
    def _find_common_memory_patterns(self, experiences: List[LearningExperience]) -> List[str]:
        """Find common memory usage patterns across experiences."""
        memory_usage = defaultdict(int)
        
        for exp in experiences:
            for memory_id in exp.memories_used:
                memory_usage[memory_id] += 1
        
        # Find memories used in at least 50% of experiences
        threshold = len(experiences) * 0.5
        common_memories = [
            memory_id for memory_id, count in memory_usage.items()
            if count >= threshold
        ]
        
        return common_memories
    
    def _identify_successful_strategies(self, experiences: List[LearningExperience]) -> List[str]:
        """Identify successful strategies from experiences."""
        # This is a simplified version - in practice, this would analyze
        # the sequence of actions, memory usage patterns, etc.
        strategies = []
        
        # Look for common task contexts
        context_patterns = defaultdict(int)
        for exp in experiences:
            for key, value in exp.task_context.items():
                pattern = f"{key}:{value}"
                context_patterns[pattern] += 1
        
        # Find patterns that appear in most successful experiences
        threshold = len(experiences) * 0.6
        for pattern, count in context_patterns.items():
            if count >= threshold:
                strategies.append(pattern)
        
        return strategies
    
    async def _identify_transfer_opportunities(self, experience: LearningExperience):
        """Identify opportunities for cross-task transfer."""
        try:
            # Find similar tasks based on memory usage and performance patterns
            current_task = experience.task_type
            current_memories = set(experience.memories_used)
            
            for other_task, performances in self.task_performance_history.items():
                if other_task == current_task or len(performances) < 3:
                    continue
                
                # Find experiences from the other task
                other_experiences = [
                    exp for exp in self.learning_experiences.values()
                    if exp.task_type == other_task and exp.success
                ]
                
                if not other_experiences:
                    continue
                
                # Calculate memory overlap
                other_memories = set()
                for exp in other_experiences:
                    other_memories.update(exp.memories_used)
                
                overlap = len(current_memories.intersection(other_memories))
                total_unique = len(current_memories.union(other_memories))
                
                if total_unique > 0:
                    similarity = overlap / total_unique
                    
                    if similarity > self.config["transfer_threshold"]:
                        # Potential transfer opportunity
                        await self._apply_cross_task_transfer(
                            experience, other_task, other_experiences, similarity
                        )
            
        except Exception as e:
            logger.error(f"Failed to identify transfer opportunities: {e}")
    
    async def _apply_cross_task_transfer(
        self,
        current_experience: LearningExperience,
        source_task: str,
        source_experiences: List[LearningExperience],
        similarity: float
    ):
        """Apply cross-task transfer learning."""
        try:
            # Boost effectiveness of memories that were successful in source task
            successful_memories = set()
            for exp in source_experiences:
                if exp.success and exp.performance_metrics:
                    avg_perf = np.mean(list(exp.performance_metrics.values()))
                    if avg_perf > 0.7:  # High performance threshold
                        successful_memories.update(exp.memories_used)
            
            # Update memory effectiveness scores with transfer boost
            transfer_boost = similarity * 0.2  # Up to 20% boost
            
            for memory_id in successful_memories:
                if memory_id in current_experience.memories_used:
                    current_score = self.memory_effectiveness_scores.get(memory_id, 0.5)
                    boosted_score = min(1.0, current_score + transfer_boost)
                    self.memory_effectiveness_scores[memory_id] = boosted_score
            
            self.stats["successful_transfers"] += 1
            
            logger.info(
                "Cross-task transfer applied",
                agent_id=self.agent_id,
                source_task=source_task,
                target_task=current_experience.task_type,
                similarity=similarity,
                transferred_memories=len(successful_memories)
            )
            
        except Exception as e:
            logger.error(f"Failed to apply cross-task transfer: {e}")
    
    async def _update_performance_weights(self, experience: LearningExperience):
        """Update performance-driven weights for memories."""
        try:
            if not experience.performance_metrics:
                return
            
            avg_performance = np.mean(list(experience.performance_metrics.values()))
            
            # Update weights based on performance
            for memory_id in experience.memories_used:
                current_effectiveness = self.memory_effectiveness_scores.get(memory_id, 0.5)
                
                # Performance-driven update
                performance_weight = avg_performance if experience.success else avg_performance * 0.5
                learning_rate = self.config["learning_rate"]
                
                new_effectiveness = (
                    (1 - learning_rate) * current_effectiveness +
                    learning_rate * performance_weight
                )
                
                self.memory_effectiveness_scores[memory_id] = new_effectiveness
            
        except Exception as e:
            logger.error(f"Failed to update performance weights: {e}")
    
    async def get_memory_recommendations(
        self,
        task_type: str,
        task_context: Dict[str, Any],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get memory recommendations based on learning history."""
        try:
            recommendations = []
            
            # Find similar past experiences
            similar_experiences = [
                exp for exp in self.learning_experiences.values()
                if exp.task_type == task_type and exp.success
            ]
            
            # Score memories based on effectiveness and relevance
            memory_scores = defaultdict(float)
            
            for exp in similar_experiences:
                exp_performance = np.mean(list(exp.performance_metrics.values())) if exp.performance_metrics else 0.5
                
                for memory_id in exp.memories_used:
                    effectiveness = self.memory_effectiveness_scores.get(memory_id, 0.5)
                    memory_scores[memory_id] += effectiveness * exp_performance
            
            # Normalize scores
            if memory_scores:
                max_score = max(memory_scores.values())
                if max_score > 0:
                    for memory_id in memory_scores:
                        memory_scores[memory_id] /= max_score
            
            # Sort and return top recommendations
            sorted_memories = sorted(
                memory_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return sorted_memories[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to get memory recommendations: {e}")
            return []
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        # Calculate additional stats
        if self.stats["total_experiences"] > 0:
            successful_experiences = sum(
                1 for exp in self.learning_experiences.values() if exp.success
            )
            success_rate = successful_experiences / self.stats["total_experiences"]
        else:
            success_rate = 0.0
        
        # Calculate average performance improvement
        all_performances = []
        for performances in self.task_performance_history.values():
            if len(performances) > 1:
                improvement = performances[-1] - performances[0]
                all_performances.append(improvement)
        
        avg_improvement = np.mean(all_performances) if all_performances else 0.0
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "avg_performance_improvement": avg_improvement,
            "total_task_types": len(self.task_performance_history),
            "memory_effectiveness_scores_count": len(self.memory_effectiveness_scores),
            "config": self.config
        }
