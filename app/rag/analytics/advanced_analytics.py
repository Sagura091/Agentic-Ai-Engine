"""
Revolutionary Advanced Analytics System for RAG 4.0.

This module provides comprehensive analytics capabilities including:
- Search pattern analysis and optimization
- Query intelligence and insights generation
- Performance analytics and recommendations
- User behavior analysis and personalization insights
- Knowledge base optimization recommendations
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re
from collections import defaultdict, Counter

import structlog
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from ..core.embeddings import EmbeddingManager
from ..core.caching import get_rag_cache, CacheType
from ..core.resilience_manager import get_resilience_manager

logger = structlog.get_logger(__name__)


class AnalyticsType(Enum):
    """Types of analytics."""
    SEARCH_PATTERNS = "search_patterns"
    QUERY_OPTIMIZATION = "query_optimization"
    USER_BEHAVIOR = "user_behavior"
    PERFORMANCE = "performance"
    KNOWLEDGE_BASE = "knowledge_base"
    CONTENT_QUALITY = "content_quality"


class InsightType(Enum):
    """Types of insights."""
    TREND = "trend"
    ANOMALY = "anomaly"
    OPTIMIZATION = "optimization"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"
    SUCCESS = "success"


@dataclass
class SearchEvent:
    """Individual search event for analytics."""
    id: str
    user_id: str
    query: str
    query_embedding: Optional[List[float]]
    results_count: int
    response_time: float
    click_through_rate: float
    satisfaction_score: Optional[float]
    context: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class SearchPattern:
    """Identified search pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    users_affected: int
    avg_satisfaction: float
    optimization_potential: float
    examples: List[str]
    metadata: Dict[str, Any]


@dataclass
class QueryOptimization:
    """Query optimization recommendation."""
    original_query: str
    optimized_query: str
    optimization_type: str
    expected_improvement: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class AnalyticsInsight:
    """Analytics insight or recommendation."""
    id: str
    insight_type: InsightType
    analytics_type: AnalyticsType
    title: str
    description: str
    impact_score: float
    confidence: float
    actionable_recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class PerformanceMetrics:
    """Performance metrics for analytics."""
    avg_response_time: float
    p95_response_time: float
    cache_hit_rate: float
    search_success_rate: float
    user_satisfaction: float
    query_complexity_score: float
    knowledge_coverage: float
    system_efficiency: float


class SearchPatternAnalyzer:
    """Analyzes search patterns and user behavior."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.search_events: List[SearchEvent] = []
        self.identified_patterns: List[SearchPattern] = []
    
    async def record_search_event(
        self,
        user_id: str,
        query: str,
        results_count: int,
        response_time: float,
        click_through_rate: float = 0.0,
        satisfaction_score: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> SearchEvent:
        """Record a search event for analysis."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_text(query)
            
            event = SearchEvent(
                id=str(uuid.uuid4()),
                user_id=user_id,
                query=query,
                query_embedding=query_embedding,
                results_count=results_count,
                response_time=response_time,
                click_through_rate=click_through_rate,
                satisfaction_score=satisfaction_score,
                context=context or {},
                timestamp=datetime.utcnow()
            )
            
            self.search_events.append(event)
            
            # Trigger pattern analysis if we have enough events
            if len(self.search_events) % 100 == 0:
                await self._analyze_patterns()
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to record search event: {str(e)}")
            raise
    
    async def analyze_search_patterns(
        self, 
        time_window: timedelta = timedelta(days=7)
    ) -> List[SearchPattern]:
        """Analyze search patterns within time window."""
        try:
            cutoff_time = datetime.utcnow() - time_window
            recent_events = [
                event for event in self.search_events 
                if event.timestamp >= cutoff_time
            ]
            
            patterns = []
            
            # Pattern 1: Frequent similar queries
            similar_query_patterns = await self._find_similar_query_patterns(recent_events)
            patterns.extend(similar_query_patterns)
            
            # Pattern 2: Low satisfaction queries
            low_satisfaction_patterns = await self._find_low_satisfaction_patterns(recent_events)
            patterns.extend(low_satisfaction_patterns)
            
            # Pattern 3: High response time queries
            slow_query_patterns = await self._find_slow_query_patterns(recent_events)
            patterns.extend(slow_query_patterns)
            
            # Pattern 4: Zero result queries
            zero_result_patterns = await self._find_zero_result_patterns(recent_events)
            patterns.extend(zero_result_patterns)
            
            # Pattern 5: Temporal patterns
            temporal_patterns = await self._find_temporal_patterns(recent_events)
            patterns.extend(temporal_patterns)
            
            self.identified_patterns = patterns
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze search patterns: {str(e)}")
            return []
    
    async def _find_similar_query_patterns(
        self, 
        events: List[SearchEvent]
    ) -> List[SearchPattern]:
        """Find patterns of similar queries."""
        patterns = []
        
        # Group queries by semantic similarity
        query_groups = defaultdict(list)
        
        for event in events:
            if not event.query_embedding:
                continue
            
            # Find similar queries
            similar_found = False
            for group_key, group_events in query_groups.items():
                if group_events:
                    # Calculate similarity with first event in group
                    similarity = self._calculate_similarity(
                        event.query_embedding, 
                        group_events[0].query_embedding
                    )
                    
                    if similarity > 0.8:  # High similarity threshold
                        query_groups[group_key].append(event)
                        similar_found = True
                        break
            
            if not similar_found:
                query_groups[event.query] = [event]
        
        # Identify patterns from groups with multiple queries
        for group_key, group_events in query_groups.items():
            if len(group_events) >= 5:  # Minimum frequency
                avg_satisfaction = np.mean([
                    e.satisfaction_score for e in group_events 
                    if e.satisfaction_score is not None
                ]) if any(e.satisfaction_score for e in group_events) else 0.5
                
                pattern = SearchPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="similar_queries",
                    description=f"Frequent similar queries: {group_key[:50]}...",
                    frequency=len(group_events),
                    users_affected=len(set(e.user_id for e in group_events)),
                    avg_satisfaction=avg_satisfaction,
                    optimization_potential=0.8 if avg_satisfaction < 0.6 else 0.3,
                    examples=[e.query for e in group_events[:3]],
                    metadata={"group_key": group_key}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _find_low_satisfaction_patterns(
        self, 
        events: List[SearchEvent]
    ) -> List[SearchPattern]:
        """Find patterns of low satisfaction queries."""
        low_satisfaction_events = [
            e for e in events 
            if e.satisfaction_score is not None and e.satisfaction_score < 0.4
        ]
        
        if len(low_satisfaction_events) < 3:
            return []
        
        # Analyze common characteristics
        common_terms = self._extract_common_terms([e.query for e in low_satisfaction_events])
        
        pattern = SearchPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type="low_satisfaction",
            description=f"Queries with low satisfaction scores (avg: {np.mean([e.satisfaction_score for e in low_satisfaction_events]):.2f})",
            frequency=len(low_satisfaction_events),
            users_affected=len(set(e.user_id for e in low_satisfaction_events)),
            avg_satisfaction=np.mean([e.satisfaction_score for e in low_satisfaction_events]),
            optimization_potential=0.9,
            examples=[e.query for e in low_satisfaction_events[:3]],
            metadata={"common_terms": common_terms}
        )
        
        return [pattern]
    
    async def _find_slow_query_patterns(
        self, 
        events: List[SearchEvent]
    ) -> List[SearchPattern]:
        """Find patterns of slow queries."""
        # Calculate response time threshold (95th percentile)
        response_times = [e.response_time for e in events]
        if not response_times:
            return []
        
        threshold = np.percentile(response_times, 95)
        slow_events = [e for e in events if e.response_time > threshold]
        
        if len(slow_events) < 3:
            return []
        
        pattern = SearchPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type="slow_queries",
            description=f"Queries with high response time (>{threshold:.2f}s)",
            frequency=len(slow_events),
            users_affected=len(set(e.user_id for e in slow_events)),
            avg_satisfaction=np.mean([
                e.satisfaction_score for e in slow_events 
                if e.satisfaction_score is not None
            ]) if any(e.satisfaction_score for e in slow_events) else 0.5,
            optimization_potential=0.7,
            examples=[e.query for e in slow_events[:3]],
            metadata={"avg_response_time": np.mean([e.response_time for e in slow_events])}
        )
        
        return [pattern]
    
    async def _find_zero_result_patterns(
        self, 
        events: List[SearchEvent]
    ) -> List[SearchPattern]:
        """Find patterns of queries returning zero results."""
        zero_result_events = [e for e in events if e.results_count == 0]
        
        if len(zero_result_events) < 3:
            return []
        
        common_terms = self._extract_common_terms([e.query for e in zero_result_events])
        
        pattern = SearchPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type="zero_results",
            description="Queries returning no results",
            frequency=len(zero_result_events),
            users_affected=len(set(e.user_id for e in zero_result_events)),
            avg_satisfaction=0.1,  # Very low satisfaction for zero results
            optimization_potential=0.95,
            examples=[e.query for e in zero_result_events[:3]],
            metadata={"common_terms": common_terms}
        )
        
        return [pattern]
    
    async def _find_temporal_patterns(
        self, 
        events: List[SearchEvent]
    ) -> List[SearchPattern]:
        """Find temporal patterns in search behavior."""
        patterns = []
        
        # Group by hour of day
        hourly_counts = defaultdict(int)
        for event in events:
            hour = event.timestamp.hour
            hourly_counts[hour] += 1
        
        # Find peak hours
        if hourly_counts:
            max_count = max(hourly_counts.values())
            peak_hours = [hour for hour, count in hourly_counts.items() if count > max_count * 0.8]
            
            if len(peak_hours) <= 3:  # Concentrated peak activity
                pattern = SearchPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="temporal_peak",
                    description=f"Peak search activity during hours: {peak_hours}",
                    frequency=sum(hourly_counts[h] for h in peak_hours),
                    users_affected=len(set(e.user_id for e in events if e.timestamp.hour in peak_hours)),
                    avg_satisfaction=0.5,  # Neutral
                    optimization_potential=0.4,
                    examples=[],
                    metadata={"peak_hours": peak_hours, "hourly_distribution": dict(hourly_counts)}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _extract_common_terms(self, queries: List[str]) -> List[str]:
        """Extract common terms from queries."""
        all_terms = []
        for query in queries:
            terms = query.lower().split()
            all_terms.extend(terms)
        
        # Count term frequency
        term_counts = Counter(all_terms)
        
        # Return terms that appear in at least 30% of queries
        min_frequency = max(1, len(queries) * 0.3)
        common_terms = [
            term for term, count in term_counts.items() 
            if count >= min_frequency and len(term) > 2
        ]
        
        return common_terms[:10]  # Top 10 common terms
    
    async def _analyze_patterns(self) -> None:
        """Periodic pattern analysis."""
        try:
            await self.analyze_search_patterns()
        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")


class QueryOptimizer:
    """Optimizes queries for better performance and results."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load query optimization rules."""
        return {
            "expansion_terms": {
                "ai": ["artificial intelligence", "machine learning", "neural networks"],
                "ml": ["machine learning", "artificial intelligence", "algorithms"],
                "nlp": ["natural language processing", "text processing", "linguistics"]
            },
            "stop_words": ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"],
            "synonyms": {
                "fast": ["quick", "rapid", "speedy"],
                "big": ["large", "huge", "massive"],
                "small": ["tiny", "little", "compact"]
            }
        }
    
    async def optimize_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryOptimization:
        """Optimize a query for better results."""
        try:
            optimizations = []
            optimized_query = query.lower().strip()
            
            # Apply optimization techniques
            
            # 1. Query expansion
            expanded_query = await self._expand_query(optimized_query)
            if expanded_query != optimized_query:
                optimizations.append("expansion")
                optimized_query = expanded_query
            
            # 2. Synonym replacement
            synonym_query = await self._replace_synonyms(optimized_query, context)
            if synonym_query != optimized_query:
                optimizations.append("synonyms")
                optimized_query = synonym_query
            
            # 3. Stop word removal (for certain contexts)
            if context and context.get("remove_stop_words", False):
                filtered_query = await self._remove_stop_words(optimized_query)
                if filtered_query != optimized_query:
                    optimizations.append("stop_word_removal")
                    optimized_query = filtered_query
            
            # 4. Query restructuring
            restructured_query = await self._restructure_query(optimized_query)
            if restructured_query != optimized_query:
                optimizations.append("restructuring")
                optimized_query = restructured_query
            
            # Calculate expected improvement
            expected_improvement = len(optimizations) * 0.15  # 15% per optimization
            expected_improvement = min(expected_improvement, 0.8)  # Cap at 80%
            
            # Generate reasoning
            reasoning = self._generate_optimization_reasoning(optimizations, query, optimized_query)
            
            optimization = QueryOptimization(
                original_query=query,
                optimized_query=optimized_query,
                optimization_type=", ".join(optimizations) if optimizations else "none",
                expected_improvement=expected_improvement,
                confidence=0.8 if optimizations else 0.1,
                reasoning=reasoning,
                metadata={"optimizations_applied": optimizations}
            )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            # Return original query as fallback
            return QueryOptimization(
                original_query=query,
                optimized_query=query,
                optimization_type="none",
                expected_improvement=0.0,
                confidence=0.0,
                reasoning="Optimization failed, using original query",
                metadata={}
            )
    
    async def _expand_query(self, query: str) -> str:
        """Expand query with related terms."""
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            
            # Check for expansion terms
            if word in self.optimization_rules["expansion_terms"]:
                expansion_terms = self.optimization_rules["expansion_terms"][word]
                # Add one expansion term
                expanded_words.append(expansion_terms[0])
        
        return " ".join(expanded_words)
    
    async def _replace_synonyms(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Replace words with better synonyms based on context."""
        words = query.split()
        replaced_words = []
        
        for word in words:
            if word in self.optimization_rules["synonyms"]:
                synonyms = self.optimization_rules["synonyms"][word]
                # Use first synonym for simplicity
                replaced_words.append(synonyms[0])
            else:
                replaced_words.append(word)
        
        return " ".join(replaced_words)
    
    async def _remove_stop_words(self, query: str) -> str:
        """Remove stop words from query."""
        words = query.split()
        filtered_words = [
            word for word in words 
            if word not in self.optimization_rules["stop_words"]
        ]
        
        return " ".join(filtered_words) if filtered_words else query
    
    async def _restructure_query(self, query: str) -> str:
        """Restructure query for better semantic understanding."""
        # Simple restructuring: move important terms to the front
        words = query.split()
        
        # Identify important terms (longer words, capitalized words)
        important_words = []
        other_words = []
        
        for word in words:
            if len(word) > 5 or word[0].isupper():
                important_words.append(word)
            else:
                other_words.append(word)
        
        # Restructure: important words first
        if important_words and len(words) > 3:
            restructured = important_words + other_words
            return " ".join(restructured)
        
        return query
    
    def _generate_optimization_reasoning(
        self, 
        optimizations: List[str], 
        original: str, 
        optimized: str
    ) -> str:
        """Generate reasoning for optimization."""
        if not optimizations:
            return "No optimizations applied - query is already well-formed"
        
        reasoning_parts = []
        
        if "expansion" in optimizations:
            reasoning_parts.append("expanded with related terms for broader coverage")
        
        if "synonyms" in optimizations:
            reasoning_parts.append("replaced words with more effective synonyms")
        
        if "stop_word_removal" in optimizations:
            reasoning_parts.append("removed stop words to focus on key terms")
        
        if "restructuring" in optimizations:
            reasoning_parts.append("restructured to prioritize important terms")
        
        return f"Query optimized by: {', '.join(reasoning_parts)}"


class PerformanceAnalyzer:
    """Analyzes system performance and generates insights."""
    
    def __init__(self):
        self.performance_history: List[PerformanceMetrics] = []
    
    async def record_performance_metrics(
        self,
        avg_response_time: float,
        p95_response_time: float,
        cache_hit_rate: float,
        search_success_rate: float,
        user_satisfaction: float,
        query_complexity_score: float,
        knowledge_coverage: float,
        system_efficiency: float
    ) -> PerformanceMetrics:
        """Record performance metrics."""
        metrics = PerformanceMetrics(
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            cache_hit_rate=cache_hit_rate,
            search_success_rate=search_success_rate,
            user_satisfaction=user_satisfaction,
            query_complexity_score=query_complexity_score,
            knowledge_coverage=knowledge_coverage,
            system_efficiency=system_efficiency
        )
        
        self.performance_history.append(metrics)
        
        # Keep only last 100 records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        return metrics
    
    async def analyze_performance_trends(self) -> List[AnalyticsInsight]:
        """Analyze performance trends and generate insights."""
        if len(self.performance_history) < 5:
            return []
        
        insights = []
        
        # Analyze response time trends
        response_times = [m.avg_response_time for m in self.performance_history[-10:]]
        if len(response_times) >= 5:
            trend = self._calculate_trend(response_times)
            if trend > 0.1:  # Increasing response time
                insights.append(AnalyticsInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.WARNING,
                    analytics_type=AnalyticsType.PERFORMANCE,
                    title="Response Time Increasing",
                    description=f"Average response time has increased by {trend*100:.1f}% recently",
                    impact_score=0.8,
                    confidence=0.9,
                    actionable_recommendations=[
                        "Review system resources and scaling",
                        "Optimize query processing pipeline",
                        "Check for performance bottlenecks"
                    ],
                    supporting_data={"trend": trend, "recent_avg": np.mean(response_times[-3:])},
                    generated_at=datetime.utcnow()
                ))
        
        # Analyze cache performance
        cache_rates = [m.cache_hit_rate for m in self.performance_history[-10:]]
        if cache_rates and np.mean(cache_rates) < 0.6:
            insights.append(AnalyticsInsight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.OPTIMIZATION,
                analytics_type=AnalyticsType.PERFORMANCE,
                title="Low Cache Hit Rate",
                description=f"Cache hit rate is {np.mean(cache_rates)*100:.1f}%, below optimal threshold",
                impact_score=0.7,
                confidence=0.8,
                actionable_recommendations=[
                    "Review cache configuration and size",
                    "Analyze cache eviction policies",
                    "Consider warming up cache with popular queries"
                ],
                supporting_data={"avg_cache_rate": np.mean(cache_rates)},
                generated_at=datetime.utcnow()
            ))
        
        # Analyze user satisfaction
        satisfaction_scores = [m.user_satisfaction for m in self.performance_history[-10:]]
        if satisfaction_scores and np.mean(satisfaction_scores) < 0.7:
            insights.append(AnalyticsInsight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.WARNING,
                analytics_type=AnalyticsType.USER_BEHAVIOR,
                title="Low User Satisfaction",
                description=f"User satisfaction is {np.mean(satisfaction_scores)*100:.1f}%, indicating potential issues",
                impact_score=0.9,
                confidence=0.8,
                actionable_recommendations=[
                    "Review search result quality and relevance",
                    "Analyze user feedback and pain points",
                    "Improve query understanding and processing"
                ],
                supporting_data={"avg_satisfaction": np.mean(satisfaction_scores)},
                generated_at=datetime.utcnow()
            ))
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by mean value
        mean_value = np.mean(values)
        if mean_value == 0:
            return 0.0
        
        return slope / mean_value


class AdvancedAnalyticsEngine:
    """
    Revolutionary advanced analytics engine for RAG 4.0.
    
    Features:
    - Search pattern analysis and optimization
    - Query intelligence and insights generation
    - Performance analytics and recommendations
    - User behavior analysis and personalization insights
    - Knowledge base optimization recommendations
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        
        # Component analyzers
        self.pattern_analyzer = SearchPatternAnalyzer(embedding_manager)
        self.query_optimizer = QueryOptimizer(embedding_manager)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Cache and resilience
        self.cache = None
        self.resilience_manager = None
        
        # Generated insights
        self.insights: List[AnalyticsInsight] = []
    
    async def initialize(self) -> None:
        """Initialize the analytics engine."""
        try:
            self.cache = await get_rag_cache()
            self.resilience_manager = await get_resilience_manager()
            
            await self.resilience_manager.register_component(
                "analytics_engine",
                recovery_strategies=["retry", "graceful_degradation"]
            )
            
            logger.info("Advanced analytics engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics engine: {str(e)}")
            raise
    
    async def generate_comprehensive_insights(self) -> List[AnalyticsInsight]:
        """Generate comprehensive analytics insights."""
        try:
            all_insights = []
            
            # Search pattern insights
            patterns = await self.pattern_analyzer.analyze_search_patterns()
            pattern_insights = await self._generate_pattern_insights(patterns)
            all_insights.extend(pattern_insights)
            
            # Performance insights
            performance_insights = await self.performance_analyzer.analyze_performance_trends()
            all_insights.extend(performance_insights)
            
            # Knowledge base insights
            kb_insights = await self._generate_knowledge_base_insights()
            all_insights.extend(kb_insights)
            
            # User behavior insights
            behavior_insights = await self._generate_user_behavior_insights()
            all_insights.extend(behavior_insights)
            
            # Sort by impact score
            all_insights.sort(key=lambda x: x.impact_score, reverse=True)
            
            self.insights = all_insights
            return all_insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            return []
    
    async def _generate_pattern_insights(self, patterns: List[SearchPattern]) -> List[AnalyticsInsight]:
        """Generate insights from search patterns."""
        insights = []
        
        for pattern in patterns:
            if pattern.optimization_potential > 0.7:
                insight = AnalyticsInsight(
                    id=str(uuid.uuid4()),
                    insight_type=InsightType.OPTIMIZATION,
                    analytics_type=AnalyticsType.SEARCH_PATTERNS,
                    title=f"High Optimization Potential: {pattern.pattern_type}",
                    description=pattern.description,
                    impact_score=pattern.optimization_potential,
                    confidence=0.8,
                    actionable_recommendations=self._generate_pattern_recommendations(pattern),
                    supporting_data=asdict(pattern),
                    generated_at=datetime.utcnow()
                )
                insights.append(insight)
        
        return insights
    
    def _generate_pattern_recommendations(self, pattern: SearchPattern) -> List[str]:
        """Generate recommendations for a search pattern."""
        recommendations = []
        
        if pattern.pattern_type == "similar_queries":
            recommendations.extend([
                "Create query templates for common patterns",
                "Implement auto-complete suggestions",
                "Add query expansion for related terms"
            ])
        elif pattern.pattern_type == "low_satisfaction":
            recommendations.extend([
                "Review and improve result ranking algorithms",
                "Enhance content quality in knowledge base",
                "Implement user feedback collection"
            ])
        elif pattern.pattern_type == "slow_queries":
            recommendations.extend([
                "Optimize query processing pipeline",
                "Implement query caching for complex queries",
                "Review indexing strategies"
            ])
        elif pattern.pattern_type == "zero_results":
            recommendations.extend([
                "Expand knowledge base coverage",
                "Implement fuzzy search capabilities",
                "Add query suggestion system"
            ])
        
        return recommendations
    
    async def _generate_knowledge_base_insights(self) -> List[AnalyticsInsight]:
        """Generate knowledge base optimization insights."""
        # Placeholder for knowledge base analysis
        return [
            AnalyticsInsight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.RECOMMENDATION,
                analytics_type=AnalyticsType.KNOWLEDGE_BASE,
                title="Knowledge Base Coverage Analysis",
                description="Regular analysis of knowledge base coverage and gaps",
                impact_score=0.6,
                confidence=0.7,
                actionable_recommendations=[
                    "Conduct regular content audits",
                    "Identify and fill knowledge gaps",
                    "Update outdated content"
                ],
                supporting_data={},
                generated_at=datetime.utcnow()
            )
        ]
    
    async def _generate_user_behavior_insights(self) -> List[AnalyticsInsight]:
        """Generate user behavior insights."""
        # Placeholder for user behavior analysis
        return [
            AnalyticsInsight(
                id=str(uuid.uuid4()),
                insight_type=InsightType.TREND,
                analytics_type=AnalyticsType.USER_BEHAVIOR,
                title="User Engagement Patterns",
                description="Analysis of user engagement and interaction patterns",
                impact_score=0.5,
                confidence=0.6,
                actionable_recommendations=[
                    "Personalize search experience",
                    "Implement user onboarding",
                    "Optimize user interface"
                ],
                supporting_data={},
                generated_at=datetime.utcnow()
            )
        ]
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        try:
            # Generate fresh insights
            insights = await self.generate_comprehensive_insights()
            
            # Categorize insights
            insight_categories = defaultdict(list)
            for insight in insights:
                insight_categories[insight.analytics_type.value].append(insight)
            
            # Calculate summary metrics
            total_search_events = len(self.pattern_analyzer.search_events)
            identified_patterns = len(self.pattern_analyzer.identified_patterns)
            high_impact_insights = len([i for i in insights if i.impact_score > 0.7])
            
            summary = {
                "total_insights": len(insights),
                "high_impact_insights": high_impact_insights,
                "insight_categories": {
                    category: len(category_insights) 
                    for category, category_insights in insight_categories.items()
                },
                "search_analytics": {
                    "total_events": total_search_events,
                    "identified_patterns": identified_patterns,
                    "recent_events": len([
                        e for e in self.pattern_analyzer.search_events 
                        if (datetime.utcnow() - e.timestamp).days <= 1
                    ])
                },
                "performance_metrics": (
                    asdict(self.performance_analyzer.performance_history[-1]) 
                    if self.performance_analyzer.performance_history 
                    else {}
                ),
                "top_insights": [asdict(insight) for insight in insights[:5]],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {str(e)}")
            return {}


# Global analytics engine instance
analytics_engine = None


async def get_analytics_engine(embedding_manager: EmbeddingManager) -> AdvancedAnalyticsEngine:
    """Get the global analytics engine instance."""
    global analytics_engine
    
    if analytics_engine is None:
        analytics_engine = AdvancedAnalyticsEngine(embedding_manager)
        await analytics_engine.initialize()
    
    return analytics_engine
