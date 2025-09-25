"""
Revolutionary Active Retrieval Engine for Agentic AI Memory.

Implements automatic context-based memory retrieval without explicit search commands,
based on state-of-the-art research in agent memory systems including MIRIX and RoboMemory.

Key Features:
- Context-aware automatic retrieval
- Multi-modal similarity matching
- Temporal relevance scoring
- Emotional context consideration
- Association-based memory activation
- Importance-weighted retrieval
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import structlog

from .memory_models import (
    MemoryEntry, MemoryType, MemoryImportance, 
    RevolutionaryMemoryCollection, CoreMemoryBlock,
    ResourceMemoryEntry, KnowledgeVaultEntry
)

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalContext:
    """Context information for active memory retrieval."""
    current_task: str = ""
    conversation_context: str = ""
    emotional_state: float = 0.0  # -1.0 to 1.0
    time_context: datetime = field(default_factory=datetime.now)
    active_tags: Set[str] = field(default_factory=set)
    priority_memories: Set[str] = field(default_factory=set)
    exclude_memories: Set[str] = field(default_factory=set)
    max_memories: int = 10
    relevance_threshold: float = 0.3


@dataclass
class RetrievalResult:
    """Result of active memory retrieval."""
    memories: List[MemoryEntry]
    relevance_scores: Dict[str, float]
    retrieval_reason: Dict[str, str]  # memory_id -> reason
    context_summary: str
    total_retrieved: int
    retrieval_time_ms: float


class ActiveRetrievalEngine:
    """
    Revolutionary Active Retrieval Engine.
    
    Automatically retrieves relevant memories based on current context
    without requiring explicit search commands from the agent.
    """
    
    def __init__(self, embedding_function: Optional[callable] = None):
        """Initialize the active retrieval engine."""
        self.embedding_function = embedding_function
        self.retrieval_stats = {
            "total_retrievals": 0,
            "avg_retrieval_time_ms": 0.0,
            "avg_memories_retrieved": 0.0,
            "context_hits": 0,
            "association_hits": 0,
            "temporal_hits": 0,
            "emotional_hits": 0
        }
        
        # Retrieval weights for different factors
        self.weights = {
            "semantic_similarity": 0.3,
            "temporal_relevance": 0.2,
            "importance": 0.2,
            "emotional_alignment": 0.1,
            "association_strength": 0.15,
            "access_frequency": 0.05
        }
        
        logger.info("Active Retrieval Engine initialized")
    
    async def retrieve_active_memories(
        self,
        memory_collection: RevolutionaryMemoryCollection,
        context: RetrievalContext
    ) -> RetrievalResult:
        """
        Actively retrieve relevant memories based on current context.
        
        This is the core revolutionary feature - automatic memory retrieval
        without explicit search commands.
        """
        start_time = time.time()
        
        try:
            # Get all candidate memories
            candidate_memories = self._get_candidate_memories(memory_collection, context)
            
            if not candidate_memories:
                return RetrievalResult(
                    memories=[],
                    relevance_scores={},
                    retrieval_reason={},
                    context_summary="No relevant memories found",
                    total_retrieved=0,
                    retrieval_time_ms=0.0
                )
            
            # Score memories for relevance
            scored_memories = await self._score_memories(candidate_memories, context)
            
            # Filter by relevance threshold
            relevant_memories = [
                (memory, score) for memory, score in scored_memories
                if score >= context.relevance_threshold
            ]
            
            # Sort by relevance score (descending)
            relevant_memories.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to max_memories
            top_memories = relevant_memories[:context.max_memories]
            
            # Build result
            memories = [memory for memory, _ in top_memories]
            relevance_scores = {memory.id: score for memory, score in top_memories}
            retrieval_reasons = self._generate_retrieval_reasons(top_memories, context)
            
            # Update statistics
            retrieval_time = (time.time() - start_time) * 1000
            self._update_stats(len(memories), retrieval_time)
            
            # Generate context summary
            context_summary = self._generate_context_summary(memories, context)
            
            logger.info(
                "Active memory retrieval completed",
                agent_id=memory_collection.agent_id,
                candidates=len(candidate_memories),
                retrieved=len(memories),
                retrieval_time_ms=f"{retrieval_time:.2f}",
                avg_relevance=f"{np.mean(list(relevance_scores.values())):.3f}" if relevance_scores else "0.000"
            )
            
            return RetrievalResult(
                memories=memories,
                relevance_scores=relevance_scores,
                retrieval_reason=retrieval_reasons,
                context_summary=context_summary,
                total_retrieved=len(memories),
                retrieval_time_ms=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Active memory retrieval failed: {e}")
            return RetrievalResult(
                memories=[],
                relevance_scores={},
                retrieval_reason={},
                context_summary=f"Retrieval failed: {str(e)}",
                total_retrieved=0,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
    
    def _get_candidate_memories(
        self,
        memory_collection: RevolutionaryMemoryCollection,
        context: RetrievalContext
    ) -> List[MemoryEntry]:
        """Get candidate memories for scoring."""
        candidates = []
        
        # Collect from all memory stores
        all_stores = [
            memory_collection.short_term_memories,
            memory_collection.long_term_memories,
            memory_collection.episodic_memories,
            memory_collection.semantic_memories,
            memory_collection.procedural_memories
        ]
        
        for store in all_stores:
            for memory in store.values():
                if memory.id not in context.exclude_memories:
                    candidates.append(memory)
        
        # Add working memories
        for memory in memory_collection.working_memories:
            if memory.id not in context.exclude_memories:
                candidates.append(memory)
        
        return candidates
    
    async def _score_memories(
        self,
        memories: List[MemoryEntry],
        context: RetrievalContext
    ) -> List[Tuple[MemoryEntry, float]]:
        """Score memories for relevance to current context."""
        scored_memories = []
        
        for memory in memories:
            score = await self._calculate_relevance_score(memory, context)
            scored_memories.append((memory, score))
        
        return scored_memories
    
    async def _calculate_relevance_score(
        self,
        memory: MemoryEntry,
        context: RetrievalContext
    ) -> float:
        """Calculate comprehensive relevance score for a memory."""
        total_score = 0.0
        
        # 1. Semantic similarity (if embedding function available)
        semantic_score = await self._calculate_semantic_similarity(memory, context)
        total_score += semantic_score * self.weights["semantic_similarity"]
        
        # 2. Temporal relevance
        temporal_score = self._calculate_temporal_relevance(memory, context)
        total_score += temporal_score * self.weights["temporal_relevance"]
        
        # 3. Importance weighting
        importance_score = self._calculate_importance_score(memory)
        total_score += importance_score * self.weights["importance"]
        
        # 4. Emotional alignment
        emotional_score = self._calculate_emotional_alignment(memory, context)
        total_score += emotional_score * self.weights["emotional_alignment"]
        
        # 5. Association strength
        association_score = self._calculate_association_strength(memory, context)
        total_score += association_score * self.weights["association_strength"]
        
        # 6. Access frequency
        frequency_score = self._calculate_frequency_score(memory)
        total_score += frequency_score * self.weights["access_frequency"]
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    async def _calculate_semantic_similarity(
        self,
        memory: MemoryEntry,
        context: RetrievalContext
    ) -> float:
        """Calculate semantic similarity between memory and context."""
        if not self.embedding_function:
            # Fallback to simple keyword matching
            return self._simple_keyword_similarity(memory, context)
        
        try:
            # Use embedding function for semantic similarity
            memory_embedding = await self.embedding_function(memory.content)
            context_text = f"{context.current_task} {context.conversation_context}"
            context_embedding = await self.embedding_function(context_text)
            
            # Calculate cosine similarity
            similarity = np.dot(memory_embedding, context_embedding) / (
                np.linalg.norm(memory_embedding) * np.linalg.norm(context_embedding)
            )
            
            return max(0.0, similarity)
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return self._simple_keyword_similarity(memory, context)
    
    def _simple_keyword_similarity(self, memory: MemoryEntry, context: RetrievalContext) -> float:
        """Simple keyword-based similarity as fallback."""
        memory_words = set(memory.content.lower().split())
        context_words = set(f"{context.current_task} {context.conversation_context}".lower().split())
        
        if not context_words:
            return 0.0
        
        intersection = memory_words.intersection(context_words)
        return len(intersection) / len(context_words)
    
    def _calculate_temporal_relevance(self, memory: MemoryEntry, context: RetrievalContext) -> float:
        """Calculate temporal relevance of memory."""
        time_diff = abs((context.time_context - memory.created_at).total_seconds())
        
        # Recent memories are more relevant
        if time_diff < 3600:  # 1 hour
            return 1.0
        elif time_diff < 86400:  # 1 day
            return 0.8
        elif time_diff < 604800:  # 1 week
            return 0.6
        elif time_diff < 2592000:  # 1 month
            return 0.4
        else:
            return 0.2
    
    def _calculate_importance_score(self, memory: MemoryEntry) -> float:
        """Calculate importance-based score."""
        importance_mapping = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TEMPORARY: 0.2
        }
        return importance_mapping.get(memory.importance, 0.6)
    
    def _calculate_emotional_alignment(self, memory: MemoryEntry, context: RetrievalContext) -> float:
        """Calculate emotional alignment between memory and context."""
        if context.emotional_state == 0.0 and memory.emotional_valence == 0.0:
            return 0.5  # Neutral alignment
        
        # Calculate emotional distance
        emotional_distance = abs(context.emotional_state - memory.emotional_valence)
        
        # Convert distance to similarity (closer = more similar)
        return 1.0 - (emotional_distance / 2.0)
    
    def _calculate_association_strength(self, memory: MemoryEntry, context: RetrievalContext) -> float:
        """Calculate association strength with priority memories."""
        if not context.priority_memories:
            return 0.0
        
        max_strength = 0.0
        for priority_id in context.priority_memories:
            if priority_id in memory.associations:
                strength = memory.associations[priority_id]
                max_strength = max(max_strength, strength)
        
        return max_strength
    
    def _calculate_frequency_score(self, memory: MemoryEntry) -> float:
        """Calculate score based on access frequency."""
        if memory.access_count == 0:
            return 0.0
        
        # Normalize access count (log scale to prevent dominance)
        return min(np.log(memory.access_count + 1) / 10.0, 1.0)
    
    def _generate_retrieval_reasons(
        self,
        scored_memories: List[Tuple[MemoryEntry, float]],
        context: RetrievalContext
    ) -> Dict[str, str]:
        """Generate human-readable reasons for memory retrieval."""
        reasons = {}
        
        for memory, score in scored_memories:
            reason_parts = []
            
            if score > 0.8:
                reason_parts.append("highly relevant")
            elif score > 0.6:
                reason_parts.append("moderately relevant")
            else:
                reason_parts.append("somewhat relevant")
            
            if memory.importance == MemoryImportance.CRITICAL:
                reason_parts.append("critical importance")
            elif memory.importance == MemoryImportance.HIGH:
                reason_parts.append("high importance")
            
            if memory.access_count > 5:
                reason_parts.append("frequently accessed")
            
            # Check for tag matches
            tag_matches = memory.tags.intersection(context.active_tags)
            if tag_matches:
                reason_parts.append(f"matches tags: {', '.join(list(tag_matches)[:3])}")
            
            reasons[memory.id] = f"Retrieved due to: {', '.join(reason_parts)}"
        
        return reasons
    
    def _generate_context_summary(
        self,
        memories: List[MemoryEntry],
        context: RetrievalContext
    ) -> str:
        """Generate a summary of the retrieved memory context."""
        if not memories:
            return "No relevant memories retrieved"
        
        memory_types = defaultdict(int)
        for memory in memories:
            memory_types[memory.memory_type.value] += 1
        
        type_summary = ", ".join([f"{count} {mtype}" for mtype, count in memory_types.items()])
        
        return f"Retrieved {len(memories)} memories: {type_summary}"
    
    def _update_stats(self, retrieved_count: int, retrieval_time_ms: float):
        """Update retrieval statistics."""
        self.retrieval_stats["total_retrievals"] += 1
        
        # Update averages using exponential moving average
        alpha = 0.1
        self.retrieval_stats["avg_retrieval_time_ms"] = (
            alpha * retrieval_time_ms + 
            (1 - alpha) * self.retrieval_stats["avg_retrieval_time_ms"]
        )
        
        self.retrieval_stats["avg_memories_retrieved"] = (
            alpha * retrieved_count + 
            (1 - alpha) * self.retrieval_stats["avg_memories_retrieved"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        return self.retrieval_stats.copy()
