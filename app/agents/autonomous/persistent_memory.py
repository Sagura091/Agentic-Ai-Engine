"""
Persistent Memory System for Autonomous Agents.

This module implements a comprehensive memory system that enables agents to:
- Store and retrieve episodic memories (experiences)
- Maintain semantic memories (learned concepts and facts)
- Manage working memory for current context
- Consolidate memories for long-term retention
- Learn from memory patterns and associations
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict

import structlog
import numpy as np
from langchain_core.language_models import BaseLanguageModel

from app.memory.memory_models import MemoryEntry as AgentMemoryEntry
# from app.services.autonomous_persistence import autonomous_persistence  # Optional service

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the agent's memory system."""
    EPISODIC = "episodic"           # Specific experiences and events
    SEMANTIC = "semantic"           # General knowledge and concepts
    PROCEDURAL = "procedural"       # Skills and procedures
    WORKING = "working"             # Current context and temporary information
    EMOTIONAL = "emotional"         # Emotional associations and responses


class MemoryImportance(str, Enum):
    """Importance levels for memory consolidation."""
    CRITICAL = "critical"           # Must be retained permanently
    HIGH = "high"                   # Important for long-term retention
    MEDIUM = "medium"               # Moderate importance
    LOW = "low"                     # Can be forgotten if space is needed
    TEMPORARY = "temporary"         # Short-term only


@dataclass
class MemoryTrace:
    """Represents a memory trace with associations and metadata."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: MemoryImportance = MemoryImportance.MEDIUM
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    confidence: float = 1.0
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    
    # Associations and context
    tags: Set[str] = field(default_factory=set)
    associations: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Consolidation information
    consolidation_level: int = 0  # 0 = new, higher = more consolidated
    decay_rate: float = 0.1
    
    # Metadata
    source: str = "agent"
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersistentMemorySystem:
    """
    Comprehensive memory system for autonomous agents.
    
    Implements multiple memory types with consolidation, retrieval,
    and learning capabilities for truly persistent agentic behavior.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseLanguageModel,
        max_working_memory: int = 20,
        max_episodic_memory: int = 10000,
        max_semantic_memory: int = 5000,
        consolidation_threshold: int = 5,
        persistence_service = None
    ):
        """
        Initialize the persistent memory system.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: Language model for memory processing
            max_working_memory: Maximum working memory items
            max_episodic_memory: Maximum episodic memory items
            max_semantic_memory: Maximum semantic memory items
            consolidation_threshold: Access count threshold for consolidation
        """
        self.agent_id = agent_id
        self.llm = llm
        self.max_working_memory = max_working_memory
        self.max_episodic_memory = max_episodic_memory
        self.max_semantic_memory = max_semantic_memory
        self.consolidation_threshold = consolidation_threshold
        self.persistence_service = persistence_service
        
        # Memory stores
        self.working_memory: deque = deque(maxlen=max_working_memory)
        self.episodic_memory: Dict[str, MemoryTrace] = {}
        self.semantic_memory: Dict[str, MemoryTrace] = {}
        self.procedural_memory: Dict[str, MemoryTrace] = {}
        
        # Memory indices for fast retrieval
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date -> memory_ids
        self.association_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Memory statistics
        self.memory_stats = {
            "total_memories": 0,
            "memories_created": 0,
            "memories_retrieved": 0,
            "memories_consolidated": 0,
            "memories_forgotten": 0,
            "consolidation_cycles": 0
        }
        
        # State tracking
        self.is_initialized = False
        self.last_consolidation = None
        
        logger.info(
            "Persistent Memory System initialized",
            agent_id=agent_id,
            max_working=max_working_memory,
            max_episodic=max_episodic_memory,
            max_semantic=max_semantic_memory
        )
    
    async def initialize(self) -> bool:
        """
        Initialize the memory system and load persistent memories.
        
        Returns:
            True if initialization successful
        """
        try:
            # Load persistent memories from database
            await self._load_persistent_memories()
            
            # Rebuild indices
            await self._rebuild_indices()
            
            self.is_initialized = True
            
            logger.info(
                "Memory system initialized",
                agent_id=self.agent_id,
                episodic_count=len(self.episodic_memory),
                semantic_count=len(self.semantic_memory),
                procedural_count=len(self.procedural_memory)
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize memory system", agent_id=self.agent_id, error=str(e))
            return False

    async def _load_persistent_memories(self) -> bool:
        """Load persistent memories from database."""
        try:
            if not self.persistence_service:
                logger.warning("No persistence service available", agent_id=self.agent_id)
                return False

            # Load memories from database
            memories = await self.persistence_service.load_agent_memories(self.agent_id)

            for memory_data in memories:
                memory = MemoryTrace(
                    memory_id=memory_data.memory_id,
                    content=memory_data.content,
                    memory_type=MemoryType(memory_data.memory_type),
                    importance=MemoryImportance(memory_data.importance),
                    emotional_valence=memory_data.emotional_valence,
                    tags=memory_data.tags or [],
                    context=memory_data.context or {},
                    created_at=memory_data.created_at,
                    expires_at=memory_data.expires_at
                )

                # Add to appropriate memory store
                if memory.memory_type == MemoryType.EPISODIC:
                    self.episodic_memory[memory.memory_id] = memory
                elif memory.memory_type == MemoryType.SEMANTIC:
                    self.semantic_memory[memory.memory_id] = memory
                elif memory.memory_type == MemoryType.PROCEDURAL:
                    self.procedural_memory[memory.memory_id] = memory
                elif memory.memory_type == MemoryType.WORKING:
                    self.working_memory[memory.memory_id] = memory

            logger.info("Persistent memories loaded",
                       agent_id=self.agent_id,
                       memories_loaded=len(memories))
            return True

        except Exception as e:
            logger.error("Failed to load persistent memories", agent_id=self.agent_id, error=str(e))
            return False

    async def _update_indices(self, memory: MemoryTrace) -> None:
        """Update memory indices for fast retrieval."""
        try:
            # Update tag index
            for tag in memory.tags:
                self.tag_index[tag].add(memory.memory_id)

            # Update temporal index
            date_key = memory.created_at.strftime("%Y-%m-%d")
            self.temporal_index[date_key].append(memory.memory_id)

            # Update content index (simple keyword extraction)
            content_words = memory.content.lower().split()
            for word in content_words:
                if len(word) > 3:  # Only index meaningful words
                    self.tag_index[f"content:{word}"].add(memory.memory_id)

            # Update context index
            for key, value in memory.context.items():
                context_key = f"context:{key}:{str(value).lower()}"
                self.tag_index[context_key].add(memory.memory_id)

            logger.debug("Memory indices updated",
                        memory_id=memory.memory_id,
                        tags_indexed=len(memory.tags),
                        content_words=len(content_words))

        except Exception as e:
            logger.error("Failed to update memory indices",
                        memory_id=memory.memory_id,
                        error=str(e))

    async def _create_associations(self, memory: MemoryTrace) -> None:
        """Create associations between memories based on content similarity."""
        try:
            # Simple association based on shared tags and content keywords
            content_words = set(memory.content.lower().split())
            memory_tags = set(memory.tags)

            # Find related memories
            for existing_id, existing_memory in self.episodic_memory.items():
                if existing_id == memory.memory_id:
                    continue

                # Calculate similarity based on shared tags
                existing_tags = set(existing_memory.tags)
                tag_similarity = len(memory_tags & existing_tags) / max(len(memory_tags | existing_tags), 1)

                # Calculate content similarity (simple word overlap)
                existing_words = set(existing_memory.content.lower().split())
                content_similarity = len(content_words & existing_words) / max(len(content_words | existing_words), 1)

                # Combined similarity score
                similarity = (tag_similarity * 0.6) + (content_similarity * 0.4)

                if similarity > 0.3:  # Threshold for creating association
                    # Create bidirectional association
                    self.association_graph[memory.memory_id][existing_id] = similarity
                    self.association_graph[existing_id][memory.memory_id] = similarity

            # Also check semantic and procedural memories
            for memory_store in [self.semantic_memory, self.procedural_memory]:
                for existing_id, existing_memory in memory_store.items():
                    existing_tags = set(existing_memory.tags)
                    tag_similarity = len(memory_tags & existing_tags) / max(len(memory_tags | existing_tags), 1)

                    if tag_similarity > 0.4:  # Higher threshold for cross-type associations
                        self.association_graph[memory.memory_id][existing_id] = tag_similarity
                        self.association_graph[existing_id][memory.memory_id] = tag_similarity

            logger.debug("Memory associations created",
                        memory_id=memory.memory_id,
                        associations_count=len(self.association_graph[memory.memory_id]))

        except Exception as e:
            logger.error("Failed to create memory associations",
                        memory_id=memory.memory_id,
                        error=str(e))

    async def _rebuild_indices(self) -> None:
        """Rebuild all memory indices for consistency."""
        try:
            # Clear existing indices
            self.tag_index.clear()
            self.temporal_index.clear()
            self.association_graph.clear()

            # Rebuild indices for all memory types
            all_memories = []
            all_memories.extend(self.episodic_memory.values())
            all_memories.extend(self.semantic_memory.values())
            all_memories.extend(self.procedural_memory.values())
            all_memories.extend(list(self.working_memory))  # working_memory is a deque

            for memory in all_memories:
                await self._update_indices(memory)
                await self._create_associations(memory)

            logger.info("Memory indices rebuilt",
                       agent_id=self.agent_id,
                       total_memories=len(all_memories),
                       total_associations=sum(len(assocs) for assocs in self.association_graph.values()))

        except Exception as e:
            logger.error("Failed to rebuild memory indices",
                        agent_id=self.agent_id,
                        error=str(e))

    async def _persist_memory(self, memory: MemoryTrace) -> bool:
        """Persist memory to database if persistence service is available."""
        try:
            if not self.persistence_service:
                logger.debug("No persistence service available", memory_id=memory.memory_id)
                return False

            # Convert memory to database format
            memory_data = {
                "agent_id": self.agent_id,
                "memory_id": memory.memory_id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "importance": memory.importance.value,
                "emotional_valence": memory.emotional_valence,
                "tags": memory.tags,
                "context": memory.context,
                "created_at": memory.created_at,
                "expires_at": memory.expires_at
            }

            # Save to database
            await self.persistence_service.save_agent_memory(memory_data)

            logger.debug("Memory persisted to database", memory_id=memory.memory_id)
            return True

        except Exception as e:
            logger.error("Failed to persist memory",
                        memory_id=memory.memory_id,
                        error=str(e))
            return False

    async def _get_associated_memories(self, memory_id: str, max_associations: int = 5) -> List[MemoryTrace]:
        """Get memories associated with the given memory ID."""
        try:
            associated_memories = []

            if memory_id not in self.association_graph:
                return associated_memories

            # Get associated memory IDs sorted by similarity score
            associations = self.association_graph[memory_id]
            sorted_associations = sorted(associations.items(), key=lambda x: x[1], reverse=True)

            # Retrieve the actual memory objects
            for assoc_id, similarity in sorted_associations[:max_associations]:
                memory = None

                # Check all memory stores
                if assoc_id in self.episodic_memory:
                    memory = self.episodic_memory[assoc_id]
                elif assoc_id in self.semantic_memory:
                    memory = self.semantic_memory[assoc_id]
                elif assoc_id in self.procedural_memory:
                    memory = self.procedural_memory[assoc_id]
                else:
                    # Check working memory (deque)
                    for working_mem in self.working_memory:
                        if working_mem.memory_id == assoc_id:
                            memory = working_mem
                            break

                if memory:
                    associated_memories.append(memory)

            logger.debug("Associated memories retrieved",
                        memory_id=memory_id,
                        associations_count=len(associated_memories))

            return associated_memories

        except Exception as e:
            logger.error("Failed to get associated memories",
                        memory_id=memory_id,
                        error=str(e))
            return []
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        emotional_valence: float = 0.0,
        tags: Optional[Set[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Store a new memory in the system.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            emotional_valence: Emotional association (-1.0 to 1.0)
            tags: Associated tags
            context: Additional context
            session_id: Session identifier
            
        Returns:
            Memory ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Create memory trace
            memory = MemoryTrace(
                content=content,
                memory_type=memory_type,
                importance=importance,
                emotional_valence=emotional_valence,
                tags=tags or set(),
                context=context or {},
                session_id=session_id
            )
            
            # Store in appropriate memory store
            if memory_type == MemoryType.WORKING:
                self.working_memory.append(memory)
            elif memory_type == MemoryType.EPISODIC:
                self.episodic_memory[memory.memory_id] = memory
            elif memory_type == MemoryType.SEMANTIC:
                self.semantic_memory[memory.memory_id] = memory
            elif memory_type == MemoryType.PROCEDURAL:
                self.procedural_memory[memory.memory_id] = memory
            
            # Update indices
            await self._update_indices(memory)
            
            # Find and create associations
            await self._create_associations(memory)
            
            # Persist to database if not working memory
            if memory_type != MemoryType.WORKING:
                await self._persist_memory(memory)
            
            self.memory_stats["memories_created"] += 1
            self.memory_stats["total_memories"] += 1
            
            logger.debug(
                "Memory stored",
                agent_id=self.agent_id,
                memory_id=memory.memory_id,
                memory_type=memory_type.value,
                importance=importance.value
            )
            
            return memory.memory_id
            
        except Exception as e:
            logger.error("Failed to store memory", agent_id=self.agent_id, error=str(e))
            raise
    
    async def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[Set[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        importance_threshold: Optional[MemoryImportance] = None,
        max_results: int = 10,
        include_associations: bool = True
    ) -> List[MemoryTrace]:
        """
        Retrieve memories based on query and filters.
        
        Args:
            query: Search query
            memory_types: Types of memory to search
            tags: Required tags
            time_range: Time range filter (start, end)
            importance_threshold: Minimum importance level
            max_results: Maximum number of results
            include_associations: Include associated memories
            
        Returns:
            List of matching memory traces
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Collect candidate memories
            candidates = []
            
            # Search in specified memory types
            search_types = memory_types or [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
            
            for memory_type in search_types:
                if memory_type == MemoryType.EPISODIC:
                    candidates.extend(self.episodic_memory.values())
                elif memory_type == MemoryType.SEMANTIC:
                    candidates.extend(self.semantic_memory.values())
                elif memory_type == MemoryType.PROCEDURAL:
                    candidates.extend(self.procedural_memory.values())
                elif memory_type == MemoryType.WORKING:
                    candidates.extend(self.working_memory)
            
            # Apply filters
            filtered_memories = []
            
            for memory in candidates:
                # Tag filter
                if tags and not tags.intersection(memory.tags):
                    continue
                
                # Time range filter
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= memory.created_at <= end_time):
                        continue
                
                # Importance filter
                if importance_threshold:
                    importance_levels = {
                        MemoryImportance.TEMPORARY: 0,
                        MemoryImportance.LOW: 1,
                        MemoryImportance.MEDIUM: 2,
                        MemoryImportance.HIGH: 3,
                        MemoryImportance.CRITICAL: 4
                    }
                    if importance_levels[memory.importance] < importance_levels[importance_threshold]:
                        continue
                
                # Content relevance (improved keyword matching)
                query_words = set(query.lower().split())
                content_words = set(memory.content.lower().split())
                tag_words = set(tag.lower() for tag in memory.tags)

                # Check if query matches content, tags, or context
                content_match = query.lower() in memory.content.lower()
                word_overlap = len(query_words & content_words) > 0
                tag_match = len(query_words & tag_words) > 0
                context_match = any(query.lower() in str(v).lower() for v in memory.context.values())

                if content_match or word_overlap or tag_match or context_match:
                    filtered_memories.append(memory)
            
            # Sort by relevance and recency
            def relevance_score(memory: MemoryTrace) -> float:
                # Simple scoring based on content match, importance, and recency
                content_score = memory.content.lower().count(query.lower()) / len(memory.content.split())
                importance_score = {
                    MemoryImportance.TEMPORARY: 0.1,
                    MemoryImportance.LOW: 0.3,
                    MemoryImportance.MEDIUM: 0.5,
                    MemoryImportance.HIGH: 0.8,
                    MemoryImportance.CRITICAL: 1.0
                }[memory.importance]
                
                # Recency score (more recent = higher score)
                days_old = (datetime.utcnow() - memory.created_at).days
                recency_score = max(0.1, 1.0 / (1.0 + days_old * 0.1))
                
                # Access frequency score
                frequency_score = min(1.0, memory.access_count * 0.1)
                
                return content_score * 0.4 + importance_score * 0.3 + recency_score * 0.2 + frequency_score * 0.1
            
            filtered_memories.sort(key=relevance_score, reverse=True)
            
            # Limit results
            results = filtered_memories[:max_results]
            
            # Update access statistics
            for memory in results:
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1
            
            # Include associated memories if requested
            if include_associations and results:
                associated_memories = []
                for memory in results:
                    assoc_mems = await self._get_associated_memories(memory.memory_id, max_results // 4)
                    associated_memories.extend(assoc_mems)

                # Remove duplicates and add to results
                existing_ids = {m.memory_id for m in results}
                unique_associated = [m for m in associated_memories if m.memory_id not in existing_ids]
                results.extend(unique_associated)
            
            self.memory_stats["memories_retrieved"] += len(results)
            
            logger.debug(
                "Memories retrieved",
                agent_id=self.agent_id,
                query=query,
                results_count=len(results),
                candidates_count=len(candidates)
            )
            
            return results
            
        except Exception as e:
            logger.error("Failed to retrieve memories", agent_id=self.agent_id, error=str(e))
            return []
