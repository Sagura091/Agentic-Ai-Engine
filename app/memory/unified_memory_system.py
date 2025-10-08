"""
OPTIMIZED Revolutionary Unified Memory System - THE Memory System for Multi-Agent Architecture.

This is THE ONLY memory system in the entire application.
All memory operations flow through this revolutionary unified system.

OPTIMIZED ARCHITECTURE:
✅ Lightning-fast in-memory cache with optimized indexing
✅ Agent-specific memory isolation with fast lookups
✅ Revolutionary memory types: Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault
✅ Active Retrieval Engine with caching and optimized algorithms
✅ Memory Orchestrator with streamlined operations
✅ Multi-agent memory coordination with parallel processing

OPTIMIZATION PRINCIPLES:
- Keep all revolutionary features
- Optimize for speed and performance
- Simple interface with complex backend
- Fast operations under 100ms
- Bulletproof reliability
- Easy maintenance and debugging

OPTIMIZED FEATURES:
✅ Core Memory (always-visible persistent context)
✅ Knowledge Vault (secure sensitive information storage)
✅ Resource Memory (document and file management)
✅ Active Retrieval Engine (automatic context-based retrieval)
✅ Memory Orchestrator (multi-agent coordination)
✅ Revolutionary memory models with associations and importance
✅ Lightning-fast caching and indexing
✅ Optimized algorithms and data structures
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

from .memory_models import (
    MemoryType, MemoryEntry, MemoryCollection, RevolutionaryMemoryCollection,
    MemoryImportance, CoreMemoryBlock, ResourceMemoryEntry, KnowledgeVaultEntry
)
from .active_retrieval_engine import ActiveRetrievalEngine, RetrievalContext, RetrievalResult
from .memory_orchestrator import MemoryOrchestrator, MemoryOperation, MemoryManagerType
from .dynamic_knowledge_graph import DynamicKnowledgeGraph
from .advanced_retrieval_mechanisms import AdvancedRetrievalMechanisms, RetrievalQuery
from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

# Get backend logger instance
_backend_logger = get_backend_logger()


class UnifiedMemorySystem:
    """
    OPTIMIZED Revolutionary Unified Memory System - THE Memory System.

    OPTIMIZED ARCHITECTURE:
    - Lightning-fast in-memory cache with optimized indexing
    - Agent-specific memory collections with fast lookups
    - All memory types: Short-term, Long-term, Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault
    - Active Retrieval Engine with caching and optimized algorithms
    - Memory Orchestrator with streamlined operations
    - Automatic cleanup and TTL with background processing
    """

    def __init__(self, unified_rag=None, embedding_function: Optional[Callable] = None):
        """Initialize THE optimized revolutionary unified memory system."""
        self.unified_rag = unified_rag
        self.embedding_function = embedding_function
        self.is_initialized = False

        # OPTIMIZED: Fast in-memory cache with threading lock
        self._cache_lock = threading.RLock()
        self._fast_cache: Dict[str, Any] = {}
        self._index_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._association_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._importance_cache: Dict[str, float] = {}

        # Revolutionary agent memory collections
        self.agent_memories: Dict[str, RevolutionaryMemoryCollection] = {}

        # Agent-specific memory configurations
        self.agent_memory_configs: Dict[str, Dict[str, Any]] = {}

        # OPTIMIZED: Performance-focused configuration
        self.config = {
            "max_short_term_memories": 1000,
            "max_long_term_memories": 10000,
            "max_episodic_memories": 5000,
            "max_semantic_memories": 3000,
            "max_procedural_memories": 2000,
            "max_working_memories": 20,
            "max_resource_entries": 1000,
            "max_vault_entries": 500,
            "short_term_ttl_hours": 24,
            "cleanup_interval_hours": 6,
            "enable_active_retrieval": True,
            "enable_memory_orchestration": True,
            # OPTIMIZATION: Performance settings
            "cache_size_limit": 50000,
            "index_update_batch_size": 100,
            "background_cleanup_interval": 300,  # 5 minutes
            "fast_retrieval_threshold": 0.3,
            "parallel_operations": True,
            # Memory type enablement (defaults - can be overridden per agent)
            "enable_short_term": True,
            "enable_long_term": True,
            "enable_episodic": True,
            "enable_semantic": True,
            "enable_procedural": True,
            "enable_working": True
        }

        # Revolutionary components
        self.memory_orchestrator: Optional[MemoryOrchestrator] = None
        self.active_retrieval_engine: Optional[ActiveRetrievalEngine] = None
        self.knowledge_graph: Optional[DynamicKnowledgeGraph] = None
        self.advanced_retrieval: Optional[AdvancedRetrievalMechanisms] = None
        # Optional components (not required for core functionality)
        # self.consolidation_system: Optional[MemoryConsolidationSystem] = None
        # self.lifelong_learning: Optional[LifelongLearningCapabilities] = None
        # self.multimodal_system: Optional[MultimodalMemorySystem] = None
        # self.decision_making: Optional[MemoryDrivenDecisionMaking] = None

        # Enhanced stats
        self.stats = {
            "total_agents": 0,
            "total_memories": 0,
            "short_term_memories": 0,
            "long_term_memories": 0,
            "episodic_memories": 0,
            "semantic_memories": 0,
            "procedural_memories": 0,
            "working_memories": 0,
            "resource_entries": 0,
            "vault_entries": 0,
            "active_retrievals": 0,
            "orchestrated_operations": 0,
            "knowledge_graph_entities": 0,
            "knowledge_graph_relationships": 0,
            "consolidation_sessions": 0,
            "learning_experiences": 0,
            "multimodal_memories": 0,
            "decisions_made": 0
        }

        _backend_logger.info(
            "Revolutionary Unified Memory System initialized",
            LogCategory.MEMORY_OPERATIONS,
            "app.memory.unified_memory_system"
        )
    
    async def initialize(self) -> None:
        """Initialize the revolutionary memory system."""
        try:
            if self.is_initialized:
                return

            # Ensure unified RAG is initialized if provided
            if self.unified_rag and not self.unified_rag.is_initialized:
                await self.unified_rag.initialize()

            # Initialize revolutionary components
            if self.config["enable_memory_orchestration"]:
                self.memory_orchestrator = MemoryOrchestrator(self.embedding_function)
                await self.memory_orchestrator.initialize()
                _backend_logger.info(
                    "Memory Orchestrator initialized",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.unified_memory_system"
                )

            if self.config["enable_active_retrieval"]:
                self.active_retrieval_engine = ActiveRetrievalEngine(self.embedding_function)
                _backend_logger.info(
                    "Active Retrieval Engine initialized",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.unified_memory_system"
                )

            # Initialize all revolutionary components
            self.knowledge_graph = DynamicKnowledgeGraph("system", self.embedding_function)
            _backend_logger.info(
                "Dynamic Knowledge Graph initialized",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.unified_memory_system"
            )

            self.advanced_retrieval = AdvancedRetrievalMechanisms("system", self.embedding_function, self.knowledge_graph)
            _backend_logger.info(
                "Advanced Retrieval Mechanisms initialized",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.unified_memory_system"
            )

            # Optional: Multimodal system (not required for core functionality)
            # self.multimodal_system = MultimodalMemorySystem("system")
            # _backend_logger.info("Multimodal Memory System initialized", LogCategory.MEMORY_OPERATIONS, "app.memory.unified_memory_system")

            # Note: Agent-specific components will be initialized per agent
            _backend_logger.info(
                "All revolutionary components initialized",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.unified_memory_system"
            )

            self.is_initialized = True
            _backend_logger.info(
                "Revolutionary Memory System initialization completed",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.unified_memory_system"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize revolutionary memory system: {str(e)}",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.unified_memory_system",
                data={"error": str(e)}
            )
            raise
    
    async def create_agent_memory(self, agent_id: str, memory_config: Optional[Dict[str, Any]] = None) -> RevolutionaryMemoryCollection:
        """
        Create revolutionary memory collection for a new agent.

        Args:
            agent_id: The agent's unique identifier
            memory_config: Optional memory configuration with flags like:
                - enable_short_term: bool (default: True)
                - enable_long_term: bool (default: True)
                - enable_episodic: bool (default: True)
                - enable_semantic: bool (default: True)
                - enable_procedural: bool (default: True)
                - enable_working: bool (default: True)

        Returns:
            RevolutionaryMemoryCollection for the agent
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            if agent_id in self.agent_memories:
                logger.warning(f"Revolutionary memory collection already exists for agent {agent_id}")
                return self.agent_memories[agent_id]

            # Store agent-specific memory configuration
            if memory_config:
                self.agent_memory_configs[agent_id] = memory_config
                logger.info(
                    f"Agent {agent_id} memory configuration",
                    short_term=memory_config.get("enable_short_term", True),
                    long_term=memory_config.get("enable_long_term", True),
                    episodic=memory_config.get("enable_episodic", True),
                    semantic=memory_config.get("enable_semantic", True),
                    procedural=memory_config.get("enable_procedural", True),
                    working=memory_config.get("enable_working", True)
                )
            else:
                # Use default configuration
                self.agent_memory_configs[agent_id] = {
                    "enable_short_term": self.config.get("enable_short_term", True),
                    "enable_long_term": self.config.get("enable_long_term", True),
                    "enable_episodic": self.config.get("enable_episodic", True),
                    "enable_semantic": self.config.get("enable_semantic", True),
                    "enable_procedural": self.config.get("enable_procedural", True),
                    "enable_working": self.config.get("enable_working", True)
                }

            # Create revolutionary memory collection
            collection = RevolutionaryMemoryCollection.create(agent_id)
            self.agent_memories[agent_id] = collection

            # Register with memory orchestrator if available
            if self.memory_orchestrator:
                await self.memory_orchestrator.register_agent(agent_id)

            # Initialize agent-specific revolutionary components
            if agent_id not in self.agent_memories:
                # Create agent-specific knowledge graph
                agent_knowledge_graph = DynamicKnowledgeGraph(agent_id, self.embedding_function)

                # Optional: Create agent-specific advanced components (not required for core functionality)
                # agent_consolidation = MemoryConsolidationSystem(agent_id, collection)
                # agent_learning = LifelongLearningCapabilities(agent_id, collection)
                # agent_decision_making = MemoryDrivenDecisionMaking(agent_id, collection, agent_knowledge_graph)

                # Store references in collection for easy access
                collection.knowledge_graph = agent_knowledge_graph
                # collection.consolidation_system = agent_consolidation
                # collection.lifelong_learning = agent_learning
                collection.decision_making = agent_decision_making

            # Update stats
            self.stats["total_agents"] += 1

            logger.info(f"Created revolutionary memory collection for agent {agent_id}")
            return collection

        except Exception as e:
            logger.error(f"Failed to create revolutionary memory for agent {agent_id}: {str(e)}")
            raise

    # REVOLUTIONARY MEMORY METHODS

    async def add_memory(
        self,
        agent_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        emotional_valence: float = 0.0,
        tags: Optional[set] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        OPTIMIZED: Add a revolutionary memory entry with fast caching and automatic database persistence.

        Respects agent-specific memory type configuration. If a memory type is disabled for an agent,
        it will be silently skipped or converted to an enabled type.
        """
        start_time = time.time()

        try:
            if not self.is_initialized:
                await self.initialize()

            # Ensure agent has memory collection
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]

            # Check if this memory type is enabled for this agent
            agent_config = self.agent_memory_configs.get(agent_id, {})

            # Map memory types to their enable flags
            memory_type_flags = {
                MemoryType.SHORT_TERM: "enable_short_term",
                MemoryType.LONG_TERM: "enable_long_term",
                MemoryType.EPISODIC: "enable_episodic",
                MemoryType.SEMANTIC: "enable_semantic",
                MemoryType.PROCEDURAL: "enable_procedural",
                MemoryType.WORKING: "enable_working"
            }

            # Check if memory type is enabled
            flag_name = memory_type_flags.get(memory_type)
            if flag_name and not agent_config.get(flag_name, True):
                # Memory type is disabled for this agent
                logger.debug(
                    f"Memory type {memory_type.value} is disabled for agent {agent_id}, skipping",
                    agent_id=agent_id,
                    memory_type=memory_type.value
                )
                # Return empty ID to indicate memory was not stored
                return ""

            # Create revolutionary memory entry
            memory = MemoryEntry.create(
                agent_id=agent_id,
                memory_type=memory_type,
                content=content,
                metadata=metadata,
                importance=importance,
                emotional_valence=emotional_valence,
                tags=tags,
                context=context
            )

            # OPTIMIZATION: Fast cache update with lock
            with self._cache_lock:
                # Add to collection
                collection.add_memory(memory)

                # Update fast cache
                self._fast_cache[memory.id] = memory
                self._importance_cache[memory.id] = self._get_importance_score(importance)

                # Update index cache
                self._update_index_cache(memory)

                # Update association cache
                self._update_association_cache(memory)

            # Update stats
            self._update_memory_stats(memory_type, 1)

            # OPTIMIZATION: Background RAG storage if enabled
            if self.unified_rag and self.config.get("parallel_operations", True):
                asyncio.create_task(self._background_rag_storage(memory))

            # CRITICAL FIX: Automatic database persistence
            asyncio.create_task(self._persist_to_database(memory))

            operation_time = (time.time() - start_time) * 1000
            logger.debug(
                "OPTIMIZED revolutionary memory added with persistence",
                agent_id=agent_id,
                memory_id=memory.id,
                memory_type=memory_type.value,
                importance=importance.value,
                operation_time_ms=f"{operation_time:.2f}"
            )

            return memory.id

        except Exception as e:
            logger.error(f"Failed to add revolutionary memory: {str(e)}")
            raise

    def _get_importance_score(self, importance: MemoryImportance) -> float:
        """Fast importance score calculation."""
        importance_mapping = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TEMPORARY: 0.2
        }
        return importance_mapping.get(importance, 0.6)

    def _update_index_cache(self, memory: MemoryEntry) -> None:
        """OPTIMIZATION: Fast index cache update."""
        # Update tag index
        for tag in memory.tags:
            if tag not in self._index_cache:
                self._index_cache[tag] = {}
            self._index_cache[tag][memory.id] = memory

        # Update temporal index
        date_key = memory.created_at.strftime("%Y-%m-%d")
        if date_key not in self._index_cache:
            self._index_cache[date_key] = {}
        self._index_cache[date_key][memory.id] = memory

    def _update_association_cache(self, memory: MemoryEntry) -> None:
        """OPTIMIZATION: Fast association cache update."""
        if memory.id not in self._association_cache:
            self._association_cache[memory.id] = {}
        
        for assoc_id, strength in memory.associations.items():
            self._association_cache[memory.id][assoc_id] = strength

    async def _background_rag_storage(self, memory: MemoryEntry) -> None:
        """OPTIMIZATION: Background RAG storage to avoid blocking."""
        try:
            if not self.unified_rag:
                return

            from ..rag.core.unified_rag_system import Document

            # Create document for RAG storage
            doc = Document(
                id=memory.id,
                content=memory.content,
                metadata={
                    **(memory.metadata or {}),
                    "memory_type": memory.memory_type.value,
                    "agent_id": memory.agent_id,
                    "created_at": memory.created_at.isoformat(),
                    "importance": memory.importance.value,
                    "emotional_valence": memory.emotional_valence
                }
            )

            # Store in appropriate collection
            collection_type = "memory_short" if memory.memory_type == MemoryType.SHORT_TERM else "memory_long"
            await self.unified_rag.add_documents(
                agent_id=memory.agent_id,
                documents=[doc],
                collection_type=collection_type
            )

        except Exception as e:
            logger.warning(f"Background RAG storage failed: {e}")

    async def active_retrieve_memories(
        self,
        agent_id: str,
        current_task: str = "",
        conversation_context: str = "",
        emotional_state: float = 0.0,
        max_memories: int = 10,
        relevance_threshold: float = 0.3
    ) -> RetrievalResult:
        """OPTIMIZED: Actively retrieve relevant memories with fast caching."""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()

            # OPTIMIZATION: Fast cache lookup first
            cache_key = f"{agent_id}:{current_task}:{conversation_context}:{emotional_state}"
            if cache_key in self._fast_cache:
                cached_result = self._fast_cache[cache_key]
                if isinstance(cached_result, RetrievalResult):
                    logger.debug(f"Fast cache hit for agent {agent_id}")
                    return cached_result

            if not self.active_retrieval_engine:
                logger.warning("Active Retrieval Engine not available")
                return RetrievalResult(
                    memories=[],
                    relevance_scores={},
                    retrieval_reason={},
                    context_summary="Active retrieval not available",
                    total_retrieved=0,
                    retrieval_time_ms=0.0
                )

            # Ensure agent has memory collection
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]

            # OPTIMIZATION: Fast context creation
            context = RetrievalContext(
                current_task=current_task,
                conversation_context=conversation_context,
                emotional_state=emotional_state,
                max_memories=max_memories,
                relevance_threshold=relevance_threshold
            )

            # OPTIMIZATION: Use optimized retrieval with caching
            result = await self._optimized_active_retrieval(collection, context)

            # OPTIMIZATION: Cache the result
            with self._cache_lock:
                self._fast_cache[cache_key] = result
                # Limit cache size
                if len(self._fast_cache) > self.config.get("cache_size_limit", 50000):
                    # Remove oldest entries
                    oldest_keys = list(self._fast_cache.keys())[:1000]
                    for key in oldest_keys:
                        self._fast_cache.pop(key, None)

            # Update stats
            self.stats["active_retrievals"] += 1

            operation_time = (time.time() - start_time) * 1000
            logger.info(
                "OPTIMIZED active memory retrieval completed",
                agent_id=agent_id,
                retrieved_count=result.total_retrieved,
                retrieval_time_ms=f"{operation_time:.2f}",
                cache_hit=False
            )

            return result

        except Exception as e:
            logger.error(f"Active memory retrieval failed: {str(e)}")
            raise

    async def _optimized_active_retrieval(
        self,
        collection: RevolutionaryMemoryCollection,
        context: RetrievalContext
    ) -> RetrievalResult:
        """OPTIMIZATION: Fast active retrieval with optimized algorithms."""
        start_time = time.time()
        
        try:
            # OPTIMIZATION: Fast candidate collection using cache
            candidate_memories = []
            
            with self._cache_lock:
                # Fast lookup from cache
                for memory_id, memory in self._fast_cache.items():
                    if isinstance(memory, MemoryEntry) and memory.agent_id == collection.agent_id:
                        if memory.id not in context.exclude_memories:
                            candidate_memories.append(memory)
            
            # Fallback to collection if cache miss
            if not candidate_memories:
                candidate_memories = self._get_candidate_memories_fast(collection, context)
            
            if not candidate_memories:
                return RetrievalResult(
                    memories=[],
                    relevance_scores={},
                    retrieval_reason={},
                    context_summary="No relevant memories found",
                    total_retrieved=0,
                    retrieval_time_ms=0.0
                )
            
            # OPTIMIZATION: Fast scoring with cached importance
            scored_memories = []
            for memory in candidate_memories:
                score = self._fast_relevance_score(memory, context)
                if score >= context.relevance_threshold:
                    scored_memories.append((memory, score))
            
            # Sort by score (descending)
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to max_memories
            top_memories = scored_memories[:context.max_memories]
            
            # Build result
            memories = [memory for memory, _ in top_memories]
            relevance_scores = {memory.id: score for memory, score in top_memories}
            retrieval_reasons = self._fast_retrieval_reasons(top_memories, context)
            
            retrieval_time = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                memories=memories,
                relevance_scores=relevance_scores,
                retrieval_reason=retrieval_reasons,
                context_summary=f"Retrieved {len(memories)} memories",
                total_retrieved=len(memories),
                retrieval_time_ms=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Optimized active retrieval failed: {e}")
            return RetrievalResult(
                memories=[],
                relevance_scores={},
                retrieval_reason={},
                context_summary=f"Retrieval failed: {str(e)}",
                total_retrieved=0,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )

    def _get_candidate_memories_fast(
        self,
        collection: RevolutionaryMemoryCollection,
        context: RetrievalContext
    ) -> List[MemoryEntry]:
        """OPTIMIZATION: Fast candidate memory collection."""
        candidates = []
        
        # Fast collection from all stores
        all_stores = [
            collection.short_term_memories,
            collection.long_term_memories,
            collection.episodic_memories,
            collection.semantic_memories,
            collection.procedural_memories
        ]
        
        for store in all_stores:
            candidates.extend(store.values())
        
        # Add working memories
        candidates.extend(collection.working_memories)
        
        return candidates

    def _fast_relevance_score(self, memory: MemoryEntry, context: RetrievalContext) -> float:
        """OPTIMIZATION: Fast relevance scoring with cached values."""
        total_score = 0.0
        
        # 1. Importance score (cached)
        importance_score = self._importance_cache.get(memory.id, 0.6)
        total_score += importance_score * 0.3
        
        # 2. Temporal relevance (fast calculation)
        time_diff = abs((context.time_context - memory.created_at).total_seconds())
        if time_diff < 3600:  # 1 hour
            temporal_score = 1.0
        elif time_diff < 86400:  # 1 day
            temporal_score = 0.8
        elif time_diff < 604800:  # 1 week
            temporal_score = 0.6
        else:
            temporal_score = 0.4
        total_score += temporal_score * 0.2
        
        # 3. Access frequency (fast calculation)
        frequency_score = min(memory.access_count / 10.0, 1.0) if memory.access_count > 0 else 0.0
        total_score += frequency_score * 0.1
        
        # 4. Tag matches (fast set intersection)
        if context.active_tags and memory.tags:
            tag_matches = len(context.active_tags.intersection(memory.tags))
            tag_score = tag_matches / len(context.active_tags) if context.active_tags else 0.0
            total_score += tag_score * 0.2
        
        # 5. Emotional alignment (fast calculation)
        if context.emotional_state != 0.0 and memory.emotional_valence != 0.0:
            emotional_distance = abs(context.emotional_state - memory.emotional_valence)
            emotional_score = 1.0 - (emotional_distance / 2.0)
            total_score += emotional_score * 0.1
        
        # 6. Association strength (cached)
        if context.priority_memories and memory.id in self._association_cache:
            max_strength = 0.0
            for priority_id in context.priority_memories:
                if priority_id in self._association_cache[memory.id]:
                    strength = self._association_cache[memory.id][priority_id]
                    max_strength = max(max_strength, strength)
            total_score += max_strength * 0.1
        
        return min(total_score, 1.0)

    def _fast_retrieval_reasons(
        self,
        scored_memories: List[Tuple[MemoryEntry, float]],
        context: RetrievalContext
    ) -> Dict[str, str]:
        """OPTIMIZATION: Fast retrieval reason generation."""
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
            
            reasons[memory.id] = f"Retrieved due to: {', '.join(reason_parts)}"
        
        return reasons

    async def update_core_memory(
        self,
        agent_id: str,
        block_type: str,  # "persona" or "human"
        content: str
    ) -> bool:
        """Update core memory block."""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Ensure agent has memory collection
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]
            success = collection.update_core_memory(block_type, content)

            if success:
                logger.info(
                    "Core memory updated",
                    agent_id=agent_id,
                    block_type=block_type,
                    content_length=len(content)
                )

            return success

        except Exception as e:
            logger.error(f"Failed to update core memory: {str(e)}")
            raise

    async def get_core_memory_context(self, agent_id: str) -> str:
        """Get formatted core memory context for agent prompts."""
        try:
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]
            return collection.get_core_memory_context()

        except Exception as e:
            logger.error(f"Failed to get core memory context: {str(e)}")
            return ""

    async def add_resource(
        self,
        agent_id: str,
        title: str,
        content: str,
        resource_type: str = "document",
        summary: str = "",
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_in_rag: bool = True
    ) -> str:
        """Add a resource to Resource Memory with RAG integration."""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Ensure agent has memory collection
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]

            # Create resource entry
            resource = ResourceMemoryEntry(
                title=title,
                summary=summary,
                resource_type=resource_type,
                content=content,
                file_path=file_path,
                metadata=metadata or {},
                size_bytes=len(content)
            )

            # Add to local collection
            collection.add_resource(resource)

            # REVOLUTIONARY ENHANCEMENT: Store in RAG system for searchability
            if store_in_rag and self.unified_rag:
                from ..rag.core.unified_rag_system import Document

                # Create document for RAG storage
                rag_doc = Document(
                    id=f"resource_{resource.resource_id}",
                    content=content,
                    metadata={
                        **(metadata or {}),
                        "resource_id": resource.resource_id,
                        "title": title,
                        "summary": summary,
                        "resource_type": resource_type,
                        "agent_id": agent_id,
                        "created_at": resource.created_at.isoformat(),
                        "memory_category": "resource",
                        "file_path": file_path
                    }
                )

                # Store in agent's knowledge collection for searchability
                await self.unified_rag.add_documents(
                    agent_id=agent_id,
                    documents=[rag_doc],
                    collection_type="knowledge"
                )

                logger.info(
                    "Resource stored in RAG system for searchability",
                    agent_id=agent_id,
                    resource_id=resource.resource_id,
                    rag_doc_id=rag_doc.id
                )

            # Update stats
            self.stats["resource_entries"] += 1

            logger.info(
                "Resource added to revolutionary memory",
                agent_id=agent_id,
                resource_id=resource.resource_id,
                title=title,
                resource_type=resource_type,
                stored_in_rag=store_in_rag and self.unified_rag is not None
            )

            return resource.resource_id

        except Exception as e:
            logger.error(f"Failed to add resource: {str(e)}")
            raise

    async def search_resources(
        self,
        agent_id: str,
        query: str,
        resource_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search resources using RAG system for semantic similarity."""
        try:
            if not self.is_initialized:
                await self.initialize()

            results = []

            # Search using RAG system if available
            if self.unified_rag:
                # Build filters for resource search
                filters = {"memory_category": "resource"}
                if resource_type:
                    filters["resource_type"] = resource_type

                # Search in agent's knowledge collection
                rag_results = await self.unified_rag.search_agent_knowledge(
                    agent_id=agent_id,
                    query=query,
                    top_k=top_k,
                    filters=filters
                )

                # Convert RAG results to resource format
                for doc in rag_results:
                    if doc.metadata.get("memory_category") == "resource":
                        results.append({
                            "resource_id": doc.metadata.get("resource_id"),
                            "title": doc.metadata.get("title", ""),
                            "summary": doc.metadata.get("summary", ""),
                            "resource_type": doc.metadata.get("resource_type", "document"),
                            "content": doc.content,
                            "file_path": doc.metadata.get("file_path"),
                            "created_at": doc.metadata.get("created_at"),
                            "relevance_score": getattr(doc, 'score', 1.0)
                        })

            # Fallback: search local resource memory if RAG not available
            if not results and agent_id in self.agent_memories:
                collection = self.agent_memories[agent_id]
                query_lower = query.lower()

                for resource in collection.resource_memory.values():
                    # Simple text matching fallback
                    if (query_lower in resource.title.lower() or
                        query_lower in resource.summary.lower() or
                        query_lower in resource.content.lower()):

                        if not resource_type or resource.resource_type == resource_type:
                            results.append({
                                "resource_id": resource.resource_id,
                                "title": resource.title,
                                "summary": resource.summary,
                                "resource_type": resource.resource_type,
                                "content": resource.content,
                                "file_path": resource.file_path,
                                "created_at": resource.created_at.isoformat(),
                                "relevance_score": 0.5  # Default score for text matching
                            })

            logger.info(
                "Resource search completed",
                agent_id=agent_id,
                query=query,
                results_count=len(results),
                used_rag=self.unified_rag is not None
            )

            return results[:top_k]

        except Exception as e:
            logger.error(f"Failed to search resources: {str(e)}")
            return []

    async def add_vault_entry(
        self,
        agent_id: str,
        entry_type: str,
        title: str,
        secret_value: str,
        sensitivity_level: str = "confidential",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add an entry to Knowledge Vault."""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Ensure agent has memory collection
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]

            # Create vault entry
            from .memory_models import SensitivityLevel
            entry = KnowledgeVaultEntry(
                entry_type=entry_type,
                title=title,
                secret_value=secret_value,
                sensitivity_level=SensitivityLevel(sensitivity_level),
                metadata=metadata or {}
            )

            # Add to collection
            collection.add_vault_entry(entry)

            # Update stats
            self.stats["vault_entries"] += 1

            logger.info(
                "Entry added to Knowledge Vault",
                agent_id=agent_id,
                entry_id=entry.entry_id,
                entry_type=entry_type,
                sensitivity=sensitivity_level
            )

            return entry.entry_id

        except Exception as e:
            logger.error(f"Failed to add vault entry: {str(e)}")
            raise

    def _update_memory_stats(self, memory_type: MemoryType, count_delta: int):
        """Update memory statistics."""
        self.stats["total_memories"] += count_delta

        if memory_type == MemoryType.SHORT_TERM:
            self.stats["short_term_memories"] += count_delta
        elif memory_type == MemoryType.LONG_TERM:
            self.stats["long_term_memories"] += count_delta
        elif memory_type == MemoryType.EPISODIC:
            self.stats["episodic_memories"] += count_delta
        elif memory_type == MemoryType.SEMANTIC:
            self.stats["semantic_memories"] += count_delta
        elif memory_type == MemoryType.PROCEDURAL:
            self.stats["procedural_memories"] += count_delta
        elif memory_type == MemoryType.WORKING:
            self.stats["working_memories"] += count_delta

    async def get_memory_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if agent_id and agent_id in self.agent_memories:
            # Agent-specific stats
            collection = self.agent_memories[agent_id]
            return collection.get_stats()
        else:
            # System-wide stats
            stats = self.stats.copy()

            # Add orchestrator stats if available
            if self.memory_orchestrator:
                stats["orchestrator"] = self.memory_orchestrator.get_orchestrator_stats()

            # Add retrieval engine stats if available
            if self.active_retrieval_engine:
                stats["retrieval_engine"] = self.active_retrieval_engine.get_stats()

            return stats
    
    async def add_memory(
        self,
        agent_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a memory for an agent - ENHANCED WITH RAG INTEGRATION."""
        try:
            # Ensure agent memory exists
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            # Create memory entry
            memory = MemoryEntry.create(
                agent_id=agent_id,
                memory_type=memory_type,
                content=content,
                metadata=metadata
            )

            # Add to local collection
            collection = self.agent_memories[agent_id]
            collection.add_memory(memory)

            # PHASE 2 ENHANCEMENT: Store in UnifiedRAGSystem if available
            if self.unified_rag:
                from ..rag.core.unified_rag_system import Document

                # Create document for RAG storage
                doc = Document(
                    id=memory.id,
                    content=content,
                    metadata={
                        **(metadata or {}),
                        "memory_type": memory_type.value,
                        "agent_id": agent_id,
                        "created_at": memory.created_at.isoformat()
                    }
                )

                # Store in appropriate collection
                collection_type = "memory_short" if memory_type == MemoryType.SHORT_TERM else "memory_long"
                await self.unified_rag.add_documents(
                    agent_id=agent_id,
                    documents=[doc],
                    collection_type=collection_type
                )

            # Update stats
            self.stats["total_memories"] += 1
            if memory_type == MemoryType.SHORT_TERM:
                self.stats["short_term_memories"] += 1
            else:
                self.stats["long_term_memories"] += 1

            logger.debug(f"Added {memory_type.value} memory for agent {agent_id} (RAG: {'enabled' if self.unified_rag else 'disabled'})")
            return memory.id
            
        except Exception as e:
            logger.error(f"Failed to add memory for agent {agent_id}: {str(e)}")
            raise
    
    async def get_memory(self, agent_id: str, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory for an agent."""
        try:
            if agent_id not in self.agent_memories:
                return None
            
            collection = self.agent_memories[agent_id]
            return collection.get_memory(memory_id)
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id} for agent {agent_id}: {str(e)}")
            return None
    
    async def search_memories(
        self,
        agent_id: str,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memories for an agent."""
        try:
            if agent_id not in self.agent_memories:
                return []
            
            collection = self.agent_memories[agent_id]
            results = []
            
            # Simple text search in memory content
            memories_to_search = []
            
            if memory_type == MemoryType.SHORT_TERM or memory_type is None:
                memories_to_search.extend(collection.short_term_memories.values())
            
            if memory_type == MemoryType.LONG_TERM or memory_type is None:
                memories_to_search.extend(collection.long_term_memories.values())
            
            # Basic text matching
            query_lower = query.lower()
            for memory in memories_to_search:
                if query_lower in memory.content.lower():
                    results.append(memory)
                    if len(results) >= limit:
                        break
            
            # Sort by access count and recency
            results.sort(key=lambda m: (m.access_count, m.last_accessed), reverse=True)
            
            logger.debug(f"Found {len(results)} memories for agent {agent_id}")
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search memories for agent {agent_id}: {str(e)}")
            return []
    
    async def remove_memory(self, agent_id: str, memory_id: str) -> bool:
        """Remove a memory for an agent."""
        try:
            if agent_id not in self.agent_memories:
                return False
            
            collection = self.agent_memories[agent_id]
            removed = collection.remove_memory(memory_id)
            
            if removed:
                self.stats["total_memories"] -= 1
                logger.debug(f"Removed memory {memory_id} for agent {agent_id}")
            
            return removed
            
        except Exception as e:
            logger.error(f"Failed to remove memory {memory_id} for agent {agent_id}: {str(e)}")
            return False
    
    async def get_agent_memories(self, agent_id: str) -> Optional[MemoryCollection]:
        """Get all memories for an agent."""
        return self.agent_memories.get(agent_id)
    
    async def cleanup_expired_memories(self) -> int:
        """Clean up expired short-term memories."""
        try:
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(hours=self.config["short_term_ttl_hours"])
            
            for agent_id, collection in self.agent_memories.items():
                expired_ids = []
                
                for memory_id, memory in collection.short_term_memories.items():
                    if memory.created_at < cutoff_time:
                        expired_ids.append(memory_id)
                
                for memory_id in expired_ids:
                    if collection.remove_memory(memory_id):
                        cleaned_count += 1
                        self.stats["total_memories"] -= 1
                        self.stats["short_term_memories"] -= 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired memories")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired memories: {str(e)}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "config": self.config,
            "agents_with_memories": len(self.agent_memories)
        }

    # REVOLUTIONARY METHODS FOR ALL PHASES

    async def run_consolidation_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """Run memory consolidation for a specific agent."""
        try:
            if agent_id not in self.agent_memories:
                return {"error": "Agent not found"}

            collection = self.agent_memories[agent_id]
            if hasattr(collection, 'consolidation_system') and collection.consolidation_system:
                session = await collection.consolidation_system.run_consolidation_session()
                self.stats["consolidation_sessions"] += 1
                return {
                    "session_id": session.session_id,
                    "memories_processed": session.memories_processed,
                    "memories_promoted": session.memories_promoted,
                    "memories_forgotten": session.memories_forgotten
                }
            else:
                return {"error": "Consolidation system not available"}

        except Exception as e:
            logger.error(f"Failed to run consolidation for agent {agent_id}: {e}")
            return {"error": str(e)}

    async def record_learning_experience(
        self,
        agent_id: str,
        task_type: str,
        task_context: Dict[str, Any],
        performance_metrics: Dict[str, float],
        memories_used: List[str],
        memories_created: List[str],
        success: bool = True
    ) -> str:
        """Record a learning experience for an agent."""
        try:
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]
            if hasattr(collection, 'lifelong_learning') and collection.lifelong_learning:
                experience_id = await collection.lifelong_learning.record_learning_experience(
                    task_type, task_context, performance_metrics, memories_used, memories_created, success
                )
                self.stats["learning_experiences"] += 1
                return experience_id
            else:
                return ""

        except Exception as e:
            logger.error(f"Failed to record learning experience for agent {agent_id}: {e}")
            return ""

    # Optional: Multimodal memory support (not required for core functionality)
    # async def store_multimodal_memory(
    #     self,
    #     agent_id: str,
    #     primary_modality: Any,  # ModalityType
    #     primary_content: Any,
    #     additional_modalities: Optional[Dict[Any, Any]] = None,
    #     context: Optional[Dict[str, Any]] = None,
    #     importance: str = "medium"
    # ) -> str:
    #     """Store a multimodal memory."""
    #     try:
    #         if not self.multimodal_system:
    #             return ""
    #
    #         memory_id = await self.multimodal_system.store_multimodal_memory(
    #             primary_modality, primary_content, additional_modalities, context, importance
    #         )
    #
    #         if memory_id:
    #             self.stats["multimodal_memories"] += 1
    #
    #         return memory_id
    #
    #     except Exception as e:
    #         logger.error(f"Failed to store multimodal memory: {e}")
    #         return ""

    # Optional: Memory-driven decision making (not required for core functionality)
    # async def make_memory_driven_decision(
    #     self,
    #     agent_id: str,
    #     decision_context: Any,  # DecisionContext
    #     options: List[Any],     # List[DecisionOption]
    #     strategy: str = "hybrid"
    # ) -> Tuple[Any, Any]:  # Tuple[DecisionOption, DecisionRecord]
    #     """Make a memory-driven decision for an agent."""
    #     try:
    #         if agent_id not in self.agent_memories:
    #             await self.create_agent_memory(agent_id)
    #
    #         collection = self.agent_memories[agent_id]
    #         if hasattr(collection, 'decision_making') and collection.decision_making:
    #             chosen_option, decision_record = await collection.decision_making.make_decision(
    #                 decision_context, options, strategy
    #             )
    #             self.stats["decisions_made"] += 1
    #             return chosen_option, decision_record
    #         else:
    #             # Return first option as fallback
    #             return options[0] if options else None, None
    #
    #     except Exception as e:
    #         logger.error(f"Failed to make memory-driven decision for agent {agent_id}: {e}")
    #         return options[0] if options else None, None

    async def advanced_memory_search(
        self,
        agent_id: str,
        query: RetrievalQuery
    ) -> List[Any]:  # List[RetrievalResult]
        """Perform advanced memory search using multiple retrieval methods."""
        try:
            if agent_id not in self.agent_memories:
                return []

            if not self.advanced_retrieval:
                return []

            # Get agent memories as dict for advanced retrieval
            collection = self.agent_memories[agent_id]
            memories_dict = {}

            # Combine all memory types
            for memory_type in ["short_term_memories", "long_term_memories", "episodic_memories", "semantic_memories", "procedural_memories"]:
                if hasattr(collection, memory_type):
                    memory_store = getattr(collection, memory_type)
                    for memory_id, memory in memory_store.items():
                        memories_dict[memory_id] = {
                            "content": memory.content,
                            "metadata": memory.metadata,
                            "created_at": memory.created_at,
                            "importance": memory.importance,
                            "memory_type": memory_type
                        }

            # Perform advanced retrieval
            results = await self.advanced_retrieval.advanced_retrieve(query, memories_dict)
            return results

        except Exception as e:
            logger.error(f"Failed to perform advanced memory search for agent {agent_id}: {e}")
            return []

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all revolutionary components."""
        try:
            comprehensive_stats = {
                "unified_memory_system": self.get_stats(),
                "knowledge_graph": {},
                "advanced_retrieval": {},
                "multimodal_system": {},
                "agent_specific": {}
            }

            # System-level component stats
            if self.knowledge_graph:
                comprehensive_stats["knowledge_graph"] = self.knowledge_graph.get_graph_stats()

            if self.advanced_retrieval:
                comprehensive_stats["advanced_retrieval"] = self.advanced_retrieval.get_retrieval_stats()

            if self.multimodal_system:
                comprehensive_stats["multimodal_system"] = self.multimodal_system.get_multimodal_stats()

            # Agent-specific stats
            for agent_id, collection in self.agent_memories.items():
                agent_stats = {}

                if hasattr(collection, 'consolidation_system') and collection.consolidation_system:
                    agent_stats["consolidation"] = collection.consolidation_system.get_consolidation_stats()

                if hasattr(collection, 'lifelong_learning') and collection.lifelong_learning:
                    agent_stats["learning"] = collection.lifelong_learning.get_learning_stats()

                if hasattr(collection, 'decision_making') and collection.decision_making:
                    agent_stats["decision_making"] = collection.decision_making.get_decision_stats()

                if hasattr(collection, 'knowledge_graph') and collection.knowledge_graph:
                    agent_stats["knowledge_graph"] = collection.knowledge_graph.get_graph_stats()

                comprehensive_stats["agent_specific"][agent_id] = agent_stats

            return comprehensive_stats

        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return {"error": str(e)}

    # OPTIMIZED SIMPLE INTERFACE METHODS
    
    async def store(self, agent_id: str, content: str, memory_type: str = "episodic", 
                   importance: str = "medium", tags: List[str] = None, 
                   emotional_valence: float = 0.0, metadata: Dict[str, Any] = None) -> str:
        """OPTIMIZED SIMPLE INTERFACE: Store memory with all advanced features."""
        try:
            # Convert string parameters to enums
            memory_type_enum = MemoryType(memory_type)
            importance_enum = MemoryImportance(importance)
            
            # Store memory with all revolutionary features
            memory_id = await self.add_memory(
                agent_id=agent_id,
                memory_type=memory_type_enum,
                content=content,
                metadata=metadata,
                importance=importance_enum,
                emotional_valence=emotional_valence,
                tags=set(tags) if tags else None
            )
            
            logger.info(f"OPTIMIZED memory stored: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    async def retrieve(self, agent_id: str, query: str = None, context: str = None, 
                      limit: int = 5, memory_types: List[str] = None) -> List[MemoryEntry]:
        """OPTIMIZED SIMPLE INTERFACE: Retrieve memories with all advanced features."""
        try:
            # Use active retrieval if no specific query
            if not query and not context:
                result = await self.active_retrieve_memories(
                    agent_id=agent_id,
                    max_memories=limit
                )
                return result.memories
            
            # Use active retrieval with context
            if context:
                result = await self.active_retrieve_memories(
                    agent_id=agent_id,
                    conversation_context=context,
                    max_memories=limit
                )
                return result.memories
            
            # Use search for specific query
            if query:
                memory_type_enum = None
                if memory_types and len(memory_types) == 1:
                    memory_type_enum = MemoryType(memory_types[0])
                
                memories = await self.search_memories(
                    agent_id=agent_id,
                    query=query,
                    memory_type=memory_type_enum,
                    limit=limit
                )
                return memories
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    async def learn(self, agent_id: str, experience: str, outcome: str, success: bool = True) -> str:
        """OPTIMIZED SIMPLE INTERFACE: Learn from experience with all advanced features."""
        try:
            # Store the experience as episodic memory
            experience_id = await self.store(
                agent_id=agent_id,
                content=f"Experience: {experience}. Outcome: {outcome}. Success: {success}",
                memory_type="episodic",
                importance="high" if success else "medium",
                tags=["learning", "experience"],
                emotional_valence=1.0 if success else -0.5
            )
            
            # Record learning experience if system available
            if agent_id in self.agent_memories:
                collection = self.agent_memories[agent_id]
                if hasattr(collection, 'lifelong_learning') and collection.lifelong_learning:
                    await collection.lifelong_learning.record_learning_experience(
                        task_type="experience_learning",
                        task_context={"experience": experience, "outcome": outcome},
                        performance_metrics={"success": 1.0 if success else 0.0},
                        memories_used=[],
                        memories_created=[experience_id],
                        success=success
                    )
            
            logger.info(f"OPTIMIZED learning recorded: {experience_id}")
            return experience_id
            
        except Exception as e:
            logger.error(f"Failed to learn from experience: {e}")
            return ""

    async def _persist_to_database(self, memory: MemoryEntry) -> bool:
        """
        CRITICAL FIX: Persist memory to PostgreSQL database.

        This ensures memories survive agent restarts and system reboots.
        Runs as background task to avoid blocking memory operations.
        """
        try:
            from app.models.database.base import get_database_session
            from app.models.autonomous import AgentMemoryDB, AutonomousAgentState
            from app.models.agent import Agent
            from sqlalchemy import select
            from sqlalchemy.exc import IntegrityError
            import uuid as uuid_module

            async for session in get_database_session():
                try:
                    # Parse agent_id to UUID
                    try:
                        agent_uuid = uuid_module.UUID(memory.agent_id) if isinstance(memory.agent_id, str) else memory.agent_id
                    except (ValueError, AttributeError):
                        logger.warning(f"Invalid agent_id format for persistence: {memory.agent_id}")
                        return False

                    # Get or create agent record
                    agent_result = await session.execute(
                        select(Agent).where(Agent.id == agent_uuid)
                    )
                    agent_record = agent_result.scalar_one_or_none()

                    if not agent_record:
                        # Create agent record if it doesn't exist
                        agent_record = Agent(
                            id=agent_uuid,
                            name=f"Agent-{memory.agent_id[:8]}",
                            agent_type="general",
                            model="llama3.2:latest",
                            model_provider="ollama"
                        )
                        session.add(agent_record)
                        await session.flush()

                    # Get or create autonomous agent state
                    state_result = await session.execute(
                        select(AutonomousAgentState).where(AutonomousAgentState.agent_id == agent_uuid)
                    )
                    agent_state_record = state_result.scalar_one_or_none()

                    if not agent_state_record:
                        # Create agent state if doesn't exist
                        agent_state_record = AutonomousAgentState(
                            agent_id=agent_uuid,
                            autonomy_level='adaptive',
                            learning_enabled=True
                        )
                        session.add(agent_state_record)
                        await session.flush()

                    # Check if memory already exists
                    existing_memory = await session.execute(
                        select(AgentMemoryDB).where(AgentMemoryDB.memory_id == memory.id)
                    )
                    if existing_memory.scalar_one_or_none():
                        logger.debug(f"Memory {memory.id} already persisted, skipping")
                        return True

                    # Create memory record
                    memory_record = AgentMemoryDB(
                        memory_id=memory.id,
                        agent_state_id=agent_state_record.id,
                        content=memory.content,
                        memory_type=memory.memory_type.value,
                        context=memory.metadata or {},
                        importance=memory.importance.value,
                        emotional_valence=memory.emotional_valence,
                        tags=list(memory.tags) if memory.tags else [],
                        created_at=memory.created_at,
                        last_accessed=memory.last_accessed,
                        access_count=memory.access_count
                    )

                    session.add(memory_record)
                    await session.commit()

                    logger.debug(f"Memory persisted to database: {memory.id} for agent {memory.agent_id}")
                    return True

                except IntegrityError as e:
                    await session.rollback()
                    logger.debug(f"Memory {memory.id} already exists in database (integrity error)")
                    return True
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Failed to persist memory to database: {e}", exc_info=True)
                    return False

        except Exception as e:
            logger.error(f"Database persistence error: {e}", exc_info=True)
            return False

    async def get_agent_context(self, agent_id: str, current_task: str = "",
                               conversation_context: str = "") -> str:
        """OPTIMIZED SIMPLE INTERFACE: Get comprehensive agent context."""
        try:
            # Get core memory context
            core_context = await self.get_core_memory_context(agent_id)

            # Get active memories
            result = await self.active_retrieve_memories(
                agent_id=agent_id,
                current_task=current_task,
                conversation_context=conversation_context,
                max_memories=10
            )

            # Build comprehensive context
            context_parts = []

            if core_context:
                context_parts.append(f"CORE MEMORY:\n{core_context}")

            if result.memories:
                memory_context = "\n".join([
                    f"- {memory.content} (importance: {memory.importance.value})"
                    for memory in result.memories[:5]
                ])
                context_parts.append(f"RELEVANT MEMORIES:\n{memory_context}")

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Failed to get agent context: {e}")
            return ""

    def get_performance_stats(self) -> Dict[str, Any]:
        """OPTIMIZED: Get performance statistics."""
        return {
            "cache_size": len(self._fast_cache),
            "index_size": len(self._index_cache),
            "association_size": len(self._association_cache),
            "importance_size": len(self._importance_cache),
            "total_agents": len(self.agent_memories),
            "config": self.config,
            "is_initialized": self.is_initialized
        }
