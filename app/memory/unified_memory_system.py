"""
Revolutionary Unified Memory System - THE Memory System for Multi-Agent Architecture.

This is THE ONLY memory system in the entire application.
All memory operations flow through this revolutionary unified system.

REVOLUTIONARY ARCHITECTURE (PHASE 3):
✅ Agent-specific memory isolation using ChromaDB collections
✅ Short-term memory (24h TTL) and Long-term memory (persistent)
✅ Integration with UnifiedRAGSystem for vector storage
✅ Revolutionary memory types: Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault
✅ Active Retrieval Engine for automatic context-based memory retrieval
✅ Memory Orchestrator with specialized managers
✅ Multi-agent memory coordination and parallel processing

DESIGN PRINCIPLES:
- One revolutionary memory system to rule them all
- Agent isolation through collections
- Revolutionary memory types based on MIRIX and state-of-the-art research
- Active retrieval without explicit search commands
- Memory orchestration with specialized managers
- No complexity unless it provides revolutionary capabilities

PHASE 3 REVOLUTIONARY ENHANCEMENTS:
✅ Core Memory (always-visible persistent context)
✅ Knowledge Vault (secure sensitive information storage)
✅ Resource Memory (document and file management)
✅ Active Retrieval Engine (automatic context-based retrieval)
✅ Memory Orchestrator (multi-agent coordination)
✅ Revolutionary memory models with associations and importance
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

import structlog

from .memory_models import (
    MemoryType, MemoryEntry, MemoryCollection, RevolutionaryMemoryCollection,
    MemoryImportance, CoreMemoryBlock, ResourceMemoryEntry, KnowledgeVaultEntry
)
from .active_retrieval_engine import ActiveRetrievalEngine, RetrievalContext, RetrievalResult
from .memory_orchestrator import MemoryOrchestrator, MemoryOperation, MemoryManagerType
from .dynamic_knowledge_graph import DynamicKnowledgeGraph
from .advanced_retrieval_mechanisms import AdvancedRetrievalMechanisms, RetrievalQuery
from .memory_consolidation_system import MemoryConsolidationSystem, ConsolidationPhase
from .lifelong_learning_capabilities import LifelongLearningCapabilities
from .multimodal_memory_support import MultimodalMemorySystem, ModalityType
from .memory_driven_decision_making import MemoryDrivenDecisionMaking

logger = structlog.get_logger(__name__)


class UnifiedMemorySystem:
    """
    Revolutionary Unified Memory System - THE Memory System.

    REVOLUTIONARY ARCHITECTURE:
    - Uses UnifiedRAGSystem for storage
    - Agent-specific revolutionary memory collections
    - All memory types: Short-term, Long-term, Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault
    - Active Retrieval Engine for automatic context-based retrieval
    - Memory Orchestrator for multi-agent coordination
    - Automatic cleanup and TTL
    """

    def __init__(self, unified_rag=None, embedding_function: Optional[Callable] = None):
        """Initialize THE revolutionary unified memory system."""
        self.unified_rag = unified_rag
        self.embedding_function = embedding_function
        self.is_initialized = False

        # Revolutionary agent memory collections
        self.agent_memories: Dict[str, RevolutionaryMemoryCollection] = {}

        # Revolutionary configuration
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
            "enable_memory_orchestration": True
        }

        # Revolutionary components
        self.memory_orchestrator: Optional[MemoryOrchestrator] = None
        self.active_retrieval_engine: Optional[ActiveRetrievalEngine] = None
        self.knowledge_graph: Optional[DynamicKnowledgeGraph] = None
        self.advanced_retrieval: Optional[AdvancedRetrievalMechanisms] = None
        self.consolidation_system: Optional[MemoryConsolidationSystem] = None
        self.lifelong_learning: Optional[LifelongLearningCapabilities] = None
        self.multimodal_system: Optional[MultimodalMemorySystem] = None
        self.decision_making: Optional[MemoryDrivenDecisionMaking] = None

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

        logger.info("Revolutionary Unified Memory System initialized")
    
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
                logger.info("Memory Orchestrator initialized")

            if self.config["enable_active_retrieval"]:
                self.active_retrieval_engine = ActiveRetrievalEngine(self.embedding_function)
                logger.info("Active Retrieval Engine initialized")

            # Initialize all revolutionary components
            self.knowledge_graph = DynamicKnowledgeGraph("system", self.embedding_function)
            logger.info("Dynamic Knowledge Graph initialized")

            self.advanced_retrieval = AdvancedRetrievalMechanisms("system", self.embedding_function, self.knowledge_graph)
            logger.info("Advanced Retrieval Mechanisms initialized")

            self.multimodal_system = MultimodalMemorySystem("system")
            logger.info("Multimodal Memory System initialized")

            # Note: Agent-specific components will be initialized per agent
            logger.info("All revolutionary components initialized")

            self.is_initialized = True
            logger.info("Revolutionary Memory System initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize revolutionary memory system: {str(e)}")
            raise
    
    async def create_agent_memory(self, agent_id: str) -> RevolutionaryMemoryCollection:
        """Create revolutionary memory collection for a new agent."""
        try:
            if not self.is_initialized:
                await self.initialize()

            if agent_id in self.agent_memories:
                logger.warning(f"Revolutionary memory collection already exists for agent {agent_id}")
                return self.agent_memories[agent_id]

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

                # Create agent-specific consolidation system
                agent_consolidation = MemoryConsolidationSystem(agent_id, collection)

                # Create agent-specific lifelong learning
                agent_learning = LifelongLearningCapabilities(agent_id, collection)

                # Create agent-specific decision making
                agent_decision_making = MemoryDrivenDecisionMaking(agent_id, collection, agent_knowledge_graph)

                # Store references in collection for easy access
                collection.knowledge_graph = agent_knowledge_graph
                collection.consolidation_system = agent_consolidation
                collection.lifelong_learning = agent_learning
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
        """Add a revolutionary memory entry."""
        try:
            if not self.is_initialized:
                await self.initialize()

            # Ensure agent has memory collection
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]

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

            # Add to collection
            collection.add_memory(memory)

            # Update stats
            self._update_memory_stats(memory_type, 1)

            logger.debug(
                "Revolutionary memory added",
                agent_id=agent_id,
                memory_id=memory.id,
                memory_type=memory_type.value,
                importance=importance.value
            )

            return memory.id

        except Exception as e:
            logger.error(f"Failed to add revolutionary memory: {str(e)}")
            raise

    async def active_retrieve_memories(
        self,
        agent_id: str,
        current_task: str = "",
        conversation_context: str = "",
        emotional_state: float = 0.0,
        max_memories: int = 10,
        relevance_threshold: float = 0.3
    ) -> RetrievalResult:
        """Actively retrieve relevant memories without explicit search."""
        try:
            if not self.is_initialized:
                await self.initialize()

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

            # Create retrieval context
            context = RetrievalContext(
                current_task=current_task,
                conversation_context=conversation_context,
                emotional_state=emotional_state,
                max_memories=max_memories,
                relevance_threshold=relevance_threshold
            )

            # Perform active retrieval
            result = await self.active_retrieval_engine.retrieve_active_memories(collection, context)

            # Update stats
            self.stats["active_retrievals"] += 1

            logger.info(
                "Active memory retrieval completed",
                agent_id=agent_id,
                retrieved_count=result.total_retrieved,
                retrieval_time_ms=f"{result.retrieval_time_ms:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Active memory retrieval failed: {str(e)}")
            raise

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

    async def store_multimodal_memory(
        self,
        agent_id: str,
        primary_modality: ModalityType,
        primary_content: Any,
        additional_modalities: Optional[Dict[ModalityType, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        importance: str = "medium"
    ) -> str:
        """Store a multimodal memory."""
        try:
            if not self.multimodal_system:
                return ""

            memory_id = await self.multimodal_system.store_multimodal_memory(
                primary_modality, primary_content, additional_modalities, context, importance
            )

            if memory_id:
                self.stats["multimodal_memories"] += 1

            return memory_id

        except Exception as e:
            logger.error(f"Failed to store multimodal memory: {e}")
            return ""

    async def make_memory_driven_decision(
        self,
        agent_id: str,
        decision_context: Any,  # DecisionContext
        options: List[Any],     # List[DecisionOption]
        strategy: str = "hybrid"
    ) -> Tuple[Any, Any]:  # Tuple[DecisionOption, DecisionRecord]
        """Make a memory-driven decision for an agent."""
        try:
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)

            collection = self.agent_memories[agent_id]
            if hasattr(collection, 'decision_making') and collection.decision_making:
                chosen_option, decision_record = await collection.decision_making.make_decision(
                    decision_context, options, strategy
                )
                self.stats["decisions_made"] += 1
                return chosen_option, decision_record
            else:
                # Return first option as fallback
                return options[0] if options else None, None

        except Exception as e:
            logger.error(f"Failed to make memory-driven decision for agent {agent_id}: {e}")
            return options[0] if options else None, None

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
