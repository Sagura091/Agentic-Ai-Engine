"""
Agent-Specific Memory Collections Manager.

This module manages agent-specific memory collections within the unified
ChromaDB instance, providing efficient memory storage and retrieval with
proper isolation and access controls.

Features:
- Agent-specific memory collection management
- Memory type segregation (episodic, semantic, procedural)
- Efficient memory consolidation and retrieval
- Cross-memory type associations
- Memory lifecycle management
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .unified_rag_system import UnifiedRAGSystem, AgentCollections
from .unified_memory_system import UnifiedMemorySystem, MemoryEntry, MemoryType, MemoryImportance
from .agent_isolation_manager import AgentIsolationManager
from .unified_rag_system import Document

logger = structlog.get_logger(__name__)


class MemoryCollectionType(str, Enum):
    """Types of memory collections."""
    EPISODIC = "episodic"           # Event-based memories
    SEMANTIC = "semantic"           # Factual knowledge
    PROCEDURAL = "procedural"       # Skills and procedures
    WORKING = "working"             # Temporary context
    CONSOLIDATED = "consolidated"   # Merged and processed memories


@dataclass
class MemoryCollectionInfo:
    """Information about a memory collection."""
    collection_name: str
    agent_id: str
    memory_type: MemoryCollectionType
    document_count: int
    last_updated: datetime
    size_bytes: int
    consolidation_level: int


class AgentMemoryCollections:
    """
    Agent-Specific Memory Collections Manager.
    
    Manages memory collections for individual agents within the unified
    ChromaDB instance with proper isolation and efficient operations.
    """
    
    def __init__(
        self,
        unified_rag: UnifiedRAGSystem,
        memory_system: UnifiedMemorySystem,
        isolation_manager: AgentIsolationManager
    ):
        """Initialize the agent memory collections manager."""
        self.unified_rag = unified_rag
        self.memory_system = memory_system
        self.isolation_manager = isolation_manager
        
        # Collection tracking
        self.agent_collections: Dict[str, Dict[MemoryCollectionType, MemoryCollectionInfo]] = {}
        
        # Memory associations
        self.memory_associations: Dict[str, Dict[str, float]] = {}  # memory_id -> {related_memory_id -> strength}
        
        # Performance tracking
        self.stats = {
            "total_memory_collections": 0,
            "total_memory_documents": 0,
            "consolidation_operations": 0,
            "association_updates": 0
        }
        
        logger.info("Agent memory collections manager initialized")
    
    async def initialize_agent_memory_collections(self, agent_id: str) -> Dict[MemoryCollectionType, MemoryCollectionInfo]:
        """
        Initialize memory collections for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary of memory collection information
        """
        try:
            if agent_id in self.agent_collections:
                logger.warning(f"Memory collections already exist for agent {agent_id}")
                return self.agent_collections[agent_id]
            
            # Ensure agent ecosystem exists
            agent_ecosystem = await self.unified_rag.get_agent_collections(agent_id)
            if not agent_ecosystem:
                agent_ecosystem = await self.unified_rag.create_agent_ecosystem(agent_id)
            
            # Create collection info for each memory type
            collections = {}
            
            # Episodic memory collection
            episodic_info = MemoryCollectionInfo(
                collection_name=agent_ecosystem.episodic_memory_collection,
                agent_id=agent_id,
                memory_type=MemoryCollectionType.EPISODIC,
                document_count=0,
                last_updated=datetime.utcnow(),
                size_bytes=0,
                consolidation_level=0
            )
            collections[MemoryCollectionType.EPISODIC] = episodic_info
            
            # Semantic memory collection
            semantic_info = MemoryCollectionInfo(
                collection_name=agent_ecosystem.semantic_memory_collection,
                agent_id=agent_id,
                memory_type=MemoryCollectionType.SEMANTIC,
                document_count=0,
                last_updated=datetime.utcnow(),
                size_bytes=0,
                consolidation_level=0
            )
            collections[MemoryCollectionType.SEMANTIC] = semantic_info
            
            # Procedural memory collection (use semantic for now)
            procedural_info = MemoryCollectionInfo(
                collection_name=f"agent_{agent_id}_memory_procedural",
                agent_id=agent_id,
                memory_type=MemoryCollectionType.PROCEDURAL,
                document_count=0,
                last_updated=datetime.utcnow(),
                size_bytes=0,
                consolidation_level=0
            )
            collections[MemoryCollectionType.PROCEDURAL] = procedural_info
            
            # Create procedural collection
            await self.unified_rag._create_collection(
                procedural_info.collection_name,
                self.unified_rag.CollectionType.AGENT_MEMORY_SEMANTIC
            )
            
            self.agent_collections[agent_id] = collections
            self.stats["total_memory_collections"] += len(collections)
            
            logger.info(f"Initialized memory collections for agent {agent_id}")
            return collections
            
        except Exception as e:
            logger.error(f"Failed to initialize memory collections for agent {agent_id}: {str(e)}")
            raise
    
    async def store_memory_in_collection(
        self,
        agent_id: str,
        memory: MemoryEntry,
        target_collection: Optional[MemoryCollectionType] = None
    ) -> str:
        """
        Store a memory in the appropriate collection.
        
        Args:
            agent_id: Agent storing the memory
            memory: Memory to store
            target_collection: Specific collection to store in (optional)
            
        Returns:
            Document ID in the collection
        """
        try:
            # Ensure collections exist
            if agent_id not in self.agent_collections:
                await self.initialize_agent_memory_collections(agent_id)
            
            # Determine target collection
            if target_collection is None:
                if memory.memory_type == MemoryType.EPISODIC:
                    target_collection = MemoryCollectionType.EPISODIC
                elif memory.memory_type == MemoryType.SEMANTIC:
                    target_collection = MemoryCollectionType.SEMANTIC
                elif memory.memory_type == MemoryType.PROCEDURAL:
                    target_collection = MemoryCollectionType.PROCEDURAL
                else:
                    target_collection = MemoryCollectionType.EPISODIC  # Default
            
            collection_info = self.agent_collections[agent_id][target_collection]
            
            # Create document from memory
            memory_doc = Document(
                title=f"Memory: {memory.memory_type.value}",
                content=memory.content,
                metadata={
                    "memory_id": memory.memory_id,
                    "agent_id": agent_id,
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance.value,
                    "created_at": memory.created_at.isoformat(),
                    "last_accessed": memory.last_accessed.isoformat(),
                    "access_count": memory.access_count,
                    "tags": list(memory.tags),
                    "emotional_valence": memory.emotional_valence,
                    "confidence": memory.confidence,
                    "session_id": memory.session_id,
                    "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
                    **memory.context
                }
            )
            
            # Store in unified RAG system
            collection_type_map = {
                MemoryCollectionType.EPISODIC: "episodic_memory",
                MemoryCollectionType.SEMANTIC: "semantic_memory",
                MemoryCollectionType.PROCEDURAL: "semantic_memory"  # Use semantic for procedural
            }
            
            document_id = await self.unified_rag.add_document_to_agent(
                agent_id=agent_id,
                document=memory_doc,
                collection_type=collection_type_map[target_collection]
            )
            
            # Update collection info
            collection_info.document_count += 1
            collection_info.last_updated = datetime.utcnow()
            collection_info.size_bytes += len(memory.content.encode('utf-8'))
            
            # Update associations
            await self._update_memory_associations(agent_id, memory)
            
            self.stats["total_memory_documents"] += 1
            
            logger.info(f"Stored memory in {target_collection.value} collection for agent {agent_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to store memory in collection: {str(e)}")
            raise
    
    async def retrieve_memories_from_collections(
        self,
        agent_id: str,
        query: str,
        collection_types: Optional[List[MemoryCollectionType]] = None,
        max_results: int = 10,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        importance_threshold: Optional[MemoryImportance] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve memories from agent's collections.
        
        Args:
            agent_id: Agent retrieving memories
            query: Search query
            collection_types: Types of collections to search
            max_results: Maximum results to return
            time_range: Time range filter
            importance_threshold: Minimum importance level
            
        Returns:
            List of matching memories
        """
        try:
            if agent_id not in self.agent_collections:
                return []
            
            # Default to all collection types
            if collection_types is None:
                collection_types = [
                    MemoryCollectionType.EPISODIC,
                    MemoryCollectionType.SEMANTIC,
                    MemoryCollectionType.PROCEDURAL
                ]
            
            # Search using unified RAG system
            rag_result = await self.unified_rag.search_agent_knowledge(
                agent_id=agent_id,
                query=query,
                top_k=max_results * 2,  # Get more results for filtering
                include_shared=False
            )
            
            # Convert results back to memory entries
            memories = []
            for result in rag_result.results:
                try:
                    # Extract memory metadata
                    metadata = result.metadata
                    
                    # Filter by collection type
                    memory_type_str = metadata.get("memory_type", "episodic")
                    if memory_type_str == "episodic" and MemoryCollectionType.EPISODIC not in collection_types:
                        continue
                    elif memory_type_str == "semantic" and MemoryCollectionType.SEMANTIC not in collection_types:
                        continue
                    elif memory_type_str == "procedural" and MemoryCollectionType.PROCEDURAL not in collection_types:
                        continue
                    
                    # Filter by time range
                    if time_range:
                        created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                        if not (time_range[0] <= created_at <= time_range[1]):
                            continue
                    
                    # Filter by importance
                    if importance_threshold:
                        importance_str = metadata.get("importance", "medium")
                        importance = MemoryImportance(importance_str)
                        importance_order = {
                            MemoryImportance.TEMPORARY: 0,
                            MemoryImportance.LOW: 1,
                            MemoryImportance.MEDIUM: 2,
                            MemoryImportance.HIGH: 3,
                            MemoryImportance.CRITICAL: 4
                        }
                        if importance_order[importance] < importance_order[importance_threshold]:
                            continue
                    
                    # Create memory entry
                    memory = MemoryEntry(
                        memory_id=metadata.get("memory_id", str(uuid.uuid4())),
                        agent_id=agent_id,
                        content=result.content,
                        memory_type=MemoryType(memory_type_str),
                        importance=MemoryImportance(metadata.get("importance", "medium")),
                        created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                        last_accessed=datetime.fromisoformat(metadata.get("last_accessed", datetime.utcnow().isoformat())),
                        access_count=metadata.get("access_count", 0),
                        tags=set(metadata.get("tags", [])),
                        context=metadata.get("context", {}),
                        emotional_valence=metadata.get("emotional_valence", 0.0),
                        confidence=metadata.get("confidence", 1.0),
                        session_id=metadata.get("session_id"),
                        expires_at=datetime.fromisoformat(metadata["expires_at"]) if metadata.get("expires_at") else None
                    )
                    
                    memories.append(memory)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse memory result: {str(e)}")
                    continue
            
            # Sort by relevance and return top results
            memories = memories[:max_results]
            
            # Update access statistics
            for memory in memories:
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories from collections: {str(e)}")
            raise
    
    async def consolidate_agent_memories(
        self,
        agent_id: str,
        consolidation_threshold: int = 5
    ) -> int:
        """
        Consolidate frequently accessed memories for an agent.
        
        Args:
            agent_id: Agent to consolidate memories for
            consolidation_threshold: Minimum access count for consolidation
            
        Returns:
            Number of memories consolidated
        """
        try:
            if agent_id not in self.agent_collections:
                return 0
            
            # Get all memories for the agent
            all_memories = await self.memory_system.retrieve_memories(
                agent_id=agent_id,
                query="",  # Empty query to get all
                max_results=1000,
                include_working=False
            )
            
            # Find memories that meet consolidation criteria
            consolidation_candidates = [
                memory for memory in all_memories
                if memory.access_count >= consolidation_threshold
            ]
            
            consolidated_count = 0
            
            # Group related memories for consolidation
            memory_groups = await self._group_related_memories(consolidation_candidates)
            
            for group in memory_groups:
                if len(group) > 1:
                    # Create consolidated memory
                    consolidated_memory = await self._create_consolidated_memory(agent_id, group)
                    
                    # Store consolidated memory
                    await self.store_memory_in_collection(
                        agent_id=agent_id,
                        memory=consolidated_memory,
                        target_collection=MemoryCollectionType.CONSOLIDATED
                    )
                    
                    consolidated_count += 1
            
            self.stats["consolidation_operations"] += 1
            
            logger.info(f"Consolidated {consolidated_count} memory groups for agent {agent_id}")
            return consolidated_count
            
        except Exception as e:
            logger.error(f"Failed to consolidate memories for agent {agent_id}: {str(e)}")
            return 0
    
    async def _update_memory_associations(self, agent_id: str, memory: MemoryEntry) -> None:
        """Update memory associations for a new memory."""
        try:
            # Find related memories based on content similarity and tags
            related_memories = await self.memory_system.retrieve_memories(
                agent_id=agent_id,
                query=memory.content[:100],  # Use first 100 chars as query
                max_results=5,
                include_working=False
            )
            
            # Calculate association strengths
            for related_memory in related_memories:
                if related_memory.memory_id != memory.memory_id:
                    # Calculate similarity based on tags and content
                    tag_overlap = len(memory.tags & related_memory.tags)
                    content_similarity = 0.5  # Simplified similarity calculation
                    
                    association_strength = (tag_overlap * 0.3 + content_similarity * 0.7)
                    
                    # Store bidirectional associations
                    if memory.memory_id not in self.memory_associations:
                        self.memory_associations[memory.memory_id] = {}
                    if related_memory.memory_id not in self.memory_associations:
                        self.memory_associations[related_memory.memory_id] = {}
                    
                    self.memory_associations[memory.memory_id][related_memory.memory_id] = association_strength
                    self.memory_associations[related_memory.memory_id][memory.memory_id] = association_strength
            
            self.stats["association_updates"] += 1
            
        except Exception as e:
            logger.error(f"Failed to update memory associations: {str(e)}")
    
    async def _group_related_memories(self, memories: List[MemoryEntry]) -> List[List[MemoryEntry]]:
        """Group related memories for consolidation."""
        try:
            groups = []
            processed = set()
            
            for memory in memories:
                if memory.memory_id in processed:
                    continue
                
                # Start a new group
                group = [memory]
                processed.add(memory.memory_id)
                
                # Find related memories
                associations = self.memory_associations.get(memory.memory_id, {})
                for related_id, strength in associations.items():
                    if strength > 0.7 and related_id not in processed:  # High association threshold
                        # Find the related memory in our list
                        related_memory = next(
                            (m for m in memories if m.memory_id == related_id),
                            None
                        )
                        if related_memory:
                            group.append(related_memory)
                            processed.add(related_id)
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Failed to group related memories: {str(e)}")
            return [[memory] for memory in memories]  # Return individual groups
    
    async def _create_consolidated_memory(self, agent_id: str, memory_group: List[MemoryEntry]) -> MemoryEntry:
        """Create a consolidated memory from a group of related memories."""
        try:
            # Combine content
            combined_content = " ".join([memory.content for memory in memory_group])
            
            # Combine tags
            combined_tags = set()
            for memory in memory_group:
                combined_tags.update(memory.tags)
            
            # Calculate average emotional valence
            avg_valence = sum(memory.emotional_valence for memory in memory_group) / len(memory_group)
            
            # Use highest importance
            max_importance = max(memory.importance for memory in memory_group)
            
            # Create consolidated memory
            consolidated_memory = MemoryEntry(
                agent_id=agent_id,
                content=f"Consolidated memory: {combined_content}",
                memory_type=MemoryType.SEMANTIC,  # Consolidated memories are semantic
                importance=max_importance,
                tags=combined_tags,
                emotional_valence=avg_valence,
                context={
                    "consolidated_from": [memory.memory_id for memory in memory_group],
                    "consolidation_date": datetime.utcnow().isoformat()
                },
                consolidation_level=1
            )
            
            return consolidated_memory
            
        except Exception as e:
            logger.error(f"Failed to create consolidated memory: {str(e)}")
            raise
    
    def get_agent_collection_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory collection statistics for an agent."""
        if agent_id not in self.agent_collections:
            return {}
        
        collections = self.agent_collections[agent_id]
        stats = {}
        
        for collection_type, info in collections.items():
            stats[collection_type.value] = {
                "document_count": info.document_count,
                "size_bytes": info.size_bytes,
                "last_updated": info.last_updated.isoformat(),
                "consolidation_level": info.consolidation_level
            }
        
        return stats
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide memory collection statistics."""
        return {
            **self.stats,
            "agents_with_collections": len(self.agent_collections),
            "total_associations": len(self.memory_associations)
        }
