"""
Unified Memory System for Multi-Agent Architecture.

This module provides a comprehensive memory system that unifies all memory
operations across the platform with agent-specific isolation and efficient
resource utilization.

Features:
- Unified short-term and long-term memory
- Agent-specific memory isolation
- Memory consolidation and decay
- Efficient retrieval and associations
- Memory sharing capabilities
- Performance optimization
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import structlog
from pydantic import BaseModel, Field

from .unified_rag_system import UnifiedRAGSystem
from .agent_isolation_manager import AgentIsolationManager, ResourceType
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the unified system."""
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
class MemoryEntry:
    """Unified memory entry structure."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: MemoryImportance = MemoryImportance.MEDIUM
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    
    # Content and context
    tags: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Associations and relationships
    related_memories: Set[str] = field(default_factory=set)
    association_strength: Dict[str, float] = field(default_factory=dict)
    
    # Memory properties
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    confidence: float = 1.0
    consolidation_level: int = 0
    
    # Session information
    session_id: Optional[str] = None
    source: str = "agent"


class UnifiedMemoryConfig(BaseModel):
    """Configuration for the unified memory system."""
    # Memory limits
    max_working_memory: int = Field(default=50, description="Max working memory items per agent")
    max_episodic_memory: int = Field(default=10000, description="Max episodic memories per agent")
    max_semantic_memory: int = Field(default=5000, description="Max semantic memories per agent")
    
    # Retention settings
    default_retention_days: int = Field(default=30, description="Default memory retention in days")
    critical_retention_days: int = Field(default=365, description="Critical memory retention in days")
    
    # Consolidation settings
    consolidation_threshold: int = Field(default=5, description="Access count for consolidation")
    decay_rate: float = Field(default=0.1, description="Memory decay rate")
    
    # Performance settings
    enable_memory_compression: bool = Field(default=True, description="Enable memory compression")
    enable_auto_cleanup: bool = Field(default=True, description="Enable automatic cleanup")
    cleanup_interval_hours: int = Field(default=24, description="Cleanup interval in hours")


class UnifiedMemorySystem:
    """
    Unified Memory System for Multi-Agent Architecture.
    
    Provides comprehensive memory management with agent isolation,
    efficient storage, and intelligent retrieval capabilities.
    """
    
    def __init__(
        self,
        unified_rag: UnifiedRAGSystem,
        isolation_manager: AgentIsolationManager,
        config: Optional[UnifiedMemoryConfig] = None
    ):
        """Initialize the unified memory system."""
        self.unified_rag = unified_rag
        self.isolation_manager = isolation_manager
        self.config = config or UnifiedMemoryConfig()
        
        # Agent memory stores
        self.agent_memories: Dict[str, Dict[MemoryType, Dict[str, MemoryEntry]]] = {}
        self.working_memories: Dict[str, deque] = {}  # agent_id -> deque of working memories
        
        # Memory indices for fast retrieval
        self.memory_indices: Dict[str, Dict[str, Set[str]]] = {}  # agent_id -> {index_key -> memory_ids}
        
        # Performance tracking
        self.stats = {
            "total_memories": 0,
            "memories_created": 0,
            "memories_retrieved": 0,
            "memories_consolidated": 0,
            "memories_expired": 0,
            "cleanup_cycles": 0
        }
        
        self.is_initialized = False
        logger.info("Unified memory system created", config=self.config.dict())
    
    async def initialize(self) -> None:
        """Initialize the unified memory system."""
        try:
            if self.is_initialized:
                logger.warning("Unified memory system already initialized")
                return
            
            logger.info("Initializing unified memory system...")
            
            # Ensure unified RAG is initialized
            if not self.unified_rag.is_initialized:
                await self.unified_rag.initialize()
            
            # Start background cleanup task if enabled
            if self.config.enable_auto_cleanup:
                asyncio.create_task(self._background_cleanup())
            
            self.is_initialized = True
            logger.info("Unified memory system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize unified memory system: {str(e)}")
            raise
    
    async def create_agent_memory(self, agent_id: str) -> None:
        """Create memory structures for a new agent."""
        try:
            if agent_id in self.agent_memories:
                logger.warning(f"Memory structures already exist for agent {agent_id}")
                return
            
            # Initialize memory stores
            self.agent_memories[agent_id] = {
                MemoryType.EPISODIC: {},
                MemoryType.SEMANTIC: {},
                MemoryType.PROCEDURAL: {},
                MemoryType.EMOTIONAL: {}
            }
            
            # Initialize working memory
            self.working_memories[agent_id] = deque(maxlen=self.config.max_working_memory)
            
            # Initialize memory indices
            self.memory_indices[agent_id] = {}
            
            # Ensure agent has collections in unified RAG
            await self.unified_rag.create_agent_ecosystem(agent_id)
            
            logger.info(f"Created memory structures for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to create memory for agent {agent_id}: {str(e)}")
            raise
    
    async def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Optional[Set[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        emotional_valence: float = 0.0
    ) -> str:
        """
        Store a memory for an agent.
        
        Args:
            agent_id: Agent storing the memory
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            tags: Memory tags
            context: Additional context
            session_id: Session identifier
            emotional_valence: Emotional association
            
        Returns:
            Memory ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Check resource quota
            if not await self.isolation_manager.check_resource_quota(agent_id, "memory_items", 1):
                raise ValueError(f"Agent {agent_id} has exceeded memory quota")
            
            # Ensure agent memory exists
            if agent_id not in self.agent_memories:
                await self.create_agent_memory(agent_id)
            
            # Create memory entry
            memory = MemoryEntry(
                agent_id=agent_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or set(),
                context=context or {},
                session_id=session_id,
                emotional_valence=emotional_valence
            )
            
            # Set expiration based on importance
            if importance == MemoryImportance.CRITICAL:
                memory.expires_at = datetime.utcnow() + timedelta(days=self.config.critical_retention_days)
            elif importance == MemoryImportance.TEMPORARY:
                memory.expires_at = datetime.utcnow() + timedelta(hours=24)
            else:
                memory.expires_at = datetime.utcnow() + timedelta(days=self.config.default_retention_days)
            
            # Store memory
            if memory_type == MemoryType.WORKING:
                self.working_memories[agent_id].append(memory)
            else:
                self.agent_memories[agent_id][memory_type][memory.memory_id] = memory
            
            # Update indices
            await self._update_memory_indices(agent_id, memory)
            
            # Store in vector database for semantic search
            await self._store_memory_in_vector_db(agent_id, memory)
            
            # Update resource usage
            await self.isolation_manager.update_resource_usage(agent_id, "memory_items", 1)
            
            self.stats["memories_created"] += 1
            self.stats["total_memories"] += 1
            
            logger.info(f"Stored {memory_type.value} memory for agent {agent_id}")
            return memory.memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory for agent {agent_id}: {str(e)}")
            raise
    
    async def retrieve_memories(
        self,
        agent_id: str,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        max_results: int = 10,
        include_working: bool = True,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve memories for an agent.
        
        Args:
            agent_id: Agent retrieving memories
            query: Search query
            memory_types: Types of memory to search
            max_results: Maximum results to return
            include_working: Include working memory
            time_range: Time range filter
            
        Returns:
            List of matching memories
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if agent_id not in self.agent_memories:
                return []
            
            # Default to all memory types
            if memory_types is None:
                memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
            
            # Collect candidate memories
            candidates = []
            
            # Search long-term memories
            for memory_type in memory_types:
                if memory_type in self.agent_memories[agent_id]:
                    for memory in self.agent_memories[agent_id][memory_type].values():
                        if not memory.expires_at or memory.expires_at > datetime.utcnow():
                            candidates.append(memory)
            
            # Include working memory if requested
            if include_working and MemoryType.WORKING in memory_types:
                candidates.extend(self.working_memories[agent_id])
            
            # Apply time range filter
            if time_range:
                start_time, end_time = time_range
                candidates = [
                    m for m in candidates 
                    if start_time <= m.created_at <= end_time
                ]
            
            # Simple relevance scoring (would use embeddings in production)
            scored_memories = []
            query_words = set(query.lower().split())
            
            for memory in candidates:
                # Calculate relevance score
                content_words = set(memory.content.lower().split())
                tag_words = set(tag.lower() for tag in memory.tags)
                
                # Word overlap scoring
                content_overlap = len(query_words & content_words) / max(len(query_words), 1)
                tag_overlap = len(query_words & tag_words) / max(len(query_words), 1)
                
                # Boost by importance and recency
                importance_boost = {
                    MemoryImportance.CRITICAL: 2.0,
                    MemoryImportance.HIGH: 1.5,
                    MemoryImportance.MEDIUM: 1.0,
                    MemoryImportance.LOW: 0.7,
                    MemoryImportance.TEMPORARY: 0.5
                }.get(memory.importance, 1.0)
                
                recency_boost = 1.0 + (1.0 / max((datetime.utcnow() - memory.created_at).days + 1, 1))
                
                score = (content_overlap * 0.7 + tag_overlap * 0.3) * importance_boost * recency_boost
                scored_memories.append((score, memory))
            
            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            results = [memory for score, memory in scored_memories[:max_results]]
            
            # Update access statistics
            for memory in results:
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1
            
            self.stats["memories_retrieved"] += len(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories for agent {agent_id}: {str(e)}")
            raise
    
    async def _update_memory_indices(self, agent_id: str, memory: MemoryEntry) -> None:
        """Update memory indices for fast retrieval."""
        try:
            indices = self.memory_indices[agent_id]
            
            # Tag index
            for tag in memory.tags:
                if tag not in indices:
                    indices[tag] = set()
                indices[tag].add(memory.memory_id)
            
            # Date index
            date_key = memory.created_at.strftime("%Y-%m-%d")
            if date_key not in indices:
                indices[date_key] = set()
            indices[date_key].add(memory.memory_id)
            
            # Type index
            type_key = f"type:{memory.memory_type.value}"
            if type_key not in indices:
                indices[type_key] = set()
            indices[type_key].add(memory.memory_id)
            
        except Exception as e:
            logger.error(f"Failed to update memory indices: {str(e)}")
    
    async def _store_memory_in_vector_db(self, agent_id: str, memory: MemoryEntry) -> None:
        """Store memory in vector database for semantic search."""
        try:
            # Create document for memory
            from .knowledge_base import Document
            
            memory_doc = Document(
                title=f"Memory: {memory.memory_type.value}",
                content=memory.content,
                metadata={
                    "memory_id": memory.memory_id,
                    "agent_id": agent_id,
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance.value,
                    "created_at": memory.created_at.isoformat(),
                    "tags": list(memory.tags),
                    **memory.context
                }
            )
            
            # Store in appropriate memory collection
            collection_type = "episodic_memory" if memory.memory_type == MemoryType.EPISODIC else "semantic_memory"
            await self.unified_rag.add_document_to_agent(agent_id, memory_doc, collection_type)
            
        except Exception as e:
            logger.error(f"Failed to store memory in vector DB: {str(e)}")
    
    async def _background_cleanup(self) -> None:
        """Background task for memory cleanup and maintenance."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                await self._cleanup_expired_memories()
                self.stats["cleanup_cycles"] += 1
                
            except Exception as e:
                logger.error(f"Background cleanup failed: {str(e)}")
    
    async def _cleanup_expired_memories(self) -> None:
        """Clean up expired memories."""
        try:
            now = datetime.utcnow()
            expired_count = 0
            
            for agent_id in self.agent_memories:
                for memory_type in self.agent_memories[agent_id]:
                    expired_ids = []
                    for memory_id, memory in self.agent_memories[agent_id][memory_type].items():
                        if memory.expires_at and memory.expires_at < now:
                            expired_ids.append(memory_id)
                    
                    for memory_id in expired_ids:
                        del self.agent_memories[agent_id][memory_type][memory_id]
                        expired_count += 1
            
            self.stats["memories_expired"] += expired_count
            self.stats["total_memories"] -= expired_count
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired memories")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired memories: {str(e)}")
    
    def get_agent_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an agent."""
        if agent_id not in self.agent_memories:
            return {}
        
        stats = {}
        for memory_type in MemoryType:
            if memory_type == MemoryType.WORKING:
                stats[memory_type.value] = len(self.working_memories.get(agent_id, []))
            else:
                stats[memory_type.value] = len(self.agent_memories[agent_id].get(memory_type, {}))
        
        return stats
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide memory statistics."""
        return {
            **self.stats,
            "total_agents_with_memory": len(self.agent_memories),
            "config": self.config.dict()
        }
