"""
Unified Memory System - THE Memory System for Multi-Agent Architecture.

This is THE ONLY memory system in the entire application.
All memory operations flow through this unified system.

CORE ARCHITECTURE:
- Agent-specific memory isolation using ChromaDB collections
- Short-term memory (24h TTL) and Long-term memory (persistent)
- Integration with UnifiedRAGSystem for vector storage
- Simple, clean, fast operations

DESIGN PRINCIPLES:
- One memory system to rule them all
- Agent isolation through collections
- Simple short/long term model
- No complexity unless absolutely necessary

PHASE 2 ENHANCEMENT:
✅ Integration with UnifiedRAGSystem
✅ ChromaDB-based storage
✅ Agent-specific memory collections
✅ Automatic cleanup and TTL
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import structlog

from .memory_models import MemoryType, MemoryEntry, MemoryCollection

logger = structlog.get_logger(__name__)


class UnifiedMemorySystem:
    """
    Unified Memory System - THE Memory System.

    SIMPLIFIED ARCHITECTURE:
    - Uses UnifiedRAGSystem for storage
    - Agent-specific memory collections
    - Short-term and long-term memory
    - Automatic cleanup and TTL
    """

    def __init__(self, unified_rag=None):
        """Initialize THE unified memory system."""
        self.unified_rag = unified_rag
        self.is_initialized = False

        # Agent memory collections - SIMPLIFIED
        self.agent_memories: Dict[str, MemoryCollection] = {}

        # Simple configuration - OPTIMIZED
        self.config = {
            "max_short_term_memories": 1000,
            "max_long_term_memories": 10000,
            "short_term_ttl_hours": 24,
            "cleanup_interval_hours": 6
        }

        # Basic stats
        self.stats = {
            "total_agents": 0,
            "total_memories": 0,
            "short_term_memories": 0,
            "long_term_memories": 0
        }
        
        logger.info("Unified memory system initialized")
    
    async def initialize(self) -> None:
        """Initialize the memory system."""
        try:
            if self.is_initialized:
                return
            
            # Ensure unified RAG is initialized if provided
            if self.unified_rag and not self.unified_rag.is_initialized:
                await self.unified_rag.initialize()
            
            self.is_initialized = True
            logger.info("Memory system initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {str(e)}")
            raise
    
    async def create_agent_memory(self, agent_id: str) -> MemoryCollection:
        """Create memory collection for a new agent."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if agent_id in self.agent_memories:
                logger.warning(f"Memory collection already exists for agent {agent_id}")
                return self.agent_memories[agent_id]
            
            # Create memory collection
            collection = MemoryCollection.create(agent_id)
            self.agent_memories[agent_id] = collection
            
            # Update stats
            self.stats["total_agents"] += 1
            
            logger.info(f"Created memory collection for agent {agent_id}")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create memory for agent {agent_id}: {str(e)}")
            raise
    
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
