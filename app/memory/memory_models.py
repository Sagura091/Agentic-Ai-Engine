"""
Memory Models - Streamlined Foundation.

Simple data models for the unified memory system.
"""

from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import uuid

import structlog

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Types of memory."""
    SHORT_TERM = "short_term"    # Temporary working memory
    LONG_TERM = "long_term"      # Persistent memory


@dataclass
class MemoryEntry:
    """Simple memory entry."""
    id: str
    agent_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    @classmethod
    def create(
        cls,
        agent_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "MemoryEntry":
        """Create a new memory entry."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            created_at=now,
            last_accessed=now
        )


@dataclass
class MemoryCollection:
    """Collection of memories for an agent."""
    agent_id: str
    short_term_memories: Dict[str, MemoryEntry]
    long_term_memories: Dict[str, MemoryEntry]
    created_at: datetime
    last_updated: datetime
    
    @classmethod
    def create(cls, agent_id: str) -> "MemoryCollection":
        """Create a new memory collection for an agent."""
        now = datetime.now()
        return cls(
            agent_id=agent_id,
            short_term_memories={},
            long_term_memories={},
            created_at=now,
            last_updated=now
        )
    
    def add_memory(self, memory: MemoryEntry) -> None:
        """Add a memory to the collection."""
        if memory.memory_type == MemoryType.SHORT_TERM:
            self.short_term_memories[memory.id] = memory
        else:
            self.long_term_memories[memory.id] = memory
        
        self.last_updated = datetime.now()
    
    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        memory = (self.short_term_memories.get(memory_id) or 
                 self.long_term_memories.get(memory_id))
        
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            self.last_updated = datetime.now()
        
        return memory
    
    def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        removed = False
        
        if memory_id in self.short_term_memories:
            del self.short_term_memories[memory_id]
            removed = True
        elif memory_id in self.long_term_memories:
            del self.long_term_memories[memory_id]
            removed = True
        
        if removed:
            self.last_updated = datetime.now()
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory collection statistics."""
        return {
            "agent_id": self.agent_id,
            "short_term_count": len(self.short_term_memories),
            "long_term_count": len(self.long_term_memories),
            "total_count": len(self.short_term_memories) + len(self.long_term_memories),
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
