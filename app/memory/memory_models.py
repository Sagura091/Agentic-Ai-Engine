"""
Revolutionary Memory Models - Enhanced Foundation for Agentic AI.

Advanced data models supporting the complete revolutionary memory architecture
with Core Memory, Knowledge Vault, Resource Memory, and Active Retrieval.
"""

from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Set, List, Union
import uuid
import json
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class MemoryType(str, Enum):
    """Revolutionary memory types based on MIRIX and state-of-the-art research."""
    # Basic memory types (existing)
    SHORT_TERM = "short_term"        # Temporary working memory
    LONG_TERM = "long_term"          # Persistent memory

    # Revolutionary memory types (new)
    CORE = "core"                    # Always-visible persistent context (persona + human facts)
    EPISODIC = "episodic"            # Time-stamped events and experiences
    SEMANTIC = "semantic"            # Abstract knowledge and concepts
    PROCEDURAL = "procedural"        # Skills, procedures, and how-to knowledge
    RESOURCE = "resource"            # Documents, files, and media storage
    KNOWLEDGE_VAULT = "knowledge_vault"  # Secure sensitive information storage
    WORKING = "working"              # Current context and temporary information


class MemoryImportance(str, Enum):
    """Memory importance levels for consolidation and retention."""
    CRITICAL = "critical"            # Must be retained permanently
    HIGH = "high"                    # Important for long-term retention
    MEDIUM = "medium"                # Moderate importance
    LOW = "low"                      # Can be forgotten if space is needed
    TEMPORARY = "temporary"          # Short-term only


class SensitivityLevel(str, Enum):
    """Sensitivity levels for Knowledge Vault entries."""
    PUBLIC = "public"                # No sensitivity
    INTERNAL = "internal"            # Internal use only
    CONFIDENTIAL = "confidential"    # Confidential information
    SECRET = "secret"                # Highly sensitive information


@dataclass
class MemoryEntry:
    """Revolutionary memory entry with enhanced capabilities."""
    id: str
    agent_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0

    # Revolutionary enhancements
    importance: MemoryImportance = MemoryImportance.MEDIUM
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    confidence: float = 1.0
    tags: Set[str] = field(default_factory=set)
    associations: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength
    context: Dict[str, Any] = field(default_factory=dict)
    consolidation_level: int = 0  # 0 = new, higher = more consolidated
    decay_rate: float = 0.1
    expires_at: Optional[datetime] = None
    source: str = "agent"
    session_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        agent_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        emotional_valence: float = 0.0,
        tags: Optional[Set[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> "MemoryEntry":
        """Create a new revolutionary memory entry."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            created_at=now,
            last_accessed=now,
            importance=importance,
            emotional_valence=emotional_valence,
            tags=tags or set(),
            context=context or {}
        )


@dataclass
class CoreMemoryBlock:
    """Core memory block for always-visible persistent context."""
    block_type: str  # "persona" or "human"
    content: str
    last_updated: datetime = field(default_factory=datetime.now)
    max_size: int = 2000  # Character limit

    def update_content(self, new_content: str) -> bool:
        """Update block content with size validation."""
        if len(new_content) > self.max_size:
            logger.warning(f"Core memory block content exceeds max size",
                         block_type=self.block_type,
                         size=len(new_content),
                         max_size=self.max_size)
            return False

        self.content = new_content
        self.last_updated = datetime.now()
        return True


@dataclass
class KnowledgeVaultEntry:
    """Secure entry for sensitive information in Knowledge Vault."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = "credential"  # credential, api_key, contact_info, bookmark
    title: str = ""
    secret_value: str = ""
    sensitivity_level: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    source: str = "user_provided"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def access(self) -> str:
        """Access the secret value with logging."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        logger.info("Knowledge vault entry accessed",
                   entry_id=self.entry_id,
                   entry_type=self.entry_type,
                   sensitivity=self.sensitivity_level.value)
        return self.secret_value


@dataclass
class ResourceMemoryEntry:
    """Entry for document and file storage in Resource Memory."""
    resource_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    summary: str = ""
    resource_type: str = "document"  # document, image, audio, video, markdown, pdf_text
    content: str = ""  # Full or excerpted content
    file_path: Optional[str] = None  # Path to actual file if stored separately
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0

    def access(self) -> str:
        """Access the resource content with logging."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.content
    
@dataclass
class RevolutionaryMemoryCollection:
    """Revolutionary memory collection supporting all memory types."""
    agent_id: str
    created_at: datetime
    last_updated: datetime

    # Core Memory (always visible)
    core_memory: Dict[str, CoreMemoryBlock] = field(default_factory=lambda: {
        "persona": CoreMemoryBlock("persona", ""),
        "human": CoreMemoryBlock("human", "")
    })

    # Traditional memory types
    short_term_memories: Dict[str, MemoryEntry] = field(default_factory=dict)
    long_term_memories: Dict[str, MemoryEntry] = field(default_factory=dict)

    # Revolutionary memory types
    episodic_memories: Dict[str, MemoryEntry] = field(default_factory=dict)
    semantic_memories: Dict[str, MemoryEntry] = field(default_factory=dict)
    procedural_memories: Dict[str, MemoryEntry] = field(default_factory=dict)
    working_memories: List[MemoryEntry] = field(default_factory=list)

    # Specialized storage
    resource_memory: Dict[str, ResourceMemoryEntry] = field(default_factory=dict)
    knowledge_vault: Dict[str, KnowledgeVaultEntry] = field(default_factory=dict)

    # Memory indices for fast retrieval
    tag_index: Dict[str, Set[str]] = field(default_factory=dict)
    temporal_index: Dict[str, List[str]] = field(default_factory=dict)  # date -> memory_ids
    association_graph: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def create(cls, agent_id: str) -> "RevolutionaryMemoryCollection":
        """Create a new revolutionary memory collection for an agent."""
        now = datetime.now()
        return cls(
            agent_id=agent_id,
            created_at=now,
            last_updated=now
        )

    def add_memory(self, memory: MemoryEntry) -> None:
        """Add a memory to the appropriate collection."""
        memory_store = self._get_memory_store(memory.memory_type)

        if memory.memory_type == MemoryType.WORKING:
            # Working memory is a list with max size
            self.working_memories.append(memory)
            if len(self.working_memories) > 20:  # Max working memory size
                self.working_memories.pop(0)
        else:
            memory_store[memory.id] = memory

        # Update indices
        self._update_indices(memory)
        self.last_updated = datetime.now()

    def _get_memory_store(self, memory_type: MemoryType) -> Dict[str, MemoryEntry]:
        """Get the appropriate memory store for a memory type."""
        store_mapping = {
            MemoryType.SHORT_TERM: self.short_term_memories,
            MemoryType.LONG_TERM: self.long_term_memories,
            MemoryType.EPISODIC: self.episodic_memories,
            MemoryType.SEMANTIC: self.semantic_memories,
            MemoryType.PROCEDURAL: self.procedural_memories,
        }
        return store_mapping.get(memory_type, self.long_term_memories)

    def _update_indices(self, memory: MemoryEntry) -> None:
        """Update memory indices for fast retrieval."""
        # Tag index
        for tag in memory.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(memory.id)

        # Temporal index
        date_key = memory.created_at.strftime("%Y-%m-%d")
        if date_key not in self.temporal_index:
            self.temporal_index[date_key] = []
        self.temporal_index[date_key].append(memory.id)

        # Association graph
        if memory.id not in self.association_graph:
            self.association_graph[memory.id] = {}
        for assoc_id, strength in memory.associations.items():
            self.association_graph[memory.id][assoc_id] = strength

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID from any collection."""
        # Search all memory stores
        all_stores = [
            self.short_term_memories,
            self.long_term_memories,
            self.episodic_memories,
            self.semantic_memories,
            self.procedural_memories
        ]

        for store in all_stores:
            if memory_id in store:
                memory = store[memory_id]
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                self.last_updated = datetime.now()
                return memory

        # Check working memory
        for memory in self.working_memories:
            if memory.id == memory_id:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                self.last_updated = datetime.now()
                return memory

        return None

    def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory by ID from any collection."""
        # Search all memory stores
        all_stores = [
            self.short_term_memories,
            self.long_term_memories,
            self.episodic_memories,
            self.semantic_memories,
            self.procedural_memories
        ]

        for store in all_stores:
            if memory_id in store:
                del store[memory_id]
                self._remove_from_indices(memory_id)
                self.last_updated = datetime.now()
                return True

        # Check working memory
        self.working_memories = [m for m in self.working_memories if m.id != memory_id]
        return False

    def _remove_from_indices(self, memory_id: str) -> None:
        """Remove memory from all indices."""
        # Remove from tag index
        for tag_set in self.tag_index.values():
            tag_set.discard(memory_id)

        # Remove from temporal index
        for memory_list in self.temporal_index.values():
            if memory_id in memory_list:
                memory_list.remove(memory_id)

        # Remove from association graph
        if memory_id in self.association_graph:
            del self.association_graph[memory_id]

        # Remove associations to this memory
        for assoc_dict in self.association_graph.values():
            assoc_dict.pop(memory_id, None)

    def add_resource(self, resource: ResourceMemoryEntry) -> None:
        """Add a resource to Resource Memory."""
        self.resource_memory[resource.resource_id] = resource
        self.last_updated = datetime.now()

    def add_vault_entry(self, entry: KnowledgeVaultEntry) -> None:
        """Add an entry to Knowledge Vault."""
        self.knowledge_vault[entry.entry_id] = entry
        self.last_updated = datetime.now()

    def update_core_memory(self, block_type: str, content: str) -> bool:
        """Update core memory block."""
        if block_type in self.core_memory:
            success = self.core_memory[block_type].update_content(content)
            if success:
                self.last_updated = datetime.now()
            return success
        return False

    def get_core_memory_context(self) -> str:
        """Get formatted core memory context for agent prompts."""
        persona_content = self.core_memory["persona"].content
        human_content = self.core_memory["human"].content

        context = ""
        if persona_content:
            context += f"<persona>\n{persona_content}\n</persona>\n"
        if human_content:
            context += f"<human>\n{human_content}\n</human>"

        return context.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory collection statistics."""
        return {
            "agent_id": self.agent_id,
            "short_term_count": len(self.short_term_memories),
            "long_term_count": len(self.long_term_memories),
            "episodic_count": len(self.episodic_memories),
            "semantic_count": len(self.semantic_memories),
            "procedural_count": len(self.procedural_memories),
            "working_count": len(self.working_memories),
            "resource_count": len(self.resource_memory),
            "vault_count": len(self.knowledge_vault),
            "total_count": (len(self.short_term_memories) + len(self.long_term_memories) +
                          len(self.episodic_memories) + len(self.semantic_memories) +
                          len(self.procedural_memories) + len(self.working_memories)),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "core_memory_size": {
                "persona": len(self.core_memory["persona"].content),
                "human": len(self.core_memory["human"].content)
            }
        }


# Backward compatibility alias
MemoryCollection = RevolutionaryMemoryCollection
