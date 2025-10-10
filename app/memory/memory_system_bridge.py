"""
Memory System Bridge

CRITICAL FIX: Bridge between UnifiedMemorySystem and PersistentMemorySystem.

This module provides a unified interface that bridges the two memory systems:
- UnifiedMemorySystem (revolutionary features, in-memory with RAG)
- PersistentMemorySystem (PostgreSQL persistence, autonomous agents)

The bridge ensures:
1. Consistent API across both systems
2. Automatic synchronization between systems
3. Seamless migration path to unified architecture
4. No breaking changes to existing code
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from app.memory.unified_memory_system import UnifiedMemorySystem
from app.memory.memory_models import MemoryType, MemoryImportance, MemoryEntry
from app.agents.autonomous.persistent_memory import PersistentMemorySystem

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class MemorySystemBridge:
    """
    Bridge between UnifiedMemorySystem and PersistentMemorySystem.
    
    Provides a unified interface that works with both memory systems,
    ensuring consistency and enabling gradual migration to a fully
    unified architecture.
    """
    
    def __init__(
        self,
        unified_system: UnifiedMemorySystem,
        enable_sync: bool = True
    ):
        """
        Initialize the memory system bridge.
        
        Args:
            unified_system: The UnifiedMemorySystem instance
            enable_sync: Whether to enable automatic synchronization
        """
        self.unified_system = unified_system
        self.enable_sync = enable_sync
        
        # Track persistent memory systems by agent_id
        self.persistent_systems: Dict[str, PersistentMemorySystem] = {}

        logger.info(
            "Memory system bridge initialized",
            LogCategory.MEMORY_OPERATIONS,
            "app.memory.memory_system_bridge.MemorySystemBridge",
            data={"enable_sync": enable_sync}
        )

    def register_persistent_system(
        self,
        agent_id: str,
        persistent_system: PersistentMemorySystem
    ):
        """
        Register a PersistentMemorySystem for an agent.

        This enables bidirectional synchronization between the two systems.

        Args:
            agent_id: The agent's unique identifier
            persistent_system: The PersistentMemorySystem instance
        """
        self.persistent_systems[agent_id] = persistent_system
        logger.info(
            f"Persistent memory system registered for agent {agent_id}",
            LogCategory.MEMORY_OPERATIONS,
            "app.memory.memory_system_bridge.MemorySystemBridge",
            agent_id=agent_id
        )
    
    async def add_memory(
        self,
        agent_id: str,
        memory_type: Union[MemoryType, str],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Union[MemoryImportance, str] = MemoryImportance.MEDIUM,
        emotional_valence: float = 0.0,
        tags: Optional[set] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add memory through the bridge.
        
        Automatically stores in both systems if agent has both.
        
        Args:
            agent_id: The agent's unique identifier
            memory_type: Type of memory (episodic, semantic, etc.)
            content: Memory content
            metadata: Optional metadata
            importance: Memory importance level
            emotional_valence: Emotional value (-1.0 to 1.0)
            tags: Optional tags
            context: Optional context
            
        Returns:
            Memory ID
        """
        # Convert string types to enums if needed
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        if isinstance(importance, str):
            importance = MemoryImportance(importance)
        
        # Always store in UnifiedMemorySystem (has database persistence now)
        memory_id = await self.unified_system.add_memory(
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata,
            importance=importance,
            emotional_valence=emotional_valence,
            tags=tags,
            context=context
        )
        
        # If agent has PersistentMemorySystem, sync to it as well
        if self.enable_sync and agent_id in self.persistent_systems:
            try:
                persistent_system = self.persistent_systems[agent_id]
                await persistent_system.store_memory(
                    content=content,
                    memory_type=memory_type.value,
                    importance=importance.value,
                    metadata=metadata or {},
                    emotional_valence=emotional_valence
                )
                logger.debug(
                    f"Memory synced to persistent system for agent {agent_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.memory_system_bridge.MemorySystemBridge",
                    data={"agent_id": agent_id, "memory_id": memory_id}
                )
            except Exception as e:
                logger.warn(
                    "Failed to sync memory to persistent system",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.memory_system_bridge.MemorySystemBridge",
                    data={"agent_id": agent_id, "error": str(e)}
                )
        
        return memory_id
    
    async def retrieve_memories(
        self,
        agent_id: str,
        query: Optional[str] = None,
        memory_types: Optional[List[Union[MemoryType, str]]] = None,
        limit: int = 10,
        relevance_threshold: float = 0.3
    ) -> List[MemoryEntry]:
        """
        Retrieve memories through the bridge.
        
        Uses the most appropriate system for the agent.
        
        Args:
            agent_id: The agent's unique identifier
            query: Optional search query
            memory_types: Optional list of memory types to filter
            limit: Maximum number of memories to return
            relevance_threshold: Minimum relevance score
            
        Returns:
            List of memory entries
        """
        # Check if agent has PersistentMemorySystem
        if agent_id in self.persistent_systems:
            try:
                persistent_system = self.persistent_systems[agent_id]
                
                # Convert memory types to strings
                type_filters = None
                if memory_types:
                    type_filters = [
                        mt.value if isinstance(mt, MemoryType) else mt
                        for mt in memory_types
                    ]
                
                memories = await persistent_system.retrieve_memories(
                    query=query or "",
                    memory_types=type_filters,
                    limit=limit
                )
                
                logger.debug(
                    f"Retrieved {len(memories)} memories from persistent system",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.memory_system_bridge.MemorySystemBridge",
                    data={"agent_id": agent_id, "count": len(memories)}
                )

                return memories

            except Exception as e:
                logger.warn(
                    "Failed to retrieve from persistent system, falling back to unified",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.memory_system_bridge.MemorySystemBridge",
                    data={"agent_id": agent_id, "error": str(e)}
                )
        
        # Use UnifiedMemorySystem
        if query:
            # Use active retrieval if query provided
            result = await self.unified_system.active_retrieve_memories(
                agent_id=agent_id,
                current_task=query,
                conversation_context="",
                max_memories=limit,
                relevance_threshold=relevance_threshold
            )
            return result.memories
        else:
            # Use search if no query
            type_filters = memory_types or [MemoryType.EPISODIC, MemoryType.SEMANTIC]
            memories = await self.unified_system.search_memories(
                agent_id=agent_id,
                query="",
                memory_types=type_filters,
                limit=limit
            )
            return memories
    
    async def consolidate_memories(self, agent_id: str) -> Dict[str, Any]:
        """
        Consolidate memories for an agent.
        
        Runs consolidation on the appropriate system.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            Consolidation results
        """
        # Check if agent has PersistentMemorySystem
        if agent_id in self.persistent_systems:
            try:
                persistent_system = self.persistent_systems[agent_id]
                result = await persistent_system.consolidate_memories()
                
                logger.info(
                    "Consolidated memories using persistent system",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.memory_system_bridge.MemorySystemBridge",
                    data={"agent_id": agent_id, "result": result}
                )

                return result

            except Exception as e:
                logger.warn(
                    "Failed to consolidate with persistent system, using unified",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.memory_system_bridge.MemorySystemBridge",
                    data={"agent_id": agent_id, "error": str(e)}
                )

        # Use UnifiedMemorySystem
        result = await self.unified_system.run_consolidation_for_agent(agent_id)

        logger.info(
            "Consolidated memories using unified system",
            LogCategory.MEMORY_OPERATIONS,
            "app.memory.memory_system_bridge.MemorySystemBridge",
            agent_id=agent_id,
            result=result
        )
        
        return result
    
    async def get_agent_context(
        self,
        agent_id: str,
        current_task: str = "",
        conversation_context: str = ""
    ) -> str:
        """
        Get comprehensive agent context.
        
        Args:
            agent_id: The agent's unique identifier
            current_task: Current task description
            conversation_context: Conversation context
            
        Returns:
            Formatted context string
        """
        return await self.unified_system.get_agent_context(
            agent_id=agent_id,
            current_task=current_task,
            conversation_context=conversation_context
        )
    
    def get_system_type(self, agent_id: str) -> str:
        """
        Get the memory system type for an agent.
        
        Args:
            agent_id: The agent's unique identifier
            
        Returns:
            "persistent", "unified", or "both"
        """
        has_persistent = agent_id in self.persistent_systems
        has_unified = agent_id in self.unified_system.agent_memories
        
        if has_persistent and has_unified:
            return "both"
        elif has_persistent:
            return "persistent"
        elif has_unified:
            return "unified"
        else:
            return "none"

