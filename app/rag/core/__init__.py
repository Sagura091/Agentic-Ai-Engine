"""
Core RAG components for unified multi-agent architecture.

This module contains the unified components for the RAG system including
the unified RAG system, collection-based knowledge management, agent isolation,
and memory systems.
"""

# Import unified system components
from .unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig, Document, KnowledgeQuery, KnowledgeResult, KnowledgeConfig
from .collection_based_kb_manager import CollectionBasedKBManager
from .agent_isolation_manager import AgentIsolationManager
from .unified_memory_system import UnifiedMemorySystem, UnifiedMemoryConfig
from .agent_memory_collections import AgentMemoryCollections
from .global_embedding_manager import GlobalEmbeddingManager
from .advanced_caching import AdvancedCacheManager

__all__ = [
    # Unified RAG System
    "UnifiedRAGSystem",
    "UnifiedRAGConfig",
    "Document",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KnowledgeConfig",

    # Knowledge and Memory Management
    "CollectionBasedKBManager",
    "AgentIsolationManager",
    "UnifiedMemorySystem",
    "UnifiedMemoryConfig",
    "AgentMemoryCollections",

    # Supporting Components
    "GlobalEmbeddingManager",
    "AdvancedCacheManager"
]
