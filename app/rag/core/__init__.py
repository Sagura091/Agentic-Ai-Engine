"""
Core RAG Components - THE Unified Multi-Agent Architecture.

This module contains THE ONLY RAG system components in the entire application.
All RAG operations flow through these unified components.

PHASE 1 FOUNDATION COMPLETE:
✅ UnifiedRAGSystem - THE single RAG system
✅ CollectionBasedKBManager - THE knowledge base system
✅ AgentIsolationManager - THE agent isolation system

ARCHITECTURE PRINCIPLES:
- One system to rule them all
- Agent isolation through collections
- Simple, clean, fast operations
- No complexity unless absolutely necessary
"""

# Import THE unified system components - PHASE 1 FOUNDATION
from .unified_rag_system import (
    UnifiedRAGSystem,
    UnifiedRAGConfig,
    Document,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeConfig,
    CollectionType,
    AgentCollections
)
from .collection_based_kb_manager import (
    CollectionBasedKBManager,
    KnowledgeBaseInfo,
    AccessLevel
)
from .agent_isolation_manager import (
    AgentIsolationManager,
    AgentIsolationProfile,
    IsolationLevel,
    ResourceQuota,
    ResourceUsage
)

# Supporting components (will be enhanced in later phases)
try:
    from .embeddings import EmbeddingManager as GlobalEmbeddingManager, get_global_embedding_manager
except ImportError:
    GlobalEmbeddingManager = None
    get_global_embedding_manager = None

__all__ = [
    # PHASE 1: THE Foundation - Core RAG System
    "UnifiedRAGSystem",
    "UnifiedRAGConfig",
    "Document",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KnowledgeConfig",
    "CollectionType",
    "AgentCollections",

    # PHASE 1: THE Knowledge Base System
    "CollectionBasedKBManager",
    "KnowledgeBaseInfo",
    "AccessLevel",

    # PHASE 1: THE Agent Isolation System
    "AgentIsolationManager",
    "AgentIsolationProfile",
    "IsolationLevel",
    "ResourceQuota",
    "ResourceUsage",

    # Supporting Components (optional)
    "GlobalEmbeddingManager",
    "get_global_embedding_manager"
]
