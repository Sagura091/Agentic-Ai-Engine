"""
Revolutionary RAG System for Agentic AI.

Unified RAG implementation with ChromaDB integration, multi-modal processing,
and seamless agent integration for revolutionary AI capabilities.
"""

# Core RAG System Components
from .core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig, AgentCollections
from .core.collection_based_kb_manager import CollectionBasedKBManager, KnowledgeBaseInfo, AccessLevel
from .core.agent_isolation_manager import AgentIsolationManager, AgentIsolationProfile, IsolationLevel, ResourceQuota

# Essential Components
from .core.embeddings import EmbeddingManager as GlobalEmbeddingManager, get_global_embedding_manager
from .ingestion.pipeline import RevolutionaryIngestionPipeline as IngestionPipeline
from .tools.knowledge_tools import KnowledgeSearchTool, DocumentIngestTool, FactCheckTool, SynthesisTool
from .tools.enhanced_knowledge_tools import EnhancedKnowledgeSearchTool, AgentDocumentIngestTool, AgentMemoryTool

# Exported components
__all__ = [
    # Core RAG System
    "UnifiedRAGSystem",
    "UnifiedRAGConfig",
    "AgentCollections",
    "CollectionBasedKBManager",
    "KnowledgeBaseInfo",
    "AccessLevel",
    "AgentIsolationManager",
    "AgentIsolationProfile",
    "IsolationLevel",
    "ResourceQuota",
    # Essential Components
    "GlobalEmbeddingManager",
    "get_global_embedding_manager",
    "IngestionPipeline",
    # Knowledge Tools
    "KnowledgeSearchTool",
    "DocumentIngestTool",
    "FactCheckTool",
    "SynthesisTool",
    "EnhancedKnowledgeSearchTool",
    "AgentDocumentIngestTool",
    "AgentMemoryTool"
]

# System Information
__version__ = "2.0.0"
__description__ = "Revolutionary RAG system for unlimited agent knowledge capabilities"

# Core Features
CORE_FEATURES = [
    "chromadb_integration",
    "multi_modal_ingestion",
    "agent_isolation",
    "knowledge_tools",
    "vision_support",
    "performance_optimization"
]

# Default Configuration
DEFAULT_CONFIG = {
    "vector_store": {
        "provider": "chromadb",
        "persist_directory": "./data/chroma"
    },
    "embeddings": {
        "model": "all-MiniLM-L6-v2",
        "batch_size": 32
    },
    "retrieval": {
        "top_k": 10,
        "score_threshold": 0.7
    }
}

def get_rag_info():
    """Get RAG system information."""
    return {
        "version": __version__,
        "description": __description__,
        "features": CORE_FEATURES,
        "config": DEFAULT_CONFIG
    }
