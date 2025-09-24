"""
Revolutionary RAG (Retrieval-Augmented Generation) System for Agentic AI.

This package provides a comprehensive RAG implementation with ChromaDB integration,
advanced document processing, and seamless LangChain/LangGraph integration for
revolutionary agent capabilities.

Key Features:
- ChromaDB vector database integration
- Multi-modal document ingestion pipeline
- Dense and sparse retrieval with vision models
- LangGraph-native RAG tools
- Multi-agent knowledge sharing
- Real-time knowledge updates

Components:
- Core: Knowledge base management and vector operations
- Ingestion: Document processing and embedding pipeline
- Retrieval: Advanced search and ranking strategies
- Agents: RAG-enabled LangGraph agents
- Tools: Knowledge tools for agent integration
"""

# Unified RAG System - Core Components
from .core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig, AgentCollections
from .core.collection_based_kb_manager import CollectionBasedKBManager, KnowledgeBaseInfo, AccessLevel
from .core.agent_isolation_manager import AgentIsolationManager, AgentIsolationProfile, IsolationLevel, ResourceQuota
# Memory system is now in app.memory, not in RAG core

# Supporting Components (optional - may not be available)
try:
    from .core.global_embedding_manager import GlobalEmbeddingManager
except ImportError:
    GlobalEmbeddingManager = None

try:
    from .core.advanced_caching import AdvancedCacheManager
except ImportError:
    AdvancedCacheManager = None

try:
    from .ingestion.pipeline import RevolutionaryIngestionPipeline as IngestionPipeline
except ImportError:
    IngestionPipeline = None

try:
    from .ingestion.processors import DocumentProcessor
except ImportError:
    DocumentProcessor = None
# Tools (optional - may not be available)
try:
    from .tools.knowledge_tools import (
        KnowledgeSearchTool,
        DocumentIngestTool,
        FactCheckTool,
        SynthesisTool
    )
except ImportError:
    KnowledgeSearchTool = None
    DocumentIngestTool = None
    FactCheckTool = None
    SynthesisTool = None

try:
    from .tools.enhanced_knowledge_tools import (
        EnhancedKnowledgeSearchTool,
        AgentDocumentIngestTool,
        AgentMemoryTool
    )
except ImportError:
    EnhancedKnowledgeSearchTool = None
    AgentDocumentIngestTool = None
    AgentMemoryTool = None

# Build __all__ list dynamically based on available components
__all__ = [
    # Unified RAG System - Core Components (always available)
    "UnifiedRAGSystem",
    "UnifiedRAGConfig",
    "AgentCollections",
    "CollectionBasedKBManager",
    "KnowledgeBaseInfo",
    "AccessLevel",
    "AgentIsolationManager",
    "AgentIsolationProfile",
    "IsolationLevel",
    "ResourceQuota"
]

# Add optional components if available
if GlobalEmbeddingManager:
    __all__.append("GlobalEmbeddingManager")
if AdvancedCacheManager:
    __all__.append("AdvancedCacheManager")
if IngestionPipeline:
    __all__.append("IngestionPipeline")
if DocumentProcessor:
    __all__.append("DocumentProcessor")
if KnowledgeSearchTool:
    __all__.extend(["KnowledgeSearchTool", "DocumentIngestTool", "FactCheckTool", "SynthesisTool"])
if EnhancedKnowledgeSearchTool:
    __all__.extend(["EnhancedKnowledgeSearchTool", "AgentDocumentIngestTool", "AgentMemoryTool"])

# Version information
__version__ = "1.0.0"
__author__ = "Agentic AI Team"
__description__ = "Revolutionary RAG system for unlimited agent knowledge capabilities"

# RAG system features - Revolutionary Multi-Agent Capabilities
RAG_SYSTEM_FEATURES = [
    # Core RAG features
    "chromadb_integration",
    "multi_modal_ingestion",
    "dense_sparse_retrieval",
    "vision_model_support",
    "langgraph_integration",
    "intelligent_chunking",
    "metadata_enrichment",

    # Revolutionary multi-agent features
    "agent_specific_knowledge",
    "hierarchical_collections",
    "knowledge_isolation",
    "memory_integration",
    "contextual_retrieval",
    "collaborative_learning",
    "knowledge_sharing_protocols",
    "adaptive_retrieval_strategies",
    "multi_tenancy_support",
    "performance_optimization",
    "real_time_updates",
    "knowledge_lifecycle_management",
    "permission_based_access",
    "query_expansion",
    "result_reranking",
    "connection_pooling",
    "advanced_caching"
]

# Supported document types
SUPPORTED_DOCUMENT_TYPES = [
    "pdf",
    "docx", 
    "txt",
    "html",
    "markdown",
    "json",
    "csv",
    "xml",
    "rtf",
    "web_pages"
]

# Embedding models configuration
EMBEDDING_MODELS = {
    "default": "all-MiniLM-L6-v2",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "code": "microsoft/codebert-base",
    "scientific": "allenai/scibert_scivocab_uncased"
}

# Revolutionary Multi-Agent ChromaDB Collections Configuration
CHROMA_COLLECTIONS = {
    # Global collections - accessible to all agents
    "global_knowledge": "Universal knowledge shared across all agents",
    "shared_procedures": "Standard operating procedures and workflows",
    "public_documents": "Publicly available documents and resources",

    # Domain-specific collections
    "domain_research": "Research-specific knowledge and methodologies",
    "domain_creative": "Creative writing and artistic knowledge",
    "domain_technical": "Technical documentation and code examples",
    "domain_analysis": "Data analysis and business intelligence",

    # Agent-specific collections (templates)
    "agent_{agent_id}_private": "Agent's private knowledge and documents",
    "agent_{agent_id}_memory": "Agent's episodic and semantic memories",
    "agent_{agent_id}_session": "Agent's session-specific temporary knowledge",
    "agent_{agent_id}_learned": "Agent's learned patterns and preferences",

    # Collaborative collections
    "shared_research": "Collaborative research between agents",
    "shared_projects": "Multi-agent project knowledge",
    "team_memories": "Shared team experiences and learnings",

    # Specialized collections
    "conversation_context": "Session-specific conversation context",
    "web_research": "Cached web research results",
    "fact_checking": "Verified facts and sources",
    "knowledge_conflicts": "Conflicting information requiring resolution"
}

def get_rag_system_info():
    """Get comprehensive information about the RAG system."""
    return {
        "version": __version__,
        "description": __description__,
        "features": RAG_SYSTEM_FEATURES,
        "supported_documents": SUPPORTED_DOCUMENT_TYPES,
        "embedding_models": EMBEDDING_MODELS,
        "collections": CHROMA_COLLECTIONS
    }

# Revolutionary Multi-Agent RAG Configuration
DEFAULT_RAG_CONFIG = {
    "vector_store": {
        "provider": "chromadb",
        "persist_directory": "./data/chroma",
        "collection_metadata": {"hnsw:space": "cosine"},
        "connection_pool_size": 10,
        "max_batch_size": 128,
        "enable_multi_collection": True
    },
    "embeddings": {
        "model": EMBEDDING_MODELS["default"],
        "batch_size": 32,
        "normalize": True,
        "cache_embeddings": True,
        "cache_size": 10000,
        "enable_hybrid_embeddings": True
    },
    "chunking": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "strategy": "semantic",
        "enable_agent_tagging": True,
        "enable_scope_classification": True
    },
    "retrieval": {
        "top_k": 10,
        "score_threshold": 0.7,
        "rerank": True,
        "enable_query_expansion": True,
        "enable_hybrid_search": True,
        "memory_boost_factor": 1.2,
        "recency_boost_factor": 1.1,
        "enable_contextual_retrieval": True
    },
    "multi_agent": {
        "enable_agent_isolation": True,
        "enable_knowledge_sharing": True,
        "enable_memory_integration": True,
        "enable_collaborative_learning": True,
        "default_retention_days": 30,
        "max_memory_items": 10000,
        "enable_auto_cleanup": True,
        "enable_permission_system": True
    },
    "performance": {
        "enable_caching": True,
        "cache_ttl": 3600,
        "enable_connection_pooling": True,
        "enable_batch_processing": True,
        "max_concurrent_queries": 50,
        "enable_async_processing": True
    },
    "knowledge_lifecycle": {
        "enable_auto_archiving": True,
        "archive_threshold_days": 90,
        "enable_knowledge_expiration": True,
        "cleanup_interval_hours": 24
    }
}

# Export configuration
__config__ = {
    "name": "rag",
    "version": __version__,
    "description": __description__,
    "features": RAG_SYSTEM_FEATURES,
    "default_config": DEFAULT_RAG_CONFIG
}
