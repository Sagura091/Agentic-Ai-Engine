"""
Advanced Retrieval Components for RAG.

This module provides comprehensive retrieval capabilities including:
- Advanced retrieval pipeline orchestration
- Query expansion strategies
- BM25 sparse retrieval
- Hybrid fusion (dense + sparse)
- Cross-encoder reranking
- MMR diversity selection
- Contextual compression
"""

from .advanced_retrieval_pipeline import (
    AdvancedRetrievalPipeline,
    PipelineConfig,
    RetrievalMode,
    RetrievalResult
)

from .query_expansion import (
    QueryExpander,
    ExpansionStrategy,
    get_query_expander
)

from .bm25_retriever import (
    BM25Retriever,
    BM25Variant
)

from .hybrid_fusion import (
    HybridFusion,
    FusionStrategy
)

from .reranker import (
    Reranker,
    RerankerModel,
    get_reranker
)

from .mmr import (
    MMRSelector,
    MMRConfig
)

from .contextual_compression import (
    ContextualCompressor,
    CompressionStrategy
)

__all__ = [
    # Advanced Retrieval Pipeline
    "AdvancedRetrievalPipeline",
    "PipelineConfig",
    "RetrievalMode",
    "RetrievalResult",
    
    # Query Expansion
    "QueryExpander",
    "ExpansionStrategy",
    "get_query_expander",
    
    # BM25 Retrieval
    "BM25Retriever",
    "BM25Variant",
    
    # Hybrid Fusion
    "HybridFusion",
    "FusionStrategy",
    
    # Reranking
    "Reranker",
    "RerankerModel",
    "get_reranker",
    
    # MMR Diversity
    "MMRSelector",
    "MMRConfig",
    
    # Contextual Compression
    "ContextualCompressor",
    "CompressionStrategy"
]

