"""
Cross-Encoder Reranking for RAG Retrieval.

This module implements production-grade reranking using cross-encoder models
to improve retrieval quality by reordering initial search results.

Cross-encoders jointly encode query and document pairs, providing more accurate
relevance scores than bi-encoders (which encode separately).

Reranking Pipeline:
1. Initial retrieval (dense + sparse) returns top-N candidates (e.g., 100)
2. Cross-encoder scores each (query, document) pair
3. Results are reordered by cross-encoder scores
4. Return top-K final results (e.g., 10)

Features:
- Multiple reranker models (BGE, MiniLM, etc.)
- Batch processing for efficiency
- GPU acceleration support
- Result caching
- Fallback to original scores on failure
- Comprehensive metrics

Author: Agentic AI System
Purpose: Improve retrieval precision through reranking
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from pydantic import BaseModel, Field
import numpy as np

# Sentence transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

# Torch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class RerankerModel(str, Enum):
    """Supported reranker models."""
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"
    BGE_RERANKER_V2_M3 = "BAAI/bge-reranker-v2-m3"
    MINILM_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MINILM_CROSS_ENCODER_V2 = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class RerankerConfig(BaseModel):
    """Configuration for reranker."""
    model_name: RerankerModel = Field(default=RerankerModel.BGE_RERANKER_BASE)
    batch_size: int = Field(default=32, ge=1, le=256)
    max_length: int = Field(default=512, ge=128, le=2048)
    device: str = Field(default="auto", description="Device: auto, cpu, cuda")
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, ge=60)
    fallback_to_original: bool = Field(default=True, description="Fallback to original scores on error")
    top_k_rerank: Optional[int] = Field(default=None, description="Only rerank top K results")


@dataclass
class RerankResult:
    """Reranked result."""
    doc_id: str
    rerank_score: float
    original_score: float
    rank: int
    original_rank: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry for reranking scores."""
    scores: List[float]
    timestamp: float
    access_count: int = 0


class Reranker:
    """
    Production-grade cross-encoder reranker.
    
    Uses cross-encoder models to rerank initial retrieval results for
    improved precision.
    
    Features:
    - Multiple cross-encoder models
    - Batch processing for efficiency
    - GPU acceleration
    - Result caching
    - Graceful fallback on errors
    - Comprehensive metrics
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize reranker."""
        self.config = config or RerankerConfig()
        
        # Model
        self._model: Optional[Any] = None
        self._model_lock = asyncio.Lock()
        self._initialized = False
        
        # Device
        if self.config.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        # Cache
        self._cache: Dict[str, CacheEntry] = {}
        
        # Metrics
        self._metrics = {
            'total_reranks': 0,
            'total_documents_reranked': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_rerank_time_ms': 0.0,
            'errors': 0,
            'fallbacks': 0
        }
        
        logger.info(
            "Reranker initialized",
            LogCategory.RAG_OPERATIONS,
            "app.rag.retrieval.reranker.Reranker",
            data={
                "model": self.config.model_name.value,
                "device": self.device,
                "caching": self.config.enable_caching
            }
        )
    
    async def initialize(self) -> None:
        """Initialize reranker model."""
        if self._initialized:
            return

        async with self._model_lock:
            if self._initialized:
                return

            if not CROSS_ENCODER_AVAILABLE:
                logger.error(
                    "sentence-transformers not available for reranking",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.retrieval.reranker.Reranker"
                )
                raise ImportError("sentence-transformers required for reranking")

            try:
                # Load model in thread pool
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    self._load_model
                )

                self._initialized = True
                logger.info(
                    f"Reranker model loaded: {self.config.model_name.value}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.retrieval.reranker.Reranker",
                    data={"model": self.config.model_name.value}
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize reranker: {e}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.retrieval.reranker.Reranker",
                    error=e
                )
                raise

    def _load_model(self) -> Any:
        """
        Load cross-encoder model (runs in thread pool).

        Checks centralized storage first, then falls back to HuggingFace download.
        """
        try:
            # Try to use centralized model storage
            from app.rag.core.embedding_model_manager import embedding_model_manager
            from app.rag.config.required_models import get_model_by_id

            # Get model spec
            model_id = self.config.model_name.value
            model_spec = get_model_by_id(model_id)

            if model_spec:
                # Check if model exists in centralized storage
                model_info = embedding_model_manager.get_model_info(model_spec.local_name)

                if model_info and model_info.is_downloaded:
                    logger.info(
                        f"Loading reranker from centralized storage: {model_spec.local_name}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.retrieval.reranker.Reranker",
                        data={
                            "model": model_spec.local_name,
                            "path": str(model_info.local_path)
                        }
                    )
                    return CrossEncoder(
                        model_info.local_path,
                        max_length=self.config.max_length,
                        device=self.device
                    )

            # Fallback: Load from HuggingFace (will download if needed)
            logger.warn(
                f"Reranker model not in centralized storage, downloading from HuggingFace: {model_id}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.retrieval.reranker.Reranker",
                data={"model_id": model_id}
            )
            return CrossEncoder(
                model_id,
                max_length=self.config.max_length,
                device=self.device
            )

        except ImportError:
            # Model manager not available, use direct loading
            logger.warn(
                "Model manager not available, loading reranker directly from HuggingFace",
                LogCategory.RAG_OPERATIONS,
                "app.rag.retrieval.reranker.Reranker"
            )
            return CrossEncoder(
                self.config.model_name.value,
                max_length=self.config.max_length,
                device=self.device
            )
    
    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Search query
            results: List of results with 'doc_id', 'score', 'content', 'metadata'
            top_k: Number of top results to return after reranking
            
        Returns:
            List of RerankResult ordered by rerank score
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        if not results:
            logger.warn(
                "No results to rerank",
                LogCategory.RAG_OPERATIONS,
                "app.rag.retrieval.reranker.Reranker"
            )
            return []
        
        # Limit reranking to top K if configured
        if self.config.top_k_rerank:
            results = results[:self.config.top_k_rerank]
        
        try:
            # Prepare query-document pairs
            pairs = []
            doc_ids = []
            original_scores = []
            original_ranks = []
            contents = []
            metadatas = []
            
            for idx, result in enumerate(results, 1):
                doc_id = result.get('doc_id')
                content = result.get('content', '')
                score = result.get('score', 0.0)
                metadata = result.get('metadata', {})
                
                if not doc_id or not content:
                    logger.warn(
                        "Skipping result with missing doc_id or content",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.retrieval.reranker.Reranker"
                    )
                    continue
                
                pairs.append([query, content])
                doc_ids.append(doc_id)
                original_scores.append(score)
                original_ranks.append(idx)
                contents.append(content)
                metadatas.append(metadata)
            
            if not pairs:
                logger.warn(
                    "No valid pairs to rerank",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.retrieval.reranker.Reranker"
                )
                return []
            
            # Check cache
            cache_key = self._get_cache_key(query, doc_ids)
            if self.config.enable_caching and cache_key in self._cache:
                entry = self._cache[cache_key]
                if time.time() - entry.timestamp < self.config.cache_ttl:
                    entry.access_count += 1
                    self._metrics['cache_hits'] += 1
                    rerank_scores = entry.scores
                    logger.debug(
                        f"Rerank cache hit for query: {query[:50]}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.retrieval.reranker.Reranker",
                        data={"query_preview": query[:50]}
                    )
                else:
                    # Cache expired
                    del self._cache[cache_key]
                    rerank_scores = await self._compute_scores(pairs)
                    self._metrics['cache_misses'] += 1
            else:
                # Compute rerank scores
                rerank_scores = await self._compute_scores(pairs)
                self._metrics['cache_misses'] += 1
                
                # Cache scores
                if self.config.enable_caching:
                    self._cache[cache_key] = CacheEntry(
                        scores=rerank_scores,
                        timestamp=time.time()
                    )
            
            # Create reranked results
            reranked_results = []
            for i, (doc_id, rerank_score, orig_score, orig_rank, content, metadata) in enumerate(
                zip(doc_ids, rerank_scores, original_scores, original_ranks, contents, metadatas)
            ):
                reranked_results.append(RerankResult(
                    doc_id=doc_id,
                    rerank_score=float(rerank_score),
                    original_score=orig_score,
                    rank=0,  # Will be set after sorting
                    original_rank=orig_rank,
                    content=content,
                    metadata=metadata
                ))
            
            # Sort by rerank score descending
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Limit to top_k and set final ranks
            if top_k:
                reranked_results = reranked_results[:top_k]
            
            for rank, result in enumerate(reranked_results, 1):
                result.rank = rank
            
            # Update metrics
            self._metrics['total_reranks'] += 1
            self._metrics['total_documents_reranked'] += len(pairs)
            
            rerank_time_ms = (time.time() - start_time) * 1000
            total_reranks = self._metrics['total_reranks']
            current_avg = self._metrics['avg_rerank_time_ms']
            self._metrics['avg_rerank_time_ms'] = (
                (current_avg * (total_reranks - 1) + rerank_time_ms) / total_reranks
            )
            
            logger.debug(
                f"Reranking completed",
                LogCategory.RAG_OPERATIONS,
                "app.rag.retrieval.reranker.Reranker",
                data={
                    "query_preview": query[:50],
                    "results_count": len(reranked_results),
                    "rerank_time_ms": rerank_time_ms
                }
            )

            return reranked_results

        except Exception as e:
            self._metrics['errors'] += 1
            logger.error(
                f"Reranking failed: {e}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.retrieval.reranker.Reranker",
                error=e,
                data={"query_preview": query[:50]}
            )
            
            # Fallback to original scores if configured
            if self.config.fallback_to_original:
                self._metrics['fallbacks'] += 1
                return self._fallback_to_original(results, top_k)
            else:
                raise

    async def _compute_scores(self, pairs: List[List[str]]) -> List[float]:
        """
        Compute reranking scores for query-document pairs.

        Args:
            pairs: List of [query, document] pairs

        Returns:
            List of reranking scores
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self._model.predict,
            pairs,
            self.config.batch_size
        )

        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    def _fallback_to_original(
        self,
        results: List[Dict[str, Any]],
        top_k: Optional[int]
    ) -> List[RerankResult]:
        """
        Fallback to original scores when reranking fails.

        Args:
            results: Original results
            top_k: Number of top results to return

        Returns:
            List of RerankResult using original scores
        """
        logger.warn(
            "Falling back to original scores",
            LogCategory.RAG_OPERATIONS,
            "app.rag.retrieval.reranker.Reranker"
        )

        reranked_results = []
        for idx, result in enumerate(results, 1):
            doc_id = result.get('doc_id')
            score = result.get('score', 0.0)
            content = result.get('content', '')
            metadata = result.get('metadata', {})

            if not doc_id:
                continue

            reranked_results.append(RerankResult(
                doc_id=doc_id,
                rerank_score=score,  # Use original score
                original_score=score,
                rank=idx,
                original_rank=idx,
                content=content,
                metadata=metadata
            ))

        # Limit to top_k
        if top_k:
            reranked_results = reranked_results[:top_k]

        return reranked_results

    def _get_cache_key(self, query: str, doc_ids: List[str]) -> str:
        """Generate cache key for query and document IDs."""
        # Sort doc_ids for consistent cache keys
        sorted_ids = sorted(doc_ids)
        key_str = f"{query}:{'|'.join(sorted_ids)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_metrics(self) -> Dict[str, Any]:
        """Get reranker metrics."""
        total_requests = self._metrics['cache_hits'] + self._metrics['cache_misses']
        cache_hit_rate = (
            self._metrics['cache_hits'] / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            'total_reranks': self._metrics['total_reranks'],
            'total_documents_reranked': self._metrics['total_documents_reranked'],
            'cache_hits': self._metrics['cache_hits'],
            'cache_misses': self._metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'avg_rerank_time_ms': self._metrics['avg_rerank_time_ms'],
            'errors': self._metrics['errors'],
            'fallbacks': self._metrics['fallbacks'],
            'cache_size': len(self._cache)
        }

    def clear_cache(self) -> None:
        """Clear reranking cache."""
        self._cache.clear()
        logger.info(
            "Reranker cache cleared",
            LogCategory.RAG_OPERATIONS,
            "app.rag.retrieval.reranker.Reranker"
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._cache.clear()
        self._model = None
        self._initialized = False
        logger.info(
            "Reranker cleaned up",
            LogCategory.RAG_OPERATIONS,
            "app.rag.retrieval.reranker.Reranker"
        )


# Global reranker instance
_global_reranker: Optional[Reranker] = None
_reranker_lock = asyncio.Lock()


async def get_reranker(config: Optional[RerankerConfig] = None) -> Reranker:
    """
    Get or create global reranker instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        Reranker instance
    """
    global _global_reranker

    if _global_reranker is None:
        async with _reranker_lock:
            if _global_reranker is None:
                _global_reranker = Reranker(config)
                await _global_reranker.initialize()

    return _global_reranker

