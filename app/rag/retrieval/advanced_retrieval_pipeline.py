"""
Advanced Retrieval Pipeline for RAG.

This module orchestrates all advanced retrieval components into a unified
pipeline that significantly improves retrieval quality over basic vector search.

Pipeline Stages:
1. Query Expansion - Expand query with synonyms and related terms
2. Parallel Retrieval - Execute dense (vector) + sparse (BM25) search in parallel
3. Hybrid Fusion - Combine results using RRF or other fusion strategies
4. Reranking - Rerank with cross-encoder for improved precision
5. MMR - Apply diversity to reduce redundancy
6. Contextual Compression - Extract relevant passages from long documents

Features:
- Configurable pipeline stages (enable/disable each component)
- Parallel execution for efficiency
- Comprehensive error handling with fallbacks
- Detailed metrics and logging
- Production-ready with async support

Author: Agentic AI System
Purpose: Orchestrate advanced retrieval for maximum quality
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field
import numpy as np

# Import retrieval components
from app.rag.retrieval.query_expansion import (
    QueryExpander,
    ExpansionStrategy,
    get_query_expander
)
from app.rag.retrieval.bm25_retriever import BM25Retriever, BM25Variant
from app.rag.retrieval.hybrid_fusion import HybridFusion, FusionStrategy
from app.rag.retrieval.reranker import Reranker, RerankerModel, get_reranker
from app.rag.retrieval.mmr import MMRSelector, MMRConfig
from app.rag.retrieval.contextual_compression import (
    ContextualCompressor,
    CompressionStrategy
)

logger = structlog.get_logger(__name__)


class RetrievalMode(str, Enum):
    """Retrieval modes."""
    BASIC = "basic"  # Vector search only
    HYBRID = "hybrid"  # Vector + BM25
    ADVANCED = "advanced"  # Full pipeline with all components


class PipelineConfig(BaseModel):
    """Configuration for advanced retrieval pipeline."""
    # Mode
    mode: RetrievalMode = Field(default=RetrievalMode.ADVANCED)
    
    # Query expansion
    enable_query_expansion: bool = Field(default=True)
    expansion_strategy: ExpansionStrategy = Field(default=ExpansionStrategy.WORDNET)
    
    # BM25
    enable_bm25: bool = Field(default=True)
    bm25_variant: BM25Variant = Field(default=BM25Variant.OKAPI)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Fusion
    fusion_strategy: FusionStrategy = Field(default=FusionStrategy.RRF)
    
    # Reranking
    enable_reranking: bool = Field(default=True)
    reranker_model: RerankerModel = Field(default=RerankerModel.BGE_RERANKER_BASE)
    rerank_top_k: int = Field(default=100, ge=10)
    
    # MMR
    enable_mmr: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    mmr_diversity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Compression
    enable_compression: bool = Field(default=False)
    compression_strategy: CompressionStrategy = Field(default=CompressionStrategy.EXTRACTIVE)
    compression_ratio: float = Field(default=0.5, ge=0.1, le=1.0)
    
    # Retrieval parameters
    initial_top_k: int = Field(default=100, ge=10, description="Initial retrieval count")
    final_top_k: int = Field(default=10, ge=1, description="Final result count")


@dataclass
class RetrievalResult:
    """Final retrieval result."""
    doc_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline details
    original_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    mmr_score: Optional[float] = None
    diversity_score: Optional[float] = None


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution."""
    total_time_ms: float
    query_expansion_time_ms: float
    dense_retrieval_time_ms: float
    sparse_retrieval_time_ms: float
    fusion_time_ms: float
    reranking_time_ms: float
    mmr_time_ms: float
    compression_time_ms: float
    
    expanded_queries: List[str]
    dense_results_count: int
    sparse_results_count: int
    fused_results_count: int
    reranked_results_count: int
    final_results_count: int


class AdvancedRetrievalPipeline:
    """
    Production-grade advanced retrieval pipeline.
    
    Orchestrates all retrieval components to provide state-of-the-art
    retrieval quality for RAG applications.
    
    Features:
    - Configurable pipeline stages
    - Parallel execution
    - Comprehensive error handling
    - Detailed metrics
    - Async support
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize advanced retrieval pipeline."""
        self.config = config or PipelineConfig()
        
        # Components (initialized lazily)
        self._query_expander: Optional[QueryExpander] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._fusion: Optional[HybridFusion] = None
        self._reranker: Optional[Reranker] = None
        self._mmr_selector: Optional[MMRSelector] = None
        self._compressor: Optional[ContextualCompressor] = None
        
        # Locks for lazy initialization
        self._init_lock = asyncio.Lock()
        self._initialized = False
        
        # Metrics
        self._metrics = {
            'total_retrievals': 0,
            'avg_total_time_ms': 0.0,
            'avg_results_count': 0.0
        }
        
        logger.info(
            "AdvancedRetrievalPipeline initialized",
            mode=self.config.mode.value,
            query_expansion=self.config.enable_query_expansion,
            bm25=self.config.enable_bm25,
            reranking=self.config.enable_reranking,
            mmr=self.config.enable_mmr,
            compression=self.config.enable_compression
        )
    
    async def initialize(self) -> None:
        """Initialize pipeline components."""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            try:
                # Initialize query expander
                if self.config.enable_query_expansion:
                    self._query_expander = await get_query_expander()
                
                # Initialize BM25 retriever
                if self.config.enable_bm25:
                    from app.rag.retrieval.bm25_retriever import BM25Config
                    bm25_config = BM25Config(variant=self.config.bm25_variant)
                    self._bm25_retriever = BM25Retriever(bm25_config)
                
                # Initialize fusion
                if self.config.enable_bm25:
                    from app.rag.retrieval.hybrid_fusion import FusionConfig
                    fusion_config = FusionConfig(
                        default_strategy=self.config.fusion_strategy
                    )
                    self._fusion = HybridFusion(fusion_config)
                
                # Initialize reranker
                if self.config.enable_reranking:
                    from app.rag.retrieval.reranker import RerankerConfig
                    reranker_config = RerankerConfig(
                        model_name=self.config.reranker_model,
                        top_k_rerank=self.config.rerank_top_k
                    )
                    self._reranker = await get_reranker(reranker_config)
                
                # Initialize MMR selector
                if self.config.enable_mmr:
                    mmr_config = MMRConfig(
                        lambda_param=self.config.mmr_lambda,
                        diversity_threshold=self.config.mmr_diversity_threshold
                    )
                    self._mmr_selector = MMRSelector(mmr_config)
                
                # Initialize compressor
                if self.config.enable_compression:
                    from app.rag.retrieval.contextual_compression import CompressionConfig
                    compression_config = CompressionConfig(
                        strategy=self.config.compression_strategy,
                        target_ratio=self.config.compression_ratio
                    )
                    self._compressor = ContextualCompressor(compression_config)
                
                self._initialized = True
                logger.info("AdvancedRetrievalPipeline components initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize pipeline: {e}")
                raise
    
    async def retrieve(
        self,
        query: str,
        dense_retriever: Callable[[str, int], Awaitable[List[Dict[str, Any]]]],
        query_embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None
    ) -> tuple[List[RetrievalResult], PipelineMetrics]:
        """
        Execute advanced retrieval pipeline.
        
        Args:
            query: Search query
            dense_retriever: Async function for dense vector search
            query_embedding: Optional pre-computed query embedding
            top_k: Override final_top_k from config
            
        Returns:
            Tuple of (results, metrics)
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        top_k = top_k or self.config.final_top_k
        
        # Initialize timing metrics
        timings = {
            'query_expansion': 0.0,
            'dense_retrieval': 0.0,
            'sparse_retrieval': 0.0,
            'fusion': 0.0,
            'reranking': 0.0,
            'mmr': 0.0,
            'compression': 0.0
        }
        
        expanded_queries = [query]
        
        try:
            # Stage 1: Query Expansion
            if self.config.enable_query_expansion and self._query_expander:
                t0 = time.time()
                expansion_result = await self._query_expander.expand_query(
                    query,
                    self.config.expansion_strategy
                )
                expanded_queries = expansion_result.expanded_queries
                timings['query_expansion'] = (time.time() - t0) * 1000
                logger.debug(f"Query expanded: {len(expanded_queries)} variants")

            # Stage 2: Parallel Retrieval (Dense + Sparse)
            dense_results = []
            sparse_results = []

            # Execute dense retrieval for all expanded queries
            t0 = time.time()
            if len(expanded_queries) > 1:
                # Retrieve for each expanded query and combine
                dense_tasks = [
                    dense_retriever(q, self.config.initial_top_k)
                    for q in expanded_queries
                ]
                all_dense_results = await asyncio.gather(*dense_tasks)

                # Combine and deduplicate
                seen_ids = set()
                for results in all_dense_results:
                    for result in results:
                        doc_id = result.get('doc_id')
                        if doc_id and doc_id not in seen_ids:
                            dense_results.append(result)
                            seen_ids.add(doc_id)
            else:
                # Single query
                dense_results = await dense_retriever(query, self.config.initial_top_k)

            timings['dense_retrieval'] = (time.time() - t0) * 1000
            logger.debug(f"Dense retrieval: {len(dense_results)} results")

            # Execute sparse retrieval (BM25) if enabled
            if self.config.enable_bm25 and self._bm25_retriever:
                t0 = time.time()
                try:
                    # Search with BM25
                    bm25_results = await self._bm25_retriever.search(
                        query,
                        top_k=self.config.initial_top_k
                    )

                    # Convert to standard format
                    sparse_results = [
                        {
                            'doc_id': r.doc_id,
                            'score': r.score,
                            'content': r.content,
                            'metadata': r.metadata,
                            'method': 'bm25'
                        }
                        for r in bm25_results
                    ]

                    timings['sparse_retrieval'] = (time.time() - t0) * 1000
                    logger.debug(f"Sparse retrieval: {len(sparse_results)} results")

                except Exception as e:
                    logger.warning(f"BM25 retrieval failed: {e}")
                    sparse_results = []

            # Stage 3: Hybrid Fusion
            if self.config.enable_bm25 and sparse_results and self._fusion:
                t0 = time.time()

                # Prepare results by method
                from app.rag.retrieval.hybrid_fusion import RetrievalResult as FusionRetrievalResult

                results_by_method = {}

                # Dense results
                if dense_results:
                    results_by_method['dense'] = [
                        FusionRetrievalResult(
                            doc_id=r.get('doc_id'),
                            score=r.get('score', 0.0),
                            rank=idx + 1,
                            method='dense',
                            content=r.get('content', ''),
                            metadata=r.get('metadata', {})
                        )
                        for idx, r in enumerate(dense_results)
                    ]

                # Sparse results
                if sparse_results:
                    results_by_method['sparse'] = [
                        FusionRetrievalResult(
                            doc_id=r.get('doc_id'),
                            score=r.get('score', 0.0),
                            rank=idx + 1,
                            method='sparse',
                            content=r.get('content', ''),
                            metadata=r.get('metadata', {})
                        )
                        for idx, r in enumerate(sparse_results)
                    ]

                # Fuse results
                fused_results = self._fusion.fuse_results(
                    results_by_method,
                    strategy=self.config.fusion_strategy,
                    top_k=self.config.initial_top_k
                )

                # Convert back to standard format
                combined_results = [
                    {
                        'doc_id': r.doc_id,
                        'score': r.fused_score,
                        'content': r.content,
                        'metadata': r.metadata,
                        'dense_score': r.method_scores.get('dense'),
                        'sparse_score': r.method_scores.get('sparse'),
                        'original_rank': r.rank
                    }
                    for r in fused_results
                ]

                timings['fusion'] = (time.time() - t0) * 1000
                logger.debug(f"Fusion: {len(combined_results)} results")
            else:
                # No fusion, use dense results only
                combined_results = [
                    {
                        'doc_id': r.get('doc_id'),
                        'score': r.get('score', 0.0),
                        'content': r.get('content', ''),
                        'metadata': r.get('metadata', {}),
                        'dense_score': r.get('score', 0.0),
                        'sparse_score': None,
                        'original_rank': idx + 1
                    }
                    for idx, r in enumerate(dense_results)
                ]

            # Stage 4: Reranking
            reranked_results = combined_results
            if self.config.enable_reranking and self._reranker:
                t0 = time.time()
                try:
                    # Limit to rerank_top_k
                    to_rerank = combined_results[:self.config.rerank_top_k]

                    rerank_output = await self._reranker.rerank(
                        query,
                        to_rerank,
                        top_k=None  # Don't limit yet
                    )

                    # Convert to standard format
                    reranked_results = [
                        {
                            'doc_id': r.doc_id,
                            'score': r.rerank_score,
                            'content': r.content,
                            'metadata': r.metadata,
                            'dense_score': next(
                                (cr['dense_score'] for cr in combined_results if cr['doc_id'] == r.doc_id),
                                None
                            ),
                            'sparse_score': next(
                                (cr['sparse_score'] for cr in combined_results if cr['doc_id'] == r.doc_id),
                                None
                            ),
                            'rerank_score': r.rerank_score,
                            'original_rank': r.original_rank
                        }
                        for r in rerank_output
                    ]

                    timings['reranking'] = (time.time() - t0) * 1000
                    logger.debug(f"Reranking: {len(reranked_results)} results")

                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
                    reranked_results = combined_results

            # Stage 5: MMR (Diversity)
            final_results = reranked_results
            if self.config.enable_mmr and self._mmr_selector:
                t0 = time.time()
                try:
                    # Prepare results with embeddings if available
                    mmr_input = []
                    for r in reranked_results:
                        # Try to get embedding from metadata
                        embedding = r.get('metadata', {}).get('embedding')
                        mmr_input.append({
                            'doc_id': r['doc_id'],
                            'score': r['score'],
                            'content': r['content'],
                            'embedding': embedding,
                            'metadata': r['metadata']
                        })

                    # Apply MMR
                    mmr_output = self._mmr_selector.select(
                        query_embedding=query_embedding,
                        results=mmr_input,
                        top_k=top_k
                    )

                    # Convert to standard format
                    final_results = [
                        {
                            'doc_id': r.doc_id,
                            'score': r.mmr_score,
                            'content': r.content,
                            'metadata': r.metadata,
                            'dense_score': next(
                                (rr['dense_score'] for rr in reranked_results if rr['doc_id'] == r.doc_id),
                                None
                            ),
                            'sparse_score': next(
                                (rr['sparse_score'] for rr in reranked_results if rr['doc_id'] == r.doc_id),
                                None
                            ),
                            'rerank_score': next(
                                (rr.get('rerank_score') for rr in reranked_results if rr['doc_id'] == r.doc_id),
                                None
                            ),
                            'mmr_score': r.mmr_score,
                            'diversity_score': r.diversity_score,
                            'original_rank': r.original_rank
                        }
                        for r in mmr_output
                    ]

                    timings['mmr'] = (time.time() - t0) * 1000
                    logger.debug(f"MMR: {len(final_results)} results")

                except Exception as e:
                    logger.warning(f"MMR failed: {e}")
                    final_results = reranked_results[:top_k]
            else:
                # No MMR, just limit to top_k
                final_results = reranked_results[:top_k]

            # Stage 6: Contextual Compression (optional)
            if self.config.enable_compression and self._compressor:
                t0 = time.time()
                try:
                    compressed_output = self._compressor.compress(
                        query,
                        final_results,
                        query_embedding
                    )

                    # Update content with compressed version
                    for i, compressed in enumerate(compressed_output):
                        if i < len(final_results):
                            final_results[i]['content'] = compressed.compressed_content
                            final_results[i]['metadata']['original_length'] = compressed.original_length
                            final_results[i]['metadata']['compressed_length'] = compressed.compressed_length
                            final_results[i]['metadata']['compression_ratio'] = compressed.compression_ratio

                    timings['compression'] = (time.time() - t0) * 1000
                    logger.debug(f"Compression: {len(compressed_output)} documents")

                except Exception as e:
                    logger.warning(f"Compression failed: {e}")

            # Convert to RetrievalResult objects
            results = []
            for idx, r in enumerate(final_results, 1):
                results.append(RetrievalResult(
                    doc_id=r['doc_id'],
                    content=r['content'],
                    score=r['score'],
                    rank=idx,
                    metadata=r.get('metadata', {}),
                    original_rank=r.get('original_rank'),
                    dense_score=r.get('dense_score'),
                    sparse_score=r.get('sparse_score'),
                    rerank_score=r.get('rerank_score'),
                    mmr_score=r.get('mmr_score'),
                    diversity_score=r.get('diversity_score')
                ))

            # Create metrics
            total_time_ms = (time.time() - start_time) * 1000
            metrics = PipelineMetrics(
                total_time_ms=total_time_ms,
                query_expansion_time_ms=timings['query_expansion'],
                dense_retrieval_time_ms=timings['dense_retrieval'],
                sparse_retrieval_time_ms=timings['sparse_retrieval'],
                fusion_time_ms=timings['fusion'],
                reranking_time_ms=timings['reranking'],
                mmr_time_ms=timings['mmr'],
                compression_time_ms=timings['compression'],
                expanded_queries=expanded_queries,
                dense_results_count=len(dense_results),
                sparse_results_count=len(sparse_results),
                fused_results_count=len(combined_results) if self.config.enable_bm25 else 0,
                reranked_results_count=len(reranked_results),
                final_results_count=len(results)
            )

            # Update global metrics
            self._metrics['total_retrievals'] += 1
            total_retrievals = self._metrics['total_retrievals']
            current_avg_time = self._metrics['avg_total_time_ms']
            self._metrics['avg_total_time_ms'] = (
                (current_avg_time * (total_retrievals - 1) + total_time_ms) / total_retrievals
            )
            current_avg_count = self._metrics['avg_results_count']
            self._metrics['avg_results_count'] = (
                (current_avg_count * (total_retrievals - 1) + len(results)) / total_retrievals
            )

            logger.info(
                f"Advanced retrieval completed",
                query=query[:50],
                results_count=len(results),
                total_time_ms=total_time_ms
            )

            return results, metrics

        except Exception as e:
            logger.error(f"Advanced retrieval failed: {e}", query=query[:50])
            raise

    async def add_documents_to_bm25(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Add documents to BM25 index.

        Args:
            documents: List of documents with 'doc_id', 'content', 'metadata'

        Returns:
            Number of documents added
        """
        if not self.config.enable_bm25 or not self._bm25_retriever:
            logger.warning("BM25 not enabled, cannot add documents")
            return 0

        return await self._bm25_retriever.add_documents(documents)

    async def delete_documents_from_bm25(
        self,
        doc_ids: List[str]
    ) -> int:
        """
        Delete documents from BM25 index.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        if not self.config.enable_bm25 or not self._bm25_retriever:
            logger.warning("BM25 not enabled, cannot delete documents")
            return 0

        return await self._bm25_retriever.delete_documents(doc_ids)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        metrics = {
            'total_retrievals': self._metrics['total_retrievals'],
            'avg_total_time_ms': self._metrics['avg_total_time_ms'],
            'avg_results_count': self._metrics['avg_results_count']
        }

        # Add component metrics
        if self._query_expander:
            metrics['query_expander'] = self._query_expander.get_metrics()

        if self._bm25_retriever:
            metrics['bm25_retriever'] = self._bm25_retriever.get_metrics()

        if self._fusion:
            metrics['fusion'] = self._fusion.get_metrics()

        if self._reranker:
            metrics['reranker'] = self._reranker.get_metrics()

        if self._mmr_selector:
            metrics['mmr_selector'] = self._mmr_selector.get_metrics()

        if self._compressor:
            metrics['compressor'] = self._compressor.get_metrics()

        return metrics

    async def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self._reranker:
            await self._reranker.cleanup()

        if self._query_expander:
            self._query_expander.clear_cache()

        if self._reranker:
            self._reranker.clear_cache()

        logger.info("AdvancedRetrievalPipeline cleaned up")

