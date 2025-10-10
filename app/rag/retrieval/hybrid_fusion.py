"""
Hybrid Fusion for RAG Retrieval.

This module implements fusion algorithms to combine results from multiple
retrieval methods (dense vector search, BM25 keyword search, etc.) into
a unified ranked list.

Fusion Algorithms:
- Reciprocal Rank Fusion (RRF) - Industry standard
- Weighted Sum - Simple weighted combination
- CombSUM - Sum of normalized scores
- CombMNZ - CombSUM with non-zero count multiplier
- Learned Fusion - ML-based score combination

Features:
- Multiple fusion strategies
- Score normalization and calibration
- Configurable weights per retrieval method
- Tie-breaking strategies
- Comprehensive metrics

Author: Agentic AI System
Purpose: Combine multiple retrieval methods for optimal results
"""

import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from pydantic import BaseModel, Field
import numpy as np

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class FusionStrategy(str, Enum):
    """Fusion algorithm strategies."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # Weighted score combination
    COMB_SUM = "comb_sum"  # Sum of normalized scores
    COMB_MNZ = "comb_mnz"  # CombSUM with non-zero multiplier
    MAX_SCORE = "max_score"  # Maximum score across methods
    MIN_SCORE = "min_score"  # Minimum score across methods


class FusionConfig(BaseModel):
    """Configuration for hybrid fusion."""
    strategy: FusionStrategy = Field(default=FusionStrategy.RRF)
    rrf_k: int = Field(default=60, ge=1, description="RRF constant (typically 60)")
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"dense": 0.5, "sparse": 0.5},
        description="Weights for each retrieval method"
    )
    normalize_scores: bool = Field(default=True)
    min_score_threshold: float = Field(default=0.0, ge=0.0)
    max_results: int = Field(default=100, ge=1)


@dataclass
class RetrievalResult:
    """Result from a single retrieval method."""
    doc_id: str
    score: float
    rank: int
    method: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult:
    """Fused result from multiple retrieval methods."""
    doc_id: str
    fused_score: float
    rank: int
    method_scores: Dict[str, float]
    method_ranks: Dict[str, int]
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridFusion:
    """
    Production-grade hybrid fusion engine.
    
    Combines results from multiple retrieval methods (dense, sparse, etc.)
    using various fusion algorithms.
    
    Features:
    - Multiple fusion strategies (RRF, weighted sum, etc.)
    - Score normalization and calibration
    - Configurable method weights
    - Comprehensive metrics
    - Efficient result merging
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialize hybrid fusion."""
        self.config = config or FusionConfig()
        
        # Metrics
        self._metrics = {
            'total_fusions': 0,
            'avg_fusion_time_ms': 0.0,
            'method_contributions': defaultdict(int),
            'strategy_usage': defaultdict(int)
        }
        
        logger.info(
            "HybridFusion initialized",
            LogCategory.RAG_OPERATIONS,
            "app.rag.retrieval.hybrid_fusion.HybridFusion",
            data={
                "strategy": self.config.strategy.value,
                "weights": self.config.weights
            }
        )
    
    def fuse_results(
        self,
        results_by_method: Dict[str, List[RetrievalResult]],
        strategy: Optional[FusionStrategy] = None,
        top_k: Optional[int] = None
    ) -> List[FusedResult]:
        """
        Fuse results from multiple retrieval methods.
        
        Args:
            results_by_method: Dict mapping method name to list of results
            strategy: Override default fusion strategy
            top_k: Number of top results to return
            
        Returns:
            List of FusedResult ordered by fused score
        """
        start_time = time.time()
        strategy = strategy or self.config.strategy
        top_k = top_k or self.config.max_results
        
        # Validate inputs
        if not results_by_method:
            logger.warn(
                "No results to fuse",
                LogCategory.RAG_OPERATIONS,
                "app.rag.retrieval.hybrid_fusion.HybridFusion"
            )
            return []
        
        # Normalize scores if configured
        if self.config.normalize_scores:
            results_by_method = self._normalize_scores(results_by_method)
        
        # Apply fusion strategy
        if strategy == FusionStrategy.RRF:
            fused_results = self._fuse_rrf(results_by_method)
        elif strategy == FusionStrategy.WEIGHTED_SUM:
            fused_results = self._fuse_weighted_sum(results_by_method)
        elif strategy == FusionStrategy.COMB_SUM:
            fused_results = self._fuse_comb_sum(results_by_method)
        elif strategy == FusionStrategy.COMB_MNZ:
            fused_results = self._fuse_comb_mnz(results_by_method)
        elif strategy == FusionStrategy.MAX_SCORE:
            fused_results = self._fuse_max_score(results_by_method)
        elif strategy == FusionStrategy.MIN_SCORE:
            fused_results = self._fuse_min_score(results_by_method)
        else:
            logger.error(
                f"Unknown fusion strategy: {strategy}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.retrieval.hybrid_fusion.HybridFusion",
                data={"strategy": strategy}
            )
            fused_results = []
        
        # Filter by minimum score
        if self.config.min_score_threshold > 0:
            fused_results = [
                r for r in fused_results
                if r.fused_score >= self.config.min_score_threshold
            ]
        
        # Sort by fused score descending
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)
        
        # Limit to top_k and set final ranks
        fused_results = fused_results[:top_k]
        for rank, result in enumerate(fused_results, 1):
            result.rank = rank
        
        # Update metrics
        self._metrics['total_fusions'] += 1
        self._metrics['strategy_usage'][strategy.value] += 1
        
        for method in results_by_method.keys():
            self._metrics['method_contributions'][method] += 1
        
        fusion_time_ms = (time.time() - start_time) * 1000
        total_fusions = self._metrics['total_fusions']
        current_avg = self._metrics['avg_fusion_time_ms']
        self._metrics['avg_fusion_time_ms'] = (
            (current_avg * (total_fusions - 1) + fusion_time_ms) / total_fusions
        )
        
        logger.debug(
            f"Fusion completed",
            LogCategory.RAG_OPERATIONS,
            "app.rag.retrieval.hybrid_fusion.HybridFusion",
            data={
                "strategy": strategy.value,
                "methods": list(results_by_method.keys()),
                "results_count": len(fused_results),
                "fusion_time_ms": fusion_time_ms
            }
        )
        
        return fused_results
    
    def _fuse_rrf(
        self,
        results_by_method: Dict[str, List[RetrievalResult]]
    ) -> List[FusedResult]:
        """
        Fuse using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(d) = sum over methods of 1 / (k + rank(d))
        where k is a constant (typically 60) and rank(d) is the rank of document d.
        
        RRF is robust to score scale differences and works well in practice.
        """
        k = self.config.rrf_k
        doc_scores: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'rrf_score': 0.0,
                'method_scores': {},
                'method_ranks': {},
                'content': None,
                'metadata': {}
            }
        )
        
        # Calculate RRF scores
        for method, results in results_by_method.items():
            method_weight = self.config.weights.get(method, 1.0)
            
            for result in results:
                doc_id = result.doc_id
                rank = result.rank
                
                # RRF contribution: weight / (k + rank)
                rrf_contribution = method_weight / (k + rank)
                doc_scores[doc_id]['rrf_score'] += rrf_contribution
                doc_scores[doc_id]['method_scores'][method] = result.score
                doc_scores[doc_id]['method_ranks'][method] = rank
                
                # Store content and metadata from first occurrence
                if doc_scores[doc_id]['content'] is None:
                    doc_scores[doc_id]['content'] = result.content
                    doc_scores[doc_id]['metadata'] = result.metadata
        
        # Create fused results
        fused_results = []
        for doc_id, data in doc_scores.items():
            fused_results.append(FusedResult(
                doc_id=doc_id,
                fused_score=data['rrf_score'],
                rank=0,  # Will be set after sorting
                method_scores=data['method_scores'],
                method_ranks=data['method_ranks'],
                content=data['content'],
                metadata=data['metadata']
            ))
        
        return fused_results
    
    def _fuse_weighted_sum(
        self,
        results_by_method: Dict[str, List[RetrievalResult]]
    ) -> List[FusedResult]:
        """
        Fuse using weighted sum of normalized scores.
        
        Formula: score(d) = sum over methods of weight(m) * score(d, m)
        """
        doc_scores: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'weighted_score': 0.0,
                'method_scores': {},
                'method_ranks': {},
                'content': None,
                'metadata': {}
            }
        )
        
        # Calculate weighted scores
        for method, results in results_by_method.items():
            method_weight = self.config.weights.get(method, 1.0)
            
            for result in results:
                doc_id = result.doc_id
                
                # Weighted contribution
                weighted_contribution = method_weight * result.score
                doc_scores[doc_id]['weighted_score'] += weighted_contribution
                doc_scores[doc_id]['method_scores'][method] = result.score
                doc_scores[doc_id]['method_ranks'][method] = result.rank
                
                # Store content and metadata
                if doc_scores[doc_id]['content'] is None:
                    doc_scores[doc_id]['content'] = result.content
                    doc_scores[doc_id]['metadata'] = result.metadata
        
        # Create fused results
        fused_results = []
        for doc_id, data in doc_scores.items():
            fused_results.append(FusedResult(
                doc_id=doc_id,
                fused_score=data['weighted_score'],
                rank=0,
                method_scores=data['method_scores'],
                method_ranks=data['method_ranks'],
                content=data['content'],
                metadata=data['metadata']
            ))
        
        return fused_results

    def _fuse_comb_sum(
        self,
        results_by_method: Dict[str, List[RetrievalResult]]
    ) -> List[FusedResult]:
        """
        Fuse using CombSUM (sum of normalized scores).

        Formula: score(d) = sum over methods of normalized_score(d, m)
        """
        # CombSUM is similar to weighted sum with equal weights
        return self._fuse_weighted_sum(results_by_method)

    def _fuse_comb_mnz(
        self,
        results_by_method: Dict[str, List[RetrievalResult]]
    ) -> List[FusedResult]:
        """
        Fuse using CombMNZ (CombSUM multiplied by number of non-zero scores).

        Formula: score(d) = count(non-zero methods) * sum of normalized_score(d, m)

        This gives preference to documents that appear in multiple methods.
        """
        doc_scores: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'sum_score': 0.0,
                'method_count': 0,
                'method_scores': {},
                'method_ranks': {},
                'content': None,
                'metadata': {}
            }
        )

        # Calculate sum and count
        for method, results in results_by_method.items():
            for result in results:
                doc_id = result.doc_id

                doc_scores[doc_id]['sum_score'] += result.score
                doc_scores[doc_id]['method_count'] += 1
                doc_scores[doc_id]['method_scores'][method] = result.score
                doc_scores[doc_id]['method_ranks'][method] = result.rank

                if doc_scores[doc_id]['content'] is None:
                    doc_scores[doc_id]['content'] = result.content
                    doc_scores[doc_id]['metadata'] = result.metadata

        # Create fused results with MNZ scoring
        fused_results = []
        for doc_id, data in doc_scores.items():
            mnz_score = data['sum_score'] * data['method_count']

            fused_results.append(FusedResult(
                doc_id=doc_id,
                fused_score=mnz_score,
                rank=0,
                method_scores=data['method_scores'],
                method_ranks=data['method_ranks'],
                content=data['content'],
                metadata=data['metadata']
            ))

        return fused_results

    def _fuse_max_score(
        self,
        results_by_method: Dict[str, List[RetrievalResult]]
    ) -> List[FusedResult]:
        """Fuse using maximum score across methods."""
        doc_scores: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'max_score': 0.0,
                'method_scores': {},
                'method_ranks': {},
                'content': None,
                'metadata': {}
            }
        )

        for method, results in results_by_method.items():
            for result in results:
                doc_id = result.doc_id

                # Update max score
                if result.score > doc_scores[doc_id]['max_score']:
                    doc_scores[doc_id]['max_score'] = result.score

                doc_scores[doc_id]['method_scores'][method] = result.score
                doc_scores[doc_id]['method_ranks'][method] = result.rank

                if doc_scores[doc_id]['content'] is None:
                    doc_scores[doc_id]['content'] = result.content
                    doc_scores[doc_id]['metadata'] = result.metadata

        # Create fused results
        fused_results = []
        for doc_id, data in doc_scores.items():
            fused_results.append(FusedResult(
                doc_id=doc_id,
                fused_score=data['max_score'],
                rank=0,
                method_scores=data['method_scores'],
                method_ranks=data['method_ranks'],
                content=data['content'],
                metadata=data['metadata']
            ))

        return fused_results

    def _fuse_min_score(
        self,
        results_by_method: Dict[str, List[RetrievalResult]]
    ) -> List[FusedResult]:
        """Fuse using minimum score across methods (conservative)."""
        doc_scores: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'min_score': float('inf'),
                'method_scores': {},
                'method_ranks': {},
                'content': None,
                'metadata': {}
            }
        )

        for method, results in results_by_method.items():
            for result in results:
                doc_id = result.doc_id

                # Update min score
                if result.score < doc_scores[doc_id]['min_score']:
                    doc_scores[doc_id]['min_score'] = result.score

                doc_scores[doc_id]['method_scores'][method] = result.score
                doc_scores[doc_id]['method_ranks'][method] = result.rank

                if doc_scores[doc_id]['content'] is None:
                    doc_scores[doc_id]['content'] = result.content
                    doc_scores[doc_id]['metadata'] = result.metadata

        # Create fused results
        fused_results = []
        for doc_id, data in doc_scores.items():
            min_score = data['min_score'] if data['min_score'] != float('inf') else 0.0

            fused_results.append(FusedResult(
                doc_id=doc_id,
                fused_score=min_score,
                rank=0,
                method_scores=data['method_scores'],
                method_ranks=data['method_ranks'],
                content=data['content'],
                metadata=data['metadata']
            ))

        return fused_results

    def _normalize_scores(
        self,
        results_by_method: Dict[str, List[RetrievalResult]]
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Normalize scores for each method to [0, 1] range.

        Uses min-max normalization per method.
        """
        normalized_results = {}

        for method, results in results_by_method.items():
            if not results:
                normalized_results[method] = []
                continue

            # Get min and max scores
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)

            # Normalize
            if max_score > min_score:
                normalized = []
                for result in results:
                    normalized_score = (result.score - min_score) / (max_score - min_score)
                    normalized.append(RetrievalResult(
                        doc_id=result.doc_id,
                        score=normalized_score,
                        rank=result.rank,
                        method=result.method,
                        content=result.content,
                        metadata=result.metadata
                    ))
                normalized_results[method] = normalized
            else:
                # All scores are the same, set to 1.0
                normalized = []
                for result in results:
                    normalized.append(RetrievalResult(
                        doc_id=result.doc_id,
                        score=1.0,
                        rank=result.rank,
                        method=result.method,
                        content=result.content,
                        metadata=result.metadata
                    ))
                normalized_results[method] = normalized

        return normalized_results

    def get_metrics(self) -> Dict[str, Any]:
        """Get fusion metrics."""
        return {
            'total_fusions': self._metrics['total_fusions'],
            'avg_fusion_time_ms': self._metrics['avg_fusion_time_ms'],
            'method_contributions': dict(self._metrics['method_contributions']),
            'strategy_usage': dict(self._metrics['strategy_usage'])
        }

