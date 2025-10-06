"""
Maximal Marginal Relevance (MMR) for Result Diversity.

This module implements MMR algorithm to diversify search results by balancing
relevance and diversity. MMR prevents redundant results by selecting documents
that are relevant to the query but dissimilar to already selected documents.

MMR Formula:
MMR = argmax[D_i in R \ S] [λ * Sim1(D_i, Q) - (1-λ) * max[D_j in S] Sim2(D_i, D_j)]

Where:
- R: Set of candidate documents
- S: Set of already selected documents
- Q: Query
- λ: Trade-off parameter (0 = max diversity, 1 = max relevance)
- Sim1: Similarity between document and query
- Sim2: Similarity between documents

Features:
- Configurable relevance-diversity trade-off
- Multiple similarity metrics (cosine, euclidean, dot product)
- Efficient incremental selection
- Embedding-based and text-based similarity
- Comprehensive metrics

Author: Agentic AI System
Purpose: Diversify retrieval results to reduce redundancy
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field
import numpy as np

logger = structlog.get_logger(__name__)


class SimilarityMetric(str, Enum):
    """Similarity metrics for MMR."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    JACCARD = "jaccard"


class MMRConfig(BaseModel):
    """Configuration for MMR."""
    lambda_param: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Trade-off: 0=max diversity, 1=max relevance"
    )
    similarity_metric: SimilarityMetric = Field(default=SimilarityMetric.COSINE)
    diversity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum diversity score (1 - max similarity)"
    )
    use_embeddings: bool = Field(
        default=True,
        description="Use embeddings for similarity (vs text-based)"
    )


@dataclass
class MMRResult:
    """MMR-selected result."""
    doc_id: str
    relevance_score: float
    diversity_score: float
    mmr_score: float
    rank: int
    original_rank: int
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MMRSelector:
    """
    Production-grade Maximal Marginal Relevance selector.
    
    Implements MMR algorithm to select diverse results that balance
    relevance to query and dissimilarity to already selected documents.
    
    Features:
    - Configurable relevance-diversity trade-off
    - Multiple similarity metrics
    - Efficient incremental selection
    - Embedding-based similarity
    - Comprehensive metrics
    """
    
    def __init__(self, config: Optional[MMRConfig] = None):
        """Initialize MMR selector."""
        self.config = config or MMRConfig()
        
        # Metrics
        self._metrics = {
            'total_selections': 0,
            'total_documents_processed': 0,
            'avg_selection_time_ms': 0.0,
            'avg_diversity_score': 0.0
        }
        
        logger.info(
            "MMRSelector initialized",
            lambda_param=self.config.lambda_param,
            similarity_metric=self.config.similarity_metric.value
        )
    
    def select(
        self,
        query_embedding: Optional[np.ndarray],
        results: List[Dict[str, Any]],
        top_k: int,
        lambda_param: Optional[float] = None
    ) -> List[MMRResult]:
        """
        Select diverse results using MMR.
        
        Args:
            query_embedding: Query embedding vector (required if use_embeddings=True)
            results: List of results with 'doc_id', 'score', 'content', 'embedding', 'metadata'
            top_k: Number of diverse results to select
            lambda_param: Override default lambda parameter
            
        Returns:
            List of MMRResult ordered by selection order
        """
        start_time = time.time()
        lambda_param = lambda_param if lambda_param is not None else self.config.lambda_param
        
        if not results:
            logger.warning("No results to select from")
            return []
        
        if top_k <= 0:
            logger.warning("top_k must be positive")
            return []
        
        # Limit to available results
        top_k = min(top_k, len(results))
        
        # Validate embeddings if required
        if self.config.use_embeddings:
            if query_embedding is None:
                logger.error("Query embedding required when use_embeddings=True")
                return self._fallback_to_top_k(results, top_k)
            
            # Check if results have embeddings
            if not all('embedding' in r and r['embedding'] is not None for r in results):
                logger.warning("Some results missing embeddings, falling back to top-k")
                return self._fallback_to_top_k(results, top_k)
        
        try:
            # Prepare candidates
            candidates = []
            for idx, result in enumerate(results):
                doc_id = result.get('doc_id')
                score = result.get('score', 0.0)
                content = result.get('content', '')
                embedding = result.get('embedding')
                metadata = result.get('metadata', {})
                
                if not doc_id:
                    continue
                
                # Convert embedding to numpy array if needed
                if embedding is not None and not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                candidates.append({
                    'doc_id': doc_id,
                    'score': score,
                    'content': content,
                    'embedding': embedding,
                    'metadata': metadata,
                    'original_rank': idx + 1
                })
            
            if not candidates:
                logger.warning("No valid candidates for MMR")
                return []
            
            # Perform MMR selection
            selected = []
            remaining = candidates.copy()
            
            for _ in range(top_k):
                if not remaining:
                    break
                
                # Select next document using MMR
                next_doc, mmr_score, diversity_score = self._select_next(
                    query_embedding,
                    remaining,
                    selected,
                    lambda_param
                )
                
                if next_doc is None:
                    break
                
                # Add to selected
                selected.append(MMRResult(
                    doc_id=next_doc['doc_id'],
                    relevance_score=next_doc['score'],
                    diversity_score=diversity_score,
                    mmr_score=mmr_score,
                    rank=len(selected),
                    original_rank=next_doc['original_rank'],
                    content=next_doc['content'],
                    embedding=next_doc['embedding'],
                    metadata=next_doc['metadata']
                ))
                
                # Remove from remaining
                remaining = [r for r in remaining if r['doc_id'] != next_doc['doc_id']]
            
            # Update metrics
            self._metrics['total_selections'] += 1
            self._metrics['total_documents_processed'] += len(candidates)
            
            if selected:
                avg_diversity = sum(r.diversity_score for r in selected) / len(selected)
                total_selections = self._metrics['total_selections']
                current_avg = self._metrics['avg_diversity_score']
                self._metrics['avg_diversity_score'] = (
                    (current_avg * (total_selections - 1) + avg_diversity) / total_selections
                )
            
            selection_time_ms = (time.time() - start_time) * 1000
            total_selections = self._metrics['total_selections']
            current_avg = self._metrics['avg_selection_time_ms']
            self._metrics['avg_selection_time_ms'] = (
                (current_avg * (total_selections - 1) + selection_time_ms) / total_selections
            )
            
            logger.debug(
                f"MMR selection completed",
                selected_count=len(selected),
                candidates_count=len(candidates),
                selection_time_ms=selection_time_ms
            )
            
            return selected
            
        except Exception as e:
            logger.error(f"MMR selection failed: {e}")
            return self._fallback_to_top_k(results, top_k)
    
    def _select_next(
        self,
        query_embedding: Optional[np.ndarray],
        remaining: List[Dict[str, Any]],
        selected: List[MMRResult],
        lambda_param: float
    ) -> Tuple[Optional[Dict[str, Any]], float, float]:
        """
        Select next document using MMR formula.
        
        Returns:
            Tuple of (selected_doc, mmr_score, diversity_score)
        """
        best_doc = None
        best_mmr_score = float('-inf')
        best_diversity_score = 0.0
        
        for candidate in remaining:
            # Calculate relevance score (similarity to query)
            if self.config.use_embeddings and query_embedding is not None:
                relevance = self._compute_similarity(
                    query_embedding,
                    candidate['embedding']
                )
            else:
                # Use original score as relevance
                relevance = candidate['score']
            
            # Calculate diversity score (dissimilarity to selected)
            if selected:
                # Max similarity to any selected document
                max_similarity = max(
                    self._compute_similarity(
                        candidate['embedding'],
                        sel.embedding
                    ) if self.config.use_embeddings and candidate['embedding'] is not None
                    else 0.0
                    for sel in selected
                )
                diversity = 1.0 - max_similarity
            else:
                # First selection, max diversity
                diversity = 1.0
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * (1 - diversity)
            
            # Update best if this is better
            if mmr_score > best_mmr_score:
                # Check diversity threshold
                if diversity >= (1.0 - self.config.diversity_threshold):
                    best_doc = candidate
                    best_mmr_score = mmr_score
                    best_diversity_score = diversity
        
        return best_doc, best_mmr_score, best_diversity_score

    def _compute_similarity(
        self,
        vec1: Optional[np.ndarray],
        vec2: Optional[np.ndarray]
    ) -> float:
        """
        Compute similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score [0, 1]
        """
        if vec1 is None or vec2 is None:
            return 0.0

        # Ensure numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)

        # Handle dimension mismatch
        if vec1.shape != vec2.shape:
            logger.warning(
                f"Vector dimension mismatch: {vec1.shape} vs {vec2.shape}",
                fallback="zero similarity"
            )
            return 0.0

        try:
            if self.config.similarity_metric == SimilarityMetric.COSINE:
                # Cosine similarity
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 == 0 or norm2 == 0:
                    return 0.0

                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                # Clamp to [0, 1]
                return float(max(0.0, min(1.0, (similarity + 1) / 2)))

            elif self.config.similarity_metric == SimilarityMetric.DOT_PRODUCT:
                # Dot product (assumes normalized vectors)
                similarity = np.dot(vec1, vec2)
                # Clamp to [0, 1]
                return float(max(0.0, min(1.0, similarity)))

            elif self.config.similarity_metric == SimilarityMetric.EUCLIDEAN:
                # Euclidean distance converted to similarity
                distance = np.linalg.norm(vec1 - vec2)
                # Convert to similarity: 1 / (1 + distance)
                similarity = 1.0 / (1.0 + distance)
                return float(similarity)

            else:
                logger.warning(f"Unknown similarity metric: {self.config.similarity_metric}")
                return 0.0

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    def _fallback_to_top_k(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[MMRResult]:
        """
        Fallback to simple top-k selection when MMR fails.

        Args:
            results: Original results
            top_k: Number of results to return

        Returns:
            List of MMRResult using original ranking
        """
        logger.warning("Falling back to top-k selection")

        mmr_results = []
        for idx, result in enumerate(results[:top_k], 1):
            doc_id = result.get('doc_id')
            score = result.get('score', 0.0)
            content = result.get('content', '')
            embedding = result.get('embedding')
            metadata = result.get('metadata', {})

            if not doc_id:
                continue

            # Convert embedding if needed
            if embedding is not None and not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            mmr_results.append(MMRResult(
                doc_id=doc_id,
                relevance_score=score,
                diversity_score=1.0,  # Assume max diversity
                mmr_score=score,
                rank=idx,
                original_rank=idx,
                content=content,
                embedding=embedding,
                metadata=metadata
            ))

        return mmr_results

    def get_metrics(self) -> Dict[str, Any]:
        """Get MMR metrics."""
        return {
            'total_selections': self._metrics['total_selections'],
            'total_documents_processed': self._metrics['total_documents_processed'],
            'avg_selection_time_ms': self._metrics['avg_selection_time_ms'],
            'avg_diversity_score': self._metrics['avg_diversity_score']
        }

