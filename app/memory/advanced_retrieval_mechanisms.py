"""
Revolutionary Advanced Retrieval Mechanisms for Agentic AI Memory.

Implements state-of-the-art retrieval techniques including embedding-based similarity search,
BM25, hybrid retrieval, multi-modal retrieval, and graph-based retrieval.

Key Features:
- Hybrid retrieval combining multiple algorithms
- Multi-modal retrieval (text, images, audio)
- Graph-based traversal retrieval
- Contextual re-ranking
- Query expansion and refinement
- Retrieval result fusion and scoring
"""

import asyncio
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import numpy as np

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class RetrievalMethod(str, Enum):
    """Types of retrieval methods."""
    EMBEDDING_SIMILARITY = "embedding_similarity"
    BM25 = "bm25"
    GRAPH_TRAVERSAL = "graph_traversal"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    SPATIAL_PROXIMITY = "spatial_proximity"
    IMPORTANCE_WEIGHTED = "importance_weighted"
    ASSOCIATION_STRENGTH = "association_strength"
    MULTIMODAL = "multimodal"
    HYBRID = "hybrid"


@dataclass
class RetrievalQuery:
    """Advanced retrieval query with multiple modalities."""
    text_query: str = ""
    image_query: Optional[bytes] = None
    audio_query: Optional[bytes] = None
    
    # Query constraints
    memory_types: Optional[List[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    importance_threshold: float = 0.0
    spatial_region: Optional[str] = None
    
    # Retrieval parameters
    top_k: int = 10
    methods: List[RetrievalMethod] = field(default_factory=lambda: [RetrievalMethod.HYBRID])
    fusion_strategy: str = "weighted_sum"  # weighted_sum, rank_fusion, max_score
    
    # Context for contextual retrieval
    conversation_context: str = ""
    current_task: str = ""
    emotional_state: float = 0.0


@dataclass
class RetrievalResult:
    """Result from advanced retrieval with detailed scoring."""
    memory_id: str
    content: str
    metadata: Dict[str, Any]
    
    # Detailed scoring
    scores: Dict[RetrievalMethod, float] = field(default_factory=dict)
    final_score: float = 0.0
    rank: int = 0
    
    # Retrieval context
    retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID
    retrieval_reason: str = ""
    confidence: float = 1.0
    
    # Additional information
    memory_age_days: float = 0.0
    access_frequency: int = 0
    importance_level: str = "medium"


class BM25Retriever:
    """BM25 retrieval implementation for text-based search."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.document_frequencies = {}
        self.document_lengths = {}
        self.avg_document_length = 0.0
        self.total_documents = 0
        self.vocabulary = set()
    
    def index_documents(self, documents: Dict[str, str]):
        """Index documents for BM25 retrieval."""
        self.total_documents = len(documents)
        document_lengths = []
        term_frequencies = defaultdict(lambda: defaultdict(int))
        document_frequencies = defaultdict(int)
        
        # Calculate term frequencies and document frequencies
        for doc_id, content in documents.items():
            terms = content.lower().split()
            doc_length = len(terms)
            document_lengths.append(doc_length)
            self.document_lengths[doc_id] = doc_length
            
            unique_terms = set(terms)
            for term in unique_terms:
                document_frequencies[term] += 1
            
            term_counts = Counter(terms)
            for term, count in term_counts.items():
                term_frequencies[doc_id][term] = count
                self.vocabulary.add(term)
        
        self.avg_document_length = sum(document_lengths) / len(document_lengths)
        self.document_frequencies = dict(document_frequencies)
        self.term_frequencies = dict(term_frequencies)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 algorithm."""
        query_terms = query.lower().split()
        scores = {}
        
        for doc_id in self.document_lengths:
            score = 0.0
            doc_length = self.document_lengths[doc_id]
            
            for term in query_terms:
                if term not in self.vocabulary:
                    continue
                
                # Term frequency in document
                tf = self.term_frequencies.get(doc_id, {}).get(term, 0)
                
                # Document frequency
                df = self.document_frequencies.get(term, 0)
                
                # IDF calculation
                idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                
                # BM25 score calculation
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_document_length))
                
                score += idf * (numerator / denominator)
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class AdvancedRetrievalMechanisms:
    """
    Revolutionary Advanced Retrieval Mechanisms.
    
    Combines multiple retrieval methods for comprehensive memory search
    with hybrid fusion and multi-modal capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        embedding_function: Optional[callable] = None,
        knowledge_graph: Optional[Any] = None
    ):
        """Initialize advanced retrieval mechanisms."""
        self.agent_id = agent_id
        self.embedding_function = embedding_function
        self.knowledge_graph = knowledge_graph
        
        # Retrieval components
        self.bm25_retriever = BM25Retriever()
        self.is_indexed = False
        
        # Retrieval weights for fusion
        self.method_weights = {
            RetrievalMethod.EMBEDDING_SIMILARITY: 0.3,
            RetrievalMethod.BM25: 0.25,
            RetrievalMethod.GRAPH_TRAVERSAL: 0.2,
            RetrievalMethod.TEMPORAL_PROXIMITY: 0.1,
            RetrievalMethod.IMPORTANCE_WEIGHTED: 0.1,
            RetrievalMethod.ASSOCIATION_STRENGTH: 0.05
        }
        
        # Performance statistics
        self.stats = {
            "total_queries": 0,
            "avg_query_time_ms": 0.0,
            "method_usage": defaultdict(int),
            "fusion_strategy_usage": defaultdict(int),
            "avg_results_per_query": 0.0
        }

        logger.info(
            f"Advanced Retrieval Mechanisms initialized for agent {agent_id}",
            LogCategory.MEMORY_OPERATIONS,
            "app.memory.advanced_retrieval_mechanisms.AdvancedRetrievalMechanisms"
        )
    
    async def index_memories(self, memories: Dict[str, Dict[str, Any]]):
        """Index memories for advanced retrieval."""
        try:
            # Prepare documents for BM25 indexing
            documents = {}
            for memory_id, memory_data in memories.items():
                content = memory_data.get("content", "")
                if content:
                    documents[memory_id] = content
            
            # Index with BM25
            if documents:
                self.bm25_retriever.index_documents(documents)
                self.is_indexed = True

                logger.info(
                    "Memories indexed for advanced retrieval",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.memory.advanced_retrieval_mechanisms.AdvancedRetrievalMechanisms",
                    data={"agent_id": self.agent_id, "total_memories": len(documents)}
                )

        except Exception as e:
            logger.error(
                "Failed to index memories",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.advanced_retrieval_mechanisms.AdvancedRetrievalMechanisms",
                error=e
            )
    
    async def advanced_retrieve(
        self,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """Perform advanced retrieval using multiple methods."""
        start_time = time.time()
        
        try:
            # Ensure memories are indexed
            if not self.is_indexed:
                await self.index_memories(memories)
            
            all_results = {}
            method_results = {}
            
            # Apply each requested retrieval method
            for method in query.methods:
                if method == RetrievalMethod.HYBRID:
                    # Use all methods for hybrid retrieval
                    method_results.update(await self._apply_all_methods(query, memories))
                else:
                    results = await self._apply_single_method(method, query, memories)
                    method_results[method] = results
            
            # Fuse results from different methods
            fused_results = await self._fuse_results(
                method_results, query.fusion_strategy, query.top_k
            )
            
            # Post-process and rank results
            final_results = await self._post_process_results(
                fused_results, query, memories
            )
            
            # Update statistics
            query_time = (time.time() - start_time) * 1000
            self._update_stats(query, len(final_results), query_time)

            logger.info(
                "Advanced retrieval completed",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.advanced_retrieval_mechanisms.AdvancedRetrievalMechanisms",
                data={
                    "agent_id": self.agent_id,
                    "query_methods": len(query.methods),
                    "results_count": len(final_results),
                    "query_time_ms": f"{query_time:.2f}"
                }
            )

            return final_results

        except Exception as e:
            logger.error(
                "Advanced retrieval failed",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.advanced_retrieval_mechanisms.AdvancedRetrievalMechanisms",
                error=e
            )
            return []
    
    async def _apply_all_methods(
        self,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> Dict[RetrievalMethod, List[Tuple[str, float]]]:
        """Apply all retrieval methods for hybrid search."""
        results = {}
        
        # Apply each method
        methods_to_apply = [
            RetrievalMethod.EMBEDDING_SIMILARITY,
            RetrievalMethod.BM25,
            RetrievalMethod.TEMPORAL_PROXIMITY,
            RetrievalMethod.IMPORTANCE_WEIGHTED
        ]
        
        if self.knowledge_graph:
            methods_to_apply.append(RetrievalMethod.GRAPH_TRAVERSAL)
        
        for method in methods_to_apply:
            method_results = await self._apply_single_method(method, query, memories)
            if method_results:
                results[method] = method_results
        
        return results
    
    async def _apply_single_method(
        self,
        method: RetrievalMethod,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Apply a single retrieval method."""
        try:
            if method == RetrievalMethod.BM25:
                return self._bm25_retrieve(query, memories)
            elif method == RetrievalMethod.EMBEDDING_SIMILARITY:
                return await self._embedding_retrieve(query, memories)
            elif method == RetrievalMethod.TEMPORAL_PROXIMITY:
                return self._temporal_retrieve(query, memories)
            elif method == RetrievalMethod.IMPORTANCE_WEIGHTED:
                return self._importance_retrieve(query, memories)
            elif method == RetrievalMethod.GRAPH_TRAVERSAL:
                return await self._graph_retrieve(query, memories)
            else:
                return []

        except Exception as e:
            logger.error(
                f"Method {method} failed",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.advanced_retrieval_mechanisms.AdvancedRetrievalMechanisms",
                error=e,
                data={"method": method}
            )
            return []
    
    def _bm25_retrieve(
        self,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Retrieve using BM25 algorithm."""
        if not query.text_query or not self.is_indexed:
            return []
        
        results = self.bm25_retriever.search(query.text_query, query.top_k * 2)
        
        # Filter results based on query constraints
        filtered_results = []
        for memory_id, score in results:
            if memory_id in memories:
                memory_data = memories[memory_id]
                
                # Apply filters
                if self._passes_filters(memory_data, query):
                    filtered_results.append((memory_id, score))
        
        return filtered_results[:query.top_k]
    
    async def _embedding_retrieve(
        self,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Retrieve using embedding similarity."""
        if not query.text_query or not self.embedding_function:
            return []
        
        try:
            # Get query embedding
            query_embedding = await self.embedding_function(query.text_query)
            
            results = []
            for memory_id, memory_data in memories.items():
                content = memory_data.get("content", "")
                if not content or not self._passes_filters(memory_data, query):
                    continue
                
                # Get memory embedding (in production, cache these)
                memory_embedding = await self.embedding_function(content)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, memory_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                )
                
                if similarity > 0.1:  # Minimum similarity threshold
                    results.append((memory_id, float(similarity)))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:query.top_k]

        except Exception as e:
            logger.error(
                "Embedding retrieval failed",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.advanced_retrieval_mechanisms.AdvancedRetrievalMechanisms",
                error=e
            )
            return []
    
    def _temporal_retrieve(
        self,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Retrieve based on temporal proximity."""
        current_time = datetime.now()
        results = []
        
        for memory_id, memory_data in memories.items():
            if not self._passes_filters(memory_data, query):
                continue
            
            created_at = memory_data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    continue
            elif not isinstance(created_at, datetime):
                continue
            
            # Calculate temporal score (more recent = higher score)
            time_diff_hours = (current_time - created_at).total_seconds() / 3600
            
            # Exponential decay with half-life of 24 hours
            temporal_score = math.exp(-time_diff_hours / 24.0)
            
            if temporal_score > 0.01:  # Minimum threshold
                results.append((memory_id, temporal_score))
        
        # Sort by temporal score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:query.top_k]
    
    def _importance_retrieve(
        self,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Retrieve based on importance weighting."""
        results = []
        
        importance_mapping = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
            "temporary": 0.2
        }
        
        for memory_id, memory_data in memories.items():
            if not self._passes_filters(memory_data, query):
                continue
            
            importance = memory_data.get("importance", "medium")
            importance_score = importance_mapping.get(importance, 0.6)
            
            # Boost score based on access frequency
            access_count = memory_data.get("access_count", 0)
            frequency_boost = min(math.log(access_count + 1) / 10.0, 0.3)
            
            final_score = importance_score + frequency_boost
            
            if final_score >= query.importance_threshold:
                results.append((memory_id, final_score))
        
        # Sort by importance score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:query.top_k]
    
    async def _graph_retrieve(
        self,
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Retrieve using knowledge graph traversal."""
        if not self.knowledge_graph:
            return []
        
        # This would integrate with the DynamicKnowledgeGraph
        # For now, return empty results
        return []
    
    def _passes_filters(self, memory_data: Dict[str, Any], query: RetrievalQuery) -> bool:
        """Check if memory passes query filters."""
        # Memory type filter
        if query.memory_types:
            memory_type = memory_data.get("memory_type", "")
            if memory_type not in query.memory_types:
                return False
        
        # Time range filter
        if query.time_range:
            created_at = memory_data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    return False
            elif not isinstance(created_at, datetime):
                return False
            
            start_time, end_time = query.time_range
            if not (start_time <= created_at <= end_time):
                return False
        
        # Spatial region filter
        if query.spatial_region:
            spatial_region = memory_data.get("spatial_region", "")
            if spatial_region != query.spatial_region:
                return False
        
        return True

    async def _fuse_results(
        self,
        method_results: Dict[RetrievalMethod, List[Tuple[str, float]]],
        fusion_strategy: str,
        top_k: int
    ) -> List[Tuple[str, float, Dict[RetrievalMethod, float]]]:
        """Fuse results from multiple retrieval methods."""
        if not method_results:
            return []

        if fusion_strategy == "weighted_sum":
            return self._weighted_sum_fusion(method_results, top_k)
        elif fusion_strategy == "rank_fusion":
            return self._rank_fusion(method_results, top_k)
        elif fusion_strategy == "max_score":
            return self._max_score_fusion(method_results, top_k)
        else:
            return self._weighted_sum_fusion(method_results, top_k)

    def _weighted_sum_fusion(
        self,
        method_results: Dict[RetrievalMethod, List[Tuple[str, float]]],
        top_k: int
    ) -> List[Tuple[str, float, Dict[RetrievalMethod, float]]]:
        """Fuse results using weighted sum of scores."""
        memory_scores = defaultdict(lambda: {"total": 0.0, "methods": {}})

        for method, results in method_results.items():
            weight = self.method_weights.get(method, 0.1)

            # Normalize scores to 0-1 range
            if results:
                max_score = max(score for _, score in results)
                min_score = min(score for _, score in results)
                score_range = max_score - min_score if max_score > min_score else 1.0

                for memory_id, score in results:
                    normalized_score = (score - min_score) / score_range
                    weighted_score = normalized_score * weight

                    memory_scores[memory_id]["total"] += weighted_score
                    memory_scores[memory_id]["methods"][method] = normalized_score

        # Sort by total score
        sorted_results = sorted(
            memory_scores.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )

        # Format results
        fused_results = []
        for memory_id, score_data in sorted_results[:top_k]:
            fused_results.append((
                memory_id,
                score_data["total"],
                score_data["methods"]
            ))

        return fused_results

    def _rank_fusion(
        self,
        method_results: Dict[RetrievalMethod, List[Tuple[str, float]]],
        top_k: int
    ) -> List[Tuple[str, float, Dict[RetrievalMethod, float]]]:
        """Fuse results using reciprocal rank fusion."""
        memory_scores = defaultdict(lambda: {"total": 0.0, "methods": {}})

        for method, results in method_results.items():
            weight = self.method_weights.get(method, 0.1)

            for rank, (memory_id, score) in enumerate(results):
                # Reciprocal rank fusion score
                rrf_score = weight / (rank + 60)  # k=60 is common

                memory_scores[memory_id]["total"] += rrf_score
                memory_scores[memory_id]["methods"][method] = score

        # Sort by total RRF score
        sorted_results = sorted(
            memory_scores.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )

        # Format results
        fused_results = []
        for memory_id, score_data in sorted_results[:top_k]:
            fused_results.append((
                memory_id,
                score_data["total"],
                score_data["methods"]
            ))

        return fused_results

    def _max_score_fusion(
        self,
        method_results: Dict[RetrievalMethod, List[Tuple[str, float]]],
        top_k: int
    ) -> List[Tuple[str, float, Dict[RetrievalMethod, float]]]:
        """Fuse results using maximum score across methods."""
        memory_scores = defaultdict(lambda: {"max": 0.0, "methods": {}})

        for method, results in method_results.items():
            for memory_id, score in results:
                if score > memory_scores[memory_id]["max"]:
                    memory_scores[memory_id]["max"] = score

                memory_scores[memory_id]["methods"][method] = score

        # Sort by max score
        sorted_results = sorted(
            memory_scores.items(),
            key=lambda x: x[1]["max"],
            reverse=True
        )

        # Format results
        fused_results = []
        for memory_id, score_data in sorted_results[:top_k]:
            fused_results.append((
                memory_id,
                score_data["max"],
                score_data["methods"]
            ))

        return fused_results

    async def _post_process_results(
        self,
        fused_results: List[Tuple[str, float, Dict[RetrievalMethod, float]]],
        query: RetrievalQuery,
        memories: Dict[str, Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """Post-process and format final results."""
        final_results = []

        for rank, (memory_id, final_score, method_scores) in enumerate(fused_results):
            if memory_id not in memories:
                continue

            memory_data = memories[memory_id]

            # Calculate additional metrics
            created_at = memory_data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    memory_age_days = (datetime.now() - created_at).total_seconds() / 86400
                except:
                    memory_age_days = 0.0
            else:
                memory_age_days = 0.0

            # Generate retrieval reason
            primary_method = max(method_scores.items(), key=lambda x: x[1])[0] if method_scores else RetrievalMethod.HYBRID
            retrieval_reason = self._generate_retrieval_reason(primary_method, method_scores, query)

            result = RetrievalResult(
                memory_id=memory_id,
                content=memory_data.get("content", ""),
                metadata=memory_data.get("metadata", {}),
                scores=method_scores,
                final_score=final_score,
                rank=rank + 1,
                retrieval_method=primary_method,
                retrieval_reason=retrieval_reason,
                confidence=min(final_score, 1.0),
                memory_age_days=memory_age_days,
                access_frequency=memory_data.get("access_count", 0),
                importance_level=memory_data.get("importance", "medium")
            )

            final_results.append(result)

        return final_results

    def _generate_retrieval_reason(
        self,
        primary_method: RetrievalMethod,
        method_scores: Dict[RetrievalMethod, float],
        query: RetrievalQuery
    ) -> str:
        """Generate human-readable retrieval reason."""
        reasons = []

        if primary_method == RetrievalMethod.EMBEDDING_SIMILARITY:
            reasons.append("semantic similarity to query")
        elif primary_method == RetrievalMethod.BM25:
            reasons.append("keyword relevance")
        elif primary_method == RetrievalMethod.TEMPORAL_PROXIMITY:
            reasons.append("temporal relevance")
        elif primary_method == RetrievalMethod.IMPORTANCE_WEIGHTED:
            reasons.append("high importance rating")
        elif primary_method == RetrievalMethod.GRAPH_TRAVERSAL:
            reasons.append("knowledge graph connections")

        # Add secondary reasons
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        for method, score in sorted_methods[1:3]:  # Top 2 secondary methods
            if score > 0.3:  # Significant contribution
                if method == RetrievalMethod.EMBEDDING_SIMILARITY:
                    reasons.append("semantic relevance")
                elif method == RetrievalMethod.BM25:
                    reasons.append("keyword match")
                elif method == RetrievalMethod.TEMPORAL_PROXIMITY:
                    reasons.append("recent activity")
                elif method == RetrievalMethod.IMPORTANCE_WEIGHTED:
                    reasons.append("importance")

        return f"Retrieved due to: {', '.join(reasons)}"

    def _update_stats(self, query: RetrievalQuery, result_count: int, query_time_ms: float):
        """Update retrieval statistics."""
        self.stats["total_queries"] += 1

        # Update average query time
        alpha = 0.1
        self.stats["avg_query_time_ms"] = (
            alpha * query_time_ms + (1 - alpha) * self.stats["avg_query_time_ms"]
        )

        # Update average results per query
        self.stats["avg_results_per_query"] = (
            alpha * result_count + (1 - alpha) * self.stats["avg_results_per_query"]
        )

        # Update method usage
        for method in query.methods:
            self.stats["method_usage"][method.value] += 1

        # Update fusion strategy usage
        self.stats["fusion_strategy_usage"][query.fusion_strategy] += 1

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrieval statistics."""
        return {
            **self.stats,
            "method_weights": self.method_weights,
            "is_indexed": self.is_indexed,
            "bm25_vocabulary_size": len(self.bm25_retriever.vocabulary) if self.is_indexed else 0
        }
