"""
Advanced Hybrid Search Implementation for RAG 4.0.

Combines dense vector search (ChromaDB) with sparse retrieval (BM25/TF-IDF)
for superior retrieval performance and relevance scoring.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib
from pathlib import Path

from .knowledge_base import Document, KnowledgeQuery, KnowledgeResult
from .vector_store import ChromaVectorStore
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class SearchStrategy(str, Enum):
    """Available search strategies for hybrid retrieval."""
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID_BALANCED = "hybrid_balanced"
    HYBRID_DENSE_WEIGHTED = "hybrid_dense_weighted"
    HYBRID_SPARSE_WEIGHTED = "hybrid_sparse_weighted"
    ADAPTIVE = "adaptive"


@dataclass
class SearchResult:
    """Enhanced search result with scoring details."""
    document: Document
    dense_score: float
    sparse_score: float
    hybrid_score: float
    confidence: float
    retrieval_method: str
    metadata: Dict[str, Any]


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search system."""
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    min_confidence_threshold: float = 0.1
    max_results: int = 50
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    cache_tfidf_models: bool = True
    adaptive_threshold: float = 0.8


class TFIDFSearchEngine:
    """TF-IDF based sparse retrieval engine."""
    
    def __init__(self, config: HybridSearchConfig):
        self.config = config
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.document_matrices: Dict[str, Any] = {}
        self.documents: Dict[str, List[Document]] = {}
        self.cache_dir = Path(get_settings().DATA_DIR) / "tfidf_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    async def index_documents(self, collection_name: str, documents: List[Document]) -> None:
        """Index documents for TF-IDF search."""
        try:
            # Prepare text corpus
            texts = [doc.content for doc in documents]
            
            # Create or load TF-IDF vectorizer
            cache_key = self._get_cache_key(collection_name, texts)
            vectorizer_path = self.cache_dir / f"vectorizer_{cache_key}.pkl"
            matrix_path = self.cache_dir / f"matrix_{cache_key}.pkl"
            
            if self.config.cache_tfidf_models and vectorizer_path.exists():
                logger.info("Loading cached TF-IDF model", collection=collection_name)
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                with open(matrix_path, 'rb') as f:
                    doc_matrix = pickle.load(f)
            else:
                logger.info("Creating new TF-IDF model", collection=collection_name)
                vectorizer = TfidfVectorizer(
                    max_features=self.config.tfidf_max_features,
                    ngram_range=self.config.tfidf_ngram_range,
                    stop_words='english',
                    lowercase=True,
                    strip_accents='unicode'
                )
                doc_matrix = vectorizer.fit_transform(texts)
                
                # Cache the model
                if self.config.cache_tfidf_models:
                    with open(vectorizer_path, 'wb') as f:
                        pickle.dump(vectorizer, f)
                    with open(matrix_path, 'wb') as f:
                        pickle.dump(doc_matrix, f)
            
            self.vectorizers[collection_name] = vectorizer
            self.document_matrices[collection_name] = doc_matrix
            self.documents[collection_name] = documents
            
            logger.info(
                "TF-IDF indexing completed",
                collection=collection_name,
                documents=len(documents),
                features=doc_matrix.shape[1] if hasattr(doc_matrix, 'shape') else 0
            )
            
        except Exception as e:
            logger.error("TF-IDF indexing failed", collection=collection_name, error=str(e))
            raise
    
    async def search(self, collection_name: str, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Perform TF-IDF search."""
        if collection_name not in self.vectorizers:
            logger.warning("Collection not indexed for TF-IDF", collection=collection_name)
            return []
        
        try:
            vectorizer = self.vectorizers[collection_name]
            doc_matrix = self.document_matrices[collection_name]
            documents = self.documents[collection_name]
            
            # Transform query
            query_vector = vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, doc_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero similarities
                    results.append((documents[idx], float(similarities[idx])))
            
            logger.debug(
                "TF-IDF search completed",
                collection=collection_name,
                query_length=len(query),
                results=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error("TF-IDF search failed", collection=collection_name, error=str(e))
            return []
    
    def _get_cache_key(self, collection_name: str, texts: List[str]) -> str:
        """Generate cache key for TF-IDF model."""
        content_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
        return f"{collection_name}_{content_hash[:8]}"


class HybridSearchEngine:
    """Advanced hybrid search engine combining dense and sparse retrieval."""
    
    def __init__(self, vector_store: ChromaVectorStore, config: Optional[HybridSearchConfig] = None):
        self.vector_store = vector_store
        self.config = config or HybridSearchConfig()
        self.tfidf_engine = TFIDFSearchEngine(self.config)
        self.performance_stats = {
            "total_searches": 0,
            "dense_searches": 0,
            "sparse_searches": 0,
            "hybrid_searches": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0
        }
        
    async def index_collection(self, collection_name: str, documents: List[Document]) -> None:
        """Index documents in both dense and sparse engines."""
        try:
            # Index in TF-IDF engine
            await self.tfidf_engine.index_documents(collection_name, documents)
            
            # Dense indexing is handled by ChromaDB automatically
            logger.info(
                "Hybrid indexing completed",
                collection=collection_name,
                documents=len(documents)
            )
            
        except Exception as e:
            logger.error("Hybrid indexing failed", collection=collection_name, error=str(e))
            raise
    
    async def search(
        self,
        collection_name: str,
        query: KnowledgeQuery,
        strategy: SearchStrategy = SearchStrategy.HYBRID_BALANCED
    ) -> List[SearchResult]:
        """Perform hybrid search with configurable strategy."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.performance_stats["total_searches"] += 1
            
            if strategy == SearchStrategy.DENSE_ONLY:
                results = await self._dense_search_only(collection_name, query)
                self.performance_stats["dense_searches"] += 1
                
            elif strategy == SearchStrategy.SPARSE_ONLY:
                results = await self._sparse_search_only(collection_name, query)
                self.performance_stats["sparse_searches"] += 1
                
            elif strategy == SearchStrategy.ADAPTIVE:
                results = await self._adaptive_search(collection_name, query)
                self.performance_stats["hybrid_searches"] += 1
                
            else:
                results = await self._hybrid_search(collection_name, query, strategy)
                self.performance_stats["hybrid_searches"] += 1
            
            # Apply confidence filtering
            filtered_results = [
                r for r in results 
                if r.confidence >= self.config.min_confidence_threshold
            ]
            
            # Limit results
            final_results = filtered_results[:self.config.max_results]
            
            # Update performance stats
            response_time = asyncio.get_event_loop().time() - start_time
            self._update_performance_stats(response_time)
            
            logger.info(
                "Hybrid search completed",
                collection=collection_name,
                strategy=strategy.value,
                results=len(final_results),
                response_time=response_time
            )
            
            return final_results
            
        except Exception as e:
            logger.error("Hybrid search failed", collection=collection_name, error=str(e))
            return []
    
    async def _dense_search_only(self, collection_name: str, query: KnowledgeQuery) -> List[SearchResult]:
        """Perform dense vector search only."""
        try:
            # Get dense results from ChromaDB
            dense_results = await self.vector_store.search(
                collection_name=collection_name,
                query_text=query.query,
                n_results=self.config.max_results,
                where=query.filters
            )
            
            results = []
            for doc, score in zip(dense_results.documents, dense_results.distances):
                # Convert distance to similarity score
                similarity = 1.0 / (1.0 + score) if score > 0 else 1.0
                
                search_result = SearchResult(
                    document=doc,
                    dense_score=similarity,
                    sparse_score=0.0,
                    hybrid_score=similarity,
                    confidence=similarity,
                    retrieval_method="dense_only",
                    metadata={"original_distance": score}
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            logger.error("Dense search failed", collection=collection_name, error=str(e))
            return []
    
    async def _sparse_search_only(self, collection_name: str, query: KnowledgeQuery) -> List[SearchResult]:
        """Perform sparse TF-IDF search only."""
        try:
            sparse_results = await self.tfidf_engine.search(
                collection_name, query.query, self.config.max_results
            )
            
            results = []
            for doc, score in sparse_results:
                search_result = SearchResult(
                    document=doc,
                    dense_score=0.0,
                    sparse_score=score,
                    hybrid_score=score,
                    confidence=score,
                    retrieval_method="sparse_only",
                    metadata={"tfidf_score": score}
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            logger.error("Sparse search failed", collection=collection_name, error=str(e))
            return []
    
    async def _hybrid_search(
        self, 
        collection_name: str, 
        query: KnowledgeQuery, 
        strategy: SearchStrategy
    ) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse results."""
        try:
            # Get both dense and sparse results concurrently
            dense_task = self._dense_search_only(collection_name, query)
            sparse_task = self._sparse_search_only(collection_name, query)
            
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            
            # Determine weights based on strategy
            if strategy == SearchStrategy.HYBRID_DENSE_WEIGHTED:
                dense_weight, sparse_weight = 0.8, 0.2
            elif strategy == SearchStrategy.HYBRID_SPARSE_WEIGHTED:
                dense_weight, sparse_weight = 0.2, 0.8
            else:  # HYBRID_BALANCED
                dense_weight, sparse_weight = self.config.dense_weight, self.config.sparse_weight
            
            # Combine results
            combined_results = self._combine_search_results(
                dense_results, sparse_results, dense_weight, sparse_weight
            )
            
            return combined_results
            
        except Exception as e:
            logger.error("Hybrid search failed", collection=collection_name, error=str(e))
            return []
    
    async def _adaptive_search(self, collection_name: str, query: KnowledgeQuery) -> List[SearchResult]:
        """Adaptive search that chooses strategy based on query characteristics."""
        try:
            # Analyze query to determine best strategy
            query_length = len(query.query.split())
            has_specific_terms = any(term.isupper() or term.isdigit() for term in query.query.split())
            
            if query_length <= 3 and has_specific_terms:
                # Short, specific queries work better with sparse search
                strategy = SearchStrategy.HYBRID_SPARSE_WEIGHTED
            elif query_length > 10:
                # Long queries work better with dense search
                strategy = SearchStrategy.HYBRID_DENSE_WEIGHTED
            else:
                # Balanced approach for medium queries
                strategy = SearchStrategy.HYBRID_BALANCED
            
            logger.debug(
                "Adaptive strategy selected",
                collection=collection_name,
                strategy=strategy.value,
                query_length=query_length,
                has_specific_terms=has_specific_terms
            )
            
            return await self._hybrid_search(collection_name, query, strategy)
            
        except Exception as e:
            logger.error("Adaptive search failed", collection=collection_name, error=str(e))
            return []
    
    def _combine_search_results(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight: float,
        sparse_weight: float
    ) -> List[SearchResult]:
        """Combine and rank results from dense and sparse search."""
        # Create lookup for efficient merging
        dense_lookup = {doc.id: result for result in dense_results for doc in [result.document]}
        sparse_lookup = {doc.id: result for result in sparse_results for doc in [result.document]}
        
        # Get all unique documents
        all_doc_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            dense_result = dense_lookup.get(doc_id)
            sparse_result = sparse_lookup.get(doc_id)
            
            # Calculate hybrid score
            dense_score = dense_result.dense_score if dense_result else 0.0
            sparse_score = sparse_result.sparse_score if sparse_result else 0.0
            hybrid_score = (dense_weight * dense_score) + (sparse_weight * sparse_score)
            
            # Calculate confidence (max of individual confidences)
            confidence = max(
                dense_result.confidence if dense_result else 0.0,
                sparse_result.confidence if sparse_result else 0.0
            )
            
            # Use document from whichever result has higher score
            document = (dense_result or sparse_result).document
            
            combined_result = SearchResult(
                document=document,
                dense_score=dense_score,
                sparse_score=sparse_score,
                hybrid_score=hybrid_score,
                confidence=confidence,
                retrieval_method="hybrid",
                metadata={
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight,
                    "found_in_dense": dense_result is not None,
                    "found_in_sparse": sparse_result is not None
                }
            )
            combined_results.append(combined_result)
        
        # Sort by hybrid score
        combined_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return combined_results
    
    def _update_performance_stats(self, response_time: float) -> None:
        """Update performance statistics."""
        total = self.performance_stats["total_searches"]
        current_avg = self.performance_stats["avg_response_time"]
        
        # Calculate new average response time
        new_avg = ((current_avg * (total - 1)) + response_time) / total
        self.performance_stats["avg_response_time"] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    async def clear_cache(self, collection_name: Optional[str] = None) -> None:
        """Clear TF-IDF cache for specific collection or all collections."""
        if collection_name:
            # Clear specific collection
            if collection_name in self.tfidf_engine.vectorizers:
                del self.tfidf_engine.vectorizers[collection_name]
            if collection_name in self.tfidf_engine.document_matrices:
                del self.tfidf_engine.document_matrices[collection_name]
            if collection_name in self.tfidf_engine.documents:
                del self.tfidf_engine.documents[collection_name]
        else:
            # Clear all caches
            self.tfidf_engine.vectorizers.clear()
            self.tfidf_engine.document_matrices.clear()
            self.tfidf_engine.documents.clear()
        
        logger.info("Hybrid search cache cleared", collection=collection_name or "all")
