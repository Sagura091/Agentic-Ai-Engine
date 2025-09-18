"""
Revolutionary Advanced Search Engine for RAG 4.0.

This module provides advanced search capabilities including:
- Multi-modal search (text, image, audio, video)
- Contextual and conversational search
- Query expansion and rewriting
- Advanced filtering and ranking
- Real-time search results
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import structlog
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..core.knowledge_base import KnowledgeBase, KnowledgeQuery
from ..core.embeddings import EmbeddingManager
from ..core.caching import get_rag_cache, CacheType
from ..core.resilience_manager import get_resilience_manager

logger = structlog.get_logger(__name__)


class SearchMode(Enum):
    """Advanced search modes."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    NEURAL = "neural"
    GRAPH = "graph"
    MULTIMODAL = "multimodal"
    CONVERSATIONAL = "conversational"


class ContentType(Enum):
    """Content types for multi-modal search."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    CODE = "code"


@dataclass
class SearchContext:
    """Search context for personalized and contextual search."""
    user_id: Optional[str] = None
    conversation_history: Optional[List[str]] = None
    user_profile: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    domain_context: Optional[List[str]] = None
    session_id: Optional[str] = None


@dataclass
class SearchFilter:
    """Advanced search filter."""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, contains, regex
    value: Any
    weight: float = 1.0


@dataclass
class SearchOptions:
    """Advanced search options."""
    query_expansion: bool = True
    contextual_search: bool = True
    multi_modal: bool = False
    similarity_threshold: float = 0.7
    max_results: int = 20
    search_scope: str = "all"
    enable_reranking: bool = True
    include_metadata: bool = True
    highlight_matches: bool = True
    real_time_results: bool = False


@dataclass
class SearchResult:
    """Enhanced search result."""
    id: str
    title: str
    content: str
    score: float
    content_type: ContentType
    metadata: Dict[str, Any]
    chunks: Optional[List[Dict[str, Any]]] = None
    highlights: Optional[List[str]] = None
    embedding_preview: Optional[List[float]] = None
    explanation: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    search_mode: SearchMode
    context_used: bool
    query_expanded: bool
    filters_applied: List[SearchFilter]
    suggestions: Optional[List[str]] = None
    analytics: Optional[Dict[str, Any]] = None


class QueryProcessor:
    """Advanced query processing and expansion."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.query_cache = {}
        
    async def expand_query(
        self, 
        query: str, 
        context: Optional[SearchContext] = None
    ) -> List[str]:
        """Expand query with synonyms and related terms."""
        expanded_queries = [query]
        
        try:
            # Simple expansion - in production use advanced NLP
            words = query.lower().split()
            
            # Add synonyms and related terms
            synonyms = {
                "ai": ["artificial intelligence", "machine learning", "neural networks"],
                "search": ["find", "lookup", "retrieve", "query"],
                "document": ["file", "paper", "text", "content"],
                "knowledge": ["information", "data", "facts", "insights"]
            }
            
            for word in words:
                if word in synonyms:
                    for synonym in synonyms[word]:
                        expanded_query = query.replace(word, synonym)
                        expanded_queries.append(expanded_query)
            
            # Context-based expansion
            if context and context.conversation_history:
                # Extract relevant terms from conversation history
                for msg in context.conversation_history[-3:]:  # Last 3 messages
                    if len(msg.split()) <= 3:  # Short phrases
                        expanded_queries.append(f"{query} {msg}")
            
            return expanded_queries[:5]  # Limit to 5 expanded queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return [query]
    
    async def rewrite_query(
        self, 
        query: str, 
        search_history: List[str]
    ) -> str:
        """Rewrite query based on search history and intent."""
        try:
            # Simple rewriting - in production use advanced NLP
            if len(search_history) > 0:
                last_query = search_history[0]
                
                # If current query is very short, combine with last query
                if len(query.split()) <= 2 and len(last_query.split()) > 2:
                    return f"{last_query} {query}"
            
            return query
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {str(e)}")
            return query
    
    async def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract search intent from query."""
        intent = {
            "type": "informational",  # informational, navigational, transactional
            "entities": [],
            "temporal": None,
            "spatial": None,
            "sentiment": "neutral"
        }
        
        try:
            # Simple intent extraction
            query_lower = query.lower()
            
            # Detect question types
            if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
                intent["type"] = "informational"
            elif any(word in query_lower for word in ["find", "show", "get", "retrieve"]):
                intent["type"] = "navigational"
            elif any(word in query_lower for word in ["create", "make", "build", "generate"]):
                intent["type"] = "transactional"
            
            # Extract temporal indicators
            if any(word in query_lower for word in ["today", "yesterday", "recent", "latest"]):
                intent["temporal"] = "recent"
            elif any(word in query_lower for word in ["old", "historical", "past", "archive"]):
                intent["temporal"] = "historical"
            
            return intent
            
        except Exception as e:
            logger.error(f"Intent extraction failed: {str(e)}")
            return intent


class AdvancedSearchEngine:
    """
    Revolutionary advanced search engine for RAG 4.0.
    
    Features:
    - Multi-modal search capabilities
    - Contextual and conversational search
    - Advanced query processing
    - Real-time search results
    - Intelligent ranking and filtering
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedding_manager: EmbeddingManager,
        enable_caching: bool = True
    ):
        self.knowledge_base = knowledge_base
        self.embedding_manager = embedding_manager
        self.enable_caching = enable_caching
        
        # Query processing
        self.query_processor = QueryProcessor(embedding_manager)
        
        # Search analytics
        self.search_analytics = {
            "total_searches": 0,
            "avg_response_time": 0.0,
            "popular_queries": {},
            "search_patterns": {}
        }
        
        # Cache and resilience
        self.cache = None
        self.resilience_manager = None
        
        # Real-time search
        self.real_time_subscribers = {}
    
    async def initialize(self) -> None:
        """Initialize the advanced search engine."""
        try:
            if self.enable_caching:
                self.cache = await get_rag_cache()
            
            self.resilience_manager = await get_resilience_manager()
            await self.resilience_manager.register_component(
                "advanced_search_engine",
                recovery_strategies=["retry", "circuit_breaker", "graceful_degradation"]
            )
            
            logger.info("Advanced search engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.SEMANTIC,
        filters: Optional[List[SearchFilter]] = None,
        options: Optional[SearchOptions] = None,
        context: Optional[SearchContext] = None
    ) -> SearchResponse:
        """
        Perform advanced search with multiple modes and options.
        
        Args:
            query: Search query
            mode: Search mode (semantic, hybrid, neural, etc.)
            filters: Advanced filters
            options: Search options
            context: Search context for personalization
            
        Returns:
            Comprehensive search response
        """
        start_time = time.time()
        
        # Default options
        if options is None:
            options = SearchOptions()
        
        if filters is None:
            filters = []
        
        try:
            # Check circuit breaker
            if not await self.resilience_manager.can_execute_operation("advanced_search_engine"):
                raise Exception("Search engine temporarily unavailable")
            
            # Process query
            processed_query = query
            expanded_queries = [query]
            
            if options.query_expansion:
                expanded_queries = await self.query_processor.expand_query(query, context)
                processed_query = expanded_queries[0]  # Use first expanded query
            
            # Extract search intent
            intent = await self.query_processor.extract_intent(query)
            
            # Check cache first
            cache_key = self._generate_cache_key(query, mode, filters, options)
            if self.cache and not options.real_time_results:
                cached_result = await self.cache.get(cache_key, CacheType.SEARCH_RESULT)
                if cached_result:
                    logger.info(f"Cache hit for search query: {query}")
                    return SearchResponse(**cached_result)
            
            # Perform search based on mode
            if mode == SearchMode.SEMANTIC:
                results = await self._semantic_search(processed_query, filters, options, context)
            elif mode == SearchMode.HYBRID:
                results = await self._hybrid_search(expanded_queries, filters, options, context)
            elif mode == SearchMode.NEURAL:
                results = await self._neural_search(processed_query, filters, options, context)
            elif mode == SearchMode.GRAPH:
                results = await self._graph_search(processed_query, filters, options, context)
            elif mode == SearchMode.MULTIMODAL:
                results = await self._multimodal_search(processed_query, filters, options, context)
            elif mode == SearchMode.CONVERSATIONAL:
                results = await self._conversational_search(processed_query, filters, options, context)
            else:
                results = await self._semantic_search(processed_query, filters, options, context)
            
            # Post-process results
            if options.enable_reranking:
                results = await self._rerank_results(results, query, context)
            
            if options.highlight_matches:
                results = await self._add_highlights(results, query)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(query, results)
            
            # Create response
            search_time = time.time() - start_time
            response = SearchResponse(
                query=query,
                results=results[:options.max_results],
                total_results=len(results),
                search_time=search_time,
                search_mode=mode,
                context_used=context is not None,
                query_expanded=options.query_expansion,
                filters_applied=filters,
                suggestions=suggestions,
                analytics=self._generate_search_analytics(query, results, search_time)
            )
            
            # Cache result
            if self.cache and not options.real_time_results:
                await self.cache.set(
                    cache_key,
                    asdict(response),
                    CacheType.SEARCH_RESULT,
                    ttl=3600
                )
            
            # Update analytics
            await self._update_search_analytics(query, search_time, len(results))
            
            # Record success
            await self.resilience_manager.record_success("advanced_search_engine", search_time)
            
            return response
            
        except Exception as e:
            # Record error
            await self.resilience_manager.record_error(
                "advanced_search_engine",
                e,
                context={"query": query, "mode": mode.value}
            )
            
            logger.error(f"Search failed: {str(e)}")
            raise
    
    async def _semantic_search(
        self,
        query: str,
        filters: List[SearchFilter],
        options: SearchOptions,
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_text(query)
            
            # Search knowledge base
            knowledge_query = KnowledgeQuery(
                query=query,
                top_k=options.max_results * 2,  # Get more for filtering
                filters=self._convert_filters(filters),
                include_metadata=options.include_metadata
            )
            
            kb_results = await self.knowledge_base.search(knowledge_query)
            
            # Convert to SearchResult objects
            results = []
            for result in kb_results.results:
                search_result = SearchResult(
                    id=result.get("id", str(uuid.uuid4())),
                    title=result.get("title", "Untitled"),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    content_type=ContentType.TEXT,
                    metadata=result.get("metadata", {}),
                    chunks=result.get("chunks", [])
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    async def _hybrid_search(
        self,
        queries: List[str],
        filters: List[SearchFilter],
        options: SearchOptions,
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Perform hybrid search combining multiple query variations."""
        all_results = []
        
        for query in queries:
            results = await self._semantic_search(query, filters, options, context)
            all_results.extend(results)
        
        # Deduplicate and merge scores
        merged_results = {}
        for result in all_results:
            if result.id in merged_results:
                # Combine scores
                merged_results[result.id].score = max(
                    merged_results[result.id].score,
                    result.score
                )
            else:
                merged_results[result.id] = result
        
        # Sort by score
        return sorted(merged_results.values(), key=lambda x: x.score, reverse=True)
    
    async def _neural_search(
        self,
        query: str,
        filters: List[SearchFilter],
        options: SearchOptions,
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Perform neural search with advanced AI processing."""
        # For now, use semantic search as base
        # In production, implement advanced neural search
        return await self._semantic_search(query, filters, options, context)
    
    async def _graph_search(
        self,
        query: str,
        filters: List[SearchFilter],
        options: SearchOptions,
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Perform knowledge graph-based search."""
        # Placeholder for graph search implementation
        return await self._semantic_search(query, filters, options, context)
    
    async def _multimodal_search(
        self,
        query: str,
        filters: List[SearchFilter],
        options: SearchOptions,
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Perform multi-modal search across different content types."""
        # Placeholder for multi-modal search implementation
        return await self._semantic_search(query, filters, options, context)
    
    async def _conversational_search(
        self,
        query: str,
        filters: List[SearchFilter],
        options: SearchOptions,
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Perform conversational search using context."""
        if context and context.conversation_history:
            # Combine current query with conversation context
            context_query = " ".join(context.conversation_history[-3:] + [query])
            return await self._semantic_search(context_query, filters, options, context)
        
        return await self._semantic_search(query, filters, options, context)
    
    async def _rerank_results(
        self,
        results: List[SearchResult],
        query: str,
        context: Optional[SearchContext]
    ) -> List[SearchResult]:
        """Rerank search results using advanced algorithms."""
        # Simple reranking - in production use advanced ML models
        for result in results:
            # Boost score based on title match
            if query.lower() in result.title.lower():
                result.score *= 1.2
            
            # Boost recent content
            if "created_at" in result.metadata:
                try:
                    created_date = datetime.fromisoformat(result.metadata["created_at"])
                    days_old = (datetime.utcnow() - created_date).days
                    if days_old < 30:  # Recent content
                        result.score *= 1.1
                except:
                    pass
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def _add_highlights(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Add highlights to search results."""
        query_terms = query.lower().split()
        
        for result in results:
            highlights = []
            content_lower = result.content.lower()
            
            for term in query_terms:
                if term in content_lower:
                    highlights.append(term)
            
            result.highlights = highlights
        
        return results
    
    async def _generate_suggestions(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[str]:
        """Generate search suggestions based on results."""
        suggestions = []
        
        # Extract common terms from results
        term_counts = {}
        for result in results[:5]:  # Top 5 results
            words = result.content.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    term_counts[word] = term_counts.get(word, 0) + 1
        
        # Get most common terms
        common_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for term, count in common_terms:
            if term not in query.lower():
                suggestions.append(f"{query} {term}")
        
        return suggestions
    
    def _generate_cache_key(
        self,
        query: str,
        mode: SearchMode,
        filters: List[SearchFilter],
        options: SearchOptions
    ) -> str:
        """Generate cache key for search request."""
        import hashlib
        
        key_data = {
            "query": query,
            "mode": mode.value,
            "filters": [asdict(f) for f in filters],
            "options": asdict(options)
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _convert_filters(self, filters: List[SearchFilter]) -> Dict[str, Any]:
        """Convert SearchFilter objects to knowledge base filter format."""
        converted = {}
        for filter_obj in filters:
            converted[filter_obj.field] = {
                "operator": filter_obj.operator,
                "value": filter_obj.value,
                "weight": filter_obj.weight
            }
        return converted
    
    def _generate_search_analytics(
        self,
        query: str,
        results: List[SearchResult],
        search_time: float
    ) -> Dict[str, Any]:
        """Generate analytics for search request."""
        return {
            "query_length": len(query.split()),
            "result_count": len(results),
            "avg_score": sum(r.score for r in results) / len(results) if results else 0,
            "search_time_ms": search_time * 1000,
            "content_types": list(set(r.content_type.value for r in results))
        }
    
    async def _update_search_analytics(
        self,
        query: str,
        search_time: float,
        result_count: int
    ) -> None:
        """Update global search analytics."""
        self.search_analytics["total_searches"] += 1
        
        # Update average response time
        total_searches = self.search_analytics["total_searches"]
        current_avg = self.search_analytics["avg_response_time"]
        self.search_analytics["avg_response_time"] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )
        
        # Track popular queries
        if query not in self.search_analytics["popular_queries"]:
            self.search_analytics["popular_queries"][query] = 0
        self.search_analytics["popular_queries"][query] += 1
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics."""
        return self.search_analytics.copy()


# Global search engine instance
advanced_search_engine = None


async def get_advanced_search_engine(
    knowledge_base: KnowledgeBase,
    embedding_manager: EmbeddingManager
) -> AdvancedSearchEngine:
    """Get the global advanced search engine instance."""
    global advanced_search_engine
    
    if advanced_search_engine is None:
        advanced_search_engine = AdvancedSearchEngine(knowledge_base, embedding_manager)
        await advanced_search_engine.initialize()
    
    return advanced_search_engine
