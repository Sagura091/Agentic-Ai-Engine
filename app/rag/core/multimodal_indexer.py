"""
Multimodal Indexer.

This module provides specialized indexing for different content types (text, code,
tables, images, etc.). It enables content-type-specific search, retrieval, and
ranking strategies.

Key Features:
- Separate indexes for different content types
- Content-type-specific embeddings
- Specialized ranking algorithms
- Multi-index search
- Type-aware result fusion
- Performance optimization per type

Author: Agentic AI System
Purpose: Multimodal content indexing and retrieval
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


class ContentType(str, Enum):
    """Content types for multimodal indexing."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class RankingStrategy(str, Enum):
    """Ranking strategies for different content types."""
    SEMANTIC = "semantic"  # Semantic similarity
    KEYWORD = "keyword"  # Keyword matching
    HYBRID = "hybrid"  # Hybrid semantic + keyword
    STRUCTURAL = "structural"  # Structure-based
    TEMPORAL = "temporal"  # Time-based


@dataclass
class ContentTypeConfig:
    """Configuration for a content type."""
    content_type: ContentType
    ranking_strategy: RankingStrategy
    boost_factor: float = 1.0  # Boost factor for this content type
    enable_specialized_embedding: bool = False
    embedding_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalSearchResult:
    """Result from multimodal search."""
    chunk_id: str
    content_type: ContentType
    score: float
    boosted_score: float
    ranking_strategy: RankingStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentTypeIndex:
    """
    Index for a specific content type.
    
    Maintains a separate index for each content type with specialized
    ranking and retrieval strategies.
    """
    
    def __init__(self, config: ContentTypeConfig):
        """
        Initialize content type index.
        
        Args:
            config: Content type configuration
        """
        self.config = config
        self.content_type = config.content_type
        
        # Chunk storage
        self._chunks: Dict[str, Dict[str, Any]] = {}  # chunk_id -> chunk data
        
        # Inverted index for keyword search
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)  # term -> chunk_ids
        
        # Metrics
        self._metrics = {
            'total_chunks': 0,
            'total_searches': 0,
            'avg_search_time_ms': 0.0
        }
        
        logger.debug(f"ContentTypeIndex initialized: {self.content_type.value}")
    
    def add_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a chunk to the index.
        
        Args:
            chunk_id: Chunk ID
            content: Chunk content
            embedding: Optional embedding vector
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        try:
            self._chunks[chunk_id] = {
                'content': content,
                'embedding': embedding,
                'metadata': metadata or {},
                'added_at': datetime.utcnow()
            }
            
            # Build keyword index
            if self.config.ranking_strategy in [RankingStrategy.KEYWORD, RankingStrategy.HYBRID]:
                terms = self._tokenize(content)
                for term in terms:
                    self._keyword_index[term].add(chunk_id)
            
            self._metrics['total_chunks'] += 1
            
            logger.debug(f"Chunk added to {self.content_type.value} index: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunk {chunk_id} to {self.content_type.value} index: {e}")
            return False
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the index.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            True if successful
        """
        try:
            if chunk_id not in self._chunks:
                return False
            
            # Remove from keyword index
            for term, chunk_ids in self._keyword_index.items():
                chunk_ids.discard(chunk_id)
            
            # Remove chunk
            del self._chunks[chunk_id]
            
            self._metrics['total_chunks'] -= 1
            
            logger.debug(f"Chunk removed from {self.content_type.value} index: {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove chunk {chunk_id} from {self.content_type.value} index: {e}")
            return False
    
    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10
    ) -> List[MultimodalSearchResult]:
        """
        Search the index.
        
        Args:
            query: Search query
            query_embedding: Optional query embedding
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        import time
        start_time = time.time()
        
        try:
            results = []
            
            if self.config.ranking_strategy == RankingStrategy.SEMANTIC:
                results = self._semantic_search(query_embedding, top_k)
            elif self.config.ranking_strategy == RankingStrategy.KEYWORD:
                results = self._keyword_search(query, top_k)
            elif self.config.ranking_strategy == RankingStrategy.HYBRID:
                results = self._hybrid_search(query, query_embedding, top_k)
            elif self.config.ranking_strategy == RankingStrategy.STRUCTURAL:
                results = self._structural_search(query, top_k)
            elif self.config.ranking_strategy == RankingStrategy.TEMPORAL:
                results = self._temporal_search(query, top_k)
            
            # Apply boost factor
            for result in results:
                result.boosted_score = result.score * self.config.boost_factor
            
            # Update metrics
            self._metrics['total_searches'] += 1
            search_time = (time.time() - start_time) * 1000
            self._metrics['avg_search_time_ms'] = (
                (self._metrics['avg_search_time_ms'] * (self._metrics['total_searches'] - 1) + search_time) /
                self._metrics['total_searches']
            )
            
            logger.debug(
                f"Search completed in {self.content_type.value} index",
                results=len(results),
                time_ms=search_time
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed in {self.content_type.value} index: {e}")
            return []
    
    def _semantic_search(
        self,
        query_embedding: Optional[List[float]],
        top_k: int
    ) -> List[MultimodalSearchResult]:
        """Semantic search using embeddings."""
        if not query_embedding:
            return []
        
        results = []
        
        for chunk_id, chunk_data in self._chunks.items():
            embedding = chunk_data.get('embedding')
            if not embedding:
                continue
            
            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, embedding)
            
            results.append(MultimodalSearchResult(
                chunk_id=chunk_id,
                content_type=self.content_type,
                score=score,
                boosted_score=score,
                ranking_strategy=RankingStrategy.SEMANTIC,
                metadata=chunk_data.get('metadata', {})
            ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[MultimodalSearchResult]:
        """Keyword search using inverted index."""
        query_terms = self._tokenize(query)
        
        # Find matching chunks
        chunk_scores: Dict[str, float] = defaultdict(float)
        
        for term in query_terms:
            if term in self._keyword_index:
                for chunk_id in self._keyword_index[term]:
                    chunk_scores[chunk_id] += 1.0
        
        # Normalize scores
        max_score = max(chunk_scores.values()) if chunk_scores else 1.0
        
        results = []
        for chunk_id, score in chunk_scores.items():
            normalized_score = score / max_score
            
            results.append(MultimodalSearchResult(
                chunk_id=chunk_id,
                content_type=self.content_type,
                score=normalized_score,
                boosted_score=normalized_score,
                ranking_strategy=RankingStrategy.KEYWORD,
                metadata=self._chunks[chunk_id].get('metadata', {})
            ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:top_k]
    
    def _hybrid_search(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        top_k: int
    ) -> List[MultimodalSearchResult]:
        """Hybrid search combining semantic and keyword."""
        # Get semantic results
        semantic_results = self._semantic_search(query_embedding, top_k * 2)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Merge results
        chunk_scores: Dict[str, float] = {}
        
        for result in semantic_results:
            chunk_scores[result.chunk_id] = result.score * 0.7  # 70% weight for semantic
        
        for result in keyword_results:
            if result.chunk_id in chunk_scores:
                chunk_scores[result.chunk_id] += result.score * 0.3  # 30% weight for keyword
            else:
                chunk_scores[result.chunk_id] = result.score * 0.3
        
        # Create results
        results = []
        for chunk_id, score in chunk_scores.items():
            results.append(MultimodalSearchResult(
                chunk_id=chunk_id,
                content_type=self.content_type,
                score=score,
                boosted_score=score,
                ranking_strategy=RankingStrategy.HYBRID,
                metadata=self._chunks[chunk_id].get('metadata', {})
            ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]

    def _structural_search(
        self,
        query: str,
        top_k: int
    ) -> List[MultimodalSearchResult]:
        """Structural search based on document structure."""
        # For structural search, prioritize chunks with specific structural properties
        results = []

        for chunk_id, chunk_data in self._chunks.items():
            metadata = chunk_data.get('metadata', {})
            score = 0.0

            # Boost based on structural properties
            if metadata.get('section_path'):
                score += 0.3
            if metadata.get('page_number'):
                score += 0.2
            if metadata.get('heading_level'):
                score += 0.5

            if score > 0:
                results.append(MultimodalSearchResult(
                    chunk_id=chunk_id,
                    content_type=self.content_type,
                    score=score,
                    boosted_score=score,
                    ranking_strategy=RankingStrategy.STRUCTURAL,
                    metadata=metadata
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]

    def _temporal_search(
        self,
        query: str,
        top_k: int
    ) -> List[MultimodalSearchResult]:
        """Temporal search based on recency."""
        results = []

        # Get all chunks with timestamps
        timestamped_chunks = [
            (chunk_id, chunk_data)
            for chunk_id, chunk_data in self._chunks.items()
            if chunk_data.get('added_at')
        ]

        # Sort by timestamp descending (most recent first)
        timestamped_chunks.sort(
            key=lambda x: x[1]['added_at'],
            reverse=True
        )

        # Create results with time-decay scoring
        now = datetime.utcnow()
        for chunk_id, chunk_data in timestamped_chunks[:top_k]:
            added_at = chunk_data['added_at']
            age_hours = (now - added_at).total_seconds() / 3600

            # Time decay: score = 1 / (1 + age_hours/24)
            score = 1.0 / (1.0 + age_hours / 24.0)

            results.append(MultimodalSearchResult(
                chunk_id=chunk_id,
                content_type=self.content_type,
                score=score,
                boosted_score=score,
                ranking_strategy=RankingStrategy.TEMPORAL,
                metadata=chunk_data.get('metadata', {})
            ))

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def get_metrics(self) -> Dict[str, Any]:
        """Get index metrics."""
        return {
            'content_type': self.content_type.value,
            'total_chunks': self._metrics['total_chunks'],
            'total_searches': self._metrics['total_searches'],
            'avg_search_time_ms': self._metrics['avg_search_time_ms'],
            'ranking_strategy': self.config.ranking_strategy.value,
            'boost_factor': self.config.boost_factor
        }


class MultimodalIndexer:
    """
    Production-grade multimodal indexer.

    Manages multiple content-type-specific indexes and provides unified
    search interface with type-aware ranking and fusion.

    Features:
    - Separate indexes per content type
    - Content-type-specific ranking
    - Multi-index search
    - Type-aware result fusion
    - Configurable boost factors
    - Performance optimization
    """

    def __init__(self):
        """Initialize multimodal indexer."""
        # Content type indexes
        self._indexes: Dict[ContentType, ContentTypeIndex] = {}

        # Default configurations
        self._default_configs = {
            ContentType.TEXT: ContentTypeConfig(
                content_type=ContentType.TEXT,
                ranking_strategy=RankingStrategy.HYBRID,
                boost_factor=1.0
            ),
            ContentType.CODE: ContentTypeConfig(
                content_type=ContentType.CODE,
                ranking_strategy=RankingStrategy.KEYWORD,
                boost_factor=1.2
            ),
            ContentType.TABLE: ContentTypeConfig(
                content_type=ContentType.TABLE,
                ranking_strategy=RankingStrategy.STRUCTURAL,
                boost_factor=1.3
            ),
            ContentType.LIST: ContentTypeConfig(
                content_type=ContentType.LIST,
                ranking_strategy=RankingStrategy.KEYWORD,
                boost_factor=1.0
            ),
            ContentType.HEADING: ContentTypeConfig(
                content_type=ContentType.HEADING,
                ranking_strategy=RankingStrategy.STRUCTURAL,
                boost_factor=1.5
            ),
            ContentType.IMAGE: ContentTypeConfig(
                content_type=ContentType.IMAGE,
                ranking_strategy=RankingStrategy.SEMANTIC,
                boost_factor=0.8
            ),
            ContentType.AUDIO: ContentTypeConfig(
                content_type=ContentType.AUDIO,
                ranking_strategy=RankingStrategy.SEMANTIC,
                boost_factor=0.9
            ),
            ContentType.VIDEO: ContentTypeConfig(
                content_type=ContentType.VIDEO,
                ranking_strategy=RankingStrategy.SEMANTIC,
                boost_factor=0.9
            ),
            ContentType.METADATA: ContentTypeConfig(
                content_type=ContentType.METADATA,
                ranking_strategy=RankingStrategy.KEYWORD,
                boost_factor=0.7
            ),
            ContentType.UNKNOWN: ContentTypeConfig(
                content_type=ContentType.UNKNOWN,
                ranking_strategy=RankingStrategy.HYBRID,
                boost_factor=0.5
            )
        }

        # Initialize indexes with default configs
        for content_type, config in self._default_configs.items():
            self._indexes[content_type] = ContentTypeIndex(config)

        # Metrics
        self._metrics = {
            'total_chunks': 0,
            'total_searches': 0,
            'chunks_by_type': defaultdict(int)
        }

        logger.info("MultimodalIndexer initialized with all content types")

    async def add_chunk(
        self,
        chunk_id: str,
        content: str,
        content_type: ContentType,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a chunk to the appropriate content type index.

        Args:
            chunk_id: Chunk ID
            content: Chunk content
            content_type: Content type
            embedding: Optional embedding vector
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            if content_type not in self._indexes:
                logger.warning(f"Unknown content type: {content_type}, using UNKNOWN index")
                content_type = ContentType.UNKNOWN

            index = self._indexes[content_type]
            success = index.add_chunk(chunk_id, content, embedding, metadata)

            if success:
                self._metrics['total_chunks'] += 1
                self._metrics['chunks_by_type'][content_type.value] += 1

            return success

        except Exception as e:
            logger.error(f"Failed to add chunk {chunk_id}: {e}")
            return False

    async def remove_chunk(
        self,
        chunk_id: str,
        content_type: ContentType
    ) -> bool:
        """
        Remove a chunk from the appropriate content type index.

        Args:
            chunk_id: Chunk ID
            content_type: Content type

        Returns:
            True if successful
        """
        try:
            if content_type not in self._indexes:
                return False

            index = self._indexes[content_type]
            success = index.remove_chunk(chunk_id)

            if success:
                self._metrics['total_chunks'] -= 1
                self._metrics['chunks_by_type'][content_type.value] -= 1

            return success

        except Exception as e:
            logger.error(f"Failed to remove chunk {chunk_id}: {e}")
            return False

    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        content_types: Optional[List[ContentType]] = None,
        top_k: int = 10
    ) -> List[MultimodalSearchResult]:
        """
        Search across multiple content type indexes.

        Args:
            query: Search query
            query_embedding: Optional query embedding
            content_types: Optional list of content types to search (default: all)
            top_k: Number of results to return

        Returns:
            List of search results
        """
        try:
            # Determine which indexes to search
            if content_types:
                indexes_to_search = [
                    (ct, self._indexes[ct])
                    for ct in content_types
                    if ct in self._indexes
                ]
            else:
                indexes_to_search = list(self._indexes.items())

            # Search each index
            all_results = []
            for content_type, index in indexes_to_search:
                results = index.search(query, query_embedding, top_k * 2)
                all_results.extend(results)

            # Sort by boosted score
            all_results.sort(key=lambda r: r.boosted_score, reverse=True)

            # Return top-k
            final_results = all_results[:top_k]

            self._metrics['total_searches'] += 1

            logger.debug(
                f"Multimodal search completed",
                indexes_searched=len(indexes_to_search),
                total_results=len(all_results),
                final_results=len(final_results)
            )

            return final_results

        except Exception as e:
            logger.error(f"Multimodal search failed: {e}")
            return []

    async def search_by_type(
        self,
        query: str,
        content_type: ContentType,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10
    ) -> List[MultimodalSearchResult]:
        """
        Search a specific content type index.

        Args:
            query: Search query
            content_type: Content type to search
            query_embedding: Optional query embedding
            top_k: Number of results to return

        Returns:
            List of search results
        """
        try:
            if content_type not in self._indexes:
                logger.warning(f"Unknown content type: {content_type}")
                return []

            index = self._indexes[content_type]
            results = index.search(query, query_embedding, top_k)

            return results

        except Exception as e:
            logger.error(f"Search by type failed for {content_type}: {e}")
            return []

    def update_config(
        self,
        content_type: ContentType,
        config: ContentTypeConfig
    ) -> bool:
        """
        Update configuration for a content type.

        Args:
            content_type: Content type
            config: New configuration

        Returns:
            True if successful
        """
        try:
            if content_type in self._indexes:
                # Create new index with updated config
                old_index = self._indexes[content_type]
                new_index = ContentTypeIndex(config)

                # Copy chunks to new index
                for chunk_id, chunk_data in old_index._chunks.items():
                    new_index.add_chunk(
                        chunk_id=chunk_id,
                        content=chunk_data['content'],
                        embedding=chunk_data.get('embedding'),
                        metadata=chunk_data.get('metadata')
                    )

                # Replace index
                self._indexes[content_type] = new_index

                logger.info(f"Config updated for {content_type.value}")
                return True
            else:
                # Create new index
                self._indexes[content_type] = ContentTypeIndex(config)
                logger.info(f"New index created for {content_type.value}")
                return True

        except Exception as e:
            logger.error(f"Failed to update config for {content_type}: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get multimodal indexer metrics.

        Returns:
            Metrics dictionary
        """
        index_metrics = {}
        for content_type, index in self._indexes.items():
            index_metrics[content_type.value] = index.get_metrics()

        return {
            **self._metrics,
            'index_metrics': index_metrics
        }


# Global singleton
_multimodal_indexer: Optional[MultimodalIndexer] = None
_indexer_lock = asyncio.Lock()


async def get_multimodal_indexer() -> MultimodalIndexer:
    """
    Get or create multimodal indexer singleton.

    Returns:
        MultimodalIndexer instance
    """
    global _multimodal_indexer

    async with _indexer_lock:
        if _multimodal_indexer is None:
            _multimodal_indexer = MultimodalIndexer()
            logger.info("MultimodalIndexer singleton created")

        return _multimodal_indexer
