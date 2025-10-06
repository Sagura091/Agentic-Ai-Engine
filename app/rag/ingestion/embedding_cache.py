"""
Embedding cache for RAG ingestion.

This module provides caching for embeddings to avoid recomputing
embeddings for duplicate or similar content.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CachedEmbedding:
    """Cached embedding entry."""
    content_sha: str
    embedding: List[float]
    model_name: str
    model_version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate (0.0-1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "hit_rate": self.get_hit_rate()
        }


class EmbeddingCache:
    """
    LRU cache for embeddings with TTL support.
    
    Features:
    - Cache by content_sha (exact match)
    - LRU eviction policy
    - TTL-based expiration
    - Model version tracking
    - Cache statistics
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_hours: int = 168,  # 7 days default
        enable_stats: bool = True
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl_hours: Time-to-live in hours
            enable_stats: Enable statistics tracking
        """
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.enable_stats = enable_stats
        
        # Cache storage: content_sha -> CachedEmbedding
        self._cache: Dict[str, CachedEmbedding] = {}
        
        # Statistics
        self.stats = CacheStats()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(
            "EmbeddingCache initialized",
            max_size=max_size,
            ttl_hours=ttl_hours
        )
    
    async def get(
        self,
        content_sha: str,
        model_name: str,
        model_version: str
    ) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            content_sha: Content SHA-256 hash
            model_name: Embedding model name
            model_version: Embedding model version
            
        Returns:
            Embedding if found and valid, None otherwise
        """
        async with self._lock:
            if self.enable_stats:
                self.stats.total_requests += 1
            
            # Check if entry exists
            entry = self._cache.get(content_sha)
            
            if entry is None:
                if self.enable_stats:
                    self.stats.cache_misses += 1
                return None
            
            # Check if model matches
            if entry.model_name != model_name or entry.model_version != model_version:
                # Model changed, invalidate cache entry
                del self._cache[content_sha]
                if self.enable_stats:
                    self.stats.cache_misses += 1
                
                logger.debug(
                    "Cache entry invalidated due to model mismatch",
                    content_sha=content_sha,
                    cached_model=f"{entry.model_name}:{entry.model_version}",
                    requested_model=f"{model_name}:{model_version}"
                )
                
                return None
            
            # Check if entry is expired
            if self._is_expired(entry):
                del self._cache[content_sha]
                if self.enable_stats:
                    self.stats.cache_misses += 1
                    self.stats.evictions += 1
                
                logger.debug(
                    "Cache entry expired",
                    content_sha=content_sha,
                    age_hours=(datetime.utcnow() - entry.created_at).total_seconds() / 3600
                )
                
                return None
            
            # Cache hit!
            entry.update_access()
            if self.enable_stats:
                self.stats.cache_hits += 1
            
            logger.debug(
                "Cache hit",
                content_sha=content_sha,
                access_count=entry.access_count
            )
            
            return entry.embedding
    
    async def put(
        self,
        content_sha: str,
        embedding: List[float],
        model_name: str,
        model_version: str
    ) -> None:
        """
        Put embedding in cache.
        
        Args:
            content_sha: Content SHA-256 hash
            embedding: Embedding vector
            model_name: Embedding model name
            model_version: Embedding model version
        """
        async with self._lock:
            # Check if cache is full
            if len(self._cache) >= self.max_size:
                # Evict LRU entry
                await self._evict_lru()
            
            # Create cache entry
            entry = CachedEmbedding(
                content_sha=content_sha,
                embedding=embedding,
                model_name=model_name,
                model_version=model_version
            )
            
            # Store in cache
            self._cache[content_sha] = entry
            
            logger.debug(
                "Embedding cached",
                content_sha=content_sha,
                cache_size=len(self._cache)
            )
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        
        # Remove it
        del self._cache[lru_key]
        
        if self.enable_stats:
            self.stats.evictions += 1
        
        logger.debug(
            "LRU eviction",
            evicted_key=lru_key,
            cache_size=len(self._cache)
        )
    
    def _is_expired(self, entry: CachedEmbedding) -> bool:
        """
        Check if cache entry is expired.
        
        Args:
            entry: Cache entry
            
        Returns:
            True if expired
        """
        age = datetime.utcnow() - entry.created_at
        return age > timedelta(hours=self.ttl_hours)
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired entries.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                del self._cache[key]
                if self.enable_stats:
                    self.stats.evictions += 1
            
            if expired_keys:
                logger.info(
                    "Expired cache entries cleaned up",
                    removed=len(expired_keys),
                    remaining=len(self._cache)
                )
            
            return len(expired_keys)
    
    async def warm_cache(
        self,
        embeddings: List[Tuple[str, List[float]]],
        model_name: str,
        model_version: str
    ) -> int:
        """
        Warm cache with pre-computed embeddings.
        
        Args:
            embeddings: List of (content_sha, embedding) tuples
            model_name: Embedding model name
            model_version: Embedding model version
            
        Returns:
            Number of embeddings cached
        """
        count = 0
        
        for content_sha, embedding in embeddings:
            await self.put(content_sha, embedding, model_name, model_version)
            count += 1
        
        logger.info(
            "Cache warmed",
            embeddings_added=count,
            cache_size=len(self._cache)
        )
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.stats.to_dict(),
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl_hours,
            "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0.0
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = CacheStats()
        logger.info("Cache statistics reset")
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            
            logger.info(
                "Cache cleared",
                entries_removed=count
            )
    
    def __len__(self) -> int:
        """Get cache size."""
        return len(self._cache)


# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None
_cache_lock = asyncio.Lock()


async def get_embedding_cache(
    max_size: int = 10000,
    ttl_hours: int = 168
) -> EmbeddingCache:
    """
    Get global embedding cache instance.
    
    Args:
        max_size: Maximum cache size
        ttl_hours: TTL in hours
        
    Returns:
        EmbeddingCache instance
    """
    global _embedding_cache
    
    if _embedding_cache is None:
        async with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache(
                    max_size=max_size,
                    ttl_hours=ttl_hours
                )
    
    return _embedding_cache

