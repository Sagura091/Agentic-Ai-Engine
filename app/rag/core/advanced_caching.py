"""
Advanced Multi-Level Caching System for RAG 4.0.

Implements embedding cache, query result cache, collection metadata cache,
cache invalidation strategies, and distributed caching capabilities.
"""

import asyncio
import hashlib
import json
import pickle
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import structlog
from pathlib import Path
import aioredis
import numpy as np
from collections import OrderedDict, defaultdict

from .unified_rag_system import Document, KnowledgeQuery, KnowledgeResult
from .hybrid_search import SearchResult
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class CacheLevel(str, Enum):
    """Cache levels in the multi-level caching system."""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_DISTRIBUTED = "l3_distributed"


class CacheType(str, Enum):
    """Types of cached data."""
    EMBEDDINGS = "embeddings"
    QUERY_RESULTS = "query_results"
    COLLECTION_METADATA = "collection_metadata"
    AGENT_CONTEXT = "agent_context"
    SEARCH_RESULTS = "search_results"
    TFIDF_MODELS = "tfidf_models"


class InvalidationStrategy(str, Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least recently used
    KNOWLEDGE_FRESHNESS = "knowledge_freshness"  # Based on knowledge updates
    MANUAL = "manual"  # Manual invalidation
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class CacheConfig:
    """Configuration for advanced caching system."""
    # Memory cache settings
    l1_max_size_mb: int = 512
    l1_max_entries: int = 10000
    
    # Disk cache settings
    l2_max_size_gb: int = 10
    l2_cache_dir: str = "cache"
    
    # Distributed cache settings
    enable_distributed: bool = False
    redis_url: Optional[str] = None
    l3_ttl_seconds: int = 3600
    
    # Cache type specific TTLs
    embedding_ttl: int = 86400  # 24 hours
    query_result_ttl: int = 3600  # 1 hour
    metadata_ttl: int = 1800  # 30 minutes
    
    # Invalidation settings
    default_invalidation: InvalidationStrategy = InvalidationStrategy.LRU
    enable_cache_warming: bool = True
    warming_batch_size: int = 100
    
    # Performance settings
    compression_enabled: bool = True
    async_write_enabled: bool = True
    cache_hit_logging: bool = True


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0
        }
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired:
                    del self.cache[key]
                    self.stats["size_bytes"] -= entry.size_bytes
                    self.stats["misses"] += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.stats["hits"] += 1
                return entry
            else:
                self.stats["misses"] += 1
                return None
    
    async def put(self, key: str, entry: CacheEntry) -> None:
        """Put item in cache."""
        async with self._lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats["size_bytes"] -= old_entry.size_bytes
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            self.stats["size_bytes"] += entry.size_bytes
            
            # Evict if necessary
            while len(self.cache) > self.max_size:
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.stats["size_bytes"] -= oldest_entry.size_bytes
                self.stats["evictions"] += 1
    
    async def remove(self, key: str) -> bool:
        """Remove item from cache."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats["size_bytes"] -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all items from cache."""
        async with self._lock:
            self.cache.clear()
            self.stats["size_bytes"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "entries": len(self.cache),
            "hit_rate": hit_rate,
            "size_mb": self.stats["size_bytes"] / (1024 * 1024)
        }


class DiskCache:
    """Disk-based cache with compression and async I/O."""
    
    def __init__(self, cache_dir: str, max_size_gb: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.index_file = self.cache_dir / "cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "size_bytes": 0
        }
        
        # Load existing index
        asyncio.create_task(self._load_index())
    
    async def _load_index(self) -> None:
        """Load cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                
                # Calculate total size
                total_size = 0
                for entry_info in self.index.values():
                    total_size += entry_info.get("size_bytes", 0)
                self.stats["size_bytes"] = total_size
                
                logger.info(
                    "Disk cache index loaded",
                    entries=len(self.index),
                    size_mb=total_size / (1024 * 1024)
                )
        except Exception as e:
            logger.error("Failed to load disk cache index", error=str(e))
            self.index = {}
    
    async def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.error("Failed to save disk cache index", error=str(e))
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from disk cache."""
        async with self._lock:
            if key not in self.index:
                self.stats["misses"] += 1
                return None
            
            entry_info = self.index[key]
            file_path = self.cache_dir / entry_info["filename"]
            
            if not file_path.exists():
                # File missing, remove from index
                del self.index[key]
                self.stats["misses"] += 1
                return None
            
            try:
                # Check if expired
                created_at = datetime.fromisoformat(entry_info["created_at"])
                ttl = entry_info.get("ttl_seconds")
                if ttl and (datetime.now() - created_at).total_seconds() > ttl:
                    await self._remove_file(key, file_path)
                    self.stats["misses"] += 1
                    return None
                
                # Load from disk
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                entry = CacheEntry(
                    key=key,
                    value=data,
                    cache_type=CacheType(entry_info["cache_type"]),
                    created_at=created_at,
                    last_accessed=datetime.now(),
                    access_count=entry_info.get("access_count", 0) + 1,
                    ttl_seconds=ttl,
                    size_bytes=entry_info["size_bytes"],
                    metadata=entry_info.get("metadata", {})
                )
                
                # Update access info
                self.index[key]["access_count"] = entry.access_count
                self.index[key]["last_accessed"] = entry.last_accessed.isoformat()
                
                self.stats["hits"] += 1
                return entry
                
            except Exception as e:
                logger.error("Failed to load from disk cache", key=key, error=str(e))
                await self._remove_file(key, file_path)
                self.stats["misses"] += 1
                return None
    
    async def put(self, key: str, entry: CacheEntry) -> None:
        """Put item in disk cache."""
        async with self._lock:
            try:
                # Generate filename
                filename = f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
                file_path = self.cache_dir / filename
                
                # Remove existing entry if present
                if key in self.index:
                    old_file = self.cache_dir / self.index[key]["filename"]
                    if old_file.exists():
                        old_file.unlink()
                    self.stats["size_bytes"] -= self.index[key]["size_bytes"]
                
                # Write to disk
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.value, f)
                
                # Calculate file size
                file_size = file_path.stat().st_size
                
                # Update index
                self.index[key] = {
                    "filename": filename,
                    "cache_type": entry.cache_type.value,
                    "created_at": entry.created_at.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "access_count": entry.access_count,
                    "ttl_seconds": entry.ttl_seconds,
                    "size_bytes": file_size,
                    "metadata": entry.metadata
                }
                
                self.stats["size_bytes"] += file_size
                self.stats["writes"] += 1
                
                # Evict if necessary
                await self._evict_if_needed()
                
                # Save index periodically
                if self.stats["writes"] % 100 == 0:
                    await self._save_index()
                
            except Exception as e:
                logger.error("Failed to write to disk cache", key=key, error=str(e))
    
    async def _evict_if_needed(self) -> None:
        """Evict entries if cache size exceeds limit."""
        while self.stats["size_bytes"] > self.max_size_bytes and self.index:
            # Find least recently used entry
            lru_key = min(
                self.index.keys(),
                key=lambda k: self.index[k]["last_accessed"]
            )
            
            file_path = self.cache_dir / self.index[lru_key]["filename"]
            await self._remove_file(lru_key, file_path)
    
    async def _remove_file(self, key: str, file_path: Path) -> None:
        """Remove file and update index."""
        try:
            if file_path.exists():
                file_path.unlink()
            
            if key in self.index:
                self.stats["size_bytes"] -= self.index[key]["size_bytes"]
                del self.index[key]
        except Exception as e:
            logger.error("Failed to remove cache file", key=key, error=str(e))
    
    async def remove(self, key: str) -> bool:
        """Remove item from disk cache."""
        async with self._lock:
            if key in self.index:
                file_path = self.cache_dir / self.index[key]["filename"]
                await self._remove_file(key, file_path)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all items from disk cache."""
        async with self._lock:
            for entry_info in self.index.values():
                file_path = self.cache_dir / entry_info["filename"]
                if file_path.exists():
                    file_path.unlink()
            
            self.index.clear()
            self.stats["size_bytes"] = 0
            await self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "entries": len(self.index),
            "hit_rate": hit_rate,
            "size_gb": self.stats["size_bytes"] / (1024 * 1024 * 1024)
        }


class DistributedCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.redis: Optional[aioredis.Redis] = None
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "errors": 0
        }
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis for distributed caching")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self.redis = None
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from distributed cache."""
        if not self.redis:
            return None
        
        try:
            data = await self.redis.get(key)
            if data:
                entry_dict = pickle.loads(data)
                entry = CacheEntry(**entry_dict)
                self.stats["hits"] += 1
                return entry
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.error("Failed to get from distributed cache", key=key, error=str(e))
            self.stats["errors"] += 1
            return None
    
    async def put(self, key: str, entry: CacheEntry) -> None:
        """Put item in distributed cache."""
        if not self.redis:
            return
        
        try:
            # Convert entry to dict for serialization
            entry_dict = {
                "key": entry.key,
                "value": entry.value,
                "cache_type": entry.cache_type.value,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "access_count": entry.access_count,
                "ttl_seconds": entry.ttl_seconds,
                "size_bytes": entry.size_bytes,
                "metadata": entry.metadata
            }
            
            data = pickle.dumps(entry_dict)
            await self.redis.setex(key, self.ttl_seconds, data)
            self.stats["writes"] += 1
            
        except Exception as e:
            logger.error("Failed to put in distributed cache", key=key, error=str(e))
            self.stats["errors"] += 1
    
    async def remove(self, key: str) -> bool:
        """Remove item from distributed cache."""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Failed to remove from distributed cache", key=key, error=str(e))
            self.stats["errors"] += 1
            return False
    
    async def clear(self) -> None:
        """Clear all items from distributed cache."""
        if not self.redis:
            return
        
        try:
            await self.redis.flushdb()
        except Exception as e:
            logger.error("Failed to clear distributed cache", error=str(e))
            self.stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "connected": self.redis is not None
        }


class AdvancedCacheManager:
    """Advanced multi-level cache manager."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize cache levels
        self.l1_cache = LRUCache(self.config.l1_max_entries)
        self.l2_cache = DiskCache(
            self.config.l2_cache_dir,
            self.config.l2_max_size_gb
        )
        
        self.l3_cache: Optional[DistributedCache] = None
        if self.config.enable_distributed and self.config.redis_url:
            self.l3_cache = DistributedCache(
                self.config.redis_url,
                self.config.l3_ttl_seconds
            )
        
        # Cache warming
        self.warming_queue: Set[str] = set()
        self.warming_in_progress = False
        
        # Statistics
        self.global_stats = {
            "total_requests": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0
        }
    
    async def initialize(self) -> None:
        """Initialize cache manager."""
        if self.l3_cache:
            await self.l3_cache.connect()
        
        logger.info("Advanced cache manager initialized")
    
    async def get(self, key: str, cache_type: CacheType) -> Optional[Any]:
        """Get item from multi-level cache."""
        self.global_stats["total_requests"] += 1
        
        # Try L1 cache first
        entry = await self.l1_cache.get(key)
        if entry:
            self.global_stats["l1_hits"] += 1
            if self.config.cache_hit_logging:
                logger.debug("Cache hit L1", key=key, cache_type=cache_type.value)
            return entry.value
        
        # Try L2 cache
        entry = await self.l2_cache.get(key)
        if entry:
            self.global_stats["l2_hits"] += 1
            # Promote to L1
            await self.l1_cache.put(key, entry)
            if self.config.cache_hit_logging:
                logger.debug("Cache hit L2", key=key, cache_type=cache_type.value)
            return entry.value
        
        # Try L3 cache
        if self.l3_cache:
            entry = await self.l3_cache.get(key)
            if entry:
                self.global_stats["l3_hits"] += 1
                # Promote to L1 and L2
                await self.l1_cache.put(key, entry)
                await self.l2_cache.put(key, entry)
                if self.config.cache_hit_logging:
                    logger.debug("Cache hit L3", key=key, cache_type=cache_type.value)
                return entry.value
        
        # Cache miss
        self.global_stats["misses"] += 1
        return None
    
    async def put(
        self, 
        key: str, 
        value: Any, 
        cache_type: CacheType,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Put item in multi-level cache."""
        # Determine TTL
        if ttl_seconds is None:
            ttl_seconds = self._get_default_ttl(cache_type)
        
        # Calculate size
        size_bytes = len(pickle.dumps(value))
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            cache_type=cache_type,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=ttl_seconds,
            size_bytes=size_bytes
        )
        
        # Store in all levels
        await self.l1_cache.put(key, entry)
        
        if self.config.async_write_enabled:
            # Async write to L2 and L3
            asyncio.create_task(self.l2_cache.put(key, entry))
            if self.l3_cache:
                asyncio.create_task(self.l3_cache.put(key, entry))
        else:
            # Sync write
            await self.l2_cache.put(key, entry)
            if self.l3_cache:
                await self.l3_cache.put(key, entry)
    
    def _get_default_ttl(self, cache_type: CacheType) -> int:
        """Get default TTL for cache type."""
        ttl_mapping = {
            CacheType.EMBEDDINGS: self.config.embedding_ttl,
            CacheType.QUERY_RESULTS: self.config.query_result_ttl,
            CacheType.COLLECTION_METADATA: self.config.metadata_ttl,
            CacheType.AGENT_CONTEXT: self.config.query_result_ttl,
            CacheType.SEARCH_RESULTS: self.config.query_result_ttl,
            CacheType.TFIDF_MODELS: self.config.embedding_ttl
        }
        return ttl_mapping.get(cache_type, 3600)
    
    async def invalidate(
        self, 
        key: str, 
        strategy: InvalidationStrategy = InvalidationStrategy.MANUAL
    ) -> None:
        """Invalidate cache entry across all levels."""
        await self.l1_cache.remove(key)
        await self.l2_cache.remove(key)
        if self.l3_cache:
            await self.l3_cache.remove(key)
        
        logger.debug("Cache invalidated", key=key, strategy=strategy.value)
    
    async def invalidate_by_pattern(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern."""
        # This is a simplified implementation
        # In production, you'd want more sophisticated pattern matching
        if "*" in pattern:
            # Clear all caches for wildcard patterns
            await self.clear_all()
        else:
            await self.invalidate(pattern)
    
    async def warm_cache(self, keys: List[str], cache_type: CacheType) -> None:
        """Warm cache with frequently accessed keys."""
        if not self.config.enable_cache_warming or self.warming_in_progress:
            return
        
        self.warming_in_progress = True
        self.warming_queue.update(keys)
        
        try:
            # Process warming queue in batches
            while self.warming_queue:
                batch = []
                for _ in range(min(self.config.warming_batch_size, len(self.warming_queue))):
                    if self.warming_queue:
                        batch.append(self.warming_queue.pop())
                
                # Warm batch (implementation depends on data source)
                await self._warm_batch(batch, cache_type)
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
        
        finally:
            self.warming_in_progress = False
    
    async def _warm_batch(self, keys: List[str], cache_type: CacheType) -> None:
        """Warm a batch of cache keys."""
        # This would be implemented based on your data sources
        # For now, it's a placeholder
        logger.debug("Cache warming batch", keys=len(keys), cache_type=cache_type.value)
    
    async def clear_all(self) -> None:
        """Clear all cache levels."""
        await self.l1_cache.clear()
        await self.l2_cache.clear()
        if self.l3_cache:
            await self.l3_cache.clear()
        
        logger.info("All caches cleared")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all cache levels."""
        total_requests = self.global_stats["total_requests"]
        overall_hit_rate = 0.0
        if total_requests > 0:
            total_hits = (
                self.global_stats["l1_hits"] + 
                self.global_stats["l2_hits"] + 
                self.global_stats["l3_hits"]
            )
            overall_hit_rate = total_hits / total_requests
        
        stats = {
            "global": {
                **self.global_stats,
                "overall_hit_rate": overall_hit_rate
            },
            "l1_memory": self.l1_cache.get_stats(),
            "l2_disk": self.l2_cache.get_stats()
        }
        
        if self.l3_cache:
            stats["l3_distributed"] = self.l3_cache.get_stats()
        
        return stats

    async def get_cache_key(
        self,
        prefix: str,
        *args,
        **kwargs
    ) -> str:
        """Generate cache key from arguments."""
        # Create deterministic key from arguments
        key_parts = [prefix]

        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])

        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
            else:
                key_parts.append(f"{k}:{hashlib.md5(str(v).encode()).hexdigest()[:8]}")

        return ":".join(key_parts)

    async def cache_embeddings(
        self,
        text: str,
        model_name: str,
        embeddings: np.ndarray
    ) -> None:
        """Cache embeddings for text."""
        key = await self.get_cache_key("embeddings", text, model_name)
        await self.put(key, embeddings.tolist(), CacheType.EMBEDDINGS)

    async def get_cached_embeddings(
        self,
        text: str,
        model_name: str
    ) -> Optional[np.ndarray]:
        """Get cached embeddings for text."""
        key = await self.get_cache_key("embeddings", text, model_name)
        cached = await self.get(key, CacheType.EMBEDDINGS)
        return np.array(cached) if cached else None

    async def cache_search_results(
        self,
        query: str,
        collection: str,
        results: List[SearchResult]
    ) -> None:
        """Cache search results."""
        key = await self.get_cache_key("search", query, collection)
        # Convert results to serializable format
        serializable_results = [
            {
                "document": {
                    "id": r.document.id,
                    "title": r.document.title,
                    "content": r.document.content,
                    "metadata": r.document.metadata
                },
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
                "hybrid_score": r.hybrid_score,
                "confidence": r.confidence,
                "retrieval_method": r.retrieval_method,
                "metadata": r.metadata
            }
            for r in results
        ]
        await self.put(key, serializable_results, CacheType.SEARCH_RESULTS)

    async def get_cached_search_results(
        self,
        query: str,
        collection: str
    ) -> Optional[List[SearchResult]]:
        """Get cached search results."""
        key = await self.get_cache_key("search", query, collection)
        cached = await self.get(key, CacheType.SEARCH_RESULTS)

        if cached:
            # Convert back to SearchResult objects
            from .hybrid_search import SearchResult
            from .unified_rag_system import Document

            results = []
            for item in cached:
                doc_data = item["document"]
                document = Document(
                    id=doc_data["id"],
                    title=doc_data["title"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"]
                )

                result = SearchResult(
                    document=document,
                    dense_score=item["dense_score"],
                    sparse_score=item["sparse_score"],
                    hybrid_score=item["hybrid_score"],
                    confidence=item["confidence"],
                    retrieval_method=item["retrieval_method"],
                    metadata=item["metadata"]
                )
                results.append(result)

            return results

        return None
