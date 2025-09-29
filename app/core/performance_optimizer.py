"""
Performance optimization system for the Agentic AI platform.

This module provides caching, connection pooling, async optimization,
and other performance improvements.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog
from functools import wraps, lru_cache
import hashlib
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import redis
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cache_hits: int = 0
    cache_misses: int = 0
    database_queries: int = 0
    network_requests: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()
        self.current_memory = 0
        self.lock = threading.RLock()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        while (len(self.cache) > self.max_size or 
               self.current_memory > self.max_memory_bytes):
            if not self.access_order:
                break
            
            key = self.access_order.popleft()
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_memory -= entry.size_bytes
    
    def _update_access_order(self, key: str):
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return None
            
            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            self._update_access_order(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in cache."""
        with self.lock:
            # Calculate size
            size = self._calculate_size(value)
            
            # Create entry
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                size_bytes=size
            )
            
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory -= old_entry.size_bytes
                if key in self.access_order:
                    self.access_order.remove(key)
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory += size
            
            # Evict if necessary
            self._evict_lru()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_memory -= entry.size_bytes
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self.current_memory,
                "memory_usage_mb": self.current_memory / (1024 * 1024),
                "max_memory_bytes": self.max_memory_bytes,
                "total_accesses": total_accesses,
                "hit_rate": total_accesses / max(len(self.cache), 1)
            }


class RedisCache:
    """Redis-based cache with connection pooling."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 max_connections: int = 20):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.pool: Optional[redis.ConnectionPool] = None
        self.redis: Optional[redis.Redis] = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection."""
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True
            )
            self.redis = redis.Redis(connection_pool=self.pool)
            # Test connection
            self.redis.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.redis = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis:
            return None
        
        try:
            value = self.redis.get(key)
            if value is None:
                return None
            
            # Try to deserialize
            try:
                return pickle.loads(value)
            except:
                return value.decode('utf-8')
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in Redis cache."""
        if not self.redis:
            return
        
        try:
            # Serialize value
            try:
                serialized = pickle.dumps(value)
            except:
                serialized = str(value).encode('utf-8')
            
            if ttl_seconds:
                self.redis.setex(key, ttl_seconds, serialized)
            else:
                self.redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.redis:
            return False
        
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries."""
        if not self.redis:
            return
        
        try:
            self.redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class ConnectionPool:
    """HTTP connection pool for performance optimization."""
    
    def __init__(self, max_connections: int = 100, max_connections_per_host: int = 30):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.session: Optional[aiohttp.ClientSession] = None
        self.lock = asyncio.Lock()
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        async with self.lock:
            if self.session is None or self.session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_connections_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(total=30)
                
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={"User-Agent": "AgenticAI/1.0"}
                )
            
            return self.session
    
    async def close(self):
        """Close the connection pool."""
        if self.session and not self.session.closed:
            await self.session.close()


class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, enable_memory_cache: bool = True, enable_redis_cache: bool = False,
                 redis_url: str = "redis://localhost:6379"):
        self.enable_memory_cache = enable_memory_cache
        self.enable_redis_cache = enable_redis_cache
        
        # Initialize caches
        self.memory_cache = MemoryCache() if enable_memory_cache else None
        self.redis_cache = RedisCache(redis_url) if enable_redis_cache else None
        
        # Connection pool
        self.connection_pool = ConnectionPool()
        
        # Performance metrics
        self.metrics: List[PerformanceMetrics] = []
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key for function."""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cached(self, ttl_seconds: Optional[int] = None, cache_type: str = "memory"):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache = self._get_cache(cache_type)
                if not cache:
                    return await func(*args, **kwargs)
                
                key = self.cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = cache.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                cache.set(key, result, ttl_seconds)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache = self._get_cache(cache_type)
                if not cache:
                    return func(*args, **kwargs)
                
                key = self.cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = cache.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                cache.set(key, result, ttl_seconds)
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _get_cache(self, cache_type: str):
        """Get cache instance by type."""
        if cache_type == "memory" and self.memory_cache:
            return self.memory_cache
        elif cache_type == "redis" and self.redis_cache:
            return self.redis_cache
        return None
    
    def monitor_performance(self, operation_name: str):
        """Decorator for monitoring performance."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    duration_ms = (end_time - start_time) * 1000
                    memory_usage_mb = (end_memory - start_memory) / (1024 * 1024)
                    
                    # Record metrics
                    metrics = PerformanceMetrics(
                        operation=operation_name,
                        duration_ms=duration_ms,
                        memory_usage_mb=memory_usage_mb
                    )
                    self.metrics.append(metrics)
                    
                    # Store operation time
                    self.operation_times[operation_name].append(duration_ms)
                    if len(self.operation_times[operation_name]) > 1000:
                        self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    duration_ms = (end_time - start_time) * 1000
                    memory_usage_mb = (end_memory - start_memory) / (1024 * 1024)
                    
                    # Record metrics
                    metrics = PerformanceMetrics(
                        operation=operation_name,
                        duration_ms=duration_ms,
                        memory_usage_mb=memory_usage_mb
                    )
                    self.metrics.append(metrics)
                    
                    # Store operation time
                    self.operation_times[operation_name].append(duration_ms)
                    if len(self.operation_times[operation_name]) > 1000:
                        self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get HTTP session from connection pool."""
        return await self.connection_pool.get_session()
    
    def run_in_thread(self, func: Callable, *args, **kwargs):
        """Run function in thread pool."""
        return self.thread_pool.submit(func, *args, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "total_operations": len(self.metrics),
            "operation_times": {},
            "cache_stats": {}
        }
        
        # Operation times
        for operation, times in self.operation_times.items():
            if times:
                stats["operation_times"][operation] = {
                    "count": len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "avg_ms": sum(times) / len(times),
                    "p95_ms": self._percentile(times, 95),
                    "p99_ms": self._percentile(times, 99)
                }
        
        # Cache stats
        if self.memory_cache:
            stats["cache_stats"]["memory"] = self.memory_cache.get_stats()
        
        if self.redis_cache:
            stats["cache_stats"]["redis"] = {
                "enabled": True,
                "url": self.redis_cache.redis_url
            }
        
        return stats
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    async def optimize_database_queries(self, queries: List[str]) -> List[str]:
        """Optimize database queries."""
        optimized_queries = []
        
        for query in queries:
            # Basic query optimization
            optimized = query.strip()
            
            # Remove unnecessary whitespace
            optimized = re.sub(r'\s+', ' ', optimized)
            
            # Add query hints if needed
            if 'SELECT' in optimized.upper() and 'LIMIT' not in optimized.upper():
                optimized += ' LIMIT 1000'
            
            optimized_queries.append(optimized)
        
        return optimized_queries
    
    async def batch_operations(self, operations: List[Callable], batch_size: int = 10):
        """Batch operations for better performance."""
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    async def close(self):
        """Close performance optimizer."""
        if self.connection_pool:
            await self.connection_pool.close()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


# Global performance optimizer
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


# Convenience decorators
def cached(ttl_seconds: Optional[int] = None, cache_type: str = "memory"):
    """Caching decorator."""
    optimizer = get_performance_optimizer()
    return optimizer.cached(ttl_seconds, cache_type)


def monitor_performance(operation_name: str):
    """Performance monitoring decorator."""
    optimizer = get_performance_optimizer()
    return optimizer.monitor_performance(operation_name)


# Export all components
__all__ = [
    "CacheEntry", "PerformanceMetrics", "MemoryCache", "RedisCache",
    "ConnectionPool", "PerformanceOptimizer", "get_performance_optimizer",
    "cached", "monitor_performance"
]


