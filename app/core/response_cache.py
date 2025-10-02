"""
Advanced Response Caching System.

This module provides intelligent response caching for API endpoints
with automatic cache invalidation, compression, and performance optimization.
"""

import asyncio
import hashlib
import json
import zlib
from typing import Optional, Any, Dict, Callable, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import structlog

from fastapi import Request, Response
from fastapi.responses import JSONResponse
import redis.asyncio as redis

logger = structlog.get_logger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies."""
    NO_CACHE = "no_cache"              # Don't cache
    CACHE_ALWAYS = "cache_always"      # Always cache
    CACHE_SUCCESS = "cache_success"    # Only cache successful responses
    CACHE_CONDITIONAL = "cache_conditional"  # Cache based on conditions


class CacheInvalidationStrategy(str, Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"                        # Time-based expiration
    EVENT_BASED = "event_based"        # Invalidate on specific events
    PATTERN_BASED = "pattern_based"    # Invalidate by pattern matching
    MANUAL = "manual"                  # Manual invalidation only


@dataclass
class CacheConfig:
    """Configuration for response caching."""
    
    # Basic settings
    enabled: bool = True
    ttl: int = 300  # Default 5 minutes
    
    # Strategy
    strategy: CacheStrategy = CacheStrategy.CACHE_SUCCESS
    invalidation_strategy: CacheInvalidationStrategy = CacheInvalidationStrategy.TTL
    
    # Compression
    compress: bool = True
    compression_threshold: int = 1024  # Compress if response > 1KB
    
    # Cache key generation
    include_query_params: bool = True
    include_headers: List[str] = None  # Specific headers to include in cache key
    exclude_params: List[str] = None   # Query params to exclude from cache key
    
    # Performance
    max_cache_size_mb: int = 100
    eviction_policy: str = "lru"  # lru, lfu, fifo
    
    # Conditional caching
    cache_condition: Optional[Callable] = None  # Function to determine if should cache


class ResponseCache:
    """Advanced response caching system."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        default_config: Optional[CacheConfig] = None
    ):
        self.redis = redis_client
        self.default_config = default_config or CacheConfig()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
            "errors": 0
        }
        
        # Event-based invalidation patterns
        self.invalidation_patterns: Dict[str, List[str]] = {}
        
        logger.info("Response cache initialized")
    
    def cache_response(
        self,
        ttl: Optional[int] = None,
        strategy: Optional[CacheStrategy] = None,
        key_prefix: Optional[str] = None,
        include_user: bool = False,
        invalidate_on: Optional[List[str]] = None
    ):
        """
        Decorator for caching endpoint responses.
        
        Args:
            ttl: Time to live in seconds
            strategy: Cache strategy to use
            key_prefix: Prefix for cache key
            include_user: Include user ID in cache key
            invalidate_on: Events that should invalidate this cache
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from args/kwargs
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                if not request:
                    request = kwargs.get('request')
                
                if not request:
                    # No request object, can't cache
                    return await func(*args, **kwargs)
                
                # Generate cache key
                cache_key = await self._generate_cache_key(
                    request=request,
                    key_prefix=key_prefix or func.__name__,
                    include_user=include_user
                )
                
                # Try to get from cache
                cached_response = await self.get(cache_key)
                if cached_response is not None:
                    self.stats["hits"] += 1
                    logger.debug(f"Cache hit for {cache_key}")
                    return JSONResponse(
                        content=cached_response,
                        headers={"X-Cache": "HIT"}
                    )
                
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for {cache_key}")
                
                # Execute function
                response = await func(*args, **kwargs)
                
                # Determine if should cache
                should_cache = await self._should_cache_response(
                    response=response,
                    strategy=strategy or self.default_config.strategy
                )
                
                if should_cache:
                    # Extract response data
                    response_data = None
                    if isinstance(response, JSONResponse):
                        response_data = json.loads(response.body.decode())
                    elif isinstance(response, dict):
                        response_data = response
                    
                    if response_data:
                        # Cache the response
                        await self.set(
                            key=cache_key,
                            value=response_data,
                            ttl=ttl or self.default_config.ttl
                        )
                        
                        # Register invalidation patterns
                        if invalidate_on:
                            for event in invalidate_on:
                                if event not in self.invalidation_patterns:
                                    self.invalidation_patterns[event] = []
                                self.invalidation_patterns[event].append(cache_key)
                
                # Add cache header
                if isinstance(response, JSONResponse):
                    response.headers["X-Cache"] = "MISS"
                
                return response
            
            return wrapper
        return decorator
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            cached_data = await self.redis.get(key)
            if cached_data:
                # Decompress if needed
                if cached_data.startswith(b'\x78\x9c'):  # zlib magic number
                    cached_data = zlib.decompress(cached_data)
                
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}", key=key)
            self.stats["errors"] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            # Serialize value
            serialized = json.dumps(value).encode()
            
            # Compress if needed
            if self.default_config.compress and len(serialized) > self.default_config.compression_threshold:
                serialized = zlib.compress(serialized)
            
            # Set in Redis with TTL
            await self.redis.set(
                key,
                serialized,
                ex=ttl or self.default_config.ttl
            )
            
            self.stats["sets"] += 1
            logger.debug(f"Cached response for {key}", ttl=ttl)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}", key=key)
            self.stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            await self.redis.delete(key)
            self.stats["invalidations"] += 1
            logger.debug(f"Invalidated cache for {key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}", key=key)
            self.stats["errors"] += 1
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                self.stats["invalidations"] += deleted
                logger.info(f"Invalidated {deleted} cache entries matching {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache pattern delete error: {str(e)}", pattern=pattern)
            self.stats["errors"] += 1
            return 0
    
    async def invalidate_by_event(self, event: str):
        """Invalidate cache entries associated with an event."""
        if event in self.invalidation_patterns:
            keys = self.invalidation_patterns[event]
            for key in keys:
                await self.delete(key)
            
            logger.info(f"Invalidated {len(keys)} cache entries for event: {event}")
    
    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        try:
            await self.redis.flushdb()
            logger.warning("Cleared all cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    async def _generate_cache_key(
        self,
        request: Request,
        key_prefix: str,
        include_user: bool = False
    ) -> str:
        """Generate cache key from request."""
        # Start with prefix and path
        key_parts = [key_prefix, request.url.path]
        
        # Add query parameters
        if self.default_config.include_query_params:
            query_params = dict(request.query_params)
            
            # Exclude specified params
            if self.default_config.exclude_params:
                for param in self.default_config.exclude_params:
                    query_params.pop(param, None)
            
            if query_params:
                # Sort for consistency
                sorted_params = sorted(query_params.items())
                params_str = json.dumps(sorted_params)
                key_parts.append(params_str)
        
        # Add user ID if requested
        if include_user:
            user_id = getattr(request.state, 'user_id', None)
            if user_id:
                key_parts.append(f"user:{user_id}")
        
        # Add specific headers if configured
        if self.default_config.include_headers:
            for header in self.default_config.include_headers:
                header_value = request.headers.get(header)
                if header_value:
                    key_parts.append(f"{header}:{header_value}")
        
        # Generate hash
        key_str = ":".join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"cache:{key_prefix}:{key_hash}"
    
    async def _should_cache_response(
        self,
        response: Any,
        strategy: CacheStrategy
    ) -> bool:
        """Determine if response should be cached."""
        if strategy == CacheStrategy.NO_CACHE:
            return False
        
        if strategy == CacheStrategy.CACHE_ALWAYS:
            return True
        
        if strategy == CacheStrategy.CACHE_SUCCESS:
            # Check if response is successful
            if isinstance(response, JSONResponse):
                return 200 <= response.status_code < 300
            elif isinstance(response, dict):
                # Assume dict responses are successful
                return True
            return False
        
        if strategy == CacheStrategy.CACHE_CONDITIONAL:
            # Use custom condition if provided
            if self.default_config.cache_condition:
                return await self.default_config.cache_condition(response)
            return False
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "invalidations": self.stats["invalidations"],
            "errors": self.stats["errors"],
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        try:
            info = await self.redis.info("memory")
            stats = await self.redis.info("stats")
            
            return {
                "memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
                "memory_peak_mb": info.get("used_memory_peak", 0) / 1024 / 1024,
                "total_keys": await self.redis.dbsize(),
                "evicted_keys": stats.get("evicted_keys", 0),
                "expired_keys": stats.get("expired_keys", 0),
                "stats": self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {"error": str(e), "stats": self.get_stats()}


# Cache invalidation helpers

async def invalidate_agent_cache(cache: ResponseCache, agent_id: str):
    """Invalidate all cache entries related to an agent."""
    await cache.delete_pattern(f"cache:*agent*{agent_id}*")


async def invalidate_workflow_cache(cache: ResponseCache, workflow_id: str):
    """Invalidate all cache entries related to a workflow."""
    await cache.delete_pattern(f"cache:*workflow*{workflow_id}*")


async def invalidate_user_cache(cache: ResponseCache, user_id: str):
    """Invalidate all cache entries for a user."""
    await cache.delete_pattern(f"cache:*user:{user_id}*")

