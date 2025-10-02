# ⚡ Performance Optimization Guide

## Overview

The Agentic AI Engine includes comprehensive performance optimization features:

✅ **Response Caching** - Intelligent API response caching  
✅ **Pagination** - Efficient data pagination  
✅ **Circuit Breakers** - Protect against slow services  
✅ **Connection Pooling** - Optimized database connections  
✅ **Async Processing** - Non-blocking operations  

---

## Response Caching

### Quick Start

```python
from app.core.response_cache import ResponseCache
from fastapi import APIRouter, Depends
import redis.asyncio as redis

router = APIRouter()

# Initialize cache
redis_client = redis.from_url("redis://localhost:6379/0")
cache = ResponseCache(redis_client)

# Cache endpoint responses
@router.get("/agents")
@cache.cache_response(ttl=300, key_prefix="agents_list")
async def list_agents(request: Request):
    agents = await agent_service.get_all_agents()
    return {"agents": agents}
```

### Advanced Caching

```python
from app.core.response_cache import ResponseCache, CacheConfig, CacheStrategy

# Custom cache configuration
cache_config = CacheConfig(
    enabled=True,
    ttl=600,  # 10 minutes
    strategy=CacheStrategy.CACHE_SUCCESS,
    compress=True,
    compression_threshold=1024,  # Compress if > 1KB
    include_query_params=True,
    exclude_params=["timestamp", "nonce"]  # Don't include in cache key
)

cache = ResponseCache(redis_client, default_config=cache_config)

# Cache with user-specific keys
@router.get("/user/profile")
@cache.cache_response(
    ttl=300,
    key_prefix="user_profile",
    include_user=True  # Include user ID in cache key
)
async def get_user_profile(request: Request):
    user_id = request.state.user_id
    profile = await user_service.get_profile(user_id)
    return profile
```

### Cache Invalidation

```python
from app.core.response_cache import (
    invalidate_agent_cache,
    invalidate_workflow_cache,
    invalidate_user_cache
)

# Invalidate specific caches
@router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, data: AgentUpdate):
    agent = await agent_service.update(agent_id, data)
    
    # Invalidate agent cache
    await invalidate_agent_cache(cache, agent_id)
    
    return agent

# Invalidate by pattern
await cache.delete_pattern("cache:agents:*")

# Invalidate by event
@router.post("/agents")
async def create_agent(data: AgentCreate):
    agent = await agent_service.create(data)
    
    # Invalidate list cache
    await cache.invalidate_by_event("agent_created")
    
    return agent
```

### Cache Monitoring

```python
# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate_percent']}%")
print(f"Total requests: {stats['total_requests']}")
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")

# Get detailed cache info
info = await cache.get_cache_info()
print(f"Memory used: {info['memory_used_mb']:.2f} MB")
print(f"Total keys: {info['total_keys']}")
print(f"Evicted keys: {info['evicted_keys']}")
```

---

## Pagination

### Basic Pagination

```python
from app.core.pagination import AdvancedQueryParams, Paginator

@router.get("/agents")
async def list_agents(
    params: AdvancedQueryParams = Depends(),
    db: AsyncSession = Depends(get_db)
):
    # Create paginator
    paginator = Paginator(session=db, cache_backend=redis_client)
    
    # Build query
    query = select(Agent)
    
    # Paginate
    result = await paginator.paginate(
        query=query,
        params=params,
        model_class=Agent
    )
    
    return {
        "items": result.items,
        "metadata": result.metadata
    }
```

### Advanced Pagination with Filtering

```python
from app.core.pagination import AdvancedQueryParams, FilterOperator

@router.get("/agents")
async def list_agents(
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    agent_type: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    # Create query params
    params = AdvancedQueryParams(
        page=page,
        size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        filters={
            "type": {"operator": "eq", "value": agent_type} if agent_type else None,
            "status": {"operator": "eq", "value": status} if status else None
        }
    )
    
    # Build query with filters
    query = select(Agent)
    if agent_type:
        query = query.where(Agent.type == agent_type)
    if status:
        query = query.where(Agent.status == status)
    
    # Paginate
    paginator = Paginator(session=db, cache_backend=redis_client, cache_ttl=300)
    result = await paginator.paginate(query, params, Agent)
    
    return result
```

### Cursor-Based Pagination

```python
from app.core.pagination import CursorPaginator

@router.get("/agents/cursor")
async def list_agents_cursor(
    cursor: Optional[str] = None,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db)
):
    # Create cursor paginator
    paginator = CursorPaginator(session=db)
    
    # Create params
    params = AdvancedQueryParams(
        cursor=cursor,
        size=page_size,
        sort_by="id",
        sort_order="desc"
    )
    
    # Build query
    query = select(Agent)
    
    # Paginate
    result = await paginator.paginate(
        query=query,
        params=params,
        cursor_field="id",
        model_class=Agent
    )
    
    return {
        "items": result.items,
        "next_cursor": result.metadata.next_cursor,
        "has_next": result.metadata.has_next
    }
```

### In-Memory List Pagination

```python
from app.core.pagination import Paginator, AdvancedQueryParams

@router.get("/agents/memory")
async def list_agents_memory(
    page: int = 1,
    page_size: int = 20
):
    # Get all agents (from cache, memory, etc.)
    all_agents = await get_all_agents_from_cache()
    
    # Create params
    params = AdvancedQueryParams(page=page, size=page_size)
    
    # Paginate in-memory list
    paginator = Paginator(session=None)
    result = await paginator.paginate_list(all_agents, params)
    
    return result
```

---

## Database Optimization

### Connection Pooling

```env
# Optimized pool settings
DATABASE__POOL_SIZE=50
DATABASE__POOL_MAX_OVERFLOW=20
DATABASE__POOL_TIMEOUT=30
DATABASE__POOL_RECYCLE=3600
```

### Query Optimization

```python
from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload

# Eager loading to avoid N+1 queries
@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    query = (
        select(Agent)
        .options(
            selectinload(Agent.tools),
            selectinload(Agent.workflows),
            joinedload(Agent.user)
        )
        .where(Agent.id == agent_id)
    )
    
    result = await db.execute(query)
    agent = result.scalar_one_or_none()
    
    return agent
```

### Batch Operations

```python
# Batch insert
@router.post("/agents/batch")
async def create_agents_batch(
    agents_data: List[AgentCreate],
    db: AsyncSession = Depends(get_db)
):
    # Create all agents in one transaction
    agents = [Agent(**data.dict()) for data in agents_data]
    db.add_all(agents)
    await db.commit()
    
    return {"created": len(agents)}

# Batch update
@router.put("/agents/batch")
async def update_agents_batch(
    updates: List[AgentUpdate],
    db: AsyncSession = Depends(get_db)
):
    # Update in batch
    for update in updates:
        await db.execute(
            update(Agent)
            .where(Agent.id == update.id)
            .values(**update.dict(exclude_unset=True))
        )
    
    await db.commit()
    return {"updated": len(updates)}
```

---

## Async Processing

### Background Tasks

```python
from fastapi import BackgroundTasks

async def process_document_async(doc_id: str):
    """Process document in background."""
    await document_processor.process(doc_id)
    await cache.invalidate_by_event(f"document_{doc_id}_processed")

@router.post("/documents/{doc_id}/process")
async def process_document(
    doc_id: str,
    background_tasks: BackgroundTasks
):
    # Add to background tasks
    background_tasks.add_task(process_document_async, doc_id)
    
    return {"status": "processing", "doc_id": doc_id}
```

### Task Queue

```python
from app.core.task_queue import task_queue

@task_queue.task
async def heavy_computation(data: dict):
    """Heavy computation task."""
    result = await perform_heavy_computation(data)
    return result

@router.post("/compute")
async def start_computation(data: dict):
    # Queue task
    task_id = await heavy_computation.delay(data)
    
    return {"task_id": task_id, "status": "queued"}

@router.get("/compute/{task_id}")
async def get_computation_result(task_id: str):
    # Get task result
    result = await task_queue.get_result(task_id)
    
    return {"task_id": task_id, "result": result}
```

---

## Circuit Breakers for Performance

### Timeout Protection

```python
from app.core.circuit_breaker import circuit_breaker_manager, CircuitBreakerConfig

# Configure with timeout
config = CircuitBreakerConfig(
    call_timeout=5.0,  # 5 second timeout
    failure_threshold=3,
    failure_timeout=30
)

@router.get("/external-data")
async def get_external_data():
    try:
        # Protected call with timeout
        data = await circuit_breaker_manager.call(
            name="external_api",
            func=external_api.fetch_data,
            config=config,
            fallback=lambda: {"status": "cached", "data": get_cached_data()}
        )
        return data
    except CircuitBreakerOpenError:
        # Circuit is open, return cached data
        return {"status": "cached", "data": get_cached_data()}
```

---

## Performance Monitoring

### Endpoint Performance

```python
from fastapi import Request
import time

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 1.0:
        logger.warning(
            "Slow request",
            path=request.url.path,
            method=request.method,
            process_time=process_time
        )
    
    return response
```

### Database Query Performance

```python
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    
    # Log slow queries
    if total > 0.5:
        logger.warning(
            "Slow query",
            query=statement,
            duration=total
        )
```

---

## Best Practices

### 1. Cache Frequently Accessed Data

```python
# Cache static/semi-static data
@cache.cache_response(ttl=3600)  # 1 hour
async def get_agent_templates():
    return await template_service.get_all()
```

### 2. Use Pagination for Large Datasets

```python
# Always paginate list endpoints
@router.get("/agents")
async def list_agents(params: AdvancedQueryParams = Depends()):
    # Paginate instead of returning all
    return await paginator.paginate(query, params, Agent)
```

### 3. Implement Circuit Breakers for External Services

```python
# Protect external service calls
async def call_external_service():
    return await circuit_breaker_manager.call(
        name="external_service",
        func=service.call,
        fallback=get_cached_response
    )
```

### 4. Use Async Operations

```python
# Use async/await for I/O operations
async def process_multiple_documents(doc_ids: List[str]):
    # Process concurrently
    tasks = [process_document(doc_id) for doc_id in doc_ids]
    results = await asyncio.gather(*tasks)
    return results
```

### 5. Monitor Performance

```python
# Regular performance monitoring
@router.get("/performance/stats")
async def get_performance_stats():
    return {
        "cache": cache.get_stats(),
        "circuit_breakers": circuit_breaker_manager.get_all_metrics(),
        "database": await get_db_stats()
    }
```

---

## Performance Tuning

### Configuration

```env
# Performance settings
PERFORMANCE__ASYNC_WORKER_COUNT=16
PERFORMANCE__MAX_CONCURRENT_AGENTS=100
PERFORMANCE__WORKER_CONNECTIONS=2000
PERFORMANCE__ENABLE_PARALLEL_PROCESSING=true
PERFORMANCE__DOCUMENT_PROCESSING_WORKERS=8

# Database
DATABASE__POOL_SIZE=50
DATABASE__POOL_MAX_OVERFLOW=20

# Redis
REDIS__POOL_SIZE=10

# RAG
RAG__CHROMA_CONNECTION_POOL_SIZE=50
RAG__CHROMA_MAX_MEMORY_MB=4096
```

### Monitoring Endpoints

```python
@router.get("/health/performance")
async def performance_health():
    return {
        "cache_hit_rate": cache.get_stats()["hit_rate_percent"],
        "circuit_breakers": {
            name: stats["state"]
            for name, stats in circuit_breaker_manager.get_all_metrics().items()
        },
        "database_pool": await get_db_pool_stats()
    }
```

---

## Summary

Performance optimization features:

✅ **Response Caching** - Up to 90% faster for cached responses  
✅ **Pagination** - Handle millions of records efficiently  
✅ **Circuit Breakers** - Protect against slow/failing services  
✅ **Connection Pooling** - Optimized database connections  
✅ **Async Processing** - Non-blocking operations  
✅ **Monitoring** - Comprehensive performance metrics  

**Result**: Faster responses, better scalability, improved reliability!

