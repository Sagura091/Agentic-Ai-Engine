# 🛡️ Error Handling & Recovery Guide

## Overview

The Agentic AI Engine features a **revolutionary intelligent error handling system** with:

✅ **Automatic Error Analysis** - AI-powered root cause identification
✅ **Smart Recovery** - Multiple recovery strategies
✅ **Detailed Error Codes** - Structured error responses
✅ **Learning System** - Improves over time

> **⚠️ NOTE:** Circuit Breakers are documented below but **not currently implemented** in the codebase. The circuit breaker module (`app/core/circuit_breaker.py`) was removed as it was never integrated. If you need circuit breaker functionality, you'll need to implement it or use a third-party library like `pybreaker`.

---

## Error Handling System

### Components

1. **Intelligent Error Handler** - Analyzes and recovers from errors
2. **Error Codes** - Structured error classification
3. **Recovery Strategies** - Multiple recovery approaches

---

## Using the Error Handler

### Basic Usage

```python
from app.core.error_handling import error_handler

try:
    # Your code here
    result = await some_operation()
except Exception as e:
    # Handle error with intelligent analysis
    error_response = await error_handler.handle_error(
        error=e,
        context={
            "operation": "some_operation",
            "user_id": user_id,
            "additional_info": "..."
        },
        request_id=request_id,
        auto_recover=True  # Attempt automatic recovery
    )
    return error_response
```

### In FastAPI Endpoints

```python
from fastapi import APIRouter, HTTPException
from app.core.error_handling import error_handler

router = APIRouter()

@router.post("/agents")
async def create_agent(request: Request, agent_data: AgentCreate):
    try:
        agent = await agent_service.create_agent(agent_data)
        return {"success": True, "agent": agent}
        
    except Exception as e:
        # Intelligent error handling
        error_response = await error_handler.handle_error(
            error=e,
            context={
                "operation": "create_agent",
                "agent_type": agent_data.type,
                "endpoint": "/agents"
            },
            request_id=request.state.request_id,
            auto_recover=True
        )
        raise HTTPException(
            status_code=500,
            detail=error_response.dict()
        )
```

---

## Recovery Strategies

The system supports multiple recovery strategies:

### 1. Retry Strategy
Automatically retries failed operations with exponential backoff.

```python
context = {
    "operation": lambda: await external_service.call(),
    "max_retries": 3,
    "retry_delay": 1.0  # Initial delay in seconds
}

error_response = await error_handler.handle_error(
    error=e,
    context=context,
    auto_recover=True
)
```

### 2. Fallback Strategy
Uses fallback data when operation fails.

```python
context = {
    "fallback_data": {
        "status": "degraded",
        "message": "Using cached data"
    }
}
```

### 3. Circuit Breaker Strategy
Opens circuit after repeated failures to prevent cascading issues.

```python
context = {
    "service_name": "llm_provider",
    "operation": "llm_request"
}
```

### 4. Graceful Degradation
Continues with limited functionality.

```python
# Automatically applied for agent and RAG errors
```

---

## Circuit Breakers

### Using Circuit Breakers

```python
from app.core.circuit_breaker import circuit_breaker_manager, CircuitBreakerConfig

# Configure circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=5,      # Open after 5 failures
    failure_timeout=60,       # Wait 60s before retry
    success_threshold=2,      # Close after 2 successes
    call_timeout=30.0,        # 30s timeout per call
    exponential_backoff=True  # Use exponential backoff
)

# Use circuit breaker
try:
    result = await circuit_breaker_manager.call(
        name="external_api",
        func=external_api.call,
        config=config,
        fallback=lambda: {"status": "unavailable"}
    )
except CircuitBreakerOpenError:
    # Circuit is open, service unavailable
    return {"error": "Service temporarily unavailable"}
```

### Direct Circuit Breaker Usage

```python
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Create circuit breaker
breaker = CircuitBreaker(
    name="llm_provider",
    config=CircuitBreakerConfig(failure_threshold=3)
)

# Use it
async def call_llm():
    return await breaker.call(
        func=llm_provider.generate,
        prompt="Hello",
        fallback=lambda: "Service unavailable"
    )
```

### Monitor Circuit Breakers

```python
from app.core.circuit_breaker import circuit_breaker_manager

# Get all metrics
metrics = circuit_breaker_manager.get_all_metrics()

for name, stats in metrics.items():
    print(f"{name}: {stats['state']} - {stats['success_rate']}% success")
```

---

## Error Codes

### Error Code Structure

Format: `ERR_XXXX`
- `1000-1999`: Client errors (validation, auth, not found)
- `2000-2999`: Service errors (database, LLM, RAG, agents)
- `3000-3999`: Configuration errors
- `9000-9999`: Internal/system errors

### Common Error Codes

| Code | Category | Description |
|------|----------|-------------|
| `ERR_1000` | Validation | Validation error |
| `ERR_1100` | Authentication | Authentication failed |
| `ERR_1200` | Authorization | Access denied |
| `ERR_1300` | Not Found | Resource not found |
| `ERR_1500` | Rate Limit | Rate limit exceeded |
| `ERR_2100` | Database | Database error |
| `ERR_2200` | LLM Provider | LLM provider error |
| `ERR_2300` | RAG System | RAG system error |
| `ERR_2400` | Agent | Agent error |
| `ERR_9000` | Internal | Internal server error |

### Using Error Codes

```python
from app.core.error_handling import (
    ValidationException,
    AuthenticationException,
    NotFoundException,
    LLMProviderException
)

# Validation error
if not data.is_valid():
    raise ValidationException(
        message="Invalid input data",
        details={"field": "name", "error": "required"}
    )

# Not found error
agent = await get_agent(agent_id)
if not agent:
    raise NotFoundException(
        resource_type="Agent",
        resource_id=agent_id
    )

# LLM provider error
try:
    response = await llm_provider.generate(prompt)
except Exception as e:
    raise LLMProviderException(
        provider="ollama",
        message="Failed to generate response",
        original_exception=e
    )
```

---

## Error Analysis

The system provides intelligent error analysis:

### Error Pattern Learning

```python
# Get error statistics
stats = error_handler.get_error_statistics()

print(f"Total errors: {stats['metrics']['total_errors']}")
print(f"Recovery rate: {stats['metrics']['recovery_success_rate']}")
print(f"Error patterns: {stats['total_patterns']}")
```

### Root Cause Analysis

The system automatically identifies root causes:

- Network connectivity issues
- Memory exhaustion
- Permission problems
- Resource not found
- Service overload

### Prevention Suggestions

For each error, the system suggests prevention measures:

```python
error_response = await error_handler.handle_error(error, context)

# Access prevention suggestions
for suggestion in error_response.prevention_suggestions:
    print(f"Suggestion: {suggestion}")
```

---

## Best Practices

### 1. Always Provide Context

```python
context = {
    "operation": "create_agent",
    "user_id": user_id,
    "agent_type": agent_type,
    "timestamp": datetime.utcnow().isoformat()
}
```

### 2. Use Specific Exceptions

```python
# Good
raise NotFoundException("Agent", agent_id)

# Avoid
raise Exception("Agent not found")
```

### 3. Enable Auto-Recovery for Transient Errors

```python
# For network/service errors
auto_recover=True

# For validation errors
auto_recover=False
```

### 4. Implement Fallbacks

```python
try:
    result = await primary_service.call()
except Exception as e:
    # Use fallback
    result = await fallback_service.call()
```

### 5. Monitor Circuit Breakers

```python
# Regular health check
metrics = circuit_breaker_manager.get_all_metrics()

for name, stats in metrics.items():
    if stats['state'] == 'open':
        logger.warning(f"Circuit breaker {name} is OPEN")
```

---

## Advanced Features

### Custom Recovery Strategies

```python
from app.core.error_handling import error_handler

# Register custom recovery strategy
async def custom_recovery(error, context):
    # Your custom recovery logic
    return {
        "success": True,
        "attempts": 1,
        "notes": ["Custom recovery applied"]
    }

error_handler.recovery_strategies["custom"] = custom_recovery
```

### Error Event Hooks

```python
# Hook into error events
async def on_error(error, analysis):
    # Send notification, log to external service, etc.
    await notification_service.send_alert(error, analysis)

# Register hook
error_handler.on_error_hooks.append(on_error)
```

---

## Monitoring & Metrics

### Error Metrics

```python
from app.core.error_handling import error_handler

stats = error_handler.get_error_statistics()

print(f"""
Error Handling Metrics:
- Total Errors: {stats['metrics']['total_errors']}
- Recovered: {stats['metrics']['recovered_errors']}
- Success Rate: {stats['metrics']['recovery_success_rate']:.2%}
- Avg Recovery Time: {stats['metrics']['average_recovery_time']:.2f}ms
- Prevented Errors: {stats['metrics']['prevented_errors']}
""")
```

### Circuit Breaker Metrics

```python
from app.core.circuit_breaker import circuit_breaker_manager

metrics = circuit_breaker_manager.get_all_metrics()

for name, stats in metrics.items():
    print(f"""
Circuit Breaker: {name}
- State: {stats['state']}
- Success Rate: {stats['success_rate']:.2%}
- Total Calls: {stats['total_calls']}
- Failed Calls: {stats['failed_calls']}
- Rejected Calls: {stats['rejected_calls']}
- Avg Response Time: {stats['average_response_time']:.2f}s
""")
```

---

## Integration Examples

### FastAPI Middleware

```python
from fastapi import FastAPI, Request
from app.core.error_handling import error_handler

app = FastAPI()

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        error_response = await error_handler.handle_error(
            error=e,
            context={"path": request.url.path, "method": request.method},
            request_id=request.state.request_id,
            auto_recover=True
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )
```

### Background Tasks

```python
from fastapi import BackgroundTasks

async def process_document(doc_id: str):
    try:
        await document_processor.process(doc_id)
    except Exception as e:
        await error_handler.handle_error(
            error=e,
            context={"operation": "process_document", "doc_id": doc_id},
            auto_recover=True
        )

@app.post("/documents/{doc_id}/process")
async def process_document_endpoint(doc_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_document, doc_id)
    return {"status": "processing"}
```

---

## Troubleshooting

### High Error Rate

1. **Check error patterns** - Identify common errors
2. **Review recovery strategies** - Are they appropriate?
3. **Monitor circuit breakers** - Are services failing?
4. **Check logs** - Look for root causes

### Circuit Breakers Always Open

1. **Check service health** - Is the service actually down?
2. **Review thresholds** - Are they too aggressive?
3. **Check timeouts** - Are they too short?
4. **Monitor metrics** - What's the failure pattern?

### Recovery Not Working

1. **Check recovery strategy** - Is it appropriate for the error?
2. **Review context** - Is all required info provided?
3. **Check logs** - What's failing during recovery?
4. **Test manually** - Can you reproduce the issue?

---

## Summary

The error handling system provides:

✅ **Intelligent Analysis** - AI-powered root cause identification  
✅ **Automatic Recovery** - Multiple recovery strategies  
✅ **Circuit Protection** - Prevent cascading failures  
✅ **Detailed Errors** - Structured error codes and messages  
✅ **Learning System** - Improves over time  
✅ **Comprehensive Monitoring** - Detailed metrics and statistics  

**Result**: More reliable system, better error recovery, easier debugging!

