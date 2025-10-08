# Backend Logging System - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will get you up and running with the production backend logging system.

---

## 📦 Basic Setup

### 1. Import the Logger

```python
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get the global logger instance
logger = get_logger()
```

### 2. Log Your First Message

```python
# Simple info log
logger.info(
    "User logged in successfully",
    LogCategory.SECURITY_EVENTS,
    "AuthService"
)

# Log with additional data
logger.info(
    "Document processed",
    LogCategory.RAG_OPERATIONS,
    "DocumentProcessor",
    data={
        "document_id": "doc_123",
        "chunks_created": 42,
        "processing_time_ms": 1250
    }
)

# Log an error
logger.error(
    "Failed to connect to database",
    LogCategory.DATABASE_OPERATIONS,
    "DatabaseService",
    error=exception,
    data={
        "host": "localhost",
        "port": 5432,
        "retry_count": 3
    }
)
```

---

## 📋 Log Categories

Choose the appropriate category for your logs:

| Category | Use Case |
|----------|----------|
| `AGENT_OPERATIONS` | Agent reasoning, decisions, executions |
| `TOOL_OPERATIONS` | Tool usage and results |
| `RAG_OPERATIONS` | Document ingestion, retrieval, chunking |
| `LLM_OPERATIONS` | LLM calls, responses, errors |
| `MEMORY_OPERATIONS` | Memory storage, retrieval, updates |
| `API_OPERATIONS` | API requests, responses |
| `DATABASE_OPERATIONS` | Database queries, connections |
| `SECURITY_EVENTS` | Authentication, authorization |
| `ERROR_TRACKING` | Application errors |
| `PERFORMANCE_MONITORING` | Performance metrics |
| `SYSTEM_HEALTH` | System status, health checks |

---

## 🎯 Common Use Cases

### Use Case 1: Agent Operations

```python
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

logger = get_logger()

# Log agent task start
logger.info(
    "Agent task started",
    LogCategory.AGENT_OPERATIONS,
    "ReActAgent",
    data={
        "task": "analyze_document",
        "user_id": "user_123",
        "agent_id": "agent_456"
    }
)

# Log agent reasoning
logger.debug(
    "Agent reasoning step",
    LogCategory.AGENT_OPERATIONS,
    "ReActAgent",
    data={
        "thought": "I need to retrieve relevant documents first",
        "step": 1,
        "total_steps": 5
    }
)

# Log tool execution
logger.info(
    "Tool executed",
    LogCategory.TOOL_OPERATIONS,
    "DocumentRetriever",
    data={
        "tool": "rag_search",
        "query": "machine learning algorithms",
        "results_count": 10,
        "execution_time_ms": 450
    }
)

# Log task completion
logger.info(
    "Agent task completed",
    LogCategory.AGENT_OPERATIONS,
    "ReActAgent",
    data={
        "task": "analyze_document",
        "success": True,
        "total_time_ms": 5200
    }
)
```

### Use Case 2: RAG Document Processing

```python
logger = get_logger()

# Log document ingestion
logger.info(
    "Document ingestion started",
    LogCategory.RAG_OPERATIONS,
    "IngestionPipeline",
    data={
        "file_name": "research_paper.pdf",
        "file_size_mb": 5.2,
        "mime_type": "application/pdf"
    }
)

# Log chunking
logger.info(
    "Document chunked",
    LogCategory.RAG_OPERATIONS,
    "SemanticChunker",
    data={
        "document_id": "doc_123",
        "chunks_created": 42,
        "avg_chunk_size": 1000,
        "chunking_strategy": "semantic"
    }
)

# Log embedding
logger.info(
    "Embeddings generated",
    LogCategory.RAG_OPERATIONS,
    "EmbeddingService",
    data={
        "chunks_embedded": 42,
        "model": "text-embedding-ada-002",
        "total_tokens": 35000,
        "duration_ms": 2500
    }
)

# Log storage
logger.info(
    "Document stored in vector database",
    LogCategory.RAG_OPERATIONS,
    "VectorStore",
    data={
        "collection": "research_papers",
        "vectors_stored": 42,
        "storage_time_ms": 150
    }
)
```

### Use Case 3: LLM Operations

```python
logger = get_logger()

# Log LLM call
logger.info(
    "LLM request initiated",
    LogCategory.LLM_OPERATIONS,
    "LLMProvider",
    data={
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000
    }
)

# Log LLM response
logger.info(
    "LLM response received",
    LogCategory.LLM_OPERATIONS,
    "LLMProvider",
    data={
        "provider": "openai",
        "model": "gpt-4",
        "tokens_used": 1250,
        "response_time_ms": 3500,
        "cost_usd": 0.025
    }
)

# Log LLM error
logger.error(
    "LLM request failed",
    LogCategory.LLM_OPERATIONS,
    "LLMProvider",
    error=exception,
    data={
        "provider": "openai",
        "model": "gpt-4",
        "error_type": "RateLimitError",
        "retry_after_seconds": 60
    }
)
```

### Use Case 4: Tool Operations

```python
logger = get_logger()

# Log tool execution
logger.info(
    "File system operation started",
    LogCategory.TOOL_OPERATIONS,
    "FileSystemTool",
    data={
        "operation": "read",
        "path": "/data/documents/report.pdf",
        "user_id": "user_123"
    }
)

# Log tool success
logger.info(
    "File system operation completed",
    LogCategory.TOOL_OPERATIONS,
    "FileSystemTool",
    data={
        "operation": "read",
        "path": "/data/documents/report.pdf",
        "file_size_bytes": 524288,
        "execution_time_ms": 45
    }
)

# Log tool error
logger.error(
    "File system operation failed",
    LogCategory.TOOL_OPERATIONS,
    "FileSystemTool",
    error=exception,
    data={
        "operation": "read",
        "path": "/data/documents/report.pdf",
        "error_type": "FileNotFoundError"
    }
)
```

---

## 🔧 Advanced Features

### Enable PII Redaction

```python
logger = get_logger()

# Enable PII redaction
logger.enable_pii_redaction(True)

# Now all logs will have PII automatically redacted
logger.info(
    "User registered",
    LogCategory.SECURITY_EVENTS,
    "AuthService",
    data={
        "email": "user@example.com",  # Will be [EMAIL_REDACTED]
        "phone": "555-123-4567",      # Will be [PHONE_REDACTED]
        "username": "john_doe"         # Not PII, preserved
    }
)
```

### Enable Log Sampling

```python
logger = get_logger()

# Enable sampling for high-volume scenarios
logger.enable_sampling(True)
logger.update_sampling_rate(0.01)  # 1% sampling

# Errors are always logged regardless of sampling
logger.error(
    "Critical error occurred",
    LogCategory.ERROR_TRACKING,
    "CriticalService",
    error=exception
)
```

### Add Breadcrumbs for Error Context

```python
logger = get_logger()

# Add breadcrumbs to track user journey
logger.add_breadcrumb(
    message="User clicked upload button",
    category="user_action",
    level="info",
    data={"button_id": "upload_btn"}
)

logger.add_breadcrumb(
    message="File selected",
    category="user_action",
    level="info",
    data={"file_name": "document.pdf", "size_mb": 5.2}
)

logger.add_breadcrumb(
    message="Upload started",
    category="system_action",
    level="info",
    data={"upload_id": "upload_123"}
)

# If an error occurs, breadcrumbs provide context
logger.error(
    "Upload failed",
    LogCategory.ERROR_TRACKING,
    "UploadService",
    error=exception,
    data={"upload_id": "upload_123"}
)
# Breadcrumbs are automatically included in error reports
```

### Create Distributed Tracing Spans

```python
logger = get_logger()

# Create a span for distributed tracing
span = logger.create_span(
    "document_processing_pipeline",
    attributes={
        "document.id": "doc_123",
        "document.type": "pdf",
        "user.id": "user_456"
    }
)

# All logs within this context are correlated
logger.info(
    "Processing started",
    LogCategory.RAG_OPERATIONS,
    "DocumentProcessor"
)

# ... processing logic ...

logger.info(
    "Processing completed",
    LogCategory.RAG_OPERATIONS,
    "DocumentProcessor",
    data={"duration_ms": 3500}
)
```

### Get Logging Metrics

```python
logger = get_logger()

# Get comprehensive metrics
metrics = logger.get_metrics()

print(f"Total logs: {metrics['log_metrics']['total_logs']}")
print(f"Error count: {metrics['log_metrics']['error_count']}")
print(f"Error rate: {metrics['log_metrics']['error_rate']:.2%}")
print(f"Current QPS: {metrics['sampling_stats']['current_qps']}")

# Performance percentiles
if 'performance_percentiles' in metrics['log_metrics']:
    perf = metrics['log_metrics']['performance_percentiles']
    print(f"P50 latency: {perf['p50']}ms")
    print(f"P95 latency: {perf['p95']}ms")
    print(f"P99 latency: {perf['p99']}ms")
```

---

## 🎨 Best Practices

### 1. Choose the Right Log Level

```python
# DEBUG - Detailed diagnostic information
logger.debug("Variable value: x=42", LogCategory.SYSTEM_HEALTH, "Component")

# INFO - General informational messages
logger.info("Operation completed", LogCategory.AGENT_OPERATIONS, "Component")

# WARNING - Warning messages for potentially harmful situations
logger.warning("Retry attempt 3/5", LogCategory.API_OPERATIONS, "Component")

# ERROR - Error events that might still allow the application to continue
logger.error("Failed to process item", LogCategory.ERROR_TRACKING, "Component", error=e)

# CRITICAL/FATAL - Very severe error events that might cause the application to abort
logger.fatal("Database connection lost", LogCategory.DATABASE_OPERATIONS, "Component", error=e)
```

### 2. Include Relevant Context

```python
# ❌ Bad - Not enough context
logger.error("Operation failed", LogCategory.ERROR_TRACKING, "Service", error=e)

# ✅ Good - Rich context
logger.error(
    "Document processing failed",
    LogCategory.RAG_OPERATIONS,
    "DocumentProcessor",
    error=e,
    data={
        "document_id": "doc_123",
        "file_name": "report.pdf",
        "processing_stage": "chunking",
        "chunks_processed": 15,
        "total_chunks": 42,
        "error_type": type(e).__name__
    }
)
```

### 3. Use Structured Data

```python
# ❌ Bad - String concatenation
logger.info(f"User {user_id} uploaded {file_name}", LogCategory.API_OPERATIONS, "API")

# ✅ Good - Structured data
logger.info(
    "File uploaded",
    LogCategory.API_OPERATIONS,
    "UploadAPI",
    data={
        "user_id": user_id,
        "file_name": file_name,
        "file_size_bytes": file_size,
        "upload_duration_ms": duration
    }
)
```

---

## 📚 Next Steps

- Read the [Full Feature Documentation](./BACKEND_LOGGING_ENHANCEMENTS.md)
- Review the [Implementation Guide](./PRODUCTION_IMPLEMENTATION_COMPLETE.md)
- Check the [Configuration Reference](../../config/logging.yaml)
- Explore [Agent Logging Patterns](./AGENT_LOGGING_GUIDE.md)

---

## 🆘 Troubleshooting

### Logs Not Appearing

1. Check if the module is enabled in configuration
2. Verify log level is appropriate
3. Check file permissions for log directory
4. Review console output for initialization errors

### High Log Volume

1. Enable log sampling: `logger.enable_sampling(True)`
2. Adjust sampling rate: `logger.update_sampling_rate(0.01)`
3. Increase sampling threshold: Configure `LOG_SAMPLING_THRESHOLD_QPS`

### PII Not Being Redacted

1. Enable PII redaction: `logger.enable_pii_redaction(True)`
2. Verify configuration: `LOG_ENABLE_PII_REDACTION=true`
3. Check custom patterns if using non-standard PII

---

**Happy Logging! 🎉**

