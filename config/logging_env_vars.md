# Environment Variables for Revolutionary Logging System

This document describes all environment variables that can be used to configure the logging system.

## Global Settings

```bash
# Logging mode: user | developer | debug
LOG_MODE=user

# Enable/disable entire logging system
LOG_ENABLED=true

# Show correlation IDs, session IDs, agent IDs
LOG_SHOW_IDS=false

# Show timestamps in console output
LOG_SHOW_TIMESTAMPS=false

# Timestamp format: simple | full | iso
LOG_TIMESTAMP_FORMAT=simple
```

## User Conversation Layer

```bash
# Enable user-facing conversation output
LOG_CONVERSATION_ENABLED=true

# Conversation style: conversational | technical | minimal
LOG_CONVERSATION_STYLE=conversational

# Show agent reasoning/thinking process
LOG_CONVERSATION_SHOW_REASONING=true

# Show tool usage and execution
LOG_CONVERSATION_SHOW_TOOL_USAGE=true

# Show tool results
LOG_CONVERSATION_SHOW_TOOL_RESULTS=true

# Show agent final responses
LOG_CONVERSATION_SHOW_RESPONSES=true

# Enable emoji enhancement
LOG_CONVERSATION_EMOJI_ENHANCED=true

# Maximum reasoning text length
LOG_CONVERSATION_MAX_REASONING_LENGTH=200

# Maximum tool result length
LOG_CONVERSATION_MAX_RESULT_LENGTH=500
```

## Module Control - Enable/Disable

```bash
# Enable/disable specific modules (true/false)
LOG_MODULE_AGENTS=false
LOG_MODULE_RAG=false
LOG_MODULE_MEMORY=false
LOG_MODULE_LLM=false
LOG_MODULE_TOOLS=false
LOG_MODULE_API=false
LOG_MODULE_CORE=true
LOG_MODULE_SERVICES=false
LOG_MODULE_ORCHESTRATION=false
LOG_MODULE_COMMUNICATION=false
LOG_MODULE_CONFIG=false
LOG_MODULE_MODELS=false
LOG_MODULE_OPTIMIZATION=false
LOG_MODULE_INTEGRATIONS=false
```

## Module Control - Log Levels

```bash
# Console log levels for each module
# Levels: DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_LEVEL_APP_AGENTS=CRITICAL
LOG_LEVEL_APP_RAG=WARNING
LOG_LEVEL_APP_MEMORY=WARNING
LOG_LEVEL_APP_LLM=ERROR
LOG_LEVEL_APP_TOOLS=CRITICAL
LOG_LEVEL_APP_API=ERROR
LOG_LEVEL_APP_CORE=WARNING
LOG_LEVEL_APP_SERVICES=ERROR
LOG_LEVEL_APP_ORCHESTRATION=WARNING
LOG_LEVEL_APP_COMMUNICATION=INFO
LOG_LEVEL_APP_CONFIG=WARNING
LOG_LEVEL_APP_MODELS=ERROR
LOG_LEVEL_APP_OPTIMIZATION=WARNING
LOG_LEVEL_APP_INTEGRATIONS=WARNING

# Sub-module specific levels (examples)
LOG_LEVEL_APP_AGENTS_BASE=CRITICAL
LOG_LEVEL_APP_AGENTS_BASE_AGENT=CRITICAL
LOG_LEVEL_APP_AGENTS_AUTONOMOUS=CRITICAL
LOG_LEVEL_APP_AGENTS_FACTORY=INFO
LOG_LEVEL_APP_AGENTS_REGISTRY=INFO

LOG_LEVEL_APP_RAG_CORE=WARNING
LOG_LEVEL_APP_RAG_CORE_UNIFIED_RAG_SYSTEM=WARNING
LOG_LEVEL_APP_RAG_INGESTION=WARNING
LOG_LEVEL_APP_RAG_INGESTION_PIPELINE=WARNING
LOG_LEVEL_APP_RAG_TOOLS=WARNING
LOG_LEVEL_APP_RAG_VISION=WARNING

LOG_LEVEL_APP_MEMORY_UNIFIED_MEMORY_SYSTEM=WARNING
LOG_LEVEL_APP_MEMORY_MEMORY_CONSOLIDATION_SYSTEM=WARNING
LOG_LEVEL_APP_MEMORY_DYNAMIC_KNOWLEDGE_GRAPH=WARNING

LOG_LEVEL_APP_LLM_MANAGER=ERROR
LOG_LEVEL_APP_LLM_PROVIDERS=ERROR

LOG_LEVEL_APP_TOOLS_PRODUCTION=CRITICAL
LOG_LEVEL_APP_TOOLS_SOCIAL_MEDIA=CRITICAL

LOG_LEVEL_APP_API_V1=ERROR
LOG_LEVEL_APP_API_WEBSOCKET=ERROR

LOG_LEVEL_APP_CORE_UNIFIED_SYSTEM_ORCHESTRATOR=WARNING
LOG_LEVEL_APP_CORE_ERROR_HANDLING=ERROR
LOG_LEVEL_APP_CORE_MONITORING=INFO
```

## Module Control - Console/File Output

```bash
# Enable console output for specific modules
LOG_CONSOLE_APP_AGENTS=false
LOG_CONSOLE_APP_RAG=false
LOG_CONSOLE_APP_MEMORY=false
LOG_CONSOLE_APP_LLM=false
LOG_CONSOLE_APP_TOOLS=false
LOG_CONSOLE_APP_API=false
LOG_CONSOLE_APP_CORE=true
LOG_CONSOLE_APP_SERVICES=false

# Enable file output for specific modules
LOG_FILE_APP_AGENTS=true
LOG_FILE_APP_RAG=true
LOG_FILE_APP_MEMORY=true
LOG_FILE_APP_LLM=true
LOG_FILE_APP_TOOLS=true
LOG_FILE_APP_API=true
LOG_FILE_APP_CORE=true
LOG_FILE_APP_SERVICES=true
```

## File Logging

```bash
# Enable file logging
LOG_FILE_ENABLED=true

# Base directory for log files
LOG_FILE_DIRECTORY=data/logs/backend

# Log file format: json | text
LOG_FILE_FORMAT=json

# Separate log files by category
LOG_FILE_SEPARATE_BY_CATEGORY=true

# Separate log files by module
LOG_FILE_SEPARATE_BY_MODULE=false

# File rotation strategy: daily | size | time
LOG_FILE_ROTATION_STRATEGY=daily

# Maximum file size in MB (for size-based rotation)
LOG_FILE_MAX_SIZE_MB=100

# Rotation time interval in hours (for time-based rotation)
LOG_FILE_ROTATION_INTERVAL_HOURS=24

# Number of backup files to keep
LOG_FILE_BACKUP_COUNT=30

# Retention period in days
LOG_FILE_RETENTION_DAYS=30

# Compress old log files
LOG_FILE_COMPRESS_OLD_LOGS=true

# Compression format: gz | zip
LOG_FILE_COMPRESSION_FORMAT=gz

# Include conversation logs in files
LOG_FILE_INCLUDE_CONVERSATION=true
```

## Backend Logging

```bash
# Enable structured logging with context
LOG_BACKEND_ENABLED=true

# Include performance metrics
LOG_BACKEND_INCLUDE_PERFORMANCE_METRICS=true

# Include correlation tracking
LOG_BACKEND_INCLUDE_CORRELATION_TRACKING=true

# Include stack traces for errors
LOG_BACKEND_INCLUDE_STACK_TRACES=true

# Maximum stack trace depth
LOG_BACKEND_MAX_STACK_TRACE_DEPTH=10

# Async logging for performance
LOG_BACKEND_ASYNC_LOGGING=true

# Log buffer size for async logging
LOG_BACKEND_LOG_BUFFER_SIZE=1000

# Flush interval in seconds
LOG_BACKEND_FLUSH_INTERVAL_SECONDS=5
```

## External Libraries

```bash
# Suppress noisy external loggers
LOG_EXTERNAL_SUPPRESS_NOISY=true

# Default level for external loggers
LOG_EXTERNAL_DEFAULT_LEVEL=ERROR
```

## Runtime Configuration

```bash
# Enable hot-reload of configuration
LOG_RUNTIME_HOT_RELOAD_ENABLED=true

# Configuration reload interval in seconds
LOG_RUNTIME_RELOAD_INTERVAL_SECONDS=60

# Enable runtime API for logging control
LOG_RUNTIME_API_ENABLED=true

# API authentication required
LOG_RUNTIME_API_AUTH_REQUIRED=true

# Allow runtime mode switching
LOG_RUNTIME_ALLOW_MODE_SWITCHING=true

# Allow runtime module control
LOG_RUNTIME_ALLOW_MODULE_CONTROL=true
```

## Example Configurations

### Example 1: End User (Clean Conversation Only)
```bash
LOG_MODE=user
LOG_CONVERSATION_ENABLED=true
LOG_MODULE_AGENTS=false
LOG_MODULE_RAG=false
LOG_MODULE_MEMORY=false
LOG_MODULE_LLM=false
LOG_MODULE_TOOLS=false
LOG_MODULE_API=false
LOG_MODULE_CORE=false
```

### Example 2: Developer Debugging RAG
```bash
LOG_MODE=developer
LOG_CONVERSATION_ENABLED=true
LOG_MODULE_AGENTS=false
LOG_MODULE_RAG=true
LOG_MODULE_MEMORY=false
LOG_LEVEL_APP_RAG_INGESTION=DEBUG
LOG_LEVEL_APP_RAG_CORE=INFO
LOG_CONSOLE_APP_RAG=true
```

### Example 3: Full System Debug
```bash
LOG_MODE=debug
LOG_CONVERSATION_ENABLED=true
LOG_MODULE_AGENTS=true
LOG_MODULE_RAG=true
LOG_MODULE_MEMORY=true
LOG_MODULE_LLM=true
LOG_MODULE_TOOLS=true
LOG_LEVEL_APP_AGENTS=DEBUG
LOG_LEVEL_APP_RAG=DEBUG
LOG_LEVEL_APP_MEMORY=DEBUG
LOG_SHOW_IDS=true
LOG_SHOW_TIMESTAMPS=true
```

### Example 4: Production (Errors Only)
```bash
LOG_MODE=user
LOG_CONVERSATION_ENABLED=true
LOG_MODULE_AGENTS=false
LOG_MODULE_RAG=false
LOG_MODULE_MEMORY=false
LOG_MODULE_CORE=true
LOG_LEVEL_APP_CORE=ERROR
LOG_FILE_ENABLED=true
LOG_FILE_RETENTION_DAYS=90
```

