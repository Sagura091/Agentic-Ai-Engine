# Revolutionary Logging System

## Overview

The Revolutionary Logging System is a comprehensive, enterprise-grade logging architecture for the Agentic AI Microservice platform. It provides clean user-facing output, granular module control, and powerful debugging capabilities.

## Key Features

✅ **Clean User Experience** - No technical spam, just agent conversation  
✅ **Granular Module Control** - Turn on/off any of 14 backend systems  
✅ **Three Logging Modes** - USER, DEVELOPER, DEBUG  
✅ **Runtime Configuration** - Change settings without restart  
✅ **Hot-Reload Support** - Zero-downtime configuration updates  
✅ **Complete Audit Trail** - File persistence with rotation  

## Quick Start

### 1. Choose Your Mode

**For End Users** (clean conversation only):
```bash
LOG_MODE=user
```

**For Developers** (selective debugging):
```bash
LOG_MODE=developer
LOG_MODULE_AGENTS=true
LOG_MODULE_RAG=true
```

**For Deep Debugging** (full verbose):
```bash
LOG_MODE=debug
```

### 2. Start the Application

```bash
python app/main.py
```

## Logging Modes

### USER Mode

**Purpose**: Clean user experience without technical details

**Output**:
```
🧑 User: What is the weather in San Francisco?
🤖 Agent: I'll help you with that. Let me think about this...
🔧 Using: weather_api
💬 The weather in San Francisco is currently sunny with a temperature of 72°F.
```

**Configuration**:
```bash
LOG_MODE=user
LOG_CONVERSATION_ENABLED=true
LOG_MODULE_AGENTS=false
LOG_MODULE_RAG=false
```

### DEVELOPER Mode

**Purpose**: Selective module debugging for development

**Output**:
```
🧑 User: What is the weather in San Francisco?
🤖 Agent: I'll help you with that...
[11:23:45] [DEBUG] [app.agents.base.agent] Agent reasoning started
[11:23:46] [INFO]  [app.rag.core] Document retrieved
🔧 Using: weather_api
💬 The weather is sunny, 72°F.
```

**Configuration**:
```bash
LOG_MODE=developer
LOG_MODULE_AGENTS=true
LOG_MODULE_RAG=true
LOG_LEVEL_APP_AGENTS=DEBUG
```

### DEBUG Mode

**Purpose**: Full verbose logging for deep debugging

**Output**:
```
[2025-10-01 11:23:45.123] [DEBUG] [app.agents.base.agent:763] Agent reasoning iteration 1 started | correlation_id=9a6dd14a... | session_id=adf492b3... | agent_id=74ba858e...
[2025-10-01 11:23:45.234] [DEBUG] [app.llm.manager:145] LLM call started | model=llama3.2 | provider=ollama
[2025-10-01 11:23:46.345] [DEBUG] [app.rag.core:234] Document retrieval started
```

**Configuration**:
```bash
LOG_MODE=debug
LOG_SHOW_IDS=true
LOG_SHOW_TIMESTAMPS=true
LOG_SHOW_CALLSITE=true
```

## Module Control

### Supported Modules (14 total)

1. **app.agents** - Agent operations and reasoning
2. **app.rag** - RAG system (ingestion, retrieval, vision)
3. **app.memory** - Memory system (consolidation, knowledge graph)
4. **app.llm** - LLM management and calls
5. **app.tools** - Tool execution
6. **app.api** - API layer
7. **app.core** - Core system components
8. **app.services** - Business logic
9. **app.orchestration** - Workflow orchestration
10. **app.communication** - Agent communication
11. **app.config** - Configuration management
12. **app.models** - Database models
13. **app.optimization** - Performance optimization
14. **app.integrations** - External integrations

### Enable/Disable Modules

**Environment Variables**:
```bash
LOG_MODULE_AGENTS=true          # Enable agents
LOG_MODULE_RAG=true             # Enable RAG
LOG_MODULE_MEMORY=false         # Disable memory
```

**YAML Configuration**:
```yaml
module_control:
  app.agents:
    enabled: true
    console_level: DEBUG
  app.rag:
    enabled: true
    console_level: INFO
  app.memory:
    enabled: false
```

**Runtime API**:
```bash
POST /api/v1/admin/logging/module/enable
{
  "module_name": "app.rag",
  "enabled": true,
  "console_level": "DEBUG"
}
```

## Configuration

### Priority Order

1. **Runtime API** (highest priority)
2. **Environment Variables**
3. **YAML Configuration**
4. **Default Values** (lowest priority)

### Environment Variables

See [.env.example](.env.example) for all available variables.

**Essential Variables**:
```bash
# Mode
LOG_MODE=developer

# Module Control
LOG_MODULE_AGENTS=true
LOG_MODULE_RAG=true

# Log Levels
LOG_LEVEL_APP_AGENTS=DEBUG
LOG_LEVEL_APP_RAG=INFO

# File Persistence
LOG_FILE_ENABLED=true
LOG_FILE_RETENTION_DAYS=30
```

### YAML Configuration

See [config/logging.yaml](../../config/logging.yaml) for the complete schema.

**Example**:
```yaml
global:
  mode: developer
  
module_control:
  app.agents:
    enabled: true
    console_level: DEBUG
    
file_persistence:
  enabled: true
  retention_days: 30
```

### Runtime API

**Get Status**:
```bash
GET /api/v1/admin/logging/status
```

**Change Mode**:
```bash
POST /api/v1/admin/logging/mode
{"mode": "debug"}
```

**Enable Module**:
```bash
POST /api/v1/admin/logging/module/enable
{"module_name": "app.rag", "console_level": "DEBUG"}
```

## Architecture

### 5-Layer Architecture

```
Layer 1: User Conversation Layer
    ↓
Layer 2: Module Control Layer
    ↓
Layer 3: Tier System Layer
    ↓
Layer 4: Backend Logging Layer
    ↓
Layer 5: File Persistence Layer
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete architecture documentation
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Configuration guide
- **[API_REFERENCE.md](API_REFERENCE.md)** - API reference
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migration guide

## Common Scenarios

### Scenario 1: Debug RAG Issues

```bash
LOG_MODE=developer
LOG_MODULE_RAG=true
LOG_LEVEL_APP_RAG=DEBUG
LOG_CONSOLE_APP_RAG=true
```

### Scenario 2: Clean User Experience

```bash
LOG_MODE=user
LOG_CONVERSATION_ENABLED=true
LOG_MODULE_AGENTS=false
LOG_MODULE_RAG=false
```

### Scenario 3: Full System Debugging

```bash
LOG_MODE=debug
LOG_SHOW_IDS=true
LOG_SHOW_TIMESTAMPS=true
LOG_MODULE_AGENTS=true
LOG_MODULE_RAG=true
LOG_MODULE_MEMORY=true
```

## File Structure

### Log Files

```
data/logs/backend/
├── agent_operations_20251001.log
├── rag_operations_20251001.log
├── memory_operations_20251001.log
├── llm_operations_20251001.log
├── tool_operations_20251001.log
├── api_layer_20251001.log
├── database_layer_20251001.log
├── external_integrations_20251001.log
├── security_events_20251001.log
├── configuration_management_20251001.log
├── resource_management_20251001.log
├── system_health_20251001.log
├── orchestration_20251001.log
├── communication_20251001.log
├── service_operations_20251001.log
├── performance_20251001.log
├── user_interaction_20251001.log
└── error_tracking_20251001.log
```

### Configuration Files

```
config/
├── logging.yaml              # Main configuration
├── logging.dev.yaml          # Development configuration
├── logging.prod.yaml         # Production configuration
├── logging_module_hierarchy.json  # Module hierarchy
├── logging_env_vars.md       # Environment variable docs
└── conversation_format_spec.md    # Conversation format spec
```

## Code Examples

### Backend Logging

```python
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

logger = get_logger()

logger.info(
    "Processing document",
    LogCategory.RAG_OPERATIONS,
    "DocumentProcessor",
    data={
        "document_id": doc_id,
        "size": doc_size,
        "format": doc_format
    }
)
```

### Conversation Logging

```python
from app.core.clean_logging import get_conversation_logger

conversation_logger = get_conversation_logger("DocumentAgent")

conversation_logger.user_query("Analyze this document")
conversation_logger.agent_thinking("Analyzing document structure...")
conversation_logger.tool_usage("document_parser", "Extracting text")
conversation_logger.agent_response("I've analyzed the document!")
```

## Performance

- **Zero overhead** for disabled modules
- **Async file writing** for minimal performance impact
- **Buffered I/O** reduces disk operations
- **Hot-reload** without service interruption

## Troubleshooting

### No Logs Appearing

1. Check `LOG_ENABLED=true`
2. Check module is enabled: `LOG_MODULE_AGENTS=true`
3. Check log level: `LOG_LEVEL_APP_AGENTS=DEBUG`
4. Check console output: `LOG_CONSOLE_APP_AGENTS=true`

### Too Many Logs

1. Switch to USER mode: `LOG_MODE=user`
2. Disable noisy modules: `LOG_MODULE_API=false`
3. Increase log level: `LOG_LEVEL_APP_RAG=WARNING`

### Configuration Not Loading

1. Check YAML syntax
2. Check file path: `LOG_HOT_RELOAD_CONFIG_PATH`
3. Check environment variables (higher priority)

## Support

For issues or questions:

1. Check [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
2. Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
3. Enable debug mode: `LOG_MODE=debug`
4. Check error logs: `data/logs/backend/error_tracking_*.log`

## License

Part of the Agentic AI Microservice platform.

