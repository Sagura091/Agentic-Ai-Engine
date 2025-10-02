# Revolutionary Logging System - Configuration Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Environment Variables](#environment-variables)
3. [YAML Configuration](#yaml-configuration)
4. [Runtime API](#runtime-api)
5. [Common Scenarios](#common-scenarios)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Choose Your Logging Mode

**For End Users** (clean conversation only):
```bash
# .env file
LOG_MODE=user
LOG_CONVERSATION_ENABLED=true
```

**For Developers** (selective module debugging):
```bash
# .env file
LOG_MODE=developer
LOG_MODULE_AGENTS=true
LOG_MODULE_RAG=true
LOG_LEVEL_APP_AGENTS=DEBUG
LOG_LEVEL_APP_RAG=INFO
```

**For Deep Debugging** (full verbose logging):
```bash
# .env file
LOG_MODE=debug
LOG_SHOW_IDS=true
LOG_SHOW_TIMESTAMPS=true
LOG_SHOW_CALLSITE=true
```

### 2. Start the Application

```bash
python app/main.py
```

The logging system will automatically load your configuration!

---

## Environment Variables

### Global Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_MODE` | string | `user` | Logging mode: `user`, `developer`, or `debug` |
| `LOG_ENABLED` | bool | `true` | Master switch for all logging |
| `LOG_SHOW_IDS` | bool | `false` | Show correlation/session/agent IDs |
| `LOG_SHOW_TIMESTAMPS` | bool | `false` | Show timestamps in console output |
| `LOG_SHOW_CALLSITE` | bool | `false` | Show filename, function, line number |
| `LOG_EXTERNAL_LOGGERS_ENABLED` | bool | `false` | Allow external library logs |

### Conversation Layer

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_CONVERSATION_ENABLED` | bool | `true` | Enable conversation layer |
| `LOG_CONVERSATION_STYLE` | string | `conversational` | Style: `conversational` or `technical` |
| `LOG_CONVERSATION_EMOJI` | bool | `true` | Use emoji in conversation output |
| `LOG_CONVERSATION_MAX_REASONING` | int | `200` | Max chars for reasoning messages |
| `LOG_CONVERSATION_MAX_TOOL_RESULT` | int | `300` | Max chars for tool results |

### Module Control (14 Modules)

**Enable/Disable Modules**:
```bash
LOG_MODULE_AGENTS=true          # app.agents
LOG_MODULE_RAG=true             # app.rag
LOG_MODULE_MEMORY=true          # app.memory
LOG_MODULE_LLM=true             # app.llm
LOG_MODULE_TOOLS=true           # app.tools
LOG_MODULE_API=false            # app.api
LOG_MODULE_CORE=false           # app.core
LOG_MODULE_SERVICES=false       # app.services
LOG_MODULE_ORCHESTRATION=false  # app.orchestration
LOG_MODULE_COMMUNICATION=false  # app.communication
LOG_MODULE_CONFIG=false         # app.config
LOG_MODULE_MODELS=false         # app.models
LOG_MODULE_OPTIMIZATION=false   # app.optimization
LOG_MODULE_INTEGRATIONS=false   # app.integrations
```

**Set Log Levels**:
```bash
LOG_LEVEL_APP_AGENTS=DEBUG      # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL_APP_RAG=INFO
LOG_LEVEL_APP_MEMORY=WARNING
```

**Console vs File Output**:
```bash
LOG_CONSOLE_APP_AGENTS=true     # Show in console
LOG_FILE_APP_AGENTS=true        # Write to file
```

### File Persistence

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_FILE_ENABLED` | bool | `true` | Enable file logging |
| `LOG_FILE_BASE_PATH` | string | `data/logs/backend` | Base directory for logs |
| `LOG_FILE_ROTATION_STRATEGY` | string | `daily` | Rotation: `daily`, `size`, `time` |
| `LOG_FILE_MAX_SIZE_MB` | int | `100` | Max file size before rotation |
| `LOG_FILE_RETENTION_DAYS` | int | `30` | Days to keep old logs |
| `LOG_FILE_COMPRESSION` | bool | `true` | Compress rotated logs |
| `LOG_FILE_FORMAT` | string | `json` | Format: `json` or `text` |

### Hot-Reload

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_HOT_RELOAD_ENABLED` | bool | `true` | Enable hot-reload |
| `LOG_HOT_RELOAD_INTERVAL` | int | `60` | Check interval (seconds) |
| `LOG_HOT_RELOAD_CONFIG_PATH` | string | `config/logging.yaml` | YAML config path |

---

## YAML Configuration

### Basic Configuration

Create `config/logging.yaml`:

```yaml
# Global settings
global:
  mode: developer              # user, developer, or debug
  enabled: true
  show_ids: false
  show_timestamps: true
  show_callsite: false
  external_loggers_enabled: false

# Conversation layer
conversation_layer:
  enabled: true
  style: conversational        # conversational or technical
  emoji_enhanced: true
  max_reasoning_length: 200
  max_tool_result_length: 300

# Module control
module_control:
  app.agents:
    enabled: true
    console_level: DEBUG
    file_level: DEBUG
    console_output: true
    file_output: true
  
  app.rag:
    enabled: true
    console_level: INFO
    file_level: DEBUG
    console_output: true
    file_output: true
  
  app.memory:
    enabled: false
    console_level: WARNING
    file_level: INFO
    console_output: false
    file_output: true

# Tier configurations
tier_configs:
  user:
    console_level: WARNING
    show_ids: false
    show_timestamps: false
    show_callsite: false
  
  developer:
    console_level: INFO
    show_ids: false
    show_timestamps: true
    show_callsite: false
  
  debug:
    console_level: DEBUG
    show_ids: true
    show_timestamps: true
    show_callsite: true

# File persistence
file_persistence:
  enabled: true
  base_path: data/logs/backend
  rotation_strategy: daily
  max_size_mb: 100
  retention_days: 30
  compression_enabled: true
  format: json
  async_writing: true
  buffer_size: 8192

# Hot-reload
hot_reload:
  enabled: true
  check_interval_seconds: 60
  config_file_path: config/logging.yaml
```

### Advanced Configuration

**Hierarchical Module Control**:
```yaml
module_control:
  # Enable all RAG modules
  app.rag:
    enabled: true
    console_level: INFO
  
  # But disable RAG ingestion specifically
  app.rag.ingestion:
    enabled: false
  
  # And set RAG vision to DEBUG
  app.rag.vision:
    enabled: true
    console_level: DEBUG
```

**Environment-Specific Configs**:

**Development** (`config/logging.dev.yaml`):
```yaml
global:
  mode: developer
  show_timestamps: true

module_control:
  app.agents:
    enabled: true
    console_level: DEBUG
  app.rag:
    enabled: true
    console_level: DEBUG
```

**Production** (`config/logging.prod.yaml`):
```yaml
global:
  mode: user
  show_timestamps: false

module_control:
  app.agents:
    enabled: false
  app.rag:
    enabled: false

file_persistence:
  retention_days: 90
  compression_enabled: true
```

---

## Runtime API

### Get Current Status

```bash
GET /api/v1/admin/logging/status
```

**Response**:
```json
{
  "mode": "developer",
  "conversation_enabled": true,
  "modules": {
    "app.agents": {
      "enabled": true,
      "console_level": "DEBUG",
      "file_level": "DEBUG"
    },
    "app.rag": {
      "enabled": true,
      "console_level": "INFO",
      "file_level": "DEBUG"
    }
  },
  "file_persistence_enabled": true,
  "hot_reload_enabled": true
}
```

### Change Logging Mode

```bash
POST /api/v1/admin/logging/mode
Content-Type: application/json

{
  "mode": "debug"
}
```

### Enable/Disable Module

```bash
POST /api/v1/admin/logging/module/enable
Content-Type: application/json

{
  "module_name": "app.rag",
  "enabled": true,
  "console_level": "DEBUG"
}
```

### Disable Module

```bash
POST /api/v1/admin/logging/module/disable
Content-Type: application/json

{
  "module_name": "app.memory"
}
```

### Reload Configuration

```bash
POST /api/v1/admin/logging/reload
```

### Update Configuration

```bash
POST /api/v1/admin/logging/config
Content-Type: application/json

{
  "conversation_layer": {
    "emoji_enhanced": false,
    "max_reasoning_length": 500
  },
  "file_persistence": {
    "retention_days": 60
  }
}
```

---

## Common Scenarios

### Scenario 1: Clean User Experience

**Goal**: Show only agent conversation, no technical logs.

**.env**:
```bash
LOG_MODE=user
LOG_CONVERSATION_ENABLED=true
LOG_MODULE_AGENTS=false
LOG_MODULE_RAG=false
LOG_MODULE_MEMORY=false
LOG_MODULE_LLM=false
```

**Output**:
```
🧑 User: What is the weather in San Francisco?
🤖 Agent: I'll help you with that. Let me think about this...
🔧 Using: weather_api
💬 The weather in San Francisco is currently sunny with a temperature of 72°F.
```

### Scenario 2: Debug RAG System

**Goal**: Debug RAG issues while keeping other modules quiet.

**.env**:
```bash
LOG_MODE=developer
LOG_MODULE_RAG=true
LOG_LEVEL_APP_RAG=DEBUG
LOG_MODULE_AGENTS=false
LOG_MODULE_MEMORY=false
```

**Output**:
```
[11:23:45] [DEBUG] [app.rag.ingestion] Processing document: example.pdf
[11:23:46] [DEBUG] [app.rag.ingestion] Created 15 chunks
[11:23:47] [INFO]  [app.rag.core] Stored in collection: documents
```

### Scenario 3: Full System Debugging

**Goal**: See everything for deep debugging.

**.env**:
```bash
LOG_MODE=debug
LOG_SHOW_IDS=true
LOG_SHOW_TIMESTAMPS=true
LOG_SHOW_CALLSITE=true
LOG_MODULE_AGENTS=true
LOG_MODULE_RAG=true
LOG_MODULE_MEMORY=true
LOG_MODULE_LLM=true
```

**Output**:
```
[2025-10-01 11:23:45.123] [DEBUG] [app.agents.base.agent:763] Agent reasoning iteration 1 started | correlation_id=9a6dd14a... | session_id=adf492b3... | agent_id=74ba858e...
[2025-10-01 11:23:45.234] [DEBUG] [app.llm.manager:145] LLM call started | model=llama3.2 | provider=ollama
```

---

## Troubleshooting

### Issue: No Logs Appearing

**Check**:
1. Is `LOG_ENABLED=true`?
2. Is the module enabled? (`LOG_MODULE_AGENTS=true`)
3. Is the log level appropriate? (DEBUG shows more than WARNING)
4. Is console output enabled? (`LOG_CONSOLE_APP_AGENTS=true`)

### Issue: Too Many Logs

**Solution**:
1. Switch to USER mode: `LOG_MODE=user`
2. Disable noisy modules: `LOG_MODULE_API=false`
3. Increase log level: `LOG_LEVEL_APP_RAG=WARNING`

### Issue: Configuration Not Loading

**Check**:
1. Is the YAML file path correct? (`LOG_HOT_RELOAD_CONFIG_PATH`)
2. Is the YAML syntax valid? (use YAML validator)
3. Are environment variables overriding? (env vars have higher priority)

### Issue: Hot-Reload Not Working

**Check**:
1. Is hot-reload enabled? (`LOG_HOT_RELOAD_ENABLED=true`)
2. Is the interval too long? (`LOG_HOT_RELOAD_INTERVAL=60`)
3. Check file permissions on `config/logging.yaml`

---

**Next**: See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation.

