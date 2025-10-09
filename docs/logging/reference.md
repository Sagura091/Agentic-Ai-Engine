# Revolutionary Logging System - API Reference

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [Models](#models)
5. [Error Codes](#error-codes)
6. [Examples](#examples)

---

## Overview

The Revolutionary Logging System provides a comprehensive REST API for runtime configuration and control. All endpoints are under the `/api/v1/admin/logging` path.

**Base URL**: `http://localhost:8000/api/v1/admin/logging`

**Content-Type**: `application/json`

---

## Authentication

All admin endpoints require authentication. Include your API key in the request headers:

```bash
Authorization: Bearer YOUR_API_KEY
```

---

## Endpoints

### 1. Get Logging Status

Get the current logging configuration and status.

**Endpoint**: `GET /api/v1/admin/logging/status`

**Request**: No body required

**Response**:
```json
{
  "mode": "developer",
  "enabled": true,
  "conversation_enabled": true,
  "show_ids": false,
  "show_timestamps": true,
  "show_callsite": false,
  "external_loggers_enabled": false,
  "modules": {
    "app.agents": {
      "enabled": true,
      "console_level": "DEBUG",
      "file_level": "DEBUG",
      "console_output": true,
      "file_output": true
    },
    "app.rag": {
      "enabled": true,
      "console_level": "INFO",
      "file_level": "DEBUG",
      "console_output": true,
      "file_output": true
    }
  },
  "file_persistence_enabled": true,
  "hot_reload_enabled": true,
  "config_source": "environment"
}
```

**Status Codes**:
- `200 OK`: Success
- `401 Unauthorized`: Missing or invalid authentication
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X GET http://localhost:8000/api/v1/admin/logging/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

### 2. Set Logging Mode

Change the logging mode at runtime.

**Endpoint**: `POST /api/v1/admin/logging/mode`

**Request Body**:
```json
{
  "mode": "debug"
}
```

**Parameters**:
- `mode` (string, required): One of `user`, `developer`, or `debug`

**Response**:
```json
{
  "success": true,
  "message": "Logging mode updated to debug",
  "previous_mode": "developer",
  "new_mode": "debug"
}
```

**Status Codes**:
- `200 OK`: Mode updated successfully
- `400 Bad Request`: Invalid mode value
- `401 Unauthorized`: Missing or invalid authentication
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/admin/logging/mode \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"mode": "debug"}'
```

---

### 3. Enable Module Logging

Enable logging for a specific module and set its log levels.

**Endpoint**: `POST /api/v1/admin/logging/module/enable`

**Request Body**:
```json
{
  "module_name": "app.rag",
  "enabled": true,
  "console_level": "DEBUG",
  "file_level": "DEBUG",
  "console_output": true,
  "file_output": true
}
```

**Parameters**:
- `module_name` (string, required): Module name (e.g., `app.rag`, `app.agents`)
- `enabled` (boolean, optional): Enable/disable module (default: `true`)
- `console_level` (string, optional): Console log level (default: `INFO`)
- `file_level` (string, optional): File log level (default: `DEBUG`)
- `console_output` (boolean, optional): Show in console (default: `true`)
- `file_output` (boolean, optional): Write to file (default: `true`)

**Response**:
```json
{
  "success": true,
  "message": "Module app.rag logging enabled",
  "module_name": "app.rag",
  "config": {
    "enabled": true,
    "console_level": "DEBUG",
    "file_level": "DEBUG",
    "console_output": true,
    "file_output": true
  }
}
```

**Status Codes**:
- `200 OK`: Module updated successfully
- `400 Bad Request`: Invalid module name or parameters
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Module not found
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/admin/logging/module/enable \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "module_name": "app.rag",
    "enabled": true,
    "console_level": "DEBUG"
  }'
```

---

### 4. Disable Module Logging

Disable logging for a specific module.

**Endpoint**: `POST /api/v1/admin/logging/module/disable`

**Request Body**:
```json
{
  "module_name": "app.memory"
}
```

**Parameters**:
- `module_name` (string, required): Module name to disable

**Response**:
```json
{
  "success": true,
  "message": "Module app.memory logging disabled",
  "module_name": "app.memory"
}
```

**Status Codes**:
- `200 OK`: Module disabled successfully
- `400 Bad Request`: Invalid module name
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Module not found
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/admin/logging/module/disable \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"module_name": "app.memory"}'
```

---

### 5. Reload Configuration

Reload logging configuration from the YAML file.

**Endpoint**: `POST /api/v1/admin/logging/reload`

**Request**: No body required

**Response**:
```json
{
  "success": true,
  "message": "Logging configuration reloaded successfully",
  "config_source": "yaml",
  "config_path": "config/logging.yaml",
  "modules_updated": 5,
  "timestamp": "2025-10-01T11:23:45.123Z"
}
```

**Status Codes**:
- `200 OK`: Configuration reloaded successfully
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Configuration file not found
- `500 Internal Server Error`: Server error or invalid YAML

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/admin/logging/reload \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

### 6. Update Configuration

Update specific configuration settings at runtime.

**Endpoint**: `POST /api/v1/admin/logging/config`

**Request Body**:
```json
{
  "global": {
    "show_ids": true,
    "show_timestamps": true
  },
  "conversation_layer": {
    "emoji_enhanced": false,
    "max_reasoning_length": 500
  },
  "file_persistence": {
    "retention_days": 60,
    "compression_enabled": true
  }
}
```

**Parameters**: Any valid configuration section (see [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md))

**Response**:
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "updated_sections": ["global", "conversation_layer", "file_persistence"],
  "timestamp": "2025-10-01T11:23:45.123Z"
}
```

**Status Codes**:
- `200 OK`: Configuration updated successfully
- `400 Bad Request`: Invalid configuration values
- `401 Unauthorized`: Missing or invalid authentication
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/admin/logging/config \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_layer": {
      "emoji_enhanced": false
    }
  }'
```

---

## Models

### LoggingMode

```python
class LoggingMode(str, Enum):
    USER = "user"
    DEVELOPER = "developer"
    DEBUG = "debug"
```

### LogLevel

```python
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
```

### ModuleConfig

```python
class ModuleConfig(BaseModel):
    module_name: str
    enabled: bool = True
    console_level: LogLevel = LogLevel.WARNING
    file_level: LogLevel = LogLevel.DEBUG
    console_output: bool = False
    file_output: bool = True
```

### ConversationConfig

```python
class ConversationConfig(BaseModel):
    enabled: bool = True
    style: str = "conversational"  # "conversational" or "technical"
    emoji_enhanced: bool = True
    max_reasoning_length: int = 200
    max_tool_result_length: int = 300
```

---

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| `400` | Invalid mode value | Mode must be `user`, `developer`, or `debug` |
| `400` | Invalid module name | Module name not recognized |
| `400` | Invalid log level | Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL |
| `400` | Invalid configuration | Configuration values are invalid |
| `401` | Unauthorized | Missing or invalid API key |
| `404` | Module not found | Specified module does not exist |
| `404` | Configuration file not found | YAML config file not found |
| `500` | Internal server error | Unexpected server error |
| `500` | Invalid YAML | YAML configuration file is malformed |

---

## Examples

### Example 1: Switch to Debug Mode

```bash
# Get current status
curl -X GET http://localhost:8000/api/v1/admin/logging/status \
  -H "Authorization: Bearer YOUR_API_KEY"

# Switch to debug mode
curl -X POST http://localhost:8000/api/v1/admin/logging/mode \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"mode": "debug"}'

# Verify the change
curl -X GET http://localhost:8000/api/v1/admin/logging/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Example 2: Enable RAG Debugging

```bash
# Enable RAG module with DEBUG level
curl -X POST http://localhost:8000/api/v1/admin/logging/module/enable \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "module_name": "app.rag",
    "enabled": true,
    "console_level": "DEBUG",
    "file_level": "DEBUG"
  }'
```

### Example 3: Disable Conversation Emojis

```bash
# Update conversation configuration
curl -X POST http://localhost:8000/api/v1/admin/logging/config \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_layer": {
      "emoji_enhanced": false
    }
  }'
```

### Example 4: Reload from YAML

```bash
# Edit config/logging.yaml, then reload
curl -X POST http://localhost:8000/api/v1/admin/logging/reload \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

**Next**: See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migration instructions.

