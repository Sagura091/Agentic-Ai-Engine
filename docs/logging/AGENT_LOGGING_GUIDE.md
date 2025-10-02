# Agent Logging Guide - Simplified

## Overview

This guide explains how to use the Revolutionary Logging System in your agents. The logging system is **already configured globally** - agents just need to import and use the loggers!

**Key Principle**: Agents USE loggers, they don't CONFIGURE them. All configuration is done globally in `config/logging.yaml`, environment variables, or runtime API.

---

## Table of Contents

1. [Quick Start (3 Steps)](#quick-start-3-steps)
2. [Conversation Logging](#conversation-logging)
3. [Backend Logging](#backend-logging)
4. [Global Configuration](#global-configuration)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

---

## Quick Start (3 Steps)

### Step 1: Import Loggers in Agent Code

```python
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory
from app.core.clean_logging import get_conversation_logger

# Initialize loggers (they're already configured globally!)
backend_logger = get_logger()
conversation_logger = get_conversation_logger("your_agent_id")
```

### Step 2: Use Loggers in Your Agent

```python
class YourAgent:
    def __init__(self):
        self.agent_id = "your_agent"
        # Just get the loggers - they're already configured!
        self.backend_logger = get_logger()
        self.conversation_logger = get_conversation_logger(self.agent_id)

    async def some_method(self):
        # User-facing conversation logging
        self.conversation_logger.user_query("Create an Excel report")
        self.conversation_logger.agent_thinking("Planning the report structure...")
        self.conversation_logger.tool_usage("revolutionary_universal_excel_tool", "Creating spreadsheet")
        self.conversation_logger.success("Report created successfully!")

        # Backend technical logging
        self.backend_logger.info(
            "Document created",
            LogCategory.AGENT_OPERATIONS,
            "YourAgent",
            data={"filename": "report.xlsx", "sheets": 3}
        )
```

### Step 3: Done!

That's it! No YAML configuration needed in your agent file. The logging system is already configured globally.

---

## Global Configuration

**All logging configuration is done in ONE place**: `config/logging.yaml`

This file controls logging for ALL agents:

```yaml
logging:
  # ===== CONVERSATION LOGGING (User-Facing) =====
  conversation:
    enabled: true                    # Enable clean conversation logging
    emoji_enhanced: true             # Use emojis (🧑 User, 🤖 Agent, 🔧 Tool)
    show_reasoning: true             # Show agent thinking/reasoning
    show_tool_usage: true            # Show tool usage messages
    show_tool_results: true          # Show tool results
    max_reasoning_length: 200        # Truncate long reasoning messages
    max_result_length: 500           # Truncate long tool results
    
  # ===== BACKEND LOGGING (Technical) =====
  backend:
    enabled: true                    # Enable backend technical logging
    log_level: "INFO"                # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # What to log
    log_tool_usage: true             # Log tool execution details
    log_decision_process: true       # Log agent decision-making
    log_llm_calls: true              # Log LLM API calls
    log_memory_operations: true      # Log memory read/write
    log_rag_operations: true         # Log RAG queries
    log_performance_metrics: true    # Log performance data
    
    # Console output control
    console_enabled: true            # Show logs in console
    console_level: "INFO"            # Console log level
    
    # File output control
    file_enabled: true               # Write logs to files
    file_level: "DEBUG"              # File log level (more verbose)
    
  # ===== MODULE-SPECIFIC LOGGING =====
  modules:
    # Agent operations (this agent's core logic)
    agents:
      enabled: true
      console_level: "INFO"
      file_level: "DEBUG"
      
    # Tool operations
    tools:
      enabled: true
      console_level: "INFO"
      file_level: "DEBUG"
      
    # LLM operations
    llm:
      enabled: true
      console_level: "INFO"
      file_level: "DEBUG"
      
    # Memory operations
    memory:
      enabled: true
      console_level: "WARNING"       # Only show warnings/errors
      file_level: "INFO"
      
    # RAG operations
    rag:
      enabled: true
      console_level: "WARNING"       # Only show warnings/errors
      file_level: "INFO"
      
  # ===== LOGGING MODE =====
  mode: "developer"                  # user, developer, debug
  
  # ===== CORRELATION TRACKING =====
  correlation:
    enabled: true                    # Enable correlation ID tracking
    include_session_id: true         # Include session ID in logs
    include_agent_id: true           # Include agent ID in logs
    include_component: true          # Include component name in logs
    
  # ===== PERFORMANCE LOGGING =====
  performance:
    enabled: true                    # Enable performance logging
    log_duration: true               # Log operation duration
    log_memory_usage: true           # Log memory usage
    log_cpu_usage: false             # Log CPU usage (can be expensive)
    
  # ===== AGENT-SPECIFIC LOGGING =====
  agent_specific:
    # Custom logging for your agent's specific operations
    log_document_creation: true      # Example: Log document creation
    log_tool_selection: true         # Example: Log tool selection decisions
    log_workflow_steps: true         # Example: Log workflow steps
```

---

## Conversation Logging

Conversation logging provides clean, user-facing output without technical details.

### Configuration

```yaml
logging:
  conversation:
    enabled: true                    # Enable/disable conversation logging
    emoji_enhanced: true             # Use emojis for visual appeal
    show_reasoning: true             # Show agent thinking process
    show_tool_usage: true            # Show when tools are used
    show_tool_results: true          # Show tool execution results
    max_reasoning_length: 200        # Truncate long reasoning (0 = no limit)
    max_result_length: 500           # Truncate long results (0 = no limit)
```

### Available Message Types

```python
# User interaction
conversation_logger.user_query("Your message")

# Agent responses
conversation_logger.agent_acknowledgment("I'll help you with that!")
conversation_logger.agent_thinking("Analyzing the requirements...")
conversation_logger.agent_response("Here's what I found...")
conversation_logger.agent_goal("My goal is to...")
conversation_logger.agent_decision("I've decided to...")

# Tool usage
conversation_logger.tool_usage("tool_name", "What I'm doing")
conversation_logger.tool_result("Result of the tool")

# Status messages
conversation_logger.progress("Processing... 50%")
conversation_logger.info("Additional information")
conversation_logger.success("Operation completed!")
conversation_logger.warning("Something to be aware of")
conversation_logger.error("An error occurred")
```

### Output Examples

**With emoji_enhanced: true**
```
🧑 User: Create an Excel report with sales data
🤖 Agent: I'll create an Excel report for you!
🔍 Thinking: Planning the report structure with charts and tables...
🔧 Using: revolutionary_universal_excel_tool - Creating spreadsheet
✅ Created Sales_Report.xlsx with 3 sheets and 5 charts
💬 Agent: I've created your sales report with interactive dashboards!
```

**With emoji_enhanced: false**
```
User: Create an Excel report with sales data
Agent: I'll create an Excel report for you!
Thinking: Planning the report structure with charts and tables...
Using: revolutionary_universal_excel_tool - Creating spreadsheet
Created Sales_Report.xlsx with 3 sheets and 5 charts
Agent: I've created your sales report with interactive dashboards!
```

---

## Backend Logging

Backend logging provides structured technical logs for debugging and monitoring.

### Configuration

```yaml
logging:
  backend:
    enabled: true                    # Enable backend logging
    log_level: "INFO"                # Minimum log level
    
    # What to log
    log_tool_usage: true             # Tool execution details
    log_decision_process: true       # Decision-making process
    log_llm_calls: true              # LLM API calls
    log_memory_operations: true      # Memory operations
    log_rag_operations: true         # RAG queries
    log_performance_metrics: true    # Performance metrics
    
    # Output control
    console_enabled: true            # Console output
    console_level: "INFO"            # Console log level
    file_enabled: true               # File output
    file_level: "DEBUG"              # File log level
```

### Usage in Code

```python
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory

backend_logger = get_logger()

# Log agent operations
backend_logger.info(
    "Agent task started",
    LogCategory.AGENT_OPERATIONS,
    "YourAgent",
    data={"task": "create_report", "user_id": "123"}
)

# Log tool usage
backend_logger.info(
    "Tool executed successfully",
    LogCategory.TOOL_OPERATIONS,
    "ExcelTool",
    data={"tool": "revolutionary_universal_excel_tool", "duration_ms": 1250}
)

# Log errors
backend_logger.error(
    "Failed to create document",
    LogCategory.AGENT_OPERATIONS,
    "YourAgent",
    data={"error": "File not found", "path": "/data/template.xlsx"}
)
```

### Log Categories

- `AGENT_OPERATIONS` - Agent core operations
- `TOOL_OPERATIONS` - Tool execution
- `LLM_OPERATIONS` - LLM API calls
- `MEMORY_OPERATIONS` - Memory read/write
- `RAG_OPERATIONS` - RAG queries
- `API_LAYER` - API requests/responses
- `DATABASE_LAYER` - Database operations
- `EXTERNAL_INTEGRATIONS` - External service calls
- `SECURITY_EVENTS` - Security-related events
- `PERFORMANCE` - Performance metrics
- `ERROR_TRACKING` - Error tracking

---

## Module-Specific Logging

Control logging for specific backend modules independently.

### Configuration

```yaml
logging:
  modules:
    # Enable/disable and set levels for each module
    agents:
      enabled: true
      console_level: "INFO"
      file_level: "DEBUG"
      
    tools:
      enabled: true
      console_level: "INFO"
      file_level: "DEBUG"
      
    llm:
      enabled: true
      console_level: "WARNING"       # Only warnings and errors
      file_level: "INFO"
      
    memory:
      enabled: false                 # Completely disable
      
    rag:
      enabled: true
      console_level: "ERROR"         # Only errors
      file_level: "INFO"
```

### Available Modules

- `agents` - Agent operations (app.agents)
- `tools` - Tool execution (app.tools)
- `llm` - LLM operations (app.llm)
- `memory` - Memory system (app.memory)
- `rag` - RAG system (app.rag)
- `api` - API layer (app.api)
- `core` - Core orchestration (app.core)
- `services` - Business logic (app.services)
- `orchestration` - Workflow management (app.orchestration)
- `communication` - Agent communication (app.communication)
- `config` - Configuration (app.config)
- `models` - Database models (app.models)
- `optimization` - Performance optimization (app.optimization)
- `integrations` - External integrations (app.integrations)

---

## Logging Modes

Three logging modes provide different levels of detail.

### USER Mode

**Purpose**: Clean conversation only, no technical logs  
**Use Case**: End-user interactions, production deployments

```yaml
logging:
  mode: "user"
```

**Output**:
```
🧑 User: Create a sales report
🤖 Agent: I'll create that for you!
✅ Report created successfully!
```

### DEVELOPER Mode

**Purpose**: Conversation + selected module logs  
**Use Case**: Development, debugging specific modules

```yaml
logging:
  mode: "developer"
  modules:
    agents: {enabled: true, console_level: "INFO"}
    tools: {enabled: true, console_level: "INFO"}
```

**Output**:
```
🧑 User: Create a sales report
[13:45:23] [INFO] [app.agents] Agent task started
🤖 Agent: I'll create that for you!
[13:45:24] [INFO] [app.tools] Tool execution started
✅ Report created successfully!
[13:45:25] [INFO] [app.tools] Tool execution completed
```

### DEBUG Mode

**Purpose**: Full verbose logging with all metadata  
**Use Case**: Deep debugging, troubleshooting

```yaml
logging:
  mode: "debug"
```

**Output**:
```
[2025-10-01 13:45:23.123] [INFO] [app.agents.base.agent:495] Agent task started | correlation_id=abc123 | session_id=xyz789 | agent_id=your_agent
🧑 User: Create a sales report
🤖 Agent: I'll create that for you!
[2025-10-01 13:45:24.456] [DEBUG] [app.tools.executor:234] Tool execution started | tool=excel_tool | correlation_id=abc123
✅ Report created successfully!
```

---

## Agent-Specific Logging

Define custom logging for your agent's specific operations.

```yaml
logging:
  agent_specific:
    # Document agent example
    log_document_creation: true
    log_document_path: true
    log_document_metadata: true
    
    # Tool selection example
    log_tool_selection: true
    log_tool_confidence: true
    
    # Workflow example
    log_workflow_steps: true
    log_planning_phase: true
    log_execution_phase: true
```

Use in code:

```python
if self.logging_config.get('agent_specific', {}).get('log_document_creation', False):
    self.backend_logger.info(
        "Document created",
        LogCategory.AGENT_OPERATIONS,
        "YourAgent",
        data={"filename": filename, "type": "excel"}
    )
```

---

## Code Integration

### Complete Agent Integration Example

```python
from pathlib import Path
import yaml
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel
from app.core.clean_logging import get_conversation_logger

class YourAgent:
    def __init__(self):
        self.agent_id = "your_agent_id"
        self.logging_config = None
        
        # Initialize loggers
        self.backend_logger = get_logger()
        self.conversation_logger = get_conversation_logger(self.agent_id)
        
    async def _load_logging_config(self):
        """Load logging configuration from YAML."""
        try:
            config_path = Path(__file__).parent.parent / "config" / "agents" / f"{self.agent_id}.yaml"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.logging_config = config.get('logging', {})
                    
                    # Apply module configurations
                    modules_config = self.logging_config.get('modules', {})
                    for module_name, module_settings in modules_config.items():
                        if module_settings.get('enabled', True):
                            console_level = module_settings.get('console_level', 'INFO')
                            self.backend_logger.module_controller.enable_module(
                                f"app.{module_name}",
                                LogLevel[console_level]
                            )
        except Exception as e:
            self.backend_logger.warning(
                f"Failed to load logging config: {e}",
                LogCategory.AGENT_OPERATIONS,
                self.agent_id
            )
    
    async def initialize(self):
        """Initialize the agent."""
        # Load logging config first
        await self._load_logging_config()
        
        # Use conversation logger for user-facing messages
        self.conversation_logger.agent_acknowledgment("Initializing agent...")
        
        # Use backend logger for technical tracking
        self.backend_logger.info(
            "Agent initialization started",
            LogCategory.AGENT_OPERATIONS,
            self.agent_id
        )
        
        # ... rest of initialization ...
        
        self.conversation_logger.success("Agent initialized successfully!")
        self.backend_logger.info(
            "Agent initialization completed",
            LogCategory.AGENT_OPERATIONS,
            self.agent_id
        )
```

---

## Examples

See the complete example in:
- **Agent Code**: `data/agents/universal_document_master_agent.py`
- **Agent Config**: `data/config/agents/universal_document_master_agent.yaml`

---

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors that may cause system failure

### 2. Separate User-Facing and Technical Logs

- Use **conversation logger** for user-facing messages
- Use **backend logger** for technical details

### 3. Include Relevant Context

```python
# Good - includes context
backend_logger.info(
    "Document created",
    LogCategory.AGENT_OPERATIONS,
    "DocumentAgent",
    data={"filename": "report.xlsx", "sheets": 3, "duration_ms": 1250}
)

# Bad - no context
backend_logger.info("Document created", LogCategory.AGENT_OPERATIONS, "DocumentAgent")
```

### 4. Use Module Control Wisely

- Enable only modules you need to debug
- Use WARNING/ERROR levels for noisy modules
- Disable modules you don't need

### 5. Choose the Right Mode

- **Production**: USER mode
- **Development**: DEVELOPER mode with selected modules
- **Debugging**: DEBUG mode temporarily

### 6. Log Performance Metrics

```python
import time

start_time = time.time()
# ... operation ...
duration_ms = (time.time() - start_time) * 1000

backend_logger.info(
    "Operation completed",
    LogCategory.AGENT_OPERATIONS,
    "YourAgent",
    data={"operation": "create_report", "duration_ms": duration_ms}
)
```

### 7. Use Correlation IDs

Correlation IDs help track requests across the system:

```python
from app.backend_logging.context import CorrelationContext

CorrelationContext.update_context(
    correlation_id="unique-id",
    session_id="session-id",
    agent_id=self.agent_id
)
```

---

## Summary

The Revolutionary Logging System provides:

✅ **Clean user-facing conversation output**  
✅ **Granular module control** for 14 backend systems  
✅ **Three logging modes** (USER, DEVELOPER, DEBUG)  
✅ **Per-agent configuration** via YAML  
✅ **Runtime configuration** without restart  
✅ **Complete audit trail** with file persistence  
✅ **Performance optimized** (async, buffered, zero overhead)  

Configure logging in your agent's YAML file and integrate the loggers in your code for a powerful, flexible logging experience!

