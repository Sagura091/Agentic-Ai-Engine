# Revolutionary Logging System - Quick Reference

## 🚀 Quick Start (3 Steps)

**Key Principle**: Agents USE loggers, they don't CONFIGURE them!

### 1. Import Loggers in Agent Code

```python
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory
from app.core.clean_logging import get_conversation_logger

# Loggers are already configured globally - just use them!
backend_logger = get_logger()
conversation_logger = get_conversation_logger("your_agent_id")
```

### 2. Use Loggers in Your Agent

```python
class YourAgent:
    def __init__(self):
        self.backend_logger = get_logger()
        self.conversation_logger = get_conversation_logger("your_agent_id")

    async def some_method(self):
        # User-facing
        self.conversation_logger.user_query("Create a report")
        self.conversation_logger.agent_response("Report created!")

        # Technical
        self.backend_logger.info("Task completed", LogCategory.AGENT_OPERATIONS, "YourAgent")
```

### 3. Done!

**No YAML configuration needed in agent files!** All logging is configured globally in `config/logging.yaml`, environment variables, or runtime API.

---

## 📝 Conversation Logger Methods

```python
# User interaction
conversation_logger.user_query("message")

# Agent responses
conversation_logger.agent_acknowledgment("message")
conversation_logger.agent_thinking("message")
conversation_logger.agent_response("message")
conversation_logger.agent_goal("message")
conversation_logger.agent_decision("message")

# Tool usage
conversation_logger.tool_usage("tool_name", "description")
conversation_logger.tool_result("result")

# Status
conversation_logger.progress("message")
conversation_logger.info("message")
conversation_logger.success("message")
conversation_logger.warning("message")
conversation_logger.error("message")
```

---

## 🔧 Backend Logger Usage

```python
# Basic logging
backend_logger.debug("message", LogCategory.AGENT_OPERATIONS, "Component")
backend_logger.info("message", LogCategory.AGENT_OPERATIONS, "Component")
backend_logger.warning("message", LogCategory.AGENT_OPERATIONS, "Component")
backend_logger.error("message", LogCategory.AGENT_OPERATIONS, "Component")
backend_logger.critical("message", LogCategory.AGENT_OPERATIONS, "Component")

# With data
backend_logger.info(
    "Operation completed",
    LogCategory.AGENT_OPERATIONS,
    "Component",
    data={"key": "value", "duration_ms": 1250}
)
```

---

## 📊 Log Categories

```python
LogCategory.AGENT_OPERATIONS          # Agent core operations
LogCategory.TOOL_OPERATIONS           # Tool execution
LogCategory.LLM_OPERATIONS            # LLM API calls
LogCategory.MEMORY_OPERATIONS         # Memory read/write
LogCategory.RAG_OPERATIONS            # RAG queries
LogCategory.API_LAYER                 # API requests
LogCategory.DATABASE_LAYER            # Database operations
LogCategory.EXTERNAL_INTEGRATIONS     # External services
LogCategory.SECURITY_EVENTS           # Security events
LogCategory.PERFORMANCE               # Performance metrics
LogCategory.ERROR_TRACKING            # Error tracking
```

---

## 🎯 Logging Modes

### USER Mode
```yaml
logging:
  mode: "user"
```
**Output**: Clean conversation only, no technical logs

### DEVELOPER Mode
```yaml
logging:
  mode: "developer"
  modules:
    agents: {enabled: true, console_level: "INFO"}
    tools: {enabled: true, console_level: "INFO"}
```
**Output**: Conversation + selected module logs

### DEBUG Mode
```yaml
logging:
  mode: "debug"
```
**Output**: Full verbose with correlation IDs and timestamps

---

## 🔌 Module Control

```yaml
logging:
  modules:
    agents:      {enabled: true,  console_level: "INFO",    file_level: "DEBUG"}
    tools:       {enabled: true,  console_level: "INFO",    file_level: "DEBUG"}
    llm:         {enabled: true,  console_level: "WARNING", file_level: "INFO"}
    memory:      {enabled: false}
    rag:         {enabled: true,  console_level: "ERROR",   file_level: "INFO"}
    api:         {enabled: true,  console_level: "INFO",    file_level: "DEBUG"}
    core:        {enabled: true,  console_level: "INFO",    file_level: "DEBUG"}
    services:    {enabled: true,  console_level: "INFO",    file_level: "DEBUG"}
```

**Available Modules**: agents, tools, llm, memory, rag, api, core, services, orchestration, communication, config, models, optimization, integrations

---

## ⚙️ Complete YAML Configuration

```yaml
logging:
  # Conversation (user-facing)
  conversation:
    enabled: true
    emoji_enhanced: true
    show_reasoning: true
    show_tool_usage: true
    show_tool_results: true
    max_reasoning_length: 200
    max_result_length: 500
    
  # Backend (technical)
  backend:
    enabled: true
    log_level: "INFO"
    log_tool_usage: true
    log_decision_process: true
    log_llm_calls: true
    log_memory_operations: true
    log_rag_operations: true
    log_performance_metrics: true
    console_enabled: true
    console_level: "INFO"
    file_enabled: true
    file_level: "DEBUG"
    
  # Module control
  modules:
    agents: {enabled: true, console_level: "INFO", file_level: "DEBUG"}
    tools:  {enabled: true, console_level: "INFO", file_level: "DEBUG"}
    llm:    {enabled: true, console_level: "INFO", file_level: "DEBUG"}
    
  # Mode
  mode: "developer"  # user, developer, debug
  
  # Correlation tracking
  correlation:
    enabled: true
    include_session_id: true
    include_agent_id: true
    include_component: true
    
  # Performance
  performance:
    enabled: true
    log_duration: true
    log_memory_usage: true
    log_cpu_usage: false
    
  # Agent-specific
  agent_specific:
    log_document_creation: true
    log_tool_selection: true
    log_workflow_steps: true
```

---

## 💻 Code Integration Pattern

```python
from pathlib import Path
import yaml
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel
from app.core.clean_logging import get_conversation_logger

class YourAgent:
    def __init__(self):
        self.agent_id = "your_agent_id"
        self.backend_logger = get_logger()
        self.conversation_logger = get_conversation_logger(self.agent_id)
        self.logging_config = None
        
    async def _load_logging_config(self):
        """Load logging config from YAML."""
        config_path = Path(__file__).parent.parent / "config" / "agents" / f"{self.agent_id}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logging_config = config.get('logging', {})
                
                # Apply module configs
                modules = self.logging_config.get('modules', {})
                for module_name, settings in modules.items():
                    if settings.get('enabled', True):
                        level = settings.get('console_level', 'INFO')
                        self.backend_logger.module_controller.enable_module(
                            f"app.{module_name}",
                            LogLevel[level]
                        )
    
    async def initialize(self):
        """Initialize agent."""
        await self._load_logging_config()
        
        self.conversation_logger.agent_acknowledgment("Initializing...")
        self.backend_logger.info(
            "Initialization started",
            LogCategory.AGENT_OPERATIONS,
            self.agent_id
        )
        
        # ... initialization code ...
        
        self.conversation_logger.success("Initialized!")
        self.backend_logger.info(
            "Initialization completed",
            LogCategory.AGENT_OPERATIONS,
            self.agent_id
        )
```

---

## 🌍 Environment Variables (Global Override)

```bash
# Global mode
LOG_MODE=developer                    # user, developer, debug

# Conversation layer
LOG_CONVERSATION_ENABLED=true
LOG_CONVERSATION_EMOJI=true

# Module control
LOG_MODULE_AGENTS=true
LOG_MODULE_TOOLS=true
LOG_MODULE_LLM=true
LOG_MODULE_MEMORY=false
LOG_MODULE_RAG=true

# Log levels
LOG_LEVEL_APP_AGENTS=INFO
LOG_LEVEL_APP_TOOLS=INFO
LOG_LEVEL_APP_LLM=WARNING
LOG_LEVEL_APP_RAG=WARNING
```

---

## 🔄 Runtime API (No Restart Required)

```bash
# Get status
curl http://localhost:8000/api/v1/admin/logging/status

# Change mode
curl -X POST http://localhost:8000/api/v1/admin/logging/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "debug"}'

# Enable module
curl -X POST http://localhost:8000/api/v1/admin/logging/module/enable \
  -H "Content-Type: application/json" \
  -d '{"module_name": "app.rag", "console_level": "DEBUG"}'

# Disable module
curl -X POST http://localhost:8000/api/v1/admin/logging/module/disable \
  -H "Content-Type: application/json" \
  -d '{"module_name": "app.memory"}'

# Reload config
curl -X POST http://localhost:8000/api/v1/admin/logging/reload
```

---

## 📁 Log Files Location

```
data/logs/backend/
├── agent_operations_YYYYMMDD.log
├── tool_operations_YYYYMMDD.log
├── llm_operations_YYYYMMDD.log
├── memory_operations_YYYYMMDD.log
├── rag_operations_YYYYMMDD.log
├── api_layer_YYYYMMDD.log
└── ... (18 categories total)
```

**Format**: JSON (one log entry per line)  
**Rotation**: Daily  
**Retention**: 30 days (configurable)

---

## 🎨 Output Examples

### USER Mode
```
🧑 User: Create an Excel report
🤖 Agent: I'll create that for you!
🔧 Using: revolutionary_universal_excel_tool
✅ Report created successfully!
```

### DEVELOPER Mode
```
🧑 User: Create an Excel report
[13:45:23] [INFO] [app.agents] Task started
🤖 Agent: I'll create that for you!
[13:45:24] [INFO] [app.tools] Tool execution started
🔧 Using: revolutionary_universal_excel_tool
[13:45:25] [INFO] [app.tools] Tool completed (1250ms)
✅ Report created successfully!
```

### DEBUG Mode
```
[2025-10-01 13:45:23.123] [INFO] [app.agents.base:495] Task started | correlation_id=abc123 | session_id=xyz789
🧑 User: Create an Excel report
🤖 Agent: I'll create that for you!
[2025-10-01 13:45:24.456] [DEBUG] [app.tools.executor:234] Tool execution | tool=excel_tool | correlation_id=abc123
🔧 Using: revolutionary_universal_excel_tool
[2025-10-01 13:45:25.789] [INFO] [app.tools.executor:267] Tool completed | duration_ms=1250 | correlation_id=abc123
✅ Report created successfully!
```

---

## 🎯 Best Practices

1. **Use conversation logger for user-facing messages**
2. **Use backend logger for technical details**
3. **Choose appropriate log levels** (DEBUG < INFO < WARNING < ERROR < CRITICAL)
4. **Include relevant context in data parameter**
5. **Enable only modules you need to debug**
6. **Use USER mode in production**
7. **Use DEVELOPER mode during development**
8. **Use DEBUG mode only for troubleshooting**
9. **Log performance metrics** (duration, memory usage)
10. **Use correlation IDs for request tracking**

---

## 📚 Full Documentation

- **Complete Guide**: `docs/logging/AGENT_LOGGING_GUIDE.md`
- **Architecture**: `docs/logging/ARCHITECTURE.md`
- **Configuration**: `docs/logging/CONFIGURATION_GUIDE.md`
- **API Reference**: `docs/logging/API_REFERENCE.md`
- **Migration Guide**: `docs/logging/MIGRATION_GUIDE.md`

---

## ✅ Checklist for New Agents

- [ ] Add `logging` section to agent YAML config
- [ ] Import loggers in agent code
- [ ] Load logging config in `__init__` or `initialize`
- [ ] Use conversation logger for user-facing messages
- [ ] Use backend logger for technical tracking
- [ ] Configure module-specific logging
- [ ] Choose appropriate logging mode
- [ ] Test all three modes (USER, DEVELOPER, DEBUG)
- [ ] Verify log files are created
- [ ] Check console output matches expectations

---

**The Revolutionary Logging System is ready to use!** 🎉

