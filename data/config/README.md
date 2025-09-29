# Agent Configuration System

This directory contains the centralized configuration system for AI agents, designed to replace hardcoded values throughout the codebase with flexible, validated, and layered configuration management.

## 🎯 **Problem Solved**

The original codebase suffered from:
- **Technology Lock-in**: Hardcoded model names (`llama3.2:latest`) and providers (`ollama`)
- **Performance Issues**: Scattered hardcoded timeouts, iterations, and thresholds
- **Inflexible System Prompts**: Long hardcoded prompts that couldn't be customized
- **Infrastructure Brittleness**: Hardcoded connection strings and health check intervals
- **Missing Safety Rails**: No validation, error recovery, or security constraints

## 📁 **File Structure**

```
data/config/
├── README.md                    # This documentation
├── agent_defaults.yaml          # Smart defaults for all configurations
├── user_config_template.yaml    # Template for user customizations
├── user_config.yaml            # User overrides (create from template)
├── global_config.json          # Legacy configuration (maintained for compatibility)
├── .env.template               # Environment variable template
├── .env                        # Environment variables (create from template)
└── migration_report.md         # Generated migration analysis
```

```
app/config/
├── agent_config_manager.py     # Core configuration management system
├── agent_config_integration.py # Integration with existing agent system
└── config_migration.py         # Migration utilities and validation
```

## 🏗️ **Configuration Layers**

The system uses a layered approach with clear precedence:

1. **Defaults** (`agent_defaults.yaml`) - Smart defaults for all settings
2. **Environment** (`.env` + environment variables) - Deployment-specific overrides
3. **User Config** (`user_config.yaml`) - User customizations
4. **Runtime** - Programmatic overrides during execution

Higher layers override lower layers.

## 🚀 **Quick Start**

### 1. Create User Configuration

```bash
# Copy the template
cp data/config/user_config_template.yaml data/config/user_config.yaml

# Edit to customize your settings
nano data/config/user_config.yaml
```

### 2. Set Environment Variables

```bash
# Copy the template
cp data/config/.env.template data/config/.env

# Edit with your environment-specific settings
nano data/config/.env
```

### 3. Use in Your Code

```python
from app.config.agent_config_integration import get_config_integration

# Get configuration integration
config_integration = get_config_integration()

# Create agent config using configuration system
agent_config = config_integration.create_agent_config(
    name="My Research Agent",
    description="Intelligent research assistant",
    agent_type="research",
    # All other values come from configuration system
)

# Create builder config
builder_config = config_integration.create_builder_config(
    name="My Custom Agent",
    description="Custom agent with configuration",
    agent_type=AgentType.REACT,
    # Configuration system provides smart defaults
)
```

## ⚙️ **Configuration Categories**

### **LLM Providers**
```yaml
llm_providers:
  default_provider: "ollama"  # Instead of hardcoded
  ollama:
    default_model: "llama3.2:latest"
    temperature: 0.7
    max_tokens: 2048
    timeout_seconds: 300
```

### **Agent Types**
```yaml
agent_types:
  react:
    framework: "react"
    default_temperature: 0.7
    max_iterations: 50        # Instead of hardcoded 3
    timeout_seconds: 300
    enable_memory: true
```

### **Performance Limits**
```yaml
performance:
  max_execution_time_seconds: 3600  # Hard safety limit
  max_iterations_hard_limit: 200
  max_memory_per_agent_mb: 1024
  default_decision_threshold: 0.6   # Instead of hardcoded
```

### **System Prompts**
```yaml
system_prompts:
  base_template: |
    You are an intelligent AI agent with access to powerful tools.
    AVAILABLE TOOLS: {tools}
    # Templatable and customizable
```

### **Security & Safety**
```yaml
security:
  requests_per_minute: 60           # Rate limiting
  max_file_size_mb: 100            # File upload limits
  allowed_file_types: [".txt", ".pdf", ".md"]
```

## 🔧 **Environment Variables**

Override any configuration with environment variables:

```bash
# LLM Settings
AGENT_DEFAULT_PROVIDER=openai
AGENT_DEFAULT_MODEL=gpt-4
AGENT_DEFAULT_TEMPERATURE=0.5

# Performance Settings
AGENT_MAX_ITERATIONS=100
AGENT_MAX_EXECUTION_TIME=1800
AGENT_MAX_MEMORY_MB=512

# Security Settings
AGENT_RATE_LIMIT_PER_MINUTE=30
AGENT_MAX_FILE_SIZE_MB=50
```

## 🛡️ **Validation & Safety**

The system includes comprehensive validation:

```python
# Automatic validation
config_manager = get_agent_config_manager()
errors = config_manager.validate_configuration()

if errors:
    print("Configuration errors:", errors)
    # Handle validation failures

# Get validated constraints
constraints = config_integration.get_validation_constraints()
# Returns: max_execution_time, max_iterations, temperature_range, etc.
```

## 📊 **Migration from Hardcoded Values**

### Run Migration Analysis
```python
from app.config.config_migration import run_configuration_migration

# Analyze current hardcoded values and generate migration files
run_configuration_migration()
```

### Migration Report
The system generates a detailed report of:
- ❌ Hardcoded values that should be configurable
- ✅ Strategic hardcoding that should remain
- 🔧 Recommended configuration changes
- 📋 Validation results

## 🎛️ **Advanced Usage**

### Custom Validation Rules
```python
from app.config.agent_config_manager import ConfigValidationRule

# Add custom validation
rule = ConfigValidationRule(
    field_path="custom.my_setting",
    rule_type="range", 
    constraint=(1, 100),
    error_message="My setting must be between 1 and 100"
)
```

### Hot Reloading
```python
# Reload configuration without restart
config_manager.reload_configuration()
```

### Configuration Watching
```python
# Monitor configuration files for changes
# (Implementation can be added for production use)
```

## 🔍 **Troubleshooting**

### Common Issues

1. **Configuration Not Loading**
   ```bash
   # Check file permissions
   ls -la data/config/
   
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('data/config/agent_defaults.yaml'))"
   ```

2. **Environment Variables Not Working**
   ```bash
   # Check environment variable names (must match exactly)
   env | grep AGENT_
   
   # Verify data types (true/false for booleans, numbers for numeric values)
   ```

3. **Validation Errors**
   ```python
   # Get detailed validation errors
   from app.config.agent_config_manager import get_agent_config_manager
   
   config_manager = get_agent_config_manager()
   errors = config_manager.validate_configuration()
   for error in errors:
       print(f"❌ {error}")
   ```

## 🚀 **Benefits Achieved**

✅ **Flexibility**: Easy to change models, providers, and parameters
✅ **Safety**: Validation prevents invalid configurations  
✅ **Consistency**: Centralized configuration eliminates scattered hardcoded values
✅ **Environment Support**: Different settings for dev/staging/production
✅ **User Customization**: Easy to override defaults without code changes
✅ **Migration Path**: Clear path from hardcoded to configurable values

## 📈 **Next Steps**

1. **Update Agent Code**: Replace hardcoded values with configuration system calls
2. **Add More Templates**: Create agent-specific prompt templates
3. **Enhance Validation**: Add more sophisticated validation rules
4. **Monitoring Integration**: Add configuration change monitoring
5. **Documentation**: Document all configuration options

## 🤝 **Contributing**

When adding new configuration options:

1. Add defaults to `agent_defaults.yaml`
2. Add environment variable mapping in `agent_config_manager.py`
3. Add validation rules if needed
4. Update templates and documentation
5. Test with migration utilities
