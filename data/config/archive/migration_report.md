# Configuration Migration Report
Generated at: unknown

## Current Configuration Status
✅ Configuration validation: PASSED

## Hardcoded Values Detected
🔴 **app/agents/base/agent.py:110**
   Issue: Hardcoded default model 'llama3.2:latest'
   Fix: Use config_manager.get_llm_config().get('default_model')

🔴 **app/agents/base/agent.py:1016**
   Issue: Hardcoded iteration limit of 3
   Fix: Use config_manager.get('performance.min_iterations', 3)

🟡 **app/agents/base/agent.py:271**
   Issue: Hardcoded tool_choice='any'
   Fix: Make tool choice configurable

🟡 **app/agents/factory/__init__.py:138**
   Issue: Hardcoded health check interval of 60 seconds
   Fix: Use config_manager.get('infrastructure.health_check_interval_seconds')

🔴 **app/agents/templates.py:multiple**
   Issue: Multiple hardcoded temperature values
   Fix: Use agent type specific temperature from config

## Recommended Actions

1. **High Priority**: Fix hardcoded model names and iteration limits
2. **Medium Priority**: Make infrastructure settings configurable
3. **Low Priority**: Add configuration validation for all parameters

## Configuration Files Status
✅ agent_defaults.yaml: Default configuration - EXISTS
❌ user_config.yaml: User overrides - MISSING
✅ global_config.json: Legacy configuration - EXISTS
❌ .env: Environment variables - MISSING