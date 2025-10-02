# 🔧 Configuration System Guide

## Overview

The Agentic AI Engine now features a **revolutionary unified configuration system** that consolidates all settings into logical groups, validates them at startup, and provides comprehensive documentation.

## Key Features

✅ **Organized Configuration Groups** - Settings grouped by functionality  
✅ **Startup Validation** - Comprehensive validation before accepting requests  
✅ **Environment Profiles** - Development, Staging, Production, Testing  
✅ **Auto-Documentation** - Self-documenting configuration  
✅ **Backward Compatible** - Existing code continues to work  
✅ **Type-Safe** - Pydantic validation for all settings  

---

## Configuration Groups

### 1. Server Configuration (`SERVER__`)
Application and server settings including host, port, workers, CORS.

### 2. Database Configuration (`DATABASE__`)
PostgreSQL settings with optimized connection pooling.

### 3. Redis Configuration (`REDIS__`)
Caching and state management settings.

### 4. LLM Provider Configuration (`LLM__`)
Multi-provider LLM configuration (Ollama, OpenAI, Anthropic, Google).

### 5. RAG Configuration (`RAG__`)
ChromaDB and RAG system settings.

### 6. Security Configuration (`SECURITY__`)
Authentication, JWT, and SSO settings.

### 7. Performance Configuration (`PERFORMANCE__`)
Async processing, concurrency, and optimization settings.

### 8. Logging Configuration (`LOGGING__`)
Logging levels, formats, and monitoring settings.

---

## Quick Start

### 1. Create `.env` File

```env
# Server
SERVER__ENVIRONMENT=production
SERVER__PORT=8888
SERVER__DEBUG=false

# Database
DATABASE__DATABASE_URL=postgresql://user:pass@localhost:5432/agentic_ai
DATABASE__POOL_SIZE=50

# Redis
REDIS__REDIS_URL=redis://localhost:6379/0

# LLM Providers
LLM__OLLAMA_ENABLED=true
LLM__OLLAMA_BASE_URL=http://localhost:11434

LLM__OPENAI_ENABLED=true
LLM__OPENAI_API_KEY=sk-your-key-here

# Security (IMPORTANT: Change this!)
SECURITY__SECRET_KEY=your-very-long-and-secure-secret-key-here

# Performance
PERFORMANCE__MAX_CONCURRENT_AGENTS=100
PERFORMANCE__ASYNC_WORKER_COUNT=16
```

### 2. Use in Code

```python
from app.config.unified_config import get_config

# Get configuration
config = get_config()

# Access settings
print(config.server.port)
print(config.database.pool_size)
print(config.llm.get_enabled_providers())

# Check environment
if config.is_production():
    print("Running in production mode")
```

### 3. Validate Configuration

```python
from app.config.unified_config import get_config

config = get_config()

# Validate all settings
is_valid = await config.validate()

if not is_valid:
    print("Configuration validation failed!")
    exit(1)
```

---

## Environment Variables Format

Use double underscore (`__`) to separate group and setting:

```
GROUP__SETTING_NAME=value
```

Examples:
- `SERVER__PORT=8080` → `config.server.port`
- `DATABASE__POOL_SIZE=50` → `config.database.pool_size`
- `LLM__OLLAMA_ENABLED=true` → `config.llm.ollama_enabled`

---

## Configuration Profiles

### Development Profile
```env
SERVER__ENVIRONMENT=development
SERVER__DEBUG=true
LOGGING__LOG_LEVEL=DEBUG
```

### Staging Profile
```env
SERVER__ENVIRONMENT=staging
SERVER__DEBUG=false
LOGGING__LOG_LEVEL=INFO
```

### Production Profile
```env
SERVER__ENVIRONMENT=production
SERVER__DEBUG=false
LOGGING__LOG_LEVEL=WARNING
SECURITY__SECRET_KEY=<strong-secret-key>
```

### Testing Profile
```env
SERVER__ENVIRONMENT=testing
DATABASE__DATABASE_URL=sqlite:///./test.db
```

---

## Validation System

The configuration system performs comprehensive validation at startup:

### Critical Validations (Must Pass)
- ✅ Database connectivity
- ✅ Required API keys for enabled providers
- ✅ Secret key is not default value
- ✅ ChromaDB directory is writable

### Warning Validations (Should Pass)
- ⚠️ Production settings (debug mode, CORS, etc.)
- ⚠️ Performance settings (pool sizes, worker counts)
- ⚠️ Security settings (token expiration, etc.)

### Info Validations (Informational)
- ℹ️ Configuration values
- ℹ️ Enabled features
- ℹ️ Resource availability

---

## Migration from Old Settings

### Before (Old Way)
```python
from app.config.settings import settings

database_url = settings.DATABASE_URL
pool_size = settings.DATABASE_POOL_SIZE
```

### After (New Way)
```python
from app.config.unified_config import get_config

config = get_config()
database_url = config.database.database_url
pool_size = config.database.pool_size
```

### Backward Compatibility
The old `settings` object still works:
```python
from app.config.unified_config import settings  # Still works!
```

---

## Advanced Features

### 1. Export Configuration
```python
config = get_config()
config_dict = config.export_to_dict()  # Credentials redacted
```

### 2. Generate Documentation
```python
from app.config.unified_config import generate_config_documentation

docs = generate_config_documentation()
print(docs)  # Markdown documentation
```

### 3. Custom Validation
```python
config = get_config()

# Skip connectivity checks (useful for testing)
is_valid = await config.validate(skip_connectivity=True)
```

---

## Complete Configuration Reference

See the auto-generated documentation:

```python
from app.config.unified_config import generate_config_documentation

print(generate_config_documentation())
```

Or run:
```bash
python -c "from app.config.unified_config import generate_config_documentation; print(generate_config_documentation())" > CONFIG_REFERENCE.md
```

---

## Troubleshooting

### Configuration Validation Fails

1. **Check validation output** - It shows exactly what failed
2. **Review critical issues** - Must be fixed before starting
3. **Address warnings** - Should be fixed for production

### Environment Variables Not Loading

1. **Check `.env` file location** - Must be in project root
2. **Verify variable names** - Use `GROUP__SETTING` format
3. **Check for typos** - Variable names are case-insensitive but must match

### Database Connection Fails

1. **Verify DATABASE__DATABASE_URL** - Check credentials and host
2. **Test connectivity** - Can you connect manually?
3. **Check firewall** - Is the port accessible?

### LLM Provider Issues

1. **Verify API keys** - Check they're correct and active
2. **Test provider URLs** - Can you reach the endpoints?
3. **Check enabled status** - Is the provider enabled?

---

## Best Practices

### 1. Use Environment-Specific Files
```
.env.development
.env.staging
.env.production
```

### 2. Never Commit Secrets
Add to `.gitignore`:
```
.env
.env.*
!.env.example
```

### 3. Validate on Startup
```python
# In main.py
config = get_config()
if not await config.validate():
    logger.error("Configuration validation failed")
    sys.exit(1)
```

### 4. Use Type Hints
```python
from app.config.unified_config import UnifiedConfig

def my_function(config: UnifiedConfig):
    # IDE autocomplete works!
    port = config.server.port
```

### 5. Document Custom Settings
```python
# Add comments in .env
# Server Configuration
SERVER__PORT=8888  # Application port
SERVER__WORKERS=4  # Number of worker processes
```

---

## Performance Optimizations

The unified configuration system includes several optimizations:

1. **Cached Configuration** - Single instance via `@lru_cache()`
2. **Lazy Loading** - Settings loaded only when accessed
3. **Validation Caching** - Validation results cached
4. **Type Conversion** - Automatic type conversion from env vars

---

## Security Considerations

### 1. Secret Key
**CRITICAL**: Change the default secret key!
```env
SECURITY__SECRET_KEY=<generate-strong-random-key>
```

Generate a strong key:
```python
import secrets
print(secrets.token_urlsafe(32))
```

### 2. API Keys
Store API keys securely:
- Use environment variables
- Never commit to version control
- Rotate regularly

### 3. Production Settings
Ensure these are set correctly:
```env
SERVER__DEBUG=false
SERVER__ENVIRONMENT=production
SECURITY__SECRET_KEY=<strong-key>
```

---

## Support

For issues or questions:
1. Check validation output
2. Review this guide
3. Check logs for detailed errors
4. Consult the auto-generated documentation

---

## Summary

The unified configuration system provides:

✅ **Better Organization** - Logical grouping of settings  
✅ **Validation** - Catch errors before they cause problems  
✅ **Documentation** - Self-documenting configuration  
✅ **Type Safety** - Pydantic validation  
✅ **Flexibility** - Multiple profiles and environments  
✅ **Backward Compatibility** - Existing code still works  

**Result**: Easier configuration, fewer errors, better developer experience!

