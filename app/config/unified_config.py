"""
Unified Configuration System - Single Source of Truth.

This module provides a unified configuration system that consolidates all
settings into logical groups, validates them at startup, and provides
easy access throughout the application.
"""

import os
from typing import Optional
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings

from app.config.config_groups import (
    ServerConfig,
    DatabaseConfig,
    RedisConfig,
    LLMProviderConfig,
    RAGConfig,
    SecurityConfig,
    PerformanceConfig,
    LoggingConfig,
    ConfigProfile
)
from app.config.config_validator import ConfigValidator


class UnifiedConfig(BaseSettings):
    """
    Unified configuration system that consolidates all settings.
    
    This replaces the scattered settings across multiple files and provides
    a single source of truth for all configuration.
    """
    
    # Configuration Groups
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables (backward compatibility)

        # Allow nested configuration from environment variables
        # e.g., SERVER__PORT=8080 sets server.port
        env_nested_delimiter = "__"
    
    async def validate(self, skip_connectivity: bool = False) -> bool:
        """
        Validate all configuration settings.
        
        Args:
            skip_connectivity: Skip connectivity checks (useful for testing)
        
        Returns:
            True if all critical validations passed
        """
        validator = ConfigValidator(
            server_config=self.server,
            database_config=self.database,
            redis_config=self.redis,
            llm_config=self.llm,
            rag_config=self.rag,
            security_config=self.security,
            performance_config=self.performance,
            logging_config=self.logging
        )
        
        all_passed, results = await validator.validate_all(skip_connectivity=skip_connectivity)
        validator.print_results()
        
        return all_passed
    
    def get_profile_name(self) -> str:
        """Get the current configuration profile name."""
        return self.server.environment.value
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.server.environment == ConfigProfile.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.server.environment == ConfigProfile.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.server.environment == ConfigProfile.TESTING
    
    def get_enabled_llm_providers(self) -> list:
        """Get list of enabled LLM providers."""
        return self.llm.get_enabled_providers()
    
    def export_to_dict(self) -> dict:
        """Export configuration to dictionary (for debugging)."""
        return {
            "server": self.server.dict(),
            "database": {
                **self.database.dict(),
                "database_url": "***REDACTED***"  # Don't expose credentials
            },
            "redis": {
                **self.redis.dict(),
                "redis_url": "***REDACTED***"
            },
            "llm": {
                **self.llm.dict(),
                "openai_api_key": "***REDACTED***" if self.llm.openai_api_key else "",
                "anthropic_api_key": "***REDACTED***" if self.llm.anthropic_api_key else "",
                "google_api_key": "***REDACTED***" if self.llm.google_api_key else "",
            },
            "rag": self.rag.dict(),
            "security": {
                **self.security.dict(),
                "secret_key": "***REDACTED***",
                "keycloak_client_secret": "***REDACTED***" if self.security.keycloak_client_secret else None
            },
            "performance": self.performance.dict(),
            "logging": self.logging.dict()
        }


@lru_cache()
def get_config() -> UnifiedConfig:
    """
    Get the unified configuration instance (cached).
    
    This function is cached to ensure we only create one instance
    of the configuration throughout the application lifecycle.
    """
    return UnifiedConfig()


# Backward compatibility: Create settings instance
# This allows existing code to continue using `from app.config.settings import settings`
settings = get_config()


# Export commonly used settings for convenience
def get_database_url() -> str:
    """Get database URL."""
    return get_config().database.database_url_async


def get_redis_url() -> str:
    """Get Redis URL."""
    return get_config().redis.redis_url


def get_secret_key() -> str:
    """Get JWT secret key."""
    return get_config().security.secret_key


def is_production() -> bool:
    """Check if running in production."""
    return get_config().is_production()


def is_development() -> bool:
    """Check if running in development."""
    return get_config().is_development()


# Configuration documentation generator
def generate_config_documentation() -> str:
    """
    Generate comprehensive configuration documentation.
    
    Returns:
        Markdown-formatted documentation of all configuration options
    """
    doc = """# Configuration Documentation

## Overview

The Agentic AI Engine uses a unified configuration system organized into logical groups.
Configuration can be set via environment variables or a `.env` file.

## Environment Variables

Environment variables use the format: `GROUP__SETTING_NAME`

Examples:
- `SERVER__PORT=8080` sets the server port
- `DATABASE__POOL_SIZE=50` sets the database pool size
- `LLM__OLLAMA_ENABLED=true` enables Ollama provider

## Configuration Groups

### 1. Server Configuration (`SERVER__`)

Application and server settings.

| Setting | Default | Description |
|---------|---------|-------------|
| `SERVER__APP_NAME` | "Agentic AI Microservice" | Application name |
| `SERVER__VERSION` | "0.1.0" | Application version |
| `SERVER__DEBUG` | false | Debug mode |
| `SERVER__ENVIRONMENT` | "development" | Environment: development, staging, production, testing |
| `SERVER__HOST` | "0.0.0.0" | Server host |
| `SERVER__PORT` | 8888 | Server port (1024-65535) |
| `SERVER__WORKERS` | 1 | Number of worker processes |
| `SERVER__BASE_URL` | "http://localhost:8888" | Base URL for callbacks |
| `SERVER__CORS_ORIGINS` | "http://localhost:3000,..." | Comma-separated CORS origins |

### 2. Database Configuration (`DATABASE__`)

PostgreSQL database settings with connection pooling.

| Setting | Default | Description |
|---------|---------|-------------|
| `DATABASE__DATABASE_URL` | "postgresql://..." | Database connection URL |
| `DATABASE__POOL_SIZE` | 50 | Connection pool size (5-100) |
| `DATABASE__POOL_MAX_OVERFLOW` | 20 | Pool max overflow (0-50) |
| `DATABASE__POOL_TIMEOUT` | 30 | Pool timeout in seconds (5-120) |
| `DATABASE__POOL_RECYCLE` | 3600 | Pool recycle time in seconds (300-7200) |

### 3. Redis Configuration (`REDIS__`)

Redis caching and state management.

| Setting | Default | Description |
|---------|---------|-------------|
| `REDIS__REDIS_URL` | "redis://localhost:6379/0" | Redis connection URL |
| `REDIS__POOL_SIZE` | 10 | Redis connection pool size (5-50) |

### 4. LLM Provider Configuration (`LLM__`)

Multi-provider LLM configuration.

#### Ollama
| Setting | Default | Description |
|---------|---------|-------------|
| `LLM__OLLAMA_ENABLED` | true | Enable Ollama provider |
| `LLM__OLLAMA_BASE_URL` | "http://localhost:11434" | Ollama base URL |
| `LLM__OLLAMA_TIMEOUT` | 120 | Timeout in seconds (30-300) |
| `LLM__OLLAMA_RETRY_ATTEMPTS` | 3 | Retry attempts (1-10) |
| `LLM__OLLAMA_KEEP_ALIVE` | "30m" | Model keep-alive duration |

#### OpenAI
| Setting | Default | Description |
|---------|---------|-------------|
| `LLM__OPENAI_ENABLED` | false | Enable OpenAI provider |
| `LLM__OPENAI_API_KEY` | "" | OpenAI API key |
| `LLM__OPENAI_BASE_URL` | "https://api.openai.com/v1" | OpenAI base URL |
| `LLM__OPENAI_ORGANIZATION` | "" | OpenAI organization ID |
| `LLM__OPENAI_TIMEOUT` | 60 | Timeout in seconds (30-300) |

#### Anthropic
| Setting | Default | Description |
|---------|---------|-------------|
| `LLM__ANTHROPIC_ENABLED` | false | Enable Anthropic provider |
| `LLM__ANTHROPIC_API_KEY` | "" | Anthropic API key |
| `LLM__ANTHROPIC_BASE_URL` | "https://api.anthropic.com" | Anthropic base URL |
| `LLM__ANTHROPIC_TIMEOUT` | 60 | Timeout in seconds (30-300) |

#### Google
| Setting | Default | Description |
|---------|---------|-------------|
| `LLM__GOOGLE_ENABLED` | false | Enable Google provider |
| `LLM__GOOGLE_API_KEY` | "" | Google API key |
| `LLM__GOOGLE_BASE_URL` | "https://generativelanguage.googleapis.com/v1beta" | Google base URL |
| `LLM__GOOGLE_TIMEOUT` | 60 | Timeout in seconds (30-300) |

#### Defaults
| Setting | Default | Description |
|---------|---------|-------------|
| `LLM__DEFAULT_PROVIDER` | "ollama" | Default LLM provider |
| `LLM__DEFAULT_MODEL` | "llama3.1:8b" | Default LLM model |
| `LLM__BACKUP_PROVIDER` | "ollama" | Backup LLM provider |
| `LLM__BACKUP_MODEL` | "llama3.2:latest" | Backup LLM model |

### 5. RAG Configuration (`RAG__`)

ChromaDB and RAG system settings.

| Setting | Default | Description |
|---------|---------|-------------|
| `RAG__CHROMA_PERSIST_DIRECTORY` | "./data/chroma" | ChromaDB persistence directory |
| `RAG__CHROMA_COLLECTION_NAME` | "knowledge_base" | Default collection name |
| `RAG__CHROMA_DISTANCE_METRIC` | "cosine" | Distance metric |
| `RAG__CHROMA_PRODUCTION_MODE` | false | Enable production features |
| `RAG__CHROMA_CLUSTER_MODE` | "standalone" | Cluster mode |
| `RAG__CHROMA_BACKUP_ENABLED` | true | Enable backup system |
| `RAG__CHROMA_BACKUP_INTERVAL_HOURS` | 6 | Backup interval (1-24) |
| `RAG__CHROMA_MONITORING_ENABLED` | true | Enable monitoring |
| `RAG__CHROMA_MAX_MEMORY_MB` | 4096 | Max memory in MB (512-16384) |
| `RAG__CHROMA_CONNECTION_POOL_SIZE` | 50 | Connection pool size (10-100) |
| `RAG__EMBEDDING_MODEL` | "all-MiniLM-L6-v2" | Default embedding model |
| `RAG__CHUNK_SIZE` | 1000 | Document chunk size (100-4000) |
| `RAG__CHUNK_OVERLAP` | 200 | Chunk overlap size (0-1000) |

### 6. Security Configuration (`SECURITY__`)

Authentication and security settings.

| Setting | Default | Description |
|---------|---------|-------------|
| `SECURITY__SECRET_KEY` | "your-secret-key-change-this" | JWT secret key (CHANGE THIS!) |
| `SECURITY__ALGORITHM` | "HS256" | JWT algorithm |
| `SECURITY__ACCESS_TOKEN_EXPIRE_MINUTES` | 30 | Access token expiration (5-1440) |
| `SECURITY__SSO_ENABLED` | false | Enable SSO authentication |
| `SECURITY__KEYCLOAK_ENABLED` | false | Enable Keycloak SSO |
| `SECURITY__KEYCLOAK_SERVER_URL` | null | Keycloak server URL |
| `SECURITY__KEYCLOAK_REALM` | null | Keycloak realm |
| `SECURITY__KEYCLOAK_CLIENT_ID` | null | Keycloak client ID |
| `SECURITY__KEYCLOAK_CLIENT_SECRET` | null | Keycloak client secret |

### 7. Performance Configuration (`PERFORMANCE__`)

Performance and optimization settings.

| Setting | Default | Description |
|---------|---------|-------------|
| `PERFORMANCE__ASYNC_WORKER_COUNT` | 16 | Async worker count (4-64) |
| `PERFORMANCE__ASYNC_TASK_TIMEOUT` | 300 | Task timeout in seconds (60-3600) |
| `PERFORMANCE__ENABLE_ASYNC_PROCESSING` | true | Enable async processing |
| `PERFORMANCE__TASK_RESULT_TTL` | 3600 | Task result TTL (300-86400) |
| `PERFORMANCE__MAX_CONCURRENT_AGENTS` | 100 | Max concurrent agents (10-1000) |
| `PERFORMANCE__AGENT_TIMEOUT_SECONDS` | 300 | Agent timeout (60-3600) |
| `PERFORMANCE__WORKER_CONNECTIONS` | 2000 | Worker connections (100-10000) |
| `PERFORMANCE__MAX_REQUEST_SIZE` | 16777216 | Max request size in bytes |
| `PERFORMANCE__REQUEST_TIMEOUT_SECONDS` | 60 | Request timeout (10-300) |
| `PERFORMANCE__MAX_FILE_SIZE_MB` | 100 | Max file size (1-500) |
| `PERFORMANCE__ENABLE_PARALLEL_PROCESSING` | true | Enable parallel processing |
| `PERFORMANCE__DOCUMENT_PROCESSING_WORKERS` | 8 | Document workers (2-32) |

### 8. Logging Configuration (`LOGGING__`)

Logging and monitoring settings.

| Setting | Default | Description |
|---------|---------|-------------|
| `LOGGING__LOG_LEVEL` | "INFO" | Console log level |
| `LOGGING__LOG_TO_CONSOLE` | true | Enable console logging |
| `LOGGING__LOG_CONSOLE_FORMAT` | "simple" | Console format: simple, json, structured |
| `LOGGING__LOG_TO_FILE` | true | Enable file logging |
| `LOGGING__LOG_FILE_LEVEL` | "DEBUG" | File log level |
| `LOGGING__LOG_JSON_FORMAT` | true | Use JSON format for files |
| `LOGGING__LOG_RETENTION_DAYS` | 30 | Log retention period (1-365) |
| `LOGGING__LOG_MAX_FILE_SIZE_MB` | 100 | Max log file size (10-1000) |
| `LOGGING__LOG_MAX_FILES` | 10 | Max log files to keep (1-100) |
| `LOGGING__LOG_INCLUDE_REQUEST_BODY` | false | Include request body in logs |
| `LOGGING__LOG_INCLUDE_RESPONSE_BODY` | false | Include response body in logs |
| `LOGGING__LOG_EXCLUDE_PATHS` | "/health,/metrics,..." | Comma-separated paths to exclude |
| `LOGGING__ENABLE_METRICS` | true | Enable Prometheus metrics |
| `LOGGING__ENABLE_TRACING` | false | Enable OpenTelemetry tracing |
| `LOGGING__METRICS_PORT` | 9090 | Metrics server port (1024-65535) |

## Configuration Profiles

The system supports different configuration profiles:

- **development**: Local development with debug enabled
- **staging**: Pre-production testing environment
- **production**: Production environment with security hardening
- **testing**: Automated testing environment

Set the profile using: `SERVER__ENVIRONMENT=production`

## Validation

All configuration is validated at startup. The system will:
- ✅ Check all required settings are present
- ✅ Validate setting values are within acceptable ranges
- ✅ Test connectivity to external services (database, Redis, LLM providers)
- ✅ Warn about insecure configurations in production
- ✅ Fail to start if critical validations fail

## Example .env File

```env
# Server
SERVER__ENVIRONMENT=production
SERVER__PORT=8888
SERVER__DEBUG=false

# Database
DATABASE__DATABASE_URL=postgresql://user:pass@db-server:5432/agentic_ai
DATABASE__POOL_SIZE=50

# Redis
REDIS__REDIS_URL=redis://redis-server:6379/0

# LLM Providers
LLM__OLLAMA_ENABLED=true
LLM__OLLAMA_BASE_URL=http://ollama-server:11434

LLM__OPENAI_ENABLED=true
LLM__OPENAI_API_KEY=sk-...

# Security
SECURITY__SECRET_KEY=your-very-long-and-secure-secret-key-here

# Performance
PERFORMANCE__MAX_CONCURRENT_AGENTS=100
PERFORMANCE__ASYNC_WORKER_COUNT=16
```
"""
    return doc

