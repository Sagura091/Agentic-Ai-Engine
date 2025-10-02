"""
Configuration Groups - Organized Settings Management.

This module provides organized configuration groups for better management
and validation of system settings. All settings are grouped logically
and validated at startup.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigProfile(str, Enum):
    """Configuration profiles for different environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ServerConfig(BaseModel):
    """Server and application configuration."""
    
    # Application
    app_name: str = Field(default="Agentic AI Microservice", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: ConfigProfile = Field(default=ConfigProfile.DEVELOPMENT, description="Environment profile")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8888, ge=1024, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of worker processes")
    base_url: str = Field(default="http://localhost:8888", description="Base URL for callbacks")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://localhost:5173", "http://localhost:8001"],
        description="Allowed CORS origins"
    )
    
    @field_validator("cors_origins", mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        use_enum_values = True


class DatabaseConfig(BaseModel):
    """Database configuration with connection pooling."""
    
    # Connection
    database_url: str = Field(
        default="postgresql://agentic_user:agentic_secure_password_2024@localhost:5432/agentic_ai",
        description="Database connection URL"
    )
    
    # Connection Pool (OPTIMIZED)
    pool_size: int = Field(default=50, ge=5, le=100, description="Connection pool size")
    pool_max_overflow: int = Field(default=20, ge=0, le=50, description="Pool max overflow")
    pool_timeout: int = Field(default=30, ge=5, le=120, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=300, le=7200, description="Pool recycle time in seconds")
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL."""
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        return self.database_url
    
    @property
    def database_url_sync(self) -> str:
        """Get sync database URL."""
        if self.database_url.startswith("postgresql+asyncpg://"):
            return self.database_url.replace("postgresql+asyncpg://", "postgresql://")
        return self.database_url
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "sqlite:///")):
            raise ValueError("DATABASE_URL must start with postgresql:// or sqlite:///")
        return v


class RedisConfig(BaseModel):
    """Redis configuration for caching and state management."""
    
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    pool_size: int = Field(default=10, ge=5, le=50, description="Redis connection pool size")
    
    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must start with redis://")
        return v


class LLMProviderConfig(BaseModel):
    """LLM provider configuration for multi-provider support."""
    
    # Ollama
    ollama_enabled: bool = Field(default=True, description="Enable Ollama provider")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_timeout: int = Field(default=120, ge=30, le=300, description="Ollama timeout in seconds")
    ollama_retry_attempts: int = Field(default=3, ge=1, le=10, description="Ollama retry attempts")
    ollama_keep_alive: str = Field(default="30m", description="Model keep-alive duration")
    
    # OpenAI
    openai_enabled: bool = Field(default=False, description="Enable OpenAI provider")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI base URL")
    openai_organization: str = Field(default="", description="OpenAI organization ID")
    openai_timeout: int = Field(default=60, ge=30, le=300, description="OpenAI timeout in seconds")
    
    # Anthropic
    anthropic_enabled: bool = Field(default=False, description="Enable Anthropic provider")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_base_url: str = Field(default="https://api.anthropic.com", description="Anthropic base URL")
    anthropic_timeout: int = Field(default=60, ge=30, le=300, description="Anthropic timeout in seconds")
    
    # Google
    google_enabled: bool = Field(default=False, description="Enable Google provider")
    google_api_key: str = Field(default="", description="Google API key")
    google_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", description="Google base URL")
    google_timeout: int = Field(default=60, ge=30, le=300, description="Google timeout in seconds")
    
    # Default Models
    default_provider: str = Field(default="ollama", description="Default LLM provider")
    default_model: str = Field(default="llama3.1:8b", description="Default LLM model")
    backup_provider: str = Field(default="ollama", description="Backup LLM provider")
    backup_model: str = Field(default="llama3.2:latest", description="Backup LLM model")
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers."""
        providers = []
        if self.ollama_enabled:
            providers.append("ollama")
        if self.openai_enabled and self.openai_api_key:
            providers.append("openai")
        if self.anthropic_enabled and self.anthropic_api_key:
            providers.append("anthropic")
        if self.google_enabled and self.google_api_key:
            providers.append("google")
        return providers


class RAGConfig(BaseModel):
    """RAG system configuration for knowledge management."""
    
    # ChromaDB
    chroma_persist_directory: str = Field(default="./data/chroma", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field(default="knowledge_base", description="Default collection name")
    chroma_distance_metric: str = Field(default="cosine", description="Distance metric")
    
    # Production Features
    chroma_production_mode: bool = Field(default=False, description="Enable production features")
    chroma_cluster_mode: str = Field(default="standalone", description="Cluster mode")
    chroma_backup_enabled: bool = Field(default=True, description="Enable backup system")
    chroma_backup_interval_hours: int = Field(default=6, ge=1, le=24, description="Backup interval")
    chroma_monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    
    # Performance
    chroma_max_memory_mb: int = Field(default=4096, ge=512, le=16384, description="Max memory in MB")
    chroma_connection_pool_size: int = Field(default=50, ge=10, le=100, description="Connection pool size")
    
    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Default embedding model")
    chunk_size: int = Field(default=1000, ge=100, le=4000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Chunk overlap size")


class SecurityConfig(BaseModel):
    """Security and authentication configuration."""
    
    # JWT
    secret_key: str = Field(default="your-secret-key-change-this", description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440, description="Access token expiration")
    
    # SSO (Optional)
    sso_enabled: bool = Field(default=False, description="Enable SSO authentication")
    keycloak_enabled: bool = Field(default=False, description="Enable Keycloak SSO")
    keycloak_server_url: Optional[str] = Field(default=None, description="Keycloak server URL")
    keycloak_realm: Optional[str] = Field(default=None, description="Keycloak realm")
    keycloak_client_id: Optional[str] = Field(default=None, description="Keycloak client ID")
    keycloak_client_secret: Optional[str] = Field(default=None, description="Keycloak client secret")
    
    @model_validator(mode='after')
    def validate_sso_config(self):
        """Validate SSO configuration if enabled."""
        if self.keycloak_enabled:
            required_fields = {
                "keycloak_server_url": self.keycloak_server_url,
                "keycloak_realm": self.keycloak_realm,
                "keycloak_client_id": self.keycloak_client_id,
                "keycloak_client_secret": self.keycloak_client_secret
            }
            for field_name, field_value in required_fields.items():
                if not field_value:
                    raise ValueError(f"{field_name} is required when Keycloak is enabled")
        return self


class PerformanceConfig(BaseModel):
    """Performance and optimization configuration."""
    
    # Async Processing (OPTIMIZED)
    async_worker_count: int = Field(default=16, ge=4, le=64, description="Async worker count")
    async_task_timeout: int = Field(default=300, ge=60, le=3600, description="Task timeout in seconds")
    enable_async_processing: bool = Field(default=True, description="Enable async processing")
    task_result_ttl: int = Field(default=3600, ge=300, le=86400, description="Task result TTL")
    
    # Concurrency (OPTIMIZED)
    max_concurrent_agents: int = Field(default=100, ge=10, le=1000, description="Max concurrent agents")
    agent_timeout_seconds: int = Field(default=300, ge=60, le=3600, description="Agent timeout")
    worker_connections: int = Field(default=2000, ge=100, le=10000, description="Worker connections")
    
    # Request Limits
    max_request_size: int = Field(default=16 * 1024 * 1024, ge=1024*1024, description="Max request size in bytes")
    request_timeout_seconds: int = Field(default=60, ge=10, le=300, description="Request timeout")
    
    # Document Processing (OPTIMIZED)
    max_file_size_mb: int = Field(default=100, ge=1, le=500, description="Max file size")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    document_processing_workers: int = Field(default=8, ge=2, le=32, description="Document workers")


class LoggingConfig(BaseModel):
    """Logging and monitoring configuration."""
    
    # Console Logging
    log_level: str = Field(default="INFO", description="Console log level")
    log_to_console: bool = Field(default=True, description="Enable console logging")
    log_console_format: str = Field(default="simple", description="Console format: simple, json, structured")
    
    # File Logging
    log_to_file: bool = Field(default=True, description="Enable file logging")
    log_file_level: str = Field(default="DEBUG", description="File log level")
    log_json_format: bool = Field(default=True, description="Use JSON format for files")
    log_retention_days: int = Field(default=30, ge=1, le=365, description="Log retention period")
    log_max_file_size_mb: int = Field(default=100, ge=10, le=1000, description="Max log file size")
    log_max_files: int = Field(default=10, ge=1, le=100, description="Max log files to keep")
    
    # Request/Response Logging
    log_include_request_body: bool = Field(default=False, description="Include request body in logs")
    log_include_response_body: bool = Field(default=False, description="Include response body in logs")
    log_exclude_paths: List[str] = Field(
        default=["/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico"],
        description="Paths to exclude from logging"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    metrics_port: int = Field(default=9090, ge=1024, le=65535, description="Metrics server port")

