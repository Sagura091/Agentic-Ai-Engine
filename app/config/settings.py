"""
Application settings and configuration management.

This module provides centralized configuration management using Pydantic settings
with support for environment variables and multiple configuration sources.
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables with the
    prefix 'AGENTIC_' (e.g., AGENTIC_DEBUG=true).
    """
    
    # Application settings
    APP_NAME: str = Field(default="Agentic AI Microservice", description="Application name")
    VERSION: str = Field(default="0.1.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment (development, staging, production)")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8888, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")

    # Agent settings
    MAX_AGENTS: int = Field(default=100, description="Maximum number of agents")
    DEFAULT_AGENT_MODEL: str = Field(default="llama3.1:8b", description="Default agent model")
    
    # Security settings
    SECRET_KEY: str = Field(default="your-secret-key-change-this", description="Secret key for JWT tokens")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration in minutes")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://localhost:5173", "http://localhost:8001", "*"],
        description="Allowed CORS origins"
    )

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_TO_FILE: bool = Field(default=True, description="Enable file logging")
    LOG_TO_CONSOLE: bool = Field(default=True, description="Enable console logging")
    LOG_JSON_FORMAT: bool = Field(default=True, description="Use JSON log format")
    LOG_RETENTION_DAYS: int = Field(default=30, description="Log retention period in days")
    LOG_MAX_FILE_SIZE_MB: int = Field(default=100, description="Maximum log file size in MB")
    LOG_MAX_FILES: int = Field(default=10, description="Maximum number of log files to keep")
    LOG_INCLUDE_REQUEST_BODY: bool = Field(default=False, description="Include request body in logs")
    LOG_INCLUDE_RESPONSE_BODY: bool = Field(default=False, description="Include response body in logs")
    LOG_EXCLUDE_PATHS: List[str] = Field(
        default=["/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico"],
        description="Paths to exclude from logging"
    )
    
    # Database settings
    DATABASE_URL: str = Field(
        default="postgresql://agentic_user:agentic_secure_password_2024@localhost:5432/agentic_ai",
        description="Database connection URL",
        env="AGENTIC_DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DATABASE_POOL_MAX_OVERFLOW: int = Field(default=5, description="Database pool max overflow")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, description="Database pool timeout")
    DATABASE_POOL_RECYCLE: int = Field(default=3600, description="Database pool recycle time")

    @property
    def database_url_async(self) -> str:
        """Get async database URL."""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return self.DATABASE_URL

    @property
    def database_url_sync(self) -> str:
        """Get sync database URL."""
        if self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        return self.DATABASE_URL
    
    # Redis settings
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_POOL_SIZE: int = Field(default=10, description="Redis connection pool size")

    # Distributed Architecture Settings
    ENABLE_DISTRIBUTED_MODE: bool = Field(default=False, description="Enable distributed agent registry")
    NODE_ID: Optional[str] = Field(default=None, description="Unique node identifier")
    CLUSTER_NAME: str = Field(default="agent_builder_cluster", description="Cluster name for node discovery")
    HEARTBEAT_INTERVAL: int = Field(default=30, description="Node heartbeat interval in seconds")

    # Async Processing Configuration
    ASYNC_WORKER_COUNT: int = Field(default=4, description="Number of async worker tasks")
    ASYNC_TASK_TIMEOUT: int = Field(default=300, description="Async task timeout in seconds")
    ENABLE_ASYNC_PROCESSING: bool = Field(default=True, description="Enable async task processing")
    TASK_RESULT_TTL: int = Field(default=3600, description="Task result cache TTL in seconds")

    # Enhanced Document Processing
    MAX_FILE_SIZE_MB: int = Field(default=100, description="Maximum file size for processing")
    SUPPORTED_FILE_EXTENSIONS: List[str] = Field(
        default=[
            ".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".json",
            ".xlsx", ".pptx", ".rtf", ".odt", ".epub", ".xml", ".yaml", ".yml",
            ".xls", ".ppt", ".zip", ".tar", ".gz"
        ],
        description="Supported file extensions for document processing"
    )
    ENABLE_PARALLEL_PROCESSING: bool = Field(default=True, description="Enable parallel document processing")
    DOCUMENT_PROCESSING_WORKERS: int = Field(default=3, description="Number of document processing workers")

    # Agent Builder UI Configuration
    ENABLE_VISUAL_BUILDER: bool = Field(default=True, description="Enable visual agent builder")
    COMPONENT_CACHE_TTL: int = Field(default=3600, description="Component cache TTL in seconds")
    MAX_CUSTOM_COMPONENTS: int = Field(default=100, description="Maximum custom components per user")
    ENABLE_TEMPLATE_SHARING: bool = Field(default=True, description="Enable template sharing between users")
    
    # LLM Integration settings - Multi-Provider Support
    # Ollama settings
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama base URL")
    OLLAMA_TIMEOUT: int = Field(default=120, description="Ollama request timeout in seconds")
    OLLAMA_RETRY_ATTEMPTS: int = Field(default=3, description="Number of retry attempts for Ollama")
    OLLAMA_KEEP_ALIVE: str = Field(default="30m", description="How long to keep models loaded in memory (e.g., '30m', '1h', '-1' for indefinite)")

    # OpenAI settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="OpenAI base URL")
    OPENAI_ORGANIZATION: str = Field(default="", description="OpenAI organization ID")
    OPENAI_PROJECT: str = Field(default="", description="OpenAI project ID")
    OPENAI_TIMEOUT: int = Field(default=60, description="OpenAI request timeout in seconds")

    # Anthropic settings
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
    ANTHROPIC_BASE_URL: str = Field(default="https://api.anthropic.com", description="Anthropic base URL")
    ANTHROPIC_TIMEOUT: int = Field(default=60, description="Anthropic request timeout in seconds")

    # Google settings
    GOOGLE_API_KEY: str = Field(default="", description="Google API key")
    GOOGLE_BASE_URL: str = Field(default="https://generativelanguage.googleapis.com/v1beta", description="Google base URL")
    GOOGLE_TIMEOUT: int = Field(default=60, description="Google request timeout in seconds")

    # Available Ollama Models for Agents (prioritized by tool calling support)
    AVAILABLE_OLLAMA_MODELS: List[str] = Field(
        default=[
            "llama3.1:8b",        # Primary: Excellent tool calling support
            "llama3.2:latest",    # Secondary: Good tool calling support
            "llama3.1:latest",    # Tertiary: Good tool calling support
            "qwen2.5:latest",     # Alternative: May support tools
            "mistral:latest",     # Alternative: May support tools
            "codellama:latest",   # Code-focused: Limited tool support
            "llama3.2:3b",        # Lightweight: Basic tool support
            "phi4:latest",        # Available but limited tool support in Ollama
            "phi3:latest"         # Legacy: Limited tool support
        ],
        description="Available Ollama models for agents (prioritized by tool calling capability)"
    )

    # OpenWebUI Integration settings (OPTIONAL - can be disabled)
    OPENWEBUI_ENABLED: bool = Field(default=False, description="Enable OpenWebUI integration")
    OPENWEBUI_BASE_URL: str = Field(default="http://open-webui:3000", description="OpenWebUI base URL")
    OPENWEBUI_INTEGRATION_MODE: str = Field(
        default="optional",
        description="Integration mode: 'optional', 'required', 'disabled'"
    )

    # Agent settings - Multi-Provider Support
    MAX_CONCURRENT_AGENTS: int = Field(default=10, description="Maximum concurrent agents")
    AGENT_TIMEOUT_SECONDS: int = Field(default=300, description="Agent execution timeout")
    DEFAULT_AGENT_MODEL: str = Field(default="llama3.1:8b", description="Default LLM model for agents")
    DEFAULT_AGENT_PROVIDER: str = Field(default="ollama", description="Default LLM provider for agents")
    BACKUP_AGENT_MODEL: str = Field(default="llama3.2:latest", description="Backup LLM model if default fails")
    BACKUP_AGENT_PROVIDER: str = Field(default="ollama", description="Backup LLM provider if default fails")

    # LLM Provider preferences
    ENABLE_OLLAMA: bool = Field(default=True, description="Enable Ollama provider")
    ENABLE_OPENAI: bool = Field(default=False, description="Enable OpenAI provider")
    ENABLE_ANTHROPIC: bool = Field(default=False, description="Enable Anthropic provider")
    ENABLE_GOOGLE: bool = Field(default=False, description="Enable Google provider")

    # Standalone Agent API Settings
    ENABLE_STANDALONE_API: bool = Field(default=True, description="Enable standalone agent API")
    STANDALONE_API_PORT: int = Field(default=8888, description="Standalone API port")
    ENABLE_AGENT_CHAT_API: bool = Field(default=True, description="Enable agent chat API")
    ENABLE_WORKFLOW_API: bool = Field(default=True, description="Enable workflow API")
    
    # LangGraph settings
    LANGGRAPH_CHECKPOINT_BACKEND: str = Field(default="redis", description="LangGraph checkpoint backend")
    LANGGRAPH_STATE_TTL_SECONDS: int = Field(default=3600, description="LangGraph state TTL")
    
    # Monitoring settings
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    ENABLE_TRACING: bool = Field(default=False, description="Enable OpenTelemetry tracing")

    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./data/chroma",
        description="Directory for ChromaDB persistence"
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="knowledge_base",
        description="Default ChromaDB collection name"
    )
    CHROMA_DISTANCE_METRIC: str = Field(
        default="cosine",
        description="Distance metric for ChromaDB"
    )

    # Production ChromaDB Configuration
    CHROMA_PRODUCTION_MODE: bool = Field(
        default=False,
        description="Enable production ChromaDB features"
    )
    CHROMA_CLUSTER_MODE: str = Field(
        default="standalone",
        description="ChromaDB cluster mode: standalone, cluster, distributed"
    )
    CHROMA_BACKUP_ENABLED: bool = Field(
        default=True,
        description="Enable ChromaDB backup system"
    )
    CHROMA_BACKUP_INTERVAL_HOURS: int = Field(
        default=6,
        description="Backup interval in hours"
    )
    CHROMA_MONITORING_ENABLED: bool = Field(
        default=True,
        description="Enable ChromaDB monitoring"
    )
    CHROMA_MAX_MEMORY_MB: int = Field(
        default=4096,
        description="Maximum memory for ChromaDB in MB"
    )
    CHROMA_CONNECTION_POOL_SIZE: int = Field(
        default=20,
        description="ChromaDB connection pool size"
    )
    METRICS_PORT: int = Field(default=9090, description="Metrics server port")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json, text)")
    
    # File storage settings
    DATA_DIR: str = Field(default="./data", description="Data directory path")
    AGENTS_DIR: str = Field(default="./data/agents", description="Agents storage directory")
    WORKFLOWS_DIR: str = Field(default="./data/workflows", description="Workflows storage directory")
    CHECKPOINTS_DIR: str = Field(default="./data/checkpoints", description="Checkpoints storage directory")
    LOGS_DIR: str = Field(default="./data/logs", description="Logs storage directory")
    
    # Performance settings
    MAX_REQUEST_SIZE: int = Field(default=16 * 1024 * 1024, description="Maximum request size in bytes")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=60, description="Request timeout in seconds")
    WORKER_CONNECTIONS: int = Field(default=1000, description="Worker connections")
    
    # GPU settings
    ENABLE_GPU: bool = Field(default=False, description="Enable GPU acceleration")
    GPU_DEVICE_IDS: List[int] = Field(default=[0], description="GPU device IDs to use")
    GPU_MEMORY_FRACTION: float = Field(default=0.8, description="GPU memory fraction to use")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("GPU_DEVICE_IDS", pre=True)
    def parse_gpu_device_ids(cls, v):
        """Parse GPU device IDs from string or list."""
        if isinstance(v, str):
            return [int(device_id.strip()) for device_id in v.split(",")]
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "sqlite:///")):
            raise ValueError("DATABASE_URL must start with postgresql:// or sqlite:///")
        return v
    
    @validator("REDIS_URL")
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must start with redis://")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
    
    @property
    def database_url_async(self) -> str:
        """Get asynchronous database URL."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.AGENTS_DIR,
            self.WORKFLOWS_DIR,
            self.CHECKPOINTS_DIR,
            self.LOGS_DIR,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_provider_credentials(self) -> Dict[str, Any]:
        """Get LLM provider credentials configuration."""
        from app.llm.models import ProviderType, ProviderCredentials

        credentials = {}

        # Ollama credentials (no API key needed)
        if self.ENABLE_OLLAMA:
            credentials[ProviderType.OLLAMA] = ProviderCredentials(
                provider=ProviderType.OLLAMA,
                base_url=self.OLLAMA_BASE_URL
            )

        # OpenAI credentials
        if self.ENABLE_OPENAI and self.OPENAI_API_KEY:
            credentials[ProviderType.OPENAI] = ProviderCredentials(
                provider=ProviderType.OPENAI,
                api_key=self.OPENAI_API_KEY,
                base_url=self.OPENAI_BASE_URL,
                organization=self.OPENAI_ORGANIZATION or None,
                project=self.OPENAI_PROJECT or None
            )

        # Anthropic credentials
        if self.ENABLE_ANTHROPIC and self.ANTHROPIC_API_KEY:
            credentials[ProviderType.ANTHROPIC] = ProviderCredentials(
                provider=ProviderType.ANTHROPIC,
                api_key=self.ANTHROPIC_API_KEY,
                base_url=self.ANTHROPIC_BASE_URL
            )

        # Google credentials
        if self.ENABLE_GOOGLE and self.GOOGLE_API_KEY:
            credentials[ProviderType.GOOGLE] = ProviderCredentials(
                provider=ProviderType.GOOGLE,
                api_key=self.GOOGLE_API_KEY,
                base_url=self.GOOGLE_BASE_URL
            )

        return credentials

    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers."""
        providers = []
        if self.ENABLE_OLLAMA:
            providers.append("ollama")
        if self.ENABLE_OPENAI and self.OPENAI_API_KEY:
            providers.append("openai")
        if self.ENABLE_ANTHROPIC and self.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if self.ENABLE_GOOGLE and self.GOOGLE_API_KEY:
            providers.append("google")
        return providers
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "AGENTIC_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Application settings instance
    """
    settings = Settings()
    settings.create_directories()
    return settings
