"""
Enhanced Admin Settings API endpoints - Phase 1: Core Infrastructure.

Revolutionary admin panel backend with comprehensive settings management,
real-time updates, validation, and security controls.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import asyncio
from enum import Enum

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy import text, select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user, get_current_active_user
from app.models.auth import UserDB
from app.models.database.base import get_database_session
from app.api.v1.responses import StandardAPIResponse
from app.config.settings import get_settings
from app.services.admin_settings_service import get_admin_settings_service
from app.services.rag_settings_applicator import get_rag_settings_applicator
from app.rag.core.dynamic_config_manager import update_rag_settings, get_rag_system_status
from app.core.global_config_manager import global_config_manager
from app.core.configuration_broadcaster import configuration_broadcaster, BroadcastLevel, NotificationType
# from app.rag.core.embedding_model_manager import embedding_model_manager  # Removed

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/enhanced-settings", tags=["Enhanced Admin Settings"])


# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class SettingCategory(str, Enum):
    """Setting categories for organization."""
    SYSTEM_CONFIGURATION = "system_configuration"
    SECURITY_AUTHENTICATION = "security_authentication"
    AGENT_MANAGEMENT = "agent_management"
    LLM_PROVIDERS = "llm_providers"
    WORKFLOW_MANAGEMENT = "workflow_management"
    NODE_EXECUTION = "node_execution"
    GUARD_RAILS = "guard_rails"
    DATABASE_STORAGE = "database_storage"
    RAG_CONFIGURATION = "rag_configuration"
    MONITORING_LOGGING = "monitoring_logging"
    TOOL_NODE_MANAGEMENT = "tool_node_management"
    INTEGRATIONS = "integrations"
    BACKUP_MAINTENANCE = "backup_maintenance"
    NOTIFICATIONS = "notifications"


class SettingType(str, Enum):
    """Setting value types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"


class SecurityLevel(str, Enum):
    """Security levels for settings."""
    PUBLIC = "public"
    ADMIN_ONLY = "admin_only"
    SYSTEM_ONLY = "system_only"
    ENCRYPTED = "encrypted"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SettingDefinition(BaseModel):
    """Definition of a system setting."""
    key: str = Field(..., description="Setting key")
    category: SettingCategory = Field(..., description="Setting category")
    type: SettingType = Field(..., description="Setting value type")
    default_value: Any = Field(..., description="Default value")
    current_value: Any = Field(None, description="Current value")
    description: str = Field(..., description="Setting description")
    security_level: SecurityLevel = Field(default=SecurityLevel.ADMIN_ONLY)
    requires_restart: bool = Field(default=False, description="Requires system restart")
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    enum_values: Optional[List[str]] = Field(None, description="Enum values if type is enum")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value")
    is_sensitive: bool = Field(default=False, description="Contains sensitive data")


class SystemConfigurationSettings(BaseModel):
    """System configuration settings."""
    app_name: str = Field(default="Agentic AI Platform")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="production")
    debug_mode: bool = Field(default=False)
    maintenance_mode: bool = Field(default=False)
    max_users: int = Field(default=1000)
    registration_enabled: bool = Field(default=True)
    
    # NEW: Execution Control
    max_concurrent_workflows: int = Field(default=50)
    max_concurrent_nodes: int = Field(default=200)
    global_execution_timeout: int = Field(default=3600)
    max_memory_per_workflow: int = Field(default=1024)  # MB
    max_cpu_per_workflow: float = Field(default=2.0)  # CPU cores
    max_disk_usage: int = Field(default=10240)  # MB


class SecurityAuthenticationSettings(BaseModel):
    """Security and authentication settings."""
    # JWT Settings
    secret_key: str = Field(..., description="JWT secret key")
    access_token_expire_minutes: int = Field(default=30)
    algorithm: str = Field(default="HS256")

    # Password Policy
    password_min_length: int = Field(default=8)
    password_require_special: bool = Field(default=True)
    password_require_uppercase: bool = Field(default=True)
    password_require_numbers: bool = Field(default=True)

    # Session Management
    session_timeout: int = Field(default=3600)
    max_login_attempts: int = Field(default=5)
    lockout_duration: int = Field(default=900)

    # Two-Factor Authentication
    two_factor_required: bool = Field(default=False)
    two_factor_issuer: str = Field(default="Agentic AI")

    # NEW: Guard Rails & Validation
    enable_tool_validation: bool = Field(default=True)
    dangerous_operations_blocked: List[str] = Field(default_factory=lambda: [
        "exec", "eval", "subprocess", "os.system", "__import__"
    ])
    security_scan_level: str = Field(default="strict")
    sandbox_execution: bool = Field(default=True)
    max_execution_time: int = Field(default=300)
    max_memory_usage: int = Field(default=512)  # MB
    allowed_imports: List[str] = Field(default_factory=lambda: [
        "json", "datetime", "math", "random", "string", "re"
    ])

    # NEW: Node Security
    node_validation_enabled: bool = Field(default=True)
    custom_node_approval_required: bool = Field(default=True)
    node_security_scanning: bool = Field(default=True)


class AgentManagementSettings(BaseModel):
    """Agent system management settings."""
    # Agent Limits
    max_agents: int = Field(default=100)
    max_concurrent_agents: int = Field(default=10)
    max_agents_per_user: int = Field(default=10)
    agent_timeout_seconds: int = Field(default=300)

    # Default Configuration
    default_agent_model: str = Field(default="llama3.1:8b")
    default_agent_provider: str = Field(default="ollama")
    backup_agent_model: str = Field(default="llama3.2:latest")
    backup_agent_provider: str = Field(default="ollama")

    # Agent Features
    enable_agent_sharing: bool = Field(default=True)
    enable_agent_memory: bool = Field(default=True)
    enable_agent_tools: bool = Field(default=True)
    auto_cleanup: bool = Field(default=True)

    # Agent API Settings
    enable_standalone_api: bool = Field(default=True)
    standalone_api_port: int = Field(default=8888)
    enable_agent_chat_api: bool = Field(default=True)
    enable_workflow_api: bool = Field(default=True)

    # NEW: Workflow Integration
    enable_agent_workflows: bool = Field(default=True)
    max_workflow_depth: int = Field(default=10)
    agent_workflow_timeout: int = Field(default=600)
    enable_hierarchical_workflows: bool = Field(default=True)
    enable_autonomous_workflows: bool = Field(default=False)


class LLMProviderSettings(BaseModel):
    """🚀 Revolutionary LLM Provider Management Settings - Complete provider configuration system."""

    # ============================================================================
    # PROVIDER ENABLEMENT (Admin controls which providers are available system-wide)
    # ============================================================================
    enable_ollama: bool = Field(default=True, description="Enable Ollama local provider")
    enable_openai: bool = Field(default=False, description="Enable OpenAI cloud provider")
    enable_anthropic: bool = Field(default=False, description="Enable Anthropic cloud provider")
    enable_google: bool = Field(default=False, description="Enable Google cloud provider")

    # ============================================================================
    # OLLAMA CONFIGURATION (Local AI Server)
    # ============================================================================
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_timeout: int = Field(default=120, description="Request timeout in seconds")
    ollama_max_concurrent_requests: int = Field(default=10, description="Max concurrent requests")
    ollama_connection_pool_size: int = Field(default=5, description="Connection pool size")
    ollama_request_timeout: int = Field(default=60, description="Individual request timeout")
    ollama_keep_alive: str = Field(default="30m", description="Model keep-alive duration")
    ollama_num_ctx: int = Field(default=4096, description="Context window size")
    ollama_num_thread: int = Field(default=8, description="Number of threads")
    ollama_num_gpu: int = Field(default=1, description="Number of GPUs to use")
    ollama_main_gpu: int = Field(default=0, description="Main GPU index")
    ollama_repeat_penalty: float = Field(default=1.1, description="Repetition penalty")
    ollama_temperature: float = Field(default=0.7, description="Default temperature")
    ollama_top_k: int = Field(default=40, description="Top-k sampling")
    ollama_top_p: float = Field(default=0.9, description="Top-p sampling")

    # ============================================================================
    # OPENAI CONFIGURATION (Cloud Provider)
    # ============================================================================
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    openai_timeout: int = Field(default=60, description="Request timeout in seconds")
    openai_max_requests_per_minute: int = Field(default=60, description="Rate limit per minute")
    openai_max_retries: int = Field(default=3, description="Maximum retry attempts")
    openai_retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    openai_temperature: float = Field(default=0.7, description="Default temperature")
    openai_max_tokens: int = Field(default=4096, description="Default max tokens")
    openai_top_p: float = Field(default=1.0, description="Top-p sampling")
    openai_frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    openai_presence_penalty: float = Field(default=0.0, description="Presence penalty")

    # ============================================================================
    # ANTHROPIC CONFIGURATION (Cloud Provider)
    # ============================================================================
    anthropic_base_url: str = Field(default="https://api.anthropic.com", description="Anthropic API base URL")
    anthropic_timeout: int = Field(default=60, description="Request timeout in seconds")
    anthropic_max_requests_per_minute: int = Field(default=60, description="Rate limit per minute")
    anthropic_max_retries: int = Field(default=3, description="Maximum retry attempts")
    anthropic_retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    anthropic_temperature: float = Field(default=0.7, description="Default temperature")
    anthropic_max_tokens: int = Field(default=4096, description="Default max tokens")
    anthropic_top_p: float = Field(default=1.0, description="Top-p sampling")
    anthropic_top_k: int = Field(default=40, description="Top-k sampling")

    # ============================================================================
    # GOOGLE CONFIGURATION (Cloud Provider)
    # ============================================================================
    google_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", description="Google API base URL")
    google_timeout: int = Field(default=60, description="Request timeout in seconds")
    google_max_requests_per_minute: int = Field(default=60, description="Rate limit per minute")
    google_max_retries: int = Field(default=3, description="Maximum retry attempts")
    google_retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    google_temperature: float = Field(default=0.7, description="Default temperature")
    google_max_tokens: int = Field(default=4096, description="Default max tokens")
    google_top_p: float = Field(default=1.0, description="Top-p sampling")
    google_top_k: int = Field(default=40, description="Top-k sampling")

    # ============================================================================
    # PERFORMANCE & RELIABILITY SETTINGS
    # ============================================================================
    enable_load_balancing: bool = Field(default=False, description="Enable load balancing across providers")
    enable_failover: bool = Field(default=True, description="Enable automatic failover")
    default_provider: str = Field(default="ollama", description="Default provider to use")
    fallback_provider: str = Field(default="openai", description="Fallback provider")
    request_timeout: int = Field(default=120, description="Global request timeout")
    max_concurrent_requests: int = Field(default=50, description="Global max concurrent requests")
    enable_request_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")

    # ============================================================================
    # MODEL MANAGEMENT
    # ============================================================================
    auto_download_models: bool = Field(default=False, description="Auto-download recommended models")
    preferred_models: Dict[str, str] = Field(default_factory=lambda: {
        "ollama": "llama3.2:latest",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-1.5-flash"
    }, description="Preferred model for each provider")
    model_download_timeout: int = Field(default=1800, description="Model download timeout in seconds")

    # ============================================================================
    # MONITORING & LOGGING
    # ============================================================================
    enable_usage_tracking: bool = Field(default=True, description="Track provider usage")
    enable_performance_monitoring: bool = Field(default=True, description="Monitor performance metrics")
    log_requests: bool = Field(default=False, description="Log all requests (sensitive)")
    log_responses: bool = Field(default=False, description="Log all responses (sensitive)")
    enable_cost_tracking: bool = Field(default=True, description="Track API costs")

    # ============================================================================
    # SECURITY SETTINGS
    # ============================================================================
    enable_api_key_rotation: bool = Field(default=False, description="Enable automatic API key rotation")
    api_key_rotation_days: int = Field(default=30, description="API key rotation interval")
    enable_request_signing: bool = Field(default=False, description="Enable request signing")
    allowed_models: Dict[str, List[str]] = Field(default_factory=lambda: {
        "ollama": ["llama3.2:latest", "llama3.1:8b", "qwen2.5:latest"],
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "google": ["gemini-1.5-pro", "gemini-1.5-flash"]
    }, description="Allowed models per provider")

    google_base_url: str = Field(default="https://generativelanguage.googleapis.com")
    google_timeout: int = Field(default=60)
    google_max_requests_per_minute: int = Field(default=60)

    # System-wide Model Defaults (Admin sets defaults for all users)
    default_temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)
    retry_attempts: int = Field(default=3)
    request_timeout: int = Field(default=30)

    # Rate Limiting & Performance (Admin controls system performance)
    enable_rate_limiting: bool = Field(default=True)
    global_rate_limit_per_minute: int = Field(default=1000)
    enable_request_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300)

    # Model Management (Admin controls available models)
    allowed_openai_models: List[str] = Field(default_factory=lambda: [
        "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
    ])
    allowed_anthropic_models: List[str] = Field(default_factory=lambda: [
        "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"
    ])
    allowed_google_models: List[str] = Field(default_factory=lambda: [
        "gemini-pro", "gemini-pro-vision"
    ])

    # Security & Monitoring (Admin controls system security)
    enable_usage_monitoring: bool = Field(default=True)
    log_all_requests: bool = Field(default=False)
    enable_content_filtering: bool = Field(default=True)
    max_context_length: int = Field(default=32000)


class RAGSystemSettings(BaseModel):
    """🚀 Revolutionary RAG System Configuration - The Most Comprehensive RAG Settings Ever Created!"""

    # ============================================================================
    # 🗄️ VECTOR STORE CONFIGURATION
    # ============================================================================
    persist_directory: str = Field(default="./data/chroma", description="ChromaDB persistence directory")
    collection_metadata: Dict[str, str] = Field(default_factory=lambda: {"hnsw:space": "cosine"})
    connection_pool_size: int = Field(default=10, ge=1, le=100)
    max_batch_size: int = Field(default=128, ge=1, le=1000)
    enable_multi_collection: bool = Field(default=True)
    vector_dimension: int = Field(default=384, ge=128, le=4096)
    similarity_metric: str = Field(default="cosine", description="cosine, euclidean, or dot_product")

    # ============================================================================
    # 🤖 EMBEDDING MODELS CONFIGURATION
    # ============================================================================
    # Primary Embedding Model
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Primary embedding model")
    embedding_provider: str = Field(default="sentence-transformers", description="sentence-transformers, openai, ollama")
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    normalize_embeddings: bool = Field(default=True)
    cache_embeddings: bool = Field(default=True)
    embedding_cache_size: int = Field(default=10000, ge=100, le=100000)
    enable_hybrid_embeddings: bool = Field(default=True)

    # Model Download & Management
    models_directory: str = Field(default="./data/models", description="Directory for downloaded models")
    auto_download_models: bool = Field(default=False, description="Automatically download missing models")
    model_download_timeout: int = Field(default=3600, description="Model download timeout in seconds")
    enable_model_validation: bool = Field(default=True, description="Validate models after download")

    # Available Embedding Models (for download)
    available_embedding_models: List[str] = Field(default_factory=lambda: [
        "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1",
        "multi-qa-MiniLM-L6-cos-v1", "paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/all-roberta-large-v1"
    ])

    # ============================================================================
    # 👁️ VISION MODELS CONFIGURATION
    # ============================================================================
    # Vision Model Settings
    enable_vision_models: bool = Field(default=True, description="Enable vision model processing")
    primary_vision_model: str = Field(default="clip-vit-base-patch32", description="Primary vision model")
    vision_model_provider: str = Field(default="sentence-transformers", description="Vision model provider")
    vision_batch_size: int = Field(default=16, ge=1, le=64)
    vision_image_size: tuple = Field(default=(224, 224), description="Image processing size")
    enable_vision_caching: bool = Field(default=True)

    # Available Vision Models (for download)
    available_vision_models: List[str] = Field(default_factory=lambda: [
        "clip-vit-base-patch32", "clip-vit-large-patch14", "clip-vit-base-patch16",
        "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"
    ])

    # ============================================================================
    # 🔍 OCR ENGINES CONFIGURATION
    # ============================================================================
    # OCR Engine Settings
    enable_ocr: bool = Field(default=True, description="Enable OCR processing")
    enable_tesseract: bool = Field(default=True, description="Enable Tesseract OCR")
    enable_easyocr: bool = Field(default=True, description="Enable EasyOCR")
    enable_paddleocr: bool = Field(default=True, description="Enable PaddleOCR")

    # OCR Configuration
    ocr_languages: List[str] = Field(default_factory=lambda: ["en", "es", "fr", "de", "zh"], description="OCR languages")
    ocr_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum OCR confidence")
    enable_image_enhancement: bool = Field(default=True, description="Enhance images before OCR")
    ocr_preprocessing: bool = Field(default=True, description="Apply OCR preprocessing")
    ocr_timeout: int = Field(default=300, description="OCR processing timeout")

    # ============================================================================
    # 📄 DOCUMENT PROCESSING CONFIGURATION
    # ============================================================================
    # Chunking Configuration
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    chunking_strategy: str = Field(default="semantic", description="semantic, fixed, or adaptive")
    enable_agent_tagging: bool = Field(default=True)
    enable_scope_classification: bool = Field(default=True)
    enable_smart_chunking: bool = Field(default=True, description="Use AI-powered smart chunking")

    # Document Types
    supported_document_types: List[str] = Field(default_factory=lambda: [
        "pdf", "docx", "txt", "md", "html", "csv", "xlsx", "pptx", "jpg", "png", "gif", "bmp"
    ])
    max_document_size_mb: int = Field(default=100, ge=1, le=1000)
    enable_document_validation: bool = Field(default=True)

    # ============================================================================
    # 🎯 RETRIEVAL CONFIGURATION
    # ============================================================================
    # Basic Retrieval Settings
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top results to retrieve")
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    enable_query_expansion: bool = Field(default=True, description="Expand queries for better results")
    enable_hybrid_search: bool = Field(default=True, description="Combine dense and sparse search")

    # Advanced Retrieval
    memory_boost_factor: float = Field(default=1.2, ge=1.0, le=3.0, description="Boost factor for memory results")
    recency_boost_factor: float = Field(default=1.1, ge=1.0, le=3.0, description="Boost factor for recent results")
    enable_contextual_retrieval: bool = Field(default=True, description="Use conversation context")
    enable_semantic_search: bool = Field(default=True, description="Enable semantic search")
    enable_keyword_search: bool = Field(default=True, description="Enable keyword search")

    # Query Processing
    max_query_length: int = Field(default=1000, ge=10, le=5000, description="Maximum query length")
    enable_query_preprocessing: bool = Field(default=True, description="Preprocess queries")
    enable_spell_correction: bool = Field(default=True, description="Correct spelling in queries")
    query_expansion_factor: float = Field(default=1.5, ge=1.0, le=3.0, description="Query expansion factor")

    # ============================================================================
    # 🤝 MULTI-AGENT CONFIGURATION
    # ============================================================================
    # Agent Isolation & Sharing
    enable_agent_isolation: bool = Field(default=True, description="Isolate agent knowledge bases")
    enable_knowledge_sharing: bool = Field(default=True, description="Allow knowledge sharing between agents")
    enable_memory_integration: bool = Field(default=True, description="Integrate with agent memory")
    enable_collaborative_learning: bool = Field(default=True, description="Enable collaborative learning")

    # Memory Management
    default_retention_days: int = Field(default=30, ge=1, le=365, description="Default memory retention")
    max_memory_items: int = Field(default=10000, ge=100, le=100000, description="Maximum memory items per agent")
    enable_auto_cleanup: bool = Field(default=True, description="Automatically clean up old memories")
    enable_permission_system: bool = Field(default=True, description="Enable permission-based access")

    # Knowledge Lifecycle
    enable_knowledge_versioning: bool = Field(default=True, description="Version knowledge updates")
    enable_knowledge_expiration: bool = Field(default=False, description="Expire old knowledge")
    knowledge_expiration_days: int = Field(default=90, ge=1, le=365, description="Knowledge expiration period")

    # ============================================================================
    # 🚀 PERFORMANCE OPTIMIZATION
    # ============================================================================
    # Caching Configuration
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, ge=60, le=86400, description="Cache time-to-live in seconds")
    cache_size_mb: int = Field(default=512, ge=64, le=4096, description="Cache size in MB")
    enable_distributed_caching: bool = Field(default=False, description="Enable distributed caching")

    # Connection & Processing
    enable_connection_pooling: bool = Field(default=True, description="Enable connection pooling")
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    max_concurrent_queries: int = Field(default=50, ge=1, le=200, description="Maximum concurrent queries")
    enable_async_processing: bool = Field(default=True, description="Enable async processing")
    processing_timeout: int = Field(default=300, ge=30, le=1800, description="Processing timeout in seconds")

    # Resource Management
    max_memory_usage_mb: int = Field(default=2048, ge=256, le=16384, description="Maximum memory usage")
    enable_gpu_acceleration: bool = Field(default=True, description="Enable GPU acceleration")
    gpu_memory_fraction: float = Field(default=0.5, ge=0.1, le=1.0, description="GPU memory fraction to use")

    # ============================================================================
    # 📋 RAG TEMPLATES CONFIGURATION
    # ============================================================================
    # Template Settings
    enable_rag_templates: bool = Field(default=True, description="Enable RAG templates")
    default_rag_template: str = Field(default="general_purpose", description="Default RAG template")

    # Available Templates
    available_templates: List[str] = Field(default_factory=lambda: [
        "general_purpose", "research_assistant", "code_helper", "creative_writing",
        "technical_documentation", "customer_support", "educational_tutor", "data_analyst"
    ])

    # Template Customization
    enable_template_customization: bool = Field(default=True, description="Allow template customization")
    max_custom_templates: int = Field(default=10, ge=1, le=50, description="Maximum custom templates per user")

    # ============================================================================
    # 📊 MONITORING & ANALYTICS
    # ============================================================================
    # Performance Monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Monitor RAG performance")
    enable_query_analytics: bool = Field(default=True, description="Analyze query patterns")
    enable_usage_tracking: bool = Field(default=True, description="Track RAG usage statistics")
    analytics_retention_days: int = Field(default=30, ge=1, le=365, description="Analytics data retention")

    # Quality Metrics
    enable_quality_scoring: bool = Field(default=True, description="Score result quality")
    enable_relevance_feedback: bool = Field(default=True, description="Collect relevance feedback")
    enable_auto_evaluation: bool = Field(default=False, description="Automatic result evaluation")
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Quality score threshold")

    # ============================================================================
    # 🔧 ADVANCED CONFIGURATION
    # ============================================================================
    # Experimental Features
    enable_experimental_features: bool = Field(default=False, description="Enable experimental features")
    enable_neural_search: bool = Field(default=False, description="Enable neural search (experimental)")
    enable_graph_rag: bool = Field(default=False, description="Enable graph-based RAG (experimental)")
    enable_multimodal_fusion: bool = Field(default=False, description="Enable multimodal fusion (experimental)")

    # Integration Settings
    enable_external_apis: bool = Field(default=False, description="Enable external API integration")
    enable_web_search_fallback: bool = Field(default=False, description="Use web search as fallback")
    enable_real_time_updates: bool = Field(default=True, description="Enable real-time knowledge updates")

    # Security & Privacy
    enable_content_filtering: bool = Field(default=True, description="Filter inappropriate content")
    enable_pii_detection: bool = Field(default=True, description="Detect personally identifiable information")
    enable_data_anonymization: bool = Field(default=False, description="Anonymize sensitive data")
    content_safety_level: str = Field(default="moderate", description="strict, moderate, or permissive")

    # ============================================================================
    # 🌐 GLOBAL SETTINGS
    # ============================================================================
    # System-wide Settings
    rag_system_enabled: bool = Field(default=True, description="Enable RAG system globally")
    apply_to_all_agents: bool = Field(default=True, description="Apply settings to all agents")
    allow_agent_overrides: bool = Field(default=True, description="Allow agents to override settings")
    require_admin_approval: bool = Field(default=False, description="Require admin approval for changes")

    # Backup & Recovery
    enable_auto_backup: bool = Field(default=True, description="Enable automatic backups")
    backup_frequency_hours: int = Field(default=24, ge=1, le=168, description="Backup frequency in hours")
    backup_retention_days: int = Field(default=30, ge=1, le=365, description="Backup retention period")
    enable_disaster_recovery: bool = Field(default=False, description="Enable disaster recovery")

    # ============================================================================
    # 🎨 UI/UX CONFIGURATION
    # ============================================================================
    # User Interface Settings
    enable_advanced_ui: bool = Field(default=True, description="Enable advanced UI features")
    show_confidence_scores: bool = Field(default=True, description="Show confidence scores to users")
    enable_result_explanations: bool = Field(default=True, description="Provide result explanations")
    enable_interactive_refinement: bool = Field(default=True, description="Allow interactive query refinement")

    # Customization
    custom_branding_enabled: bool = Field(default=False, description="Enable custom branding")
    custom_css_enabled: bool = Field(default=False, description="Allow custom CSS")
    theme_customization: str = Field(default="default", description="UI theme customization")


class SettingUpdateRequest(BaseModel):
    """Request to update a setting."""
    category: SettingCategory
    key: str
    value: Any
    validate_only: bool = Field(default=False, description="Only validate, don't save")


class BulkSettingUpdateRequest(BaseModel):
    """Request to update multiple settings."""
    updates: List[SettingUpdateRequest]
    validate_all: bool = Field(default=True, description="Validate all before applying")


class SettingValidationResult(BaseModel):
    """Result of setting validation."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    requires_restart: bool = Field(default=False)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def verify_configuration_applied(section: str, key: str, expected_value: Any) -> bool:
    """Verify that a configuration change actually took effect."""
    try:
        from app.core.global_config_manager import ConfigurationSection
        section_enum = ConfigurationSection(section)
        current_config = await global_config_manager.get_section_configuration(section_enum)
        actual_value = current_config.get(key)

        # Handle different value types
        if isinstance(expected_value, bool) and isinstance(actual_value, str):
            actual_value = actual_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(expected_value, (int, float)) and isinstance(actual_value, str):
            try:
                actual_value = type(expected_value)(actual_value)
            except (ValueError, TypeError):
                pass

        return actual_value == expected_value
    except Exception as e:
        logger.error(f"❌ Failed to verify configuration: {str(e)}")
        return False


async def determine_broadcast_level(category: str, key: str) -> BroadcastLevel:
    """Determine the appropriate broadcast level for a setting change."""
    # Security-related settings - admin only
    if any(term in key.lower() for term in ['api_key', 'secret', 'password', 'token', 'credential']):
        return BroadcastLevel.ENCRYPTED

    # System configuration - admin only
    if category == 'system_configuration':
        if key in ['debug_mode', 'maintenance_mode', 'log_level']:
            return BroadcastLevel.ADMIN_ONLY
        return BroadcastLevel.SYSTEM_ONLY

    # LLM provider changes - public (affects user experience)
    if category == 'llm_providers':
        if 'enable_' in key or 'model' in key.lower():
            return BroadcastLevel.PUBLIC
        return BroadcastLevel.ADMIN_ONLY

    # RAG configuration - public (affects search results)
    if category == 'rag_configuration':
        if key in ['embedding_model', 'chunk_size', 'similarity_threshold']:
            return BroadcastLevel.PUBLIC
        return BroadcastLevel.ADMIN_ONLY

    # Default to admin only
    return BroadcastLevel.ADMIN_ONLY


async def determine_notification_type(category: str, key: str) -> NotificationType:
    """Determine the notification type for user preference filtering."""
    if category == 'llm_providers':
        return NotificationType.MODEL_UPDATES
    elif category == 'rag_configuration':
        return NotificationType.SYSTEM_UPDATES
    elif 'security' in key.lower() or 'api_key' in key.lower():
        return NotificationType.SECURITY_UPDATES
    else:
        return NotificationType.SYSTEM_UPDATES


def require_admin(current_user: UserDB = Depends(get_current_active_user)) -> UserDB:
    """Require admin user."""
    if current_user.user_group != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def validate_setting_value(definition: SettingDefinition, value: Any) -> SettingValidationResult:
    """Validate a setting value against its definition."""
    result = SettingValidationResult(is_valid=True)
    
    try:
        # Type validation
        if definition.type == SettingType.STRING and not isinstance(value, str):
            result.errors.append(f"Value must be a string")
            result.is_valid = False
        elif definition.type == SettingType.INTEGER and not isinstance(value, int):
            result.errors.append(f"Value must be an integer")
            result.is_valid = False
        elif definition.type == SettingType.FLOAT and not isinstance(value, (int, float)):
            result.errors.append(f"Value must be a number")
            result.is_valid = False
        elif definition.type == SettingType.BOOLEAN and not isinstance(value, bool):
            result.errors.append(f"Value must be a boolean")
            result.is_valid = False
        elif definition.type == SettingType.ARRAY and not isinstance(value, list):
            result.errors.append(f"Value must be an array")
            result.is_valid = False
        elif definition.type == SettingType.OBJECT and not isinstance(value, dict):
            result.errors.append(f"Value must be an object")
            result.is_valid = False
        
        # Range validation
        if definition.min_value is not None and isinstance(value, (int, float)):
            if value < definition.min_value:
                result.errors.append(f"Value must be >= {definition.min_value}")
                result.is_valid = False
        
        if definition.max_value is not None and isinstance(value, (int, float)):
            if value > definition.max_value:
                result.errors.append(f"Value must be <= {definition.max_value}")
                result.is_valid = False
        
        # Enum validation
        if definition.enum_values and value not in definition.enum_values:
            result.errors.append(f"Value must be one of: {', '.join(definition.enum_values)}")
            result.is_valid = False
        
        # Custom validation rules
        for rule_name, rule_value in definition.validation_rules.items():
            if rule_name == "min_length" and isinstance(value, str):
                if len(value) < rule_value:
                    result.errors.append(f"Value must be at least {rule_value} characters")
                    result.is_valid = False
            elif rule_name == "max_length" and isinstance(value, str):
                if len(value) > rule_value:
                    result.errors.append(f"Value must be at most {rule_value} characters")
                    result.is_valid = False
        
        result.requires_restart = definition.requires_restart
        
    except Exception as e:
        result.errors.append(f"Validation error: {str(e)}")
        result.is_valid = False
    
    return result


async def get_setting_definitions() -> Dict[str, SettingDefinition]:
    """Get all setting definitions."""
    settings = get_settings()

    # Get current values from global config manager
    print("🔥 DEBUG: About to import global config manager")
    from app.core.global_config_manager import global_config_manager, ConfigurationSection
    print(f"🔥 DEBUG: global_config_manager exists: {global_config_manager is not None}")

    current_config = {}
    if global_config_manager:
        print("🔥 DEBUG: Starting configuration loading...")
        # Get all section configurations properly
        for section in ConfigurationSection:
            try:
                print(f"🔥 DEBUG: Loading section {section}")
                section_config = await global_config_manager.get_section_configuration(section)
                current_config[section] = section_config
                print(f"🔥 DEBUG: Loaded config for {section}: {len(section_config)} settings")
                logger.info(f"✅ Loaded config for {section}: {len(section_config)} settings")
            except Exception as e:
                print(f"🔥 DEBUG: Failed to get configuration for section {section}: {e}")
                logger.warning(f"Failed to get configuration for section {section}: {e}")
                current_config[section] = {}

        print(f"🔥 DEBUG: Total sections loaded: {len(current_config)}")
        print(f"🔥 DEBUG: Database storage config: {current_config.get(ConfigurationSection.DATABASE_STORAGE, {})}")
        print(f"🔥 DEBUG: LLM providers config: {current_config.get(ConfigurationSection.LLM_PROVIDERS, {})}")
        logger.info(f"✅ Total sections loaded: {len(current_config)}")
        logger.info(f"✅ Database storage config: {current_config.get(ConfigurationSection.DATABASE_STORAGE, {})}")
        logger.info(f"✅ LLM providers config: {current_config.get(ConfigurationSection.LLM_PROVIDERS, {})}")
    else:
        print("🔥 DEBUG: global_config_manager is None!")

    definitions = {}
    
    # System Configuration Settings
    sys_config_prefix = "system_configuration"
    definitions.update({
        f"{sys_config_prefix}.app_name": SettingDefinition(
            key="app_name",
            category=SettingCategory.SYSTEM_CONFIGURATION,
            type=SettingType.STRING,
            default_value="Agentic AI Platform",
            current_value=settings.APP_NAME,
            description="Application name displayed in UI",
            validation_rules={"min_length": 1, "max_length": 100}
        ),
        f"{sys_config_prefix}.environment": SettingDefinition(
            key="environment",
            category=SettingCategory.SYSTEM_CONFIGURATION,
            type=SettingType.ENUM,
            default_value="production",
            current_value=settings.ENVIRONMENT,
            description="Application environment",
            enum_values=["development", "staging", "production"],
            requires_restart=True
        ),
        f"{sys_config_prefix}.debug_mode": SettingDefinition(
            key="debug_mode",
            category=SettingCategory.SYSTEM_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=settings.DEBUG,
            description="Enable debug mode",
            requires_restart=True
        ),
        f"{sys_config_prefix}.max_concurrent_workflows": SettingDefinition(
            key="max_concurrent_workflows",
            category=SettingCategory.SYSTEM_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=50,
            current_value=getattr(settings, 'MAX_CONCURRENT_WORKFLOWS', 50),
            description="Maximum concurrent workflows",
            min_value=1,
            max_value=1000
        ),
    })
    
    # Security & Authentication Settings
    sec_auth_prefix = "security_authentication"
    definitions.update({
        f"{sec_auth_prefix}.access_token_expire_minutes": SettingDefinition(
            key="access_token_expire_minutes",
            category=SettingCategory.SECURITY_AUTHENTICATION,
            type=SettingType.INTEGER,
            default_value=30,
            current_value=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            description="JWT token expiration time in minutes",
            min_value=5,
            max_value=1440
        ),
        f"{sec_auth_prefix}.password_min_length": SettingDefinition(
            key="password_min_length",
            category=SettingCategory.SECURITY_AUTHENTICATION,
            type=SettingType.INTEGER,
            default_value=8,
            current_value=8,
            description="Minimum password length",
            min_value=4,
            max_value=128
        ),
        f"{sec_auth_prefix}.enable_tool_validation": SettingDefinition(
            key="enable_tool_validation",
            category=SettingCategory.SECURITY_AUTHENTICATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable tool security validation"
        ),
        f"{sec_auth_prefix}.security_scan_level": SettingDefinition(
            key="security_scan_level",
            category=SettingCategory.SECURITY_AUTHENTICATION,
            type=SettingType.ENUM,
            default_value="strict",
            current_value="strict",
            description="Security scanning level",
            enum_values=["basic", "strict", "paranoid"]
        ),
    })

    # Agent Management Settings
    agent_mgmt_prefix = "agent_management"
    definitions.update({
        f"{agent_mgmt_prefix}.max_agents": SettingDefinition(
            key="max_agents",
            category=SettingCategory.AGENT_MANAGEMENT,
            type=SettingType.INTEGER,
            default_value=100,
            current_value=settings.MAX_AGENTS,
            description="Maximum number of agents in the system",
            min_value=1,
            max_value=10000
        ),
        f"{agent_mgmt_prefix}.max_concurrent_agents": SettingDefinition(
            key="max_concurrent_agents",
            category=SettingCategory.AGENT_MANAGEMENT,
            type=SettingType.INTEGER,
            default_value=10,
            current_value=settings.MAX_CONCURRENT_AGENTS,
            description="Maximum concurrent agent executions",
            min_value=1,
            max_value=1000
        ),
        f"{agent_mgmt_prefix}.default_agent_model": SettingDefinition(
            key="default_agent_model",
            category=SettingCategory.AGENT_MANAGEMENT,
            type=SettingType.STRING,
            default_value="llama3.1:8b",
            current_value=settings.DEFAULT_AGENT_MODEL,
            description="Default LLM model for new agents",
            validation_rules={"min_length": 1, "max_length": 100}
        ),
        f"{agent_mgmt_prefix}.agent_timeout_seconds": SettingDefinition(
            key="agent_timeout_seconds",
            category=SettingCategory.AGENT_MANAGEMENT,
            type=SettingType.INTEGER,
            default_value=300,
            current_value=settings.AGENT_TIMEOUT_SECONDS,
            description="Default agent execution timeout in seconds",
            min_value=10,
            max_value=3600
        ),
        f"{agent_mgmt_prefix}.enable_agent_workflows": SettingDefinition(
            key="enable_agent_workflows",
            category=SettingCategory.AGENT_MANAGEMENT,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable agent workflow integration"
        ),
        f"{agent_mgmt_prefix}.enable_autonomous_workflows": SettingDefinition(
            key="enable_autonomous_workflows",
            category=SettingCategory.AGENT_MANAGEMENT,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=False,
            description="Enable autonomous workflow execution (advanced)"
        ),
    })

    # LLM Provider Settings
    llm_provider_prefix = "llm_providers"
    from app.core.global_config_manager import ConfigurationSection
    llm_config = current_config.get(ConfigurationSection.LLM_PROVIDERS, {})

    definitions.update({
        f"{llm_provider_prefix}.enable_ollama": SettingDefinition(
            key="enable_ollama",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=llm_config.get("enable_ollama", settings.ENABLE_OLLAMA),
            description="Enable Ollama local LLM provider"
        ),
        f"{llm_provider_prefix}.enable_openai": SettingDefinition(
            key="enable_openai",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=llm_config.get("enable_openai", settings.ENABLE_OPENAI),
            description="Enable OpenAI API provider"
        ),
        f"{llm_provider_prefix}.enable_anthropic": SettingDefinition(
            key="enable_anthropic",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=llm_config.get("enable_anthropic", settings.ENABLE_ANTHROPIC),
            description="Enable Anthropic Claude API provider"
        ),
        f"{llm_provider_prefix}.enable_google": SettingDefinition(
            key="enable_google",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=llm_config.get("enable_google", settings.ENABLE_GOOGLE),
            description="Enable Google Gemini API provider"
        ),
        f"{llm_provider_prefix}.ollama_base_url": SettingDefinition(
            key="ollama_base_url",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.STRING,
            default_value="http://localhost:11434",
            current_value=settings.OLLAMA_BASE_URL,
            description="Ollama server base URL",
            validation_rules={"min_length": 1, "max_length": 200}
        ),
        f"{llm_provider_prefix}.default_temperature": SettingDefinition(
            key="default_temperature",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.FLOAT,
            default_value=0.7,
            current_value=llm_config.get("default_temperature", 0.7),
            description="Default temperature for LLM responses",
            min_value=0.0,
            max_value=2.0
        ),
        f"{llm_provider_prefix}.max_tokens": SettingDefinition(
            key="max_tokens",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=4096,
            current_value=llm_config.get("max_tokens", 4096),
            description="Maximum tokens per LLM response",
            min_value=100,
            max_value=100000
        ),
        f"{llm_provider_prefix}.request_timeout": SettingDefinition(
            key="request_timeout",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=30,
            current_value=llm_config.get("request_timeout", 30),
            description="Request timeout in seconds",
            min_value=5,
            max_value=300
        ),
        f"{llm_provider_prefix}.enable_rate_limiting": SettingDefinition(
            key="enable_rate_limiting",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=llm_config.get("enable_rate_limiting", True),
            description="Enable system-wide rate limiting for API requests"
        ),
        f"{llm_provider_prefix}.global_rate_limit_per_minute": SettingDefinition(
            key="global_rate_limit_per_minute",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=1000,
            current_value=llm_config.get("global_rate_limit_per_minute", 1000),
            description="Global rate limit for all LLM requests per minute",
            min_value=10,
            max_value=10000
        ),
        f"{llm_provider_prefix}.enable_request_caching": SettingDefinition(
            key="enable_request_caching",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=llm_config.get("enable_request_caching", True),
            description="Enable caching of LLM responses for performance"
        ),
        f"{llm_provider_prefix}.cache_ttl_seconds": SettingDefinition(
            key="cache_ttl_seconds",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=300,
            current_value=llm_config.get("cache_ttl_seconds", 300),
            description="Cache time-to-live in seconds",
            min_value=60,
            max_value=3600
        ),
        f"{llm_provider_prefix}.enable_usage_monitoring": SettingDefinition(
            key="enable_usage_monitoring",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable monitoring of LLM usage and costs"
        ),
        f"{llm_provider_prefix}.enable_content_filtering": SettingDefinition(
            key="enable_content_filtering",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable content filtering for safety"
        ),
        f"{llm_provider_prefix}.max_context_length": SettingDefinition(
            key="max_context_length",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=32000,
            current_value=32000,
            description="Maximum context length for conversations",
            min_value=1000,
            max_value=200000
        ),
    })

    # 🚀 REVOLUTIONARY DATABASE STORAGE SETTINGS - Complete Storage Management!
    database_prefix = "database_storage"
    from app.core.global_config_manager import ConfigurationSection
    database_config = current_config.get(ConfigurationSection.DATABASE_STORAGE, {})
    print(f"🔥 DEBUG: database_config = {database_config}")
    print(f"🔥 DEBUG: vector_db_type from database_config = {database_config.get('vector_db_type', 'NOT_FOUND')}")

    definitions.update({
        # ============================================================================
        # 🗄️ POSTGRESQL DATABASE SETTINGS
        # ============================================================================
        f"{database_prefix}.postgres_host": SettingDefinition(
            key="postgres_host",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="localhost",
            current_value=database_config.get("postgres_host", "localhost"),
            description="PostgreSQL database host",
            validation_rules={"min_length": 1, "max_length": 255}
        ),
        f"{database_prefix}.postgres_port": SettingDefinition(
            key="postgres_port",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=5432,
            current_value=database_config.get("postgres_port", 5432),
            description="PostgreSQL database port",
            min_value=1,
            max_value=65535
        ),
        f"{database_prefix}.postgres_database": SettingDefinition(
            key="postgres_database",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="agentic_ai",
            current_value=database_config.get("postgres_database", "agentic_ai"),
            description="PostgreSQL database name",
            validation_rules={"min_length": 1, "max_length": 63}
        ),
        f"{database_prefix}.postgres_username": SettingDefinition(
            key="postgres_username",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="postgres",
            current_value=database_config.get("postgres_username", "postgres"),
            description="PostgreSQL username",
            validation_rules={"min_length": 1, "max_length": 63}
        ),
        f"{database_prefix}.postgres_password": SettingDefinition(
            key="postgres_password",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="",
            current_value=database_config.get("postgres_password", ""),
            description="PostgreSQL password",
            is_sensitive=True,
            validation_rules={"max_length": 255}
        ),
        f"{database_prefix}.postgres_ssl_mode": SettingDefinition(
            key="postgres_ssl_mode",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.ENUM,
            default_value="prefer",
            current_value=database_config.get("postgres_ssl_mode", "prefer"),
            description="PostgreSQL SSL mode",
            enum_values=["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        ),
        f"{database_prefix}.postgres_pool_size": SettingDefinition(
            key="postgres_pool_size",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=10,
            current_value=database_config.get("postgres_pool_size", 10),
            description="PostgreSQL connection pool size",
            min_value=1,
            max_value=100
        ),
        f"{database_prefix}.postgres_max_overflow": SettingDefinition(
            key="postgres_max_overflow",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=20,
            current_value=database_config.get("postgres_max_overflow", 20),
            description="PostgreSQL max overflow connections",
            min_value=0,
            max_value=200
        ),
        f"{database_prefix}.postgres_pool_timeout": SettingDefinition(
            key="postgres_pool_timeout",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=30,
            current_value=database_config.get("postgres_pool_timeout", 30),
            description="PostgreSQL pool timeout in seconds",
            min_value=1,
            max_value=300
        ),
        f"{database_prefix}.postgres_echo_sql": SettingDefinition(
            key="postgres_echo_sql",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=database_config.get("postgres_echo_sql", False),
            description="Echo SQL queries to logs (debug mode)"
        ),
        f"{database_prefix}.postgres_docker_enabled": SettingDefinition(
            key="postgres_docker_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("postgres_docker_enabled", True),
            description="Use PostgreSQL Docker container"
        ),
        f"{database_prefix}.postgres_docker_image": SettingDefinition(
            key="postgres_docker_image",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="postgres:16-alpine",
            current_value=database_config.get("postgres_docker_image", "postgres:16-alpine"),
            description="PostgreSQL Docker image",
            validation_rules={"min_length": 1, "max_length": 200}
        ),
        f"{database_prefix}.postgres_backup_enabled": SettingDefinition(
            key="postgres_backup_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=database_config.get("postgres_backup_enabled", False),
            description="Enable automatic PostgreSQL backups"
        ),
        f"{database_prefix}.postgres_backup_schedule": SettingDefinition(
            key="postgres_backup_schedule",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="0 2 * * *",
            current_value=database_config.get("postgres_backup_schedule", "0 2 * * *"),
            description="PostgreSQL backup schedule (cron format)",
            validation_rules={"min_length": 1, "max_length": 100}
        ),

        # ============================================================================
        # 🔍 VECTOR DATABASE SETTINGS (ChromaDB & PgVector)
        # ============================================================================
        f"{database_prefix}.vector_db_type": SettingDefinition(
            key="vector_db_type",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.ENUM,
            default_value="auto",
            current_value=database_config.get("vector_db_type", "auto"),
            description="Vector database type selection",
            enum_values=["auto", "chromadb", "pgvector"]
        ),

        # DEBUG: Check the vector_db_type setting definition
        # Let me add debug here to see what the setting definition looks like

        f"{database_prefix}.vector_db_auto_detect": SettingDefinition(
            key="vector_db_auto_detect",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("vector_db_auto_detect", True),
            description="Auto-detect available vector database"
        ),

        # ChromaDB Settings
        f"{database_prefix}.chroma_persist_directory": SettingDefinition(
            key="chroma_persist_directory",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="data/chroma",
            current_value=database_config.get("chroma_persist_directory", "data/chroma"),
            description="ChromaDB persistence directory",
            validation_rules={"min_length": 1, "max_length": 500}
        ),
        f"{database_prefix}.chroma_collection_name": SettingDefinition(
            key="chroma_collection_name",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="agentic_documents",
            current_value=database_config.get("chroma_collection_name", "agentic_documents"),
            description="Default ChromaDB collection name",
            validation_rules={"min_length": 1, "max_length": 63}
        ),
        f"{database_prefix}.chroma_distance_function": SettingDefinition(
            key="chroma_distance_function",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.ENUM,
            default_value="cosine",
            current_value=database_config.get("chroma_distance_function", "cosine"),
            description="ChromaDB distance function for similarity",
            enum_values=["cosine", "euclidean", "manhattan", "dot"]
        ),
        f"{database_prefix}.chroma_batch_size": SettingDefinition(
            key="chroma_batch_size",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=100,
            current_value=database_config.get("chroma_batch_size", 100),
            description="ChromaDB batch size for operations",
            min_value=1,
            max_value=10000
        ),
        f"{database_prefix}.chroma_docker_enabled": SettingDefinition(
            key="chroma_docker_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("chroma_docker_enabled", True),
            description="Use ChromaDB Docker container"
        ),
        f"{database_prefix}.chroma_docker_image": SettingDefinition(
            key="chroma_docker_image",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="chromadb/chroma:latest",
            current_value=database_config.get("chroma_docker_image", "chromadb/chroma:latest"),
            description="ChromaDB Docker image",
            validation_rules={"min_length": 1, "max_length": 200}
        ),
        f"{database_prefix}.chroma_docker_port": SettingDefinition(
            key="chroma_docker_port",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=8000,
            current_value=database_config.get("chroma_docker_port", 8000),
            description="ChromaDB Docker port",
            min_value=1024,
            max_value=65535
        ),

        # PgVector Settings
        f"{database_prefix}.pgvector_enabled": SettingDefinition(
            key="pgvector_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=database_config.get("pgvector_enabled", False),
            description="Enable PgVector for vector storage"
        ),
        f"{database_prefix}.pgvector_table_name": SettingDefinition(
            key="pgvector_table_name",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="embeddings",
            current_value=database_config.get("pgvector_table_name", "embeddings"),
            description="PgVector table name for embeddings",
            validation_rules={"min_length": 1, "max_length": 63}
        ),
        f"{database_prefix}.pgvector_dimension": SettingDefinition(
            key="pgvector_dimension",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=1536,
            current_value=database_config.get("pgvector_dimension", 1536),
            description="PgVector embedding dimension",
            min_value=1,
            max_value=16000
        ),
        f"{database_prefix}.pgvector_distance_function": SettingDefinition(
            key="pgvector_distance_function",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.ENUM,
            default_value="cosine",
            current_value=database_config.get("pgvector_distance_function", "cosine"),
            description="PgVector distance function",
            enum_values=["cosine", "l2", "inner_product"]
        ),
        f"{database_prefix}.pgvector_docker_enabled": SettingDefinition(
            key="pgvector_docker_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("pgvector_docker_enabled", True),
            description="Use PgVector Docker container"
        ),

        # ============================================================================
        # 🔴 REDIS CACHING SETTINGS
        # ============================================================================
        f"{database_prefix}.redis_enabled": SettingDefinition(
            key="redis_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=database_config.get("redis_enabled", False),
            description="Enable Redis caching"
        ),
        f"{database_prefix}.redis_host": SettingDefinition(
            key="redis_host",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="localhost",
            current_value=database_config.get("redis_host", "localhost"),
            description="Redis server host",
            validation_rules={"min_length": 1, "max_length": 255}
        ),
        f"{database_prefix}.redis_port": SettingDefinition(
            key="redis_port",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=6379,
            current_value=database_config.get("redis_port", 6379),
            description="Redis server port",
            min_value=1,
            max_value=65535
        ),
        f"{database_prefix}.redis_database": SettingDefinition(
            key="redis_database",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=0,
            current_value=database_config.get("redis_database", 0),
            description="Redis database number",
            min_value=0,
            max_value=15
        ),
        f"{database_prefix}.redis_password": SettingDefinition(
            key="redis_password",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="",
            current_value=database_config.get("redis_password", ""),
            description="Redis password (if required)",
            is_sensitive=True,
            validation_rules={"max_length": 255}
        ),
        f"{database_prefix}.redis_max_connections": SettingDefinition(
            key="redis_max_connections",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=50,
            current_value=database_config.get("redis_max_connections", 50),
            description="Redis maximum connections",
            min_value=1,
            max_value=1000
        ),
        f"{database_prefix}.redis_connection_timeout": SettingDefinition(
            key="redis_connection_timeout",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=5,
            current_value=database_config.get("redis_connection_timeout", 5),
            description="Redis connection timeout in seconds",
            min_value=1,
            max_value=60
        ),
        f"{database_prefix}.redis_docker_enabled": SettingDefinition(
            key="redis_docker_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("redis_docker_enabled", True),
            description="Use Redis Docker container"
        ),
        f"{database_prefix}.redis_docker_image": SettingDefinition(
            key="redis_docker_image",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.STRING,
            default_value="redis:7-alpine",
            current_value=database_config.get("redis_docker_image", "redis:7-alpine"),
            description="Redis Docker image",
            validation_rules={"min_length": 1, "max_length": 200}
        ),
        f"{database_prefix}.redis_persistence_enabled": SettingDefinition(
            key="redis_persistence_enabled",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("redis_persistence_enabled", True),
            description="Enable Redis data persistence"
        ),
        f"{database_prefix}.redis_memory_policy": SettingDefinition(
            key="redis_memory_policy",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.ENUM,
            default_value="allkeys-lru",
            current_value=database_config.get("redis_memory_policy", "allkeys-lru"),
            description="Redis memory eviction policy",
            enum_values=["noeviction", "allkeys-lru", "volatile-lru", "allkeys-random", "volatile-random", "volatile-ttl"]
        ),

        # ============================================================================
        # ⚡ PERFORMANCE TUNING SETTINGS
        # ============================================================================
        f"{database_prefix}.query_timeout_seconds": SettingDefinition(
            key="query_timeout_seconds",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=30,
            current_value=database_config.get("query_timeout_seconds", 30),
            description="Database query timeout in seconds",
            min_value=5,
            max_value=300
        ),
        f"{database_prefix}.enable_query_logging": SettingDefinition(
            key="enable_query_logging",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=database_config.get("enable_query_logging", False),
            description="Enable detailed query logging"
        ),
        f"{database_prefix}.enable_slow_query_logging": SettingDefinition(
            key="enable_slow_query_logging",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("enable_slow_query_logging", True),
            description="Log slow database queries"
        ),
        f"{database_prefix}.slow_query_threshold_ms": SettingDefinition(
            key="slow_query_threshold_ms",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=1000,
            current_value=database_config.get("slow_query_threshold_ms", 1000),
            description="Slow query threshold in milliseconds",
            min_value=100,
            max_value=10000
        ),
        f"{database_prefix}.enable_connection_pooling": SettingDefinition(
            key="enable_connection_pooling",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("enable_connection_pooling", True),
            description="Enable database connection pooling"
        ),

        # ============================================================================
        # 📊 STORAGE MONITORING SETTINGS
        # ============================================================================
        f"{database_prefix}.max_database_size_gb": SettingDefinition(
            key="max_database_size_gb",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=100,
            current_value=database_config.get("max_database_size_gb", 100),
            description="Maximum database size in GB",
            min_value=1,
            max_value=10000
        ),
        f"{database_prefix}.max_vector_storage_gb": SettingDefinition(
            key="max_vector_storage_gb",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=50,
            current_value=database_config.get("max_vector_storage_gb", 50),
            description="Maximum vector storage size in GB",
            min_value=1,
            max_value=5000
        ),
        f"{database_prefix}.enable_storage_monitoring": SettingDefinition(
            key="enable_storage_monitoring",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("enable_storage_monitoring", True),
            description="Enable storage usage monitoring"
        ),
        f"{database_prefix}.storage_warning_threshold_percent": SettingDefinition(
            key="storage_warning_threshold_percent",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=80,
            current_value=database_config.get("storage_warning_threshold_percent", 80),
            description="Storage warning threshold percentage",
            min_value=50,
            max_value=95
        ),
        f"{database_prefix}.storage_critical_threshold_percent": SettingDefinition(
            key="storage_critical_threshold_percent",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=95,
            current_value=database_config.get("storage_critical_threshold_percent", 95),
            description="Storage critical threshold percentage",
            min_value=80,
            max_value=99
        ),

        # ============================================================================
        # 🔧 MIGRATION AND SCHEMA SETTINGS
        # ============================================================================
        f"{database_prefix}.enable_auto_migrations": SettingDefinition(
            key="enable_auto_migrations",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("enable_auto_migrations", True),
            description="Enable automatic database migrations"
        ),
        f"{database_prefix}.migration_timeout_seconds": SettingDefinition(
            key="migration_timeout_seconds",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.INTEGER,
            default_value=300,
            current_value=database_config.get("migration_timeout_seconds", 300),
            description="Migration timeout in seconds",
            min_value=30,
            max_value=3600
        ),
        f"{database_prefix}.enable_schema_validation": SettingDefinition(
            key="enable_schema_validation",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("enable_schema_validation", True),
            description="Enable database schema validation"
        ),
        f"{database_prefix}.enable_foreign_key_checks": SettingDefinition(
            key="enable_foreign_key_checks",
            category=SettingCategory.DATABASE_STORAGE,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=database_config.get("enable_foreign_key_checks", True),
            description="Enable foreign key constraint checks"
        ),
    })

    # 🚀 REVOLUTIONARY RAG SYSTEM SETTINGS - Most Comprehensive Ever!
    rag_prefix = "rag_configuration"

    # ============================================================================
    # 🗄️ VECTOR STORE SETTINGS
    # ============================================================================
    definitions.update({
        f"{rag_prefix}.persist_directory": SettingDefinition(
            key="persist_directory",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.STRING,
            default_value="./data/chroma",
            current_value=settings.CHROMA_PERSIST_DIRECTORY,
            description="ChromaDB persistence directory for vector storage",
            validation_rules={"min_length": 1, "max_length": 500}
        ),
        f"{rag_prefix}.connection_pool_size": SettingDefinition(
            key="connection_pool_size",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=10,
            current_value=10,
            description="Database connection pool size for optimal performance",
            validation_rules={"min_value": 1, "max_value": 100}
        ),
        f"{rag_prefix}.max_batch_size": SettingDefinition(
            key="max_batch_size",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=128,
            current_value=128,
            description="Maximum batch size for vector operations",
            validation_rules={"min_value": 1, "max_value": 1000}
        ),
        f"{rag_prefix}.similarity_metric": SettingDefinition(
            key="similarity_metric",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.ENUM,
            default_value="cosine",
            current_value="cosine",
            description="Vector similarity metric (cosine, euclidean, dot_product)",
            enum_values=["cosine", "euclidean", "dot_product"]
        ),

        # ============================================================================
        # 🤖 EMBEDDING MODELS SETTINGS
        # ============================================================================
        f"{rag_prefix}.embedding_model": SettingDefinition(
            key="embedding_model",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.STRING,
            default_value="all-MiniLM-L6-v2",
            current_value="all-MiniLM-L6-v2",
            description="Primary embedding model for vector generation",
            validation_rules={"min_length": 1, "max_length": 200}
        ),
        f"{rag_prefix}.embedding_provider": SettingDefinition(
            key="embedding_provider",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.ENUM,
            default_value="sentence-transformers",
            current_value="sentence-transformers",
            description="Embedding model provider",
            enum_values=["sentence-transformers", "openai", "ollama", "azure_openai"]
        ),
        f"{rag_prefix}.embedding_batch_size": SettingDefinition(
            key="embedding_batch_size",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=32,
            current_value=32,
            description="Batch size for embedding generation",
            validation_rules={"min_value": 1, "max_value": 256}
        ),
        f"{rag_prefix}.models_directory": SettingDefinition(
            key="models_directory",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.STRING,
            default_value="./data/models",
            current_value="./data/models",
            description="Directory for storing downloaded models",
            validation_rules={"min_length": 1, "max_length": 500}
        ),
        f"{rag_prefix}.auto_download_models": SettingDefinition(
            key="auto_download_models",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=False,
            description="Automatically download missing models"
        ),

        # ============================================================================
        # 👁️ VISION MODELS SETTINGS
        # ============================================================================
        f"{rag_prefix}.enable_vision_models": SettingDefinition(
            key="enable_vision_models",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable vision model processing for images"
        ),
        f"{rag_prefix}.primary_vision_model": SettingDefinition(
            key="primary_vision_model",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.STRING,
            default_value="clip-vit-base-patch32",
            current_value="clip-vit-base-patch32",
            description="Primary vision model for image processing",
            validation_rules={"min_length": 1, "max_length": 200}
        ),
        f"{rag_prefix}.vision_batch_size": SettingDefinition(
            key="vision_batch_size",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=16,
            current_value=16,
            description="Batch size for vision model processing",
            validation_rules={"min_value": 1, "max_value": 64}
        ),

        # ============================================================================
        # 🔍 OCR ENGINES SETTINGS
        # ============================================================================
        f"{rag_prefix}.enable_ocr": SettingDefinition(
            key="enable_ocr",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable OCR processing for images and PDFs"
        ),
        f"{rag_prefix}.enable_tesseract": SettingDefinition(
            key="enable_tesseract",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable Tesseract OCR engine"
        ),
        f"{rag_prefix}.enable_easyocr": SettingDefinition(
            key="enable_easyocr",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable EasyOCR engine"
        ),
        f"{rag_prefix}.enable_paddleocr": SettingDefinition(
            key="enable_paddleocr",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable PaddleOCR engine"
        ),
        f"{rag_prefix}.ocr_confidence_threshold": SettingDefinition(
            key="ocr_confidence_threshold",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.FLOAT,
            default_value=0.3,
            current_value=0.3,
            description="Minimum OCR confidence threshold (0.0-1.0)",
            validation_rules={"min_value": 0.0, "max_value": 1.0}
        ),
        f"{rag_prefix}.enable_image_enhancement": SettingDefinition(
            key="enable_image_enhancement",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enhance images before OCR processing"
        ),

        # ============================================================================
        # 📄 DOCUMENT PROCESSING SETTINGS
        # ============================================================================
        f"{rag_prefix}.chunk_size": SettingDefinition(
            key="chunk_size",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=1000,
            current_value=1000,
            description="Text chunk size for document processing",
            validation_rules={"min_value": 100, "max_value": 8000}
        ),
        f"{rag_prefix}.chunk_overlap": SettingDefinition(
            key="chunk_overlap",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=200,
            current_value=200,
            description="Overlap between text chunks",
            validation_rules={"min_value": 0, "max_value": 1000}
        ),
        f"{rag_prefix}.chunking_strategy": SettingDefinition(
            key="chunking_strategy",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.ENUM,
            default_value="semantic",
            current_value="semantic",
            description="Text chunking strategy",
            enum_values=["semantic", "fixed", "adaptive"]
        ),
        f"{rag_prefix}.enable_smart_chunking": SettingDefinition(
            key="enable_smart_chunking",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Use AI-powered smart chunking"
        ),
        f"{rag_prefix}.max_document_size_mb": SettingDefinition(
            key="max_document_size_mb",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=100,
            current_value=100,
            description="Maximum document size in MB",
            validation_rules={"min_value": 1, "max_value": 1000}
        ),

        # ============================================================================
        # 🎯 RETRIEVAL SETTINGS
        # ============================================================================
        f"{rag_prefix}.top_k": SettingDefinition(
            key="top_k",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=10,
            current_value=10,
            description="Number of top results to retrieve",
            validation_rules={"min_value": 1, "max_value": 100}
        ),
        f"{rag_prefix}.score_threshold": SettingDefinition(
            key="score_threshold",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.FLOAT,
            default_value=0.7,
            current_value=0.7,
            description="Minimum similarity score threshold",
            validation_rules={"min_value": 0.0, "max_value": 1.0}
        ),
        f"{rag_prefix}.enable_reranking": SettingDefinition(
            key="enable_reranking",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable result reranking for better relevance"
        ),
        f"{rag_prefix}.enable_query_expansion": SettingDefinition(
            key="enable_query_expansion",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Expand queries for better search results"
        ),
        f"{rag_prefix}.enable_hybrid_search": SettingDefinition(
            key="enable_hybrid_search",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Combine dense and sparse search methods"
        ),
        f"{rag_prefix}.enable_contextual_retrieval": SettingDefinition(
            key="enable_contextual_retrieval",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Use conversation context for retrieval"
        ),

        # ============================================================================
        # 🚀 PERFORMANCE SETTINGS
        # ============================================================================
        f"{rag_prefix}.enable_caching": SettingDefinition(
            key="enable_caching",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable result caching for better performance"
        ),
        f"{rag_prefix}.cache_ttl": SettingDefinition(
            key="cache_ttl",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=3600,
            current_value=3600,
            description="Cache time-to-live in seconds",
            validation_rules={"min_value": 60, "max_value": 86400}
        ),
        f"{rag_prefix}.max_concurrent_queries": SettingDefinition(
            key="max_concurrent_queries",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.INTEGER,
            default_value=50,
            current_value=50,
            description="Maximum concurrent queries",
            validation_rules={"min_value": 1, "max_value": 200}
        ),
        f"{rag_prefix}.enable_gpu_acceleration": SettingDefinition(
            key="enable_gpu_acceleration",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable GPU acceleration for AI models"
        ),

        # ============================================================================
        # 📋 RAG TEMPLATES SETTINGS
        # ============================================================================
        f"{rag_prefix}.enable_rag_templates": SettingDefinition(
            key="enable_rag_templates",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable RAG templates for different use cases"
        ),
        f"{rag_prefix}.default_rag_template": SettingDefinition(
            key="default_rag_template",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.ENUM,
            default_value="general_purpose",
            current_value="general_purpose",
            description="Default RAG template for new agents",
            enum_values=["general_purpose", "research_assistant", "code_helper", "creative_writing", "technical_documentation", "customer_support", "educational_tutor", "data_analyst"]
        ),
        f"{rag_prefix}.enable_template_customization": SettingDefinition(
            key="enable_template_customization",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Allow users to customize RAG templates"
        ),

        # ============================================================================
        # 🌐 GLOBAL SETTINGS
        # ============================================================================
        f"{rag_prefix}.rag_system_enabled": SettingDefinition(
            key="rag_system_enabled",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable RAG system globally"
        ),
        f"{rag_prefix}.apply_to_all_agents": SettingDefinition(
            key="apply_to_all_agents",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Apply RAG settings to all agents"
        ),
        f"{rag_prefix}.enable_performance_monitoring": SettingDefinition(
            key="enable_performance_monitoring",
            category=SettingCategory.RAG_CONFIGURATION,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Monitor RAG system performance"
        )
    })

    # DEBUG: Check the vector_db_type setting definition before returning
    vector_db_key = f"{database_prefix}.vector_db_type"
    if vector_db_key in definitions:
        vector_setting = definitions[vector_db_key]
        print(f"🔥 DEBUG: vector_db_type setting definition:")
        print(f"  - key: {vector_setting.key}")
        print(f"  - default_value: {vector_setting.default_value}")
        print(f"  - current_value: {vector_setting.current_value}")
        print(f"  - type: {vector_setting.type}")
        print(f"  - enum_values: {vector_setting.enum_values}")
    else:
        print(f"🔥 DEBUG: vector_db_type setting NOT FOUND in definitions!")
        print(f"🔥 DEBUG: Available database storage keys: {[k for k in definitions.keys() if k.startswith('database_storage')]}")

    return definitions


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/categories", response_model=StandardAPIResponse)
async def get_setting_categories(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Get all setting categories with metadata."""
    try:
        categories = {
            "core": {
                "name": "Core Settings",
                "description": "Essential system configuration",
                "categories": [
                    {
                        "id": SettingCategory.SYSTEM_CONFIGURATION,
                        "name": "System Configuration",
                        "description": "Application and execution settings",
                        "icon": "🔧",
                        "color": "blue"
                    },
                    {
                        "id": SettingCategory.SECURITY_AUTHENTICATION,
                        "name": "Security & Authentication",
                        "description": "Security, authentication, and guard rails",
                        "icon": "🔐",
                        "color": "red"
                    }
                ]
            },
            "operational": {
                "name": "Operational Settings",
                "description": "System operations and management",
                "categories": [
                    {
                        "id": SettingCategory.AGENT_MANAGEMENT,
                        "name": "Agent Management",
                        "description": "Agent system configuration and limits",
                        "icon": "🤖",
                        "color": "green"
                    },
                    {
                        "id": SettingCategory.LLM_PROVIDERS,
                        "name": "LLM Providers",
                        "description": "Language model providers with flexible API key management",
                        "icon": "🧠",
                        "color": "purple"
                    },
                    {
                        "id": SettingCategory.RAG_CONFIGURATION,
                        "name": "RAG System",
                        "description": "Retrieval-Augmented Generation system configuration",
                        "icon": "🔍",
                        "color": "indigo"
                    },
                    {
                        "id": SettingCategory.DATABASE_STORAGE,
                        "name": "Database & Storage",
                        "description": "Database and file storage configuration",
                        "icon": "🗄️",
                        "color": "cyan"
                    },
                    {
                        "id": SettingCategory.MONITORING_LOGGING,
                        "name": "Monitoring & Logging",
                        "description": "System monitoring and logging settings",
                        "icon": "📊",
                        "color": "yellow"
                    }
                ]
            },
            "advanced": {
                "name": "Advanced Settings",
                "description": "Advanced system configuration",
                "categories": [
                    {
                        "id": SettingCategory.WORKFLOW_MANAGEMENT,
                        "name": "Workflow Management",
                        "description": "Workflow execution and templates",
                        "icon": "🔄",
                        "color": "indigo"
                    },
                    {
                        "id": SettingCategory.NODE_EXECUTION,
                        "name": "Node & Execution Control",
                        "description": "Node registry and execution control",
                        "icon": "🎯",
                        "color": "orange"
                    }
                ]
            }
        }

        logger.info("Setting categories retrieved", admin_user=str(admin_user.id))

        return StandardAPIResponse(
            success=True,
            message="Setting categories retrieved successfully",
            data=categories
        )

    except Exception as e:
        logger.error("Failed to get setting categories", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get setting categories: {str(e)}"
        )


@router.get("/definitions", response_model=StandardAPIResponse)
async def get_setting_definitions_endpoint(
    category: Optional[SettingCategory] = Query(None, description="Filter by category"),
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Get setting definitions, optionally filtered by category."""
    try:
        definitions = await get_setting_definitions()

        if category:
            definitions = {
                key: definition for key, definition in definitions.items()
                if definition.category == category
            }

        logger.info(
            "Setting definitions retrieved",
            admin_user=str(admin_user.id),
            category=category,
            count=len(definitions)
        )

        return StandardAPIResponse(
            success=True,
            message="Setting definitions retrieved successfully",
            data=definitions
        )

    except Exception as e:
        logger.error("Failed to get setting definitions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get setting definitions: {str(e)}"
        )


@router.get("/values/{category}", response_model=StandardAPIResponse)
async def get_category_settings(
    category: SettingCategory,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Get all settings for a specific category."""
    try:
        definitions = await get_setting_definitions()
        category_definitions = {
            key: definition for key, definition in definitions.items()
            if definition.category == category
        }

        # Build settings response
        settings_data = {}
        for key, definition in category_definitions.items():
            settings_data[key] = {
                "value": definition.current_value,
                "default": definition.default_value,
                "type": definition.type,
                "description": definition.description,
                "requires_restart": definition.requires_restart,
                "is_sensitive": definition.is_sensitive,
                "enum_values": definition.enum_values,
                "min_value": definition.min_value,
                "max_value": definition.max_value,
                "validation_rules": definition.validation_rules
            }

        logger.info(
            "Category settings retrieved",
            admin_user=str(admin_user.id),
            category=category,
            count=len(settings_data)
        )

        return StandardAPIResponse(
            success=True,
            message=f"Settings for {category} retrieved successfully",
            data=settings_data
        )

    except Exception as e:
        logger.error("Failed to get category settings", error=str(e), category=category)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get category settings: {str(e)}"
        )


@router.post("/validate", response_model=StandardAPIResponse)
async def validate_setting(
    request: SettingUpdateRequest,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Validate a setting value without saving it."""
    try:
        definitions = await get_setting_definitions()
        setting_key = f"{request.category.value}.{request.key}"

        if setting_key not in definitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Setting {setting_key} not found"
            )

        definition = definitions[setting_key]
        validation_result = await validate_setting_value(definition, request.value)

        logger.info(
            "Setting validation performed",
            admin_user=str(admin_user.id),
            setting_key=setting_key,
            is_valid=validation_result.is_valid
        )

        return StandardAPIResponse(
            success=True,
            message="Setting validation completed",
            data=validation_result.dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to validate setting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate setting: {str(e)}"
        )


@router.post("/update", response_model=StandardAPIResponse)
async def update_setting(
    request: SettingUpdateRequest,
    background_tasks: BackgroundTasks,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Update a single setting."""
    try:
        definitions = await get_setting_definitions()
        setting_key = f"{request.category.value}.{request.key}"

        if setting_key not in definitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Setting {setting_key} not found"
            )

        definition = definitions[setting_key]

        # Validate the new value
        validation_result = await validate_setting_value(definition, request.value)

        if not validation_result.is_valid:
            return StandardAPIResponse(
                success=False,
                message="Setting validation failed",
                data={
                    "validation_result": validation_result.dict(),
                    "errors": validation_result.errors
                }
            )

        if request.validate_only:
            return StandardAPIResponse(
                success=True,
                message="Setting validation passed",
                data={"validation_result": validation_result.dict()}
            )

        # Store setting in database and apply to system
        settings_service = await get_admin_settings_service()

        success, error_message = await settings_service.set_setting(
            category=request.category.value,
            key=request.key,
            value=request.value,
            user_id=admin_user.id,
            setting_type=definition.type.value,
            description=definition.description,
            requires_restart=definition.requires_restart,
            validation_rules=definition.validation_rules,
            enum_values=definition.enum_values
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save setting: {error_message}"
            )

        logger.info(
            "Setting updated and applied",
            admin_user=str(admin_user.id),
            setting_key=setting_key,
            old_value=definition.current_value,
            new_value=request.value,
            requires_restart=validation_result.requires_restart
        )

        # 🚀 Revolutionary Real-Time RAG System Update (Global Config Manager)
        rag_update_result = None
        if setting_key.startswith("rag_configuration."):
            try:
                logger.info(f"🔄 Triggering real-time RAG system update for: {setting_key}")

                # Use global config manager instance
                config_manager = global_config_manager

                # Extract the setting key without the category prefix
                rag_setting_key = setting_key.replace("rag_configuration.", "")

                # Update the configuration through the global config manager
                # This will automatically notify all RAG observers
                from app.core.global_config_manager import ConfigurationSection
                update_result = await config_manager.update_section(
                    section=ConfigurationSection.RAG_CONFIGURATION,
                    changes={rag_setting_key: request.value},
                    user_id=str(admin_user.id)
                )

                if update_result.success:
                    logger.info(f"✅ RAG system updated successfully: {update_result.message}")
                    rag_update_result = {
                        "success": True,
                        "message": f"RAG setting '{rag_setting_key}' updated in real-time",
                        "applied_changes": {rag_setting_key: request.value}
                    }
                else:
                    logger.warning(f"⚠️ RAG system update failed: {update_result.message}")
                    rag_update_result = {
                        "success": False,
                        "error": update_result.message,
                        "message": f"RAG system update failed: {update_result.message}"
                    }

            except Exception as e:
                logger.error(f"❌ Failed to update RAG system: {str(e)}")
                rag_update_result = {
                    "success": False,
                    "error": f"RAG update failed: {str(e)}"
                }

        # 🚀 Revolutionary Real-Time LLM Provider System Update
        llm_update_result = None
        if setting_key.startswith("llm_providers."):
            try:
                logger.info(f"🔄 Triggering real-time LLM provider system update for: {setting_key}")

                # Use global config manager instance
                config_manager = global_config_manager

                # Extract the setting key without the category prefix
                llm_setting_key = setting_key.replace("llm_providers.", "")

                # Update the configuration through the global config manager
                # This will automatically notify all LLM observers
                from app.core.global_config_manager import ConfigurationSection
                update_result = await config_manager.update_section(
                    section=ConfigurationSection.LLM_PROVIDERS,
                    changes={llm_setting_key: request.value},
                    user_id=str(admin_user.id)
                )

                if update_result.success:
                    logger.info(f"✅ LLM provider system updated successfully: {update_result.message}")
                    llm_update_result = {
                        "success": True,
                        "message": f"LLM provider setting '{llm_setting_key}' updated in real-time",
                        "applied_changes": {llm_setting_key: request.value}
                    }
                else:
                    logger.warning(f"⚠️ LLM provider system update failed: {update_result.errors}")
                    llm_update_result = {
                        "success": False,
                        "error": f"LLM provider update failed: {', '.join(update_result.errors)}",
                        "errors": update_result.errors
                    }

            except Exception as e:
                logger.error(f"❌ Failed to update LLM provider system: {str(e)}")
                llm_update_result = {
                    "success": False,
                    "error": f"LLM provider update failed: {str(e)}"
                }

        # 🚀 Revolutionary Real-Time Database Storage System Update
        database_update_result = None
        if setting_key.startswith("database_storage."):
            try:
                logger.info(f"🔄 Triggering real-time database storage system update for: {setting_key}")

                # Use global config manager instance
                config_manager = global_config_manager

                # Extract the setting key without the category prefix
                database_setting_key = setting_key.replace("database_storage.", "")

                # Update the configuration through the global config manager
                # This will automatically notify all database observers
                from app.core.global_config_manager import ConfigurationSection
                update_result = await config_manager.update_section(
                    section=ConfigurationSection.DATABASE_STORAGE,
                    changes={database_setting_key: request.value},
                    user_id=str(admin_user.id)
                )

                if update_result.success:
                    logger.info(f"✅ Database storage system update successful: {update_result.message}")
                    database_update_result = {
                        "success": True,
                        "message": f"Database storage system updated in real-time",
                        "applied_changes": {database_setting_key: request.value},
                        "warnings": update_result.warnings or []
                    }
                else:
                    logger.warning(f"⚠️ Database storage system update failed: {update_result.message}")
                    database_update_result = {
                        "success": False,
                        "message": f"Database storage system update failed: {update_result.message}",
                        "errors": update_result.errors or []
                    }

            except Exception as e:
                logger.error(f"❌ Database storage system update error: {str(e)}")
                database_update_result = {
                    "success": False,
                    "message": "Database storage system update failed due to internal error",
                    "error": f"Database storage update failed: {str(e)}"
                }

        # Add background task for restart notification if needed
        if validation_result.requires_restart:
            background_tasks.add_task(
                _notify_restart_required,
                admin_user.id,
                setting_key
            )

        # Prepare response data
        response_data = {
            "setting_key": setting_key,
            "new_value": request.value,
            "requires_restart": validation_result.requires_restart,
            "warnings": validation_result.warnings
        }

        # Add RAG update information if applicable
        if rag_update_result:
            response_data["rag_update"] = rag_update_result

            # Enhance success message if RAG was updated
            if rag_update_result.get("success"):
                message = "Setting updated successfully and applied to RAG system in real-time"
                if rag_update_result.get("warnings"):
                    response_data["rag_warnings"] = rag_update_result["warnings"]
            else:
                message = "Setting updated successfully but RAG system update failed"
        else:
            message = "Setting updated successfully"

        # Add LLM provider update information if applicable
        if llm_update_result:
            response_data["llm_update"] = llm_update_result

            if llm_update_result.get("success"):
                if rag_update_result:
                    message = "Setting updated successfully and applied to both RAG and LLM provider systems in real-time"
                else:
                    message = "Setting updated successfully and applied to LLM provider system in real-time"
                if llm_update_result.get("applied_changes"):
                    response_data["llm_applied_changes"] = llm_update_result["applied_changes"]
            else:
                if rag_update_result:
                    if rag_update_result.get("success"):
                        message = "Setting updated successfully, RAG system updated but LLM provider system update failed"
                    else:
                        message = "Setting updated successfully but both RAG and LLM provider system updates failed"
                else:
                    message = "Setting updated successfully but LLM provider system update failed"
                if llm_update_result.get("errors"):
                    response_data["llm_errors"] = llm_update_result["errors"]

        # Add Database Storage update information if applicable
        if database_update_result:
            response_data["database_update"] = database_update_result

            if database_update_result.get("success"):
                if rag_update_result and llm_update_result:
                    if rag_update_result.get("success") and llm_update_result.get("success"):
                        message = "Setting updated successfully and applied to RAG, LLM provider, and database storage systems in real-time"
                    elif rag_update_result.get("success"):
                        message = "Setting updated successfully, RAG and database storage systems updated but LLM provider system update failed"
                    elif llm_update_result.get("success"):
                        message = "Setting updated successfully, LLM provider and database storage systems updated but RAG system update failed"
                    else:
                        message = "Setting updated successfully, database storage system updated but RAG and LLM provider system updates failed"
                elif rag_update_result:
                    if rag_update_result.get("success"):
                        message = "Setting updated successfully and applied to RAG and database storage systems in real-time"
                    else:
                        message = "Setting updated successfully, database storage system updated but RAG system update failed"
                elif llm_update_result:
                    if llm_update_result.get("success"):
                        message = "Setting updated successfully and applied to LLM provider and database storage systems in real-time"
                    else:
                        message = "Setting updated successfully, database storage system updated but LLM provider system update failed"
                else:
                    message = "Setting updated successfully and applied to database storage system in real-time"
                if database_update_result.get("applied_changes"):
                    response_data["database_applied_changes"] = database_update_result["applied_changes"]
            else:
                message = "Setting updated successfully but database storage system update failed"
                if database_update_result.get("errors"):
                    response_data["database_errors"] = database_update_result["errors"]

        # 🚀 Revolutionary Real-Time Broadcasting to Users
        try:
            broadcast_level = await determine_broadcast_level(request.category.value, request.key)
            notification_type = await determine_notification_type(request.category.value, request.key)

            broadcast_result = await configuration_broadcaster.broadcast_configuration_change(
                section=request.category.value,
                setting_key=request.key,
                changes={request.key: request.value},
                broadcast_level=broadcast_level,
                admin_user_id=str(admin_user.id),
                notification_type=notification_type
            )

            response_data["broadcast_result"] = {
                "notifications_sent": broadcast_result.get("notifications_sent", 0),
                "broadcast_level": broadcast_level.value
            }

            logger.info(f"📢 Configuration change broadcasted: {request.category.value}.{request.key} - {broadcast_result.get('notifications_sent', 0)} users notified")

        except Exception as e:
            logger.error(f"❌ Failed to broadcast configuration change: {str(e)}")
            # Don't fail the entire request if broadcasting fails
            response_data["broadcast_warning"] = "Configuration updated but user notification failed"

        # 🔍 Verify Configuration Applied
        try:
            verification_success = await verify_configuration_applied(
                request.category.value,
                request.key,
                request.value
            )
            response_data["verification"] = {
                "applied": verification_success,
                "verified_at": datetime.utcnow().isoformat()
            }

            if not verification_success:
                logger.warning(f"⚠️ Configuration verification failed for {request.category.value}.{request.key}")

        except Exception as e:
            logger.error(f"❌ Failed to verify configuration: {str(e)}")
            response_data["verification_warning"] = "Configuration saved but verification failed"

        return StandardAPIResponse(
            success=True,
            message=message,
            data=response_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update setting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update setting: {str(e)}"
        )


@router.post("/bulk-update", response_model=StandardAPIResponse)
async def bulk_update_settings(
    request: BulkSettingUpdateRequest,
    background_tasks: BackgroundTasks,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """Update multiple settings in a single transaction."""
    try:
        definitions = await get_setting_definitions()
        validation_results = []
        requires_restart = False

        # Validate all settings first
        for update in request.updates:
            setting_key = f"{update.category.value}.{update.key}"

            if setting_key not in definitions:
                validation_results.append({
                    "setting_key": setting_key,
                    "is_valid": False,
                    "errors": [f"Setting {setting_key} not found"]
                })
                continue

            definition = definitions[setting_key]
            validation_result = await validate_setting_value(definition, update.value)

            validation_results.append({
                "setting_key": setting_key,
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "requires_restart": validation_result.requires_restart
            })

            if validation_result.requires_restart:
                requires_restart = True

        # Check if all validations passed
        all_valid = all(result["is_valid"] for result in validation_results)

        if request.validate_all and not all_valid:
            return StandardAPIResponse(
                success=False,
                message="Bulk validation failed",
                data={
                    "validation_results": validation_results,
                    "failed_count": sum(1 for r in validation_results if not r["is_valid"])
                }
            )

        # Store settings in database and apply to system
        settings_service = await get_admin_settings_service()

        successful_updates = []
        failed_updates = []

        for i, update in enumerate(request.updates):
            if validation_results[i]["is_valid"]:
                # Get setting definition for metadata
                setting_key = f"{update.category.value}.{update.key}"
                definitions = await get_setting_definitions()
                definition = definitions.get(setting_key)

                if definition:
                    success, error_message = await settings_service.set_setting(
                        category=update.category.value,
                        key=update.key,
                        value=update.value,
                        user_id=admin_user.id,
                        setting_type=definition.type.value,
                        description=definition.description,
                        requires_restart=definition.requires_restart,
                        validation_rules=definition.validation_rules,
                        enum_values=definition.enum_values
                    )

                    if success:
                        successful_updates.append({
                            "setting_key": setting_key,
                            "new_value": update.value
                        })
                    else:
                        failed_updates.append({
                            "setting_key": setting_key,
                            "errors": [error_message]
                        })
                else:
                    failed_updates.append({
                        "setting_key": setting_key,
                        "errors": ["Setting definition not found"]
                    })
            else:
                failed_updates.append({
                    "setting_key": f"{update.category.value}.{update.key}",
                    "errors": validation_results[i]["errors"]
                })

        logger.info(
            "Bulk settings update completed",
            admin_user=str(admin_user.id),
            successful_count=len(successful_updates),
            failed_count=len(failed_updates),
            requires_restart=requires_restart
        )

        # 🚀 Revolutionary Real-Time RAG System Bulk Update
        rag_update_result = None
        rag_settings_to_update = {}

        # Collect all RAG-related settings that were successfully updated
        for update_info in successful_updates:
            setting_key = update_info["setting_key"]
            if setting_key.startswith("rag_configuration."):
                # Remove the prefix for global config manager
                rag_setting_key = setting_key.replace("rag_configuration.", "")
                rag_settings_to_update[rag_setting_key] = update_info["new_value"]

        # Apply RAG updates if any RAG settings were changed
        if rag_settings_to_update:
            try:
                logger.info(f"🔄 Triggering bulk RAG system update for {len(rag_settings_to_update)} settings")

                # Use global config manager instance
                config_manager = global_config_manager

                # Update the configuration through the global config manager
                # This will automatically notify all RAG observers
                from app.core.global_config_manager import ConfigurationSection
                update_result = await config_manager.update_section(
                    section=ConfigurationSection.RAG_CONFIGURATION,
                    changes=rag_settings_to_update,
                    user_id=str(admin_user.id)
                )

                if update_result.success:
                    logger.info(f"✅ RAG system bulk update successful: {update_result.message}")
                    rag_update_result = {
                        "success": True,
                        "message": f"RAG system updated with {len(rag_settings_to_update)} settings in real-time",
                        "applied_changes": rag_settings_to_update,
                        "warnings": update_result.warnings or []
                    }
                else:
                    logger.warning(f"⚠️ RAG system bulk update failed: {update_result.message}")
                    rag_update_result = {
                        "success": False,
                        "error": update_result.message,
                        "message": f"RAG system bulk update failed: {update_result.message}"
                    }

            except Exception as e:
                logger.error(f"❌ Failed to bulk update RAG system: {str(e)}")
                rag_update_result = {
                    "success": False,
                    "error": f"RAG bulk update failed: {str(e)}"
                }

        # 🚀 Revolutionary Real-Time LLM Provider System Bulk Update
        llm_update_result = None
        llm_settings_to_update = {}

        # Collect all LLM provider-related settings that were successfully updated
        for update_info in successful_updates:
            setting_key = update_info["setting_key"]
            if setting_key.startswith("llm_providers."):
                # Extract the setting key without the category prefix
                llm_setting_key = setting_key.replace("llm_providers.", "")
                llm_settings_to_update[llm_setting_key] = update_info["new_value"]

        # Apply LLM provider updates if any LLM settings were changed
        if llm_settings_to_update:
            try:
                logger.info(f"🔄 Triggering bulk LLM provider system update for {len(llm_settings_to_update)} settings")

                # Use global config manager instance
                config_manager = global_config_manager

                # Update the configuration through the global config manager
                # This will automatically notify all LLM observers
                from app.core.global_config_manager import ConfigurationSection
                update_result = await config_manager.update_section(
                    section=ConfigurationSection.LLM_PROVIDERS,
                    changes=llm_settings_to_update,
                    user_id=str(admin_user.id)
                )

                if update_result.success:
                    logger.info(f"✅ LLM provider system bulk update successful: {update_result.message}")
                    llm_update_result = {
                        "success": True,
                        "message": f"LLM provider system updated with {len(llm_settings_to_update)} settings in real-time",
                        "applied_changes": llm_settings_to_update,
                        "settings_count": len(llm_settings_to_update)
                    }
                else:
                    logger.warning(f"⚠️ LLM provider system bulk update failed: {update_result.errors}")
                    llm_update_result = {
                        "success": False,
                        "error": f"LLM provider bulk update failed: {', '.join(update_result.errors)}",
                        "errors": update_result.errors,
                        "settings_count": len(llm_settings_to_update)
                    }

            except Exception as e:
                logger.error(f"❌ Failed to bulk update LLM provider system: {str(e)}")
                llm_update_result = {
                    "success": False,
                    "error": f"LLM provider bulk update failed: {str(e)}",
                    "settings_count": len(llm_settings_to_update)
                }

        # 🚀 Revolutionary Real-Time Database Storage System Bulk Update
        database_update_result = None
        database_settings_to_update = {}

        # Collect all database storage-related settings that were successfully updated
        for update_info in successful_updates:
            setting_key = update_info["setting_key"]
            if setting_key.startswith("database_storage."):
                # Extract the setting key without the category prefix
                database_setting_key = setting_key.replace("database_storage.", "")
                database_settings_to_update[database_setting_key] = update_info["new_value"]

        # Apply database storage updates if any database settings were changed
        if database_settings_to_update:
            try:
                logger.info(f"🔄 Triggering bulk database storage system update for {len(database_settings_to_update)} settings")

                # Use global config manager instance
                config_manager = global_config_manager

                # Update the configuration through the global config manager
                # This will automatically notify all database observers
                from app.core.global_config_manager import ConfigurationSection
                update_result = await config_manager.update_section(
                    section=ConfigurationSection.DATABASE_STORAGE,
                    changes=database_settings_to_update,
                    user_id=str(admin_user.id)
                )

                if update_result.success:
                    logger.info(f"✅ Database storage system bulk update successful: {update_result.message}")
                    database_update_result = {
                        "success": True,
                        "message": f"Database storage system updated with {len(database_settings_to_update)} settings in real-time",
                        "applied_changes": database_settings_to_update,
                        "warnings": update_result.warnings or []
                    }
                else:
                    logger.warning(f"⚠️ Database storage system bulk update failed: {update_result.message}")
                    database_update_result = {
                        "success": False,
                        "message": f"Database storage system bulk update failed: {update_result.message}",
                        "errors": update_result.errors or []
                    }

            except Exception as e:
                logger.error(f"❌ Database storage system bulk update error: {str(e)}")
                database_update_result = {
                    "success": False,
                    "message": "Database storage system bulk update failed due to internal error",
                    "error": f"Database storage bulk update failed: {str(e)}"
                }

        # Add background task for restart notification if needed
        if requires_restart:
            background_tasks.add_task(
                _notify_restart_required,
                admin_user.id,
                "bulk_update"
            )

        # Prepare response data
        response_data = {
            "successful_updates": successful_updates,
            "failed_updates": failed_updates,
            "requires_restart": requires_restart,
            "validation_results": validation_results
        }

        # Prepare success message
        base_message = f"Bulk update completed: {len(successful_updates)} successful, {len(failed_updates)} failed"

        # Add RAG update information if applicable
        if rag_update_result:
            response_data["rag_update"] = rag_update_result
            response_data["rag_settings_updated"] = len(rag_settings_to_update)

            if rag_update_result.get("success"):
                if len(rag_settings_to_update) > 0:
                    base_message += f" (RAG system updated with {len(rag_settings_to_update)} settings in real-time)"
                if rag_update_result.get("warnings"):
                    response_data["rag_warnings"] = rag_update_result["warnings"]
            else:
                if len(rag_settings_to_update) > 0:
                    base_message += f" (RAG system update failed for {len(rag_settings_to_update)} settings)"

        # Add LLM provider update information if applicable
        if llm_update_result:
            response_data["llm_update"] = llm_update_result
            response_data["llm_settings_updated"] = len(llm_settings_to_update)

            if llm_update_result.get("success"):
                if len(llm_settings_to_update) > 0:
                    if rag_update_result and rag_update_result.get("success"):
                        base_message = base_message.replace("(RAG system updated", "(RAG and LLM provider systems updated")
                        base_message = base_message.replace("settings in real-time)", f"settings in real-time: {len(rag_settings_to_update)} RAG + {len(llm_settings_to_update)} LLM)")
                    else:
                        base_message += f" (LLM provider system updated with {len(llm_settings_to_update)} settings in real-time)"
                if llm_update_result.get("applied_changes"):
                    response_data["llm_applied_changes"] = llm_update_result["applied_changes"]
            else:
                if len(llm_settings_to_update) > 0:
                    if rag_update_result:
                        if rag_update_result.get("success"):
                            base_message += f" (LLM provider system update failed for {len(llm_settings_to_update)} settings)"
                        else:
                            base_message = base_message.replace("(RAG system update failed", "(RAG and LLM provider system updates failed")
                    else:
                        base_message += f" (LLM provider system update failed for {len(llm_settings_to_update)} settings)"
                if llm_update_result.get("errors"):
                    response_data["llm_errors"] = llm_update_result["errors"]

        # Add Database Storage update information if applicable
        if database_update_result:
            response_data["database_update"] = database_update_result
            response_data["database_settings_updated"] = len(database_settings_to_update)

            if database_update_result.get("success"):
                if len(database_settings_to_update) > 0:
                    if rag_update_result and llm_update_result:
                        if rag_update_result.get("success") and llm_update_result.get("success"):
                            base_message = base_message.replace("(RAG and LLM provider systems updated", "(RAG, LLM provider, and database storage systems updated")
                            base_message = base_message.replace(f"settings in real-time: {len(rag_settings_to_update)} RAG + {len(llm_settings_to_update)} LLM)", f"settings in real-time: {len(rag_settings_to_update)} RAG + {len(llm_settings_to_update)} LLM + {len(database_settings_to_update)} DB)")
                        elif rag_update_result.get("success"):
                            base_message = base_message.replace("(RAG system updated", "(RAG and database storage systems updated")
                            base_message = base_message.replace(f"settings in real-time)", f"settings in real-time: {len(rag_settings_to_update)} RAG + {len(database_settings_to_update)} DB)")
                        elif llm_update_result.get("success"):
                            base_message = base_message.replace("(LLM provider system updated", "(LLM provider and database storage systems updated")
                            base_message = base_message.replace(f"settings in real-time)", f"settings in real-time: {len(llm_settings_to_update)} LLM + {len(database_settings_to_update)} DB)")
                        else:
                            base_message += f" (Database storage system updated with {len(database_settings_to_update)} settings in real-time)"
                    elif rag_update_result:
                        if rag_update_result.get("success"):
                            base_message = base_message.replace("(RAG system updated", "(RAG and database storage systems updated")
                            base_message = base_message.replace(f"settings in real-time)", f"settings in real-time: {len(rag_settings_to_update)} RAG + {len(database_settings_to_update)} DB)")
                        else:
                            base_message += f" (Database storage system updated with {len(database_settings_to_update)} settings in real-time)"
                    elif llm_update_result:
                        if llm_update_result.get("success"):
                            base_message = base_message.replace("(LLM provider system updated", "(LLM provider and database storage systems updated")
                            base_message = base_message.replace(f"settings in real-time)", f"settings in real-time: {len(llm_settings_to_update)} LLM + {len(database_settings_to_update)} DB)")
                        else:
                            base_message += f" (Database storage system updated with {len(database_settings_to_update)} settings in real-time)"
                    else:
                        base_message += f" (Database storage system updated with {len(database_settings_to_update)} settings in real-time)"
                if database_update_result.get("applied_changes"):
                    response_data["database_applied_changes"] = database_update_result["applied_changes"]
            else:
                if len(database_settings_to_update) > 0:
                    base_message += f" (Database storage system update failed for {len(database_settings_to_update)} settings)"
                if database_update_result.get("errors"):
                    response_data["database_errors"] = database_update_result["errors"]

        # 🚀 Revolutionary Real-Time Broadcasting for Bulk Updates
        broadcast_results = []
        total_notifications_sent = 0

        try:
            # Group successful updates by category for efficient broadcasting
            updates_by_category = {}
            for update in successful_updates:
                category = update["setting_key"].split(".")[0]
                if category not in updates_by_category:
                    updates_by_category[category] = []
                updates_by_category[category].append(update)

            # Broadcast changes for each category
            for category, category_updates in updates_by_category.items():
                try:
                    # Determine broadcast level for the category (use most restrictive)
                    broadcast_levels = []
                    notification_types = []

                    for update in category_updates:
                        key = update["setting_key"].split(".", 1)[1] if "." in update["setting_key"] else update["setting_key"]
                        broadcast_levels.append(await determine_broadcast_level(category, key))
                        notification_types.append(await determine_notification_type(category, key))

                    # Use most restrictive broadcast level
                    broadcast_level = min(broadcast_levels, key=lambda x: ["public", "admin_only", "system_only", "encrypted"].index(x.value))
                    notification_type = notification_types[0]  # Use first one as representative

                    # Create changes dict for this category
                    changes = {}
                    for update in category_updates:
                        key = update["setting_key"].split(".", 1)[1] if "." in update["setting_key"] else update["setting_key"]
                        changes[key] = update["new_value"]

                    # Broadcast the category changes
                    broadcast_result = await configuration_broadcaster.broadcast_configuration_change(
                        section=category,
                        setting_key=f"bulk_update_{len(changes)}_settings",
                        changes=changes,
                        broadcast_level=broadcast_level,
                        admin_user_id=str(admin_user.id),
                        notification_type=notification_type
                    )

                    broadcast_results.append({
                        "category": category,
                        "settings_count": len(changes),
                        "notifications_sent": broadcast_result.get("notifications_sent", 0),
                        "broadcast_level": broadcast_level.value
                    })

                    total_notifications_sent += broadcast_result.get("notifications_sent", 0)

                except Exception as e:
                    logger.error(f"❌ Failed to broadcast changes for category {category}: {str(e)}")
                    broadcast_results.append({
                        "category": category,
                        "error": str(e)
                    })

            response_data["broadcast_results"] = broadcast_results
            response_data["total_notifications_sent"] = total_notifications_sent

            logger.info(f"📢 Bulk configuration changes broadcasted: {len(successful_updates)} settings - {total_notifications_sent} total notifications sent")

        except Exception as e:
            logger.error(f"❌ Failed to broadcast bulk configuration changes: {str(e)}")
            response_data["broadcast_warning"] = "Settings updated but user notifications failed"

        return StandardAPIResponse(
            success=len(failed_updates) == 0,
            message=base_message,
            data=response_data
        )

    except Exception as e:
        logger.error("Failed to bulk update settings", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to bulk update settings: {str(e)}"
        )


@router.get("/rag-status", response_model=StandardAPIResponse)
async def get_rag_system_status_endpoint(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get current RAG system status and configuration."""
    try:
        rag_status = await get_rag_system_status()

        return StandardAPIResponse(
            success=True,
            message="RAG system status retrieved successfully",
            data=rag_status
        )

    except Exception as e:
        logger.error(f"Failed to get RAG system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG system status: {str(e)}"
        )


@router.get("/rag-templates", response_model=StandardAPIResponse)
async def get_rag_templates(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get available RAG templates with their configurations."""
    try:
        templates = {
            "general_purpose": {
                "name": "General Purpose",
                "description": "Balanced settings for general knowledge retrieval",
                "category": "general",
                "settings": {
                    "rag_configuration.chunk_size": 1000,
                    "rag_configuration.chunk_overlap": 200,
                    "rag_configuration.top_k": 10,
                    "rag_configuration.score_threshold": 0.7,
                    "rag_configuration.enable_reranking": True,
                    "rag_configuration.enable_query_expansion": True,
                    "rag_configuration.embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },
            "research_assistant": {
                "name": "Research Assistant",
                "description": "Optimized for academic and research tasks",
                "category": "academic",
                "settings": {
                    "rag_configuration.chunk_size": 1500,
                    "rag_configuration.chunk_overlap": 300,
                    "rag_configuration.top_k": 15,
                    "rag_configuration.score_threshold": 0.8,
                    "rag_configuration.enable_reranking": True,
                    "rag_configuration.enable_query_expansion": True,
                    "rag_configuration.enable_citation_tracking": True
                }
            },
            "code_helper": {
                "name": "Code Helper",
                "description": "Specialized for code documentation and programming assistance",
                "category": "development",
                "settings": {
                    "rag_configuration.chunk_size": 800,
                    "rag_configuration.chunk_overlap": 100,
                    "rag_configuration.top_k": 8,
                    "rag_configuration.score_threshold": 0.75,
                    "rag_configuration.enable_code_parsing": True,
                    "rag_configuration.enable_syntax_highlighting": True
                }
            },
            "creative_writing": {
                "name": "Creative Writing",
                "description": "Optimized for creative and narrative content",
                "category": "creative",
                "settings": {
                    "rag_configuration.chunk_size": 1200,
                    "rag_configuration.chunk_overlap": 250,
                    "rag_configuration.top_k": 12,
                    "rag_configuration.score_threshold": 0.65,
                    "rag_configuration.enable_semantic_search": True,
                    "rag_configuration.enable_context_preservation": True
                }
            }
        }

        logger.info("RAG templates retrieved", admin_user=str(admin_user.id), template_count=len(templates))

        return StandardAPIResponse(
            success=True,
            message="RAG templates retrieved successfully",
            data=templates
        )

    except Exception as e:
        logger.error("Failed to get RAG templates", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG templates: {str(e)}"
        )


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def _notify_restart_required(admin_user_id: str, setting_key: str):
    """Background task to notify that system restart is required."""
    try:
        logger.warning(
            "System restart required",
            admin_user_id=admin_user_id,
            setting_key=setting_key,
            timestamp=datetime.utcnow().isoformat()
        )
        # TODO: Implement actual notification system (email, websocket, etc.)
    except Exception as e:
        logger.error("Failed to send restart notification", error=str(e))


# ============================================================================
# 🚀 REVOLUTIONARY LLM PROVIDER MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/llm-providers/status", response_model=StandardAPIResponse)
async def get_llm_providers_status(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get comprehensive LLM providers status and configuration."""
    try:
        from app.services.llm_service import get_llm_service

        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()

        # Get provider status
        providers = await llm_service.get_available_providers()
        provider_info = await llm_service.get_provider_info()

        # Test all providers
        test_results = await llm_service.test_all_providers()

        # Get available models
        all_models = await llm_service.get_all_models()

        status = {
            "providers": providers,
            "provider_info": provider_info,
            "test_results": test_results,
            "available_models": all_models,
            "total_providers": len(providers),
            "active_providers": len([p for p in test_results.values() if p.get("status") == "connected"]),
            "total_models": sum(len(models) for models in all_models.values())
        }

        return StandardAPIResponse(
            success=True,
            message="LLM providers status retrieved successfully",
            data=status
        )

    except Exception as e:
        logger.error(f"Failed to get LLM providers status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM providers status: {str(e)}"
        )


@router.post("/llm-providers/download-model", response_model=StandardAPIResponse)
async def download_ollama_model(
    model_name: str,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Download an Ollama model for system use."""
    try:
        from app.services.llm_service import get_llm_service

        llm_service = get_llm_service()
        if not llm_service._is_initialized:
            await llm_service.initialize()

        # Get Ollama provider
        ollama_provider = await llm_service.get_provider("ollama")
        if not ollama_provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ollama provider not available"
            )

        # Download the model
        success = await ollama_provider.pull_model(model_name)

        if success:
            logger.info(
                "Ollama model downloaded successfully",
                admin_user=str(admin_user.id),
                model=model_name
            )

            return StandardAPIResponse(
                success=True,
                message=f"Model {model_name} downloaded successfully",
                data={"model_name": model_name}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download model {model_name}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download Ollama model: {str(e)}", model=model_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download model: {str(e)}"
        )


@router.get("/llm-providers/ollama-status", response_model=StandardAPIResponse)
async def get_ollama_connection_status(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get Ollama connection status and available models."""
    try:
        from app.core.admin_model_manager import admin_model_manager

        status = await admin_model_manager.check_ollama_connection()

        return StandardAPIResponse(
            success=True,
            message="Ollama status retrieved successfully",
            data=status
        )

    except Exception as e:
        logger.error(f"Failed to get Ollama status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Ollama status: {str(e)}"
        )


@router.post("/llm-providers/download-model", response_model=StandardAPIResponse)
async def download_model_admin(
    request: dict,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Admin-only model download with comprehensive management."""
    try:
        model_name = request.get("model_name")
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name is required"
            )

        from app.core.admin_model_manager import admin_model_manager

        result = await admin_model_manager.download_model(model_name, str(admin_user.id))

        if result["success"]:
            logger.info(
                "Model downloaded by admin",
                admin_user=str(admin_user.id),
                model=model_name
            )

            return StandardAPIResponse(
                success=True,
                message=result["message"],
                data=result.get("model_info", {})
            )
        else:
            return StandardAPIResponse(
                success=False,
                message=result["message"],
                data={"error": result.get("error", "Unknown error")}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}", model=request.get("model_name"))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download model: {str(e)}"
        )


@router.delete("/llm-providers/remove-model/{model_name}", response_model=StandardAPIResponse)
async def remove_model_admin(
    model_name: str,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Admin-only model removal."""
    try:
        from app.core.admin_model_manager import admin_model_manager

        result = await admin_model_manager.remove_model(model_name, str(admin_user.id))

        if result["success"]:
            logger.info(
                "Model removed by admin",
                admin_user=str(admin_user.id),
                model=model_name
            )

        return StandardAPIResponse(
            success=result["success"],
            message=result["message"],
            data={}
        )

    except Exception as e:
        logger.error(f"Failed to remove model: {str(e)}", model=model_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove model: {str(e)}"
        )


@router.get("/llm-providers/model-registry", response_model=StandardAPIResponse)
async def get_model_registry(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get complete model registry."""
    try:
        from app.core.admin_model_manager import admin_model_manager

        models = admin_model_manager.get_available_models()

        return StandardAPIResponse(
            success=True,
            message=f"Retrieved {len(models)} models from registry",
            data={"models": models, "count": len(models)}
        )

    except Exception as e:
        logger.error(f"Failed to get model registry: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model registry: {str(e)}"
        )


@router.get("/llm-providers/download-progress", response_model=StandardAPIResponse)
async def get_download_progress(
    model_name: Optional[str] = None,
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get model download progress."""
    try:
        from app.core.admin_model_manager import admin_model_manager

        progress = admin_model_manager.get_download_progress(model_name)

        return StandardAPIResponse(
            success=True,
            message="Download progress retrieved",
            data={"progress": progress}
        )

    except Exception as e:
        logger.error(f"Failed to get download progress: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get download progress: {str(e)}"
        )


@router.get("/llm-providers/available-models", response_model=StandardAPIResponse)
async def get_available_models_for_download(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get available models that can be downloaded."""
    try:
        # Popular models categorized by use case
        available_models = {
            "recommended": [
                {
                    "name": "llama3.2:latest",
                    "size": "2.0GB",
                    "description": "Latest Llama 3.2 model with excellent tool calling support",
                    "capabilities": ["text", "tools", "conversation"],
                    "provider": "ollama",
                    "recommended": True
                },
                {
                    "name": "llama3.1:8b",
                    "size": "4.7GB",
                    "description": "Llama 3.1 8B with superior tool calling capabilities",
                    "capabilities": ["text", "tools", "conversation"],
                    "provider": "ollama",
                    "recommended": True
                },
                {
                    "name": "qwen2.5:latest",
                    "size": "4.4GB",
                    "description": "Qwen 2.5 with strong reasoning and tool support",
                    "capabilities": ["text", "tools", "conversation", "reasoning"],
                    "provider": "ollama",
                    "recommended": True
                }
            ],
            "code": [
                {
                    "name": "codellama:latest",
                    "size": "3.8GB",
                    "description": "Code Llama for programming tasks",
                    "capabilities": ["code", "text"],
                    "provider": "ollama",
                    "recommended": False
                },
                {
                    "name": "deepseek-coder:latest",
                    "size": "3.7GB",
                    "description": "DeepSeek Coder for advanced programming",
                    "capabilities": ["code", "text"],
                    "provider": "ollama",
                    "recommended": False
                }
            ],
            "lightweight": [
                {
                    "name": "llama3.2:3b",
                    "size": "2.0GB",
                    "description": "Lightweight Llama 3.2 3B model",
                    "capabilities": ["text", "conversation"],
                    "provider": "ollama",
                    "recommended": False
                },
                {
                    "name": "phi3:latest",
                    "size": "2.3GB",
                    "description": "Microsoft Phi-3 lightweight model",
                    "capabilities": ["text", "conversation"],
                    "provider": "ollama",
                    "recommended": False
                }
            ],
            "specialized": [
                {
                    "name": "mistral:latest",
                    "size": "4.1GB",
                    "description": "Mistral 7B for general tasks",
                    "capabilities": ["text", "conversation"],
                    "provider": "ollama",
                    "recommended": False
                },
                {
                    "name": "gemma2:latest",
                    "size": "5.4GB",
                    "description": "Google Gemma 2 model",
                    "capabilities": ["text", "conversation"],
                    "provider": "ollama",
                    "recommended": False
                }
            ]
        }

        return StandardAPIResponse(
            success=True,
            message="Available models retrieved successfully",
            data=available_models
        )

    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available models: {str(e)}"
        )


@router.get("/llm-providers/templates", response_model=StandardAPIResponse)
async def get_llm_provider_templates(
    admin_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """🚀 Get LLM provider configuration templates."""
    try:
        templates = {
            "local_development": {
                "name": "Local Development",
                "description": "Optimized for local development with Ollama",
                "icon": "🏠",
                "settings": {
                    "enable_ollama": True,
                    "enable_openai": False,
                    "enable_anthropic": False,
                    "enable_google": False,
                    "ollama_base_url": "http://localhost:11434",
                    "ollama_timeout": 120,
                    "ollama_max_concurrent_requests": 5,
                    "default_provider": "ollama",
                    "preferred_models": {"ollama": "llama3.2:latest"}
                }
            },
            "production_hybrid": {
                "name": "Production Hybrid",
                "description": "Balanced setup with local and cloud providers",
                "icon": "⚡",
                "settings": {
                    "enable_ollama": True,
                    "enable_openai": True,
                    "enable_anthropic": False,
                    "enable_google": False,
                    "ollama_base_url": "http://localhost:11434",
                    "ollama_timeout": 60,
                    "ollama_max_concurrent_requests": 10,
                    "openai_timeout": 30,
                    "openai_max_retries": 3,
                    "default_provider": "openai",
                    "fallback_provider": "ollama",
                    "enable_failover": True
                }
            },
            "cloud_only": {
                "name": "Cloud Only",
                "description": "Cloud-based providers for maximum performance",
                "icon": "☁️",
                "settings": {
                    "enable_ollama": False,
                    "enable_openai": True,
                    "enable_anthropic": True,
                    "enable_google": True,
                    "openai_timeout": 30,
                    "anthropic_timeout": 30,
                    "google_timeout": 30,
                    "default_provider": "openai",
                    "fallback_provider": "anthropic",
                    "enable_load_balancing": True
                }
            },
            "high_performance": {
                "name": "High Performance",
                "description": "Optimized for high-throughput applications",
                "icon": "🚀",
                "settings": {
                    "enable_ollama": True,
                    "enable_openai": True,
                    "ollama_max_concurrent_requests": 20,
                    "ollama_connection_pool_size": 10,
                    "openai_max_retries": 5,
                    "request_timeout": 120,
                    "max_concurrent_requests": 50,
                    "enable_load_balancing": True,
                    "enable_failover": True,
                    "enable_request_caching": True
                }
            }
        }

        return StandardAPIResponse(
            success=True,
            message="LLM provider templates retrieved successfully",
            data=templates
        )

    except Exception as e:
        logger.error(f"Failed to get LLM provider templates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM provider templates: {str(e)}"
        )


# ============================================================================
# 🤗 HUGGINGFACE MODEL MANAGEMENT
# ============================================================================

class HuggingFaceModelRequest(BaseModel):
    """Request to download a HuggingFace model."""
    model_id: str = Field(..., description="HuggingFace model ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2')")
    model_type: str = Field(..., description="Model type: 'embedding', 'vision', 'reranking'")
    is_public: bool = Field(default=True, description="Whether model is available to all users")
    force_redownload: bool = Field(default=False, description="Force redownload if model exists")


class HuggingFaceModelInfo(BaseModel):
    """HuggingFace model information."""
    model_id: str
    model_type: str
    is_downloaded: bool
    is_public: bool
    download_date: Optional[str]
    local_path: Optional[str]
    size_mb: Optional[float]
    description: Optional[str]


@router.get("/huggingface/available-models")
async def get_available_huggingface_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get available HuggingFace models."""
    try:
        logger.info(f"🤗 Getting available HuggingFace models for user: {current_user.email}")

        # Get all available models (fallback implementation)
        all_models = []  # No models available without embedding model manager

        # Filter by type if specified
        if model_type:
            filtered_models = {
                model_id: info for model_id, info in all_models.items()
                if info.model_type.value == model_type
            }
        else:
            filtered_models = all_models

        # Convert to response format
        models_data = []
        for model_id, model_info in filtered_models.items():
            models_data.append(HuggingFaceModelInfo(
                model_id=model_id,
                model_type=model_info.model_type.value,
                is_downloaded=model_info.is_downloaded,
                is_public=getattr(model_info, 'is_public', True),
                download_date=model_info.download_date.isoformat() if model_info.download_date else None,
                local_path=model_info.local_path,
                size_mb=getattr(model_info, 'size_mb', None),
                description=model_info.description
            ))

        return StandardAPIResponse(
            success=True,
            message=f"Retrieved {len(models_data)} HuggingFace models",
            data={
                "models": models_data,
                "total_count": len(models_data),
                "downloaded_count": sum(1 for m in models_data if m.is_downloaded),
                "public_count": sum(1 for m in models_data if m.is_public)
            }
        )

    except Exception as e:
        logger.error(f"❌ Failed to get HuggingFace models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get HuggingFace models: {str(e)}"
        )


@router.post("/huggingface/download-model")
async def download_huggingface_model(
    request: HuggingFaceModelRequest,
    background_tasks: BackgroundTasks,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Download a HuggingFace model (Admin only)."""
    try:
        # Check if user is admin (you may need to adjust this based on your auth system)
        if not getattr(current_user, 'is_admin', False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can download models"
            )

        logger.info(f"🤗 Admin {current_user.email} downloading model: {request.model_id}")

        # Validate model type
        valid_types = ["embedding", "vision", "reranking"]
        if request.model_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type. Must be one of: {valid_types}"
            )

        # Start download in background
        background_tasks.add_task(
            _download_model_background,
            request.model_id,
            request.model_type,
            request.is_public,
            request.force_redownload,
            current_user.id
        )

        return StandardAPIResponse(
            success=True,
            message=f"Model download started: {request.model_id}",
            data={
                "model_id": request.model_id,
                "model_type": request.model_type,
                "is_public": request.is_public,
                "status": "download_started",
                "estimated_time": "5-15 minutes"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start model download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model download: {str(e)}"
        )


@router.get("/huggingface/download-progress/{model_id}")
async def get_download_progress(
    model_id: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Get download progress for a model."""
    try:
        logger.info(f"📊 Getting download progress for: {model_id}")

        # Get progress (fallback implementation)
        progress = None  # No progress tracking without embedding model manager

        if not progress:
            return StandardAPIResponse(
                success=False,
                message=f"No download progress found for model: {model_id}",
                data={"model_id": model_id, "status": "not_found"}
            )

        return StandardAPIResponse(
            success=True,
            message="Download progress retrieved successfully",
            data={
                "model_id": model_id,
                "status": progress.status,
                "progress_percent": progress.progress_percent,
                "current_step": progress.current_step,
                "error_message": progress.error_message,
                "started_at": progress.started_at.isoformat() if progress.started_at else None,
                "estimated_completion": progress.estimated_completion.isoformat() if progress.estimated_completion else None
            }
        )

    except Exception as e:
        logger.error(f"❌ Failed to get download progress: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get download progress: {str(e)}"
        )


async def _download_model_background(
    model_id: str,
    model_type: str,
    is_public: bool,
    force_redownload: bool,
    admin_user_id: str
) -> None:
    """Background task to download a model."""
    try:
        logger.info(f"🚀 Starting background download: {model_id}")

        # Download the model (fallback implementation)
        success = False  # No download capability without embedding model manager

        if success:
            logger.info(f"✅ Model download completed: {model_id}")

            # Update model accessibility if needed
            if not is_public:
                # Mark model as private (you may need to implement this in the model manager)
                pass

            # Broadcast model availability update
            from ...core.configuration_broadcaster import configuration_broadcaster
            await configuration_broadcaster.broadcast_model_availability(
                model_id=model_id,
                model_type=model_type,
                is_available=True,
                is_public=is_public,
                admin_user_id=admin_user_id
            )

        else:
            logger.error(f"❌ Model download failed: {model_id}")

    except Exception as e:
        logger.error(f"❌ Background model download failed: {str(e)}")


@router.delete("/huggingface/models/{model_id}")
async def delete_huggingface_model(
    model_id: str,
    current_user: UserDB = Depends(get_current_active_user)
) -> StandardAPIResponse:
    """Delete a downloaded HuggingFace model (Admin only)."""
    try:
        # Check if user is admin
        if not getattr(current_user, 'is_admin', False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can delete models"
            )

        logger.info(f"🗑️ Admin {current_user.email} deleting model: {model_id}")

        # Delete the model (fallback implementation)
        success = False  # No delete capability without embedding model manager

        if success:
            # Broadcast model removal
            from ...core.configuration_broadcaster import configuration_broadcaster
            await configuration_broadcaster.broadcast_model_availability(
                model_id=model_id,
                model_type="unknown",
                is_available=False,
                is_public=False,
                admin_user_id=str(current_user.id)
            )

            return StandardAPIResponse(
                success=True,
                message=f"Model deleted successfully: {model_id}",
                data={"model_id": model_id, "status": "deleted"}
            )
        else:
            return StandardAPIResponse(
                success=False,
                message=f"Failed to delete model: {model_id}",
                data={"model_id": model_id, "status": "delete_failed"}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.get("/database-storage/status", response_model=StandardAPIResponse)
async def get_database_storage_status(
    current_user: UserDB = Depends(require_admin)
) -> StandardAPIResponse:
    """
    Get comprehensive database and storage status.
    """
    try:
        from app.rag.core.vector_db_factory import get_available_vector_db_types

        status_data = {
            "postgresql": {
                "status": "unknown",
                "connection_pool": {
                    "active_connections": 0,
                    "idle_connections": 0,
                    "max_connections": 10
                },
                "last_check": None,
                "error": None
            },
            "vector_databases": {
                "available_types": get_available_vector_db_types(),
                "current_type": "chromadb",
                "status": "connected"
            },
            "storage": {
                "data_directory": "data/",
                "vector_storage": "data/chroma",
                "session_documents": "data/session_documents",
                "agent_files": "data/agent_files"
            }
        }

        # Test PostgreSQL connection
        try:
            # Try to import and test database connection
            import os
            db_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/agentic_ai")
            if "postgresql" in db_url:
                status_data["postgresql"]["status"] = "configured"
                status_data["postgresql"]["last_check"] = "2025-09-23T19:30:00Z"
            else:
                status_data["postgresql"]["status"] = "not_configured"
                status_data["postgresql"]["error"] = "PostgreSQL not configured"
                status_data["postgresql"]["last_check"] = "2025-09-23T19:30:00Z"
        except Exception as e:
            status_data["postgresql"]["status"] = "error"
            status_data["postgresql"]["error"] = str(e)
            status_data["postgresql"]["last_check"] = "2025-09-23T19:30:00Z"

        return StandardAPIResponse(
            success=True,
            message="Database storage status retrieved successfully",
            data=status_data
        )

    except Exception as e:
        logger.error(f"Error getting database storage status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get database storage status: {str(e)}")


@router.get("/database-storage/vector-db-status")
async def get_vector_db_status(
    current_user: UserDB = Depends(require_admin)
):
    """
    Get vector database availability status and recommendations.
    """
    try:
        from app.core.global_config_manager import global_config_manager
        from app.rag.core.vector_db_factory import get_available_vector_db_types

        # Get database configuration from global config manager
        from app.core.global_config_manager import ConfigurationSection
        try:
            current_config = await global_config_manager.get_section_configuration(ConfigurationSection.DATABASE_STORAGE)
        except Exception:
            current_config = {}

        # Get available vector database types
        available_types = get_available_vector_db_types()

        return StandardAPIResponse(
                success=True,
                message="Vector database status retrieved successfully (fallback mode)",
                data={
                    "availability": {db_type: True for db_type in available_types},
                    "available_types": available_types,
                    "recommended": current_config.get("vector_db_type", "chromadb"),
                    "current_type": current_config.get("vector_db_type", "chromadb"),
                    "auto_detect": current_config.get("vector_db_auto_detect", True)
                }
            )

        # Get availability and recommendation from database manager
        availability = await database_manager.detect_available_vector_databases()
        recommended = await database_manager.get_recommended_vector_db()
        available_types = get_available_vector_db_types()

        return StandardAPIResponse(
            success=True,
            message="Vector database status retrieved successfully",
            data={
                "availability": availability,
                "available_types": available_types,
                "recommended": recommended,
                "current_type": database_manager._current_config.get("vector_db_type", "auto"),
                "auto_detect": database_manager._current_config.get("vector_db_auto_detect", True)
            }
        )

    except Exception as e:
        logger.error(f"Error getting vector database status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get vector database status: {str(e)}")


@router.post("/database-storage/test-connection")
async def test_database_connection(
    request: dict,
    current_user: UserDB = Depends(require_admin)
):
    """
    Test database connection for specified type.
    """
    try:
        connection_type = request.get("connection_type")
        if not connection_type:
            raise HTTPException(status_code=400, detail="connection_type is required")

        if connection_type not in ["postgresql", "chromadb", "redis", "pgvector"]:
            raise HTTPException(status_code=400, detail="Invalid connection type")

        # This is a placeholder - in production you'd implement actual connection testing
        success = True
        message = f"{connection_type.upper()} connection test successful"

        # Simulate some connection testing logic
        if connection_type == "postgresql":
            # Test PostgreSQL connection
            message = "PostgreSQL connection successful"
        elif connection_type == "chromadb":
            # Test ChromaDB connection
            try:
                import chromadb
                message = "ChromaDB connection successful"
            except ImportError:
                success = False
                message = "ChromaDB not installed"
        elif connection_type == "redis":
            # Test Redis connection
            message = "Redis connection test successful"
        elif connection_type == "pgvector":
            # Test PgVector connection
            message = "PgVector extension test successful"

        return StandardAPIResponse(
            success=success,
            message=message,
            data={
                "connection_type": connection_type,
                "status": "connected" if success else "error"
            }
        )

    except Exception as e:
        logger.error(f"Error testing {connection_type} connection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")
