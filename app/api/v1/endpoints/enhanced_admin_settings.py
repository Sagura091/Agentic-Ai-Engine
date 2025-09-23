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
    """LLM provider management settings - admin controls for system-wide provider configuration."""
    # Provider Enablement (Admin controls which providers are available system-wide)
    enable_ollama: bool = Field(default=True)
    enable_openai: bool = Field(default=False)
    enable_anthropic: bool = Field(default=False)
    enable_google: bool = Field(default=False)

    # Ollama Configuration (Admin-managed local server)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_timeout: int = Field(default=60)
    ollama_max_concurrent_requests: int = Field(default=10)

    # Provider Connection Settings (Admin-managed infrastructure)
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_timeout: int = Field(default=60)
    openai_max_requests_per_minute: int = Field(default=60)

    anthropic_base_url: str = Field(default="https://api.anthropic.com")
    anthropic_timeout: int = Field(default=60)
    anthropic_max_requests_per_minute: int = Field(default=60)

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
    definitions.update({
        f"{llm_provider_prefix}.enable_ollama": SettingDefinition(
            key="enable_ollama",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=settings.ENABLE_OLLAMA,
            description="Enable Ollama local LLM provider"
        ),
        f"{llm_provider_prefix}.enable_openai": SettingDefinition(
            key="enable_openai",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=settings.ENABLE_OPENAI,
            description="Enable OpenAI API provider"
        ),
        f"{llm_provider_prefix}.enable_anthropic": SettingDefinition(
            key="enable_anthropic",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=settings.ENABLE_ANTHROPIC,
            description="Enable Anthropic Claude API provider"
        ),
        f"{llm_provider_prefix}.enable_google": SettingDefinition(
            key="enable_google",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=False,
            current_value=settings.ENABLE_GOOGLE,
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
            current_value=0.7,
            description="Default temperature for LLM responses",
            min_value=0.0,
            max_value=2.0
        ),
        f"{llm_provider_prefix}.max_tokens": SettingDefinition(
            key="max_tokens",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=4096,
            current_value=4096,
            description="Maximum tokens per LLM response",
            min_value=100,
            max_value=100000
        ),
        f"{llm_provider_prefix}.request_timeout": SettingDefinition(
            key="request_timeout",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=30,
            current_value=30,
            description="Request timeout in seconds",
            min_value=5,
            max_value=300
        ),
        f"{llm_provider_prefix}.enable_rate_limiting": SettingDefinition(
            key="enable_rate_limiting",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable system-wide rate limiting for API requests"
        ),
        f"{llm_provider_prefix}.global_rate_limit_per_minute": SettingDefinition(
            key="global_rate_limit_per_minute",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=1000,
            current_value=1000,
            description="Global rate limit for all LLM requests per minute",
            min_value=10,
            max_value=10000
        ),
        f"{llm_provider_prefix}.enable_request_caching": SettingDefinition(
            key="enable_request_caching",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.BOOLEAN,
            default_value=True,
            current_value=True,
            description="Enable caching of LLM responses for performance"
        ),
        f"{llm_provider_prefix}.cache_ttl_seconds": SettingDefinition(
            key="cache_ttl_seconds",
            category=SettingCategory.LLM_PROVIDERS,
            type=SettingType.INTEGER,
            default_value=300,
            current_value=300,
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
            settings_data[definition.key] = {
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
        setting_key = f"{request.category}.{request.key}"

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
        setting_key = f"{request.category}.{request.key}"

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

        # 🚀 Revolutionary Real-Time RAG System Update
        rag_update_result = None
        if setting_key.startswith("rag_configuration."):
            try:
                logger.info(f"🔄 Triggering real-time RAG system update for: {setting_key}")
                rag_update_result = await update_rag_settings({setting_key: request.value})

                if rag_update_result.get("success"):
                    logger.info(f"✅ RAG system updated successfully: {rag_update_result.get('message')}")
                else:
                    logger.warning(f"⚠️ RAG system update failed: {rag_update_result.get('error')}")

            except Exception as e:
                logger.error(f"❌ Failed to update RAG system: {str(e)}")
                rag_update_result = {
                    "success": False,
                    "error": f"RAG update failed: {str(e)}"
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
            setting_key = f"{update.category}.{update.key}"

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
                rag_settings_to_update[setting_key] = update_info["new_value"]

        # Apply RAG updates if any RAG settings were changed
        if rag_settings_to_update:
            try:
                logger.info(f"🔄 Triggering bulk RAG system update for {len(rag_settings_to_update)} settings")
                rag_update_result = await update_rag_settings(rag_settings_to_update)

                if rag_update_result.get("success"):
                    logger.info(f"✅ RAG system bulk update successful: {rag_update_result.get('message')}")
                else:
                    logger.warning(f"⚠️ RAG system bulk update failed: {rag_update_result.get('error')}")

            except Exception as e:
                logger.error(f"❌ Failed to bulk update RAG system: {str(e)}")
                rag_update_result = {
                    "success": False,
                    "error": f"RAG bulk update failed: {str(e)}"
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
