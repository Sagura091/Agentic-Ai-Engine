"""
🔥 REVOLUTIONARY SESSION DOCUMENT CONFIGURATION
===============================================

Configuration settings for the Revolutionary Session-Based Document Workspace.
Provides centralized configuration management for all session document features.

CONFIGURATION AREAS:
- Storage settings and paths
- Document size and count limits
- Expiration and cleanup policies
- Vector store configuration
- Integration settings
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import timedelta

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SessionDocumentStorageConfig:
    """Configuration for session document storage."""
    
    # Base storage directory
    base_storage_dir: Path = field(default_factory=lambda: Path("./data/session_documents"))
    
    # Temporary file directory
    temp_dir: Path = field(default_factory=lambda: Path("./data/temp/session_docs"))
    
    # Download directory for processed files
    download_dir: Path = field(default_factory=lambda: Path("./data/downloads/session_docs"))
    
    # Storage organization
    organize_by_date: bool = True
    organize_by_session: bool = True
    
    # File permissions
    file_permissions: int = 0o644
    dir_permissions: int = 0o755
    
    # Storage limits
    max_storage_size: int = 10 * 1024 * 1024 * 1024  # 10GB total
    cleanup_threshold: float = 0.8  # Cleanup when 80% full
    
    def __post_init__(self):
        """Ensure directories exist."""
        for directory in [self.base_storage_dir, self.temp_dir, self.download_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Storage directory ensured: {directory}")


@dataclass
class SessionDocumentLimitsConfig:
    """Configuration for document limits and constraints."""
    
    # Document limits per session
    max_documents_per_session: int = 100
    max_total_size_per_session: int = 1024 * 1024 * 1024  # 1GB per session
    
    # Individual document limits
    max_document_size: int = 100 * 1024 * 1024  # 100MB per document
    min_document_size: int = 1  # 1 byte minimum
    
    # Filename constraints
    max_filename_length: int = 255
    allowed_extensions: Optional[set] = None  # None = allow all
    blocked_extensions: set = field(default_factory=lambda: {'.exe', '.bat', '.cmd', '.scr'})
    
    # Content type constraints
    allowed_content_types: Optional[set] = None  # None = allow all
    blocked_content_types: set = field(default_factory=lambda: {
        'application/x-executable',
        'application/x-msdownload',
        'application/x-msdos-program'
    })
    
    def __post_init__(self):
        """Set default allowed extensions if not specified."""
        if self.allowed_extensions is None:
            self.allowed_extensions = {
                # Documents
                '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt',
                # Spreadsheets
                '.xls', '.xlsx', '.csv', '.ods',
                # Presentations
                '.ppt', '.pptx', '.odp',
                # Images
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg',
                # Archives
                '.zip', '.rar', '.7z', '.tar', '.gz',
                # Data formats
                '.json', '.xml', '.yaml', '.yml',
                # Web formats
                '.html', '.htm', '.css', '.js',
                # Other
                '.md', '.log'
            }


@dataclass
class SessionDocumentExpirationConfig:
    """Configuration for document expiration and cleanup."""
    
    # Default expiration times
    default_document_expiration: timedelta = field(default_factory=lambda: timedelta(hours=24))
    default_workspace_expiration: timedelta = field(default_factory=lambda: timedelta(hours=48))
    
    # Extended expiration for specific types
    extended_expiration_types: Dict[str, timedelta] = field(default_factory=lambda: {
        'generated': timedelta(hours=72),  # Generated documents last longer
        'modified': timedelta(hours=48),   # Modified documents last longer
    })
    
    # Cleanup configuration
    cleanup_interval: timedelta = field(default_factory=lambda: timedelta(hours=1))
    cleanup_batch_size: int = 100
    cleanup_enabled: bool = True
    
    # Grace period before actual deletion
    deletion_grace_period: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    
    # Notification settings
    expiration_warning_threshold: timedelta = field(default_factory=lambda: timedelta(hours=2))
    send_expiration_warnings: bool = True


@dataclass
class SessionDocumentVectorConfig:
    """Configuration for session document vector storage."""
    
    # Vector store settings
    enable_vector_search: bool = True
    vector_store_type: str = "chroma"  # chroma, faiss, or memory
    
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Search configuration
    default_search_k: int = 5
    max_search_k: int = 20
    similarity_threshold: float = 0.7
    
    # Performance settings
    batch_size: int = 32
    max_concurrent_embeddings: int = 4
    
    # Storage settings
    vector_storage_dir: Path = field(default_factory=lambda: Path("./data/session_vectors"))
    
    def __post_init__(self):
        """Ensure vector storage directory exists."""
        self.vector_storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🔍 Vector storage directory ensured: {self.vector_storage_dir}")


@dataclass
class SessionDocumentIntegrationConfig:
    """Configuration for integration with other systems."""
    
    # Revolutionary Document Intelligence Tool integration
    enable_document_intelligence: bool = True
    intelligence_tool_timeout: int = 300  # 5 minutes
    
    # RAG system integration
    enable_rag_integration: bool = True
    rag_collection_prefix: str = "session_"
    
    # Memory system integration
    enable_memory_integration: bool = True
    memory_retention_hours: int = 24
    
    # API integration settings
    api_rate_limit: int = 100  # requests per minute
    api_timeout: int = 30  # seconds
    
    # Background processing
    enable_background_processing: bool = True
    max_concurrent_jobs: int = 5
    job_timeout: int = 600  # 10 minutes
    
    # Notification settings
    enable_notifications: bool = True
    notification_channels: list = field(default_factory=lambda: ['websocket', 'log'])


@dataclass
class SessionDocumentSecurityConfig:
    """Configuration for security and access control."""
    
    # Access control
    require_authentication: bool = True
    session_validation: bool = True
    
    # Content scanning
    enable_content_scanning: bool = True
    scan_for_malware: bool = False  # Requires additional setup
    scan_for_sensitive_data: bool = True
    
    # Encryption settings
    encrypt_at_rest: bool = False  # Requires encryption key setup
    encryption_algorithm: str = "AES-256-GCM"
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    
    # Download security
    secure_download_links: bool = True
    download_link_expiration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    download_link_single_use: bool = True


class SessionDocumentConfig:
    """Main configuration class for session document system."""
    
    def __init__(self):
        """Initialize configuration from environment and defaults."""
        
        # Load configuration sections
        self.storage = SessionDocumentStorageConfig()
        self.limits = SessionDocumentLimitsConfig()
        self.expiration = SessionDocumentExpirationConfig()
        self.vector = SessionDocumentVectorConfig()
        self.integration = SessionDocumentIntegrationConfig()
        self.security = SessionDocumentSecurityConfig()
        
        # Apply environment overrides
        self._apply_environment_overrides()
        
        logger.info("🔥 Revolutionary Session Document Configuration initialized")
    
    def _apply_environment_overrides(self):
        """Apply configuration overrides from environment variables."""
        
        # Storage overrides
        if base_dir := os.getenv("SESSION_DOCS_BASE_DIR"):
            self.storage.base_storage_dir = Path(base_dir)
        
        if temp_dir := os.getenv("SESSION_DOCS_TEMP_DIR"):
            self.storage.temp_dir = Path(temp_dir)
        
        # Limits overrides
        if max_docs := os.getenv("SESSION_DOCS_MAX_PER_SESSION"):
            self.limits.max_documents_per_session = int(max_docs)
        
        if max_size := os.getenv("SESSION_DOCS_MAX_SIZE"):
            self.limits.max_document_size = int(max_size)
        
        # Expiration overrides
        if doc_expiry := os.getenv("SESSION_DOCS_EXPIRY_HOURS"):
            self.expiration.default_document_expiration = timedelta(hours=int(doc_expiry))
        
        if workspace_expiry := os.getenv("SESSION_WORKSPACE_EXPIRY_HOURS"):
            self.expiration.default_workspace_expiration = timedelta(hours=int(workspace_expiry))
        
        # Vector store overrides
        if vector_enabled := os.getenv("SESSION_DOCS_VECTOR_ENABLED"):
            self.vector.enable_vector_search = vector_enabled.lower() == "true"
        
        if embedding_model := os.getenv("SESSION_DOCS_EMBEDDING_MODEL"):
            self.vector.embedding_model = embedding_model
        
        # Integration overrides
        if intelligence_enabled := os.getenv("SESSION_DOCS_INTELLIGENCE_ENABLED"):
            self.integration.enable_document_intelligence = intelligence_enabled.lower() == "true"
        
        # Security overrides
        if auth_required := os.getenv("SESSION_DOCS_AUTH_REQUIRED"):
            self.security.require_authentication = auth_required.lower() == "true"
        
        if content_scan := os.getenv("SESSION_DOCS_CONTENT_SCAN"):
            self.security.enable_content_scanning = content_scan.lower() == "true"
        
        logger.info("🔧 Environment configuration overrides applied")
    
    def validate_configuration(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate storage paths
            if not self.storage.base_storage_dir.exists():
                logger.warning(f"Base storage directory does not exist: {self.storage.base_storage_dir}")
            
            # Validate limits
            if self.limits.max_document_size <= 0:
                raise ValueError("Max document size must be positive")
            
            if self.limits.max_documents_per_session <= 0:
                raise ValueError("Max documents per session must be positive")
            
            # Validate expiration settings
            if self.expiration.default_document_expiration.total_seconds() <= 0:
                raise ValueError("Document expiration must be positive")
            
            # Validate vector settings
            if self.vector.enable_vector_search and self.vector.embedding_dimension <= 0:
                raise ValueError("Embedding dimension must be positive")
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_storage_path(self, session_id: str, document_id: str) -> Path:
        """Get storage path for a document."""
        base_path = self.storage.base_storage_dir
        
        if self.storage.organize_by_session:
            base_path = base_path / session_id
        
        if self.storage.organize_by_date:
            from datetime import datetime
            date_str = datetime.now().strftime("%Y/%m/%d")
            base_path = base_path / date_str
        
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / document_id
    
    def get_vector_collection_name(self, session_id: str) -> str:
        """Get vector collection name for a session."""
        return f"{self.vector.vector_store_type}_{self.integration.rag_collection_prefix}{session_id}"
    
    def is_file_allowed(self, filename: str, content_type: str) -> bool:
        """Check if file is allowed based on configuration."""
        from pathlib import Path
        
        # Check extension
        extension = Path(filename).suffix.lower()
        if extension in self.limits.blocked_extensions:
            return False
        
        if self.limits.allowed_extensions and extension not in self.limits.allowed_extensions:
            return False
        
        # Check content type
        if content_type in self.limits.blocked_content_types:
            return False
        
        if self.limits.allowed_content_types and content_type not in self.limits.allowed_content_types:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "storage": {
                "base_storage_dir": str(self.storage.base_storage_dir),
                "temp_dir": str(self.storage.temp_dir),
                "download_dir": str(self.storage.download_dir),
                "max_storage_size": self.storage.max_storage_size,
            },
            "limits": {
                "max_documents_per_session": self.limits.max_documents_per_session,
                "max_document_size": self.limits.max_document_size,
                "max_total_size_per_session": self.limits.max_total_size_per_session,
            },
            "expiration": {
                "default_document_expiration_hours": self.expiration.default_document_expiration.total_seconds() / 3600,
                "default_workspace_expiration_hours": self.expiration.default_workspace_expiration.total_seconds() / 3600,
                "cleanup_enabled": self.expiration.cleanup_enabled,
            },
            "vector": {
                "enable_vector_search": self.vector.enable_vector_search,
                "embedding_model": self.vector.embedding_model,
                "chunk_size": self.vector.chunk_size,
            },
            "integration": {
                "enable_document_intelligence": self.integration.enable_document_intelligence,
                "enable_rag_integration": self.integration.enable_rag_integration,
                "enable_background_processing": self.integration.enable_background_processing,
            },
            "security": {
                "require_authentication": self.security.require_authentication,
                "enable_content_scanning": self.security.enable_content_scanning,
                "secure_download_links": self.security.secure_download_links,
            }
        }


# Global configuration instance
session_document_config = SessionDocumentConfig()

# Validate configuration on import
if not session_document_config.validate_configuration():
    logger.warning("⚠️ Configuration validation failed - some features may not work correctly")

logger.info("🔥 Revolutionary Session Document Configuration ready")
