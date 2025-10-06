"""
Version information for RAG ingestion pipeline.

This module tracks:
- Pipeline version
- Schema version
- Embedding model versions
- Migration utilities
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


# Pipeline version (semantic versioning)
__version__ = "2.0.0"

# Schema version for data compatibility
SCHEMA_VERSION = "2.0"

# Supported schema versions (for backward compatibility)
SUPPORTED_SCHEMA_VERSIONS = ["1.0", "2.0"]


@dataclass
class VersionInfo:
    """Version information for a component."""
    component: str
    version: str
    schema_version: str
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "version": self.version,
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionInfo":
        """Create from dictionary."""
        return cls(
            component=data["component"],
            version=data["version"],
            schema_version=data["schema_version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )


def get_pipeline_version() -> str:
    """
    Get pipeline version.
    
    Returns:
        Version string
    """
    return __version__


def get_schema_version() -> str:
    """
    Get schema version.
    
    Returns:
        Schema version string
    """
    return SCHEMA_VERSION


def is_schema_compatible(schema_version: str) -> bool:
    """
    Check if schema version is compatible.
    
    Args:
        schema_version: Schema version to check
        
    Returns:
        True if compatible
    """
    return schema_version in SUPPORTED_SCHEMA_VERSIONS


def get_version_metadata() -> Dict[str, Any]:
    """
    Get comprehensive version metadata.
    
    Returns:
        Version metadata dictionary
    """
    return {
        "pipeline_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "supported_schemas": SUPPORTED_SCHEMA_VERSIONS,
        "timestamp": datetime.utcnow().isoformat()
    }


def create_chunk_version_metadata(
    embedding_model: str,
    embedding_version: str,
    embedding_dim: int
) -> Dict[str, Any]:
    """
    Create version metadata for a chunk.
    
    Args:
        embedding_model: Embedding model name
        embedding_version: Embedding model version
        embedding_dim: Embedding dimension
        
    Returns:
        Version metadata dictionary
    """
    return {
        "pipeline_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "embedding_model": embedding_model,
        "embedding_version": embedding_version,
        "embedding_dim": embedding_dim,
        "created_at": datetime.utcnow().isoformat()
    }


class VersionMigrator:
    """
    Utility for migrating data between schema versions.
    
    Supports:
    - Schema version upgrades
    - Embedding model migrations
    - Backward compatibility
    """
    
    def __init__(self):
        """Initialize version migrator."""
        self.migrations = {
            ("1.0", "2.0"): self._migrate_1_0_to_2_0
        }
        
        logger.info("VersionMigrator initialized")
    
    def can_migrate(self, from_version: str, to_version: str) -> bool:
        """
        Check if migration is supported.
        
        Args:
            from_version: Source schema version
            to_version: Target schema version
            
        Returns:
            True if migration is supported
        """
        return (from_version, to_version) in self.migrations
    
    def migrate(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """
        Migrate data between schema versions.
        
        Args:
            data: Data to migrate
            from_version: Source schema version
            to_version: Target schema version
            
        Returns:
            Migrated data
            
        Raises:
            ValueError: If migration not supported
        """
        if not self.can_migrate(from_version, to_version):
            raise ValueError(
                f"Migration from {from_version} to {to_version} not supported"
            )
        
        migration_func = self.migrations[(from_version, to_version)]
        migrated_data = migration_func(data)
        
        logger.info(
            "Data migrated",
            from_version=from_version,
            to_version=to_version
        )
        
        return migrated_data
    
    def _migrate_1_0_to_2_0(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate from schema 1.0 to 2.0.
        
        Changes in 2.0:
        - Added content_sha and norm_text_sha
        - Added section_path
        - Added chunk_config metadata
        - Enhanced version tracking
        
        Args:
            data: Data in schema 1.0
            
        Returns:
            Data in schema 2.0
        """
        from .utils_hash import compute_content_sha, compute_norm_text_sha
        
        migrated = data.copy()
        
        # Add missing hashes
        if "content" in migrated and "content_sha" not in migrated:
            migrated["content_sha"] = compute_content_sha(migrated["content"])
        
        if "content" in migrated and "norm_text_sha" not in migrated:
            migrated["norm_text_sha"] = compute_norm_text_sha(migrated["content"])
        
        # Add section_path if missing
        if "section_path" not in migrated:
            migrated["section_path"] = None
        
        # Add chunk_config if missing
        if "metadata" in migrated and "chunk_config" not in migrated["metadata"]:
            migrated["metadata"]["chunk_config"] = {
                "min_size": 200,
                "max_size": 800,
                "overlap_pct": 0.15
            }
        
        # Update schema version
        migrated["schema_version"] = "2.0"
        
        # Add migration metadata
        if "metadata" not in migrated:
            migrated["metadata"] = {}
        
        migrated["metadata"]["migrated_from"] = "1.0"
        migrated["metadata"]["migrated_at"] = datetime.utcnow().isoformat()
        
        return migrated


class EmbeddingVersionManager:
    """
    Manager for handling multiple embedding model versions.
    
    Supports:
    - Side-by-side embedding versions
    - Gradual migration to new models
    - Version-specific queries
    """
    
    def __init__(self):
        """Initialize embedding version manager."""
        self.active_versions: Dict[str, VersionInfo] = {}
        
        logger.info("EmbeddingVersionManager initialized")
    
    def register_version(
        self,
        model_name: str,
        model_version: str,
        embedding_dim: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an embedding model version.
        
        Args:
            model_name: Model name
            model_version: Model version
            embedding_dim: Embedding dimension
            metadata: Additional metadata
        """
        version_key = f"{model_name}:{model_version}"
        
        version_info = VersionInfo(
            component="embedding_model",
            version=model_version,
            schema_version=SCHEMA_VERSION,
            created_at=datetime.utcnow(),
            metadata={
                "model_name": model_name,
                "embedding_dim": embedding_dim,
                **(metadata or {})
            }
        )
        
        self.active_versions[version_key] = version_info
        
        logger.info(
            "Embedding version registered",
            model=model_name,
            version=model_version,
            dim=embedding_dim
        )
    
    def get_version_info(self, model_name: str, model_version: str) -> Optional[VersionInfo]:
        """
        Get version information.
        
        Args:
            model_name: Model name
            model_version: Model version
            
        Returns:
            VersionInfo if found
        """
        version_key = f"{model_name}:{model_version}"
        return self.active_versions.get(version_key)
    
    def list_versions(self) -> Dict[str, VersionInfo]:
        """
        List all registered versions.
        
        Returns:
            Dictionary of version_key -> VersionInfo
        """
        return self.active_versions.copy()


# Global instances
_version_migrator: Optional[VersionMigrator] = None
_embedding_version_manager: Optional[EmbeddingVersionManager] = None


def get_version_migrator() -> VersionMigrator:
    """
    Get global version migrator instance.
    
    Returns:
        VersionMigrator instance
    """
    global _version_migrator
    
    if _version_migrator is None:
        _version_migrator = VersionMigrator()
    
    return _version_migrator


def get_embedding_version_manager() -> EmbeddingVersionManager:
    """
    Get global embedding version manager instance.
    
    Returns:
        EmbeddingVersionManager instance
    """
    global _embedding_version_manager
    
    if _embedding_version_manager is None:
        _embedding_version_manager = EmbeddingVersionManager()
    
    return _embedding_version_manager

