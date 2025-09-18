"""
OpenWebUI-Inspired RAG Configuration System.

This module provides comprehensive configuration management for the RAG system,
following OpenWebUI's patterns for environment variables, persistent config,
and data directory management.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)

class PersistentConfig:
    """
    Persistent configuration class inspired by OpenWebUI.
    
    Manages configuration values that can be set via environment variables
    or persisted in the database/config files.
    """
    
    def __init__(self, env_name: str, config_path: str, default_value: Any):
        self.env_name = env_name
        self.config_path = config_path
        self.default_value = default_value
        self._value = None
        self._load_value()
    
    def _load_value(self):
        """Load value from environment or use default."""
        env_value = os.environ.get(self.env_name)
        if env_value is not None:
            # Try to parse as appropriate type
            if isinstance(self.default_value, bool):
                self._value = env_value.lower() in ("true", "1", "yes", "on")
            elif isinstance(self.default_value, int):
                try:
                    self._value = int(env_value)
                except ValueError:
                    self._value = self.default_value
            elif isinstance(self.default_value, float):
                try:
                    self._value = float(env_value)
                except ValueError:
                    self._value = self.default_value
            else:
                self._value = env_value
        else:
            self._value = self.default_value
    
    @property
    def value(self):
        """Get the current value."""
        return self._value
    
    def set_value(self, value: Any):
        """Set a new value."""
        self._value = value

class RAGConfig(BaseModel):
    """Comprehensive RAG configuration."""
    
    # Data directories
    data_dir: str = Field(default="data", description="Base data directory")
    vector_db_dir: str = Field(default="data/vector_databases", description="Vector database directory")
    uploads_dir: str = Field(default="data/uploads", description="File uploads directory")
    cache_dir: str = Field(default="data/cache", description="Cache directory")
    models_dir: str = Field(default="data/models", description="Models directory")
    
    # Vector Database Configuration
    vector_db: str = Field(default="chroma", description="Vector database type")
    
    # ChromaDB Configuration
    chroma_data_path: str = Field(default="data/vector_databases/chroma", description="ChromaDB data path")
    chroma_http_host: str = Field(default="", description="ChromaDB HTTP host")
    chroma_http_port: int = Field(default=8000, description="ChromaDB HTTP port")
    chroma_http_ssl: bool = Field(default=False, description="ChromaDB HTTP SSL")
    chroma_tenant: str = Field(default="default_tenant", description="ChromaDB tenant")
    chroma_database: str = Field(default="default_database", description="ChromaDB database")
    
    # Embedding Configuration
    embedding_engine: str = Field(default="", description="Embedding engine: '', 'ollama', 'openai', 'azure_openai'")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    embedding_batch_size: int = Field(default=32, description="Embedding batch size")
    embedding_auto_update: bool = Field(default=True, description="Auto-update embedding models")
    embedding_trust_remote_code: bool = Field(default=True, description="Trust remote code for models")
    
    # RAG Parameters
    rag_top_k: int = Field(default=5, description="Top K results for RAG")
    rag_chunk_size: int = Field(default=1000, description="Text chunk size")
    rag_chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    rag_similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    
    # Hybrid Search Configuration
    enable_hybrid_search: bool = Field(default=True, description="Enable hybrid search")
    hybrid_bm25_weight: float = Field(default=0.3, description="BM25 weight in hybrid search")
    
    # Reranking Configuration
    reranking_engine: str = Field(default="", description="Reranking engine")
    reranking_model: str = Field(default="", description="Reranking model")
    reranking_top_k: int = Field(default=10, description="Top K for reranking")
    
    # Performance Configuration
    max_concurrent_operations: int = Field(default=10, description="Max concurrent operations")
    connection_pool_size: int = Field(default=10, description="Connection pool size")
    
    # Security and Access Control
    bypass_embedding_and_retrieval: bool = Field(default=False, description="Bypass embedding for full context")
    enable_access_control: bool = Field(default=True, description="Enable access control")

class OpenWebUIRAGConfig:
    """
    OpenWebUI-inspired RAG configuration manager.
    
    This class manages all RAG-related configuration following OpenWebUI's patterns
    for environment variables, persistent configuration, and data management.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "rag_config.json"
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent configurations
        self._init_persistent_configs()
        
        # Load or create configuration
        self.config = self._load_config()
        
        logger.info("✅ RAG configuration initialized", data_dir=str(self.data_dir))
    
    def _init_persistent_configs(self):
        """Initialize persistent configuration objects."""
        
        # Vector Database Configuration
        self.VECTOR_DB = PersistentConfig(
            "VECTOR_DB", "vector_db.type", "chroma"
        )
        
        # ChromaDB Configuration
        self.CHROMA_HTTP_HOST = PersistentConfig(
            "CHROMA_HTTP_HOST", "chroma.http_host", ""
        )
        self.CHROMA_HTTP_PORT = PersistentConfig(
            "CHROMA_HTTP_PORT", "chroma.http_port", 8000
        )
        self.CHROMA_HTTP_SSL = PersistentConfig(
            "CHROMA_HTTP_SSL", "chroma.http_ssl", False
        )
        
        # Embedding Configuration
        self.RAG_EMBEDDING_ENGINE = PersistentConfig(
            "RAG_EMBEDDING_ENGINE", "rag.embedding_engine", ""
        )
        self.RAG_EMBEDDING_MODEL = PersistentConfig(
            "RAG_EMBEDDING_MODEL", "rag.embedding_model", "all-MiniLM-L6-v2"
        )
        self.RAG_EMBEDDING_BATCH_SIZE = PersistentConfig(
            "RAG_EMBEDDING_BATCH_SIZE", "rag.embedding_batch_size", 32
        )
        
        # RAG Parameters
        self.RAG_TOP_K = PersistentConfig(
            "RAG_TOP_K", "rag.top_k", 5
        )
        self.RAG_CHUNK_SIZE = PersistentConfig(
            "RAG_CHUNK_SIZE", "rag.chunk_size", 1000
        )
        self.RAG_CHUNK_OVERLAP = PersistentConfig(
            "RAG_CHUNK_OVERLAP", "rag.chunk_overlap", 200
        )
        
        # Hybrid Search
        self.ENABLE_HYBRID_SEARCH = PersistentConfig(
            "ENABLE_HYBRID_SEARCH", "rag.hybrid_search.enable", True
        )
        self.HYBRID_BM25_WEIGHT = PersistentConfig(
            "HYBRID_BM25_WEIGHT", "rag.hybrid_search.bm25_weight", 0.3
        )
        
        # Security
        self.BYPASS_EMBEDDING_AND_RETRIEVAL = PersistentConfig(
            "BYPASS_EMBEDDING_AND_RETRIEVAL", "rag.bypass_embedding_and_retrieval", False
        )
    
    def _load_config(self) -> RAGConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return RAGConfig(**config_data)
            except Exception as e:
                logger.warning("Failed to load config file, using defaults", error=str(e))
        
        # Create default configuration
        config = RAGConfig(
            data_dir=str(self.data_dir),
            vector_db_dir=str(self.data_dir / "vector_databases"),
            uploads_dir=str(self.data_dir / "uploads"),
            cache_dir=str(self.data_dir / "cache"),
            models_dir=str(self.data_dir / "models"),
            
            # Apply persistent config values
            vector_db=self.VECTOR_DB.value,
            chroma_http_host=self.CHROMA_HTTP_HOST.value,
            chroma_http_port=self.CHROMA_HTTP_PORT.value,
            chroma_http_ssl=self.CHROMA_HTTP_SSL.value,
            
            embedding_engine=self.RAG_EMBEDDING_ENGINE.value,
            embedding_model=self.RAG_EMBEDDING_MODEL.value,
            embedding_batch_size=self.RAG_EMBEDDING_BATCH_SIZE.value,
            
            rag_top_k=self.RAG_TOP_K.value,
            rag_chunk_size=self.RAG_CHUNK_SIZE.value,
            rag_chunk_overlap=self.RAG_CHUNK_OVERLAP.value,
            
            enable_hybrid_search=self.ENABLE_HYBRID_SEARCH.value,
            hybrid_bm25_weight=self.HYBRID_BM25_WEIGHT.value,
            
            bypass_embedding_and_retrieval=self.BYPASS_EMBEDDING_AND_RETRIEVAL.value,
        )
        
        self._save_config(config)
        return config
    
    def _save_config(self, config: RAGConfig):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config.model_dump(), f, indent=2)
            logger.info("Configuration saved", file=str(self.config_file))
        except Exception as e:
            logger.error("Failed to save configuration", error=str(e))
    
    def update_config(self, **kwargs):
        """Update configuration values."""
        config_dict = self.config.model_dump()
        config_dict.update(kwargs)
        self.config = RAGConfig(**config_dict)
        self._save_config(self.config)
        logger.info("Configuration updated", updates=list(kwargs.keys()))
    
    def get_data_directories(self) -> Dict[str, Path]:
        """Get all data directories as Path objects."""
        return {
            "base": Path(self.config.data_dir),
            "vector_db": Path(self.config.vector_db_dir),
            "uploads": Path(self.config.uploads_dir),
            "cache": Path(self.config.cache_dir),
            "models": Path(self.config.models_dir),
        }
    
    def ensure_directories(self):
        """Ensure all data directories exist."""
        directories = self.get_data_directories()
        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Directory ensured", name=name, path=str(path))

# Global configuration instance
_global_config: Optional[OpenWebUIRAGConfig] = None

def get_rag_config(data_dir: str = "data") -> OpenWebUIRAGConfig:
    """Get the global RAG configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = OpenWebUIRAGConfig(data_dir)
    return _global_config

def reset_rag_config():
    """Reset the global configuration (useful for testing)."""
    global _global_config
    _global_config = None
