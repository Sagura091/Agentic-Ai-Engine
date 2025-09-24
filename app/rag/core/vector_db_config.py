"""
Vector Database Configuration Manager for Multi-Database Support.

This module provides comprehensive configuration management for multiple vector databases
including ChromaDB, pgvector, Weaviate, Qdrant, and more. It supports environment variables,
configuration files, and runtime switching between different vector database backends.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)


class VectorDBType(str, Enum):
    """Supported vector database types."""
    AUTO = "auto"
    CHROMADB = "chromadb"
    PGVECTOR = "pgvector"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    REDIS = "redis"


@dataclass
class VectorDBConnectionConfig:
    """Base configuration for vector database connections."""
    db_type: VectorDBType
    enabled: bool = True
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    ssl: bool = False
    timeout: int = 30
    pool_size: int = 10
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChromaDBConfig(VectorDBConnectionConfig):
    """ChromaDB specific configuration."""
    db_type: VectorDBType = VectorDBType.CHROMADB
    persist_directory: str = "./data/chroma"
    collection_metadata: Dict[str, str] = field(default_factory=lambda: {"hnsw:space": "cosine"})
    anonymized_telemetry: bool = False
    allow_reset: bool = False
    tenant: str = "default_tenant"
    database: str = "default_database"


@dataclass
class PgVectorConfig(VectorDBConnectionConfig):
    """PostgreSQL with pgvector extension configuration."""
    db_type: VectorDBType = VectorDBType.PGVECTOR
    host: str = "localhost"
    port: int = 5432
    database: str = "agentic_ai"
    username: str = "agentic_user"
    table_name: str = "embeddings"
    dimension: int = 384
    distance_function: str = "cosine"  # cosine, l2, inner_product
    index_type: str = "ivfflat"  # ivfflat, hnsw
    index_lists: int = 100


@dataclass
class WeaviateConfig(VectorDBConnectionConfig):
    """Weaviate configuration."""
    db_type: VectorDBType = VectorDBType.WEAVIATE
    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051
    scheme: str = "http"
    api_key: Optional[str] = None
    additional_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class QdrantConfig(VectorDBConnectionConfig):
    """Qdrant configuration."""
    db_type: VectorDBType = VectorDBType.QDRANT
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    prefer_grpc: bool = False


class VectorDBConfigManager:
    """
    Comprehensive vector database configuration manager.
    
    Supports multiple configuration sources:
    1. Environment variables
    2. Configuration files (JSON/YAML)
    3. Runtime configuration
    4. Default fallbacks
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv("VECTOR_DB_CONFIG_FILE", "data/config/vector_db_config.json")
        self.configs: Dict[VectorDBType, VectorDBConnectionConfig] = {}
        self.active_db_type: VectorDBType = VectorDBType.AUTO
        
        # Load configurations
        self._load_default_configs()
        self._load_from_file()
        self._load_from_environment()
        
        # Determine active database type
        self._determine_active_db_type()
        
        logger.info("Vector database configuration manager initialized", 
                   active_db=self.active_db_type.value,
                   available_dbs=list(self.configs.keys()))
    
    def _load_default_configs(self):
        """Load default configurations for all supported databases."""
        self.configs = {
            VectorDBType.CHROMADB: ChromaDBConfig(),
            VectorDBType.PGVECTOR: PgVectorConfig(),
            VectorDBType.WEAVIATE: WeaviateConfig(),
            VectorDBType.QDRANT: QdrantConfig(),
        }
    
    def _load_from_file(self):
        """Load configuration from file if it exists."""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                for db_type_str, config_data in file_config.items():
                    try:
                        db_type = VectorDBType(db_type_str)
                        if db_type in self.configs:
                            # Update existing config with file data
                            config_obj = self.configs[db_type]
                            for key, value in config_data.items():
                                if hasattr(config_obj, key):
                                    setattr(config_obj, key, value)
                    except ValueError:
                        logger.warning(f"Unknown database type in config file: {db_type_str}")
                        
                logger.info(f"Loaded vector database configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file {config_path}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Global settings
        if os.getenv("VECTOR_DB_TYPE"):
            try:
                self.active_db_type = VectorDBType(os.getenv("VECTOR_DB_TYPE"))
            except ValueError:
                logger.warning(f"Invalid VECTOR_DB_TYPE environment variable: {os.getenv('VECTOR_DB_TYPE')}")
        
        # ChromaDB environment variables
        chroma_config = self.configs[VectorDBType.CHROMADB]
        chroma_config.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", chroma_config.persist_directory)
        chroma_config.host = os.getenv("CHROMA_HOST", chroma_config.host)
        chroma_config.port = int(os.getenv("CHROMA_PORT", chroma_config.port or 8000))
        chroma_config.ssl = os.getenv("CHROMA_SSL", "false").lower() == "true"
        
        # PostgreSQL/pgvector environment variables
        pgvector_config = self.configs[VectorDBType.PGVECTOR]
        pgvector_config.host = os.getenv("PGVECTOR_HOST", pgvector_config.host)
        pgvector_config.port = int(os.getenv("PGVECTOR_PORT", pgvector_config.port))
        pgvector_config.database = os.getenv("PGVECTOR_DATABASE", pgvector_config.database)
        pgvector_config.username = os.getenv("PGVECTOR_USERNAME", pgvector_config.username)
        pgvector_config.password = os.getenv("PGVECTOR_PASSWORD", pgvector_config.password)
        pgvector_config.table_name = os.getenv("PGVECTOR_TABLE", pgvector_config.table_name)
        
        # Weaviate environment variables
        weaviate_config = self.configs[VectorDBType.WEAVIATE]
        weaviate_config.host = os.getenv("WEAVIATE_HOST", weaviate_config.host)
        weaviate_config.port = int(os.getenv("WEAVIATE_PORT", weaviate_config.port))
        weaviate_config.scheme = os.getenv("WEAVIATE_SCHEME", weaviate_config.scheme)
        weaviate_config.api_key = os.getenv("WEAVIATE_API_KEY", weaviate_config.api_key)
        
        # Qdrant environment variables
        qdrant_config = self.configs[VectorDBType.QDRANT]
        qdrant_config.host = os.getenv("QDRANT_HOST", qdrant_config.host)
        qdrant_config.port = int(os.getenv("QDRANT_PORT", qdrant_config.port))
        qdrant_config.api_key = os.getenv("QDRANT_API_KEY", qdrant_config.api_key)
        
        logger.debug("Loaded vector database configuration from environment variables")
    
    def _determine_active_db_type(self):
        """Determine which database type should be active."""
        if self.active_db_type == VectorDBType.AUTO:
            # Auto-detection logic
            for db_type in [VectorDBType.CHROMADB, VectorDBType.PGVECTOR, VectorDBType.WEAVIATE, VectorDBType.QDRANT]:
                if self._is_database_available(db_type):
                    self.active_db_type = db_type
                    logger.info(f"Auto-detected vector database: {db_type.value}")
                    break
            else:
                # Fallback to ChromaDB
                self.active_db_type = VectorDBType.CHROMADB
                logger.info("No vector database detected, falling back to ChromaDB")
    
    def _is_database_available(self, db_type: VectorDBType) -> bool:
        """Check if a specific database type is available."""
        config = self.configs.get(db_type)
        if not config or not config.enabled:
            return False
        
        # Basic availability checks
        if db_type == VectorDBType.CHROMADB:
            try:
                import chromadb
                return True
            except ImportError:
                return False
        
        elif db_type == VectorDBType.PGVECTOR:
            try:
                import asyncpg
                # Actually test PostgreSQL connection
                import asyncio
                async def test_connection():
                    try:
                        conn = await asyncpg.connect(
                            host=config.host,
                            port=config.port,
                            user=config.username,
                            password=config.password,
                            database=config.database,
                            timeout=2.0
                        )
                        await conn.close()
                        return True
                    except Exception:
                        return False

                # Only return True if we can actually connect
                try:
                    return asyncio.run(test_connection())
                except Exception:
                    return False
            except ImportError:
                return False
        
        elif db_type == VectorDBType.WEAVIATE:
            try:
                import weaviate
                return True
            except ImportError:
                return False
        
        elif db_type == VectorDBType.QDRANT:
            try:
                import qdrant_client
                return True
            except ImportError:
                return False
        
        return False
    
    def get_active_config(self) -> VectorDBConnectionConfig:
        """Get the configuration for the currently active database."""
        return self.configs[self.active_db_type]
    
    def get_config(self, db_type: VectorDBType) -> Optional[VectorDBConnectionConfig]:
        """Get configuration for a specific database type."""
        return self.configs.get(db_type)
    
    def set_active_database(self, db_type: VectorDBType):
        """Set the active database type."""
        if db_type not in self.configs:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        if not self._is_database_available(db_type):
            raise RuntimeError(f"Database type {db_type} is not available")
        
        self.active_db_type = db_type
        logger.info(f"Switched active vector database to: {db_type.value}")
    
    def get_available_databases(self) -> List[VectorDBType]:
        """Get list of available database types."""
        return [db_type for db_type in self.configs.keys() if self._is_database_available(db_type)]
    
    def save_config(self):
        """Save current configuration to file."""
        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {}
        for db_type, config in self.configs.items():
            config_data[db_type.value] = {
                "enabled": config.enabled,
                "host": config.host,
                "port": config.port,
                "username": config.username,
                "password": config.password,
                "database": config.database,
                "ssl": config.ssl,
                "timeout": config.timeout,
                "pool_size": config.pool_size,
                "extra_params": config.extra_params
            }
            
            # Add database-specific fields
            if isinstance(config, ChromaDBConfig):
                config_data[db_type.value].update({
                    "persist_directory": config.persist_directory,
                    "collection_metadata": config.collection_metadata,
                    "anonymized_telemetry": config.anonymized_telemetry,
                    "allow_reset": config.allow_reset,
                    "tenant": config.tenant
                })
            elif isinstance(config, PgVectorConfig):
                config_data[db_type.value].update({
                    "table_name": config.table_name,
                    "dimension": config.dimension,
                    "distance_function": config.distance_function,
                    "index_type": config.index_type,
                    "index_lists": config.index_lists
                })
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved vector database configuration to {config_path}")


# Global configuration manager instance
_config_manager: Optional[VectorDBConfigManager] = None

def get_vector_db_config_manager() -> VectorDBConfigManager:
    """Get the global vector database configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = VectorDBConfigManager()
    return _config_manager

def get_active_vector_db_config() -> VectorDBConnectionConfig:
    """Get the active vector database configuration."""
    return get_vector_db_config_manager().get_active_config()

def get_active_vector_db_type() -> VectorDBType:
    """Get the active vector database type."""
    return get_vector_db_config_manager().active_db_type
