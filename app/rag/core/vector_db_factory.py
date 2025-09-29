"""
Vector Database Factory inspired by OpenWebUI.

This module provides a factory pattern for creating different vector database clients
with unified interfaces, supporting ChromaDB, Qdrant, and other vector databases.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Protocol
from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

# Import configuration
from app.config.settings import get_settings
from app.rag.core.vector_db_config import (
    VectorDBType, VectorDBConfigManager,
    get_vector_db_config_manager, get_active_vector_db_type
)

logger = structlog.get_logger(__name__)

class VectorItem(BaseModel):
    """Vector item for storage."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document: Optional[str] = None

class SearchResult(BaseModel):
    """Search result from vector database."""
    ids: List[List[str]]
    distances: List[List[float]]
    documents: List[List[str]]
    metadatas: List[List[Dict[str, Any]]]

class GetResult(BaseModel):
    """Get result from vector database."""
    ids: List[List[str]]
    documents: List[List[str]]
    metadatas: List[List[Dict[str, Any]]]

class VectorDBBase(ABC):
    """Base class for vector database implementations."""
    
    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        pass
    
    @abstractmethod
    def search(
        self, 
        collection_name: str, 
        vectors: List[List[float]], 
        limit: int
    ) -> Optional[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def add(
        self,
        collection_name: str,
        items: List[VectorItem]
    ) -> bool:
        """Add items to collection."""
        pass
    
    @abstractmethod
    def get(self, collection_name: str) -> Optional[GetResult]:
        """Get all items from collection."""
        pass
    
    @abstractmethod
    def delete(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Delete items by IDs."""
        pass

class ChromaDBClient(VectorDBBase):
    """ChromaDB implementation following OpenWebUI patterns."""
    
    def __init__(self, config=None, data_dir: str = "data", **kwargs):
        from app.rag.core.vector_db_config import ChromaDBConfig, VectorDBType

        # Use provided config or get from config manager
        if config is None:
            config_manager = get_vector_db_config_manager()
            config = config_manager.get_config(VectorDBType.CHROMADB)

        self.config = config
        self.data_dir = Path(data_dir)
        self.chroma_data_path = self.data_dir / "chroma"
        self.chroma_data_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        settings_dict = {
            "allow_reset": config.allow_reset if config else True,
            "anonymized_telemetry": config.anonymized_telemetry if config else False,
        }

        # Check for HTTP configuration (prioritize config over environment)
        chroma_http_host = (config.host if config else None) or os.environ.get("CHROMA_HTTP_HOST", "")
        if chroma_http_host:
            import chromadb
            port = (config.port if config else None) or int(os.environ.get("CHROMA_HTTP_PORT", "8000"))
            ssl = (config.ssl if config else None) or (os.environ.get("CHROMA_HTTP_SSL", "false").lower() == "true")

            self.client = chromadb.HttpClient(
                host=chroma_http_host,
                port=port,
                ssl=ssl,
                tenant=config.tenant if config else os.environ.get("CHROMA_TENANT", chromadb.DEFAULT_TENANT),
                database=config.database if config else os.environ.get("CHROMA_DATABASE", chromadb.DEFAULT_DATABASE),
                settings=chromadb.Settings(**settings_dict),
            )
            logger.info("✅ ChromaDB HTTP client initialized", host=chroma_http_host, port=port)
        else:
            import chromadb
            persist_dir = (config.persist_directory if config else None) or str(self.chroma_data_path)

            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=chromadb.Settings(**settings_dict),
                tenant=config.tenant if config else os.environ.get("CHROMA_TENANT", chromadb.DEFAULT_TENANT),
                database=config.database if config else os.environ.get("CHROMA_DATABASE", chromadb.DEFAULT_DATABASE),
            )
            logger.info("✅ ChromaDB persistent client initialized", path=persist_dir)
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            return collection_name in collection_names
        except Exception as e:
            logger.error("Failed to check collection existence", collection=collection_name, error=str(e))
            return False
    
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info("Collection deleted", collection=collection_name)
        except Exception as e:
            logger.error("Failed to delete collection", collection=collection_name, error=str(e))
            raise
    
    def search(
        self, 
        collection_name: str, 
        vectors: List[List[float]], 
        limit: int
    ) -> Optional[SearchResult]:
        """Search for similar vectors."""
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection:
                result = collection.query(
                    query_embeddings=vectors,
                    n_results=limit,
                )
                
                # ChromaDB has cosine distance, 2 (worst) -> 0 (best). Re-ordering to 0 -> 1
                distances: List = result["distances"][0] if result["distances"] else []
                distances = [2 - dist for dist in distances]
                distances = [[dist / 2 for dist in distances]]
                
                return SearchResult(
                    ids=result["ids"],
                    distances=distances,
                    documents=result["documents"],
                    metadatas=result["metadatas"],
                )
            return None
        except Exception as e:
            logger.error("Search failed", collection=collection_name, error=str(e))
            return None
    
    def add(
        self,
        collection_name: str,
        items: List[VectorItem]
    ) -> bool:
        """Add items to collection."""
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # Prepare data for ChromaDB
            ids = [item.id for item in items]
            embeddings = [item.vector for item in items]
            documents = [item.document or "" for item in items]
            metadatas = []
            
            # Ensure metadata values are compatible with ChromaDB
            for item in items:
                metadata = {}
                for key, value in item.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
                    elif isinstance(value, list):
                        # Convert lists to strings
                        metadata[key] = str(value)
                    else:
                        metadata[key] = str(value)
                metadatas.append(metadata)
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info("Items added to collection", collection=collection_name, count=len(items))
            return True
            
        except Exception as e:
            logger.error("Failed to add items", collection=collection_name, error=str(e))
            return False
    
    def get(self, collection_name: str) -> Optional[GetResult]:
        """Get all items from collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection:
                result = collection.get()
                return GetResult(
                    ids=[result["ids"]],
                    documents=[result["documents"]],
                    metadatas=[result["metadatas"]],
                )
            return None
        except Exception as e:
            logger.error("Failed to get collection data", collection=collection_name, error=str(e))
            return None
    
    def delete(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Delete items by IDs."""
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection:
                collection.delete(ids=ids)
                logger.info("Items deleted from collection", collection=collection_name, count=len(ids))
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete items", collection=collection_name, error=str(e))
            return False

class PgVectorClient(VectorDBBase):
    """PostgreSQL with pgvector extension implementation."""

    def __init__(self, config=None, connection_string: str = None, table_name: str = "embeddings", dimension: int = 384, **kwargs):
        from app.rag.core.vector_db_config import PgVectorConfig, VectorDBType

        # Use provided config or get from config manager
        if config is None:
            config_manager = get_vector_db_config_manager()
            config = config_manager.get_config(VectorDBType.PGVECTOR)

        self.config = config

        # Build connection string from config or use provided one
        if connection_string:
            self.connection_string = connection_string
        elif config:
            # Build connection string from config
            password_part = f":{config.password}" if config.password else ""
            self.connection_string = f"postgresql://{config.username}{password_part}@{config.host}:{config.port}/{config.database}"
        else:
            self.connection_string = os.environ.get("DATABASE_URL")

        self.table_name = (config.table_name if config else None) or table_name
        self.dimension = (config.dimension if config else None) or dimension
        self.pool = None

        if not self.connection_string:
            raise ValueError("Database connection string is required for pgvector")

        logger.info("✅ PgVector client initialized",
                   table=self.table_name, dimension=self.dimension,
                   host=config.host if config else "unknown")

    async def _get_connection(self):
        """Get database connection from pool."""
        if not self.pool:
            try:
                import asyncpg
                self.pool = await asyncpg.create_pool(self.connection_string)

                # Create table and extension if not exists
                async with self.pool.acquire() as conn:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id TEXT PRIMARY KEY,
                            collection_name TEXT NOT NULL,
                            embedding vector({self.dimension}),
                            document TEXT,
                            metadata JSONB DEFAULT '{{}}'::jsonb,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_collection ON {self.table_name} (collection_name)")
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)")

            except ImportError:
                raise ImportError("asyncpg is required for pgvector support. Install with: pip install asyncpg")
            except Exception as e:
                logger.error("Failed to initialize pgvector connection", error=str(e))
                raise

        return self.pool.acquire()

    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            import asyncio
            return asyncio.run(self._has_collection_async(collection_name))
        except Exception as e:
            logger.error("Failed to check collection existence", collection=collection_name, error=str(e))
            return False

    async def _has_collection_async(self, collection_name: str) -> bool:
        """Async version of has_collection."""
        try:
            async with await self._get_connection() as conn:
                result = await conn.fetchval(
                    f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE collection_name = $1)",
                    collection_name
                )
                return bool(result)
        except Exception as e:
            logger.error("Failed to check collection existence", collection=collection_name, error=str(e))
            return False

    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            import asyncio
            asyncio.run(self._delete_collection_async(collection_name))
            logger.info("Collection deleted", collection=collection_name)
        except Exception as e:
            logger.error("Failed to delete collection", collection=collection_name, error=str(e))
            raise

    async def _delete_collection_async(self, collection_name: str):
        """Async version of delete_collection."""
        async with await self._get_connection() as conn:
            await conn.execute(f"DELETE FROM {self.table_name} WHERE collection_name = $1", collection_name)

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        limit: int
    ) -> Optional[SearchResult]:
        """Search for similar vectors."""
        try:
            import asyncio
            return asyncio.run(self._search_async(collection_name, vectors, limit))
        except Exception as e:
            logger.error("Search failed", collection=collection_name, error=str(e))
            return None

    async def _search_async(
        self,
        collection_name: str,
        vectors: List[List[float]],
        limit: int
    ) -> Optional[SearchResult]:
        """Async version of search."""
        try:
            if not vectors:
                return None

            query_vector = vectors[0]  # Use first vector for search

            async with await self._get_connection() as conn:
                rows = await conn.fetch(f"""
                    SELECT id, document, metadata, embedding <=> $1 as distance
                    FROM {self.table_name}
                    WHERE collection_name = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, query_vector, collection_name, limit)

                if not rows:
                    return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]])

                ids = [row['id'] for row in rows]
                documents = [row['document'] or "" for row in rows]
                metadatas = [dict(row['metadata']) if row['metadata'] else {} for row in rows]
                distances = [1.0 - row['distance'] for row in rows]  # Convert distance to similarity

                return SearchResult(
                    ids=[ids],
                    distances=[distances],
                    documents=[documents],
                    metadatas=[metadatas]
                )

        except Exception as e:
            logger.error("Search failed", collection=collection_name, error=str(e))
            return None

    def add(
        self,
        collection_name: str,
        items: List[VectorItem]
    ) -> bool:
        """Add items to collection."""
        try:
            import asyncio
            return asyncio.run(self._add_async(collection_name, items))
        except Exception as e:
            logger.error("Failed to add items", collection=collection_name, error=str(e))
            return False

    async def _add_async(
        self,
        collection_name: str,
        items: List[VectorItem]
    ) -> bool:
        """Async version of add."""
        try:
            async with await self._get_connection() as conn:
                for item in items:
                    await conn.execute(f"""
                        INSERT INTO {self.table_name} (id, collection_name, embedding, document, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            document = EXCLUDED.document,
                            metadata = EXCLUDED.metadata
                    """, item.id, collection_name, item.vector, item.document, item.metadata)

                logger.info("Items added to collection", collection=collection_name, count=len(items))
                return True

        except Exception as e:
            logger.error("Failed to add items", collection=collection_name, error=str(e))
            return False

    def get(self, collection_name: str) -> Optional[GetResult]:
        """Get all items from collection."""
        try:
            import asyncio
            return asyncio.run(self._get_async(collection_name))
        except Exception as e:
            logger.error("Failed to get collection data", collection=collection_name, error=str(e))
            return None

    async def _get_async(self, collection_name: str) -> Optional[GetResult]:
        """Async version of get."""
        try:
            async with await self._get_connection() as conn:
                rows = await conn.fetch(f"""
                    SELECT id, document, metadata
                    FROM {self.table_name}
                    WHERE collection_name = $1
                    ORDER BY created_at
                """, collection_name)

                if not rows:
                    return GetResult(ids=[[]], documents=[[]], metadatas=[[]])

                ids = [row['id'] for row in rows]
                documents = [row['document'] or "" for row in rows]
                metadatas = [dict(row['metadata']) if row['metadata'] else {} for row in rows]

                return GetResult(
                    ids=[ids],
                    documents=[documents],
                    metadatas=[metadatas]
                )

        except Exception as e:
            logger.error("Failed to get collection data", collection=collection_name, error=str(e))
            return None

    def delete(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Delete items by IDs."""
        try:
            import asyncio
            return asyncio.run(self._delete_async(collection_name, ids))
        except Exception as e:
            logger.error("Failed to delete items", collection=collection_name, error=str(e))
            return False

    async def _delete_async(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """Async version of delete."""
        try:
            async with await self._get_connection() as conn:
                await conn.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE collection_name = $1 AND id = ANY($2)
                """, collection_name, ids)

                logger.info("Items deleted from collection", collection=collection_name, count=len(ids))
                return True

        except Exception as e:
            logger.error("Failed to delete items", collection=collection_name, error=str(e))
            return False

class VectorDBFactory:
    """Factory for creating vector database clients."""

    _clients: Dict[str, VectorDBBase] = {}
    _config_manager: Optional[VectorDBConfigManager] = None

    @classmethod
    def _get_config_manager(cls) -> VectorDBConfigManager:
        """Get or create the configuration manager."""
        if cls._config_manager is None:
            cls._config_manager = get_vector_db_config_manager()
        return cls._config_manager

    @classmethod
    def get_client(cls, db_type: str = None, **kwargs) -> VectorDBBase:
        """Get or create a vector database client."""
        config_manager = cls._get_config_manager()

        # If no db_type specified, get from configuration
        if db_type is None:
            db_type = config_manager.active_db_type.value

        # Normalize db_type
        db_type = db_type.lower()

        if db_type not in cls._clients:
            # Get configuration for the database type
            try:
                db_type_enum = VectorDBType(db_type)
                config = config_manager.get_config(db_type_enum)

                if not config:
                    raise ValueError(f"No configuration found for database type: {db_type}")

                # Create client based on type
                if db_type_enum == VectorDBType.CHROMADB:
                    cls._clients[db_type] = ChromaDBClient(config, **kwargs)
                elif db_type_enum == VectorDBType.PGVECTOR:
                    cls._clients[db_type] = PgVectorClient(config, **kwargs)
                elif db_type_enum == VectorDBType.WEAVIATE:
                    from app.rag.core.vector_db_clients import WeaviateClient
                    cls._clients[db_type] = WeaviateClient(config, **kwargs)
                elif db_type_enum == VectorDBType.QDRANT:
                    from app.rag.core.vector_db_clients import QdrantClient
                    cls._clients[db_type] = QdrantClient(config, **kwargs)
                else:
                    raise ValueError(f"Unsupported vector database type: {db_type}")

            except ValueError as e:
                logger.warning(f"Invalid database type {db_type}: {e}, falling back to active database")
                # Fallback to active database type
                active_type = config_manager.active_db_type.value
                if active_type not in cls._clients:
                    return cls.get_client(active_type, **kwargs)
                db_type = active_type

        return cls._clients[db_type]

    @classmethod
    def _get_configured_db_type(cls) -> str:
        """Get the configured vector database type from settings."""
        config_manager = cls._get_config_manager()
        return config_manager.active_db_type.value

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available vector database types."""
        config_manager = cls._get_config_manager()
        available_types = []

        # Get available types from config manager
        for db_type in config_manager.get_available_databases():
            available_types.append(db_type.value)

        # Add legacy aliases for backward compatibility
        if "chromadb" in available_types:
            available_types.append("chroma")
        if "pgvector" in available_types:
            available_types.extend(["postgres", "postgresql"])

        return available_types

    @classmethod
    def reset_clients(cls):
        """Reset all clients (useful for testing)."""
        cls._clients.clear()

# Global client instance - will be created based on configuration
VECTOR_DB_CLIENT = None

def get_vector_db_client(db_type: str = None, **kwargs) -> VectorDBBase:
    """Get vector database client."""
    global VECTOR_DB_CLIENT

    # If no specific type requested and no global client exists, create one
    if db_type is None and VECTOR_DB_CLIENT is None:
        VECTOR_DB_CLIENT = VectorDBFactory.get_client()
        return VECTOR_DB_CLIENT

    # If specific type requested, always create/get that type
    if db_type is not None:
        return VectorDBFactory.get_client(db_type, **kwargs)

    # Return existing global client
    return VECTOR_DB_CLIENT or VectorDBFactory.get_client()

def reset_vector_db_client():
    """Reset the global vector database client."""
    global VECTOR_DB_CLIENT
    VECTOR_DB_CLIENT = None
    VectorDBFactory.reset_clients()

def get_available_vector_db_types() -> List[str]:
    """Get list of available vector database types."""
    return VectorDBFactory.get_available_types()
