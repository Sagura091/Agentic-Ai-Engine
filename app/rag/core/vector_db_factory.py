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
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.chroma_data_path = self.data_dir / "vector_databases" / "chroma"
        self.chroma_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        settings_dict = {
            "allow_reset": True,
            "anonymized_telemetry": False,
        }
        
        # Check for HTTP configuration
        chroma_http_host = os.environ.get("CHROMA_HTTP_HOST", "")
        if chroma_http_host:
            import chromadb
            self.client = chromadb.HttpClient(
                host=chroma_http_host,
                port=int(os.environ.get("CHROMA_HTTP_PORT", "8000")),
                ssl=os.environ.get("CHROMA_HTTP_SSL", "false").lower() == "true",
                tenant=os.environ.get("CHROMA_TENANT", chromadb.DEFAULT_TENANT),
                database=os.environ.get("CHROMA_DATABASE", chromadb.DEFAULT_DATABASE),
                settings=chromadb.Settings(**settings_dict),
            )
            logger.info("✅ ChromaDB HTTP client initialized", host=chroma_http_host)
        else:
            import chromadb
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_data_path),
                settings=chromadb.Settings(**settings_dict),
                tenant=os.environ.get("CHROMA_TENANT", chromadb.DEFAULT_TENANT),
                database=os.environ.get("CHROMA_DATABASE", chromadb.DEFAULT_DATABASE),
            )
            logger.info("✅ ChromaDB persistent client initialized", path=str(self.chroma_data_path))
    
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

class VectorDBFactory:
    """Factory for creating vector database clients."""
    
    _clients: Dict[str, VectorDBBase] = {}
    
    @classmethod
    def get_client(cls, db_type: str = "chroma", **kwargs) -> VectorDBBase:
        """Get or create a vector database client."""
        if db_type not in cls._clients:
            if db_type == "chroma":
                cls._clients[db_type] = ChromaDBClient(**kwargs)
            else:
                raise ValueError(f"Unsupported vector database type: {db_type}")
        
        return cls._clients[db_type]
    
    @classmethod
    def reset_clients(cls):
        """Reset all clients (useful for testing)."""
        cls._clients.clear()

# Global client instance
VECTOR_DB_CLIENT = VectorDBFactory.get_client()

def get_vector_db_client(db_type: str = "chroma", **kwargs) -> VectorDBBase:
    """Get vector database client."""
    return VectorDBFactory.get_client(db_type, **kwargs)
