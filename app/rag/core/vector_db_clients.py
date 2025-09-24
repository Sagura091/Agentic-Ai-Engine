"""
Vector Database Client Implementations for Multi-Database Support.

This module provides concrete implementations for various vector databases
including Weaviate, Qdrant, Pinecone, and others. Each client implements
the VectorDBBase interface for consistent usage across the RAG system.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

import structlog
from app.rag.core.vector_db_factory import VectorDBBase, VectorItem, SearchResult, GetResult
from app.rag.core.vector_db_config import (
    VectorDBType, WeaviateConfig, QdrantConfig, 
    get_vector_db_config_manager
)

logger = structlog.get_logger(__name__)


class WeaviateClient(VectorDBBase):
    """Weaviate vector database client implementation."""
    
    def __init__(self, config: WeaviateConfig = None):
        self.config = config or get_vector_db_config_manager().get_config(VectorDBType.WEAVIATE)
        self.client = None
        self._initialize_client()
        
        logger.info("✅ Weaviate client initialized", 
                   host=self.config.host, port=self.config.port)
    
    def _initialize_client(self):
        """Initialize Weaviate client."""
        try:
            import weaviate
            
            # Build connection URL
            url = f"{self.config.scheme}://{self.config.host}:{self.config.port}"
            
            # Setup authentication if API key is provided
            auth_config = None
            if self.config.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.config.api_key)
            
            # Initialize client
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=auth_config,
                additional_headers=self.config.additional_headers,
                timeout_config=(self.config.timeout, self.config.timeout)
            )
            
            # Test connection
            if self.client.is_ready():
                logger.info("Weaviate client connected successfully")
            else:
                logger.warning("Weaviate client connection test failed")
                
        except ImportError:
            raise ImportError("weaviate-client is required for Weaviate support. Install with: pip install weaviate-client")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise
    
    def _get_class_name(self, collection_name: str) -> str:
        """Convert collection name to Weaviate class name."""
        # Weaviate class names must start with uppercase
        return collection_name.replace("_", "").replace("-", "").capitalize()
    
    def _ensure_class_exists(self, collection_name: str):
        """Ensure Weaviate class exists for the collection."""
        class_name = self._get_class_name(collection_name)
        
        try:
            # Check if class exists
            existing_classes = self.client.schema.get()["classes"]
            class_exists = any(cls["class"] == class_name for cls in existing_classes)
            
            if not class_exists:
                # Create class schema
                class_schema = {
                    "class": class_name,
                    "description": f"Collection for {collection_name}",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "document",
                            "dataType": ["text"],
                            "description": "Document content"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Document metadata"
                        }
                    ]
                }
                
                self.client.schema.create_class(class_schema)
                logger.info(f"Created Weaviate class: {class_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure class exists: {e}")
            raise
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            class_name = self._get_class_name(collection_name)
            existing_classes = self.client.schema.get()["classes"]
            return any(cls["class"] == class_name for cls in existing_classes)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            class_name = self._get_class_name(collection_name)
            self.client.schema.delete_class(class_name)
            logger.info(f"Deleted Weaviate class: {class_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def search(self, collection_name: str, vectors: List[List[float]], limit: int) -> Optional[SearchResult]:
        """Search for similar vectors."""
        try:
            if not vectors:
                return None
            
            self._ensure_class_exists(collection_name)
            class_name = self._get_class_name(collection_name)
            query_vector = vectors[0]  # Use first vector for search
            
            # Perform vector search
            result = (
                self.client.query
                .get(class_name, ["document", "metadata"])
                .with_near_vector({"vector": query_vector})
                .with_limit(limit)
                .with_additional(["id", "distance"])
                .do()
            )
            
            if "data" not in result or "Get" not in result["data"]:
                return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]])
            
            objects = result["data"]["Get"][class_name]
            if not objects:
                return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]])
            
            ids = [obj["_additional"]["id"] for obj in objects]
            documents = [obj.get("document", "") for obj in objects]
            metadatas = [obj.get("metadata", {}) for obj in objects]
            distances = [1.0 - obj["_additional"]["distance"] for obj in objects]  # Convert to similarity
            
            return SearchResult(
                ids=[ids],
                distances=[distances],
                documents=[documents],
                metadatas=[metadatas]
            )
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            return None
    
    def add(self, collection_name: str, items: List[VectorItem]) -> bool:
        """Add items to collection."""
        try:
            self._ensure_class_exists(collection_name)
            class_name = self._get_class_name(collection_name)
            
            # Batch insert objects
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for item in items:
                    data_object = {
                        "document": item.document,
                        "metadata": item.metadata
                    }
                    
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=class_name,
                        uuid=item.id,
                        vector=item.vector
                    )
            
            logger.info(f"Added {len(items)} items to Weaviate collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add items to Weaviate: {e}")
            return False
    
    def get(self, collection_name: str) -> Optional[GetResult]:
        """Get all items from collection."""
        try:
            self._ensure_class_exists(collection_name)
            class_name = self._get_class_name(collection_name)
            
            # Get all objects
            result = (
                self.client.query
                .get(class_name, ["document", "metadata"])
                .with_additional(["id"])
                .do()
            )
            
            if "data" not in result or "Get" not in result["data"]:
                return GetResult(ids=[[]], documents=[[]], metadatas=[[]])
            
            objects = result["data"]["Get"][class_name]
            if not objects:
                return GetResult(ids=[[]], documents=[[]], metadatas=[[]])
            
            ids = [obj["_additional"]["id"] for obj in objects]
            documents = [obj.get("document", "") for obj in objects]
            metadatas = [obj.get("metadata", {}) for obj in objects]
            
            return GetResult(
                ids=[ids],
                documents=[documents],
                metadatas=[metadatas]
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection data from Weaviate: {e}")
            return None
    
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete items by IDs."""
        try:
            class_name = self._get_class_name(collection_name)
            
            for item_id in ids:
                self.client.data_object.delete(
                    uuid=item_id,
                    class_name=class_name
                )
            
            logger.info(f"Deleted {len(ids)} items from Weaviate collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete items from Weaviate: {e}")
            return False


class QdrantClient(VectorDBBase):
    """Qdrant vector database client implementation."""
    
    def __init__(self, config: QdrantConfig = None):
        self.config = config or get_vector_db_config_manager().get_config(VectorDBType.QDRANT)
        self.client = None
        self._initialize_client()
        
        logger.info("✅ Qdrant client initialized", 
                   host=self.config.host, port=self.config.port)
    
    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient as QdrantClientLib
            from qdrant_client.models import Distance, VectorParams
            
            # Initialize client
            self.client = QdrantClientLib(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                prefer_grpc=self.config.prefer_grpc,
                timeout=self.config.timeout
            )
            
            # Test connection
            info = self.client.get_collections()
            logger.info("Qdrant client connected successfully")
            
        except ImportError:
            raise ImportError("qdrant-client is required for Qdrant support. Install with: pip install qdrant-client")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def _ensure_collection_exists(self, collection_name: str, vector_size: int = 384):
        """Ensure Qdrant collection exists."""
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == collection_name for col in collections.collections)
            
            if not collection_exists:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def search(self, collection_name: str, vectors: List[List[float]], limit: int) -> Optional[SearchResult]:
        """Search for similar vectors."""
        try:
            if not vectors:
                return None
            
            self._ensure_collection_exists(collection_name, len(vectors[0]))
            query_vector = vectors[0]  # Use first vector for search
            
            # Perform search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            if not search_result:
                return SearchResult(ids=[[]], distances=[[]], documents=[[]], metadatas=[[]])
            
            ids = [str(point.id) for point in search_result]
            documents = [point.payload.get("document", "") for point in search_result]
            metadatas = [point.payload.get("metadata", {}) for point in search_result]
            distances = [point.score for point in search_result]
            
            return SearchResult(
                ids=[ids],
                distances=[distances],
                documents=[documents],
                metadatas=[metadatas]
            )
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return None
    
    def add(self, collection_name: str, items: List[VectorItem]) -> bool:
        """Add items to collection."""
        try:
            if not items:
                return True
            
            self._ensure_collection_exists(collection_name, len(items[0].vector))
            
            from qdrant_client.models import PointStruct
            
            # Prepare points
            points = []
            for item in items:
                point = PointStruct(
                    id=item.id,
                    vector=item.vector,
                    payload={
                        "document": item.document,
                        "metadata": item.metadata
                    }
                )
                points.append(point)
            
            # Upsert points
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Added {len(items)} items to Qdrant collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add items to Qdrant: {e}")
            return False
    
    def get(self, collection_name: str) -> Optional[GetResult]:
        """Get all items from collection."""
        try:
            # Qdrant doesn't have a direct "get all" method, so we'll use scroll
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=10000,  # Adjust as needed
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                return GetResult(ids=[[]], documents=[[]], metadatas=[[]])
            
            ids = [str(point.id) for point in points]
            documents = [point.payload.get("document", "") for point in points]
            metadatas = [point.payload.get("metadata", {}) for point in points]
            
            return GetResult(
                ids=[ids],
                documents=[documents],
                metadatas=[metadatas]
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection data from Qdrant: {e}")
            return None
    
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """Delete items by IDs."""
        try:
            from qdrant_client.models import PointIdsList
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=ids)
            )
            
            logger.info(f"Deleted {len(ids)} items from Qdrant collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete items from Qdrant: {e}")
            return False
