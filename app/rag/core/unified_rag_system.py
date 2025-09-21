"""
Unified RAG System - THE Single RAG System for Multi-Agent Architecture.

This is THE ONLY RAG system in the entire application. All RAG operations
flow through this unified system, providing:

CORE ARCHITECTURE:
- Single ChromaDB instance with collection-based isolation
- Agent-specific knowledge bases (kb_agent_{id})
- Agent-specific memory collections (memory_agent_{id})
- Unified tool integration
- Performance-optimized operations
- Clean, simple, fast architecture

DESIGN PRINCIPLES:
- One RAG system to rule them all
- Agent isolation through collections
- Shared infrastructure, private data
- Simple, clean, fast operations
- No complexity unless absolutely necessary
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

import structlog
from pydantic import BaseModel, Field

# Core RAG components
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

# Internal imports
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


# Data classes for compatibility with existing code
@dataclass
class Document:
    """Document representation for RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentChunk:
    """Document chunk representation for RAG system."""
    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class KnowledgeQuery:
    """Query representation for knowledge search."""
    query: str
    collection: Optional[str] = None
    top_k: int = 10
    filters: Dict[str, Any] = None
    include_metadata: bool = True

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class KnowledgeResult:
    """Result from knowledge search."""
    documents: List[Document]
    scores: List[float]
    total_results: int
    query_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KnowledgeConfig(BaseModel):
    """Configuration for knowledge base operations."""
    default_collection: str = Field(default="default")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    enable_caching: bool = Field(default=True)


class CollectionType(str, Enum):
    """Types of collections in the unified system."""
    AGENT_KNOWLEDGE = "kb_agent"          # Agent knowledge base: kb_agent_{id}
    AGENT_MEMORY_SHORT = "memory_short"   # Short-term memory: memory_short_{id}
    AGENT_MEMORY_LONG = "memory_long"     # Long-term memory: memory_long_{id}
    SHARED_KNOWLEDGE = "shared"           # Shared knowledge: shared_{domain}
    GLOBAL_KNOWLEDGE = "global"           # Global knowledge: global


@dataclass
class AgentCollections:
    """Collections associated with a specific agent - SIMPLIFIED."""
    agent_id: str
    knowledge_collection: str      # kb_agent_{id}
    short_memory_collection: str   # memory_short_{id}
    long_memory_collection: str    # memory_long_{id}
    created_at: datetime
    last_accessed: datetime

    @classmethod
    def create_for_agent(cls, agent_id: str) -> "AgentCollections":
        """Create standard collections for an agent."""
        now = datetime.utcnow()
        return cls(
            agent_id=agent_id,
            knowledge_collection=f"kb_agent_{agent_id}",
            short_memory_collection=f"memory_short_{agent_id}",
            long_memory_collection=f"memory_long_{agent_id}",
            created_at=now,
            last_accessed=now
        )


class UnifiedRAGConfig(BaseModel):
    """Streamlined configuration for the unified RAG system."""
    # Core settings - SIMPLIFIED
    persist_directory: str = Field(default="./data/chroma")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Performance settings - OPTIMIZED
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=128)

    # Agent isolation - STRICT BY DEFAULT
    auto_create_collections: bool = Field(default=True)
    strict_isolation: bool = Field(default=True)

    # Memory settings - NEW
    short_term_ttl_hours: int = Field(default=24)  # Short-term memory TTL
    long_term_max_items: int = Field(default=10000)  # Long-term memory limit


class UnifiedRAGSystem:
    """
    Unified RAG System - Single source of truth for all RAG operations.
    
    This system manages:
    - Single ChromaDB instance with multiple collections
    - Agent-specific knowledge bases and memory
    - Shared knowledge collections
    - Access control and isolation
    - Performance optimization
    """
    
    def __init__(self, config: Optional[UnifiedRAGConfig] = None):
        """Initialize the unified RAG system."""
        self.config = config or UnifiedRAGConfig()
        
        # Core components
        self.chroma_client = None
        self.embedding_function = None

        # Agent management
        self.agent_collections: Dict[str, AgentCollections] = {}
        
        # Performance tracking
        self.stats = {
            "total_agents": 0,
            "total_collections": 0,
            "total_queries": 0,
            "total_documents": 0
        }

        self.is_initialized = False
        logger.info("Unified RAG System initialized")
    
    async def initialize(self) -> None:
        """Initialize the unified RAG system."""
        try:
            if self.is_initialized:
                logger.warning("Unified RAG system already initialized")
                return

            if not chromadb:
                raise RuntimeError("ChromaDB not available")

            logger.info("Initializing unified RAG system...")

            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.persist_directory
            )

            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )

            self.is_initialized = True
            logger.info("Unified RAG system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize unified RAG system: {str(e)}")
            raise
    
    def _get_collection(self, collection_name: str):
        """Get or create a collection."""
        try:
            return self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Failed to get/create collection {collection_name}: {str(e)}")
            raise

    async def create_agent_ecosystem(self, agent_id: str) -> AgentCollections:
        """
        Create isolated collections for an agent - SIMPLIFIED APPROACH.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            AgentCollections object with collection names
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            if agent_id in self.agent_collections:
                logger.warning(f"Agent ecosystem already exists for {agent_id}")
                return self.agent_collections[agent_id]

            # Create agent collections using simplified naming
            agent_collections = AgentCollections.create_for_agent(agent_id)

            # Create collections (ChromaDB will create them on first use)
            self._get_collection(agent_collections.knowledge_collection)
            self._get_collection(agent_collections.short_memory_collection)
            self._get_collection(agent_collections.long_memory_collection)

            # Store agent collections
            self.agent_collections[agent_id] = agent_collections
            self.stats["total_agents"] += 1
            self.stats["total_collections"] += 3  # KB + short memory + long memory

            logger.info(f"Created agent ecosystem for {agent_id} with collections: {agent_collections.knowledge_collection}, {agent_collections.short_memory_collection}, {agent_collections.long_memory_collection}")
            return agent_collections

        except Exception as e:
            logger.error(f"Failed to create agent ecosystem for {agent_id}: {str(e)}")
            raise

    async def get_agent_collections(self, agent_id: str) -> Optional[AgentCollections]:
        """Get collections for a specific agent."""
        return self.agent_collections.get(agent_id)

    async def search_agent_knowledge(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search knowledge for a specific agent.

        Args:
            agent_id: Agent performing the search
            query: Search query
            top_k: Number of results to return
            filters: Additional metadata filters

        Returns:
            List of matching documents
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get or create agent collections
            agent_collections = await self.get_agent_collections(agent_id)
            if not agent_collections:
                agent_collections = await self.create_agent_ecosystem(agent_id)

            # Get the knowledge collection
            collection = self._get_collection(agent_collections.knowledge_collection)

            # Perform search
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters
            )

            # Convert to Document objects
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append(Document(
                        id=results['ids'][0][i] if results['ids'] else f"doc_{i}",
                        content=doc,
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    ))

            self.stats["total_queries"] += 1
            return documents

        except Exception as e:
            logger.error(f"Failed to search knowledge for agent {agent_id}: {str(e)}")
            raise

    async def search_documents(
        self,
        agent_id: str,
        query: str,
        collection_type: str = "knowledge",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        🚀 HYBRID RAG INTERFACE: Search documents with compatibility layer.

        This method provides a standardized interface for the hybrid RAG system
        while maintaining your revolutionary architecture underneath.

        Args:
            agent_id: Agent performing the search
            query: Search query
            collection_type: Type of collection to search ("knowledge" or "memory")
            top_k: Number of results to return
            filters: Additional metadata filters

        Returns:
            List of search results in hybrid-compatible format
        """
        try:
            # Route to appropriate search method based on collection type
            if collection_type == "memory":
                documents = await self.search_agent_memory(agent_id, query, "both", top_k, filters)
            else:  # Default to knowledge
                documents = await self.search_agent_knowledge(agent_id, query, top_k, filters)

            # Convert to hybrid-compatible format
            results = []
            for doc in documents:
                result = {
                    'content': doc.content,
                    'metadata': doc.metadata or {},
                    'score': 1.0,  # ChromaDB doesn't return scores by default
                    'id': doc.id
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to search documents for agent {agent_id}: {str(e)}")
            return []

    async def add_documents(
        self,
        agent_id: str,
        documents: List[Document],
        collection_type: str = "knowledge"
    ) -> bool:
        """
        Add documents to an agent's collection.

        Args:
            agent_id: Agent identifier
            documents: List of documents to add
            collection_type: Type of collection ("knowledge" or "memory")

        Returns:
            True if successful
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get or create agent collections
            agent_collections = await self.get_agent_collections(agent_id)
            if not agent_collections:
                agent_collections = await self.create_agent_ecosystem(agent_id)

            # Select collection based on type - SIMPLIFIED
            if collection_type == "memory_short":
                collection_name = agent_collections.short_memory_collection
            elif collection_type == "memory_long":
                collection_name = agent_collections.long_memory_collection
            else:  # Default to knowledge
                collection_name = agent_collections.knowledge_collection

            collection = self._get_collection(collection_name)

            # Prepare data for ChromaDB
            ids = [doc.id for doc in documents]
            texts = [doc.content for doc in documents]
            metadatas = [doc.metadata or {} for doc in documents]

            # Add timestamp to metadata
            for metadata in metadatas:
                metadata["added_at"] = datetime.utcnow().isoformat()
                metadata["agent_id"] = agent_id

            # Add documents
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            self.stats["total_documents"] += len(documents)
            logger.info(f"Added {len(documents)} documents to {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents for agent {agent_id}: {str(e)}")
            raise

    async def delete_agent_ecosystem(self, agent_id: str) -> bool:
        """
        Delete all collections for an agent - CLEAN REMOVAL.

        Args:
            agent_id: Agent identifier

        Returns:
            True if successful
        """
        try:
            if agent_id not in self.agent_collections:
                logger.warning(f"No ecosystem found for agent {agent_id}")
                return True

            agent_collections = self.agent_collections[agent_id]

            # Delete all agent collections
            collections_to_delete = [
                agent_collections.knowledge_collection,
                agent_collections.short_memory_collection,
                agent_collections.long_memory_collection
            ]

            for collection_name in collections_to_delete:
                try:
                    self.chroma_client.delete_collection(collection_name)
                    logger.debug(f"Deleted collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"Error deleting collection {collection_name}: {str(e)}")

            # Remove from tracking
            del self.agent_collections[agent_id]
            self.stats["total_agents"] -= 1
            self.stats["total_collections"] -= 3

            logger.info(f"Deleted agent ecosystem for {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete agent ecosystem for {agent_id}: {str(e)}")
            return False

    async def search_agent_memory(
        self,
        agent_id: str,
        query: str,
        memory_type: str = "both",  # "short", "long", or "both"
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search an agent's memory collections - SIMPLIFIED.

        Args:
            agent_id: Agent to search memory for
            query: Search query
            memory_type: Type of memory to search ("short", "long", or "both")
            top_k: Number of results to return
            filters: Additional metadata filters

        Returns:
            List of memory documents
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get agent collections
            agent_collections = await self.get_agent_collections(agent_id)
            if not agent_collections:
                agent_collections = await self.create_agent_ecosystem(agent_id)

            all_documents = []

            # Search short-term memory
            if memory_type in ["short", "both"]:
                short_collection = self._get_collection(agent_collections.short_memory_collection)
                short_results = short_collection.query(
                    query_texts=[query],
                    n_results=top_k // 2 if memory_type == "both" else top_k,
                    where=filters
                )

                if short_results['documents'] and short_results['documents'][0]:
                    for i, doc in enumerate(short_results['documents'][0]):
                        all_documents.append(Document(
                            id=short_results['ids'][0][i],
                            content=doc,
                            metadata={
                                **(short_results['metadatas'][0][i] if short_results['metadatas'] else {}),
                                "memory_type": "short_term"
                            }
                        ))

            # Search long-term memory
            if memory_type in ["long", "both"]:
                long_collection = self._get_collection(agent_collections.long_memory_collection)
                long_results = long_collection.query(
                    query_texts=[query],
                    n_results=top_k // 2 if memory_type == "both" else top_k,
                    where=filters
                )

                if long_results['documents'] and long_results['documents'][0]:
                    for i, doc in enumerate(long_results['documents'][0]):
                        all_documents.append(Document(
                            id=long_results['ids'][0][i],
                            content=doc,
                            metadata={
                                **(long_results['metadatas'][0][i] if long_results['metadatas'] else {}),
                                "memory_type": "long_term"
                            }
                        ))

            self.stats["total_queries"] += 1
            return all_documents[:top_k]  # Limit to top_k results

        except Exception as e:
            logger.error(f"Failed to search memory for agent {agent_id}: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "active_agents": list(self.agent_collections.keys()),
            "config": {
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "strict_isolation": self.config.strict_isolation
            }
        }

    async def cleanup_expired_memories(self, agent_id: str) -> int:
        """
        Clean up expired short-term memories for an agent.

        Args:
            agent_id: Agent to clean up memories for

        Returns:
            Number of memories cleaned up
        """
        try:
            agent_collections = await self.get_agent_collections(agent_id)
            if not agent_collections:
                return 0

            # Get short-term memory collection
            collection = self._get_collection(agent_collections.short_memory_collection)

            # Calculate expiry time
            expiry_time = datetime.utcnow() - timedelta(hours=self.config.short_term_ttl_hours)

            # Query for expired memories
            results = collection.get(
                where={"added_at": {"$lt": expiry_time.isoformat()}}
            )

            if results['ids']:
                # Delete expired memories
                collection.delete(ids=results['ids'])
                logger.info(f"Cleaned up {len(results['ids'])} expired memories for agent {agent_id}")
                return len(results['ids'])

            return 0

        except Exception as e:
            logger.error(f"Failed to cleanup memories for agent {agent_id}: {str(e)}")
            return 0

    async def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for a specific agent."""
        try:
            agent_collections = await self.get_agent_collections(agent_id)
            if not agent_collections:
                return {"error": "Agent not found"}

            stats = {
                "agent_id": agent_id,
                "collections": {
                    "knowledge": agent_collections.knowledge_collection,
                    "short_memory": agent_collections.short_memory_collection,
                    "long_memory": agent_collections.long_memory_collection
                },
                "created_at": agent_collections.created_at.isoformat(),
                "last_accessed": agent_collections.last_accessed.isoformat()
            }

            # Get collection counts
            for collection_type, collection_name in stats["collections"].items():
                try:
                    collection = self._get_collection(collection_name)
                    count_result = collection.count()
                    stats[f"{collection_type}_count"] = count_result
                except Exception as e:
                    stats[f"{collection_type}_count"] = 0
                    logger.warning(f"Could not get count for {collection_name}: {str(e)}")

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats for agent {agent_id}: {str(e)}")
            return {"error": str(e)}


