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

# Vector database factory
from app.rag.core.vector_db_factory import get_vector_db_client, VectorItem, SearchResult, GetResult, VectorDBBase

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


class VectorCollectionWrapper:
    """Wrapper to provide ChromaDB-like interface for other vector databases."""

    def __init__(self, collection_name: str, vector_client: VectorDBBase):
        self.name = collection_name
        self.vector_client = vector_client

    def add(self, ids: List[str], documents: List[str], metadatas: List[dict] = None):
        """Add documents to the collection."""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        # For non-ChromaDB clients, we need to generate embeddings ourselves
        # This is a simplified approach - in production you'd want proper embedding generation
        items = []
        for i, (doc_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
            # Create a simple vector (this should be replaced with actual embeddings)
            vector = [0.0] * 384  # Default dimension
            items.append(VectorItem(
                id=doc_id,
                vector=vector,
                document=document,
                metadata=metadata
            ))

        return self.vector_client.add(self.name, items)

    def query(self, query_texts: List[str], n_results: int = 10):
        """Query the collection."""
        # For non-ChromaDB clients, we need to generate query embeddings
        # This is a simplified approach
        query_vectors = [[0.0] * 384 for _ in query_texts]  # Should be actual embeddings

        result = self.vector_client.search(self.name, query_vectors, n_results)
        if result:
            return {
                'ids': result.ids,
                'distances': result.distances,
                'documents': result.documents,
                'metadatas': result.metadatas
            }
        return {'ids': [[]], 'distances': [[]], 'documents': [[]], 'metadatas': [[]]}

    def get(self):
        """Get all documents from the collection."""
        result = self.vector_client.get(self.name)
        if result:
            return {
                'ids': result.ids,
                'documents': result.documents,
                'metadatas': result.metadatas
            }
        return {'ids': [[]], 'documents': [[]], 'metadatas': [[]]}

    def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        return self.vector_client.delete(self.name, ids)


class UnifiedRAGSystem:
    """
    Unified RAG System - Single source of truth for all RAG operations.

    This system manages:
    - Configurable vector database (ChromaDB, pgvector, etc.)
    - Agent-specific knowledge bases and memory
    - Shared knowledge collections
    - Access control and isolation
    - Performance optimization
    """

    def __init__(self, config: Optional[UnifiedRAGConfig] = None):
        """Initialize the unified RAG system."""
        self.config = config or UnifiedRAGConfig()

        # Core components
        self.vector_client = None
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
        self._config_lock = asyncio.Lock()  # For thread-safe config updates
        logger.info("Unified RAG System initialized")
    
    async def initialize(self) -> None:
        """Initialize the unified RAG system."""
        try:
            if self.is_initialized:
                logger.warning("Unified RAG system already initialized")
                return

            logger.info("Initializing unified RAG system...")

            # Initialize vector database client (will auto-detect type from config)
            self.vector_client = get_vector_db_client()
            logger.info("Vector database client initialized", type=type(self.vector_client).__name__)

            # Initialize embedding function and ChromaDB client if needed
            if CHROMADB_AVAILABLE and hasattr(self.vector_client, '__class__') and 'Chroma' in self.vector_client.__class__.__name__:
                # For ChromaDB, use the existing client from vector_client to avoid conflicts
                self.chroma_client = self.vector_client.client
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedding_model
                )
                logger.info("ChromaDB client reused from vector client to avoid conflicts")
            else:
                # For other vector databases, we'll handle embeddings differently
                self.chroma_client = None
                logger.info("Using external embedding handling for non-ChromaDB vector database")

            self.is_initialized = True
            logger.info("Unified RAG system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize unified RAG system: {str(e)}")
            raise
    
    def _get_collection(self, collection_name: str):
        """Get or create a collection."""
        try:
            # For ChromaDB compatibility, we still use the direct client if available
            if hasattr(self, 'chroma_client') and self.chroma_client:
                return self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )

            # For other vector databases, ensure collection exists
            if not self.vector_client.has_collection(collection_name):
                logger.info(f"Collection {collection_name} will be created on first use")

            # Return a collection wrapper that works with our vector client
            return VectorCollectionWrapper(collection_name, self.vector_client)

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
                    # For ChromaDB compatibility
                    if hasattr(self, 'chroma_client') and self.chroma_client:
                        self.chroma_client.delete_collection(collection_name)
                    else:
                        # For other vector databases
                        self.vector_client.delete_collection(collection_name)
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

    # ============================================================================
    # 🚀 REVOLUTIONARY DYNAMIC RECONFIGURATION SYSTEM
    # ============================================================================

    async def update_configuration(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        🚀 Revolutionary method to update RAG system configuration in real-time
        without requiring server restart.

        Args:
            new_config: Dictionary of configuration updates

        Returns:
            Dictionary with update results and any warnings
        """
        async with self._config_lock:
            try:
                logger.info("🔄 Starting dynamic RAG configuration update...")

                # Track what was updated
                updates_applied = []
                warnings = []

                # Create new config with updates
                current_config_dict = self.config.dict()
                current_config_dict.update(new_config)

                # Validate new configuration
                try:
                    new_rag_config = UnifiedRAGConfig(**current_config_dict)
                except Exception as e:
                    logger.error(f"❌ Invalid configuration: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Invalid configuration: {str(e)}",
                        "updates_applied": [],
                        "warnings": []
                    }

                # Check if embedding model changed
                if new_rag_config.embedding_model != self.config.embedding_model:
                    await self._update_embedding_model(new_rag_config.embedding_model)
                    updates_applied.append(f"embedding_model: {self.config.embedding_model} → {new_rag_config.embedding_model}")

                # Check if chunk settings changed
                if (new_rag_config.chunk_size != self.config.chunk_size or
                    new_rag_config.chunk_overlap != self.config.chunk_overlap):
                    updates_applied.append(f"chunk_size: {self.config.chunk_size} → {new_rag_config.chunk_size}")
                    updates_applied.append(f"chunk_overlap: {self.config.chunk_overlap} → {new_rag_config.chunk_overlap}")
                    warnings.append("Chunk size changes will apply to new documents only. Existing documents retain their original chunking.")

                # Check if retrieval settings changed
                if new_rag_config.top_k != self.config.top_k:
                    updates_applied.append(f"top_k: {self.config.top_k} → {new_rag_config.top_k}")

                if new_rag_config.batch_size != self.config.batch_size:
                    updates_applied.append(f"batch_size: {self.config.batch_size} → {new_rag_config.batch_size}")

                # Check if isolation settings changed
                if new_rag_config.strict_isolation != self.config.strict_isolation:
                    updates_applied.append(f"strict_isolation: {self.config.strict_isolation} → {new_rag_config.strict_isolation}")
                    if not new_rag_config.strict_isolation:
                        warnings.append("Disabling strict isolation may allow agents to access each other's data.")

                # Check if memory settings changed
                if new_rag_config.short_term_ttl_hours != self.config.short_term_ttl_hours:
                    updates_applied.append(f"short_term_ttl_hours: {self.config.short_term_ttl_hours} → {new_rag_config.short_term_ttl_hours}")

                if new_rag_config.long_term_max_items != self.config.long_term_max_items:
                    updates_applied.append(f"long_term_max_items: {self.config.long_term_max_items} → {new_rag_config.long_term_max_items}")

                # Apply the new configuration
                self.config = new_rag_config

                logger.info(f"✅ RAG configuration updated successfully. Applied {len(updates_applied)} changes.")

                return {
                    "success": True,
                    "message": f"RAG configuration updated successfully with {len(updates_applied)} changes",
                    "updates_applied": updates_applied,
                    "warnings": warnings,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"❌ Failed to update RAG configuration: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to update configuration: {str(e)}",
                    "updates_applied": [],
                    "warnings": []
                }

    async def _update_embedding_model(self, new_model: str) -> None:
        """Update the embedding model dynamically."""
        try:
            logger.info(f"🔄 Updating embedding model to: {new_model}")

            # Create new embedding function
            if new_model.startswith("openai/"):
                # OpenAI embedding model
                model_name = new_model.replace("openai/", "")
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    model_name=model_name
                )
            elif new_model.startswith("cohere/"):
                # Cohere embedding model
                model_name = new_model.replace("cohere/", "")
                self.embedding_function = embedding_functions.CohereEmbeddingFunction(
                    model_name=model_name
                )
            else:
                # Default to sentence transformers
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=new_model
                )

            logger.info(f"✅ Embedding model updated to: {new_model}")

        except Exception as e:
            logger.error(f"❌ Failed to update embedding model: {str(e)}")
            raise

    async def get_current_configuration(self) -> Dict[str, Any]:
        """Get the current RAG system configuration."""
        return {
            "config": self.config.dict(),
            "stats": self.stats,
            "is_initialized": self.is_initialized,
            "timestamp": datetime.now().isoformat()
        }

    async def reload_system(self) -> Dict[str, Any]:
        """
        🚀 Revolutionary method to completely reload the RAG system
        with current configuration (useful for major changes).
        """
        try:
            logger.info("🔄 Reloading RAG system...")

            # Store current config
            current_config = self.config

            # Reinitialize the system
            self.is_initialized = False
            await self.initialize()

            logger.info("✅ RAG system reloaded successfully")

            return {
                "success": True,
                "message": "RAG system reloaded successfully",
                "config": current_config.dict(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to reload RAG system: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to reload system: {str(e)}"
            }


