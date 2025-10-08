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

from pydantic import BaseModel, Field

# Import backend logging system
from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

# Get backend logger instance
_backend_logger = get_backend_logger()

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

# Advanced retrieval pipeline (optional)
try:
    from app.rag.retrieval.advanced_retrieval_pipeline import (
        AdvancedRetrievalPipeline,
        PipelineConfig,
        RetrievalMode
    )
    ADVANCED_RETRIEVAL_AVAILABLE = True
except ImportError:
    ADVANCED_RETRIEVAL_AVAILABLE = False
    AdvancedRetrievalPipeline = None
    PipelineConfig = None
    RetrievalMode = None

# Structured KB components (optional)
try:
    from app.rag.core.metadata_index import (
        get_metadata_index_manager,
        TermFilter,
        RangeFilter,
        MetadataIndexManager
    )
    from app.rag.core.chunk_relationship_manager import (
        get_chunk_relationship_manager,
        ChunkRelationshipManager
    )
    from app.rag.core.multimodal_indexer import (
        get_multimodal_indexer,
        MultimodalIndexer,
        ContentType as MultimodalContentType
    )
    STRUCTURED_KB_AVAILABLE = True
except ImportError:
    STRUCTURED_KB_AVAILABLE = False
    get_metadata_index_manager = None
    get_chunk_relationship_manager = None
    get_multimodal_indexer = None


# Data classes for compatibility with existing code
@dataclass
class Document:
    """
    Document representation for RAG system.

    Enhanced to support the new ingestion pipeline with proper
    metadata fields and deduplication support.
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    # Enhanced fields for new pipeline
    title: Optional[str] = None
    document_type: Optional[str] = None
    source: Optional[str] = None
    chunk_count: int = 0

    # Hashes for deduplication
    content_sha: Optional[str] = None
    norm_text_sha: Optional[str] = None

    # Processing metadata
    language: str = "unknown"
    confidence: float = 1.0
    processor_name: Optional[str] = None
    processing_time_ms: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class DocumentChunk:
    """
    Document chunk representation for RAG system.

    Enhanced to support semantic chunking, deduplication, and
    advanced retrieval features.
    """
    id: str
    content: str
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    # Hashes for deduplication
    content_sha: Optional[str] = None
    norm_text_sha: Optional[str] = None

    # Provenance
    source_uri: Optional[str] = None
    section_path: Optional[str] = None
    page: Optional[int] = None

    # Content metadata
    lang: str = "en"
    char_count: int = 0
    token_count: Optional[int] = None

    # Semantic metadata
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Access control
    labels: Dict[str, str] = field(default_factory=dict)

    # Versioning
    ts_ingested: Optional[datetime] = None
    version: int = 1
    embedding_model: Optional[str] = None


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
    """Wrapper to provide ChromaDB-like interface for other vector databases with REAL embeddings."""

    def __init__(self, collection_name: str, vector_client: VectorDBBase, embedding_manager=None):
        self.name = collection_name
        self.vector_client = vector_client
        self.embedding_manager = embedding_manager

        # Initialize fallback embedding function if no manager provided
        if not self.embedding_manager:
            try:
                from sentence_transformers import SentenceTransformer
                self._fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
                _backend_logger.info(
                    "✅ Fallback SentenceTransformer model loaded for VectorCollectionWrapper",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
            except ImportError:
                _backend_logger.warn(
                    "⚠️ No embedding manager and SentenceTransformers unavailable - using hash-based fallback",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
                self._fallback_model = None

    async def add(self, ids: List[str], documents: List[str], metadatas: List[dict] = None):
        """Add documents to the collection with REAL embeddings."""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        # Generate REAL embeddings
        try:
            if self.embedding_manager:
                # Use the sophisticated embedding manager
                from .embeddings import EmbeddingType
                embeddings = await self.embedding_manager.generate_embeddings(
                    texts=documents,
                    embedding_type=EmbeddingType.DENSE
                )
                vectors = embeddings
            elif self._fallback_model:
                # Use fallback SentenceTransformer
                vectors = self._fallback_model.encode(documents).tolist()
            else:
                # Hash-based fallback (better than zeros!)
                import hashlib
                vectors = []
                for doc in documents:
                    # Create deterministic embedding from document hash
                    doc_hash = hashlib.sha256(doc.encode()).hexdigest()
                    # Convert hex to normalized floats
                    embedding = []
                    for i in range(0, min(len(doc_hash), 96), 2):  # 384/8 = 48 pairs * 2 = 96 chars
                        val = int(doc_hash[i:i+2], 16) / 255.0  # Normalize to 0-1
                        embedding.extend([val] * 8)  # Repeat to get 384 dimensions
                    # Pad to exactly 384 dimensions
                    while len(embedding) < 384:
                        embedding.append(0.0)
                    vectors.append(embedding[:384])

            _backend_logger.debug(
                f"Generated {len(vectors)} real embeddings for documents",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to generate embeddings: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            # Emergency fallback - still better than all zeros
            vectors = [[0.1] * 384 for _ in documents]  # At least not all zeros

        # Create vector items
        items = []
        for i, (doc_id, document, metadata, vector) in enumerate(zip(ids, documents, metadatas, vectors)):
            items.append(VectorItem(
                id=doc_id,
                vector=vector,
                document=document,
                metadata=metadata
            ))

        return self.vector_client.add(self.name, items)

    async def query(self, query_texts: List[str], n_results: int = 10, where: dict = None):
        """Query the collection with REAL query embeddings."""
        try:
            # Generate REAL query embeddings
            if self.embedding_manager:
                from .embeddings import EmbeddingType
                query_embeddings = await self.embedding_manager.generate_embeddings(
                    texts=query_texts,
                    embedding_type=EmbeddingType.DENSE
                )
                query_vectors = query_embeddings
            elif self._fallback_model:
                query_vectors = self._fallback_model.encode(query_texts).tolist()
            else:
                # Hash-based fallback for queries too
                import hashlib
                query_vectors = []
                for query in query_texts:
                    query_hash = hashlib.sha256(query.encode()).hexdigest()
                    embedding = []
                    for i in range(0, min(len(query_hash), 96), 2):
                        val = int(query_hash[i:i+2], 16) / 255.0
                        embedding.extend([val] * 8)
                    while len(embedding) < 384:
                        embedding.append(0.0)
                    query_vectors.append(embedding[:384])

            result = self.vector_client.search(self.name, query_vectors, n_results)
            if result:
                # Convert distances to similarity scores (1 - distance)
                scores = []
                if result.distances:
                    for distance_list in result.distances:
                        score_list = [max(0.0, 1.0 - dist) for dist in distance_list]
                        scores.append(score_list)
                else:
                    scores = [[1.0] * len(result.ids[0])] if result.ids else [[]]

                return {
                    'ids': result.ids,
                    'distances': result.distances,
                    'documents': result.documents,
                    'metadatas': result.metadatas,
                    'scores': scores  # Add real similarity scores
                }
        except Exception as e:
            _backend_logger.error(
                f"Query failed: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

        return {'ids': [[]], 'distances': [[]], 'documents': [[]], 'metadatas': [[]], 'scores': [[]]}

    def get(self):
        """Get all documents from the collection."""
        result = self.vector_client.get(self.name)
        if result:
            return {
                'ids': result.ids,
                'documents': result.documents,
                'metadatas': result.metadatas
            }
        return {'ids': [], 'documents': [], 'metadatas': []}

    def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        return self.vector_client.delete(self.name, ids)

    def count(self):
        """Get document count in collection."""
        try:
            result = self.get()
            return len(result.get('ids', []))
        except:
            return 0


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
        self.embedding_manager = None  # NEW: Integrated embedding manager
        self.advanced_retrieval_pipeline: Optional[Any] = None  # Advanced retrieval pipeline

        # Structured KB components (optional)
        self.metadata_index_manager: Optional[Any] = None
        self.chunk_relationship_manager: Optional[Any] = None
        self.multimodal_indexer: Optional[Any] = None

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
        _backend_logger.info(
            "Unified RAG System initialized",
            LogCategory.RAG_OPERATIONS,
            "app.rag.core.unified_rag_system"
        )

    async def initialize(self) -> None:
        """Initialize the unified RAG system with integrated embedding manager."""
        try:
            if self.is_initialized:
                _backend_logger.warn(
                    "Unified RAG system already initialized",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
                return

            _backend_logger.info(
                "Initializing unified RAG system...",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # ========================================================================
            # PHASE 0: ENSURE DEFAULT MODELS ARE AVAILABLE
            # ========================================================================
            # Silently ensure the 3 default models are downloaded:
            # 1. all-MiniLM-L6-v2 (embedding)
            # 2. bge-reranker-base (reranking)
            # 3. clip-ViT-B-32 (vision)

            try:
                from app.rag.core.model_initialization_service import (
                    get_model_initialization_service,
                    ModelInitializationConfig
                )

                # Create model initialization service with silent mode
                model_init_config = ModelInitializationConfig(
                    check_huggingface_cache=True,
                    reuse_cached_models=True,
                    copy_to_centralized=True,
                    validate_models=True,
                    auto_download_defaults=True,  # Download 3 default models
                    silent_mode=True  # No CLI output
                )

                model_service = await get_model_initialization_service(model_init_config)

                # Ensure default models are available (silent)
                model_locations = await model_service.ensure_models_available()

                # Log simple success message
                stats = model_service.get_statistics()
                _backend_logger.info(
                    "Default models ready",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system",
                    data={
                        "found": stats['models_found'],
                        "downloaded": stats['models_downloaded'],
                        "reused": stats['models_reused']
                    }
                )

            except Exception as e:
                _backend_logger.warn(
                    f"Model initialization failed: {e}. Models will be downloaded on first use.",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

            # ========================================================================
            # PHASE 1: INITIALIZE VECTOR DATABASE
            # ========================================================================
            _backend_logger.info(
                "Phase 1: Initializing vector database...",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Initialize vector database client (will auto-detect type from config)
            self.vector_client = get_vector_db_client()
            _backend_logger.info(
                "Vector database client initialized",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system",
                data={"type": type(self.vector_client).__name__}
            )

            # ========================================================================
            # PHASE 2: INITIALIZE EMBEDDING MANAGER
            # ========================================================================
            _backend_logger.info(
                "Phase 2: Initializing embedding manager...",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Initialize the revolutionary embedding manager
            try:
                from .embeddings import EmbeddingManager, EmbeddingConfig, EmbeddingType

                embedding_config = EmbeddingConfig(
                    model_name=self.config.embedding_model,
                    embedding_type=EmbeddingType.DENSE,  # Default to dense, can be changed per operation
                    batch_size=self.config.batch_size,
                    cache_embeddings=True,
                    use_model_manager=True  # ALWAYS use model manager for centralized storage
                )

                self.embedding_manager = EmbeddingManager(embedding_config)
                await self.embedding_manager.initialize()
                _backend_logger.info(
                    "✅ Revolutionary embedding manager initialized successfully",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

            except Exception as e:
                _backend_logger.warn(
                    f"Failed to initialize embedding manager: {str(e)}, using fallback",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
                self.embedding_manager = None

            # ========================================================================
            # PHASE 3: INITIALIZE CHROMADB EMBEDDING FUNCTION
            # ========================================================================
            _backend_logger.info(
                "Phase 3: Initializing ChromaDB embedding function...",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Initialize embedding function and ChromaDB client if needed
            if CHROMADB_AVAILABLE and hasattr(self.vector_client, '__class__') and 'Chroma' in self.vector_client.__class__.__name__:
                # For ChromaDB, use the existing client from vector_client to avoid conflicts
                self.chroma_client = self.vector_client.client
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedding_model
                )
                _backend_logger.info(
                    "ChromaDB client reused from vector client to avoid conflicts",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
            else:
                # For other vector databases, we'll handle embeddings through our manager
                self.chroma_client = None
                _backend_logger.info(
                    "Using embedding manager for non-ChromaDB vector database",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

            # ========================================================================
            # PHASE 4: INITIALIZE ADVANCED RETRIEVAL PIPELINE
            # ========================================================================
            _backend_logger.info(
                "Phase 4: Initializing advanced retrieval pipeline...",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Initialize advanced retrieval pipeline if available
            if ADVANCED_RETRIEVAL_AVAILABLE:
                try:
                    pipeline_config = PipelineConfig(
                        mode=RetrievalMode.ADVANCED,
                        enable_query_expansion=True,
                        enable_bm25=True,
                        enable_reranking=True,
                        enable_mmr=True,
                        enable_compression=False,  # Disabled by default
                        initial_top_k=100,
                        final_top_k=10
                    )
                    self.advanced_retrieval_pipeline = AdvancedRetrievalPipeline(pipeline_config)
                    await self.advanced_retrieval_pipeline.initialize()
                    _backend_logger.info(
                        "✅ Advanced retrieval pipeline initialized successfully",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                except Exception as e:
                    _backend_logger.warn(
                        f"Failed to initialize advanced retrieval pipeline: {str(e)}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                    self.advanced_retrieval_pipeline = None
            else:
                _backend_logger.info(
                    "Advanced retrieval pipeline not available (optional dependency)",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
                self.advanced_retrieval_pipeline = None

            # ========================================================================
            # PHASE 5: INITIALIZE STRUCTURED KB COMPONENTS
            # ========================================================================
            _backend_logger.info(
                "Phase 5: Initializing structured KB components...",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Initialize structured KB components if available
            if STRUCTURED_KB_AVAILABLE:
                try:
                    self.metadata_index_manager = await get_metadata_index_manager()
                    self.chunk_relationship_manager = await get_chunk_relationship_manager()
                    self.multimodal_indexer = await get_multimodal_indexer()
                    _backend_logger.info(
                        "✅ Structured KB components initialized successfully",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                except Exception as e:
                    _backend_logger.warn(
                        f"Failed to initialize structured KB components: {str(e)}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                    self.metadata_index_manager = None
                    self.chunk_relationship_manager = None
                    self.multimodal_indexer = None
            else:
                _backend_logger.info(
                    "Structured KB components not available (optional dependency)",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

            # ========================================================================
            # INITIALIZATION COMPLETE
            # ========================================================================
            self.is_initialized = True
            _backend_logger.info(
                "=" * 80,
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            _backend_logger.info(
                "✅ UNIFIED RAG SYSTEM INITIALIZED SUCCESSFULLY",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            _backend_logger.info(
                "=" * 80,
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            _backend_logger.info(
                f"   Vector DB: {type(self.vector_client).__name__}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            _backend_logger.info(
                f"   Embedding Model: {self.config.embedding_model}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            _backend_logger.info(
                f"   Advanced Retrieval: {'Enabled' if self.advanced_retrieval_pipeline else 'Disabled'}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            _backend_logger.info(
                f"   Structured KB: {'Enabled' if self.metadata_index_manager else 'Disabled'}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            _backend_logger.info(
                "=" * 80,
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize unified RAG system: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            raise
    
    def _get_collection(self, collection_name: str):
        """Get or create a collection with proper embedding support."""
        try:
            # For ChromaDB compatibility, we still use the direct client if available
            if hasattr(self, 'chroma_client') and self.chroma_client:
                return self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )

            # For other vector databases, ensure collection exists
            if not self.vector_client.has_collection(collection_name):
                _backend_logger.info(
                    f"Collection {collection_name} will be created on first use",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

            # Return a collection wrapper with embedding manager integration
            return VectorCollectionWrapper(
                collection_name=collection_name,
                vector_client=self.vector_client,
                embedding_manager=self.embedding_manager  # Pass the embedding manager!
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to get/create collection {collection_name}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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
                _backend_logger.warn(
                    f"Agent ecosystem already exists for {agent_id}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
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

            _backend_logger.info(
                f"Created agent ecosystem for {agent_id} with collections: {agent_collections.knowledge_collection}, {agent_collections.short_memory_collection}, {agent_collections.long_memory_collection}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            return agent_collections

        except Exception as e:
            _backend_logger.error(
                f"Failed to create agent ecosystem for {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            raise

    async def get_agent_collections(self, agent_id: str) -> Optional[AgentCollections]:
        """Get collections for a specific agent."""
        return self.agent_collections.get(agent_id)

    async def search_agent_knowledge(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "dense",
        use_advanced_retrieval: bool = True
    ) -> List[Document]:
        """
        Search knowledge for a specific agent with advanced search types.

        Args:
            agent_id: Agent performing the search
            query: Search query
            top_k: Number of results to return
            filters: Additional metadata filters
            search_type: Type of search ("dense", "sparse", "hybrid", "vision", "advanced")
            use_advanced_retrieval: Use advanced retrieval pipeline if available

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

            # Use advanced retrieval pipeline if available and requested
            if (use_advanced_retrieval and
                search_type in ["advanced", "hybrid"] and
                self.advanced_retrieval_pipeline is not None):

                _backend_logger.info(
                    f"Using advanced retrieval pipeline for agent {agent_id}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

                # Create dense retriever function for the pipeline
                async def dense_retriever(q: str, k: int) -> List[Dict[str, Any]]:
                    """Dense retrieval function for advanced pipeline."""
                    if hasattr(collection, 'query') and asyncio.iscoroutinefunction(collection.query):
                        results = await collection.query(
                            query_texts=[q],
                            n_results=k,
                            where=filters
                        )
                    else:
                        results = collection.query(
                            query_texts=[q],
                            n_results=k,
                            where=filters
                        )

                    # Convert to standard format
                    docs = []
                    if results['documents'] and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0]):
                            metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                            doc_id = results['ids'][0][i] if results['ids'] and results['ids'][0] else f"doc_{i}"

                            # Get score
                            score = 1.0
                            if 'scores' in results and results['scores'] and results['scores'][0]:
                                score = results['scores'][0][i] if i < len(results['scores'][0]) else 1.0
                            elif 'distances' in results and results['distances'] and results['distances'][0]:
                                dist = results['distances'][0][i] if i < len(results['distances'][0]) else 0.0
                                score = max(0.0, 1.0 - dist)

                            # Get embedding if available
                            embedding = metadata.get('embedding')

                            docs.append({
                                'doc_id': doc_id,
                                'content': doc,
                                'score': score,
                                'metadata': metadata,
                                'embedding': embedding
                            })

                    return docs

                # Get query embedding if available
                query_embedding = None
                if self.embedding_manager:
                    try:
                        from .embeddings import EmbeddingType
                        embeddings = await self.embedding_manager.generate_embeddings(
                            texts=[query],
                            embedding_type=EmbeddingType.DENSE
                        )
                        if embeddings and len(embeddings) > 0:
                            query_embedding = embeddings[0]
                    except Exception as e:
                        _backend_logger.warn(
                            f"Failed to generate query embedding: {e}",
                            LogCategory.RAG_OPERATIONS,
                            "app.rag.core.unified_rag_system"
                        )

                # Execute advanced retrieval
                try:
                    retrieval_results, metrics = await self.advanced_retrieval_pipeline.retrieve(
                        query=query,
                        dense_retriever=dense_retriever,
                        query_embedding=query_embedding,
                        top_k=top_k
                    )

                    # Convert to Document objects
                    documents = []
                    for result in retrieval_results:
                        metadata = result.metadata.copy()
                        metadata['similarity_score'] = result.score
                        metadata['rank'] = result.rank

                        # Add pipeline details
                        if result.dense_score is not None:
                            metadata['dense_score'] = result.dense_score
                        if result.sparse_score is not None:
                            metadata['sparse_score'] = result.sparse_score
                        if result.rerank_score is not None:
                            metadata['rerank_score'] = result.rerank_score
                        if result.mmr_score is not None:
                            metadata['mmr_score'] = result.mmr_score
                        if result.diversity_score is not None:
                            metadata['diversity_score'] = result.diversity_score

                        documents.append(Document(
                            id=result.doc_id,
                            content=result.content,
                            metadata=metadata
                        ))

                    _backend_logger.info(
                        "Advanced retrieval completed",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system",
                        data={
                            "agent_id": agent_id,
                            "results_count": len(documents),
                            "total_time_ms": metrics.total_time_ms
                        }
                    )

                    self.stats["total_queries"] += 1
                    return documents

                except Exception as e:
                    _backend_logger.error(
                        f"Advanced retrieval failed: {e}, falling back to basic search",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                    # Fall through to basic search

            # Configure search based on type
            if search_type in ["sparse", "hybrid"] and self.embedding_manager:
                # Use embedding manager for advanced search types
                try:
                    from .embeddings import EmbeddingType
                    if search_type == "sparse":
                        embedding_type = EmbeddingType.SPARSE
                    elif search_type == "hybrid":
                        embedding_type = EmbeddingType.HYBRID
                    else:
                        embedding_type = EmbeddingType.DENSE

                    # Generate embeddings with specific type
                    query_embeddings = await self.embedding_manager.generate_embeddings(
                        texts=[query],
                        embedding_type=embedding_type
                    )
                    _backend_logger.debug(
                        f"Generated {search_type} embeddings for query",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                except Exception as e:
                    _backend_logger.warn(
                        f"Failed to generate {search_type} embeddings: {str(e)}, falling back to dense",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                    search_type = "dense"

            # Perform search
            if hasattr(collection, 'query') and asyncio.iscoroutinefunction(collection.query):
                # Async query for VectorCollectionWrapper
                results = await collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=filters
                )
            else:
                # Sync query for ChromaDB
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=filters
                )

            # Convert to Document objects with real similarity scores
            documents = []
            if results['documents'] and results['documents'][0]:
                # Get similarity scores (prefer scores over distances)
                scores = []
                if 'scores' in results and results['scores'] and results['scores'][0]:
                    scores = results['scores'][0]
                elif 'distances' in results and results['distances'] and results['distances'][0]:
                    # Convert distances to similarity scores (1 - normalized_distance)
                    distances = results['distances'][0]
                    max_distance = max(distances) if distances else 1.0
                    scores = [max(0.0, 1.0 - (dist / max_distance)) for dist in distances]
                else:
                    # Fallback scores
                    scores = [1.0] * len(results['documents'][0])

                for i, doc in enumerate(results['documents'][0]):
                    # Create document with metadata including similarity score
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    metadata['similarity_score'] = scores[i] if i < len(scores) else 1.0

                    documents.append(Document(
                        id=results['ids'][0][i] if results['ids'] and results['ids'][0] else f"doc_{i}",
                        content=doc,
                        metadata=metadata
                    ))

            self.stats["total_queries"] += 1
            return documents

        except Exception as e:
            _backend_logger.error(
                f"Failed to search knowledge for agent {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            raise

    async def search_agent_knowledge_structured(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        content_types: Optional[List[str]] = None,
        section_path: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        expand_context: bool = False,
        context_size: int = 2,
        use_advanced_retrieval: bool = True
    ) -> List[Document]:
        """
        Search knowledge with structured KB capabilities.

        Enhanced search with:
        - Content type filtering
        - Section path filtering
        - Page number filtering
        - Metadata-based filtering
        - Context expansion (surrounding chunks)
        - Parent document retrieval

        Args:
            agent_id: Agent performing the search
            query: Search query
            top_k: Number of results to return
            content_types: Filter by content types (e.g., ["text", "code", "table"])
            section_path: Filter by section path
            page_number: Filter by page number
            metadata_filters: Additional metadata filters
            expand_context: Whether to expand results with surrounding chunks
            context_size: Number of surrounding chunks to include
            use_advanced_retrieval: Use advanced retrieval pipeline

        Returns:
            List of matching documents
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Check if structured KB is available
            if not STRUCTURED_KB_AVAILABLE or not self.metadata_index_manager:
                _backend_logger.warn(
                    "Structured KB not available, falling back to basic search",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
                return await self.search_agent_knowledge(
                    agent_id=agent_id,
                    query=query,
                    top_k=top_k,
                    filters=metadata_filters,
                    use_advanced_retrieval=use_advanced_retrieval
                )

            # Build metadata filters
            term_filters = []
            range_filters = []

            if content_types:
                term_filters.append(TermFilter(
                    field='content_type',
                    values=content_types
                ))

            if section_path:
                term_filters.append(TermFilter(
                    field='section_path',
                    values=[section_path]
                ))

            if page_number is not None:
                range_filters.append(RangeFilter(
                    field='page_number',
                    min_value=page_number,
                    max_value=page_number
                ))

            # Add custom metadata filters
            if metadata_filters:
                for field, value in metadata_filters.items():
                    if isinstance(value, (list, tuple)):
                        term_filters.append(TermFilter(field=field, values=list(value)))
                    elif isinstance(value, dict) and ('min' in value or 'max' in value):
                        range_filters.append(RangeFilter(
                            field=field,
                            min_value=value.get('min'),
                            max_value=value.get('max')
                        ))
                    else:
                        term_filters.append(TermFilter(field=field, values=[value]))

            # Query metadata index to get candidate chunk IDs
            candidate_chunk_ids = self.metadata_index_manager.query(
                term_filters=term_filters,
                range_filters=range_filters
            )

            _backend_logger.debug(
                f"Metadata filtering found {len(candidate_chunk_ids)} candidates",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system",
                data={
                    "content_types": content_types,
                    "section_path": section_path,
                    "page_number": page_number
                }
            )

            # If no candidates, return empty
            if not candidate_chunk_ids:
                return []

            # Perform vector search with candidate filtering
            # Build filters for vector search
            vector_filters = metadata_filters.copy() if metadata_filters else {}

            # Perform basic search first
            all_results = await self.search_agent_knowledge(
                agent_id=agent_id,
                query=query,
                top_k=top_k * 3,  # Get more results for filtering
                filters=vector_filters,
                use_advanced_retrieval=use_advanced_retrieval
            )

            # Filter results to only include candidates from metadata index
            filtered_results = [
                doc for doc in all_results
                if doc.id in candidate_chunk_ids
            ][:top_k]

            # Expand context if requested
            if expand_context and self.chunk_relationship_manager:
                expanded_results = []
                seen_ids = set()

                for doc in filtered_results:
                    # Add the main document
                    if doc.id not in seen_ids:
                        expanded_results.append(doc)
                        seen_ids.add(doc.id)

                    # Get surrounding chunks
                    surrounding_ids = self.chunk_relationship_manager.get_surrounding_chunks(
                        chunk_id=doc.id,
                        context_size=context_size,
                        include_siblings=True
                    )

                    # Fetch surrounding chunks from vector DB
                    for chunk_id in surrounding_ids:
                        if chunk_id not in seen_ids:
                            # Get chunk from vector DB
                            agent_collections = await self.get_agent_collections(agent_id)
                            if agent_collections:
                                collection = self._get_collection(agent_collections.knowledge_collection)

                                # Query by ID
                                try:
                                    if hasattr(collection, 'get') and asyncio.iscoroutinefunction(collection.get):
                                        result = await collection.get(ids=[chunk_id])
                                    else:
                                        result = collection.get(ids=[chunk_id])

                                    if result and result.get('documents'):
                                        metadata = result['metadatas'][0] if result.get('metadatas') else {}
                                        metadata['is_context'] = True
                                        metadata['context_for'] = doc.id

                                        expanded_results.append(Document(
                                            id=chunk_id,
                                            content=result['documents'][0],
                                            metadata=metadata
                                        ))
                                        seen_ids.add(chunk_id)
                                except Exception as e:
                                    _backend_logger.warn(
                                        f"Failed to fetch context chunk {chunk_id}: {e}",
                                        LogCategory.RAG_OPERATIONS,
                                        "app.rag.core.unified_rag_system"
                                    )

                filtered_results = expanded_results

            _backend_logger.info(
                "Structured search completed",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system",
                data={
                    "agent_id": agent_id,
                    "results_count": len(filtered_results),
                    "expanded": expand_context
                }
            )

            return filtered_results

        except Exception as e:
            _backend_logger.error(
                f"Structured search failed: {e}, falling back to basic search",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            return await self.search_agent_knowledge(
                agent_id=agent_id,
                query=query,
                top_k=top_k,
                use_advanced_retrieval=use_advanced_retrieval
            )

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

            # Convert to hybrid-compatible format with REAL scores
            results = []
            for doc in documents:
                # Extract real similarity score from metadata
                metadata = doc.metadata or {}
                similarity_score = metadata.get('similarity_score', 1.0)

                result = {
                    'content': doc.content,
                    'metadata': metadata,
                    'score': similarity_score,  # REAL similarity score!
                    'id': doc.id
                }
                results.append(result)

            return results

        except Exception as e:
            _backend_logger.error(
                f"Failed to search documents for agent {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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

            # Add documents (handle both sync and async)
            if hasattr(collection, 'add') and asyncio.iscoroutinefunction(collection.add):
                # Async add for VectorCollectionWrapper
                await collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
            else:
                # Sync add for ChromaDB
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )

            self.stats["total_documents"] += len(documents)
            _backend_logger.info(
                f"Added {len(documents)} documents to {collection_name}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Add to BM25 index if advanced retrieval is enabled and this is knowledge collection
            if (self.advanced_retrieval_pipeline is not None and
                collection_type == "knowledge"):
                try:
                    # Prepare documents for BM25
                    bm25_docs = [
                        {
                            'doc_id': doc.id,
                            'content': doc.content,
                            'metadata': doc.metadata or {}
                        }
                        for doc in documents
                    ]

                    added_count = await self.advanced_retrieval_pipeline.add_documents_to_bm25(bm25_docs)
                    _backend_logger.info(
                        f"Added {added_count} documents to BM25 index",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                except Exception as e:
                    _backend_logger.warn(
                        f"Failed to add documents to BM25 index: {e}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )

            return True

        except Exception as e:
            _backend_logger.error(
                f"Failed to add documents for agent {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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
                _backend_logger.warn(
                    f"No ecosystem found for agent {agent_id}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
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
                    _backend_logger.debug(
                        f"Deleted collection: {collection_name}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
                except Exception as e:
                    _backend_logger.warn(
                        f"Error deleting collection {collection_name}: {str(e)}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )

            # Remove from tracking
            del self.agent_collections[agent_id]
            self.stats["total_agents"] -= 1
            self.stats["total_collections"] -= 3

            _backend_logger.info(
                f"Deleted agent ecosystem for {agent_id}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            return True

        except Exception as e:
            _backend_logger.error(
                f"Failed to delete agent ecosystem for {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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
            _backend_logger.error(
                f"Failed to search memory for agent {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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
                _backend_logger.info(
                    f"Cleaned up {len(results['ids'])} expired memories for agent {agent_id}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
                return len(results['ids'])

            return 0

        except Exception as e:
            _backend_logger.error(
                f"Failed to cleanup memories for agent {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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
                    _backend_logger.warn(
                        f"Could not get count for {collection_name}: {str(e)}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )

            return stats

        except Exception as e:
            _backend_logger.error(
                f"Failed to get stats for agent {agent_id}: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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
                _backend_logger.info(
                    "🔄 Starting dynamic RAG configuration update...",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

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
                    _backend_logger.error(
                        f"❌ Invalid configuration: {str(e)}",
                        LogCategory.RAG_OPERATIONS,
                        "app.rag.core.unified_rag_system"
                    )
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

                _backend_logger.info(
                    f"✅ RAG configuration updated successfully. Applied {len(updates_applied)} changes.",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )

                return {
                    "success": True,
                    "message": f"RAG configuration updated successfully with {len(updates_applied)} changes",
                    "updates_applied": updates_applied,
                    "warnings": warnings,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                _backend_logger.error(
                    f"❌ Failed to update RAG configuration: {str(e)}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.core.unified_rag_system"
                )
                return {
                    "success": False,
                    "error": f"Failed to update configuration: {str(e)}",
                    "updates_applied": [],
                    "warnings": []
                }

    async def _update_embedding_model(self, new_model: str) -> None:
        """Update the embedding model dynamically."""
        try:
            _backend_logger.info(
                f"🔄 Updating embedding model to: {new_model}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

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

            _backend_logger.info(
                f"✅ Embedding model updated to: {new_model}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to update embedding model: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
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
            _backend_logger.info(
                "🔄 Reloading RAG system...",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Store current config
            current_config = self.config

            # Reinitialize the system
            self.is_initialized = False
            await self.initialize()

            _backend_logger.info(
                "✅ RAG system reloaded successfully",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            return {
                "success": True,
                "message": "RAG system reloaded successfully",
                "config": current_config.dict(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to reload RAG system: {str(e)}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            return {
                "success": False,
                "error": f"Failed to reload system: {str(e)}"
            }

    # ============================================================================
    # MODEL MANAGEMENT API
    # ============================================================================

    async def download_model(
        self,
        model_id: str,
        model_type: str = "embedding"
    ) -> Dict[str, Any]:
        """
        Download a specific model for use in the RAG system.

        This allows users to download additional models beyond the 3 defaults.

        Args:
            model_id: Model ID (short name like 'all-MiniLM-L6-v2' or full HuggingFace ID)
            model_type: Type of model ('embedding', 'reranking', 'vision')

        Returns:
            Dictionary with download status and model location

        Example:
            # Download a larger embedding model
            result = await rag_system.download_model('all-mpnet-base-v2', 'embedding')

            # Download a different reranker
            result = await rag_system.download_model('ms-marco-MiniLM-L-12-v2', 'reranking')
        """
        try:
            from app.rag.config.required_models import get_model_by_id
            from app.rag.core.model_initialization_service import get_model_initialization_service

            # Get model spec
            model_spec = get_model_by_id(model_id)
            if not model_spec:
                return {
                    "success": False,
                    "error": f"Unknown model: {model_id}",
                    "available_models": self.list_available_models(model_type)
                }

            # Verify model type matches
            if model_spec.model_type != model_type:
                return {
                    "success": False,
                    "error": f"Model {model_id} is type '{model_spec.model_type}', not '{model_type}'"
                }

            _backend_logger.info(
                f"Downloading model: {model_id} ({model_type})",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            # Get model service
            service = await get_model_initialization_service()

            # Download model
            locations = await service.ensure_models_available([model_spec])
            location = locations.get(model_spec.model_id)

            if location and location.is_valid:
                return {
                    "success": True,
                    "model_id": model_spec.model_id,
                    "model_type": model_spec.model_type,
                    "location": str(location.path),
                    "size_mb": location.size_mb,
                    "message": f"Model {model_id} is ready"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to download model: {model_id}",
                    "validation_errors": location.validation_errors if location else []
                }

        except Exception as e:
            _backend_logger.error(
                f"Failed to download model {model_id}: {e}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            return {
                "success": False,
                "error": str(e)
            }

    async def switch_embedding_model(self, model_id: str) -> Dict[str, Any]:
        """
        Switch to a different embedding model.

        Args:
            model_id: Model ID (short name or full HuggingFace ID)

        Returns:
            Dictionary with switch status

        Example:
            # Switch to a larger, more accurate model
            result = await rag_system.switch_embedding_model('all-mpnet-base-v2')
        """
        try:
            from app.rag.config.required_models import get_model_by_id
            from .embeddings import EmbeddingManager, EmbeddingConfig, EmbeddingType

            # Get model spec
            model_spec = get_model_by_id(model_id)
            if not model_spec or model_spec.model_type != 'embedding':
                return {
                    "success": False,
                    "error": f"Unknown embedding model: {model_id}",
                    "available_models": self.list_available_models('embedding')
                }

            # Download if not available
            download_result = await self.download_model(model_id, 'embedding')
            if not download_result['success']:
                return download_result

            # Update config
            old_model = self.config.embedding_model
            self.config.embedding_model = model_spec.model_id

            # Reinitialize embedding manager
            if self.embedding_manager:
                embedding_config = EmbeddingConfig(
                    model_name=model_spec.model_id,
                    embedding_type=EmbeddingType.DENSE,
                    batch_size=self.config.batch_size,
                    cache_embeddings=True,
                    use_model_manager=True
                )

                self.embedding_manager = EmbeddingManager(embedding_config)
                await self.embedding_manager.initialize()

            _backend_logger.info(
                f"Switched embedding model: {old_model} → {model_spec.model_id}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )

            return {
                "success": True,
                "old_model": old_model,
                "new_model": model_spec.model_id,
                "message": f"Embedding model switched to {model_spec.model_id}"
            }

        except Exception as e:
            _backend_logger.error(
                f"Failed to switch embedding model: {e}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            return {
                "success": False,
                "error": str(e)
            }

    def list_available_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available models.

        Args:
            model_type: Optional filter by type ('embedding', 'reranking', 'vision')

        Returns:
            List of model information dictionaries

        Example:
            # List all embedding models
            models = rag_system.list_available_models('embedding')
            for model in models:
                print(f"{model['id']}: {model['description']} ({model['size']})")
        """
        try:
            from app.rag.config.required_models import get_models_by_type, ALL_MODELS, format_size

            if model_type:
                models = get_models_by_type(model_type)
            else:
                models = list(ALL_MODELS.values())

            return [
                {
                    "id": m.model_id,
                    "short_name": m.local_name,
                    "type": m.model_type,
                    "priority": m.priority.value,
                    "size": format_size(m.size_mb),
                    "size_mb": m.size_mb,
                    "description": m.description,
                    "dimension": m.dimension,
                    "requires_feature": m.requires_feature
                }
                for m in models
            ]

        except Exception as e:
            _backend_logger.error(
                f"Failed to list models: {e}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.core.unified_rag_system"
            )
            return []

    def get_current_models(self) -> Dict[str, str]:
        """
        Get currently active models.

        Returns:
            Dictionary mapping model type to model ID

        Example:
            models = rag_system.get_current_models()
            print(f"Embedding: {models['embedding']}")
            print(f"Reranking: {models.get('reranking', 'Not configured')}")
        """
        current = {
            "embedding": self.config.embedding_model
        }

        # Check if advanced retrieval is configured
        if self.advanced_retrieval_pipeline:
            try:
                reranker_config = self.advanced_retrieval_pipeline.config
                if hasattr(reranker_config, 'reranker_model'):
                    current["reranking"] = reranker_config.reranker_model
            except:
                pass

        return current


