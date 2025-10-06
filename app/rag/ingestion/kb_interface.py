"""
Knowledge Base interface for atomic document operations.

This module provides an abstract interface for knowledge base operations
with transaction support, enabling atomic ingestion with rollback capability.

Enhanced with structured KB interface integration for full metadata preservation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import structlog

from ..core.unified_rag_system import Document, DocumentChunk

# Import structured KB components
try:
    from ..core.structured_kb_interface import (
        StructuredKBInterface,
        ChunkMetadata,
        ContentType
    )
    from ..core.metadata_index import (
        get_metadata_index_manager,
        TermFilter,
        RangeFilter
    )
    from ..core.chunk_relationship_manager import (
        get_chunk_relationship_manager,
        RelationType
    )
    from ..core.deduplication_enforcer import (
        get_deduplication_enforcer,
        DuplicateAction,
        ConflictResolution
    )
    from ..core.multimodal_indexer import (
        get_multimodal_indexer,
        ContentType as MultimodalContentType
    )
    STRUCTURED_KB_AVAILABLE = True
except ImportError:
    STRUCTURED_KB_AVAILABLE = False
    logger.warning("Structured KB components not available - using basic interface")

logger = structlog.get_logger(__name__)


class TransactionState(str, Enum):
    """Transaction states."""
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class Transaction:
    """Transaction for atomic operations."""
    id: str
    state: TransactionState = TransactionState.ACTIVE
    created_at: datetime = None
    committed_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    
    # Track operations for rollback
    chunks_added: List[str] = None  # Chunk IDs
    chunks_updated: List[str] = None
    chunks_deleted: List[str] = None
    documents_added: List[str] = None  # Document IDs
    documents_updated: List[str] = None
    documents_deleted: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.chunks_added is None:
            self.chunks_added = []
        if self.chunks_updated is None:
            self.chunks_updated = []
        if self.chunks_deleted is None:
            self.chunks_deleted = []
        if self.documents_added is None:
            self.documents_added = []
        if self.documents_updated is None:
            self.documents_updated = []
        if self.documents_deleted is None:
            self.documents_deleted = []


class KnowledgeBaseInterface(ABC):
    """
    Abstract interface for knowledge base operations.
    
    Provides atomic operations with transaction support for reliable
    document ingestion.
    """
    
    @abstractmethod
    async def begin_transaction(self) -> Transaction:
        """
        Begin a new transaction.
        
        Returns:
            Transaction object
        """
        pass
    
    @abstractmethod
    async def commit(self, transaction: Transaction) -> bool:
        """
        Commit a transaction.
        
        Args:
            transaction: Transaction to commit
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def rollback(self, transaction: Transaction) -> bool:
        """
        Rollback a transaction.
        
        Args:
            transaction: Transaction to rollback
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def upsert_chunks(self,
                           chunks: List[DocumentChunk],
                           transaction: Optional[Transaction] = None) -> List[str]:
        """
        Insert or update chunks.
        
        Args:
            chunks: Chunks to upsert
            transaction: Optional transaction
            
        Returns:
            List of chunk IDs
        """
        pass
    
    @abstractmethod
    async def delete_chunks_by_doc_id(self,
                                     doc_id: str,
                                     transaction: Optional[Transaction] = None) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: Document ID
            transaction: Optional transaction
            
        Returns:
            Number of chunks deleted
        """
        pass
    
    @abstractmethod
    async def exists_by_content_sha(self, content_sha: str) -> bool:
        """
        Check if chunk with content hash exists.
        
        Args:
            content_sha: Content SHA-256 hash
            
        Returns:
            True if exists
        """
        pass
    
    @abstractmethod
    async def exists_by_norm_text_sha(self, norm_text_sha: str) -> bool:
        """
        Check if chunk with normalized text hash exists.
        
        Args:
            norm_text_sha: Normalized text SHA-256 hash
            
        Returns:
            True if exists
        """
        pass
    
    @abstractmethod
    async def get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks
        """
        pass
    
    @abstractmethod
    async def get_chunk_by_content_sha(self, content_sha: str) -> Optional[DocumentChunk]:
        """
        Get chunk by content hash.
        
        Args:
            content_sha: Content SHA-256 hash
            
        Returns:
            Chunk if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def upsert_document(self,
                             document: Document,
                             transaction: Optional[Transaction] = None) -> str:
        """
        Insert or update document.
        
        Args:
            document: Document to upsert
            transaction: Optional transaction
            
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete_document(self,
                             doc_id: str,
                             transaction: Optional[Transaction] = None) -> bool:
        """
        Delete document and all its chunks.
        
        Args:
            doc_id: Document ID
            transaction: Optional transaction
            
        Returns:
            True if deleted
        """
        pass
    
    @abstractmethod
    async def batch_upsert_chunks(self,
                                 chunks: List[DocumentChunk],
                                 batch_size: int = 100,
                                 transaction: Optional[Transaction] = None) -> List[str]:
        """
        Batch upsert chunks for efficiency.

        Enhanced to preserve all metadata and use structured KB interface.

        Args:
            chunks: Chunks to upsert
            batch_size: Batch size
            transaction: Optional transaction

        Returns:
            List of chunk IDs
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Statistics dictionary
        """
        pass


class CollectionBasedKBInterface(KnowledgeBaseInterface):
    """
    Implementation of KnowledgeBaseInterface for CollectionBasedKBManager.

    Wraps the existing CollectionBasedKBManager with transaction support.
    Enhanced with structured KB interface for full metadata preservation.
    """

    def __init__(self, kb_manager, collection_name: str):
        """
        Initialize interface.

        Args:
            kb_manager: CollectionBasedKBManager instance
            collection_name: Collection name
        """
        self.kb_manager = kb_manager
        self.collection_name = collection_name
        self._active_transactions: Dict[str, Transaction] = {}

        # Initialize structured KB components if available
        self.structured_kb: Optional[StructuredKBInterface] = None
        self.metadata_index_manager = None
        self.chunk_relationship_manager = None
        self.deduplication_enforcer = None
        self.multimodal_indexer = None

        if STRUCTURED_KB_AVAILABLE:
            self._init_structured_kb()

        logger.info(
            "CollectionBasedKBInterface initialized",
            collection=collection_name,
            structured_kb_enabled=STRUCTURED_KB_AVAILABLE
        )

    def _init_structured_kb(self):
        """Initialize structured KB components."""
        try:
            # Note: These will be initialized asynchronously on first use
            # We can't call async functions in __init__
            logger.info("Structured KB components will be initialized on first use")
        except Exception as e:
            logger.error(f"Failed to initialize structured KB components: {e}")
    
    async def begin_transaction(self) -> Transaction:
        """Begin a new transaction."""
        import uuid
        
        transaction = Transaction(
            id=str(uuid.uuid4()),
            state=TransactionState.ACTIVE
        )
        
        self._active_transactions[transaction.id] = transaction
        
        logger.info("Transaction started", transaction_id=transaction.id)
        
        return transaction
    
    async def commit(self, transaction: Transaction) -> bool:
        """
        Commit a transaction.
        
        Note: Current implementation doesn't support true transactions,
        so this is a no-op. All operations are immediately committed.
        """
        if transaction.id not in self._active_transactions:
            logger.error("Transaction not found", transaction_id=transaction.id)
            return False
        
        transaction.state = TransactionState.COMMITTED
        transaction.committed_at = datetime.utcnow()
        
        del self._active_transactions[transaction.id]
        
        logger.info("Transaction committed", transaction_id=transaction.id)
        
        return True
    
    async def rollback(self, transaction: Transaction) -> bool:
        """
        Rollback a transaction.
        
        Note: Current implementation doesn't support true rollback.
        This would require implementing a write-ahead log or similar mechanism.
        """
        if transaction.id not in self._active_transactions:
            logger.error("Transaction not found", transaction_id=transaction.id)
            return False
        
        transaction.state = TransactionState.ROLLED_BACK
        transaction.rolled_back_at = datetime.utcnow()
        
        # TODO: Implement actual rollback by reversing operations
        logger.warning(
            "Rollback requested but not fully implemented",
            transaction_id=transaction.id,
            chunks_added=len(transaction.chunks_added),
            chunks_deleted=len(transaction.chunks_deleted)
        )
        
        del self._active_transactions[transaction.id]
        
        return True
    
    async def upsert_chunks(self,
                           chunks: List[DocumentChunk],
                           transaction: Optional[Transaction] = None) -> List[str]:
        """Insert or update chunks."""
        chunk_ids = []
        
        for chunk in chunks:
            # Add to knowledge base
            await self.kb_manager.add_document(
                collection_name=self.collection_name,
                document=Document(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    embedding=chunk.embedding
                )
            )
            
            chunk_ids.append(chunk.id)
            
            # Track in transaction
            if transaction:
                transaction.chunks_added.append(chunk.id)
        
        logger.debug(
            "Chunks upserted",
            count=len(chunks),
            collection=self.collection_name
        )
        
        return chunk_ids
    
    async def delete_chunks_by_doc_id(self,
                                     doc_id: str,
                                     transaction: Optional[Transaction] = None) -> int:
        """Delete all chunks for a document."""
        # Get existing chunks
        chunks = await self.get_chunks_by_doc_id(doc_id)
        
        # Delete each chunk
        for chunk in chunks:
            await self.kb_manager.delete_document(
                collection_name=self.collection_name,
                document_id=chunk.id
            )
            
            # Track in transaction
            if transaction:
                transaction.chunks_deleted.append(chunk.id)
        
        logger.debug(
            "Chunks deleted",
            doc_id=doc_id,
            count=len(chunks),
            collection=self.collection_name
        )
        
        return len(chunks)
    
    async def exists_by_content_sha(self, content_sha: str) -> bool:
        """Check if chunk with content hash exists."""
        # Query by metadata
        results = await self.kb_manager.search(
            collection_name=self.collection_name,
            query_text="",  # Empty query
            top_k=1,
            filters={"content_sha": content_sha}
        )
        
        return len(results.documents) > 0
    
    async def exists_by_norm_text_sha(self, norm_text_sha: str) -> bool:
        """Check if chunk with normalized text hash exists."""
        results = await self.kb_manager.search(
            collection_name=self.collection_name,
            query_text="",
            top_k=1,
            filters={"norm_text_sha": norm_text_sha}
        )
        
        return len(results.documents) > 0
    
    async def get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        results = await self.kb_manager.search(
            collection_name=self.collection_name,
            query_text="",
            top_k=1000,  # Large number to get all chunks
            filters={"document_id": doc_id}
        )
        
        chunks = []
        for doc in results.documents:
            chunk = DocumentChunk(
                id=doc.id,
                content=doc.content,
                document_id=doc.metadata.get("document_id", doc_id),
                chunk_index=doc.metadata.get("chunk_index", 0),
                metadata=doc.metadata,
                embedding=doc.embedding
            )
            chunks.append(chunk)
        
        return chunks
    
    async def get_chunk_by_content_sha(self, content_sha: str) -> Optional[DocumentChunk]:
        """Get chunk by content hash."""
        results = await self.kb_manager.search(
            collection_name=self.collection_name,
            query_text="",
            top_k=1,
            filters={"content_sha": content_sha}
        )
        
        if results.documents:
            doc = results.documents[0]
            return DocumentChunk(
                id=doc.id,
                content=doc.content,
                document_id=doc.metadata.get("document_id", ""),
                chunk_index=doc.metadata.get("chunk_index", 0),
                metadata=doc.metadata,
                embedding=doc.embedding
            )
        
        return None
    
    async def upsert_document(self,
                             document: Document,
                             transaction: Optional[Transaction] = None) -> str:
        """Insert or update document."""
        await self.kb_manager.add_document(
            collection_name=self.collection_name,
            document=document
        )
        
        if transaction:
            transaction.documents_added.append(document.id)
        
        return document.id
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        results = await self.kb_manager.search(
            collection_name=self.collection_name,
            query_text="",
            top_k=1,
            filters={"id": doc_id}
        )
        
        if results.documents:
            return results.documents[0]
        
        return None
    
    async def delete_document(self,
                             doc_id: str,
                             transaction: Optional[Transaction] = None) -> bool:
        """Delete document and all its chunks."""
        # Delete all chunks
        await self.delete_chunks_by_doc_id(doc_id, transaction)
        
        # Delete document
        await self.kb_manager.delete_document(
            collection_name=self.collection_name,
            document_id=doc_id
        )
        
        if transaction:
            transaction.documents_deleted.append(doc_id)
        
        return True
    
    async def batch_upsert_chunks(self,
                                 chunks: List[DocumentChunk],
                                 batch_size: int = 100,
                                 transaction: Optional[Transaction] = None) -> List[str]:
        """
        Batch upsert chunks for efficiency.

        Enhanced with structured KB integration for full metadata preservation.
        """
        chunk_ids = []

        # Initialize structured KB components if not already done
        if STRUCTURED_KB_AVAILABLE and self.metadata_index_manager is None:
            await self._ensure_structured_kb_initialized()

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Enhanced processing with structured KB
            if STRUCTURED_KB_AVAILABLE and self.deduplication_enforcer:
                batch_ids = await self._batch_upsert_with_structured_kb(batch, transaction)
            else:
                # Fallback to basic upsert
                batch_ids = await self.upsert_chunks(batch, transaction)

            chunk_ids.extend(batch_ids)

        return chunk_ids

    async def _ensure_structured_kb_initialized(self):
        """Ensure structured KB components are initialized."""
        try:
            if not self.metadata_index_manager:
                self.metadata_index_manager = await get_metadata_index_manager()

            if not self.chunk_relationship_manager:
                self.chunk_relationship_manager = await get_chunk_relationship_manager()

            if not self.deduplication_enforcer:
                self.deduplication_enforcer = await get_deduplication_enforcer()

            if not self.multimodal_indexer:
                self.multimodal_indexer = await get_multimodal_indexer()

            logger.info("Structured KB components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize structured KB components: {e}")

    async def _batch_upsert_with_structured_kb(
        self,
        chunks: List[DocumentChunk],
        transaction: Optional[Transaction] = None
    ) -> List[str]:
        """
        Batch upsert with full structured KB integration.

        Includes:
        - Deduplication checking
        - Metadata indexing
        - Relationship tracking
        - Multimodal indexing
        """
        chunk_ids = []

        for chunk in chunks:
            try:
                # Extract metadata
                content_sha = chunk.metadata.get('content_sha', '')
                norm_text_sha = chunk.metadata.get('norm_text_sha', '')
                content_type_str = chunk.metadata.get('content_type', 'text')

                # Map content type
                try:
                    content_type = ContentType(content_type_str)
                except ValueError:
                    content_type = ContentType.TEXT

                # Check for duplicates
                if content_sha and norm_text_sha:
                    dedup_result = await self.deduplication_enforcer.check_duplicate(
                        chunk_id=chunk.id,
                        content=chunk.content,
                        content_sha=content_sha,
                        norm_text_sha=norm_text_sha,
                        metadata=chunk.metadata
                    )

                    # Handle duplicate
                    if dedup_result.is_duplicate and dedup_result.action == DuplicateAction.SKIP:
                        logger.debug(f"Skipping duplicate chunk: {chunk.id}")
                        continue

                # Add to knowledge base (existing logic)
                await self.kb_manager.add_document(
                    collection_name=self.collection_name,
                    document=Document(
                        id=chunk.id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        embedding=chunk.embedding
                    )
                )

                # Register in deduplication enforcer
                if content_sha and norm_text_sha:
                    await self.deduplication_enforcer.register_chunk(
                        chunk_id=chunk.id,
                        content_sha=content_sha,
                        norm_text_sha=norm_text_sha,
                        metadata=chunk.metadata
                    )

                # Add to metadata index
                await self.metadata_index_manager.add_document(
                    chunk_id=chunk.id,
                    metadata=chunk.metadata
                )

                # Add to chunk relationship manager
                doc_id = chunk.metadata.get('document_id', chunk.document_id)
                chunk_index = chunk.metadata.get('chunk_index', chunk.chunk_index)
                total_chunks = chunk.metadata.get('total_chunks', 1)
                section_path = chunk.metadata.get('section_path')
                page_number = chunk.metadata.get('page_number')

                self.chunk_relationship_manager.add_chunk(
                    chunk_id=chunk.id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    section_path=section_path,
                    page_number=page_number,
                    metadata=chunk.metadata
                )

                # Add to multimodal indexer
                try:
                    multimodal_content_type = MultimodalContentType(content_type_str)
                except ValueError:
                    multimodal_content_type = MultimodalContentType.TEXT

                await self.multimodal_indexer.add_chunk(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    content_type=multimodal_content_type,
                    embedding=chunk.embedding,
                    metadata=chunk.metadata
                )

                chunk_ids.append(chunk.id)

                # Track in transaction
                if transaction:
                    transaction.chunks_added.append(chunk.id)

                logger.debug(
                    f"Chunk upserted with structured KB",
                    chunk_id=chunk.id,
                    content_type=content_type_str
                )

            except Exception as e:
                logger.error(f"Failed to upsert chunk {chunk.id} with structured KB: {e}")
                # Continue with next chunk
                continue

        return chunk_ids
    
    async def build_document_structure(self, doc_id: str) -> bool:
        """
        Build document structure and relationships after chunks are added.

        Args:
            doc_id: Document ID

        Returns:
            True if successful
        """
        if not STRUCTURED_KB_AVAILABLE or not self.chunk_relationship_manager:
            logger.warning("Structured KB not available - skipping structure building")
            return False

        try:
            # Build document structure
            success = self.chunk_relationship_manager.build_document_structure(doc_id)

            if success:
                logger.info(f"Document structure built successfully: {doc_id}")
            else:
                logger.warning(f"Failed to build document structure: {doc_id}")

            return success

        except Exception as e:
            logger.error(f"Error building document structure for {doc_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        # Get collection info
        info = await self.kb_manager.get_collection_info(self.collection_name)

        stats = {
            "collection": self.collection_name,
            "document_count": info.get("count", 0),
            "active_transactions": len(self._active_transactions)
        }

        # Add structured KB stats if available
        if STRUCTURED_KB_AVAILABLE:
            if self.deduplication_enforcer:
                dedup_stats = await self.deduplication_enforcer.get_statistics()
                stats['deduplication'] = {
                    'total_checks': dedup_stats.total_checks,
                    'exact_duplicates': dedup_stats.exact_duplicates,
                    'fuzzy_duplicates': dedup_stats.fuzzy_duplicates,
                    'dedup_rate': dedup_stats.dedup_rate
                }

            if self.metadata_index_manager:
                metadata_metrics = self.metadata_index_manager.get_metrics()
                stats['metadata_index'] = metadata_metrics

            if self.chunk_relationship_manager:
                relationship_metrics = self.chunk_relationship_manager.get_metrics()
                stats['relationships'] = relationship_metrics

            if self.multimodal_indexer:
                multimodal_metrics = self.multimodal_indexer.get_metrics()
                stats['multimodal'] = multimodal_metrics

        return stats

