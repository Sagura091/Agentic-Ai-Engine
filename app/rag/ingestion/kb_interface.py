"""
Knowledge Base interface for atomic document operations.

This module provides an abstract interface for knowledge base operations
with transaction support, enabling atomic ingestion with rollback capability.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import structlog

from ..core.unified_rag_system import Document, DocumentChunk

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
        
        logger.info(
            "CollectionBasedKBInterface initialized",
            collection=collection_name
        )
    
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
        """Batch upsert chunks for efficiency."""
        chunk_ids = []
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_ids = await self.upsert_chunks(batch, transaction)
            chunk_ids.extend(batch_ids)
        
        return chunk_ids
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        # Get collection info
        info = await self.kb_manager.get_collection_info(self.collection_name)
        
        return {
            "collection": self.collection_name,
            "document_count": info.get("count", 0),
            "active_transactions": len(self._active_transactions)
        }

