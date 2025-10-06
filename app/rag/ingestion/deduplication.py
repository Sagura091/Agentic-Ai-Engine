"""
Document and chunk deduplication for RAG ingestion.

This module provides deduplication capabilities to:
- Detect exact duplicates (content_sha)
- Detect fuzzy duplicates (norm_text_sha)
- Track document versions
- Update only changed chunks on re-ingestion
- Maintain deduplication metrics
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import structlog

from ..core.unified_rag_system import Document, DocumentChunk
from .utils_hash import compute_content_sha, compute_norm_text_sha
from .kb_interface import KnowledgeBaseInterface

logger = structlog.get_logger(__name__)


class DuplicateType(str, Enum):
    """Types of duplicates."""
    EXACT = "exact"  # Exact content match (content_sha)
    FUZZY = "fuzzy"  # Normalized content match (norm_text_sha)
    NONE = "none"    # Not a duplicate


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""
    is_duplicate: bool
    duplicate_type: DuplicateType
    existing_chunk_id: Optional[str] = None
    existing_document_id: Optional[str] = None
    similarity_score: float = 0.0  # 1.0 for exact, <1.0 for fuzzy
    
    def __bool__(self):
        """Allow boolean check."""
        return self.is_duplicate


@dataclass
class DeduplicationStats:
    """Statistics for deduplication operations."""
    total_chunks_checked: int = 0
    exact_duplicates: int = 0
    fuzzy_duplicates: int = 0
    unique_chunks: int = 0
    chunks_updated: int = 0
    chunks_skipped: int = 0
    
    def get_dedup_rate(self) -> float:
        """Get deduplication rate (0.0-1.0)."""
        if self.total_chunks_checked == 0:
            return 0.0
        return (self.exact_duplicates + self.fuzzy_duplicates) / self.total_chunks_checked
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_chunks_checked": self.total_chunks_checked,
            "exact_duplicates": self.exact_duplicates,
            "fuzzy_duplicates": self.fuzzy_duplicates,
            "unique_chunks": self.unique_chunks,
            "chunks_updated": self.chunks_updated,
            "chunks_skipped": self.chunks_skipped,
            "dedup_rate": self.get_dedup_rate()
        }


class DeduplicationEngine:
    """
    Engine for detecting and handling duplicate documents and chunks.
    
    Provides:
    - Exact duplicate detection (content_sha)
    - Fuzzy duplicate detection (norm_text_sha)
    - Incremental updates (only changed chunks)
    - Deduplication metrics
    """
    
    def __init__(self, kb_interface: KnowledgeBaseInterface):
        """
        Initialize deduplication engine.
        
        Args:
            kb_interface: Knowledge base interface for querying existing content
        """
        self.kb_interface = kb_interface
        self.stats = DeduplicationStats()
        
        logger.info("DeduplicationEngine initialized")
    
    async def check_chunk_duplicate(self, chunk: DocumentChunk) -> DeduplicationResult:
        """
        Check if a chunk is a duplicate.
        
        Args:
            chunk: Chunk to check
            
        Returns:
            DeduplicationResult
        """
        self.stats.total_chunks_checked += 1
        
        # Ensure hashes are computed
        if not hasattr(chunk, 'content_sha') or not chunk.content_sha:
            chunk.content_sha = compute_content_sha(chunk.content)
        if not hasattr(chunk, 'norm_text_sha') or not chunk.norm_text_sha:
            chunk.norm_text_sha = compute_norm_text_sha(chunk.content)
        
        # Check for exact duplicate
        if await self.kb_interface.exists_by_content_sha(chunk.content_sha):
            self.stats.exact_duplicates += 1
            
            # Get existing chunk
            existing_chunk = await self.kb_interface.get_chunk_by_content_sha(chunk.content_sha)
            
            logger.debug(
                "Exact duplicate detected",
                chunk_id=chunk.id,
                existing_chunk_id=existing_chunk.id if existing_chunk else None,
                content_sha=chunk.content_sha
            )
            
            return DeduplicationResult(
                is_duplicate=True,
                duplicate_type=DuplicateType.EXACT,
                existing_chunk_id=existing_chunk.id if existing_chunk else None,
                existing_document_id=existing_chunk.document_id if existing_chunk else None,
                similarity_score=1.0
            )
        
        # Check for fuzzy duplicate
        if await self.kb_interface.exists_by_norm_text_sha(chunk.norm_text_sha):
            self.stats.fuzzy_duplicates += 1
            
            logger.debug(
                "Fuzzy duplicate detected",
                chunk_id=chunk.id,
                norm_text_sha=chunk.norm_text_sha
            )
            
            return DeduplicationResult(
                is_duplicate=True,
                duplicate_type=DuplicateType.FUZZY,
                similarity_score=0.95  # High similarity but not exact
            )
        
        # Not a duplicate
        self.stats.unique_chunks += 1
        
        return DeduplicationResult(
            is_duplicate=False,
            duplicate_type=DuplicateType.NONE
        )
    
    async def deduplicate_chunks(
        self,
        chunks: List[DocumentChunk],
        skip_duplicates: bool = True
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Deduplicate a list of chunks.
        
        Args:
            chunks: Chunks to deduplicate
            skip_duplicates: If True, skip duplicates; if False, include all
            
        Returns:
            Tuple of (unique_chunks, duplicate_chunks)
        """
        unique_chunks = []
        duplicate_chunks = []
        
        for chunk in chunks:
            result = await self.check_chunk_duplicate(chunk)
            
            if result.is_duplicate:
                duplicate_chunks.append(chunk)
                if not skip_duplicates:
                    unique_chunks.append(chunk)
                else:
                    self.stats.chunks_skipped += 1
            else:
                unique_chunks.append(chunk)
        
        logger.info(
            "Chunk deduplication complete",
            total=len(chunks),
            unique=len(unique_chunks),
            duplicates=len(duplicate_chunks),
            skipped=self.stats.chunks_skipped
        )
        
        return unique_chunks, duplicate_chunks
    
    async def handle_document_update(
        self,
        document_id: str,
        new_chunks: List[DocumentChunk]
    ) -> Tuple[List[DocumentChunk], List[str]]:
        """
        Handle document re-ingestion by updating only changed chunks.
        
        Args:
            document_id: Document ID
            new_chunks: New chunks from re-processing
            
        Returns:
            Tuple of (chunks_to_add, chunk_ids_to_delete)
        """
        # Get existing chunks for this document
        existing_chunks = await self.kb_interface.get_chunks_by_doc_id(document_id)
        
        # Build hash maps for comparison
        existing_by_content_sha = {
            chunk.metadata.get('content_sha', ''): chunk 
            for chunk in existing_chunks
            if chunk.metadata.get('content_sha')
        }
        
        new_by_content_sha = {
            chunk.content_sha: chunk 
            for chunk in new_chunks
            if chunk.content_sha
        }
        
        # Find chunks to add (new or changed)
        chunks_to_add = []
        for content_sha, new_chunk in new_by_content_sha.items():
            if content_sha not in existing_by_content_sha:
                # New chunk
                chunks_to_add.append(new_chunk)
                self.stats.unique_chunks += 1
            else:
                # Chunk exists - check if metadata changed
                existing_chunk = existing_by_content_sha[content_sha]
                if self._has_metadata_changed(existing_chunk, new_chunk):
                    chunks_to_add.append(new_chunk)
                    self.stats.chunks_updated += 1
                else:
                    # Unchanged - skip
                    self.stats.chunks_skipped += 1
        
        # Find chunks to delete (removed from document)
        chunk_ids_to_delete = []
        for content_sha, existing_chunk in existing_by_content_sha.items():
            if content_sha not in new_by_content_sha:
                chunk_ids_to_delete.append(existing_chunk.id)
        
        logger.info(
            "Document update analysis complete",
            document_id=document_id,
            existing_chunks=len(existing_chunks),
            new_chunks=len(new_chunks),
            to_add=len(chunks_to_add),
            to_delete=len(chunk_ids_to_delete),
            unchanged=len(existing_chunks) - len(chunk_ids_to_delete) - len(chunks_to_add)
        )
        
        return chunks_to_add, chunk_ids_to_delete
    
    def _has_metadata_changed(self, old_chunk: DocumentChunk, new_chunk: DocumentChunk) -> bool:
        """
        Check if chunk metadata has changed significantly.
        
        Args:
            old_chunk: Existing chunk
            new_chunk: New chunk
            
        Returns:
            True if metadata changed
        """
        # Compare important metadata fields
        important_fields = [
            'section_path', 'page', 'chunk_index',
            'document_type', 'language'
        ]
        
        for field in important_fields:
            old_val = old_chunk.metadata.get(field)
            new_val = new_chunk.metadata.get(field)
            if old_val != new_val:
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.stats.to_dict()
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = DeduplicationStats()
        logger.info("Deduplication statistics reset")


async def deduplicate_chunks(
    chunks: List[DocumentChunk],
    kb_interface: KnowledgeBaseInterface,
    skip_duplicates: bool = True
) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
    """
    Convenience function to deduplicate chunks.
    
    Args:
        chunks: Chunks to deduplicate
        kb_interface: Knowledge base interface
        skip_duplicates: If True, skip duplicates
        
    Returns:
        Tuple of (unique_chunks, duplicate_chunks)
    """
    engine = DeduplicationEngine(kb_interface)
    return await engine.deduplicate_chunks(chunks, skip_duplicates)

