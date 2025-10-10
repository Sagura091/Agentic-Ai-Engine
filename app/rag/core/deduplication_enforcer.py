"""
Deduplication Enforcer.

This module enforces deduplication at the knowledge base level using content
hashes and fuzzy matching. It prevents duplicate content from being indexed
and handles updates vs new documents intelligently.

Key Features:
- Exact deduplication using content_sha
- Fuzzy deduplication using norm_text_sha
- Configurable similarity thresholds
- Update detection (same doc, different content)
- Conflict resolution strategies
- Deduplication statistics and reporting
- Batch deduplication

Author: Agentic AI System
Purpose: Enforce deduplication at KB level
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class DuplicateAction(str, Enum):
    """Actions to take when duplicate is detected."""
    SKIP = "skip"  # Skip the duplicate
    UPDATE = "update"  # Update existing chunk
    MERGE = "merge"  # Merge metadata
    KEEP_BOTH = "keep_both"  # Keep both versions
    REPLACE = "replace"  # Replace existing


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""
    KEEP_EXISTING = "keep_existing"
    KEEP_NEW = "keep_new"
    KEEP_HIGHER_CONFIDENCE = "keep_higher_confidence"
    KEEP_NEWER = "keep_newer"
    MERGE_METADATA = "merge_metadata"


@dataclass
class DuplicateMatch:
    """A duplicate match result."""
    existing_chunk_id: str
    new_chunk_id: str
    match_type: str  # "exact", "fuzzy"
    similarity: float  # 0.0-1.0
    content_sha_match: bool
    norm_text_sha_match: bool
    recommended_action: DuplicateAction
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""
    is_duplicate: bool
    action: DuplicateAction
    existing_chunk_id: Optional[str] = None
    similarity: float = 0.0
    match_type: Optional[str] = None
    reason: str = ""


@dataclass
class DeduplicationStats:
    """Deduplication statistics."""
    total_checks: int = 0
    exact_duplicates: int = 0
    fuzzy_duplicates: int = 0
    updates: int = 0
    skipped: int = 0
    merged: int = 0
    replaced: int = 0
    unique_chunks: int = 0
    dedup_rate: float = 0.0


class DeduplicationEnforcer:
    """
    Production-grade deduplication enforcer.
    
    Enforces deduplication at KB level using multiple strategies:
    - Exact matching (content_sha)
    - Fuzzy matching (norm_text_sha)
    - Configurable thresholds
    - Intelligent conflict resolution
    
    Features:
    - Multiple deduplication strategies
    - Configurable similarity thresholds
    - Update detection
    - Conflict resolution
    - Batch processing
    - Comprehensive statistics
    """
    
    def __init__(
        self,
        fuzzy_threshold: float = 0.95,
        default_action: DuplicateAction = DuplicateAction.SKIP,
        conflict_resolution: ConflictResolution = ConflictResolution.KEEP_EXISTING
    ):
        """
        Initialize deduplication enforcer.
        
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
            default_action: Default action for duplicates
            conflict_resolution: Conflict resolution strategy
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.default_action = default_action
        self.conflict_resolution = conflict_resolution
        
        # Deduplication indexes
        self._content_sha_index: Dict[str, str] = {}  # content_sha -> chunk_id
        self._norm_text_sha_index: Dict[str, Set[str]] = defaultdict(set)  # norm_text_sha -> chunk_ids
        self._chunk_metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata
        
        # Statistics
        self._stats = DeduplicationStats()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            "DeduplicationEnforcer initialized",
            LogCategory.MEMORY_OPERATIONS,
            "app.rag.core.deduplication_enforcer.DeduplicationEnforcer",
            data={
                "fuzzy_threshold": fuzzy_threshold,
                "default_action": default_action.value,
                "conflict_resolution": conflict_resolution.value
            }
        )
    
    async def check_duplicate(
        self,
        chunk_id: str,
        content: str,
        content_sha: str,
        norm_text_sha: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DeduplicationResult:
        """
        Check if chunk is a duplicate.
        
        Args:
            chunk_id: New chunk ID
            content: Chunk content
            content_sha: Content SHA hash
            norm_text_sha: Normalized text SHA hash
            metadata: Optional chunk metadata
            
        Returns:
            DeduplicationResult
        """
        async with self._lock:
            self._stats.total_checks += 1
            
            # Check exact duplicate (content_sha)
            if content_sha in self._content_sha_index:
                existing_chunk_id = self._content_sha_index[content_sha]
                
                # Check if it's the same chunk (update scenario)
                if existing_chunk_id == chunk_id:
                    self._stats.updates += 1
                    return DeduplicationResult(
                        is_duplicate=False,
                        action=DuplicateAction.UPDATE,
                        existing_chunk_id=existing_chunk_id,
                        similarity=1.0,
                        match_type="exact_update",
                        reason="Same chunk ID - this is an update"
                    )
                
                # Different chunk with same content
                self._stats.exact_duplicates += 1
                
                action = await self._resolve_conflict(
                    existing_chunk_id,
                    chunk_id,
                    metadata
                )
                
                if action == DuplicateAction.SKIP:
                    self._stats.skipped += 1
                elif action == DuplicateAction.REPLACE:
                    self._stats.replaced += 1
                elif action == DuplicateAction.MERGE:
                    self._stats.merged += 1
                
                return DeduplicationResult(
                    is_duplicate=True,
                    action=action,
                    existing_chunk_id=existing_chunk_id,
                    similarity=1.0,
                    match_type="exact",
                    reason="Exact content match (content_sha)"
                )
            
            # Check fuzzy duplicate (norm_text_sha)
            if norm_text_sha in self._norm_text_sha_index:
                existing_chunk_ids = self._norm_text_sha_index[norm_text_sha]
                
                # Filter out self
                existing_chunk_ids = {cid for cid in existing_chunk_ids if cid != chunk_id}
                
                if existing_chunk_ids:
                    # Pick the first existing chunk (could be improved with better selection)
                    existing_chunk_id = next(iter(existing_chunk_ids))
                    
                    self._stats.fuzzy_duplicates += 1
                    
                    action = await self._resolve_conflict(
                        existing_chunk_id,
                        chunk_id,
                        metadata
                    )
                    
                    if action == DuplicateAction.SKIP:
                        self._stats.skipped += 1
                    elif action == DuplicateAction.REPLACE:
                        self._stats.replaced += 1
                    elif action == DuplicateAction.MERGE:
                        self._stats.merged += 1
                    
                    return DeduplicationResult(
                        is_duplicate=True,
                        action=action,
                        existing_chunk_id=existing_chunk_id,
                        similarity=self.fuzzy_threshold,
                        match_type="fuzzy",
                        reason="Fuzzy content match (norm_text_sha)"
                    )
            
            # Not a duplicate
            self._stats.unique_chunks += 1
            return DeduplicationResult(
                is_duplicate=False,
                action=DuplicateAction.UPDATE,
                similarity=0.0,
                reason="Unique content"
            )
    
    async def register_chunk(
        self,
        chunk_id: str,
        content_sha: str,
        norm_text_sha: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a chunk in deduplication indexes.
        
        Args:
            chunk_id: Chunk ID
            content_sha: Content SHA hash
            norm_text_sha: Normalized text SHA hash
            metadata: Optional chunk metadata
            
        Returns:
            True if successful
        """
        async with self._lock:
            try:
                # Register in content_sha index
                self._content_sha_index[content_sha] = chunk_id
                
                # Register in norm_text_sha index
                self._norm_text_sha_index[norm_text_sha].add(chunk_id)
                
                # Store metadata
                if metadata:
                    self._chunk_metadata[chunk_id] = metadata

                logger.debug(
                    f"Chunk registered: {chunk_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.rag.core.deduplication_enforcer.DeduplicationEnforcer",
                    data={"chunk_id": chunk_id}
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to register chunk {chunk_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.rag.core.deduplication_enforcer.DeduplicationEnforcer",
                    error=e,
                    data={"chunk_id": chunk_id}
                )
                return False
    
    async def unregister_chunk(self, chunk_id: str) -> bool:
        """
        Unregister a chunk from deduplication indexes.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            True if successful
        """
        async with self._lock:
            try:
                # Remove from content_sha index
                content_sha_to_remove = None
                for sha, cid in self._content_sha_index.items():
                    if cid == chunk_id:
                        content_sha_to_remove = sha
                        break
                
                if content_sha_to_remove:
                    del self._content_sha_index[content_sha_to_remove]
                
                # Remove from norm_text_sha index
                for sha, chunk_ids in self._norm_text_sha_index.items():
                    if chunk_id in chunk_ids:
                        chunk_ids.discard(chunk_id)
                
                # Remove metadata
                if chunk_id in self._chunk_metadata:
                    del self._chunk_metadata[chunk_id]

                logger.debug(
                    f"Chunk unregistered: {chunk_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.rag.core.deduplication_enforcer.DeduplicationEnforcer",
                    data={"chunk_id": chunk_id}
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to unregister chunk {chunk_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.rag.core.deduplication_enforcer.DeduplicationEnforcer",
                    error=e,
                    data={"chunk_id": chunk_id}
                )
                return False

    async def batch_check_duplicates(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[DeduplicationResult]:
        """
        Batch check for duplicates.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'content', 'content_sha', 'norm_text_sha', 'metadata'

        Returns:
            List of DeduplicationResult
        """
        results = []

        for chunk in chunks:
            result = await self.check_duplicate(
                chunk_id=chunk['chunk_id'],
                content=chunk.get('content', ''),
                content_sha=chunk['content_sha'],
                norm_text_sha=chunk['norm_text_sha'],
                metadata=chunk.get('metadata')
            )
            results.append(result)

        return results

    async def find_duplicates(
        self,
        content_sha: Optional[str] = None,
        norm_text_sha: Optional[str] = None
    ) -> List[str]:
        """
        Find duplicate chunks.

        Args:
            content_sha: Optional content SHA to search for
            norm_text_sha: Optional normalized text SHA to search for

        Returns:
            List of chunk IDs
        """
        async with self._lock:
            duplicates = []

            if content_sha and content_sha in self._content_sha_index:
                duplicates.append(self._content_sha_index[content_sha])

            if norm_text_sha and norm_text_sha in self._norm_text_sha_index:
                duplicates.extend(self._norm_text_sha_index[norm_text_sha])

            return list(set(duplicates))

    async def get_duplicate_groups(self) -> List[List[str]]:
        """
        Get groups of duplicate chunks.

        Returns:
            List of duplicate groups (each group is a list of chunk IDs)
        """
        async with self._lock:
            groups = []

            # Group by norm_text_sha
            for chunk_ids in self._norm_text_sha_index.values():
                if len(chunk_ids) > 1:
                    groups.append(list(chunk_ids))

            return groups

    async def get_statistics(self) -> DeduplicationStats:
        """
        Get deduplication statistics.

        Returns:
            DeduplicationStats
        """
        async with self._lock:
            # Calculate dedup rate
            total_processed = (
                self._stats.exact_duplicates +
                self._stats.fuzzy_duplicates +
                self._stats.unique_chunks
            )

            if total_processed > 0:
                self._stats.dedup_rate = (
                    (self._stats.exact_duplicates + self._stats.fuzzy_duplicates) /
                    total_processed
                )

            return self._stats

    async def reset_statistics(self) -> None:
        """Reset deduplication statistics."""
        async with self._lock:
            self._stats = DeduplicationStats()
            logger.info(
                "Deduplication statistics reset",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.deduplication_enforcer.DeduplicationEnforcer"
            )

    async def clear_indexes(self) -> None:
        """Clear all deduplication indexes."""
        async with self._lock:
            self._content_sha_index.clear()
            self._norm_text_sha_index.clear()
            self._chunk_metadata.clear()
            logger.info(
                "Deduplication indexes cleared",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.deduplication_enforcer.DeduplicationEnforcer"
            )

    async def _resolve_conflict(
        self,
        existing_chunk_id: str,
        new_chunk_id: str,
        new_metadata: Optional[Dict[str, Any]]
    ) -> DuplicateAction:
        """
        Resolve conflict between existing and new chunk.

        Args:
            existing_chunk_id: Existing chunk ID
            new_chunk_id: New chunk ID
            new_metadata: New chunk metadata

        Returns:
            DuplicateAction
        """
        existing_metadata = self._chunk_metadata.get(existing_chunk_id, {})

        if self.conflict_resolution == ConflictResolution.KEEP_EXISTING:
            return DuplicateAction.SKIP

        elif self.conflict_resolution == ConflictResolution.KEEP_NEW:
            return DuplicateAction.REPLACE

        elif self.conflict_resolution == ConflictResolution.KEEP_HIGHER_CONFIDENCE:
            existing_confidence = existing_metadata.get('confidence', 0.0)
            new_confidence = new_metadata.get('confidence', 0.0) if new_metadata else 0.0

            if new_confidence > existing_confidence:
                return DuplicateAction.REPLACE
            else:
                return DuplicateAction.SKIP

        elif self.conflict_resolution == ConflictResolution.KEEP_NEWER:
            existing_timestamp = existing_metadata.get('ingestion_timestamp')
            new_timestamp = new_metadata.get('ingestion_timestamp') if new_metadata else None

            if new_timestamp and existing_timestamp:
                if isinstance(new_timestamp, str):
                    new_timestamp = datetime.fromisoformat(new_timestamp)
                if isinstance(existing_timestamp, str):
                    existing_timestamp = datetime.fromisoformat(existing_timestamp)

                if new_timestamp > existing_timestamp:
                    return DuplicateAction.REPLACE

            return DuplicateAction.SKIP

        elif self.conflict_resolution == ConflictResolution.MERGE_METADATA:
            return DuplicateAction.MERGE

        else:
            return self.default_action

    async def merge_metadata(
        self,
        existing_metadata: Dict[str, Any],
        new_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge metadata from two chunks.

        Args:
            existing_metadata: Existing chunk metadata
            new_metadata: New chunk metadata

        Returns:
            Merged metadata
        """
        merged = existing_metadata.copy()

        # Merge custom metadata
        for key, value in new_metadata.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Merge dicts
                merged[key] = {**merged[key], **value}
            else:
                # Keep existing value for conflicts
                pass

        return merged

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get deduplication metrics.

        Returns:
            Metrics dictionary
        """
        return {
            'total_chunks_indexed': len(self._content_sha_index),
            'total_norm_text_groups': len(self._norm_text_sha_index),
            'fuzzy_threshold': self.fuzzy_threshold,
            'default_action': self.default_action.value,
            'conflict_resolution': self.conflict_resolution.value,
            'statistics': {
                'total_checks': self._stats.total_checks,
                'exact_duplicates': self._stats.exact_duplicates,
                'fuzzy_duplicates': self._stats.fuzzy_duplicates,
                'updates': self._stats.updates,
                'skipped': self._stats.skipped,
                'merged': self._stats.merged,
                'replaced': self._stats.replaced,
                'unique_chunks': self._stats.unique_chunks,
                'dedup_rate': self._stats.dedup_rate
            }
        }


# Global singleton
_deduplication_enforcer: Optional[DeduplicationEnforcer] = None
_enforcer_lock = asyncio.Lock()


async def get_deduplication_enforcer(
    fuzzy_threshold: float = 0.95,
    default_action: DuplicateAction = DuplicateAction.SKIP,
    conflict_resolution: ConflictResolution = ConflictResolution.KEEP_EXISTING
) -> DeduplicationEnforcer:
    """
    Get or create deduplication enforcer singleton.

    Args:
        fuzzy_threshold: Similarity threshold for fuzzy matching
        default_action: Default action for duplicates
        conflict_resolution: Conflict resolution strategy

    Returns:
        DeduplicationEnforcer instance
    """
    global _deduplication_enforcer

    async with _enforcer_lock:
        if _deduplication_enforcer is None:
            _deduplication_enforcer = DeduplicationEnforcer(
                fuzzy_threshold=fuzzy_threshold,
                default_action=default_action,
                conflict_resolution=conflict_resolution
            )
            logger.info(
                "DeduplicationEnforcer singleton created",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.deduplication_enforcer"
            )

        return _deduplication_enforcer
