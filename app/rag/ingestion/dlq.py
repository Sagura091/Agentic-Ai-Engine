"""
Dead Letter Queue (DLQ) for failed ingestion jobs.

This module provides a DLQ for jobs that fail processing:
- Persistent storage of failed jobs
- Retry logic with exponential backoff
- Failure categorization
- Manual intervention support
- Metrics and monitoring
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import asyncio

import structlog
import aiofiles

logger = structlog.get_logger(__name__)


class FailureReason(str, Enum):
    """Reasons for job failure."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class DLQEntry:
    """Dead letter queue entry."""
    job_id: str
    file_path: str
    failure_reason: FailureReason
    error_message: str
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    last_retry_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "file_path": self.file_path,
            "failure_reason": self.failure_reason.value,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "is_resolved": self.is_resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DLQEntry":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            file_path=data["file_path"],
            failure_reason=FailureReason(data["failure_reason"]),
            error_message=data["error_message"],
            error_details=data.get("error_details", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            last_retry_at=datetime.fromisoformat(data["last_retry_at"]) if data.get("last_retry_at") else None,
            next_retry_at=datetime.fromisoformat(data["next_retry_at"]) if data.get("next_retry_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            is_resolved=data.get("is_resolved", False),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolution_notes=data.get("resolution_notes")
        )
    
    def can_retry(self) -> bool:
        """Check if entry can be retried."""
        if self.is_resolved:
            return False
        
        if self.retry_count >= self.max_retries:
            return False
        
        if self.next_retry_at and datetime.utcnow() < self.next_retry_at:
            return False
        
        return True
    
    def schedule_retry(self, base_delay_seconds: int = 60):
        """
        Schedule next retry with exponential backoff.
        
        Args:
            base_delay_seconds: Base delay in seconds
        """
        delay = base_delay_seconds * (2 ** self.retry_count)
        self.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
        self.updated_at = datetime.utcnow()


class DeadLetterQueue:
    """
    Dead Letter Queue for failed ingestion jobs.
    
    Features:
    - Persistent storage
    - Automatic retry with exponential backoff
    - Failure categorization
    - Manual resolution
    - Metrics tracking
    """
    
    def __init__(self, storage_path: Path, max_retries: int = 3):
        """
        Initialize DLQ.
        
        Args:
            storage_path: Path to DLQ storage directory
            max_retries: Maximum retry attempts
        """
        self.storage_path = Path(storage_path)
        self.max_retries = max_retries
        
        # In-memory cache
        self._entries: Dict[str, DLQEntry] = {}
        self._lock = asyncio.Lock()
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "DeadLetterQueue initialized",
            storage_path=str(self.storage_path),
            max_retries=max_retries
        )
    
    async def add(
        self,
        job_id: str,
        file_path: str,
        failure_reason: FailureReason,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DLQEntry:
        """
        Add entry to DLQ.
        
        Args:
            job_id: Job ID
            file_path: File path
            failure_reason: Failure reason
            error_message: Error message
            error_details: Error details
            metadata: Additional metadata
            
        Returns:
            DLQEntry
        """
        async with self._lock:
            entry = DLQEntry(
                job_id=job_id,
                file_path=file_path,
                failure_reason=failure_reason,
                error_message=error_message,
                error_details=error_details or {},
                max_retries=self.max_retries,
                metadata=metadata or {}
            )
            
            # Schedule first retry
            entry.schedule_retry()
            
            # Store in memory
            self._entries[job_id] = entry
            
            # Persist to disk
            await self._persist_entry(entry)
            
            logger.warning(
                "Job added to DLQ",
                job_id=job_id,
                failure_reason=failure_reason.value,
                error_message=error_message,
                next_retry_at=entry.next_retry_at.isoformat() if entry.next_retry_at else None
            )
            
            return entry
    
    async def get(self, job_id: str) -> Optional[DLQEntry]:
        """
        Get DLQ entry.
        
        Args:
            job_id: Job ID
            
        Returns:
            DLQEntry if found
        """
        async with self._lock:
            return self._entries.get(job_id)
    
    async def get_all(self, include_resolved: bool = False) -> List[DLQEntry]:
        """
        Get all DLQ entries.
        
        Args:
            include_resolved: Include resolved entries
            
        Returns:
            List of DLQEntry
        """
        async with self._lock:
            entries = list(self._entries.values())
            
            if not include_resolved:
                entries = [e for e in entries if not e.is_resolved]
            
            return entries
    
    async def get_retryable(self) -> List[DLQEntry]:
        """
        Get entries that can be retried.
        
        Returns:
            List of retryable DLQEntry
        """
        async with self._lock:
            now = datetime.utcnow()
            
            retryable = [
                entry for entry in self._entries.values()
                if entry.can_retry() and (
                    entry.next_retry_at is None or entry.next_retry_at <= now
                )
            ]
            
            return retryable
    
    async def mark_retry_attempted(self, job_id: str) -> bool:
        """
        Mark that retry was attempted.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successful
        """
        async with self._lock:
            entry = self._entries.get(job_id)
            
            if not entry:
                return False
            
            entry.retry_count += 1
            entry.last_retry_at = datetime.utcnow()
            
            # Schedule next retry if not exhausted
            if entry.retry_count < entry.max_retries:
                entry.schedule_retry()
            else:
                entry.next_retry_at = None
            
            entry.updated_at = datetime.utcnow()
            
            # Persist changes
            await self._persist_entry(entry)
            
            logger.info(
                "Retry attempted",
                job_id=job_id,
                retry_count=entry.retry_count,
                max_retries=entry.max_retries,
                next_retry_at=entry.next_retry_at.isoformat() if entry.next_retry_at else None
            )
            
            return True
    
    async def resolve(self, job_id: str, resolution_notes: Optional[str] = None) -> bool:
        """
        Mark entry as resolved.
        
        Args:
            job_id: Job ID
            resolution_notes: Resolution notes
            
        Returns:
            True if successful
        """
        async with self._lock:
            entry = self._entries.get(job_id)
            
            if not entry:
                return False
            
            entry.is_resolved = True
            entry.resolved_at = datetime.utcnow()
            entry.resolution_notes = resolution_notes
            entry.updated_at = datetime.utcnow()
            
            # Persist changes
            await self._persist_entry(entry)
            
            logger.info(
                "DLQ entry resolved",
                job_id=job_id,
                resolution_notes=resolution_notes
            )
            
            return True
    
    async def remove(self, job_id: str) -> bool:
        """
        Remove entry from DLQ.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if successful
        """
        async with self._lock:
            if job_id not in self._entries:
                return False
            
            del self._entries[job_id]
            
            # Remove from disk
            entry_path = self.storage_path / f"{job_id}.json"
            if entry_path.exists():
                entry_path.unlink()
            
            logger.info("DLQ entry removed", job_id=job_id)
            
            return True
    
    async def load_from_disk(self) -> int:
        """
        Load entries from disk.
        
        Returns:
            Number of entries loaded
        """
        async with self._lock:
            count = 0
            
            for entry_file in self.storage_path.glob("*.json"):
                try:
                    async with aiofiles.open(entry_file, 'r') as f:
                        data = json.loads(await f.read())
                    
                    entry = DLQEntry.from_dict(data)
                    self._entries[entry.job_id] = entry
                    count += 1
                    
                except Exception as e:
                    logger.error(
                        "Failed to load DLQ entry",
                        file=str(entry_file),
                        error=str(e)
                    )
            
            logger.info("DLQ entries loaded from disk", count=count)
            
            return count
    
    async def _persist_entry(self, entry: DLQEntry):
        """Persist entry to disk."""
        entry_path = self.storage_path / f"{entry.job_id}.json"
        
        try:
            async with aiofiles.open(entry_path, 'w') as f:
                await f.write(json.dumps(entry.to_dict(), indent=2))
        except Exception as e:
            logger.error(
                "Failed to persist DLQ entry",
                job_id=entry.job_id,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get DLQ statistics.
        
        Returns:
            Statistics dictionary
        """
        total = len(self._entries)
        resolved = sum(1 for e in self._entries.values() if e.is_resolved)
        unresolved = total - resolved
        
        retryable = sum(1 for e in self._entries.values() if e.can_retry())
        exhausted = sum(
            1 for e in self._entries.values()
            if not e.is_resolved and e.retry_count >= e.max_retries
        )
        
        by_reason = {}
        for entry in self._entries.values():
            if not entry.is_resolved:
                reason = entry.failure_reason.value
                by_reason[reason] = by_reason.get(reason, 0) + 1
        
        return {
            "total_entries": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "retryable": retryable,
            "exhausted": exhausted,
            "by_reason": by_reason
        }


# Global DLQ instance
_dlq: Optional[DeadLetterQueue] = None
_dlq_lock = asyncio.Lock()


async def get_dlq(storage_path: Path, max_retries: int = 3) -> DeadLetterQueue:
    """
    Get global DLQ instance.
    
    Args:
        storage_path: Storage path
        max_retries: Maximum retries
        
    Returns:
        DeadLetterQueue instance
    """
    global _dlq
    
    if _dlq is None:
        async with _dlq_lock:
            if _dlq is None:
                _dlq = DeadLetterQueue(storage_path, max_retries)
                await _dlq.load_from_disk()
    
    return _dlq

