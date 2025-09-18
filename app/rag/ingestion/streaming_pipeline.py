"""
Revolutionary Real-time Document Processing Pipeline for RAG 4.0.

This module provides streaming document ingestion with:
- Real-time document processing
- Incremental indexing
- Async processing queues
- Live index updates
- Progressive embedding generation
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncIterator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import structlog
from asyncio import Queue, Event
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from ..core.knowledge_base import Document, KnowledgeBase
from ..core.caching import get_rag_cache, CacheType
from ..core.embeddings import EmbeddingManager

logger = structlog.get_logger(__name__)


class ProcessingStatus(Enum):
    """Document processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingJob:
    """Document processing job."""
    job_id: str
    document: Document
    collection: str
    priority: int = 1
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.QUEUED
    progress: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    avg_processing_time: float = 0.0
    throughput_per_minute: float = 0.0
    queue_size: int = 0
    active_workers: int = 0


class StreamingDocumentProcessor:
    """
    Revolutionary streaming document processor.
    
    Features:
    - Real-time document processing
    - Async processing queues with priority
    - Incremental embedding generation
    - Live progress tracking
    - Fault tolerance and recovery
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedding_manager: EmbeddingManager,
        max_workers: int = 4,
        queue_size: int = 1000,
        batch_size: int = 10
    ):
        self.knowledge_base = knowledge_base
        self.embedding_manager = embedding_manager
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        
        # Processing queues
        self.processing_queue = Queue(maxsize=queue_size)
        self.priority_queue = Queue(maxsize=queue_size // 2)
        
        # Worker management
        self.workers = []
        self.worker_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.shutdown_event = Event()
        
        # Job tracking
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: Dict[str, ProcessingJob] = {}
        self.job_callbacks: Dict[str, List[Callable]] = {}
        
        # Metrics
        self.metrics = ProcessingMetrics()
        self.processing_times = []
        
        # Cache integration
        self.cache = None
    
    async def initialize(self) -> None:
        """Initialize the streaming processor."""
        try:
            # Initialize cache
            self.cache = await get_rag_cache()
            
            # Start worker tasks
            self.is_running = True
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self.workers.append(worker)
            
            # Start metrics collector
            asyncio.create_task(self._metrics_collector())
            
            logger.info(f"Streaming processor initialized with {self.max_workers} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize streaming processor: {str(e)}")
            raise
    
    async def submit_document(
        self,
        document: Document,
        collection: str,
        priority: int = 1,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit document for streaming processing.
        
        Args:
            document: Document to process
            collection: Target collection
            priority: Processing priority (higher = more urgent)
            callback: Optional callback for completion
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            document=document,
            collection=collection,
            priority=priority,
            metadata={
                "submitted_by": "streaming_processor",
                "document_size": len(document.content),
                "document_type": document.document_type
            }
        )
        
        # Store job
        self.active_jobs[job_id] = job
        
        # Register callback
        if callback:
            if job_id not in self.job_callbacks:
                self.job_callbacks[job_id] = []
            self.job_callbacks[job_id].append(callback)
        
        # Queue job based on priority
        if priority > 5:
            await self.priority_queue.put(job)
        else:
            await self.processing_queue.put(job)
        
        self.metrics.total_jobs += 1
        self.metrics.queue_size += 1
        
        logger.info(f"Document submitted for processing: {job_id}")
        return job_id
    
    async def submit_document_stream(
        self,
        document_stream: AsyncIterator[Document],
        collection: str,
        batch_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Submit stream of documents for batch processing.
        
        Args:
            document_stream: Async iterator of documents
            collection: Target collection
            batch_callback: Callback for batch completion
            
        Returns:
            List of job IDs
        """
        job_ids = []
        batch_count = 0
        
        async for document in document_stream:
            job_id = await self.submit_document(document, collection)
            job_ids.append(job_id)
            batch_count += 1
            
            # Process in batches
            if batch_count >= self.batch_size:
                if batch_callback:
                    await batch_callback(job_ids[-self.batch_size:])
                batch_count = 0
        
        # Process remaining documents
        if batch_count > 0 and batch_callback:
            await batch_callback(job_ids[-batch_count:])
        
        logger.info(f"Submitted {len(job_ids)} documents from stream")
        return job_ids
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get processing job status."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "error_message": job.error_message,
                "metadata": job.metadata
            }
        
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status.value,
                "progress": 100.0,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message,
                "metadata": job.metadata
            }
        
        return None
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics."""
        self.metrics.queue_size = self.processing_queue.qsize() + self.priority_queue.qsize()
        self.metrics.active_workers = len([w for w in self.workers if not w.done()])
        
        return asdict(self.metrics)
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker processing loop."""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get job from priority queue first, then regular queue
                job = None
                try:
                    job = await asyncio.wait_for(self.priority_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    try:
                        job = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                
                if job:
                    await self._process_job(job, worker_id)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_job(self, job: ProcessingJob, worker_id: str) -> None:
        """Process a single document job."""
        start_time = time.time()
        job.started_at = datetime.utcnow()
        job.status = ProcessingStatus.PROCESSING
        
        try:
            logger.info(f"Worker {worker_id} processing job {job.job_id}")
            
            # Step 1: Document chunking
            job.status = ProcessingStatus.CHUNKING
            job.progress = 20.0
            chunks = await self._chunk_document(job.document)
            
            # Step 2: Generate embeddings
            job.status = ProcessingStatus.EMBEDDING
            job.progress = 50.0
            embeddings = await self._generate_embeddings(chunks)
            
            # Step 3: Index in knowledge base
            job.status = ProcessingStatus.INDEXING
            job.progress = 80.0
            await self._index_document(job.document, chunks, embeddings, job.collection)
            
            # Step 4: Cache results
            await self._cache_results(job.document, chunks, embeddings)
            
            # Complete job
            job.status = ProcessingStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.utcnow()
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.metrics.completed_jobs += 1
            self.metrics.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            del self.active_jobs[job.job_id]
            
            # Execute callbacks
            await self._execute_callbacks(job.job_id, job)
            
            logger.info(f"Job {job.job_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            self.metrics.failed_jobs += 1
            
            logger.error(f"Job {job.job_id} failed: {str(e)}")
            
            # Move to completed jobs even if failed
            self.completed_jobs[job.job_id] = job
            del self.active_jobs[job.job_id]
            
            # Execute callbacks with error
            await self._execute_callbacks(job.job_id, job)
    
    async def _chunk_document(self, document: Document) -> List[str]:
        """Chunk document into smaller pieces."""
        # Simple chunking - in production use semantic chunking
        chunk_size = 1000
        chunks = []
        
        content = document.content
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        embeddings = []
        
        for chunk in chunks:
            # Check cache first
            if self.cache:
                cached_embedding = await self.cache.get(chunk, CacheType.EMBEDDING, use_similarity=True)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    continue
            
            # Generate new embedding
            embedding = await self.embedding_manager.embed_text(chunk)
            embeddings.append(embedding)
            
            # Cache the embedding
            if self.cache:
                await self.cache.set(chunk, embedding, CacheType.EMBEDDING)
        
        return embeddings
    
    async def _index_document(
        self,
        document: Document,
        chunks: List[str],
        embeddings: List[List[float]],
        collection: str
    ) -> None:
        """Index document in knowledge base."""
        # Add document with chunks and embeddings
        await self.knowledge_base.add_document_with_embeddings(
            document, chunks, embeddings, collection
        )
    
    async def _cache_results(
        self,
        document: Document,
        chunks: List[str],
        embeddings: List[List[float]]
    ) -> None:
        """Cache processing results."""
        if not self.cache:
            return
        
        # Cache document metadata
        doc_metadata = {
            "title": document.title,
            "document_type": document.document_type,
            "chunk_count": len(chunks),
            "processed_at": datetime.utcnow().isoformat()
        }
        
        await self.cache.set(
            f"doc_metadata:{document.id}",
            doc_metadata,
            CacheType.METADATA
        )
    
    async def _execute_callbacks(self, job_id: str, job: ProcessingJob) -> None:
        """Execute job completion callbacks."""
        if job_id in self.job_callbacks:
            callbacks = self.job_callbacks[job_id]
            for callback in callbacks:
                try:
                    await callback(job)
                except Exception as e:
                    logger.error(f"Callback execution failed: {str(e)}")
            
            # Clean up callbacks
            del self.job_callbacks[job_id]
    
    async def _metrics_collector(self) -> None:
        """Collect and update processing metrics."""
        while self.is_running:
            try:
                # Calculate throughput
                if len(self.processing_times) > 0:
                    recent_times = self.processing_times[-60:]  # Last 60 jobs
                    if recent_times:
                        avg_time = sum(recent_times) / len(recent_times)
                        self.metrics.throughput_per_minute = 60.0 / avg_time if avg_time > 0 else 0.0
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {str(e)}")
                await asyncio.sleep(30)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the processor."""
        logger.info("Shutting down streaming processor...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for workers to complete
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown thread pool
        self.worker_pool.shutdown(wait=True)
        
        logger.info("Streaming processor shutdown complete")


# Global processor instance
streaming_processor = None


async def get_streaming_processor(
    knowledge_base: KnowledgeBase,
    embedding_manager: EmbeddingManager
) -> StreamingDocumentProcessor:
    """Get the global streaming processor instance."""
    global streaming_processor
    
    if streaming_processor is None:
        streaming_processor = StreamingDocumentProcessor(knowledge_base, embedding_manager)
        await streaming_processor.initialize()
    
    return streaming_processor
