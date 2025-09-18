"""
Revolutionary RAG File Upload System.

This module provides comprehensive file upload endpoints for the RAG system
with real-time processing, progress tracking, and multi-format support.
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import mimetypes
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from app.rag.ingestion.pipeline import IngestionPipeline, IngestionConfig
from app.rag.core.knowledge_base import KnowledgeBase, KnowledgeConfig
from app.services.rag_service import RAGService
from app.core.dependencies import get_current_user

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/rag/upload", tags=["RAG Upload"])

# Global RAG service instance
rag_service: Optional[RAGService] = None


class FileUploadResponse(BaseModel):
    """Response model for file upload."""
    success: bool
    job_id: str
    file_name: str
    file_size: int
    mime_type: str
    collection: str
    message: str
    estimated_processing_time: Optional[int] = None


class UploadProgress(BaseModel):
    """Upload and processing progress."""
    job_id: str
    status: str  # "uploading", "processing", "completed", "failed"
    progress_percent: float
    current_step: str
    total_steps: int
    completed_steps: int
    file_name: str
    collection: str
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    document_id: Optional[str] = None
    chunks_created: Optional[int] = None


class KnowledgeBaseInfo(BaseModel):
    """Knowledge base information."""
    name: str
    collection: str
    description: str
    document_count: int
    total_chunks: int
    created_at: datetime
    last_updated: datetime
    size_mb: float
    embedding_model: str


# In-memory progress tracking (in production, use Redis or database)
upload_progress: Dict[str, UploadProgress] = {}


async def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
        await rag_service.initialize()
    return rag_service


@router.post("/file", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    description: Optional[str] = Form(default=None),
    chunk_size: Optional[int] = Form(default=1000),
    chunk_overlap: Optional[int] = Form(default=200),
    extract_metadata: bool = Form(default=True),
    current_user: Dict = Depends(get_current_user)
):
    """
    Upload a file for RAG processing.
    
    Supports multiple file formats:
    - PDF documents
    - Word documents (DOCX)
    - Text files (TXT, MD)
    - CSV files
    - JSON files
    - And more...
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (limit to 100MB)
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # Validate file type
        supported_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
            "application/xml",
            "text/html"
        ]
        
        if mime_type not in supported_types:
            raise HTTPException(
                status_code=415, 
                detail=f"Unsupported file type: {mime_type}. Supported types: {', '.join(supported_types)}"
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create progress tracker
        progress = UploadProgress(
            job_id=job_id,
            status="uploading",
            progress_percent=0.0,
            current_step="File uploaded, preparing for processing",
            total_steps=5,
            completed_steps=0,
            file_name=file.filename,
            collection=collection,
            started_at=datetime.utcnow()
        )
        upload_progress[job_id] = progress
        
        # Estimate processing time based on file size
        estimated_time = min(max(file_size // (1024 * 1024), 5), 300)  # 5 seconds to 5 minutes
        
        # Start background processing
        background_tasks.add_task(
            process_uploaded_file,
            job_id=job_id,
            file_content=file_content,
            file_name=file.filename,
            mime_type=mime_type,
            collection=collection,
            description=description,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_metadata=extract_metadata,
            user_id=current_user.get("id", "anonymous")
        )
        
        logger.info(
            "File upload initiated",
            job_id=job_id,
            file_name=file.filename,
            file_size=file_size,
            mime_type=mime_type,
            collection=collection,
            user_id=current_user.get("id")
        )
        
        return FileUploadResponse(
            success=True,
            job_id=job_id,
            file_name=file.filename,
            file_size=file_size,
            mime_type=mime_type,
            collection=collection,
            message="File uploaded successfully. Processing started.",
            estimated_processing_time=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/progress/{job_id}", response_model=UploadProgress)
async def get_upload_progress(job_id: str):
    """Get upload and processing progress for a job."""
    if job_id not in upload_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return upload_progress[job_id]


@router.get("/collections", response_model=List[KnowledgeBaseInfo])
async def list_knowledge_bases(current_user: Dict = Depends(get_current_user)):
    """List all available knowledge bases/collections."""
    try:
        service = await get_rag_service()
        
        # Get collection information
        collections = await service.list_collections()
        
        knowledge_bases = []
        for collection_name in collections:
            stats = await service.get_collection_stats(collection_name)
            
            knowledge_bases.append(KnowledgeBaseInfo(
                name=collection_name.replace("_", " ").title(),
                collection=collection_name,
                description=f"Knowledge base: {collection_name}",
                document_count=stats.get("document_count", 0),
                total_chunks=stats.get("chunk_count", 0),
                created_at=stats.get("created_at", datetime.utcnow()),
                last_updated=stats.get("last_updated", datetime.utcnow()),
                size_mb=stats.get("size_mb", 0.0),
                embedding_model=stats.get("embedding_model", "default")
            ))
        
        return knowledge_bases
        
    except Exception as e:
        logger.error(f"Failed to list knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{collection_name}")
async def create_knowledge_base(
    collection_name: str,
    description: Optional[str] = None,
    embedding_model: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new knowledge base/collection."""
    try:
        service = await get_rag_service()
        
        # Create collection
        success = await service.create_collection(
            collection_name,
            description=description,
            embedding_model=embedding_model
        )
        
        if success:
            logger.info(f"Knowledge base created: {collection_name}")
            return {"success": True, "message": f"Knowledge base '{collection_name}' created successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create knowledge base")
            
    except Exception as e:
        logger.error(f"Failed to create knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_knowledge_base(
    collection_name: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a knowledge base/collection."""
    try:
        service = await get_rag_service()
        
        # Delete collection
        success = await service.delete_collection(collection_name)
        
        if success:
            logger.info(f"Knowledge base deleted: {collection_name}")
            return {"success": True, "message": f"Knowledge base '{collection_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to delete knowledge base")
            
    except Exception as e:
        logger.error(f"Failed to delete knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_uploaded_file(
    job_id: str,
    file_content: bytes,
    file_name: str,
    mime_type: str,
    collection: str,
    description: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    extract_metadata: bool,
    user_id: str
):
    """Background task to process uploaded file."""
    progress = upload_progress[job_id]
    
    try:
        # Step 1: Initialize RAG service
        progress.current_step = "Initializing RAG service"
        progress.progress_percent = 10.0
        progress.completed_steps = 1
        
        service = await get_rag_service()
        
        # Step 2: Process file content
        progress.current_step = "Processing file content"
        progress.progress_percent = 30.0
        progress.completed_steps = 2
        
        # Save file temporarily
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{job_id}_{file_name}"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Step 3: Ingest document
        progress.current_step = "Ingesting document into knowledge base"
        progress.progress_percent = 60.0
        progress.completed_steps = 3
        
        result = await service.ingest_file(
            file_path=str(temp_file_path),
            collection=collection,
            metadata={
                "description": description,
                "uploaded_by": user_id,
                "upload_job_id": job_id,
                "original_filename": file_name,
                "mime_type": mime_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
        
        # Step 4: Clean up
        progress.current_step = "Finalizing processing"
        progress.progress_percent = 90.0
        progress.completed_steps = 4
        
        # Remove temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
        
        # Step 5: Complete
        progress.status = "completed"
        progress.current_step = "Processing completed successfully"
        progress.progress_percent = 100.0
        progress.completed_steps = 5
        progress.completed_at = datetime.utcnow()
        progress.document_id = result.get("document_id")
        progress.chunks_created = result.get("chunks_created")
        
        logger.info(
            "File processing completed",
            job_id=job_id,
            document_id=result.get("document_id"),
            chunks_created=result.get("chunks_created")
        )
        
    except Exception as e:
        progress.status = "failed"
        progress.error_message = str(e)
        progress.completed_at = datetime.utcnow()
        
        logger.error(f"File processing failed for job {job_id}: {str(e)}")
        
        # Clean up temporary file on error
        temp_file_path = Path("./temp_uploads") / f"{job_id}_{file_name}"
        if temp_file_path.exists():
            temp_file_path.unlink()
