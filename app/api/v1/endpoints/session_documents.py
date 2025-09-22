"""
🔥 REVOLUTIONARY SESSION DOCUMENT API ENDPOINTS - PHASE 3
========================================================

FastAPI endpoints for session-based document processing with hybrid architecture.
Provides REST API access to the revolutionary session document system.

REVOLUTIONARY FEATURES:
- Session-scoped document upload and management
- Hybrid architecture leveraging existing RAG infrastructure
- Multi-modal processing (text, images, video, audio)
- AI-powered document analysis and querying
- Complete session isolation (no permanent storage pollution)
- File upload support with streaming
- Comprehensive error handling and validation

Author: Revolutionary AI Agent System
Version: 3.0.0 - FastAPI REST API
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import structlog

# Import session document system
try:
    from app.services.session_document_manager import session_document_manager
    from app.models.session_document_models import (
        SessionDocumentCreate,
        SessionDocumentResponse,
        SessionDocumentQuery,
        SessionDocumentAnalysisRequest,
        SessionWorkspaceStats,
        SessionDocumentType,
        DocumentProcessingStatus
    )
    SESSION_SYSTEM_AVAILABLE = True
except ImportError:
    SESSION_SYSTEM_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/session-documents", tags=["Session Documents"])


# Request/Response Models
class SessionDocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    session_id: str = Field(..., description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class SessionDocumentQueryRequest(BaseModel):
    """Request model for document queries."""
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: bool = Field(default=True, description="Include document metadata")


class SessionDocumentAnalysisRequest(BaseModel):
    """Request model for document analysis."""
    session_id: str = Field(..., description="Session identifier")
    document_id: Optional[str] = Field(default=None, description="Specific document ID (optional)")
    analysis_type: str = Field(default="comprehensive", description="Analysis type")
    custom_prompt: Optional[str] = Field(default=None, description="Custom analysis prompt")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = ['comprehensive', 'summary', 'key_points', 'sentiment', 'custom']
        if v not in allowed_types:
            raise ValueError(f"Analysis type must be one of: {allowed_types}")
        return v


class SessionDocumentListRequest(BaseModel):
    """Request model for document listing."""
    session_id: str = Field(..., description="Session identifier")
    include_content: bool = Field(default=False, description="Include document content")
    include_metadata: bool = Field(default=True, description="Include document metadata")


class SessionDocumentDeleteRequest(BaseModel):
    """Request model for document deletion."""
    session_id: str = Field(..., description="Session identifier")
    document_id: Optional[str] = Field(default=None, description="Document ID to delete (optional)")


class SessionWorkspaceCreateRequest(BaseModel):
    """Request model for workspace creation."""
    session_id: str = Field(..., description="Session identifier")
    max_documents: Optional[int] = Field(default=None, description="Maximum documents allowed")
    max_size: Optional[int] = Field(default=None, description="Maximum total size in bytes")
    expires_in_hours: Optional[int] = Field(default=24, description="Expiration time in hours")


# Response Models
class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class SessionDocumentQueryResponse(BaseModel):
    """Response model for document queries."""
    success: bool = Field(..., description="Query success status")
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Original query")
    results_count: int = Field(..., description="Number of results found")
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    message: str = Field(..., description="Response message")


class SessionDocumentAnalysisResponse(BaseModel):
    """Response model for document analysis."""
    success: bool = Field(..., description="Analysis success status")
    session_id: str = Field(..., description="Session identifier")
    analysis_type: str = Field(..., description="Type of analysis performed")
    documents_analyzed: int = Field(..., description="Number of documents analyzed")
    results: List[Dict[str, Any]] = Field(..., description="Analysis results")
    message: str = Field(..., description="Response message")


# Dependency to check system availability
async def check_system_availability():
    """Check if session document system is available."""
    if not SESSION_SYSTEM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Session document system not available. Please ensure the system is properly configured."
        )
    
    if not session_document_manager.is_initialized:
        await session_document_manager.initialize()
    
    return session_document_manager


# Endpoints
@router.post("/workspaces", response_model=APIResponse)
async def create_workspace(
    request: SessionWorkspaceCreateRequest,
    manager = Depends(check_system_availability)
):
    """
    🚀 Create a new session document workspace.
    
    Creates a new workspace for session-based document processing with:
    - Session isolation
    - Configurable limits
    - Automatic expiration
    - Hybrid architecture integration
    """
    try:
        workspace = await manager.create_workspace(session_id=request.session_id)
        
        logger.info(
            "🏗️ Workspace created via API",
            session_id=request.session_id,
            max_documents=workspace.max_documents,
            max_size=workspace.max_size
        )
        
        return APIResponse(
            success=True,
            message=f"Workspace created successfully for session {request.session_id}",
            data={
                "session_id": workspace.session_id,
                "max_documents": workspace.max_documents,
                "max_size": workspace.max_size,
                "created_at": workspace.created_at.isoformat(),
                "expires_at": workspace.expires_at.isoformat() if workspace.expires_at else None
            }
        )
        
    except Exception as e:
        logger.error(f"Workspace creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=APIResponse)
async def upload_document(
    session_id: str = Form(..., description="Session identifier"),
    file: UploadFile = File(..., description="Document file to upload"),
    metadata: Optional[str] = Form(default="{}", description="JSON metadata"),
    manager = Depends(check_system_availability)
):
    """
    📤 Upload a document to session workspace.
    
    Uploads and processes documents with:
    - Multi-format support (PDF, Word, Excel, PowerPoint, images, etc.)
    - Hybrid RAG processing
    - Vector embedding generation
    - Session-scoped storage
    - Multi-modal processing capabilities
    """
    try:
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON metadata")
        
        # Read file content
        file_content = await file.read()
        
        # Upload document
        document_response = await manager.upload_document(
            session_id=session_id,
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            metadata=metadata_dict
        )
        
        logger.info(
            "📤 Document uploaded via API",
            session_id=session_id,
            document_id=document_response.document_id,
            filename=document_response.filename,
            size=document_response.file_size
        )
        
        return APIResponse(
            success=True,
            message=f"Document '{document_response.filename}' uploaded successfully",
            data={
                "document_id": document_response.document_id,
                "filename": document_response.filename,
                "content_type": document_response.content_type,
                "file_size": document_response.file_size,
                "processing_status": document_response.processing_status.value if hasattr(document_response.processing_status, 'value') else str(document_response.processing_status),
                "uploaded_at": document_response.uploaded_at.isoformat(),
                "metadata": document_response.metadata
            }
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=SessionDocumentQueryResponse)
async def query_documents(
    request: SessionDocumentQueryRequest,
    manager = Depends(check_system_availability)
):
    """
    🔍 Query session documents with natural language.
    
    Performs intelligent document search with:
    - Vector similarity search
    - Hybrid RAG architecture
    - Semantic understanding
    - Relevance scoring
    - Session-scoped results
    """
    try:
        results = await manager.query_session_documents(
            session_id=request.session_id,
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                "document_id": result["document_id"],
                "similarity_score": result.get("similarity_score", 0.0),
                "content_snippet": result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""),
                "chunk_index": result.get("chunk_index", 0)
            }
            
            if request.include_metadata:
                formatted_result["metadata"] = result.get("metadata", {})
            
            formatted_results.append(formatted_result)
        
        logger.info(
            "🔍 Document query executed via API",
            session_id=request.session_id,
            query=request.query,
            results_count=len(results)
        )
        
        return SessionDocumentQueryResponse(
            success=True,
            session_id=request.session_id,
            query=request.query,
            results_count=len(results),
            results=formatted_results,
            message=f"Found {len(results)} relevant document segments"
        )
        
    except Exception as e:
        logger.error(f"Document query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=SessionDocumentAnalysisResponse)
async def analyze_documents(
    request: SessionDocumentAnalysisRequest,
    manager = Depends(check_system_availability)
):
    """
    🧠 AI-powered document analysis and insights.

    Provides intelligent document analysis with:
    - Revolutionary Document Intelligence Tool integration
    - Multiple analysis types (summary, key points, sentiment, etc.)
    - Custom analysis prompts
    - Multi-document analysis
    - Session-scoped processing
    """
    try:
        # Check if Revolutionary Document Intelligence Tool is available
        if not manager.intelligence_tool:
            raise HTTPException(
                status_code=503,
                detail="Document intelligence tool not available. AI-powered analysis requires Revolutionary Document Intelligence Tool."
            )

        # Get documents for analysis
        if request.document_id:
            # Analyze specific document
            document = await manager.get_document(request.session_id, request.document_id)
            if not document:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {request.document_id}"
                )
            documents = [document]
        else:
            # Analyze all documents in session
            documents = await manager.list_session_documents(request.session_id)

        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found for analysis in session"
            )

        # Perform analysis using Revolutionary Document Intelligence Tool
        analysis_results = []
        for doc in documents:
            try:
                # Prepare analysis request
                if request.custom_prompt:
                    analysis_prompt = request.custom_prompt
                else:
                    analysis_prompts = {
                        "comprehensive": "Provide a comprehensive analysis of this document including key themes, important information, and insights.",
                        "summary": "Provide a concise summary of the main points and key information in this document.",
                        "key_points": "Extract and list the key points, important facts, and main takeaways from this document.",
                        "sentiment": "Analyze the sentiment and tone of this document, identifying emotional indicators and overall mood."
                    }
                    analysis_prompt = analysis_prompts.get(request.analysis_type, analysis_prompts["comprehensive"])

                # Use Revolutionary Document Intelligence Tool for analysis
                intelligence_input = f"analyze:{doc.filename}:{analysis_prompt}"
                analysis_result = await manager.intelligence_tool._arun(intelligence_input)

                analysis_results.append({
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "analysis_type": request.analysis_type,
                    "analysis": analysis_result,
                    "timestamp": datetime.utcnow().isoformat()
                })

            except Exception as e:
                logger.warning(f"Analysis failed for document {doc.document_id}: {e}")
                analysis_results.append({
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "analysis_type": request.analysis_type,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

        logger.info(
            "🧠 Document analysis completed via API",
            session_id=request.session_id,
            analysis_type=request.analysis_type,
            documents_analyzed=len(analysis_results)
        )

        return SessionDocumentAnalysisResponse(
            success=True,
            session_id=request.session_id,
            analysis_type=request.analysis_type,
            documents_analyzed=len(analysis_results),
            results=analysis_results,
            message=f"Analysis completed for {len(analysis_results)} documents"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/documents", response_model=APIResponse)
async def list_session_documents(
    session_id: str,
    include_content: bool = Query(default=False, description="Include document content"),
    include_metadata: bool = Query(default=True, description="Include document metadata"),
    manager = Depends(check_system_availability)
):
    """
    📋 List all documents in a session workspace.

    Retrieves session documents with:
    - Document metadata
    - Optional content preview
    - Processing status
    - Upload timestamps
    - Session statistics
    """
    try:
        # Get session documents
        documents = await manager.list_session_documents(session_id)

        # Format document information
        formatted_docs = []
        for doc in documents:
            doc_info = {
                "document_id": doc.document_id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "file_size": doc.file_size,
                "document_type": doc.document_type.value if hasattr(doc.document_type, 'value') else str(doc.document_type),
                "processing_status": doc.processing_status.value if hasattr(doc.processing_status, 'value') else str(doc.processing_status),
                "uploaded_at": doc.uploaded_at.isoformat()
            }

            if include_metadata:
                doc_info["metadata"] = doc.metadata

            if include_content:
                # Include content preview (first 500 characters)
                content_preview = str(doc.content)[:500]
                if len(str(doc.content)) > 500:
                    content_preview += "..."
                doc_info["content_preview"] = content_preview

            formatted_docs.append(doc_info)

        # Get workspace statistics
        try:
            workspace_stats = await manager.get_workspace_stats(session_id)
            stats_data = {
                "total_documents": workspace_stats.total_documents if workspace_stats else 0,
                "total_size": workspace_stats.total_size if workspace_stats else 0,
                "created_at": workspace_stats.created_at.isoformat() if workspace_stats else None,
                "expires_at": workspace_stats.expires_at.isoformat() if workspace_stats and workspace_stats.expires_at else None
            } if workspace_stats else None
        except Exception as e:
            logger.warning(f"Failed to get workspace stats: {e}")
            stats_data = None

        logger.info(
            "📋 Document listing completed via API",
            session_id=session_id,
            documents_count=len(documents)
        )

        return APIResponse(
            success=True,
            message=f"Found {len(documents)} documents in session workspace",
            data={
                "session_id": session_id,
                "documents_count": len(documents),
                "documents": formatted_docs,
                "workspace_stats": stats_data
            }
        )

    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}/documents/{document_id}", response_model=APIResponse)
async def delete_document(
    session_id: str,
    document_id: str,
    manager = Depends(check_system_availability)
):
    """
    🗑️ Delete a specific document from session workspace.

    Removes document with:
    - File system cleanup
    - Vector embedding removal
    - Session isolation maintenance
    - Complete cleanup
    """
    try:
        success = await manager.delete_document(session_id, document_id)

        if success:
            logger.info(
                "🗑️ Document deleted via API",
                session_id=session_id,
                document_id=document_id
            )

            return APIResponse(
                success=True,
                message=f"Document {document_id} deleted successfully",
                data={
                    "session_id": session_id,
                    "document_id": document_id,
                    "deleted_at": datetime.utcnow().isoformat()
                }
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found or failed to delete: {document_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=APIResponse)
async def cleanup_session_workspace(
    session_id: str,
    manager = Depends(check_system_availability)
):
    """
    🧹 Clean up entire session workspace.

    Removes all session data with:
    - All documents deletion
    - Vector embeddings cleanup
    - File system cleanup
    - Complete session isolation
    - No permanent storage pollution
    """
    try:
        success = await manager.cleanup_workspace(session_id)

        if success:
            logger.info(
                "🧹 Session workspace cleaned via API",
                session_id=session_id
            )

            return APIResponse(
                success=True,
                message="Session workspace cleaned successfully - all documents removed",
                data={
                    "session_id": session_id,
                    "action": "cleanup_workspace",
                    "cleaned_at": datetime.utcnow().isoformat()
                }
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session workspace not found or failed to cleanup: {session_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/stats", response_model=APIResponse)
async def get_session_statistics(
    session_id: str,
    manager = Depends(check_system_availability)
):
    """
    📊 Get comprehensive session workspace statistics.

    Provides detailed statistics including:
    - Document counts and sizes
    - Processing status
    - Workspace limits
    - Expiration information
    - Global system statistics
    """
    try:
        # Get workspace statistics
        workspace_stats = await manager.get_workspace_stats(session_id)

        # Get global manager statistics
        global_stats = manager.get_global_stats()

        # Get vector store statistics
        vector_stats = manager.vector_store.get_global_stats()

        logger.info(
            "📊 Statistics retrieved via API",
            session_id=session_id
        )

        return APIResponse(
            success=True,
            message="Session workspace statistics retrieved successfully",
            data={
                "session_id": session_id,
                "workspace_stats": {
                    "total_documents": workspace_stats.total_documents if workspace_stats else 0,
                    "total_size": workspace_stats.total_size if workspace_stats else 0,
                    "created_at": workspace_stats.created_at.isoformat() if workspace_stats else None,
                    "expires_at": workspace_stats.expires_at.isoformat() if workspace_stats and workspace_stats.expires_at else None
                } if workspace_stats else None,
                "global_stats": {
                    "active_workspaces": global_stats.get("active_workspaces", 0),
                    "total_documents": global_stats.get("total_documents", 0),
                    "successful_operations": global_stats.get("successful_operations", 0),
                    "hybrid_architecture": global_stats.get("hybrid_architecture", False),
                    "storage_type": global_stats.get("storage_type", "unknown"),
                    "permanent_storage": global_stats.get("permanent_storage", True)
                },
                "vector_stats": {
                    "active_sessions": vector_stats.get("active_sessions", 0),
                    "total_embeddings": vector_stats.get("total_embeddings", 0),
                    "hybrid_architecture": vector_stats.get("hybrid_architecture", False),
                    "storage_type": vector_stats.get("storage_type", "unknown")
                }
            }
        )

    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/documents/{document_id}", response_model=APIResponse)
async def get_document(
    session_id: str,
    document_id: str,
    include_content: bool = Query(default=False, description="Include full document content"),
    manager = Depends(check_system_availability)
):
    """
    📄 Get specific document information and content.

    Retrieves document with:
    - Complete metadata
    - Optional full content
    - Processing status
    - Upload information
    """
    try:
        document = await manager.get_document(session_id, document_id)

        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        doc_data = {
            "document_id": document.document_id,
            "session_id": document.session_id,
            "filename": document.filename,
            "content_type": document.content_type,
            "file_size": document.file_size,
            "document_type": document.document_type.value if hasattr(document.document_type, 'value') else str(document.document_type),
            "processing_status": document.processing_status.value if hasattr(document.processing_status, 'value') else str(document.processing_status),
            "uploaded_at": document.uploaded_at.isoformat(),
            "metadata": document.metadata
        }

        if include_content:
            doc_data["content"] = str(document.content) if document.content else None

        logger.info(
            "📄 Document retrieved via API",
            session_id=session_id,
            document_id=document_id
        )

        return APIResponse(
            success=True,
            message=f"Document {document_id} retrieved successfully",
            data=doc_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=APIResponse)
async def health_check():
    """
    🏥 Health check endpoint for session document system.

    Provides system health information including:
    - System availability
    - Component status
    - Configuration validation
    - Integration status
    """
    try:
        health_data = {
            "system_available": SESSION_SYSTEM_AVAILABLE,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.0.0",
            "architecture": "hybrid"
        }

        if SESSION_SYSTEM_AVAILABLE:
            # Check manager initialization
            if not session_document_manager.is_initialized:
                await session_document_manager.initialize()

            # Get system statistics
            global_stats = session_document_manager.get_global_stats()
            vector_stats = session_document_manager.vector_store.get_global_stats()

            health_data.update({
                "manager_initialized": session_document_manager.is_initialized,
                "hybrid_architecture": global_stats.get("hybrid_architecture", False),
                "rag_integration": global_stats.get("rag_integration", {}),
                "storage_type": global_stats.get("storage_type", "unknown"),
                "permanent_storage": global_stats.get("permanent_storage", True),
                "vector_search_enabled": vector_stats.get("vector_search_enabled", False),
                "active_workspaces": global_stats.get("active_workspaces", 0),
                "total_documents": global_stats.get("total_documents", 0)
            })

        return APIResponse(
            success=True,
            message="Session document system is healthy",
            data=health_data
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return APIResponse(
            success=False,
            message="Session document system health check failed",
            error=str(e),
            data={
                "system_available": SESSION_SYSTEM_AVAILABLE,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


# Export router
logger.info("🔥 Revolutionary Session Document API Endpoints ready!")
