"""
RAG API Endpoints for Revolutionary Knowledge Management.

This module provides comprehensive API endpoints for RAG operations including
knowledge search, document ingestion, collection management, and system monitoring.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from pydantic import BaseModel, Field

import structlog
from app.services.rag_service import get_rag_service, RAGService
from app.services.knowledge_base_service import get_knowledge_base_service, KnowledgeBaseService
from app.core.auth import get_current_user
from app.models.user import User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


# Request/Response Models
class KnowledgeSearchRequest(BaseModel):
    """Request model for knowledge search."""
    query: str = Field(..., description="Search query")
    collection: Optional[str] = Field(default=None, description="Target collection")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    collection: Optional[str] = Field(default=None, description="Target collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    document_type: str = Field(default="text", description="Document type")


class CollectionCreateRequest(BaseModel):
    """Request model for collection creation."""
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    use_case: Optional[str] = Field(None, description="Use case category")
    tags: Optional[List[str]] = Field(default_factory=list, description="Collection tags")


class KnowledgeBaseCreateRequest(BaseModel):
    """Request model for creating a new knowledge base."""
    name: str = Field(..., description="Knowledge base name", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="Knowledge base description", max_length=500)
    use_case: str = Field(..., description="Use case category (e.g., 'customer_support', 'research', 'legal')")
    tags: Optional[List[str]] = Field(default_factory=list, description="Knowledge base tags")
    is_public: bool = Field(default=False, description="Whether the knowledge base is public")


class KnowledgeBaseUpdateRequest(BaseModel):
    """Request model for updating a knowledge base."""
    description: Optional[str] = Field(None, description="Knowledge base description")
    tags: Optional[List[str]] = Field(None, description="Knowledge base tags")
    is_public: Optional[bool] = Field(None, description="Whether the knowledge base is public")


class DocumentUploadRequest(BaseModel):
    """Request model for uploading documents to a knowledge base."""
    knowledge_base_id: str = Field(..., description="Target knowledge base ID")
    title: Optional[str] = Field(None, description="Document title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")


class KnowledgeSearchResponse(BaseModel):
    """Response model for knowledge search."""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    collection: str


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    document_id: str
    title: str
    collection: str
    chunks_created: int


class IngestionStatusResponse(BaseModel):
    """Response model for ingestion status."""
    job_id: str
    status: str
    progress: float
    file_name: str
    document_id: Optional[str]
    chunks_created: int
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Search the knowledge base for relevant information.
    
    This endpoint enables semantic search across the knowledge base with support for:
    - Collection-specific searches
    - Metadata filtering
    - Relevance scoring
    - Configurable result limits
    """
    try:
        result = await rag_service.search_knowledge(
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
            filters=request.filters
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Search failed"))
        
        return KnowledgeSearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Knowledge search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    request: DocumentIngestRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Ingest a document into the knowledge base.
    
    This endpoint processes and stores documents with:
    - Automatic text chunking
    - Embedding generation
    - Metadata enrichment
    - Collection organization
    """
    try:
        result = await rag_service.ingest_document(
            title=request.title,
            content=request.content,
            collection=request.collection,
            metadata={
                **request.metadata,
                "ingested_by": current_user.username,
                "user_id": str(current_user.id)
            },
            document_type=request.document_type
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Ingestion failed"))
        
        return DocumentIngestResponse(**result)
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Ingest a file into the knowledge base.
    
    This endpoint supports various file formats including:
    - PDF documents
    - Word documents (DOCX)
    - Plain text files
    - HTML files
    - JSON files
    """
    try:
        # Read file content
        content = await file.read()
        
        # Create ingestion job
        job_id = await rag_service.ingestion_pipeline.ingest_content(
            content=content,
            file_name=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            collection=collection,
            metadata={
                "uploaded_by": current_user.username,
                "user_id": str(current_user.id),
                "original_filename": file.filename,
                "file_size": len(content)
            }
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "File ingestion started",
            "filename": file.filename,
            "collection": collection
        }
        
    except Exception as e:
        logger.error(f"File ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingest/status/{job_id}")
async def get_ingestion_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get the status of a file ingestion job.
    
    Returns detailed information about the ingestion process including:
    - Current status (pending, processing, completed, failed)
    - Progress percentage
    - Error messages if any
    - Results summary
    """
    try:
        status = await rag_service.get_ingestion_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ingestion status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def list_collections(
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    List all available knowledge collections.
    
    Returns a list of all collections in the knowledge base with
    their names and basic metadata.
    """
    try:
        collections = await rag_service.get_collections()
        
        return {
            "success": True,
            "collections": collections,
            "total": len(collections)
        }
        
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections")
async def create_collection(
    request: CollectionCreateRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Create a new knowledge collection.
    
    Collections help organize knowledge by domain, project, or purpose.
    Each collection maintains its own vector space for optimized retrieval.
    """
    try:
        success = await rag_service.create_collection(request.name)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create collection")
        
        return {
            "success": True,
            "collection": request.name,
            "message": f"Collection '{request.name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_rag_stats(
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get comprehensive RAG system statistics.
    
    Returns detailed information about:
    - Knowledge base metrics
    - Ingestion pipeline status
    - Performance statistics
    - System health
    """
    try:
        stats = await rag_service.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# KNOWLEDGE BASE MANAGEMENT MODELS
# ============================================================================

class KnowledgeBaseCreateRequest(BaseModel):
    """Request model for creating a new knowledge base."""
    name: str = Field(..., min_length=1, max_length=100, description="Knowledge base name")
    description: Optional[str] = Field(None, max_length=500, description="Knowledge base description")
    use_case: str = Field(..., description="Use case category (e.g., 'customer_support', 'research', 'legal')")
    tags: Optional[List[str]] = Field(default=[], description="Tags for organization")
    is_public: bool = Field(default=False, description="Whether the knowledge base is public")


class KnowledgeBaseUpdateRequest(BaseModel):
    """Request model for updating a knowledge base."""
    description: Optional[str] = Field(None, max_length=500, description="Knowledge base description")
    tags: Optional[List[str]] = Field(None, description="Tags for organization")
    is_public: Optional[bool] = Field(None, description="Whether the knowledge base is public")


class DocumentUploadRequest(BaseModel):
    """Request model for uploading documents to a knowledge base."""
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional document metadata")


# ============================================================================
# KNOWLEDGE BASE MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/knowledge-bases")
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    🚀 Create a Revolutionary Knowledge Base!

    Each knowledge base gets its own RAG system with:
    - Dedicated ChromaDB collection
    - Individual document processing pipeline
    - Shared global embedding configuration
    """
    try:
        logger.info(f"📥 Creating knowledge base: {request.name}")
        logger.info(f"📋 Request data: {request}")

        # Create knowledge base using the knowledge base service
        kb_id = await kb_service.create_knowledge_base(
            name=request.name,
            description=request.description or "",
            use_case=request.use_case,
            tags=request.tags or [],
            is_public=request.is_public,
            created_by=str(current_user.id)
        )

        # Get global embedding configuration for response
        from app.rag.core.global_embedding_manager import get_global_embedding_config
        global_config = get_global_embedding_config()

        # Return knowledge base info
        knowledge_base = {
            "id": kb_id,
            "name": request.name,
            "description": request.description,
            "use_case": request.use_case,
            "tags": request.tags,
            "embedding_model": global_config.get("embedding_model", "global") if global_config else "global",
            "embedding_engine": global_config.get("embedding_engine", "") if global_config else "",
            "is_public": request.is_public,
            "created_by": current_user.id,
            "created_at": datetime.utcnow().isoformat(),
            "document_count": 0,
            "size_mb": 0.0
        }

        response = {
            "success": True,
            "knowledge_base": knowledge_base,
            "message": f"Knowledge base '{request.name}' created successfully with its own RAG system"
        }

        logger.info(f"🎉 Knowledge base created successfully: {kb_id}")
        logger.info(f"📤 Returning response: {response}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-bases")
async def list_knowledge_bases(
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    📚 List All Knowledge Bases

    Get a list of all knowledge bases with their metadata and statistics.
    Each knowledge base has its own RAG system and document collection.
    """
    try:
        knowledge_bases = await kb_service.list_knowledge_bases()

        return {
            "success": True,
            "knowledge_bases": knowledge_bases,
            "total_count": len(knowledge_bases)
        }

    except Exception as e:
        logger.error(f"Failed to list knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-bases/{kb_id}/documents")
async def upload_document_to_knowledge_base(
    kb_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}"),
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    📄 Upload Document to Knowledge Base

    Upload and process a document into a specific knowledge base.
    The document will be:
    - Processed using the knowledge base's RAG system
    - Embedded using the global embedding configuration
    - Stored in the knowledge base's dedicated collection
    """
    try:
        # Parse metadata
        try:
            doc_metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            doc_metadata = {}

        # Add upload info to metadata
        doc_metadata.update({
            "uploaded_by": str(current_user.id),
            "uploaded_at": datetime.utcnow().isoformat()
        })

        # Read file content
        content = await file.read()

        # Upload document using knowledge base service
        job_id = await kb_service.upload_document(
            kb_id=kb_id,
            content=content,
            filename=file.filename or "unknown",
            title=title,
            metadata=doc_metadata
        )

        return {
            "success": True,
            "job_id": job_id,
            "knowledge_base_id": kb_id,
            "filename": file.filename,
            "message": f"Document uploaded to knowledge base '{kb_id}' successfully"
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upload document to knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-bases/{kb_id}/search")
async def search_knowledge_base(
    kb_id: str,
    query: str = Form(...),
    top_k: int = Form(default=5),
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    🔍 Search Knowledge Base

    Search within a specific knowledge base using its dedicated RAG system.
    Returns relevant documents and chunks from that knowledge base only.
    """
    try:
        result = await kb_service.search_knowledge_base(
            kb_id=kb_id,
            query=query,
            top_k=top_k
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to search knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    🗑️ Delete Knowledge Base

    Delete a knowledge base and all its documents.
    This will remove the RAG system and ChromaDB collection.
    """
    try:
        success = await kb_service.delete_knowledge_base(kb_id)

        if success:
            return {
                "success": True,
                "message": f"Knowledge base '{kb_id}' deleted successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to delete knowledge base")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-bases/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    📖 Get Knowledge Base Details

    Get detailed information about a specific knowledge base.
    """
    try:
        knowledge_bases = await kb_service.list_knowledge_bases()
        kb = next((kb for kb in knowledge_bases if kb['id'] == kb_id), None)

        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        return {
            "success": True,
            "knowledge_base": kb
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-bases/{kb_id}/documents")
async def get_knowledge_base_documents(
    kb_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    📄 Get Knowledge Base Documents

    Get all documents in a specific knowledge base with revolutionary document management.
    """
    try:
        documents = await kb_service.get_documents(kb_id, limit, offset)

        return {
            "success": True,
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get documents for knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-bases/{kb_id}/documents/{doc_id}/chunks")
async def get_document_chunks(
    kb_id: str,
    doc_id: str,
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    🧠 Get Document Embedding Chunks

    Get the embedding chunks for a specific document to show how it was processed.
    Revolutionary visualization of document chunking and embedding.
    """
    try:
        chunks = await kb_service.get_document_chunks(kb_id, doc_id)

        return {
            "success": True,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get chunks for document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-bases/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: str,
    doc_id: str,
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    🗑️ Delete Document

    Delete a specific document from a knowledge base.
    Revolutionary document deletion with complete cleanup.
    """
    try:
        success = await kb_service.delete_document(kb_id, doc_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "success": True,
            "message": f"Document {doc_id} deleted successfully"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-bases/{kb_id}/documents")
async def upload_document_to_knowledge_base(
    kb_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}"),
    is_public: bool = Form(False),
    current_user: User = Depends(get_current_user),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service)
):
    """
    📤 Upload Document to Knowledge Base

    Upload a document to a specific knowledge base with revolutionary processing:
    - Secure encrypted storage in PostgreSQL
    - Intelligent chunking and embedding
    - ChromaDB vector storage
    - Unique UUID system
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Parse metadata
        try:
            doc_metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

        # Read file content
        content = await file.read()

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Upload document
        document_id = await kb_service.upload_document(
            kb_id=kb_id,
            content=content,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            uploaded_by=str(current_user.id),
            title=title,
            metadata=doc_metadata,
            is_public=is_public
        )

        return {
            "success": True,
            "document_id": document_id,
            "knowledge_base_id": kb_id,
            "filename": file.filename,
            "size": len(content),
            "message": "Document uploaded successfully and queued for processing"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upload document to knowledge base {kb_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))





