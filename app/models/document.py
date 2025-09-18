"""
Document Storage Models for Revolutionary Knowledge Base System.

This module provides comprehensive document storage with:
- PostgreSQL for metadata and encrypted content
- ChromaDB for vector embeddings
- Unique UUIDs for each document
- Knowledge base isolation
- Secure blob storage
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean, LargeBinary, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, ConfigDict

from app.models.database.base import Base


class DocumentDB(Base):
    """
    PostgreSQL model for document storage.
    
    Stores document metadata and encrypted content blobs.
    Vector embeddings are stored separately in ChromaDB.
    """
    __tablename__ = "documents"
    
    # Primary key - unique UUID
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Knowledge base association
    knowledge_base_id = Column(String(255), nullable=False, index=True)
    
    # Document metadata
    title = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    
    # Content information
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    
    # Encrypted content blob (stored as binary)
    encrypted_content = Column(LargeBinary, nullable=True)
    
    # Processing status
    status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)
    
    # Chunking and embedding info
    chunk_count = Column(Integer, nullable=False, default=0)
    embedding_model = Column(String(100), nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    
    # Document classification
    document_type = Column(String(50), nullable=False, default="text")
    language = Column(String(10), nullable=False, default="en")
    
    # Metadata (searchable JSON) - renamed to avoid SQLAlchemy reserved word
    doc_metadata = Column(JSONB, nullable=False, default=dict)
    
    # Access control
    is_public = Column(Boolean, nullable=False, default=False)
    uploaded_by = Column(String(255), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Indexes for performance
    __table_args__ = (
        {'schema': 'rag'}
    )


class DocumentChunkDB(Base):
    """
    PostgreSQL model for document chunk metadata.
    
    Stores chunk information while actual vectors are in ChromaDB.
    This provides a bridge between PostgreSQL and ChromaDB.
    """
    __tablename__ = "document_chunks"
    
    # Primary key - unique UUID
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Document association
    document_id = Column(UUID(as_uuid=True), ForeignKey('rag.documents.id', ondelete='CASCADE'), nullable=False)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    
    # Position in original document
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    
    # ChromaDB reference
    chromadb_id = Column(String(255), nullable=False, unique=True, index=True)
    collection_name = Column(String(255), nullable=False)
    
    # Embedding info
    embedding_model = Column(String(100), nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    
    # Chunk metadata - renamed to avoid SQLAlchemy reserved word
    chunk_metadata = Column(JSONB, nullable=False, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    document = relationship("DocumentDB", backref="chunks")
    
    __table_args__ = (
        {'schema': 'rag'}
    )


# Pydantic models for API responses
class DocumentMetadata(BaseModel):
    """Document metadata model for API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    knowledge_base_id: str
    title: str
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    content_hash: str
    status: str
    processing_error: Optional[str] = None
    chunk_count: int
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    document_type: str
    language: str
    doc_metadata: Dict[str, Any]
    is_public: bool
    uploaded_by: str
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None


class DocumentChunkMetadata(BaseModel):
    """Document chunk metadata model for API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    start_char: int
    end_char: int
    chromadb_id: str
    collection_name: str
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    chunk_metadata: Dict[str, Any]
    created_at: datetime


class DocumentCreateRequest(BaseModel):
    """Request model for document creation."""
    title: Optional[str] = None
    doc_metadata: Optional[Dict[str, Any]] = None
    is_public: bool = False


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    document_id: str
    knowledge_base_id: str
    title: str
    filename: str
    status: str
    message: str


class DocumentSearchResult(BaseModel):
    """Document search result model."""
    document: DocumentMetadata
    chunks: List[DocumentChunkMetadata]
    relevance_score: float
    matched_chunks: int
