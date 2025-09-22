"""
🔥 REVOLUTIONARY SESSION DOCUMENT MODELS
========================================

Data models for the Revolutionary Session-Based Document Workspace.
Provides temporary document storage with session lifecycle management.

CORE FEATURES:
- Session-scoped document storage
- Temporary vector embeddings
- Automatic cleanup and expiration
- Integration with existing session management
- Revolutionary document intelligence support
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime, JSON, Text, Integer, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

logger = structlog.get_logger(__name__)

# Use existing Base from the application
try:
    from app.models.enhanced_user import Base
except ImportError:
    # Fallback if import fails
    Base = declarative_base()


class DocumentProcessingStatus(str, Enum):
    """Document processing status enumeration."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    MODIFIED = "modified"
    GENERATED = "generated"
    ERROR = "error"
    EXPIRED = "expired"


class SessionDocumentType(str, Enum):
    """Session document type enumeration."""
    UPLOADED = "uploaded"
    GENERATED = "generated"
    MODIFIED = "modified"
    TEMPORARY = "temporary"


# Pydantic Models for API and Business Logic
class SessionDocumentBase(BaseModel):
    """Base session document model."""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the document")
    file_size: int = Field(..., description="File size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    document_type: SessionDocumentType = Field(default=SessionDocumentType.UPLOADED)
    
    class Config:
        use_enum_values = True


class SessionDocumentCreate(SessionDocumentBase):
    """Model for creating session documents."""
    content: bytes = Field(..., description="Document content")
    session_id: str = Field(..., description="Session identifier")
    
    @validator('content')
    def validate_content(cls, v):
        if not v:
            raise ValueError("Document content cannot be empty")
        if len(v) > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("Document size cannot exceed 100MB")
        return v


class SessionDocumentResponse(SessionDocumentBase):
    """Model for session document responses."""
    document_id: str = Field(..., description="Unique document identifier")
    session_id: str = Field(..., description="Session identifier")
    processing_status: DocumentProcessingStatus = Field(default=DocumentProcessingStatus.UPLOADED)
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    download_url: Optional[str] = Field(None, description="Temporary download URL")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SessionDocumentQuery(BaseModel):
    """Model for querying session documents."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    include_content: bool = Field(default=False, description="Include document content in results")
    document_types: Optional[List[SessionDocumentType]] = Field(None, description="Filter by document types")
    
    class Config:
        use_enum_values = True


class SessionDocumentAnalysisRequest(BaseModel):
    """Model for document analysis requests."""
    document_id: str = Field(..., description="Document identifier")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")
    extract_forms: bool = Field(default=True, description="Extract form fields")
    extract_tables: bool = Field(default=True, description="Extract tables")
    ai_insights: bool = Field(default=True, description="Generate AI insights")


class SessionWorkspaceStats(BaseModel):
    """Model for session workspace statistics."""
    session_id: str = Field(..., description="Session identifier")
    total_documents: int = Field(..., description="Total number of documents")
    total_size: int = Field(..., description="Total size in bytes")
    document_types: Dict[str, int] = Field(..., description="Count by document type")
    processing_status: Dict[str, int] = Field(..., description="Count by processing status")
    created_at: datetime = Field(..., description="Workspace creation time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    expires_at: datetime = Field(..., description="Workspace expiration time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# SQLAlchemy Models for Database Storage
class SessionDocumentDB(Base):
    """Database model for session documents."""
    
    __tablename__ = "session_documents"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Document identification
    document_id = Column(String(255), nullable=False, unique=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    
    # Document metadata
    filename = Column(String(500), nullable=False)
    content_type = Column(String(200), nullable=False)
    file_size = Column(Integer, nullable=False)
    document_type = Column(String(50), nullable=False, default=SessionDocumentType.UPLOADED.value)
    processing_status = Column(String(50), nullable=False, default=DocumentProcessingStatus.UPLOADED.value)
    
    # Storage information
    storage_path = Column(String(1000), nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    
    # Metadata and analysis
    document_metadata = Column(JSON, default=dict)
    analysis_results = Column(JSON, default=dict)
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Relationships (if needed for future features)
    # user_session = relationship("UserSession", back_populates="documents")
    
    def __repr__(self):
        return f"<SessionDocument(id={self.document_id}, session={self.session_id}, filename='{self.filename}')>"
    
    def to_response_model(self) -> SessionDocumentResponse:
        """Convert to response model."""
        return SessionDocumentResponse(
            document_id=self.document_id,
            session_id=self.session_id,
            filename=self.filename,
            content_type=self.content_type,
            file_size=self.file_size,
            document_type=SessionDocumentType(self.document_type),
            processing_status=DocumentProcessingStatus(self.processing_status),
            metadata=self.document_metadata or {},
            uploaded_at=self.uploaded_at,
            expires_at=self.expires_at,
            last_accessed=self.last_accessed,
            analysis_results=self.analysis_results
        )


class SessionWorkspaceDB(Base):
    """Database model for session workspaces."""
    
    __tablename__ = "session_workspaces"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Workspace identification
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Workspace metadata
    total_documents = Column(Integer, default=0)
    total_size = Column(Integer, default=0)
    workspace_metadata = Column(JSON, default=dict)
    
    # Configuration
    max_documents = Column(Integer, default=100)
    max_size = Column(Integer, default=1024*1024*1024)  # 1GB default
    auto_cleanup = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    def __repr__(self):
        return f"<SessionWorkspace(session_id={self.session_id}, docs={self.total_documents})>"


# Dataclasses for Internal Processing
@dataclass
class SessionDocument:
    """Internal session document representation."""
    document_id: str
    session_id: str
    filename: str
    content: bytes
    content_type: str
    file_size: int
    document_type: SessionDocumentType
    processing_status: DocumentProcessingStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Optional[Dict[str, Any]] = None
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    storage_path: Optional[Path] = None
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.expires_at is None:
            # Default expiration: 24 hours from upload
            self.expires_at = self.uploaded_at + timedelta(hours=24)
        
        if self.last_accessed is None:
            self.last_accessed = self.uploaded_at
    
    @property
    def is_expired(self) -> bool:
        """Check if document is expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def time_until_expiry(self) -> timedelta:
        """Get time until expiration."""
        return self.expires_at - datetime.utcnow()
    
    def to_response_model(self) -> SessionDocumentResponse:
        """Convert to response model."""
        return SessionDocumentResponse(
            document_id=self.document_id,
            session_id=self.session_id,
            filename=self.filename,
            content_type=self.content_type,
            file_size=self.file_size,
            document_type=self.document_type,
            processing_status=self.processing_status,
            metadata=self.metadata,
            uploaded_at=self.uploaded_at,
            expires_at=self.expires_at,
            last_accessed=self.last_accessed,
            analysis_results=self.analysis_results
        )


@dataclass
class SessionDocumentWorkspace:
    """Session document workspace representation."""
    session_id: str
    documents: Dict[str, SessionDocument] = field(default_factory=dict)
    vector_store: Optional[Any] = None
    processing_jobs: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    max_documents: int = 100
    max_size: int = 1024 * 1024 * 1024  # 1GB
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.expires_at is None:
            # Default workspace expiration: 48 hours
            self.expires_at = self.created_at + timedelta(hours=48)
    
    @property
    def is_expired(self) -> bool:
        """Check if workspace is expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def total_documents(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    @property
    def total_size(self) -> int:
        """Get total size of all documents."""
        return sum(doc.file_size for doc in self.documents.values())
    
    @property
    def document_types_count(self) -> Dict[str, int]:
        """Get count by document type."""
        counts = {}
        for doc in self.documents.values():
            doc_type = doc.document_type.value
            counts[doc_type] = counts.get(doc_type, 0) + 1
        return counts
    
    @property
    def processing_status_count(self) -> Dict[str, int]:
        """Get count by processing status."""
        counts = {}
        for doc in self.documents.values():
            status = doc.processing_status.value
            counts[status] = counts.get(status, 0) + 1
        return counts
    
    def can_add_document(self, file_size: int) -> bool:
        """Check if document can be added to workspace."""
        if self.total_documents >= self.max_documents:
            return False
        if self.total_size + file_size > self.max_size:
            return False
        return True
    
    def to_stats_model(self) -> SessionWorkspaceStats:
        """Convert to stats model."""
        return SessionWorkspaceStats(
            session_id=self.session_id,
            total_documents=self.total_documents,
            total_size=self.total_size,
            document_types=self.document_types_count,
            processing_status=self.processing_status_count,
            created_at=self.created_at,
            last_activity=self.last_accessed,
            expires_at=self.expires_at
        )


# Exception Classes
class SessionDocumentError(Exception):
    """Base exception for session document operations."""
    pass


class SessionDocumentNotFoundError(SessionDocumentError):
    """Raised when session document is not found."""
    pass


class SessionWorkspaceNotFoundError(SessionDocumentError):
    """Raised when session workspace is not found."""
    pass


class SessionDocumentExpiredError(SessionDocumentError):
    """Raised when session document has expired."""
    pass


class SessionWorkspaceFullError(SessionDocumentError):
    """Raised when session workspace is full."""
    pass


logger.info("🔥 Revolutionary Session Document Models initialized")
