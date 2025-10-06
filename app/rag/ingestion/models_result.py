"""
Production-grade result models for document processing pipeline.

This module defines the uniform schema that all processors must return,
ensuring type safety, validation, and consistent interfaces across the
entire ingestion pipeline.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingStage(str, Enum):
    """Processing stages for error tracking."""
    INTAKE = "intake"
    VALIDATION = "validation"
    EXTRACTION = "extraction"
    OCR = "ocr"
    TRANSCRIPTION = "transcription"
    CHUNKING = "chunking"
    NORMALIZATION = "normalization"
    EMBEDDING = "embedding"
    INDEXING = "indexing"


class ErrorCode(str, Enum):
    """Standard error codes for processor failures."""
    # Intake errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_MIME_TYPE = "INVALID_MIME_TYPE"
    BLOCKED_EXTENSION = "BLOCKED_EXTENSION"
    MALWARE_DETECTED = "MALWARE_DETECTED"
    
    # Processing errors
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    OCR_FAILED = "OCR_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    PARSING_FAILED = "PARSING_FAILED"
    ENCODING_ERROR = "ENCODING_ERROR"
    
    # Resource errors
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    SUBPROCESS_FAILED = "SUBPROCESS_FAILED"
    
    # Data errors
    EMPTY_CONTENT = "EMPTY_CONTENT"
    INVALID_FORMAT = "INVALID_FORMAT"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    
    # System errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ProcessorError(BaseModel):
    """
    Structured error information from document processing.
    
    Provides detailed, actionable error information that can be used
    for debugging, retry logic, and user feedback.
    """
    code: ErrorCode = Field(..., description="Standard error code")
    message: str = Field(..., description="Human-readable error message")
    stage: ProcessingStage = Field(..., description="Stage where error occurred")
    retriable: bool = Field(default=False, description="Whether this error is retriable")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When error occurred")
    
    class Config:
        use_enum_values = True


class ImageData(BaseModel):
    """
    Extracted image data from document processing.
    
    Stores image content, metadata, and OCR results.
    """
    data: Optional[str] = Field(None, description="Base64-encoded image data")
    format: Optional[str] = Field(None, description="Image format (PNG, JPEG, etc.)")
    width: Optional[int] = Field(None, ge=0, description="Image width in pixels")
    height: Optional[int] = Field(None, ge=0, description="Image height in pixels")
    ocr_text: Optional[str] = Field(None, description="Text extracted via OCR")
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="OCR confidence score")
    page_number: Optional[int] = Field(None, ge=1, description="Page number if from document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional image metadata")


class DocumentStructure(BaseModel):
    """
    Document structure information.
    
    Captures the logical structure of the document for better chunking
    and retrieval.
    """
    type: str = Field(..., description="Document type (text, pdf, image, video, etc.)")
    has_text: bool = Field(default=True, description="Whether document contains text")
    has_images: bool = Field(default=False, description="Whether document contains images")
    has_tables: bool = Field(default=False, description="Whether document contains tables")
    has_code: bool = Field(default=False, description="Whether document contains code")
    page_count: Optional[int] = Field(None, ge=0, description="Number of pages")
    section_count: Optional[int] = Field(None, ge=0, description="Number of sections")
    headings: List[str] = Field(default_factory=list, description="Document headings")
    toc: List[Dict[str, Any]] = Field(default_factory=list, description="Table of contents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional structure metadata")


class ProcessResult(BaseModel):
    """
    Uniform result schema for all document processors.
    
    This is the contract that every processor must fulfill. It ensures
    consistent handling of results across the entire pipeline.
    """
    # Core content
    text: str = Field(..., description="Extracted text content")
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata (file info, processing info, etc.)"
    )
    
    # Multi-modal data
    images: List[ImageData] = Field(
        default_factory=list,
        description="Extracted images with OCR results"
    )
    
    # Structure information
    structure: DocumentStructure = Field(
        ...,
        description="Document structure information"
    )
    
    # Language and quality
    language: str = Field(
        default="unknown",
        description="Detected language code (ISO 639-1)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Processing confidence score"
    )
    
    # Error tracking
    errors: List[ProcessorError] = Field(
        default_factory=list,
        description="Non-fatal errors encountered during processing"
    )
    
    # Processing metadata
    processor_name: Optional[str] = Field(None, description="Name of processor used")
    processing_time_ms: Optional[float] = Field(None, ge=0, description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure text is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Extracted text cannot be empty")
        return v
    
    @field_validator('language')
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        """Validate language code format."""
        if v and v != "unknown":
            # Basic validation - should be 2-3 character code
            if not (2 <= len(v) <= 3 and v.isalpha()):
                raise ValueError(f"Invalid language code: {v}")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate consistency across fields."""
        # If structure says has_images, should have images
        if self.structure.has_images and not self.images:
            # This is a warning, not an error - add to errors list
            self.errors.append(ProcessorError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Structure indicates images but none extracted",
                stage=ProcessingStage.EXTRACTION,
                retriable=False,
                details={"has_images": True, "image_count": 0}
            ))
        
        return self
    
    def has_errors(self) -> bool:
        """Check if any errors occurred during processing."""
        return len(self.errors) > 0
    
    def has_fatal_errors(self) -> bool:
        """Check if any non-retriable errors occurred."""
        return any(not err.retriable for err in self.errors)
    
    def get_error_summary(self) -> str:
        """Get human-readable summary of errors."""
        if not self.errors:
            return "No errors"
        
        error_counts = {}
        for err in self.errors:
            error_counts[err.code] = error_counts.get(err.code, 0) + 1
        
        summary_parts = [f"{code}: {count}" for code, count in error_counts.items()]
        return ", ".join(summary_parts)


class ValidationResult(BaseModel):
    """
    Result from intake validation.
    
    Used by IntakeGuard to communicate validation results.
    """
    valid: bool = Field(..., description="Whether file passed validation")
    mime_type: Optional[str] = Field(None, description="Detected MIME type")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    errors: List[ProcessorError] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional validation metadata")
    
    def add_error(self, code: ErrorCode, message: str, stage: ProcessingStage = ProcessingStage.VALIDATION, **details):
        """Add a validation error."""
        self.valid = False
        self.errors.append(ProcessorError(
            code=code,
            message=message,
            stage=stage,
            retriable=False,
            details=details
        ))
    
    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)


class ChunkMetadata(BaseModel):
    """
    Metadata for a document chunk.
    
    Enriched metadata that supports deduplication, provenance tracking,
    and advanced retrieval.
    """
    # Identity
    doc_id: str = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., ge=0, description="Chunk index in document")
    
    # Hashes for deduplication
    content_sha: str = Field(..., description="SHA-256 of chunk content")
    norm_text_sha: str = Field(..., description="SHA-256 of normalized text")
    
    # Provenance
    source_uri: Optional[str] = Field(None, description="Source URI (e.g., s3://bucket/file.pdf#page=5)")
    section_path: Optional[str] = Field(None, description="Section path (e.g., '2.1 Pricing')")
    page: Optional[int] = Field(None, ge=1, description="Page number")
    
    # Content metadata
    lang: str = Field(default="en", description="Language code")
    char_count: int = Field(..., ge=0, description="Character count")
    token_count: Optional[int] = Field(None, ge=0, description="Token count (if available)")
    
    # Semantic metadata
    entities: List[str] = Field(default_factory=list, description="Named entities")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    
    # Access control
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels for filtering (project, security, etc.)")
    
    # Versioning
    ts_ingested: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp")
    version: int = Field(default=1, ge=1, description="Chunk version")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    
    # Additional metadata
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

