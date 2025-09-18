"""
Revolutionary Document Service.

This service provides comprehensive document management with:
- PostgreSQL storage for metadata and encrypted content
- ChromaDB integration for vector embeddings
- Unique UUID system for documents and chunks
- Knowledge base isolation
- Secure content encryption
- Revolutionary ingestion pipeline
"""

import hashlib
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import mimetypes
import structlog

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.orm import selectinload
from cryptography.fernet import Fernet

from app.models.document import (
    DocumentDB, DocumentChunkDB, DocumentMetadata, 
    DocumentChunkMetadata, DocumentCreateRequest,
    DocumentUploadResponse, DocumentSearchResult
)
from app.models.database.base import get_database_session
from app.rag.core.knowledge_base import Document, DocumentChunk
from app.rag.ingestion.pipeline import IngestionPipeline
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class DocumentEncryption:
    """Handles document content encryption/decryption."""
    
    def __init__(self):
        settings = get_settings()
        # Use a key from settings or generate one
        self.key = getattr(settings, 'DOCUMENT_ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.key)
    
    def encrypt_content(self, content: bytes) -> bytes:
        """Encrypt document content."""
        return self.cipher.encrypt(content)
    
    def decrypt_content(self, encrypted_content: bytes) -> bytes:
        """Decrypt document content."""
        return self.cipher.decrypt(encrypted_content)


class DocumentService:
    """
    Revolutionary Document Service.
    
    Manages the complete document lifecycle:
    1. Upload and validation
    2. Content encryption and storage in PostgreSQL
    3. Processing and chunking
    4. Embedding generation and ChromaDB storage
    5. Metadata management and search
    """
    
    def __init__(self):
        self.encryption = DocumentEncryption()
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the document service."""
        if self.is_initialized:
            return
            
        logger.info("Document service initialized")
        self.is_initialized = True
    
    async def upload_document(
        self,
        knowledge_base_id: str,
        file_content: bytes,
        filename: str,
        content_type: str,
        uploaded_by: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_public: bool = False
    ) -> DocumentUploadResponse:
        """
        Upload and store a document.
        
        Steps:
        1. Validate and hash content
        2. Encrypt content for PostgreSQL storage
        3. Create document record
        4. Queue for processing
        """
        try:
            async with get_database_session() as session:
                # Generate unique document ID
                document_id = uuid.uuid4()
                
                # Calculate content hash for deduplication
                content_hash = hashlib.sha256(file_content).hexdigest()
                
                # Check for duplicate content in this knowledge base
                existing_doc = await session.execute(
                    select(DocumentDB).where(
                        and_(
                            DocumentDB.knowledge_base_id == knowledge_base_id,
                            DocumentDB.content_hash == content_hash
                        )
                    )
                )
                if existing_doc.scalar_one_or_none():
                    raise ValueError(f"Document with identical content already exists in knowledge base")
                
                # Encrypt content
                encrypted_content = self.encryption.encrypt_content(file_content)
                
                # Detect document type
                document_type = self._detect_document_type(content_type)
                
                # Create document record
                document = DocumentDB(
                    id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    title=title or Path(filename).stem,
                    filename=self._sanitize_filename(filename),
                    original_filename=filename,
                    content_type=content_type,
                    file_size=len(file_content),
                    content_hash=content_hash,
                    encrypted_content=encrypted_content,
                    status="pending",
                    document_type=document_type,
                    doc_metadata=metadata or {},
                    is_public=is_public,
                    uploaded_by=uploaded_by
                )
                
                session.add(document)
                await session.commit()
                
                logger.info(
                    "Document uploaded successfully",
                    document_id=str(document_id),
                    knowledge_base_id=knowledge_base_id,
                    filename=filename,
                    size=len(file_content)
                )
                
                # Process document using revolutionary ingestion engine
                await self._process_document_with_revolutionary_engine(
                    document_id=str(document_id),
                    knowledge_base_id=knowledge_base_id,
                    file_content=file_content,
                    filename=filename,
                    content_type=content_type,
                    metadata=metadata or {}
                )

                return DocumentUploadResponse(
                    success=True,
                    document_id=str(document_id),
                    knowledge_base_id=knowledge_base_id,
                    title=document.title,
                    filename=document.filename,
                    status="processed",
                    message="Document uploaded and processed successfully with revolutionary ingestion engine"
                )
                
        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            raise

    async def _process_document_with_revolutionary_engine(
        self,
        document_id: str,
        knowledge_base_id: str,
        file_content: bytes,
        filename: str,
        content_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Process document using the revolutionary ingestion engine."""
        try:
            logger.info(f"Processing document {document_id} with revolutionary ingestion engine")

            # Import revolutionary processor registry
            from app.rag.ingestion.processors import get_revolutionary_processor_registry

            # Get processor registry
            registry = await get_revolutionary_processor_registry()

            # Process document
            processing_result = await registry.process_document(
                content=file_content,
                filename=filename,
                mime_type=content_type,
                metadata=metadata
            )

            # Extract processed data
            extracted_text = processing_result['text']
            enhanced_metadata = processing_result['metadata']
            document_structure = processing_result['structure']
            detected_language = processing_result['language']
            confidence_score = processing_result['confidence']

            # Chunk the text for vector storage
            chunks = await self._chunk_text_intelligently(
                text=extracted_text,
                document_id=document_id,
                structure=document_structure
            )

            # Generate embeddings and store in vector database
            await self._store_chunks_in_vector_db(
                chunks=chunks,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                metadata=enhanced_metadata
            )

            # Update document status in database
            await self._update_document_status(
                document_id=document_id,
                status="processed",
                chunk_count=len(chunks),
                enhanced_metadata=enhanced_metadata,
                language=detected_language,
                confidence=confidence_score
            )

            logger.info(
                f"Document {document_id} processed successfully",
                chunks_created=len(chunks),
                confidence=confidence_score,
                language=detected_language
            )

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            # Update document status to failed
            await self._update_document_status(
                document_id=document_id,
                status="failed",
                error_message=str(e)
            )
            raise

    async def _chunk_text_intelligently(
        self,
        text: str,
        document_id: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk text intelligently based on document structure."""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            # Determine chunking strategy based on document type
            doc_type = structure.get('type', 'text')

            if doc_type == 'code':
                chunk_size = 800
                chunk_overlap = 100
                separators = ['\nclass ', '\ndef ', '\n\n', '\n', ' ', '']
            elif doc_type == 'markdown':
                chunk_size = 1200
                chunk_overlap = 150
                separators = ['\n## ', '\n### ', '\n\n', '\n', ' ', '']
            else:
                chunk_size = 1000
                chunk_overlap = 200
                separators = ['\n\n', '\n', '. ', ' ', '']

            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len
            )

            # Split text into chunks
            text_chunks = text_splitter.split_text(text)

            # Create chunk metadata
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'index': i,
                    'text': chunk_text,
                    'metadata': {
                        'document_id': document_id,
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'document_type': doc_type
                    }
                })

            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk text for document {document_id}: {e}")
            # Fallback: create single chunk
            return [{
                'index': 0,
                'text': text,
                'metadata': {
                    'document_id': document_id,
                    'chunk_index': 0,
                    'chunk_size': len(text),
                    'document_type': 'fallback'
                }
            }]

    async def _store_chunks_in_vector_db(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        knowledge_base_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Store document chunks in vector database."""
        try:
            # Get embedding service
            from app.services.embedding_service import get_embedding_service
            embedding_service = await get_embedding_service()

            # Get vector store for this knowledge base
            from app.rag.core.vector_store import get_vector_store
            vector_store = await get_vector_store(knowledge_base_id)

            # Process each chunk
            for chunk in chunks:
                # Generate embedding
                embedding = await embedding_service.embed_text(chunk['text'])

                # Store in vector database
                vector_id = await vector_store.add_document(
                    content=chunk['text'],
                    embedding=embedding,
                    metadata={
                        **chunk['metadata'],
                        **metadata,
                        'knowledge_base_id': knowledge_base_id
                    }
                )

                # Store chunk in PostgreSQL
                await self._store_chunk_in_postgres(
                    document_id=document_id,
                    chunk_index=chunk['index'],
                    content=chunk['text'],
                    vector_id=vector_id,
                    metadata=chunk['metadata']
                )

            logger.info(f"Stored {len(chunks)} chunks for document {document_id}")

        except Exception as e:
            logger.error(f"Failed to store chunks for document {document_id}: {e}")
            raise

    async def _store_chunk_in_postgres(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        vector_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Store document chunk in PostgreSQL."""
        try:
            async with get_database_session() as session:
                chunk = DocumentChunkDB(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=content,
                    vector_id=vector_id,
                    chunk_metadata=metadata
                )

                session.add(chunk)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to store chunk {chunk_index} for document {document_id}: {e}")
            raise

    async def _update_document_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        enhanced_metadata: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        confidence: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Update document processing status."""
        try:
            async with get_database_session() as session:
                result = await session.execute(
                    select(DocumentDB).where(DocumentDB.id == document_id)
                )
                document = result.scalar_one_or_none()

                if document:
                    document.status = status
                    document.processed_at = datetime.utcnow()

                    if chunk_count is not None:
                        document.chunk_count = chunk_count

                    if enhanced_metadata:
                        document.doc_metadata = {
                            **(document.doc_metadata or {}),
                            **enhanced_metadata
                        }

                    if language:
                        document.language = language

                    if confidence is not None:
                        document.processing_confidence = confidence

                    if error_message:
                        document.error_message = error_message

                    await session.commit()

        except Exception as e:
            logger.error(f"Failed to update document status for {document_id}: {e}")
            # Don't raise here to avoid cascading failures
    
    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID."""
        try:
            async with get_database_session() as session:
                result = await session.execute(
                    select(DocumentDB).where(DocumentDB.id == document_id)
                )
                document = result.scalar_one_or_none()
                
                if not document:
                    return None
                
                return DocumentMetadata.model_validate(document)
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise
    
    async def list_documents(
        self,
        knowledge_base_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentMetadata]:
        """List documents in a knowledge base."""
        try:
            async with get_database_session() as session:
                result = await session.execute(
                    select(DocumentDB)
                    .where(DocumentDB.knowledge_base_id == knowledge_base_id)
                    .order_by(DocumentDB.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                documents = result.scalars().all()

                return [DocumentMetadata.model_validate(doc) for doc in documents]

        except Exception as e:
            logger.warning(f"Failed to list documents for KB {knowledge_base_id}: {e}")
            # Return empty list if database tables don't exist yet
            return []
    
    async def get_document_chunks(
        self, 
        document_id: str
    ) -> List[DocumentChunkMetadata]:
        """Get all chunks for a document."""
        try:
            async with get_database_session() as session:
                result = await session.execute(
                    select(DocumentChunkDB)
                    .where(DocumentChunkDB.document_id == document_id)
                    .order_by(DocumentChunkDB.chunk_index)
                )
                chunks = result.scalars().all()
                
                return [DocumentChunkMetadata.model_validate(chunk) for chunk in chunks]
                
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            raise
    
    def _detect_document_type(self, content_type: str) -> str:
        """Detect document type from content type."""
        if content_type.startswith('text/'):
            return 'text'
        elif content_type.startswith('image/'):
            return 'image'
        elif content_type == 'application/pdf':
            return 'pdf'
        elif content_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return 'document'
        elif content_type.startswith('application/'):
            return 'application'
        else:
            return 'unknown'
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components and dangerous characters
        safe_filename = Path(filename).name
        # Replace spaces and special characters
        safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in '.-_')
        return safe_filename[:255]  # Limit length

    async def _process_document(self, document_id: str) -> None:
        """
        Process document: extract text, chunk, embed, and store in ChromaDB.

        This is the revolutionary ingestion pipeline that:
        1. Decrypts and extracts text from document
        2. Chunks the content intelligently
        3. Generates embeddings using global embedding model
        4. Stores chunks in ChromaDB with unique collection per KB
        5. Updates PostgreSQL with chunk metadata
        """
        try:
            async with get_database_session() as session:
                # Get document
                result = await session.execute(
                    select(DocumentDB).where(DocumentDB.id == document_id)
                )
                document = result.scalar_one_or_none()

                if not document:
                    logger.error(f"Document {document_id} not found for processing")
                    return

                # Update status to processing
                await session.execute(
                    update(DocumentDB)
                    .where(DocumentDB.id == document_id)
                    .values(status="processing")
                )
                await session.commit()

                try:
                    # Decrypt content
                    decrypted_content = self.encryption.decrypt_content(document.encrypted_content)

                    # Extract text content based on document type
                    text_content = await self._extract_text_content(
                        decrypted_content,
                        document.content_type,
                        document.filename
                    )

                    # Chunk the content
                    chunks = await self._chunk_content(text_content, document_id)

                    # Generate embeddings and store in ChromaDB
                    await self._embed_and_store_chunks(chunks, document.knowledge_base_id, session)

                    # Update document status
                    await session.execute(
                        update(DocumentDB)
                        .where(DocumentDB.id == document_id)
                        .values(
                            status="completed",
                            chunk_count=len(chunks),
                            processed_at=datetime.utcnow()
                        )
                    )
                    await session.commit()

                    logger.info(
                        "Document processed successfully",
                        document_id=document_id,
                        chunks=len(chunks)
                    )

                except Exception as e:
                    # Update status to failed
                    await session.execute(
                        update(DocumentDB)
                        .where(DocumentDB.id == document_id)
                        .values(
                            status="failed",
                            processing_error=str(e)
                        )
                    )
                    await session.commit()
                    raise

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")

    async def _extract_text_content(
        self,
        content: bytes,
        content_type: str,
        filename: str
    ) -> str:
        """Extract text content from various file types."""
        try:
            if content_type.startswith('text/'):
                return content.decode('utf-8')
            elif content_type == 'application/pdf':
                # Use PyPDF2 or similar for PDF extraction
                return await self._extract_pdf_text(content)
            elif content_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                # Use python-docx for Word documents
                return await self._extract_docx_text(content)
            else:
                # Try to decode as text
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    return f"Binary file: {filename} (content extraction not supported)"

        except Exception as e:
            logger.warning(f"Failed to extract text from {filename}: {e}")
            return f"Failed to extract text from {filename}"

    async def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content."""
        # Placeholder - implement with PyPDF2 or similar
        return "PDF text extraction not implemented yet"

    async def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content."""
        # Placeholder - implement with python-docx
        return "DOCX text extraction not implemented yet"

    async def _chunk_content(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Chunk text content intelligently."""
        # Simple chunking for now - can be enhanced with semantic chunking
        chunk_size = 1000
        chunk_overlap = 200

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > start + chunk_size // 2:
                    end = start + break_point + 1
                    chunk_text = text[start:end]

            chunks.append({
                'chunk_index': chunk_index,
                'text': chunk_text.strip(),
                'start_char': start,
                'end_char': end,
                'document_id': document_id
            })

            start = end - chunk_overlap
            chunk_index += 1

        return chunks

    async def _embed_and_store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        knowledge_base_id: str,
        session: AsyncSession
    ) -> None:
        """Generate embeddings and store chunks in ChromaDB and PostgreSQL."""
        try:
            # Get global embedding manager
            from app.rag.core.global_embedding_manager import get_global_embedding_manager
            embedding_manager = await get_global_embedding_manager()

            # Get knowledge base service for ChromaDB access
            from app.services.knowledge_base_service import get_knowledge_base_service
            kb_service = await get_knowledge_base_service()

            # Get the knowledge base
            if knowledge_base_id not in kb_service.knowledge_bases:
                raise ValueError(f"Knowledge base {knowledge_base_id} not found")

            knowledge_base = kb_service.knowledge_bases[knowledge_base_id]

            # Process chunks in batches
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Extract texts for embedding
                texts = [chunk['text'] for chunk in batch]

                # Generate embeddings
                embedding_result = await embedding_manager.generate_embeddings(texts)

                # Store each chunk
                for j, chunk in enumerate(batch):
                    chunk_id = str(uuid.uuid4())
                    chromadb_id = f"chunk_{chunk_id}"

                    # Create document chunk for ChromaDB
                    doc_chunk = DocumentChunk(
                        id=chromadb_id,
                        content=chunk['text'],
                        document_id=chunk['document_id'],
                        chunk_index=chunk['chunk_index'],
                        embedding=embedding_result.embeddings[j],
                        metadata={
                            'document_id': chunk['document_id'],
                            'chunk_index': chunk['chunk_index'],
                            'start_char': chunk['start_char'],
                            'end_char': chunk['end_char'],
                            'knowledge_base_id': knowledge_base_id
                        }
                    )

                    # Store in ChromaDB
                    await knowledge_base.vector_store.add_documents([doc_chunk], knowledge_base_id)

                    # Store metadata in PostgreSQL
                    chunk_db = DocumentChunkDB(
                        id=chunk_id,
                        document_id=chunk['document_id'],
                        chunk_index=chunk['chunk_index'],
                        chunk_text=chunk['text'],
                        start_char=chunk['start_char'],
                        end_char=chunk['end_char'],
                        chromadb_id=chromadb_id,
                        collection_name=knowledge_base_id,
                        embedding_model=embedding_result.model_name,
                        embedding_dimensions=len(embedding_result.embeddings[j]),
                        chunk_metadata={
                            'knowledge_base_id': knowledge_base_id,
                            'processing_timestamp': datetime.utcnow().isoformat()
                        }
                    )

                    session.add(chunk_db)

                # Commit batch
                await session.commit()

                logger.info(f"Processed batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1)//batch_size}")

        except Exception as e:
            logger.error(f"Failed to embed and store chunks: {e}")
            raise

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks."""
        try:
            async with get_database_session() as session:
                # Get document
                result = await session.execute(
                    select(DocumentDB).where(DocumentDB.id == document_id)
                )
                document = result.scalar_one_or_none()

                if not document:
                    return False

                # Get all chunks
                chunks_result = await session.execute(
                    select(DocumentChunkDB).where(DocumentChunkDB.document_id == document_id)
                )
                chunks = chunks_result.scalars().all()

                # Delete from ChromaDB
                if chunks:
                    from app.services.knowledge_base_service import get_knowledge_base_service
                    kb_service = await get_knowledge_base_service()

                    if document.knowledge_base_id in kb_service.knowledge_bases:
                        knowledge_base = kb_service.knowledge_bases[document.knowledge_base_id]
                        chunk_ids = [chunk.chromadb_id for chunk in chunks]
                        await knowledge_base.vector_store.delete_documents(chunk_ids, document.knowledge_base_id)

                # Delete from PostgreSQL (cascades to chunks)
                await session.execute(
                    delete(DocumentDB).where(DocumentDB.id == document_id)
                )
                await session.commit()

                logger.info(f"Document {document_id} deleted successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise


# Global instance
_document_service: Optional[DocumentService] = None


async def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    global _document_service

    if _document_service is None:
        _document_service = DocumentService()
        await _document_service.initialize()

    return _document_service
