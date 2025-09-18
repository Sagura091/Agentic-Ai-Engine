"""
Revolutionary Document Ingestion Engine for Agentic AI System.

This module provides a comprehensive document ingestion pipeline that:
- Processes multiple document formats (PDF, DOCX, TXT, MD, HTML, etc.)
- Performs intelligent chunking with semantic boundaries
- Integrates with global embedding models
- Stores vectors in ChromaDB with metadata
- Preserves original files in PostgreSQL
- Provides document viewing capabilities for LLM perspective
"""

import asyncio
import hashlib
import io
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import aiofiles
import structlog
from fastapi import UploadFile
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    JSONLoader
)

from app.config.settings import get_settings
from app.models.document import DocumentDB, DocumentChunkDB, DocumentMetadata, DocumentChunkMetadata
from app.models.database.base import get_database_session
from app.rag.core.vector_store import get_vector_store
from app.services.embedding_service import get_embedding_service

logger = structlog.get_logger(__name__)


class RevolutionaryIngestionEngine:
    """
    Revolutionary document ingestion engine with advanced processing capabilities.
    
    Features:
    - Multi-format document support
    - Intelligent semantic chunking
    - Global embedding integration
    - Vector storage with metadata
    - Original file preservation
    - LLM perspective viewing
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.supported_formats = {
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'text/plain': self._process_text,
            'text/markdown': self._process_markdown,
            'text/html': self._process_html,
            'text/csv': self._process_csv,
            'application/json': self._process_json,
            'text/x-python': self._process_python,
            'application/x-python-code': self._process_python,
        }
        
        # Chunking configurations for different content types
        self.chunking_configs = {
            'text': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'separators': ['\n\n', '\n', '. ', ' ', '']
            },
            'code': {
                'chunk_size': 800,
                'chunk_overlap': 100,
                'separators': ['\nclass ', '\ndef ', '\n\n', '\n', ' ', '']
            },
            'markdown': {
                'chunk_size': 1200,
                'chunk_overlap': 150,
                'headers_to_split_on': [('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')]
            },
            'html': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'headers_to_split_on': [('h1', 'Header 1'), ('h2', 'Header 2'), ('h3', 'Header 3')]
            }
        }
    
    async def ingest_document(
        self,
        file: UploadFile,
        knowledge_base_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "default_user"
    ) -> DocumentMetadata:
        """
        Ingest a document through the revolutionary pipeline.
        
        Args:
            file: Uploaded file object
            knowledge_base_id: Target knowledge base ID
            title: Optional custom title
            metadata: Additional metadata
            user_id: User uploading the document
            
        Returns:
            DocumentMetadata: Processed document metadata
        """
        try:
            logger.info(
                "Starting revolutionary document ingestion",
                filename=file.filename,
                content_type=file.content_type,
                knowledge_base_id=knowledge_base_id
            )
            
            # Step 1: Read and validate file
            file_content = await file.read()
            file_size = len(file_content)
            content_hash = hashlib.sha256(file_content).hexdigest()
            
            # Step 2: Detect content type
            content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
            
            # Step 3: Generate document ID and metadata
            document_id = str(uuid.uuid4())
            document_title = title or file.filename or f"Document_{document_id[:8]}"
            
            # Step 4: Process document content
            processed_content, extracted_metadata = await self._process_document_content(
                file_content, content_type, file.filename
            )
            
            # Step 5: Perform intelligent chunking
            chunks = await self._chunk_document(processed_content, content_type, document_title)
            
            # Step 6: Generate embeddings
            embedding_service = await get_embedding_service()
            chunk_embeddings = []
            
            for i, chunk in enumerate(chunks):
                embedding = await embedding_service.embed_text(chunk.page_content)
                chunk_embeddings.append({
                    'chunk_index': i,
                    'content': chunk.page_content,
                    'embedding': embedding,
                    'metadata': chunk.metadata
                })
            
            # Step 7: Store in vector database
            vector_store = await get_vector_store(knowledge_base_id)
            vector_ids = []
            
            for chunk_data in chunk_embeddings:
                vector_id = await vector_store.add_document(
                    content=chunk_data['content'],
                    embedding=chunk_data['embedding'],
                    metadata={
                        **chunk_data['metadata'],
                        'document_id': document_id,
                        'chunk_index': chunk_data['chunk_index'],
                        'knowledge_base_id': knowledge_base_id
                    }
                )
                vector_ids.append(vector_id)
            
            # Step 8: Store document and chunks in PostgreSQL
            document_metadata = await self._store_document_in_database(
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                title=document_title,
                filename=file.filename,
                content_type=content_type,
                file_size=file_size,
                content_hash=content_hash,
                encrypted_content=file_content,  # In production, encrypt this
                chunks=chunk_embeddings,
                vector_ids=vector_ids,
                metadata={**(metadata or {}), **extracted_metadata},
                user_id=user_id
            )
            
            logger.info(
                "Revolutionary document ingestion completed",
                document_id=document_id,
                chunks_created=len(chunks),
                total_size=file_size
            )
            
            return document_metadata
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file.filename}: {e}")
            raise
    
    async def _process_document_content(
        self, 
        content: bytes, 
        content_type: str, 
        filename: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Process document content based on type."""
        if content_type in self.supported_formats:
            return await self.supported_formats[content_type](content, filename)
        else:
            # Fallback to text processing
            try:
                text_content = content.decode('utf-8')
                return text_content, {'processing_method': 'text_fallback'}
            except UnicodeDecodeError:
                return content.decode('utf-8', errors='ignore'), {
                    'processing_method': 'text_fallback_with_errors',
                    'warning': 'Some characters may have been lost during decoding'
                }
    
    async def _process_pdf(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process PDF documents."""
        # Save temporarily for PyPDFLoader
        temp_path = f"/tmp/{uuid.uuid4()}_{filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            text_content = '\n\n'.join([doc.page_content for doc in documents])
            
            metadata = {
                'processing_method': 'pypdf',
                'page_count': len(documents),
                'total_characters': len(text_content)
            }
            
            return text_content, metadata
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
    
    async def _process_docx(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process DOCX documents."""
        temp_path = f"/tmp/{uuid.uuid4()}_{filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        try:
            loader = Docx2txtLoader(temp_path)
            documents = loader.load()
            text_content = '\n\n'.join([doc.page_content for doc in documents])
            
            metadata = {
                'processing_method': 'docx2txt',
                'total_characters': len(text_content)
            }
            
            return text_content, metadata
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    async def _process_text(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process plain text documents."""
        text_content = content.decode('utf-8')
        metadata = {
            'processing_method': 'text',
            'total_characters': len(text_content),
            'line_count': text_content.count('\n') + 1
        }
        return text_content, metadata
    
    async def _process_markdown(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process Markdown documents."""
        text_content = content.decode('utf-8')
        metadata = {
            'processing_method': 'markdown',
            'total_characters': len(text_content),
            'header_count': text_content.count('#')
        }
        return text_content, metadata
    
    async def _process_html(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process HTML documents."""
        temp_path = f"/tmp/{uuid.uuid4()}_{filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        try:
            loader = UnstructuredHTMLLoader(temp_path)
            documents = loader.load()
            text_content = '\n\n'.join([doc.page_content for doc in documents])
            
            metadata = {
                'processing_method': 'unstructured_html',
                'total_characters': len(text_content)
            }
            
            return text_content, metadata
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    async def _process_csv(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process CSV documents."""
        temp_path = f"/tmp/{uuid.uuid4()}_{filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        try:
            loader = CSVLoader(temp_path)
            documents = loader.load()
            text_content = '\n\n'.join([doc.page_content for doc in documents])
            
            metadata = {
                'processing_method': 'csv',
                'total_characters': len(text_content),
                'row_count': len(documents)
            }
            
            return text_content, metadata
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    async def _process_json(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process JSON documents."""
        temp_path = f"/tmp/{uuid.uuid4()}_{filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)
        
        try:
            loader = JSONLoader(temp_path, jq_schema='.', text_content=False)
            documents = loader.load()
            text_content = '\n\n'.join([doc.page_content for doc in documents])
            
            metadata = {
                'processing_method': 'json',
                'total_characters': len(text_content)
            }
            
            return text_content, metadata
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    async def _process_python(self, content: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Process Python code documents."""
        text_content = content.decode('utf-8')
        metadata = {
            'processing_method': 'python_code',
            'total_characters': len(text_content),
            'line_count': text_content.count('\n') + 1,
            'function_count': text_content.count('def '),
            'class_count': text_content.count('class ')
        }
        return text_content, metadata


# Global instance
_ingestion_engine = None

async def get_ingestion_engine() -> RevolutionaryIngestionEngine:
    """Get the global ingestion engine instance."""
    global _ingestion_engine
    if _ingestion_engine is None:
        _ingestion_engine = RevolutionaryIngestionEngine()
    return _ingestion_engine
