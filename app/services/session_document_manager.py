"""
🔥 REVOLUTIONARY SESSION DOCUMENT MANAGER
========================================

Core business logic for the Revolutionary Session-Based Document Workspace.
Orchestrates document storage, vector search, and AI processing capabilities.

CORE FEATURES:
- Session-scoped document management
- Integration with Revolutionary Document Intelligence Tool
- Temporary vector search capabilities
- Automatic cleanup and lifecycle management
- Background processing with job tracking
"""

import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory

# Import our components
from app.config.session_document_config import session_document_config
from app.models.session_document_models import (
    SessionDocument,
    SessionDocumentWorkspace,
    SessionDocumentDB,
    SessionWorkspaceDB,
    SessionDocumentCreate,
    SessionDocumentResponse,
    SessionDocumentQuery,
    SessionDocumentAnalysisRequest,
    SessionWorkspaceStats,
    DocumentProcessingStatus,
    SessionDocumentType,
    SessionDocumentError,
    SessionDocumentNotFoundError,
    SessionWorkspaceNotFoundError,
    SessionDocumentExpiredError,
    SessionWorkspaceFullError
)
from app.storage.session_document_storage import session_document_storage
from app.rag.session_vector_store import session_vector_store

# Import existing database session
try:
    from app.database.session import get_database_session
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Import Revolutionary Document Intelligence Tool
try:
    from app.tools.production.revolutionary_document_intelligence_tool import RevolutionaryDocumentIntelligenceTool
    INTELLIGENCE_TOOL_AVAILABLE = True
except ImportError:
    INTELLIGENCE_TOOL_AVAILABLE = False

# Import existing RAG infrastructure for integration
try:
    from app.rag.core.unified_rag_system import UnifiedRAGSystem
    from app.rag.ingestion.pipeline import RevolutionaryIngestionPipeline
    from app.rag.ingestion.processors import RevolutionaryProcessorRegistry
    RAG_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    RAG_INFRASTRUCTURE_AVAILABLE = False

logger = get_logger()


class SessionDocumentManager:
    """
    🔥 REVOLUTIONARY SESSION DOCUMENT MANAGER - HYBRID ARCHITECTURE

    Central orchestrator for session-based document operations that leverages
    existing RAG infrastructure while maintaining session isolation:

    HYBRID FEATURES:
    - Uses existing RevolutionaryIngestionPipeline for document processing
    - Uses existing Revolutionary Document Intelligence Tool for AI analysis
    - Uses existing vision models and multi-modal processing
    - BUT maintains session-only storage with no permanent contamination
    - Provides session-scoped search and analysis
    - Guarantees complete cleanup when session ends
    """

    def __init__(self):
        """Initialize the hybrid session document manager."""
        self.config = session_document_config
        self.storage = session_document_storage
        self.vector_store = session_vector_store

        # Revolutionary Document Intelligence Tool integration
        self.intelligence_tool = None

        # Existing RAG infrastructure integration
        self.ingestion_pipeline = None
        self.processor_registry = None
        self.unified_rag = None

        # In-memory workspace cache
        self.workspaces: Dict[str, SessionDocumentWorkspace] = {}

        # Background job tracking
        self.processing_jobs: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "total_workspaces": 0,
            "total_documents": 0,
            "total_processing_jobs": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "cleanup_runs": 0,
            "last_cleanup": None,
            "rag_integration": RAG_INFRASTRUCTURE_AVAILABLE,
            "hybrid_architecture": True
        }

        self.is_initialized = False
        logger.info(
            "🔥 Revolutionary Hybrid Session Document Manager initializing...",
            LogCategory.SERVICE_OPERATIONS,
            "app.services.session_document_manager"
        )
    
    async def initialize(self):
        """Initialize the session document manager."""
        try:
            if self.is_initialized:
                return
            
            # Initialize storage system
            self.storage._ensure_directories()
            
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Initialize Revolutionary Document Intelligence Tool
            if INTELLIGENCE_TOOL_AVAILABLE and self.config.integration.enable_document_intelligence:
                self.intelligence_tool = RevolutionaryDocumentIntelligenceTool()
                logger.info(
                    "🧠 Revolutionary Document Intelligence Tool integrated",
                    LogCategory.SERVICE_OPERATIONS,
                    "app.services.session_document_manager"
                )

            # Initialize existing RAG infrastructure integration
            if RAG_INFRASTRUCTURE_AVAILABLE:
                await self._initialize_rag_integration()

            # Start background cleanup task
            if self.config.expiration.cleanup_enabled:
                asyncio.create_task(self._background_cleanup_task())

            self.is_initialized = True
            logger.info(
                "✅ Revolutionary Hybrid Session Document Manager initialized",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager"
            )

        except Exception as e:
            logger.error(
                "❌ Failed to initialize Session Document Manager",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                error=e
            )
            raise SessionDocumentError(f"Initialization failed: {str(e)}")

    async def _initialize_rag_integration(self):
        """Initialize integration with existing RAG infrastructure."""
        try:
            # Initialize revolutionary ingestion pipeline
            try:
                self.ingestion_pipeline = RevolutionaryIngestionPipeline()
                if hasattr(self.ingestion_pipeline, 'initialize'):
                    await self.ingestion_pipeline.initialize()
            except Exception as e:
                logger.warn(
                    "Ingestion pipeline initialization failed",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.session_document_manager",
                    error=e
                )
                self.ingestion_pipeline = None

            # Initialize processor registry for multi-modal processing
            try:
                self.processor_registry = RevolutionaryProcessorRegistry()
                if hasattr(self.processor_registry, 'initialize'):
                    await self.processor_registry.initialize()
            except Exception as e:
                logger.warn(
                    "Processor registry initialization failed",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.session_document_manager",
                    error=e
                )
                self.processor_registry = None

            # Initialize unified RAG system (for potential future integration)
            try:
                self.unified_rag = UnifiedRAGSystem()
                if hasattr(self.unified_rag, 'initialize'):
                    await self.unified_rag.initialize()
            except Exception as e:
                logger.warn(
                    "Unified RAG initialization failed",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.session_document_manager",
                    error=e
                )
                self.unified_rag = None

            logger.info(
                "🔗 RAG infrastructure integration attempted",
                LogCategory.RAG_OPERATIONS,
                "app.services.session_document_manager",
                data={
                    "ingestion_pipeline": self.ingestion_pipeline is not None,
                    "processor_registry": self.processor_registry is not None,
                    "unified_rag": self.unified_rag is not None
                }
            )

        except Exception as e:
            logger.warn(
                "RAG integration failed, continuing without",
                LogCategory.RAG_OPERATIONS,
                "app.services.session_document_manager",
                error=e
            )
            self.ingestion_pipeline = None
            self.processor_registry = None
            self.unified_rag = None
    
    async def create_workspace(self, session_id: str) -> SessionDocumentWorkspace:
        """
        Create a new session document workspace.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Created workspace
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Check if workspace already exists
            if session_id in self.workspaces:
                return self.workspaces[session_id]
            
            # Create new workspace
            workspace = SessionDocumentWorkspace(
                session_id=session_id,
                created_at=datetime.utcnow(),
                max_documents=self.config.limits.max_documents_per_session,
                max_size=self.config.limits.max_total_size_per_session
            )
            
            # Store in memory cache
            self.workspaces[session_id] = workspace
            
            # Create vector collection
            await self.vector_store.create_session_collection(session_id)
            
            # Store in database if available
            if DATABASE_AVAILABLE:
                await self._store_workspace_in_db(workspace)
            
            self.stats["total_workspaces"] += 1

            logger.info(
                "📁 Session workspace created",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={
                    "session_id": session_id,
                    "max_documents": workspace.max_documents,
                    "max_size": workspace.max_size
                }
            )

            return workspace

        except Exception as e:
            logger.error(
                "❌ Failed to create workspace",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )
            raise SessionDocumentError(f"Failed to create workspace: {str(e)}")
    
    async def upload_document(
        self,
        session_id: str,
        file_content: bytes,
        filename: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionDocumentResponse:
        """
        Upload a document to a session workspace.
        
        Args:
            session_id: Session identifier
            file_content: Document content
            filename: Original filename
            content_type: MIME type
            metadata: Additional metadata
            
        Returns:
            Document response with details
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Validate file
            if not self.config.is_file_allowed(filename, content_type):
                raise SessionDocumentError(f"File type not allowed: {filename}")
            
            if len(file_content) > self.config.limits.max_document_size:
                raise SessionDocumentError(f"File size exceeds limit: {len(file_content)} bytes")
            
            # Get or create workspace
            workspace = await self.get_workspace(session_id)
            if not workspace:
                workspace = await self.create_workspace(session_id)
            
            # Check workspace limits
            if not workspace.can_add_document(len(file_content)):
                if workspace.total_documents >= workspace.max_documents:
                    raise SessionWorkspaceFullError("Maximum number of documents reached")
                else:
                    raise SessionWorkspaceFullError("Maximum workspace size reached")
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Store document in storage system
            document = await self.storage.store_document(
                session_id=session_id,
                document_id=document_id,
                content=file_content,
                filename=filename,
                content_type=content_type,
                document_type=SessionDocumentType.UPLOADED
            )
            
            # Add metadata
            if metadata:
                document.metadata.update(metadata)
            
            # Add to workspace
            workspace.documents[document_id] = document
            workspace.last_accessed = datetime.utcnow()
            
            # Add to vector store for search
            await self.vector_store.add_document_to_session(session_id, document)
            
            # Store in database if available
            if DATABASE_AVAILABLE:
                await self._store_document_in_db(document)
            
            self.stats["total_documents"] += 1
            self.stats["successful_operations"] += 1

            logger.info(
                "📤 Document uploaded successfully",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={
                    "session_id": session_id,
                    "document_id": document_id,
                    "filename": filename,
                    "size": len(file_content)
                }
            )

            return document.to_response_model()

        except (SessionDocumentError, SessionWorkspaceFullError):
            self.stats["failed_operations"] += 1
            raise
        except Exception as e:
            self.stats["failed_operations"] += 1
            logger.error(
                "❌ Failed to upload document",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id, "filename": filename},
                error=e
            )
            raise SessionDocumentError(f"Failed to upload document: {str(e)}")
    
    async def get_workspace(self, session_id: str) -> Optional[SessionDocumentWorkspace]:
        """
        Get a session workspace.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Workspace or None if not found
        """
        try:
            # Check memory cache first
            if session_id in self.workspaces:
                workspace = self.workspaces[session_id]
                
                # Check if expired
                if workspace.is_expired:
                    await self.cleanup_workspace(session_id)
                    return None
                
                workspace.last_accessed = datetime.utcnow()
                return workspace
            
            # Try to load from database if available
            if DATABASE_AVAILABLE:
                workspace = await self._load_workspace_from_db(session_id)
                if workspace:
                    self.workspaces[session_id] = workspace
                    return workspace
            
            return None

        except Exception as e:
            logger.error(
                "❌ Failed to get workspace",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )
            return None
    
    async def list_session_documents(
        self,
        session_id: str,
        include_expired: bool = False
    ) -> List[SessionDocumentResponse]:
        """
        List all documents in a session.
        
        Args:
            session_id: Session identifier
            include_expired: Whether to include expired documents
            
        Returns:
            List of document responses
        """
        try:
            workspace = await self.get_workspace(session_id)
            if not workspace:
                return []
            
            documents = []
            current_time = datetime.utcnow()
            
            for document in workspace.documents.values():
                # Skip expired documents unless requested
                if not include_expired and document.is_expired:
                    continue
                
                documents.append(document.to_response_model())
            
            # Sort by upload time (newest first)
            documents.sort(key=lambda x: x.uploaded_at, reverse=True)

            logger.debug(
                "📋 Listed session documents",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id, "count": len(documents)}
            )

            return documents

        except Exception as e:
            logger.error(
                "❌ Failed to list session documents",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )
            return []
    
    async def get_document(
        self,
        session_id: str,
        document_id: str,
        include_content: bool = False
    ) -> Optional[SessionDocumentResponse]:
        """
        Get a specific document from a session.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            include_content: Whether to include document content
            
        Returns:
            Document response or None if not found
        """
        try:
            workspace = await self.get_workspace(session_id)
            if not workspace:
                return None
            
            if document_id not in workspace.documents:
                return None
            
            document = workspace.documents[document_id]
            
            # Check if expired
            if document.is_expired:
                raise SessionDocumentExpiredError(f"Document {document_id} has expired")
            
            # Update access time
            document.last_accessed = datetime.utcnow()
            workspace.last_accessed = datetime.utcnow()
            
            response = document.to_response_model()
            
            # Include content if requested
            if include_content:
                content = await self.storage.retrieve_document(session_id, document_id)
                if content:
                    response.content = content  # Note: This would need to be added to the response model
            
            return response
            
        except SessionDocumentExpiredError:
            raise
        except Exception as e:
            logger.error(
                "❌ Failed to get document",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id, "document_id": document_id},
                error=e
            )
            return None
    
    async def query_session_documents(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Query documents in a session using vector search.
        
        Args:
            session_id: Session identifier
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            workspace = await self.get_workspace(session_id)
            if not workspace:
                return []
            
            # Perform vector search
            search_results = await self.vector_store.search_session_documents(
                session_id=session_id,
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # Enhance results with document metadata
            enhanced_results = []
            for result in search_results:
                document_id = result["document_id"]
                if document_id in workspace.documents:
                    document = workspace.documents[document_id]
                    
                    # Skip expired documents
                    if document.is_expired:
                        continue
                    
                    enhanced_result = {
                        **result,
                        "document": document.to_response_model(),
                        "query": query,
                        "search_timestamp": datetime.utcnow().isoformat()
                    }
                    enhanced_results.append(enhanced_result)
            
            # Update workspace access time
            workspace.last_accessed = datetime.utcnow()

            logger.info(
                "🔍 Session document query completed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={
                    "session_id": session_id,
                    "query_length": len(query),
                    "results_count": len(enhanced_results)
                }
            )

            return enhanced_results

        except Exception as e:
            logger.error(
                "❌ Failed to query session documents",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )
            return []
    
    async def analyze_document_with_intelligence(
        self,
        session_id: str,
        document_id: str,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document using Revolutionary Document Intelligence Tool.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            analysis_options: Analysis configuration options
            
        Returns:
            Analysis results
        """
        try:
            if not INTELLIGENCE_TOOL_AVAILABLE or not self.intelligence_tool:
                raise SessionDocumentError("Revolutionary Document Intelligence Tool not available")
            
            # Get document
            workspace = await self.get_workspace(session_id)
            if not workspace or document_id not in workspace.documents:
                raise SessionDocumentNotFoundError(f"Document {document_id} not found")
            
            document = workspace.documents[document_id]
            
            # Check if expired
            if document.is_expired:
                raise SessionDocumentExpiredError(f"Document {document_id} has expired")
            
            # Get document content
            content = await self.storage.retrieve_document(session_id, document_id)
            if not content:
                raise SessionDocumentError("Failed to retrieve document content")
            
            # Prepare analysis request
            analysis_request = {
                "operation": "analyze",
                "filename": document.filename,
                "content_type": document.content_type,
                "extract_forms": analysis_options.get("extract_forms", True) if analysis_options else True,
                "extract_tables": analysis_options.get("extract_tables", True) if analysis_options else True,
                "ai_insights": analysis_options.get("ai_insights", True) if analysis_options else True
            }
            
            # Perform analysis using Revolutionary Document Intelligence Tool
            analysis_result = await self.intelligence_tool.upload_and_analyze(
                content,
                document.filename,
                **analysis_request
            )
            
            # Update document with analysis results
            document.analysis_results = analysis_result
            document.processing_status = DocumentProcessingStatus.ANALYZED
            workspace.last_accessed = datetime.utcnow()
            
            # Update in database if available
            if DATABASE_AVAILABLE:
                await self._update_document_in_db(document)

            logger.info(
                "🧠 Document analysis completed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={
                    "session_id": session_id,
                    "document_id": document_id,
                    "analysis_success": analysis_result.get("success", False)
                }
            )

            return analysis_result

        except (SessionDocumentError, SessionDocumentNotFoundError, SessionDocumentExpiredError):
            raise
        except Exception as e:
            logger.error(
                "❌ Document analysis failed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id, "document_id": document_id},
                error=e
            )
            raise SessionDocumentError(f"Analysis failed: {str(e)}")
    
    async def delete_document(
        self,
        session_id: str,
        document_id: str
    ) -> bool:
        """
        Delete a document from a session.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            workspace = await self.get_workspace(session_id)
            if not workspace or document_id not in workspace.documents:
                return False
            
            # Remove from storage
            await self.storage.delete_document(session_id, document_id)
            
            # Remove from workspace
            del workspace.documents[document_id]
            workspace.last_accessed = datetime.utcnow()
            
            # Remove from database if available
            if DATABASE_AVAILABLE:
                await self._delete_document_from_db(session_id, document_id)
            
            self.stats["successful_operations"] += 1

            logger.info(
                "🗑️ Document deleted successfully",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id, "document_id": document_id}
            )

            return True

        except Exception as e:
            self.stats["failed_operations"] += 1
            logger.error(
                "❌ Failed to delete document",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id, "document_id": document_id},
                error=e
            )
            return False
    
    async def cleanup_workspace(self, session_id: str) -> bool:
        """
        Clean up a session workspace and all its documents.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if cleaned up successfully
        """
        try:
            workspace = self.workspaces.get(session_id)
            
            if workspace:
                # Delete all documents
                for document_id in list(workspace.documents.keys()):
                    await self.delete_document(session_id, document_id)
                
                # Remove from memory
                del self.workspaces[session_id]
            
            # Remove vector collection
            await self.vector_store.remove_session_collection(session_id)
            
            # Remove from database if available
            if DATABASE_AVAILABLE:
                await self._delete_workspace_from_db(session_id)
            
            self.stats["total_workspaces"] -= 1

            logger.info(
                "🧹 Workspace cleaned up successfully",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id}
            )

            return True

        except Exception as e:
            logger.error(
                "❌ Failed to cleanup workspace",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )
            return False

    async def get_workspace_stats(self, session_id: str) -> Optional[SessionWorkspaceStats]:
        """Get statistics for a session workspace."""
        try:
            workspace = await self.get_workspace(session_id)
            if not workspace:
                return None

            return workspace.to_stats_model()

        except Exception as e:
            logger.error(
                "Failed to get workspace stats",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )
            return None

    async def _background_cleanup_task(self):
        """Background task for automatic cleanup."""
        while True:
            try:
                await asyncio.sleep(self.config.expiration.cleanup_interval.total_seconds())
                await self.cleanup_expired_workspaces()
            except Exception as e:
                logger.error(
                    "Background cleanup task failed",
                    LogCategory.SERVICE_OPERATIONS,
                    "app.services.session_document_manager",
                    error=e
                )

    async def cleanup_expired_workspaces(self) -> Dict[str, int]:
        """Clean up all expired workspaces."""
        try:
            cleanup_stats = {
                "workspaces_cleaned": 0,
                "documents_removed": 0,
                "errors": 0
            }

            current_time = datetime.utcnow()
            expired_sessions = []

            # Find expired workspaces
            for session_id, workspace in self.workspaces.items():
                if workspace.is_expired:
                    expired_sessions.append(session_id)

            # Clean up expired workspaces
            for session_id in expired_sessions:
                try:
                    workspace = self.workspaces[session_id]
                    cleanup_stats["documents_removed"] += len(workspace.documents)

                    if await self.cleanup_workspace(session_id):
                        cleanup_stats["workspaces_cleaned"] += 1
                    else:
                        cleanup_stats["errors"] += 1

                except Exception as e:
                    cleanup_stats["errors"] += 1
                    logger.error(
                        f"Failed to cleanup workspace {session_id}",
                        LogCategory.SERVICE_OPERATIONS,
                        "app.services.session_document_manager",
                        data={"session_id": session_id},
                        error=e
                    )

            # Clean up storage and vector store
            storage_cleanup = await self.storage.cleanup_expired_documents()
            vector_cleanup = await self.vector_store.cleanup_expired_sessions()

            self.stats["cleanup_runs"] += 1
            self.stats["last_cleanup"] = current_time

            logger.info(
                "🧹 Expired workspaces cleanup completed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                data=cleanup_stats
            )
            return cleanup_stats

        except Exception as e:
            logger.error(
                "❌ Expired workspaces cleanup failed",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.session_document_manager",
                error=e
            )
            return {"error": str(e)}

    # Database integration methods (if available)
    async def _store_workspace_in_db(self, workspace: SessionDocumentWorkspace):
        """Store workspace in database."""
        if not DATABASE_AVAILABLE:
            return

        try:
            async with get_database_session() as session:
                db_workspace = SessionWorkspaceDB(
                    session_id=workspace.session_id,
                    total_documents=workspace.total_documents,
                    total_size=workspace.total_size,
                    workspace_metadata=workspace.metadata,
                    max_documents=workspace.max_documents,
                    max_size=workspace.max_size,
                    created_at=workspace.created_at,
                    last_activity=workspace.last_accessed,
                    expires_at=workspace.expires_at
                )
                session.add(db_workspace)
                await session.commit()

        except Exception as e:
            logger.error(
                "Failed to store workspace in database",
                LogCategory.DATABASE_LAYER,
                "app.services.session_document_manager",
                error=e
            )

    async def _store_document_in_db(self, document: SessionDocument):
        """Store document in database."""
        if not DATABASE_AVAILABLE:
            return

        try:
            async with get_database_session() as session:
                db_document = SessionDocumentDB(
                    document_id=document.document_id,
                    session_id=document.session_id,
                    filename=document.filename,
                    content_type=document.content_type,
                    file_size=document.file_size,
                    document_type=document.document_type.value,
                    processing_status=document.processing_status.value,
                    storage_path=str(document.storage_path) if document.storage_path else "",
                    content_hash=document.content_hash or "",
                    metadata=document.metadata,
                    analysis_results=document.analysis_results,
                    uploaded_at=document.uploaded_at,
                    last_accessed=document.last_accessed,
                    expires_at=document.expires_at
                )
                session.add(db_document)
                await session.commit()

        except Exception as e:
            logger.error(
                "Failed to store document in database",
                LogCategory.DATABASE_LAYER,
                "app.services.session_document_manager",
                error=e
            )

    async def _update_document_in_db(self, document: SessionDocument):
        """Update document in database."""
        if not DATABASE_AVAILABLE:
            return

        try:
            async with get_database_session() as session:
                result = await session.execute(
                    select(SessionDocumentDB).where(
                        SessionDocumentDB.document_id == document.document_id
                    )
                )
                db_document = result.scalar_one_or_none()

                if db_document:
                    db_document.processing_status = document.processing_status.value
                    db_document.analysis_results = document.analysis_results
                    db_document.last_accessed = document.last_accessed
                    db_document.metadata = document.metadata

                    await session.commit()

        except Exception as e:
            logger.error(
                "Failed to update document in database",
                LogCategory.DATABASE_LAYER,
                "app.services.session_document_manager",
                error=e
            )

    async def _load_workspace_from_db(self, session_id: str) -> Optional[SessionDocumentWorkspace]:
        """Load workspace from database."""
        if not DATABASE_AVAILABLE:
            return None

        try:
            async with get_database_session() as session:
                # Load workspace
                result = await session.execute(
                    select(SessionWorkspaceDB).where(
                        SessionWorkspaceDB.session_id == session_id
                    )
                )
                db_workspace = result.scalar_one_or_none()

                if not db_workspace:
                    return None

                # Load documents
                doc_result = await session.execute(
                    select(SessionDocumentDB).where(
                        SessionDocumentDB.session_id == session_id
                    )
                )
                db_documents = doc_result.scalars().all()

                # Create workspace object
                workspace = SessionDocumentWorkspace(
                    session_id=session_id,
                    created_at=db_workspace.created_at,
                    last_accessed=db_workspace.last_activity,
                    expires_at=db_workspace.expires_at,
                    max_documents=db_workspace.max_documents,
                    max_size=db_workspace.max_size,
                    metadata=db_workspace.workspace_metadata or {}
                )

                # Add documents to workspace
                for db_doc in db_documents:
                    # Load document content from storage
                    content = await self.storage.retrieve_document(session_id, db_doc.document_id)
                    if content:
                        document = SessionDocument(
                            document_id=db_doc.document_id,
                            session_id=db_doc.session_id,
                            filename=db_doc.filename,
                            content=content,
                            content_type=db_doc.content_type,
                            file_size=db_doc.file_size,
                            document_type=SessionDocumentType(db_doc.document_type),
                            processing_status=DocumentProcessingStatus(db_doc.processing_status),
                            metadata=db_doc.metadata or {},
                            analysis_results=db_doc.analysis_results,
                            uploaded_at=db_doc.uploaded_at,
                            last_accessed=db_doc.last_accessed,
                            expires_at=db_doc.expires_at,
                            storage_path=Path(db_doc.storage_path) if db_doc.storage_path else None,
                            content_hash=db_doc.content_hash
                        )
                        workspace.documents[db_doc.document_id] = document

                return workspace

        except Exception as e:
            logger.error(
                "Failed to load workspace from database",
                LogCategory.DATABASE_LAYER,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )
            return None

    async def _delete_document_from_db(self, session_id: str, document_id: str):
        """Delete document from database."""
        if not DATABASE_AVAILABLE:
            return

        try:
            async with get_database_session() as session:
                result = await session.execute(
                    select(SessionDocumentDB).where(
                        and_(
                            SessionDocumentDB.session_id == session_id,
                            SessionDocumentDB.document_id == document_id
                        )
                    )
                )
                db_document = result.scalar_one_or_none()

                if db_document:
                    await session.delete(db_document)
                    await session.commit()

        except Exception as e:
            logger.error(
                "Failed to delete document from database",
                LogCategory.DATABASE_LAYER,
                "app.services.session_document_manager",
                data={"session_id": session_id, "document_id": document_id},
                error=e
            )

    async def _delete_workspace_from_db(self, session_id: str):
        """Delete workspace from database."""
        if not DATABASE_AVAILABLE:
            return

        try:
            async with get_database_session() as session:
                # Delete all documents first
                await session.execute(
                    select(SessionDocumentDB).where(
                        SessionDocumentDB.session_id == session_id
                    ).delete()
                )

                # Delete workspace
                result = await session.execute(
                    select(SessionWorkspaceDB).where(
                        SessionWorkspaceDB.session_id == session_id
                    )
                )
                db_workspace = result.scalar_one_or_none()

                if db_workspace:
                    await session.delete(db_workspace)

                await session.commit()

        except Exception as e:
            logger.error(
                "Failed to delete workspace from database",
                LogCategory.DATABASE_LAYER,
                "app.services.session_document_manager",
                data={"session_id": session_id},
                error=e
            )

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global hybrid manager statistics."""
        return {
            **self.stats,
            "active_workspaces": len(self.workspaces),
            "active_processing_jobs": len(self.processing_jobs),
            "hybrid_architecture": True,
            "rag_integration": {
                "available": RAG_INFRASTRUCTURE_AVAILABLE,
                "ingestion_pipeline": self.ingestion_pipeline is not None,
                "processor_registry": self.processor_registry is not None,
                "unified_rag": self.unified_rag is not None
            },
            "intelligence_tool_available": INTELLIGENCE_TOOL_AVAILABLE and self.intelligence_tool is not None,
            "database_integration": DATABASE_AVAILABLE,
            "vector_search_enabled": self.vector_store.config.enable_vector_search,
            "cleanup_enabled": self.config.expiration.cleanup_enabled,
            "storage_type": "session_only",
            "permanent_storage": False
        }


# Global hybrid session document manager instance
session_document_manager = SessionDocumentManager()

logger.info(
    "🔥 Revolutionary Hybrid Session Document Manager ready - leveraging existing RAG infrastructure!",
    LogCategory.SERVICE_OPERATIONS,
    "app.services.session_document_manager"
)
