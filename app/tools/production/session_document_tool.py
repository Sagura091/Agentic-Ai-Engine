"""
🔥 REVOLUTIONARY SESSION DOCUMENT TOOL - PHASE 2
==============================================

Agent tool for interacting with session-based documents that leverages existing
RAG infrastructure while maintaining complete session isolation.

REVOLUTIONARY FEATURES:
- Session-scoped document upload and management
- Hybrid architecture using existing RAG infrastructure
- Multi-modal processing (text, images, video, audio)
- AI-powered document analysis and querying
- Complete session cleanup (no permanent storage pollution)
- Integration with Revolutionary Document Intelligence Tool

Author: Revolutionary AI Agent System
Version: 2.0.0 - Hybrid Architecture
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory

# Import session document system
try:
    from app.services.session_document_manager import session_document_manager
    from app.models.session_document_models import (
        SessionDocumentType,
        DocumentProcessingStatus,
        SessionDocumentQuery,
        SessionDocumentAnalysisRequest
    )
    SESSION_SYSTEM_AVAILABLE = True
except ImportError:
    SESSION_SYSTEM_AVAILABLE = False

logger = get_logger()


class SessionDocumentUploadInput(BaseModel):
    """Input for uploading documents to session workspace."""
    session_id: str = Field(description="Session identifier for document workspace")
    file_content: Union[str, bytes] = Field(description="Document content (text or base64 encoded bytes)")
    filename: str = Field(description="Name of the file being uploaded")
    content_type: str = Field(default="text/plain", description="MIME type of the document")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the document")
    encoding: str = Field(default="utf-8", description="Text encoding (for text files)")


class SessionDocumentQueryInput(BaseModel):
    """Input for querying session documents."""
    session_id: str = Field(description="Session identifier")
    query: str = Field(description="Search query for finding relevant documents")
    top_k: int = Field(default=5, description="Maximum number of results to return")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity score for results")
    include_metadata: bool = Field(default=True, description="Include document metadata in results")


class SessionDocumentAnalysisInput(BaseModel):
    """Input for AI-powered document analysis."""
    session_id: str = Field(description="Session identifier")
    document_id: Optional[str] = Field(default=None, description="Specific document ID to analyze (optional)")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: comprehensive, summary, key_points, sentiment")
    custom_prompt: Optional[str] = Field(default=None, description="Custom analysis prompt")


class SessionDocumentListInput(BaseModel):
    """Input for listing session documents."""
    session_id: str = Field(description="Session identifier")
    include_content: bool = Field(default=False, description="Include document content in response")
    include_metadata: bool = Field(default=True, description="Include document metadata")


class SessionDocumentDeleteInput(BaseModel):
    """Input for deleting session documents."""
    session_id: str = Field(description="Session identifier")
    document_id: Optional[str] = Field(default=None, description="Specific document ID to delete (optional - if not provided, cleans entire session)")


class SessionDocumentTool(BaseTool):
    """
    🔥 REVOLUTIONARY SESSION DOCUMENT TOOL - HYBRID ARCHITECTURE

    Provides agents with powerful session-based document capabilities:

    CORE OPERATIONS:
    - upload: Upload documents to session workspace
    - query: Search and retrieve relevant document content
    - analyze: AI-powered document analysis and insights
    - list: List all documents in session workspace
    - delete: Remove specific documents or clean entire session
    - stats: Get session workspace statistics

    HYBRID FEATURES:
    - Leverages existing RAG infrastructure for processing
    - Maintains session-only storage (no permanent contamination)
    - Multi-modal processing through existing vision models
    - Revolutionary Document Intelligence integration
    - Complete cleanup when session ends
    """

    name: str = "session_document_tool"
    description: str = """
    Revolutionary session document tool for temporary document processing.

    Operations:
    - upload: Upload documents to session workspace
    - query: Search documents with natural language queries
    - analyze: AI-powered document analysis and insights
    - list: List all documents in session
    - delete: Remove documents or clean session
    - stats: Get workspace statistics

    Format: operation:session_id:parameters
    Example: upload:session123:{"filename":"doc.pdf","content":"..."}
    """

    def __init__(self):
        super().__init__()
        logger.info(
            "🔥 Revolutionary Session Document Tool initialized",
            LogCategory.TOOL_OPERATIONS,
            "SessionDocumentTool"
        )

    @property
    def manager(self):
        """Get the session document manager instance."""
        return session_document_manager if SESSION_SYSTEM_AVAILABLE else None
    
    async def _arun(self, operation_input: str) -> str:
        """Execute session document operations asynchronously."""
        try:
            if not SESSION_SYSTEM_AVAILABLE:
                return json.dumps({
                    "success": False,
                    "error": "Session document system not available",
                    "message": "Please ensure session document system is properly installed"
                })
            
            if not self.manager.is_initialized:
                await self.manager.initialize()
            
            # Parse operation input
            parts = operation_input.split(":", 2)
            if len(parts) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Invalid input format",
                    "message": "Expected format: operation:session_id:parameters"
                })
            
            operation = parts[0].lower().strip()
            session_id = parts[1].strip()
            parameters = json.loads(parts[2]) if len(parts) > 2 and parts[2].strip() else {}
            
            # Route to appropriate operation
            if operation == "upload":
                return await self._handle_upload(session_id, parameters)
            elif operation == "query":
                return await self._handle_query(session_id, parameters)
            elif operation == "analyze":
                return await self._handle_analysis(session_id, parameters)
            elif operation == "list":
                return await self._handle_list(session_id, parameters)
            elif operation == "delete":
                return await self._handle_delete(session_id, parameters)
            elif operation == "stats":
                return await self._handle_stats(session_id, parameters)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "available_operations": ["upload", "query", "analyze", "list", "delete", "stats"]
                })
                
        except Exception as e:
            logger.error(
                f"Session document tool error: {e}",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": "Session document operation failed"
            })
    
    def _run(self, operation_input: str) -> str:
        """Synchronous wrapper for async operations."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._arun(operation_input))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(operation_input))
            finally:
                loop.close()

    async def _handle_upload(self, session_id: str, parameters: Dict[str, Any]) -> str:
        """Handle document upload to session workspace."""
        try:
            # Validate required parameters
            if "filename" not in parameters or "content" not in parameters:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameters",
                    "required": ["filename", "content"]
                })

            # Prepare content
            content = parameters["content"]
            if isinstance(content, str):
                # Handle base64 encoded content or plain text
                if content.startswith("data:"):
                    # Extract base64 content from data URL
                    import base64
                    header, encoded = content.split(",", 1)
                    content = base64.b64decode(encoded)
                else:
                    # Plain text content
                    encoding = parameters.get("encoding", "utf-8")
                    content = content.encode(encoding)

            # Upload document
            document_response = await self.manager.upload_document(
                session_id=session_id,
                file_content=content,
                filename=parameters["filename"],
                content_type=parameters.get("content_type", "text/plain"),
                metadata=parameters.get("metadata", {})
            )

            logger.info(
                "📤 Document uploaded via agent tool",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                data={
                    "session_id": session_id,
                    "document_id": document_response.document_id,
                    "filename": document_response.filename
                }
            )

            return json.dumps({
                "success": True,
                "operation": "upload",
                "session_id": session_id,
                "document": {
                    "document_id": document_response.document_id,
                    "filename": document_response.filename,
                    "content_type": document_response.content_type,
                    "file_size": document_response.file_size,
                    "processing_status": document_response.processing_status.value if hasattr(document_response.processing_status, 'value') else str(document_response.processing_status),
                    "uploaded_at": document_response.uploaded_at.isoformat(),
                    "metadata": document_response.metadata
                },
                "message": f"Document '{document_response.filename}' uploaded successfully to session workspace"
            })

        except Exception as e:
            logger.error(
                f"Upload operation failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": "upload",
                "session_id": session_id
            })

    async def _handle_query(self, session_id: str, parameters: Dict[str, Any]) -> str:
        """Handle document query and search operations."""
        try:
            # Validate required parameters
            if "query" not in parameters:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: query"
                })

            # Execute query
            results = await self.manager.query_session_documents(
                session_id=session_id,
                query=parameters["query"],
                top_k=parameters.get("top_k", 5),
                similarity_threshold=parameters.get("similarity_threshold", 0.3)
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

                if parameters.get("include_metadata", True):
                    formatted_result["metadata"] = result.get("metadata", {})

                formatted_results.append(formatted_result)

            logger.info(
                "🔍 Document query executed via agent tool",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                data={
                    "session_id": session_id,
                    "query": parameters["query"],
                    "results_count": len(results)
                }
            )

            return json.dumps({
                "success": True,
                "operation": "query",
                "session_id": session_id,
                "query": parameters["query"],
                "results_count": len(results),
                "results": formatted_results,
                "message": f"Found {len(results)} relevant document segments"
            })

        except Exception as e:
            logger.error(
                f"Query operation failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": "query",
                "session_id": session_id
            })

    async def _handle_analysis(self, session_id: str, parameters: Dict[str, Any]) -> str:
        """Handle AI-powered document analysis operations."""
        try:
            # Check if Revolutionary Document Intelligence Tool is available
            if not self.manager.intelligence_tool:
                return json.dumps({
                    "success": False,
                    "error": "Document intelligence tool not available",
                    "message": "AI-powered analysis requires Revolutionary Document Intelligence Tool"
                })

            analysis_type = parameters.get("analysis_type", "comprehensive")
            document_id = parameters.get("document_id")
            custom_prompt = parameters.get("custom_prompt")

            # Get documents for analysis
            if document_id:
                # Analyze specific document
                document = await self.manager.get_document(session_id, document_id)
                if not document:
                    return json.dumps({
                        "success": False,
                        "error": f"Document not found: {document_id}"
                    })
                documents = [document]
            else:
                # Analyze all documents in session
                documents = await self.manager.list_session_documents(session_id)

            if not documents:
                return json.dumps({
                    "success": False,
                    "error": "No documents found for analysis",
                    "session_id": session_id
                })

            # Perform analysis using Revolutionary Document Intelligence Tool
            analysis_results = []
            for doc in documents:
                try:
                    # Prepare analysis request
                    if custom_prompt:
                        analysis_prompt = custom_prompt
                    else:
                        analysis_prompts = {
                            "comprehensive": "Provide a comprehensive analysis of this document including key themes, important information, and insights.",
                            "summary": "Provide a concise summary of the main points and key information in this document.",
                            "key_points": "Extract and list the key points, important facts, and main takeaways from this document.",
                            "sentiment": "Analyze the sentiment and tone of this document, identifying emotional indicators and overall mood."
                        }
                        analysis_prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])

                    # Use Revolutionary Document Intelligence Tool for analysis
                    intelligence_input = f"analyze:{doc.filename}:{analysis_prompt}"
                    analysis_result = await self.manager.intelligence_tool._arun(intelligence_input)

                    analysis_results.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "analysis_type": analysis_type,
                        "analysis": analysis_result,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                except Exception as e:
                    logger.warning(
                        f"Analysis failed for document {doc.document_id}: {e}",
                        LogCategory.TOOL_OPERATIONS,
                        "SessionDocumentTool",
                        data={"document_id": doc.document_id},
                        error=e
                    )
                    analysis_results.append({
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "analysis_type": analysis_type,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })

            logger.info(
                "🧠 Document analysis completed via agent tool",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                data={
                    "session_id": session_id,
                    "analysis_type": analysis_type,
                    "documents_analyzed": len(analysis_results)
                }
            )

            return json.dumps({
                "success": True,
                "operation": "analyze",
                "session_id": session_id,
                "analysis_type": analysis_type,
                "documents_analyzed": len(analysis_results),
                "results": analysis_results,
                "message": f"Analysis completed for {len(analysis_results)} documents"
            })

        except Exception as e:
            logger.error(
                f"Analysis operation failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": "analyze",
                "session_id": session_id
            })

    async def _handle_list(self, session_id: str, parameters: Dict[str, Any]) -> str:
        """Handle document listing operations."""
        try:
            # Get session documents
            documents = await self.manager.list_session_documents(session_id)

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

                if parameters.get("include_metadata", True):
                    doc_info["metadata"] = doc.metadata

                if parameters.get("include_content", False):
                    # Include content preview (first 500 characters)
                    content_preview = str(doc.content)[:500]
                    if len(str(doc.content)) > 500:
                        content_preview += "..."
                    doc_info["content_preview"] = content_preview

                formatted_docs.append(doc_info)

            # Get workspace statistics
            workspace_stats = await self.manager.get_workspace_stats(session_id)

            logger.info(
                "📋 Document listing completed via agent tool",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                data={
                    "session_id": session_id,
                    "documents_count": len(documents)
                }
            )

            return json.dumps({
                "success": True,
                "operation": "list",
                "session_id": session_id,
                "documents_count": len(documents),
                "documents": formatted_docs,
                "workspace_stats": {
                    "total_documents": workspace_stats.total_documents if workspace_stats else 0,
                    "total_size": workspace_stats.total_size if workspace_stats else 0,
                    "max_documents": workspace_stats.max_documents if workspace_stats else 0,
                    "max_size": workspace_stats.max_size if workspace_stats else 0
                } if workspace_stats else None,
                "message": f"Found {len(documents)} documents in session workspace"
            })

        except Exception as e:
            logger.error(
                f"List operation failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": "list",
                "session_id": session_id
            })

    async def _handle_delete(self, session_id: str, parameters: Dict[str, Any]) -> str:
        """Handle document deletion operations."""
        try:
            document_id = parameters.get("document_id")

            if document_id:
                # Delete specific document
                success = await self.manager.delete_document(session_id, document_id)

                if success:
                    logger.info(
                        "🗑️ Document deleted via agent tool",
                        LogCategory.TOOL_OPERATIONS,
                        "SessionDocumentTool",
                        data={
                            "session_id": session_id,
                            "document_id": document_id
                        }
                    )

                    return json.dumps({
                        "success": True,
                        "operation": "delete",
                        "session_id": session_id,
                        "document_id": document_id,
                        "message": f"Document {document_id} deleted successfully"
                    })
                else:
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to delete document: {document_id}",
                        "operation": "delete",
                        "session_id": session_id
                    })
            else:
                # Clean entire session workspace
                success = await self.manager.cleanup_workspace(session_id)

                if success:
                    logger.info(
                        "🧹 Session workspace cleaned via agent tool",
                        LogCategory.TOOL_OPERATIONS,
                        "SessionDocumentTool",
                        data={"session_id": session_id}
                    )

                    return json.dumps({
                        "success": True,
                        "operation": "delete",
                        "session_id": session_id,
                        "action": "cleanup_workspace",
                        "message": "Session workspace cleaned successfully - all documents removed"
                    })
                else:
                    return json.dumps({
                        "success": False,
                        "error": "Failed to cleanup session workspace",
                        "operation": "delete",
                        "session_id": session_id
                    })

        except Exception as e:
            logger.error(
                f"Delete operation failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": "delete",
                "session_id": session_id
            })

    async def _handle_stats(self, session_id: str, parameters: Dict[str, Any]) -> str:
        """Handle workspace statistics operations."""
        try:
            # Get workspace statistics
            workspace_stats = await self.manager.get_workspace_stats(session_id)

            # Get global manager statistics
            global_stats = self.manager.get_global_stats()

            # Get vector store statistics
            vector_stats = self.manager.vector_store.get_global_stats()

            logger.info(
                "📊 Statistics retrieved via agent tool",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                data={"session_id": session_id}
            )

            return json.dumps({
                "success": True,
                "operation": "stats",
                "session_id": session_id,
                "workspace_stats": {
                    "total_documents": workspace_stats.total_documents if workspace_stats else 0,
                    "total_size": workspace_stats.total_size if workspace_stats else 0,
                    "max_documents": workspace_stats.max_documents if workspace_stats else 0,
                    "max_size": workspace_stats.max_size if workspace_stats else 0,
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
                },
                "message": "Session workspace statistics retrieved successfully"
            })

        except Exception as e:
            logger.error(
                f"Stats operation failed: {e}",
                LogCategory.TOOL_OPERATIONS,
                "SessionDocumentTool",
                error=e
            )
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": "stats",
                "session_id": session_id
            })


# Global tool instance
session_document_tool = SessionDocumentTool()

logger.info(
    "🔥 Revolutionary Session Document Tool ready for agent integration!",
    LogCategory.TOOL_OPERATIONS,
    "SessionDocumentTool"
)
