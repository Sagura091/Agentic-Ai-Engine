"""
Enhanced Knowledge Tools for Multi-Agent RAG System.

These tools provide revolutionary RAG capabilities for agents with:
- Agent-specific knowledge management
- Context-aware retrieval
- Memory integration
- Advanced search strategies
- Knowledge sharing protocols
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Type
from datetime import datetime

import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.unified_rag_system import UnifiedRAGSystem
from ..core.collection_based_kb_manager import CollectionBasedKBManager, AccessLevel

logger = structlog.get_logger(__name__)


class EnhancedKnowledgeSearchInput(BaseModel):
    """Input schema for enhanced knowledge search tool."""
    query: str = Field(..., description="Search query for knowledge base")
    agent_id: str = Field(..., description="Agent performing the search")
    scopes: Optional[List[str]] = Field(default=None, description="Knowledge scopes to search")
    collections: Optional[List[str]] = Field(default=None, description="Specific collections to search")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    include_memories: bool = Field(default=True, description="Include agent memories in search")
    session_id: Optional[str] = Field(default=None, description="Current session ID")
    use_advanced_retrieval: bool = Field(default=True, description="Use advanced retrieval strategies")


class AgentDocumentIngestInput(BaseModel):
    """Input schema for agent document ingestion tool."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    agent_id: str = Field(..., description="Agent adding the document")
    scope: str = Field(default="private", description="Knowledge scope (private, shared, global)")
    collection: Optional[str] = Field(default=None, description="Target collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    document_type: str = Field(default="text", description="Document type")
    session_id: Optional[str] = Field(default=None, description="Current session ID")


class AgentMemoryInput(BaseModel):
    """Input schema for agent memory creation tool."""
    content: str = Field(..., description="Memory content")
    agent_id: str = Field(..., description="Agent creating the memory")
    memory_type: str = Field(default="episodic", description="Type of memory")
    context: Dict[str, Any] = Field(default_factory=dict, description="Memory context")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Memory importance")
    session_id: Optional[str] = Field(default=None, description="Current session ID")
    tags: List[str] = Field(default_factory=list, description="Memory tags")


class EnhancedKnowledgeSearchTool(BaseTool):
    """
    Enhanced knowledge search tool with agent-specific context.
    
    Provides revolutionary search capabilities including:
    - Agent-specific knowledge isolation
    - Context-aware retrieval
    - Memory integration
    - Advanced search strategies
    - Query expansion and re-ranking
    """
    
    name: str = "enhanced_knowledge_search"
    description: str = """
    Search the knowledge base with advanced agent-specific capabilities.
    
    Use this tool when you need to:
    - Find information relevant to your specific agent context
    - Search across your private knowledge and memories
    - Access domain-specific or global knowledge
    - Get contextually relevant results based on your history
    
    The tool provides:
    - Agent-specific knowledge isolation
    - Memory integration for personal context
    - Advanced retrieval with query expansion
    - Re-ranking based on agent preferences
    - Multi-scope search capabilities
    """
    args_schema: Type[BaseModel] = EnhancedKnowledgeSearchInput
    
    def __init__(self, rag_service: EnhancedRAGService):
        super().__init__()
        self.rag_service = rag_service
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async search."""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute enhanced knowledge search."""
        try:
            # Parse input
            search_input = EnhancedKnowledgeSearchInput(**kwargs)
            
            # Convert scope strings to enums
            scopes = None
            if search_input.scopes:
                scopes = [KnowledgeScope(scope) for scope in search_input.scopes]
            
            # Execute search
            result = await self.rag_service.search_knowledge(
                agent_id=search_input.agent_id,
                query=search_input.query,
                scopes=scopes,
                collections=search_input.collections,
                top_k=search_input.top_k,
                filters=search_input.filters,
                include_memories=search_input.include_memories,
                session_id=search_input.session_id,
                use_advanced_retrieval=search_input.use_advanced_retrieval
            )
            
            # Format results
            if not result.results:
                return json.dumps({
                    "success": True,
                    "message": "No relevant information found",
                    "query": search_input.query,
                    "agent_id": search_input.agent_id,
                    "results": []
                })
            
            formatted_results = []
            for r in result.results:
                formatted_results.append({
                    "content": r.content,
                    "score": round(r.score, 3),
                    "metadata": r.metadata,
                    "source": r.metadata.get("scope", "unknown"),
                    "type": r.metadata.get("type", "document")
                })
            
            response = {
                "success": True,
                "query": search_input.query,
                "agent_id": search_input.agent_id,
                "results": formatted_results,
                "total_results": result.total_results,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Enhanced knowledge search failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "query": kwargs.get("query", "unknown"),
                "agent_id": kwargs.get("agent_id", "unknown")
            })


class AgentDocumentIngestTool(BaseTool):
    """
    Agent-specific document ingestion tool.
    
    Enables agents to add documents to their knowledge base with:
    - Scope-aware storage (private, shared, global)
    - Agent ownership tracking
    - Automatic metadata enrichment
    - Collection management
    """
    
    name: str = "agent_document_ingest"
    description: str = """
    Add a document to your agent-specific knowledge base.
    
    Use this tool when you need to:
    - Store information for future reference
    - Add documents to your private knowledge
    - Share knowledge with other agents
    - Organize information in specific collections
    
    The tool provides:
    - Agent-specific knowledge isolation
    - Scope-based access control
    - Automatic metadata enrichment
    - Collection organization
    """
    args_schema: Type[BaseModel] = AgentDocumentIngestInput
    
    def __init__(self, rag_service: EnhancedRAGService):
        super().__init__()
        self.rag_service = rag_service
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async ingestion."""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute agent document ingestion."""
        try:
            # Parse input
            ingest_input = AgentDocumentIngestInput(**kwargs)
            
            # Create document
            document = Document(
                title=ingest_input.title,
                content=ingest_input.content,
                metadata={
                    **ingest_input.metadata,
                    "ingested_by": ingest_input.agent_id,
                    "ingested_at": datetime.utcnow().isoformat(),
                    "document_type": ingest_input.document_type
                },
                document_type=ingest_input.document_type,
                source="agent_ingestion"
            )
            
            # Convert scope string to enum
            scope = KnowledgeScope(ingest_input.scope)
            
            # Add document
            document_id = await self.rag_service.add_document(
                agent_id=ingest_input.agent_id,
                document=document,
                scope=scope,
                collection=ingest_input.collection,
                session_id=ingest_input.session_id
            )
            
            response = {
                "success": True,
                "document_id": document_id,
                "title": ingest_input.title,
                "agent_id": ingest_input.agent_id,
                "scope": ingest_input.scope,
                "collection": ingest_input.collection,
                "message": "Document successfully added to agent knowledge base"
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Agent document ingestion failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "title": kwargs.get("title", "unknown"),
                "agent_id": kwargs.get("agent_id", "unknown")
            })


class AgentMemoryTool(BaseTool):
    """
    Agent memory creation and management tool.
    
    Enables agents to create and manage their episodic and semantic memories:
    - Episodic memories for experiences and events
    - Semantic memories for learned facts and concepts
    - Importance-based retention
    - Context-aware storage
    """
    
    name: str = "agent_memory"
    description: str = """
    Create and store memories for your agent.
    
    Use this tool when you need to:
    - Remember important experiences or events
    - Store learned facts or concepts
    - Create context for future interactions
    - Build your agent's knowledge base
    
    The tool provides:
    - Episodic and semantic memory types
    - Importance-based retention
    - Context-aware storage
    - Automatic expiration management
    """
    args_schema: Type[BaseModel] = AgentMemoryInput
    
    def __init__(self, rag_service: EnhancedRAGService):
        super().__init__()
        self.rag_service = rag_service
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async memory creation."""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute agent memory creation."""
        try:
            # Parse input
            memory_input = AgentMemoryInput(**kwargs)
            
            # Add memory
            memory_id = await self.rag_service.add_memory(
                agent_id=memory_input.agent_id,
                content=memory_input.content,
                memory_type=memory_input.memory_type,
                context=memory_input.context,
                importance=memory_input.importance,
                session_id=memory_input.session_id,
                tags=memory_input.tags
            )
            
            response = {
                "success": True,
                "memory_id": memory_id,
                "agent_id": memory_input.agent_id,
                "memory_type": memory_input.memory_type,
                "importance": memory_input.importance,
                "message": "Memory successfully created and stored"
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Agent memory creation failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "agent_id": kwargs.get("agent_id", "unknown")
            })
