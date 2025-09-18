"""
Knowledge Tools for Revolutionary RAG-Enabled Agents.

This module provides LangChain-compatible tools that enable agents to interact
with the knowledge base for advanced search, document ingestion, fact-checking,
and knowledge synthesis capabilities.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union, Type
from datetime import datetime

import structlog
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.collection_based_kb_manager import CollectionBasedKBManager
from ..ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger(__name__)


class KnowledgeSearchInput(BaseModel):
    """Input schema for knowledge search tool."""
    query: str = Field(..., description="Search query for knowledge base")
    collection: Optional[str] = Field(default=None, description="Specific collection to search")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")


class DocumentIngestInput(BaseModel):
    """Input schema for document ingestion tool."""
    content: str = Field(..., description="Document content to ingest")
    title: str = Field(..., description="Document title")
    collection: Optional[str] = Field(default=None, description="Target collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    document_type: str = Field(default="text", description="Type of document")


class FactCheckInput(BaseModel):
    """Input schema for fact checking tool."""
    statement: str = Field(..., description="Statement to fact-check")
    collection: Optional[str] = Field(default=None, description="Collection to check against")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")


class SynthesisInput(BaseModel):
    """Input schema for knowledge synthesis tool."""
    topic: str = Field(..., description="Topic to synthesize knowledge about")
    collections: List[str] = Field(default_factory=list, description="Collections to synthesize from")
    max_sources: int = Field(default=20, ge=5, le=100, description="Maximum sources to consider")


class KnowledgeSearchTool(BaseTool):
    """
    Advanced knowledge search tool for agents.
    
    Enables agents to search the knowledge base with sophisticated queries,
    filters, and retrieval strategies.
    """
    
    name: str = "knowledge_search"
    description: str = """
    Search the knowledge base for relevant information.
    
    Use this tool when you need to:
    - Find specific information or facts
    - Research a topic comprehensively
    - Gather context for answering questions
    - Retrieve relevant documents or passages
    
    The tool supports:
    - Semantic search across all knowledge
    - Collection-specific searches
    - Metadata filtering
    - Relevance scoring
    """
    args_schema: Type[BaseModel] = KnowledgeSearchInput
    
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async search."""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute knowledge search."""
        try:
            # Parse input
            search_input = KnowledgeSearchInput(**kwargs)
            
            # Create knowledge query
            query = KnowledgeQuery(
                query=search_input.query,
                collection=search_input.collection,
                top_k=search_input.top_k,
                filters=search_input.filters
            )
            
            # Execute search
            result = await self.knowledge_base.search(query)
            
            # Format results
            if not result.results:
                return json.dumps({
                    "success": True,
                    "message": "No relevant information found",
                    "query": search_input.query,
                    "results": []
                })
            
            # Prepare response
            formatted_results = []
            for i, search_result in enumerate(result.results):
                formatted_results.append({
                    "rank": i + 1,
                    "content": search_result.content[:500] + "..." if len(search_result.content) > 500 else search_result.content,
                    "score": round(search_result.score, 3),
                    "metadata": search_result.metadata,
                    "source": search_result.metadata.get("document_title", "Unknown")
                })
            
            response = {
                "success": True,
                "query": search_input.query,
                "total_results": result.total_results,
                "processing_time": round(result.processing_time, 3),
                "collection": result.collection,
                "results": formatted_results
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "query": kwargs.get("query", "unknown")
            })


class DocumentIngestTool(BaseTool):
    """
    Document ingestion tool for agents.
    
    Enables agents to add new documents to the knowledge base,
    expanding the available knowledge dynamically.
    """
    
    name: str = "document_ingest"
    description: str = """
    Add a new document to the knowledge base.
    
    Use this tool when you need to:
    - Store important information for future reference
    - Add research findings to the knowledge base
    - Preserve conversation insights
    - Build up domain-specific knowledge
    
    The tool will:
    - Process and chunk the document
    - Generate embeddings
    - Store in the specified collection
    - Make it searchable immediately
    """
    args_schema: Type[BaseModel] = DocumentIngestInput
    
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async ingestion."""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute document ingestion."""
        try:
            # Parse input
            ingest_input = DocumentIngestInput(**kwargs)
            
            # Create document
            document = Document(
                title=ingest_input.title,
                content=ingest_input.content,
                metadata={
                    **ingest_input.metadata,
                    "ingested_by": "agent",
                    "ingested_at": datetime.utcnow().isoformat()
                },
                document_type=ingest_input.document_type,
                source="agent_ingestion"
            )
            
            # Add to knowledge base
            document_id = await self.knowledge_base.add_document(
                document,
                collection=ingest_input.collection
            )
            
            response = {
                "success": True,
                "message": "Document successfully added to knowledge base",
                "document_id": document_id,
                "title": ingest_input.title,
                "collection": ingest_input.collection or self.knowledge_base.config.default_collection,
                "chunks_created": document.chunk_count,
                "content_length": len(ingest_input.content)
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "title": kwargs.get("title", "unknown")
            })


class FactCheckTool(BaseTool):
    """
    Fact-checking tool using knowledge base.
    
    Enables agents to verify statements against stored knowledge
    and provide confidence assessments.
    """
    
    name: str = "fact_check"
    description: str = """
    Verify a statement against the knowledge base.
    
    Use this tool when you need to:
    - Check the accuracy of information
    - Verify claims against stored knowledge
    - Assess confidence in statements
    - Find supporting or contradicting evidence
    
    The tool will:
    - Search for relevant information
    - Compare against the statement
    - Provide confidence assessment
    - Return supporting evidence
    """
    args_schema: Type[BaseModel] = FactCheckInput
    
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async fact checking."""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute fact checking."""
        try:
            # Parse input
            fact_input = FactCheckInput(**kwargs)
            
            # Search for relevant information
            query = KnowledgeQuery(
                query=fact_input.statement,
                collection=fact_input.collection,
                top_k=10
            )
            
            result = await self.knowledge_base.search(query)
            
            if not result.results:
                return json.dumps({
                    "success": True,
                    "statement": fact_input.statement,
                    "verification": "insufficient_data",
                    "confidence": 0.0,
                    "message": "No relevant information found to verify this statement",
                    "evidence": []
                })
            
            # Analyze results for fact checking
            supporting_evidence = []
            contradicting_evidence = []
            
            for search_result in result.results:
                if search_result.score >= fact_input.confidence_threshold:
                    # Simple keyword-based analysis (can be enhanced with NLP)
                    content_lower = search_result.content.lower()
                    statement_lower = fact_input.statement.lower()
                    
                    # Extract key terms from statement
                    statement_words = set(statement_lower.split())
                    content_words = set(content_lower.split())
                    
                    overlap = len(statement_words.intersection(content_words))
                    
                    evidence_item = {
                        "content": search_result.content[:300] + "..." if len(search_result.content) > 300 else search_result.content,
                        "score": round(search_result.score, 3),
                        "source": search_result.metadata.get("document_title", "Unknown"),
                        "overlap_score": overlap / len(statement_words) if statement_words else 0
                    }
                    
                    supporting_evidence.append(evidence_item)
            
            # Calculate overall confidence
            if supporting_evidence:
                avg_score = sum(e["score"] for e in supporting_evidence) / len(supporting_evidence)
                confidence = min(avg_score, 1.0)
                verification = "supported" if confidence >= fact_input.confidence_threshold else "uncertain"
            else:
                confidence = 0.0
                verification = "unsupported"
            
            response = {
                "success": True,
                "statement": fact_input.statement,
                "verification": verification,
                "confidence": round(confidence, 3),
                "evidence_count": len(supporting_evidence),
                "evidence": supporting_evidence[:5],  # Top 5 pieces of evidence
                "threshold": fact_input.confidence_threshold
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Fact checking failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "statement": kwargs.get("statement", "unknown")
            })


class SynthesisTool(BaseTool):
    """
    Knowledge synthesis tool for comprehensive analysis.
    
    Enables agents to synthesize information from multiple sources
    to create comprehensive understanding of topics.
    """
    
    name: str = "knowledge_synthesis"
    description: str = """
    Synthesize knowledge from multiple sources about a topic.
    
    Use this tool when you need to:
    - Create comprehensive overviews
    - Combine information from multiple sources
    - Generate insights from diverse knowledge
    - Build holistic understanding
    
    The tool will:
    - Search across multiple collections
    - Gather diverse perspectives
    - Identify key themes and patterns
    - Provide synthesized summary
    """
    args_schema: Type[BaseModel] = SynthesisInput
    
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, **kwargs) -> str:
        """Synchronous wrapper for async synthesis."""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute knowledge synthesis."""
        try:
            # Parse input
            synthesis_input = SynthesisInput(**kwargs)
            
            # Determine collections to search
            collections = synthesis_input.collections
            if not collections:
                collections = [self.knowledge_base.config.default_collection]
            
            # Search across collections
            all_results = []
            for collection in collections:
                query = KnowledgeQuery(
                    query=synthesis_input.topic,
                    collection=collection,
                    top_k=synthesis_input.max_sources // len(collections)
                )
                
                result = await self.knowledge_base.search(query)
                all_results.extend(result.results)
            
            if not all_results:
                return json.dumps({
                    "success": True,
                    "topic": synthesis_input.topic,
                    "message": "No relevant information found for synthesis",
                    "sources": 0,
                    "synthesis": {}
                })
            
            # Sort by relevance
            all_results.sort(key=lambda x: x.score, reverse=True)
            top_results = all_results[:synthesis_input.max_sources]
            
            # Synthesize information
            synthesis = {
                "topic": synthesis_input.topic,
                "total_sources": len(top_results),
                "collections_searched": collections,
                "key_sources": [],
                "themes": [],
                "summary": ""
            }
            
            # Extract key sources
            for i, result in enumerate(top_results[:10]):
                synthesis["key_sources"].append({
                    "rank": i + 1,
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "score": round(result.score, 3),
                    "source": result.metadata.get("document_title", "Unknown")
                })
            
            # Simple theme extraction (can be enhanced with NLP)
            all_content = " ".join([r.content for r in top_results])
            words = all_content.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 4:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top themes
            top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            synthesis["themes"] = [{"term": term, "frequency": freq} for term, freq in top_themes]
            
            # Create summary
            synthesis["summary"] = f"Found {len(top_results)} relevant sources about '{synthesis_input.topic}' across {len(collections)} collections. Key themes include: {', '.join([t['term'] for t in synthesis['themes'][:5]])}."
            
            response = {
                "success": True,
                "synthesis": synthesis
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Knowledge synthesis failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "topic": kwargs.get("topic", "unknown")
            })


class KnowledgeManagementTool(BaseTool):
    """
    Knowledge base management tool for agents.
    
    Enables agents to manage collections, get statistics,
    and perform administrative tasks.
    """
    
    name: str = "knowledge_management"
    description: str = """
    Manage the knowledge base and get information about collections.
    
    Use this tool when you need to:
    - List available collections
    - Get knowledge base statistics
    - Create new collections
    - Check system status
    
    Available operations:
    - list_collections: Get all available collections
    - get_stats: Get knowledge base statistics
    - create_collection: Create a new collection
    - collection_stats: Get statistics for a specific collection
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, operation: str, **kwargs) -> str:
        """Synchronous wrapper for async management operations."""
        return asyncio.run(self._arun(operation, **kwargs))
    
    async def _arun(self, operation: str, **kwargs) -> str:
        """Execute knowledge management operations."""
        try:
            if operation == "list_collections":
                collections = await self.knowledge_base.get_collections()
                return json.dumps({
                    "success": True,
                    "operation": "list_collections",
                    "collections": collections,
                    "total": len(collections)
                }, indent=2)
            
            elif operation == "get_stats":
                stats = await self.knowledge_base.get_stats()
                return json.dumps({
                    "success": True,
                    "operation": "get_stats",
                    "stats": stats
                }, indent=2)
            
            elif operation == "create_collection":
                collection_name = kwargs.get("name")
                if not collection_name:
                    raise ValueError("Collection name is required")
                
                await self.knowledge_base.create_collection(collection_name)
                return json.dumps({
                    "success": True,
                    "operation": "create_collection",
                    "collection": collection_name,
                    "message": f"Collection '{collection_name}' created successfully"
                }, indent=2)
            
            elif operation == "collection_stats":
                collection_name = kwargs.get("collection")
                stats = await self.knowledge_base.vector_store.get_collection_stats(collection_name)
                return json.dumps({
                    "success": True,
                    "operation": "collection_stats",
                    "collection": collection_name,
                    "stats": stats
                }, indent=2)
            
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "available_operations": ["list_collections", "get_stats", "create_collection", "collection_stats"]
                })
                
        except Exception as e:
            logger.error(f"Knowledge management operation failed: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "operation": operation
            })
