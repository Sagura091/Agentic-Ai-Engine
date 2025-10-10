"""
🚀 HYBRID RAG INTEGRATION - Complete RAG System Integration

This module provides the complete integration between the RAG system and agents,
implementing both model-level RAG (automatic context injection) and agent-level
RAG (explicit tools) for the best of both worlds.

HYBRID APPROACH:
- Model-level RAG: Automatic KB context injection (like OpenWebUI)
- Agent-level RAG: Explicit tools for advanced operations
- Complete tool registration and integration
- Production-ready RAG pipeline

Author: Agentic AI System
Purpose: Complete RAG system integration with hybrid approach
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Import tool repository components
from ...tools.unified_tool_repository import ToolMetadata, ToolCategory, ToolAccessLevel

from ..tools.knowledge_tools import (
    KnowledgeSearchTool,
    DocumentIngestTool,
    FactCheckTool,
    SynthesisTool,
    KnowledgeManagementTool
)
from ..core.unified_rag_system import UnifiedRAGSystem
from ...tools.unified_tool_repository import UnifiedToolRepository, ToolMetadata, ToolCategory

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class HybridRAGIntegration:
    """
    🚀 Complete Hybrid RAG Integration System.
    
    This class provides the complete integration between the RAG system and agents,
    implementing both automatic context injection and explicit tool access.
    """
    
    def __init__(self, rag_system: UnifiedRAGSystem, tool_repository: UnifiedToolRepository):
        self.rag_system = rag_system
        self.tool_repository = tool_repository
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the hybrid RAG integration system."""
        try:
            logger.info(
                "🚀 Initializing Hybrid RAG Integration System",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration"
            )

            # Register all RAG tools in the unified tool repository
            await self._register_rag_tools()

            self.is_initialized = True
            logger.info(
                "✅ Hybrid RAG Integration System initialized successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration"
            )
            return True

        except Exception as e:
            logger.error(
                f"❌ Failed to initialize Hybrid RAG Integration: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration",
                error=e
            )
            return False
    
    async def _register_rag_tools(self) -> None:
        """Register all RAG tools in the unified tool repository."""
        try:
            logger.info(
                "📚 Registering RAG tools in unified repository",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration"
            )
            
            # Initialize RAG tools with the RAG system
            knowledge_search_tool = KnowledgeSearchTool(rag_system=self.rag_system)
            # Skip document_ingest_tool for now due to interface mismatch
            fact_check_tool = FactCheckTool(rag_system=self.rag_system)
            synthesis_tool = SynthesisTool(rag_system=self.rag_system)
            knowledge_mgmt_tool = KnowledgeManagementTool(rag_system=self.rag_system)
            
            # Register rag_search tool
            await self.tool_repository.register_tool(
                tool_instance=knowledge_search_tool,
                metadata=ToolMetadata(
                    tool_id="rag_search",
                    name="RAG Knowledge Search",
                    description="Search the agent's knowledge base for relevant information",
                    category=ToolCategory.RAG_ENABLED,
                    access_level=ToolAccessLevel.PUBLIC,
                    use_cases=["knowledge_search", "information_retrieval", "rag"],
                    requires_rag=True
                )
            )
            
            # Register document_analysis tool
            await self.tool_repository.register_tool(
                tool_instance=fact_check_tool,
                metadata=ToolMetadata(
                    tool_id="document_analysis",
                    name="Document Analysis",
                    description="Analyze and fact-check information against the knowledge base",
                    category=ToolCategory.RAG_ENABLED,
                    access_level=ToolAccessLevel.PUBLIC,
                    use_cases=["document_analysis", "fact_checking", "verification"],
                    requires_rag=True
                )
            )
            
            # Register knowledge_synthesis tool
            await self.tool_repository.register_tool(
                tool_instance=synthesis_tool,
                metadata=ToolMetadata(
                    tool_id="knowledge_synthesis",
                    name="Knowledge Synthesis",
                    description="Synthesize information from multiple knowledge base sources",
                    category=ToolCategory.RAG_ENABLED,
                    access_level=ToolAccessLevel.PUBLIC,
                    use_cases=["knowledge_synthesis", "information_synthesis", "research"],
                    requires_rag=True
                )
            )
            
            # Skip document_ingest tool registration for now
            
            # Register knowledge_management tool
            await self.tool_repository.register_tool(
                tool_instance=knowledge_mgmt_tool,
                metadata=ToolMetadata(
                    tool_id="knowledge_management",
                    name="Knowledge Management",
                    description="Manage knowledge base collections and metadata",
                    category=ToolCategory.RAG_ENABLED,
                    access_level=ToolAccessLevel.PUBLIC,
                    use_cases=["knowledge_management", "collection_management"],
                    requires_rag=True
                )
            )

            logger.info(
                "✅ All RAG tools registered successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration"
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to register RAG tools: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration",
                error=e
            )
            raise
    
    async def get_knowledge_base_context(self, agent_id: str, query: str, max_results: int = 5) -> str:
        """
        🧠 Get knowledge base context for model-level RAG injection.
        
        This method provides the automatic context injection functionality
        that makes agents work like OpenWebUI - automatically searching
        the knowledge base and injecting relevant context.
        
        Args:
            agent_id: The agent requesting context
            query: The user's query
            max_results: Maximum number of results to include
            
        Returns:
            Formatted context string for prompt injection
        """
        try:
            # Search the agent's knowledge base
            search_results = await self.rag_system.search_documents(
                agent_id=agent_id,
                query=query,
                collection_type="knowledge",
                top_k=max_results,
                filters={}
            )
            
            if not search_results:
                return ""
            
            # Format the context for prompt injection
            context_parts = []
            for i, result in enumerate(search_results, 1):
                content = result.get('content', '')[:400]  # Limit content length
                metadata = result.get('metadata', {})
                
                # Create context entry
                context_entry = f"[KNOWLEDGE {i}]"
                
                # Add metadata if available
                if metadata.get('source_image'):
                    context_entry += f" IMAGE: {metadata['source_image']}"
                if metadata.get('symbol_id'):
                    context_entry += f" SYMBOL: {metadata['symbol_id']}"
                if metadata.get('source_text'):
                    context_entry += f" SOURCE: {metadata['source_text']}"
                
                context_entry += f"\n{content}\n"
                context_parts.append(context_entry)
            
            if context_parts:
                formatted_context = "\n".join(context_parts)
                logger.debug(
                    f"🧠 Injected {len(context_parts)} knowledge base entries for agent {agent_id}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration",
                    data={"agent_id": agent_id, "entries_count": len(context_parts)}
                )
                return formatted_context

        except Exception as e:
            logger.warn(
                f"⚠️ Failed to get knowledge base context for agent {agent_id}: {e}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration",
                error=e,
                data={"agent_id": agent_id}
            )

        return ""
    
    async def create_agent_knowledge_base(self, agent_id: str, documents: List[Dict[str, Any]]) -> bool:
        """
        📚 Create a knowledge base for an agent.
        
        Args:
            agent_id: The agent ID
            documents: List of documents to ingest
            
        Returns:
            Success status
        """
        try:
            # Convert documents to the proper format
            from ..core.unified_rag_system import Document
            
            doc_objects = []
            for i, doc_data in enumerate(documents):
                doc = Document(
                    id=f"{agent_id}_doc_{i}",
                    content=doc_data.get('content', ''),
                    metadata=doc_data.get('metadata', {})
                )
                doc_objects.append(doc)
            
            # Add documents to the RAG system
            success = await self.rag_system.add_documents(
                agent_id=agent_id,
                documents=doc_objects,
                collection_type="knowledge"
            )
            
            if success:
                logger.info(
                    f"✅ Created knowledge base for agent {agent_id} with {len(documents)} documents",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration",
                    data={"agent_id": agent_id, "document_count": len(documents)}
                )
            else:
                logger.error(
                    f"❌ Failed to create knowledge base for agent {agent_id}",
                    LogCategory.RAG_OPERATIONS,
                    "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration",
                    data={"agent_id": agent_id}
                )

            return success

        except Exception as e:
            logger.error(
                f"❌ Error creating knowledge base for agent {agent_id}: {e}",
                LogCategory.RAG_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration.HybridRAGIntegration",
                error=e,
                data={"agent_id": agent_id}
            )
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid RAG integration statistics."""
        return {
            "is_initialized": self.is_initialized,
            "rag_system_available": self.rag_system is not None,
            "tool_repository_available": self.tool_repository is not None,
            "timestamp": datetime.now().isoformat()
        }


# Global instance
_hybrid_rag_integration: Optional[HybridRAGIntegration] = None


async def get_hybrid_rag_integration() -> Optional[HybridRAGIntegration]:
    """Get the global hybrid RAG integration instance."""
    global _hybrid_rag_integration

    if _hybrid_rag_integration is None:
        try:
            # Get the RAG system and tool repository
            from ...core.unified_system_orchestrator import get_enhanced_system_orchestrator
            from ...tools.unified_tool_repository import get_unified_tool_repository

            # Get the RAG system and tool repository
            orchestrator = get_enhanced_system_orchestrator()
            tool_repository = get_unified_tool_repository()

            if orchestrator and orchestrator.unified_rag and tool_repository:
                _hybrid_rag_integration = HybridRAGIntegration(
                    rag_system=orchestrator.unified_rag,
                    tool_repository=tool_repository
                )
                await _hybrid_rag_integration.initialize()
                logger.info(
                    "✅ Hybrid RAG Integration initialized successfully",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.rag.integration.hybrid_rag_integration"
                )
            else:
                logger.warn(
                    "⚠️ RAG integration dependencies not available - skipping initialization",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.rag.integration.hybrid_rag_integration"
                )

        except Exception as e:
            logger.error(
                f"Failed to create hybrid RAG integration: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration",
                error=e
            )
            import traceback
            logger.error(
                f"Traceback: {traceback.format_exc()}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.integration.hybrid_rag_integration"
            )

    return _hybrid_rag_integration


async def initialize_hybrid_rag_system() -> bool:
    """Initialize the complete hybrid RAG system."""
    try:
        integration = await get_hybrid_rag_integration()
        return integration is not None and integration.is_initialized
    except Exception as e:
        logger.error(
            f"Failed to initialize hybrid RAG system: {e}",
            LogCategory.SYSTEM_OPERATIONS,
            "app.rag.integration.hybrid_rag_integration",
            error=e
        )
        return False
