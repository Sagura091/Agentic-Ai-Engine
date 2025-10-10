"""
Collection-Based Knowledge Base Manager - THE Knowledge Base System.

This is THE ONLY knowledge base manager in the entire application.
All knowledge base operations flow through this unified manager.

CORE ARCHITECTURE:
- One-to-one mapping: Agent -> Knowledge Base -> ChromaDB Collection
- Simple access control: PRIVATE (owner only) or PUBLIC (all agents)
- Clean, fast operations with minimal complexity
- Agent isolation through collection naming

DESIGN PRINCIPLES:
- One knowledge base per agent by default
- Simple, clean, fast operations
- No complexity unless absolutely necessary
- Agent isolation is paramount
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from .unified_rag_system import UnifiedRAGSystem, AgentCollections, Document

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class AccessLevel(str, Enum):
    """Access levels for knowledge bases - SIMPLIFIED."""
    PRIVATE = "private"    # Only owner can access
    PUBLIC = "public"      # All agents can access
    SHARED = "shared"      # Specific agents can access


@dataclass
class KnowledgeBaseInfo:
    """Information about a knowledge base - SIMPLIFIED."""
    kb_id: str
    name: str
    description: str
    owner_agent_id: str
    access_level: AccessLevel
    collection_name: str
    created_at: datetime
    last_updated: datetime
    document_count: int = 0
    shared_with: Set[str] = field(default_factory=set)  # Agent IDs with access

    def can_access(self, agent_id: str) -> bool:
        """Check if an agent can access this knowledge base."""
        if self.access_level == AccessLevel.PRIVATE:
            return agent_id == self.owner_agent_id
        elif self.access_level == AccessLevel.PUBLIC:
            return True
        elif self.access_level == AccessLevel.SHARED:
            return agent_id == self.owner_agent_id or agent_id in self.shared_with
        return False


class CollectionBasedKBManager:
    """
    Collection-Based Knowledge Base Manager - THE Knowledge Base System.

    SIMPLIFIED ARCHITECTURE:
    - One knowledge base per agent by default
    - Uses agent's dedicated ChromaDB collection
    - Simple access control (private/public/shared)
    - Clean, fast operations
    """

    def __init__(self, unified_rag: UnifiedRAGSystem):
        """Initialize THE knowledge base manager."""
        self.unified_rag = unified_rag
        self.is_initialized = False

        # Knowledge base registry - SIMPLIFIED
        self.knowledge_bases: Dict[str, KnowledgeBaseInfo] = {}
        self.agent_kb_mapping: Dict[str, str] = {}  # agent_id -> primary_kb_id (one-to-one)

        # Simple stats
        self.stats = {
            "total_knowledge_bases": 0,
            "total_documents": 0,
            "total_queries": 0,
            "active_agents": 0
        }

        logger.info(
            "THE Collection-based KB manager initialized",
            LogCategory.MEMORY_OPERATIONS,
            "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager"
        )

    async def initialize(self) -> None:
        """Initialize THE knowledge base manager."""
        try:
            if self.is_initialized:
                return

            # Ensure unified RAG is initialized
            if not self.unified_rag.is_initialized:
                await self.unified_rag.initialize()

            self.is_initialized = True
            logger.info(
                "THE KB manager initialization completed",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager"
            )

        except Exception as e:
            logger.error(
                "Failed to initialize THE KB manager",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                error=e
            )
            raise

    async def create_knowledge_base(
        self,
        name: str,
        description: str,
        owner_agent_id: str,
        access_level: AccessLevel = AccessLevel.PRIVATE
    ) -> str:
        """
        Create a knowledge base for an agent - SIMPLIFIED ONE-TO-ONE MAPPING.

        Args:
            name: Knowledge base name
            description: Description of the knowledge base
            owner_agent_id: Agent that owns this knowledge base
            access_level: Access level for the knowledge base

        Returns:
            Knowledge base ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Check if agent already has a knowledge base (one-to-one mapping)
            if owner_agent_id in self.agent_kb_mapping:
                existing_kb_id = self.agent_kb_mapping[owner_agent_id]
                logger.warn(
                    f"Agent {owner_agent_id} already has knowledge base: {existing_kb_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                    data={"owner_agent_id": owner_agent_id, "existing_kb_id": existing_kb_id}
                )
                return existing_kb_id

            # Generate simple KB ID
            kb_id = f"kb_{owner_agent_id}"

            # Ensure agent ecosystem exists
            agent_collections = await self.unified_rag.get_agent_collections(owner_agent_id)
            if not agent_collections:
                agent_collections = await self.unified_rag.create_agent_ecosystem(owner_agent_id)

            # Use agent's knowledge collection
            collection_name = agent_collections.knowledge_collection

            # Create knowledge base info
            kb_info = KnowledgeBaseInfo(
                kb_id=kb_id,
                name=name,
                description=description,
                owner_agent_id=owner_agent_id,
                access_level=access_level,
                collection_name=collection_name,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                document_count=0
            )

            # Register knowledge base (one-to-one mapping)
            self.knowledge_bases[kb_id] = kb_info
            self.agent_kb_mapping[owner_agent_id] = kb_id

            # Update stats
            self.stats["total_knowledge_bases"] += 1
            self.stats["active_agents"] += 1

            logger.info(
                f"Created THE knowledge base: {kb_id} for agent {owner_agent_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                data={"kb_id": kb_id, "owner_agent_id": owner_agent_id, "access_level": access_level}
            )
            return kb_id

        except Exception as e:
            logger.error(
                "Failed to create knowledge base",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                error=e
            )
            raise

    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseInfo]:
        """Get knowledge base information."""
        return self.knowledge_bases.get(kb_id)

    async def list_agent_knowledge_bases(self, agent_id: str) -> List[KnowledgeBaseInfo]:
        """List all knowledge bases for an agent."""
        kb_ids = self.agent_kb_mapping.get(agent_id, [])
        return [self.knowledge_bases[kb_id] for kb_id in kb_ids if kb_id in self.knowledge_bases]

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base."""
        try:
            if kb_id not in self.knowledge_bases:
                return False

            kb_info = self.knowledge_bases[kb_id]

            # Remove from agent mapping
            if kb_info.owner_agent_id in self.agent_kb_mapping:
                self.agent_kb_mapping[kb_info.owner_agent_id].remove(kb_id)

            # Remove from registry
            del self.knowledge_bases[kb_id]

            # Update stats
            self.stats["total_knowledge_bases"] -= 1

            logger.info(
                f"Deleted knowledge base: {kb_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                data={"kb_id": kb_id}
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to delete knowledge base {kb_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                error=e,
                data={"kb_id": kb_id}
            )
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "active_knowledge_bases": len(self.knowledge_bases)
        }



    async def add_document_to_kb(
        self,
        kb_id: str,
        document: Document,
        agent_id: str
    ) -> str:
        """
        Add a document to a knowledge base.

        Args:
            kb_id: Knowledge base ID
            document: Document to add
            agent_id: Agent adding the document

        Returns:
            Document ID
        """
        try:
            # Check if knowledge base exists
            kb_info = await self.get_knowledge_base(kb_id)
            if not kb_info:
                raise ValueError(f"Knowledge base {kb_id} not found")

            # Simple permission check - only owner can add
            if agent_id != kb_info.owner_agent_id:
                raise PermissionError(f"Agent {agent_id} does not have write access to {kb_id}")

            # Add document metadata
            document.metadata.update({
                "kb_id": kb_id,
                "added_by": agent_id,
                "added_at": datetime.utcnow().isoformat()
            })

            # Add document using unified RAG system
            await self.unified_rag.add_documents(
                agent_id=kb_info.owner_agent_id,
                documents=[document],
                collection_type="knowledge"
            )

            # Update knowledge base stats
            kb_info.document_count += 1
            kb_info.last_updated = datetime.utcnow()

            self.stats["total_documents"] += 1

            logger.info(
                f"Added document to knowledge base {kb_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                data={"kb_id": kb_id, "document_id": document.id}
            )
            return document.id

        except Exception as e:
            logger.error(
                f"Failed to add document to KB {kb_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                error=e,
                data={"kb_id": kb_id}
            )
            raise
    
    async def search_knowledge_base(
        self,
        kb_id: str,
        query: str,
        agent_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search a specific knowledge base.

        Args:
            kb_id: Knowledge base ID
            query: Search query
            agent_id: Agent performing the search
            top_k: Number of results to return
            filters: Additional filters

        Returns:
            List of matching documents
        """
        try:
            # Check if knowledge base exists
            kb_info = await self.get_knowledge_base(kb_id)
            if not kb_info:
                raise ValueError(f"Knowledge base {kb_id} not found")

            # Simple permission check - owner or public access
            if agent_id != kb_info.owner_agent_id and kb_info.access_level != AccessLevel.PUBLIC:
                raise PermissionError(f"Agent {agent_id} does not have access to {kb_id}")

            # Perform search using unified RAG system
            results = await self.unified_rag.search_agent_knowledge(
                agent_id=kb_info.owner_agent_id,
                query=query,
                top_k=top_k,
                filters=filters
            )

            # Update stats
            self.stats["total_queries"] += 1

            logger.debug(
                f"Searched knowledge base {kb_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                data={"kb_id": kb_id, "query": query, "top_k": top_k}
            )
            return results

        except Exception as e:
            logger.error(
                f"Failed to search KB {kb_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.rag.core.collection_based_kb_manager.CollectionBasedKBManager",
                error=e,
                data={"kb_id": kb_id}
            )
            raise


