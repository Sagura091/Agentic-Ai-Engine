"""
Collection-Based Knowledge Base Manager.

This module provides a knowledge base manager that uses collection-based isolation
within a single ChromaDB instance for optimal resource utilization and performance.

Features:
- Collection-based knowledge base isolation
- Agent-specific knowledge management
- Shared knowledge collections
- Access control and permissions
- Efficient resource utilization
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .unified_rag_system import UnifiedRAGSystem, AgentCollections
from .unified_rag_system import Document, KnowledgeQuery, KnowledgeResult
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class AccessLevel(str, Enum):
    """Access levels for knowledge bases."""
    PRIVATE = "private"           # Agent-only access
    SHARED_TEAM = "shared_team"   # Team-level access
    SHARED_DOMAIN = "shared_domain"  # Domain-level access
    PUBLIC = "public"             # Global access


@dataclass
class KnowledgeBaseInfo:
    """Information about a knowledge base."""
    kb_id: str
    name: str
    description: str
    owner_agent_id: str
    access_level: AccessLevel
    collection_name: str
    created_at: datetime
    last_updated: datetime
    document_count: int
    size_bytes: int
    tags: List[str]


class CollectionBasedKBManager:
    """
    Collection-Based Knowledge Base Manager.
    
    Manages multiple knowledge bases using collection-based isolation
    within a single ChromaDB instance for optimal performance.
    """
    
    def __init__(self, unified_rag: UnifiedRAGSystem):
        """Initialize the collection-based KB manager."""
        self.unified_rag = unified_rag
        
        # Knowledge base registry
        self.knowledge_bases: Dict[str, KnowledgeBaseInfo] = {}
        self.agent_kb_mapping: Dict[str, List[str]] = {}  # agent_id -> kb_ids
        
        # Access control
        self.access_permissions: Dict[str, Set[str]] = {}  # kb_id -> agent_ids
        
        # Performance tracking
        self.stats = {
            "total_knowledge_bases": 0,
            "total_documents": 0,
            "total_queries": 0,
            "avg_query_time": 0.0
        }
        
        logger.info("Collection-based KB manager initialized")
    
    async def create_knowledge_base(
        self,
        name: str,
        description: str,
        owner_agent_id: str,
        access_level: AccessLevel = AccessLevel.PRIVATE,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new knowledge base for an agent.
        
        Args:
            name: Knowledge base name
            description: Description of the knowledge base
            owner_agent_id: Agent that owns this knowledge base
            access_level: Access level for the knowledge base
            tags: Optional tags for categorization
            
        Returns:
            Knowledge base ID
        """
        try:
            # Generate unique KB ID
            kb_id = f"kb_{name.lower().replace(' ', '_').replace('-', '_')}_{owner_agent_id}"
            
            if kb_id in self.knowledge_bases:
                raise ValueError(f"Knowledge base '{name}' already exists for agent {owner_agent_id}")
            
            # Ensure agent ecosystem exists
            agent_collections = await self.unified_rag.get_agent_collections(owner_agent_id)
            if not agent_collections:
                agent_collections = await self.unified_rag.create_agent_ecosystem(owner_agent_id)
            
            # For private knowledge bases, use the agent's knowledge collection
            if access_level == AccessLevel.PRIVATE:
                collection_name = agent_collections.knowledge_collection
            else:
                # For shared knowledge bases, create a dedicated collection
                collection_name = f"shared_{kb_id}"
                # Create the shared collection
                await self.unified_rag._create_collection(
                    collection_name, 
                    self.unified_rag.CollectionType.SHARED_DOMAIN
                )
            
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
                document_count=0,
                size_bytes=0,
                tags=tags or []
            )
            
            # Register knowledge base
            self.knowledge_bases[kb_id] = kb_info
            
            # Update agent mapping
            if owner_agent_id not in self.agent_kb_mapping:
                self.agent_kb_mapping[owner_agent_id] = []
            self.agent_kb_mapping[owner_agent_id].append(kb_id)
            
            # Set initial permissions
            self.access_permissions[kb_id] = {owner_agent_id}
            
            self.stats["total_knowledge_bases"] += 1
            
            logger.info(f"Created knowledge base: {kb_id} for agent {owner_agent_id}")
            return kb_id
            
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {str(e)}")
            raise
    
    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseInfo]:
        """Get knowledge base information."""
        return self.knowledge_bases.get(kb_id)
    
    async def list_agent_knowledge_bases(self, agent_id: str) -> List[KnowledgeBaseInfo]:
        """List all knowledge bases accessible to an agent."""
        try:
            accessible_kbs = []
            
            # Get agent's own knowledge bases
            agent_kb_ids = self.agent_kb_mapping.get(agent_id, [])
            for kb_id in agent_kb_ids:
                if kb_id in self.knowledge_bases:
                    accessible_kbs.append(self.knowledge_bases[kb_id])
            
            # Get shared knowledge bases the agent has access to
            for kb_id, kb_info in self.knowledge_bases.items():
                if (kb_info.access_level != AccessLevel.PRIVATE and 
                    agent_id in self.access_permissions.get(kb_id, set())):
                    accessible_kbs.append(kb_info)
            
            return accessible_kbs
            
        except Exception as e:
            logger.error(f"Failed to list knowledge bases for agent {agent_id}: {str(e)}")
            raise
    
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
            
            # Check permissions
            if not await self._check_write_permission(kb_id, agent_id):
                raise PermissionError(f"Agent {agent_id} does not have write access to {kb_id}")
            
            # Add document metadata
            document.metadata.update({
                "kb_id": kb_id,
                "added_by": agent_id,
                "added_at": datetime.utcnow().isoformat()
            })
            
            # Add document using unified RAG system
            document_id = await self.unified_rag.add_document_to_agent(
                agent_id=kb_info.owner_agent_id,
                document=document,
                collection_type="knowledge"
            )
            
            # Update knowledge base stats
            kb_info.document_count += 1
            kb_info.last_updated = datetime.utcnow()
            kb_info.size_bytes += len(document.content.encode('utf-8'))
            
            self.stats["total_documents"] += 1
            
            logger.info(f"Added document to knowledge base {kb_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to add document to KB {kb_id}: {str(e)}")
            raise
    
    async def search_knowledge_base(
        self,
        kb_id: str,
        query: str,
        agent_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> KnowledgeResult:
        """
        Search a specific knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            query: Search query
            agent_id: Agent performing the search
            top_k: Number of results to return
            filters: Additional filters
            
        Returns:
            Search results
        """
        try:
            # Check if knowledge base exists
            kb_info = await self.get_knowledge_base(kb_id)
            if not kb_info:
                raise ValueError(f"Knowledge base {kb_id} not found")
            
            # Check permissions
            if not await self._check_read_permission(kb_id, agent_id):
                raise PermissionError(f"Agent {agent_id} does not have read access to {kb_id}")
            
            # Perform search using unified RAG system
            result = await self.unified_rag.search_agent_knowledge(
                agent_id=kb_info.owner_agent_id,
                query=query,
                top_k=top_k,
                include_shared=False,  # Search only this specific KB
                filters=filters
            )
            
            # Update metadata
            result.metadata.update({
                "kb_id": kb_id,
                "kb_name": kb_info.name,
                "searched_by": agent_id
            })
            
            self.stats["total_queries"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to search KB {kb_id}: {str(e)}")
            raise
    
    async def _check_read_permission(self, kb_id: str, agent_id: str) -> bool:
        """Check if agent has read permission for knowledge base."""
        return agent_id in self.access_permissions.get(kb_id, set())
    
    async def _check_write_permission(self, kb_id: str, agent_id: str) -> bool:
        """Check if agent has write permission for knowledge base."""
        kb_info = self.knowledge_bases.get(kb_id)
        if not kb_info:
            return False
        
        # Owner always has write permission
        if agent_id == kb_info.owner_agent_id:
            return True
        
        # Check if agent has explicit write permission
        return agent_id in self.access_permissions.get(kb_id, set())
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get knowledge base manager statistics."""
        return {
            **self.stats,
            "knowledge_bases_count": len(self.knowledge_bases),
            "agents_with_kbs": len(self.agent_kb_mapping)
        }
