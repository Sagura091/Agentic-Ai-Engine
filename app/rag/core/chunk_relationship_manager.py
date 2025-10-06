"""
Chunk Relationship Manager.

This module manages relationships between chunks and document structure.
It enables context expansion, parent document retrieval, and hierarchical
navigation through document chunks.

Key Features:
- Parent-child relationships
- Sibling chunk tracking
- Document hierarchy
- Context expansion (retrieve surrounding chunks)
- Graph-based relationship storage
- Efficient relationship queries

Author: Agentic AI System
Purpose: Manage chunk relationships and document structure
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


class RelationType(str, Enum):
    """Types of relationships between chunks."""
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    NEXT = "next"
    PREVIOUS = "previous"
    REFERENCE = "reference"
    CONTAINS = "contains"
    PART_OF = "part_of"


@dataclass
class Relationship:
    """A relationship between two chunks."""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChunkNode:
    """A node in the chunk graph."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    total_chunks: int
    section_path: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentHierarchy:
    """Hierarchical structure of a document."""
    doc_id: str
    root_chunk_id: Optional[str] = None
    total_chunks: int = 0
    chunk_nodes: Dict[str, ChunkNode] = field(default_factory=dict)
    hierarchy_tree: Dict[str, List[str]] = field(default_factory=dict)  # parent_id -> [child_ids]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkRelationshipManager:
    """
    Production-grade chunk relationship manager.
    
    Manages relationships between chunks using a graph-based approach.
    Enables efficient context expansion and hierarchical navigation.
    
    Features:
    - Bidirectional relationship tracking
    - Graph traversal
    - Context expansion
    - Parent document retrieval
    - Sibling navigation
    - Relationship queries
    """
    
    def __init__(self):
        """Initialize chunk relationship manager."""
        # Relationship graph (adjacency list)
        self._outgoing_edges: Dict[str, List[Relationship]] = defaultdict(list)
        self._incoming_edges: Dict[str, List[Relationship]] = defaultdict(list)
        
        # Chunk nodes
        self._chunks: Dict[str, ChunkNode] = {}
        
        # Document hierarchies
        self._documents: Dict[str, DocumentHierarchy] = {}
        
        # Document to chunks mapping
        self._doc_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        
        # Metrics
        self._metrics = {
            'total_chunks': 0,
            'total_documents': 0,
            'total_relationships': 0,
            'context_expansions': 0,
            'parent_retrievals': 0
        }
        
        logger.info("ChunkRelationshipManager initialized")
    
    def add_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        total_chunks: int,
        section_path: Optional[str] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a chunk node.
        
        Args:
            chunk_id: Unique chunk identifier
            doc_id: Parent document ID
            chunk_index: Index of chunk in document
            total_chunks: Total number of chunks in document
            section_path: Section path (e.g., "Chapter 1/Section 1.1")
            page_number: Page number
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            node = ChunkNode(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                section_path=section_path,
                page_number=page_number,
                metadata=metadata or {}
            )
            
            self._chunks[chunk_id] = node
            self._doc_to_chunks[doc_id].add(chunk_id)
            
            # Initialize document hierarchy if needed
            if doc_id not in self._documents:
                self._documents[doc_id] = DocumentHierarchy(
                    doc_id=doc_id,
                    total_chunks=total_chunks
                )
                self._metrics['total_documents'] += 1
            
            # Add to document hierarchy
            self._documents[doc_id].chunk_nodes[chunk_id] = node
            
            self._metrics['total_chunks'] += 1
            
            logger.debug(f"Chunk added: {chunk_id} (doc: {doc_id}, index: {chunk_index})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunk {chunk_id}: {e}")
            return False
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a relationship between chunks.
        
        Args:
            source_id: Source chunk ID
            target_id: Target chunk ID
            relation_type: Type of relationship
            weight: Relationship weight (0.0-1.0)
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            relationship = Relationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight,
                metadata=metadata or {}
            )
            
            # Add to outgoing edges
            self._outgoing_edges[source_id].append(relationship)
            
            # Add to incoming edges
            self._incoming_edges[target_id].append(relationship)
            
            self._metrics['total_relationships'] += 1
            
            logger.debug(
                f"Relationship added",
                source=source_id,
                target=target_id,
                type=relation_type.value
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            return False
    
    def build_document_structure(self, doc_id: str) -> bool:
        """
        Build hierarchical structure for a document.
        
        Automatically creates NEXT/PREVIOUS relationships between sequential chunks
        and PARENT/CHILD relationships based on section paths.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            if doc_id not in self._documents:
                logger.error(f"Document not found: {doc_id}")
                return False
            
            doc = self._documents[doc_id]
            chunk_ids = sorted(
                self._doc_to_chunks[doc_id],
                key=lambda cid: self._chunks[cid].chunk_index
            )
            
            # Create sequential relationships
            for i in range(len(chunk_ids)):
                current_id = chunk_ids[i]
                
                # NEXT relationship
                if i < len(chunk_ids) - 1:
                    next_id = chunk_ids[i + 1]
                    self.add_relationship(
                        current_id,
                        next_id,
                        RelationType.NEXT
                    )
                
                # PREVIOUS relationship
                if i > 0:
                    prev_id = chunk_ids[i - 1]
                    self.add_relationship(
                        current_id,
                        prev_id,
                        RelationType.PREVIOUS
                    )
                
                # SIBLING relationships (same section level)
                current_node = self._chunks[current_id]
                for j in range(len(chunk_ids)):
                    if i != j:
                        other_id = chunk_ids[j]
                        other_node = self._chunks[other_id]
                        
                        # Check if same section level
                        if (current_node.section_path and other_node.section_path and
                            self._same_section_level(current_node.section_path, other_node.section_path)):
                            self.add_relationship(
                                current_id,
                                other_id,
                                RelationType.SIBLING,
                                weight=0.5
                            )
            
            # Build hierarchy tree based on section paths
            for chunk_id in chunk_ids:
                node = self._chunks[chunk_id]
                if node.section_path:
                    parent_path = self._get_parent_section(node.section_path)
                    if parent_path:
                        # Find parent chunk
                        for other_id in chunk_ids:
                            other_node = self._chunks[other_id]
                            if other_node.section_path == parent_path:
                                # Add parent-child relationship
                                self.add_relationship(
                                    other_id,
                                    chunk_id,
                                    RelationType.CHILD
                                )
                                self.add_relationship(
                                    chunk_id,
                                    other_id,
                                    RelationType.PARENT
                                )
                                
                                # Update hierarchy tree
                                if other_id not in doc.hierarchy_tree:
                                    doc.hierarchy_tree[other_id] = []
                                doc.hierarchy_tree[other_id].append(chunk_id)
                                break
            
            # Set root chunk (first chunk or chunk with no parent)
            if chunk_ids:
                root_candidates = [
                    cid for cid in chunk_ids
                    if not any(
                        r.relation_type == RelationType.PARENT
                        for r in self._outgoing_edges.get(cid, [])
                    )
                ]
                doc.root_chunk_id = root_candidates[0] if root_candidates else chunk_ids[0]
            
            logger.info(f"Document structure built: {doc_id} ({len(chunk_ids)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build document structure for {doc_id}: {e}")
            return False

    def get_surrounding_chunks(
        self,
        chunk_id: str,
        context_size: int = 2,
        include_siblings: bool = False
    ) -> List[str]:
        """
        Get surrounding chunks for context expansion.

        Args:
            chunk_id: Chunk ID
            context_size: Number of chunks before and after
            include_siblings: Include sibling chunks

        Returns:
            List of chunk IDs (including the original)
        """
        try:
            if chunk_id not in self._chunks:
                logger.warning(f"Chunk not found: {chunk_id}")
                return [chunk_id]

            node = self._chunks[chunk_id]
            doc_id = node.doc_id

            # Get all chunks in document sorted by index
            doc_chunks = sorted(
                [self._chunks[cid] for cid in self._doc_to_chunks[doc_id]],
                key=lambda n: n.chunk_index
            )

            # Find current chunk index
            current_idx = next(
                (i for i, n in enumerate(doc_chunks) if n.chunk_id == chunk_id),
                None
            )

            if current_idx is None:
                return [chunk_id]

            # Get surrounding chunks
            start_idx = max(0, current_idx - context_size)
            end_idx = min(len(doc_chunks), current_idx + context_size + 1)

            result = [doc_chunks[i].chunk_id for i in range(start_idx, end_idx)]

            # Add siblings if requested
            if include_siblings:
                sibling_rels = self.get_relationships(
                    chunk_id,
                    relation_type=RelationType.SIBLING
                )
                for rel in sibling_rels:
                    if rel.target_id not in result:
                        result.append(rel.target_id)

            self._metrics['context_expansions'] += 1

            logger.debug(
                f"Context expansion",
                chunk_id=chunk_id,
                context_size=context_size,
                result_count=len(result)
            )

            return result

        except Exception as e:
            logger.error(f"Context expansion failed for {chunk_id}: {e}")
            return [chunk_id]

    def get_parent_document(self, chunk_id: str) -> Optional[str]:
        """
        Get parent document ID for a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            Parent document ID if exists
        """
        if chunk_id in self._chunks:
            self._metrics['parent_retrievals'] += 1
            return self._chunks[chunk_id].doc_id
        return None

    def get_parent_chunk(self, chunk_id: str) -> Optional[str]:
        """
        Get parent chunk ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Parent chunk ID if exists
        """
        parent_rels = self.get_relationships(
            chunk_id,
            relation_type=RelationType.PARENT
        )

        if parent_rels:
            return parent_rels[0].target_id
        return None

    def get_child_chunks(self, chunk_id: str) -> List[str]:
        """
        Get child chunk IDs.

        Args:
            chunk_id: Chunk ID

        Returns:
            List of child chunk IDs
        """
        child_rels = self.get_relationships(
            chunk_id,
            relation_type=RelationType.CHILD
        )

        return [rel.target_id for rel in child_rels]

    def get_relationships(
        self,
        chunk_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "outgoing"
    ) -> List[Relationship]:
        """
        Get relationships for a chunk.

        Args:
            chunk_id: Chunk ID
            relation_type: Optional filter by relation type
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationships
        """
        relationships = []

        if direction in ["outgoing", "both"]:
            relationships.extend(self._outgoing_edges.get(chunk_id, []))

        if direction in ["incoming", "both"]:
            relationships.extend(self._incoming_edges.get(chunk_id, []))

        if relation_type:
            relationships = [
                r for r in relationships
                if r.relation_type == relation_type
            ]

        return relationships

    def get_document_hierarchy(self, doc_id: str) -> Optional[DocumentHierarchy]:
        """
        Get document hierarchy.

        Args:
            doc_id: Document ID

        Returns:
            Document hierarchy if exists
        """
        return self._documents.get(doc_id)

    def traverse_hierarchy(
        self,
        start_chunk_id: str,
        max_depth: int = 3,
        relation_types: Optional[List[RelationType]] = None
    ) -> List[str]:
        """
        Traverse chunk hierarchy using BFS.

        Args:
            start_chunk_id: Starting chunk ID
            max_depth: Maximum traversal depth
            relation_types: Optional filter by relation types

        Returns:
            List of chunk IDs in traversal order
        """
        if start_chunk_id not in self._chunks:
            return []

        visited = set()
        queue = [(start_chunk_id, 0)]  # (chunk_id, depth)
        result = []

        while queue:
            chunk_id, depth = queue.pop(0)

            if chunk_id in visited or depth > max_depth:
                continue

            visited.add(chunk_id)
            result.append(chunk_id)

            # Get outgoing relationships
            relationships = self._outgoing_edges.get(chunk_id, [])

            if relation_types:
                relationships = [
                    r for r in relationships
                    if r.relation_type in relation_types
                ]

            # Add neighbors to queue
            for rel in relationships:
                if rel.target_id not in visited:
                    queue.append((rel.target_id, depth + 1))

        return result

    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk and its relationships.

        Args:
            chunk_id: Chunk ID

        Returns:
            True if successful
        """
        try:
            if chunk_id not in self._chunks:
                return False

            node = self._chunks[chunk_id]
            doc_id = node.doc_id

            # Remove from chunks
            del self._chunks[chunk_id]

            # Remove from document mapping
            self._doc_to_chunks[doc_id].discard(chunk_id)

            # Remove from document hierarchy
            if doc_id in self._documents:
                doc = self._documents[doc_id]
                if chunk_id in doc.chunk_nodes:
                    del doc.chunk_nodes[chunk_id]

                # Remove from hierarchy tree
                for parent_id, children in doc.hierarchy_tree.items():
                    if chunk_id in children:
                        children.remove(chunk_id)

                if chunk_id in doc.hierarchy_tree:
                    del doc.hierarchy_tree[chunk_id]

            # Remove outgoing relationships
            if chunk_id in self._outgoing_edges:
                for rel in self._outgoing_edges[chunk_id]:
                    # Remove from incoming edges of target
                    if rel.target_id in self._incoming_edges:
                        self._incoming_edges[rel.target_id] = [
                            r for r in self._incoming_edges[rel.target_id]
                            if r.source_id != chunk_id
                        ]
                del self._outgoing_edges[chunk_id]

            # Remove incoming relationships
            if chunk_id in self._incoming_edges:
                for rel in self._incoming_edges[chunk_id]:
                    # Remove from outgoing edges of source
                    if rel.source_id in self._outgoing_edges:
                        self._outgoing_edges[rel.source_id] = [
                            r for r in self._outgoing_edges[rel.source_id]
                            if r.target_id != chunk_id
                        ]
                del self._incoming_edges[chunk_id]

            self._metrics['total_chunks'] -= 1

            logger.debug(f"Chunk removed: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove chunk {chunk_id}: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get relationship manager metrics.

        Returns:
            Metrics dictionary
        """
        return {
            **self._metrics,
            'avg_relationships_per_chunk': (
                self._metrics['total_relationships'] / self._metrics['total_chunks']
                if self._metrics['total_chunks'] > 0
                else 0.0
            )
        }

    def _same_section_level(self, path1: str, path2: str) -> bool:
        """
        Check if two section paths are at the same level.

        Args:
            path1: First section path
            path2: Second section path

        Returns:
            True if same level
        """
        parts1 = path1.split('/')
        parts2 = path2.split('/')

        # Same level if same depth and same parent
        if len(parts1) != len(parts2):
            return False

        if len(parts1) > 1:
            return '/'.join(parts1[:-1]) == '/'.join(parts2[:-1])

        return True

    def _get_parent_section(self, section_path: str) -> Optional[str]:
        """
        Get parent section path.

        Args:
            section_path: Section path

        Returns:
            Parent section path if exists
        """
        parts = section_path.split('/')
        if len(parts) > 1:
            return '/'.join(parts[:-1])
        return None


# Global singleton
_chunk_relationship_manager: Optional[ChunkRelationshipManager] = None
_manager_lock = asyncio.Lock()


async def get_chunk_relationship_manager() -> ChunkRelationshipManager:
    """
    Get or create chunk relationship manager singleton.

    Returns:
        ChunkRelationshipManager instance
    """
    global _chunk_relationship_manager

    async with _manager_lock:
        if _chunk_relationship_manager is None:
            _chunk_relationship_manager = ChunkRelationshipManager()
            logger.info("ChunkRelationshipManager singleton created")

        return _chunk_relationship_manager
