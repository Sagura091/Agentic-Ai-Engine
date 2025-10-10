"""
Revolutionary Dynamic Knowledge Graph System for Agentic AI Memory.

Implements incremental knowledge graph construction with spatial relationship tracking,
based on state-of-the-art research in dynamic graph neural networks and spatial reasoning.

Key Features:
- Incremental graph updates with memory associations
- Spatial relationship modeling and tracking
- Entity extraction and relationship inference
- Graph-based memory consolidation
- Cross-modal knowledge representation
- Temporal graph evolution tracking
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import numpy as np

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    # Semantic relationships
    IS_A = "is_a"                    # Entity classification
    PART_OF = "part_of"              # Compositional relationships
    RELATED_TO = "related_to"        # General semantic relation
    SIMILAR_TO = "similar_to"        # Similarity relationships
    
    # Temporal relationships
    BEFORE = "before"                # Temporal precedence
    AFTER = "after"                  # Temporal succession
    DURING = "during"                # Temporal containment
    CONCURRENT = "concurrent"        # Simultaneous events
    
    # Spatial relationships
    NEAR = "near"                    # Spatial proximity
    INSIDE = "inside"                # Spatial containment
    ADJACENT = "adjacent"            # Spatial adjacency
    ABOVE = "above"                  # Vertical spatial relation
    BELOW = "below"                  # Vertical spatial relation
    
    # Causal relationships
    CAUSES = "causes"                # Causal relationships
    ENABLES = "enables"              # Enabling relationships
    PREVENTS = "prevents"            # Prevention relationships
    
    # Memory-specific relationships
    TRIGGERS = "triggers"            # Memory activation
    REINFORCES = "reinforces"        # Memory strengthening
    CONFLICTS = "conflicts"          # Contradictory information


@dataclass
class GraphEntity:
    """Entity in the dynamic knowledge graph."""
    entity_id: str
    entity_type: str  # person, place, concept, event, object, etc.
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Spatial properties
    spatial_coordinates: Optional[Tuple[float, float, float]] = None  # x, y, z
    spatial_region: Optional[str] = None
    spatial_scale: str = "unknown"  # micro, local, regional, global
    
    # Temporal properties
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    temporal_span: Optional[Tuple[datetime, datetime]] = None
    
    # Graph properties
    centrality_score: float = 0.0
    importance_score: float = 0.5
    access_frequency: int = 0
    
    # Memory associations
    associated_memories: Set[str] = field(default_factory=set)
    activation_strength: float = 0.0


@dataclass
class GraphRelationship:
    """Relationship between entities in the knowledge graph."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    strength: float = 1.0  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    
    # Contextual information
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)  # Memory IDs supporting this relationship
    
    # Temporal properties
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.01  # How quickly the relationship weakens
    
    # Spatial properties (for spatial relationships)
    spatial_distance: Optional[float] = None
    spatial_direction: Optional[str] = None
    
    def update_strength(self, reinforcement: float = 0.1):
        """Update relationship strength with reinforcement."""
        self.strength = min(1.0, self.strength + reinforcement)
        self.last_reinforced = datetime.now()
    
    def apply_decay(self):
        """Apply temporal decay to relationship strength."""
        time_since_reinforcement = (datetime.now() - self.last_reinforced).total_seconds() / 86400  # days
        decay_factor = np.exp(-self.decay_rate * time_since_reinforcement)
        self.strength *= decay_factor


@dataclass
class GraphCluster:
    """Cluster of related entities in the knowledge graph."""
    cluster_id: str
    cluster_type: str  # semantic, spatial, temporal, functional
    entities: Set[str] = field(default_factory=set)
    centroid_entity: Optional[str] = None
    coherence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class DynamicKnowledgeGraph:
    """
    Revolutionary Dynamic Knowledge Graph System.
    
    Builds and maintains an evolving knowledge graph from agent memories
    with spatial relationship tracking and incremental updates.
    """
    
    def __init__(self, agent_id: str, embedding_function: Optional[callable] = None):
        """Initialize the dynamic knowledge graph."""
        self.agent_id = agent_id
        self.embedding_function = embedding_function
        
        # Graph storage
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        self.clusters: Dict[str, GraphCluster] = {}
        
        # Indices for fast lookup
        self.entity_type_index: Dict[str, Set[str]] = defaultdict(set)
        self.relationship_type_index: Dict[RelationshipType, Set[str]] = defaultdict(set)
        self.spatial_index: Dict[str, Set[str]] = defaultdict(set)  # region -> entity_ids
        self.temporal_index: Dict[str, Set[str]] = defaultdict(set)  # date -> entity_ids
        
        # Graph statistics
        self.stats = {
            "total_entities": 0,
            "total_relationships": 0,
            "total_clusters": 0,
            "graph_updates": 0,
            "consolidation_cycles": 0,
            "avg_entity_connections": 0.0,
            "graph_density": 0.0
        }
        
        # Configuration
        self.config = {
            "max_entities": 10000,
            "max_relationships": 50000,
            "entity_similarity_threshold": 0.8,
            "relationship_strength_threshold": 0.1,
            "consolidation_interval_hours": 24,
            "spatial_proximity_threshold": 100.0,  # meters
            "enable_automatic_clustering": True,
            "enable_spatial_reasoning": True,
            "enable_temporal_reasoning": True
        }

        logger.info(
            f"Dynamic Knowledge Graph initialized for agent {agent_id}",
            LogCategory.MEMORY_OPERATIONS,
            "app.memory.dynamic_knowledge_graph.DynamicKnowledgeGraph"
        )
    
    async def add_entity_from_memory(
        self,
        memory_id: str,
        memory_content: str,
        memory_metadata: Dict[str, Any]
    ) -> List[str]:
        """Extract and add entities from memory content."""
        try:
            # Extract entities from memory content
            extracted_entities = await self._extract_entities(memory_content, memory_metadata)
            
            entity_ids = []
            for entity_data in extracted_entities:
                entity_id = await self._add_or_update_entity(entity_data, memory_id)
                entity_ids.append(entity_id)
            
            # Extract and add relationships between entities
            if len(entity_ids) > 1:
                relationships = await self._extract_relationships(
                    entity_ids, memory_content, memory_metadata
                )
                for rel_data in relationships:
                    await self._add_or_update_relationship(rel_data, memory_id)
            
            # Update graph statistics
            self._update_graph_stats()

            logger.debug(
                "Entities added from memory",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.dynamic_knowledge_graph.DynamicKnowledgeGraph",
                data={
                    "agent_id": self.agent_id,
                    "memory_id": memory_id,
                    "entities_count": len(entity_ids)
                }
            )

            return entity_ids

        except Exception as e:
            logger.error(
                "Failed to add entities from memory",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.dynamic_knowledge_graph.DynamicKnowledgeGraph",
                error=e
            )
            return []
    
    async def _extract_entities(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract entities from content using NLP techniques."""
        entities = []
        
        # Simple entity extraction (in production, use NER models)
        words = content.split()
        
        # Extract potential entities (capitalized words, proper nouns)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                # Determine entity type based on context
                entity_type = self._classify_entity_type(word, words, i)
                
                # Extract spatial information if available
                spatial_info = self._extract_spatial_info(words, i)
                
                entities.append({
                    "name": word,
                    "type": entity_type,
                    "description": f"Entity extracted from: {content[:100]}...",
                    "spatial_info": spatial_info,
                    "context_words": words[max(0, i-3):i+4]
                })
        
        return entities
    
    def _classify_entity_type(self, word: str, words: List[str], position: int) -> str:
        """Classify entity type based on context."""
        # Simple classification rules (in production, use ML models)
        context = " ".join(words[max(0, position-2):position+3]).lower()
        
        if any(indicator in context for indicator in ["person", "people", "human", "individual"]):
            return "person"
        elif any(indicator in context for indicator in ["place", "location", "city", "country"]):
            return "place"
        elif any(indicator in context for indicator in ["event", "happened", "occurred"]):
            return "event"
        elif any(indicator in context for indicator in ["concept", "idea", "theory"]):
            return "concept"
        else:
            return "object"
    
    def _extract_spatial_info(self, words: List[str], position: int) -> Dict[str, Any]:
        """Extract spatial information from context."""
        spatial_info = {}
        
        # Look for spatial indicators in context
        context_window = words[max(0, position-5):position+6]
        context_text = " ".join(context_window).lower()
        
        # Spatial prepositions and indicators
        spatial_indicators = {
            "near": ["near", "close to", "next to", "beside"],
            "inside": ["in", "inside", "within"],
            "above": ["above", "over", "on top of"],
            "below": ["below", "under", "beneath"],
            "adjacent": ["adjacent", "neighboring", "bordering"]
        }
        
        for relation, indicators in spatial_indicators.items():
            if any(indicator in context_text for indicator in indicators):
                spatial_info["relation"] = relation
                break
        
        return spatial_info
    
    async def _add_or_update_entity(
        self,
        entity_data: Dict[str, Any],
        memory_id: str
    ) -> str:
        """Add or update an entity in the graph."""
        # Generate entity ID
        entity_id = f"entity_{hash(entity_data['name'])}_{self.agent_id}"
        
        if entity_id in self.entities:
            # Update existing entity
            entity = self.entities[entity_id]
            entity.last_updated = datetime.now()
            entity.access_frequency += 1
            entity.associated_memories.add(memory_id)
            
            # Update description if more informative
            if len(entity_data.get("description", "")) > len(entity.description):
                entity.description = entity_data["description"]
        else:
            # Create new entity
            entity = GraphEntity(
                entity_id=entity_id,
                entity_type=entity_data["type"],
                name=entity_data["name"],
                description=entity_data.get("description", ""),
                associated_memories={memory_id}
            )
            
            # Add spatial information if available
            spatial_info = entity_data.get("spatial_info", {})
            if spatial_info:
                entity.spatial_region = spatial_info.get("relation")
            
            self.entities[entity_id] = entity
            
            # Update indices
            self.entity_type_index[entity.entity_type].add(entity_id)
            if entity.spatial_region:
                self.spatial_index[entity.spatial_region].add(entity_id)
            
            self.stats["total_entities"] += 1
        
        return entity_id
    
    async def _extract_relationships(
        self,
        entity_ids: List[str],
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        # Generate relationships between all entity pairs
        for i, source_id in enumerate(entity_ids):
            for target_id in entity_ids[i+1:]:
                # Determine relationship type based on content analysis
                rel_type = await self._infer_relationship_type(
                    source_id, target_id, content, metadata
                )
                
                if rel_type:
                    relationships.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "type": rel_type,
                        "strength": 0.7,  # Default strength
                        "confidence": 0.6,  # Default confidence
                        "evidence": [content[:200]]  # Evidence snippet
                    })
        
        return relationships
    
    async def _infer_relationship_type(
        self,
        source_id: str,
        target_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> Optional[RelationshipType]:
        """Infer relationship type between entities."""
        source_entity = self.entities.get(source_id)
        target_entity = self.entities.get(target_id)
        
        if not source_entity or not target_entity:
            return None
        
        content_lower = content.lower()
        
        # Spatial relationship inference
        spatial_keywords = {
            RelationshipType.NEAR: ["near", "close", "next to", "beside"],
            RelationshipType.INSIDE: ["in", "inside", "within"],
            RelationshipType.ABOVE: ["above", "over", "on top"],
            RelationshipType.BELOW: ["below", "under", "beneath"]
        }
        
        for rel_type, keywords in spatial_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return rel_type
        
        # Temporal relationship inference
        temporal_keywords = {
            RelationshipType.BEFORE: ["before", "prior to", "earlier"],
            RelationshipType.AFTER: ["after", "following", "later"],
            RelationshipType.DURING: ["during", "while", "throughout"]
        }
        
        for rel_type, keywords in temporal_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return rel_type
        
        # Causal relationship inference
        causal_keywords = {
            RelationshipType.CAUSES: ["causes", "leads to", "results in"],
            RelationshipType.ENABLES: ["enables", "allows", "facilitates"],
            RelationshipType.PREVENTS: ["prevents", "stops", "blocks"]
        }
        
        for rel_type, keywords in causal_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return rel_type
        
        # Default to general relation
        return RelationshipType.RELATED_TO
    
    async def _add_or_update_relationship(
        self,
        rel_data: Dict[str, Any],
        memory_id: str
    ) -> str:
        """Add or update a relationship in the graph."""
        # Generate relationship ID
        rel_id = f"rel_{rel_data['source_id']}_{rel_data['target_id']}_{rel_data['type'].value}"
        
        if rel_id in self.relationships:
            # Update existing relationship
            relationship = self.relationships[rel_id]
            relationship.update_strength(0.1)  # Reinforce relationship
            relationship.evidence.append(memory_id)
        else:
            # Create new relationship
            relationship = GraphRelationship(
                relationship_id=rel_id,
                source_entity_id=rel_data["source_id"],
                target_entity_id=rel_data["target_id"],
                relationship_type=rel_data["type"],
                strength=rel_data["strength"],
                confidence=rel_data["confidence"],
                evidence=[memory_id]
            )
            
            self.relationships[rel_id] = relationship
            
            # Update indices
            self.relationship_type_index[relationship.relationship_type].add(rel_id)
            
            self.stats["total_relationships"] += 1
        
        return rel_id
    
    def _update_graph_stats(self):
        """Update graph statistics."""
        if self.stats["total_entities"] > 0:
            self.stats["avg_entity_connections"] = (
                self.stats["total_relationships"] * 2 / self.stats["total_entities"]
            )
            
            max_possible_edges = (
                self.stats["total_entities"] * (self.stats["total_entities"] - 1) / 2
            )
            if max_possible_edges > 0:
                self.stats["graph_density"] = (
                    self.stats["total_relationships"] / max_possible_edges
                )
        
        self.stats["graph_updates"] += 1

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_depth: int = 2,
        min_strength: float = 0.1
    ) -> Dict[str, Any]:
        """Get entities related to a given entity with traversal."""
        try:
            if entity_id not in self.entities:
                return {"entities": [], "relationships": [], "paths": []}

            related_entities = {}
            related_relationships = {}
            visited = set()

            # BFS traversal
            queue = [(entity_id, 0)]  # (entity_id, depth)
            visited.add(entity_id)

            while queue and len(related_entities) < 100:  # Limit results
                current_id, depth = queue.pop(0)

                if depth >= max_depth:
                    continue

                # Find relationships involving current entity
                for rel_id, relationship in self.relationships.items():
                    if relationship.strength < min_strength:
                        continue

                    if relationship_types and relationship.relationship_type not in relationship_types:
                        continue

                    target_id = None
                    if relationship.source_entity_id == current_id:
                        target_id = relationship.target_entity_id
                    elif relationship.target_entity_id == current_id:
                        target_id = relationship.source_entity_id

                    if target_id and target_id not in visited:
                        visited.add(target_id)
                        related_entities[target_id] = self.entities[target_id]
                        related_relationships[rel_id] = relationship

                        if depth + 1 < max_depth:
                            queue.append((target_id, depth + 1))

            return {
                "entities": list(related_entities.values()),
                "relationships": list(related_relationships.values()),
                "total_found": len(related_entities)
            }

        except Exception as e:
            logger.error(
                "Failed to get related entities",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.dynamic_knowledge_graph.DynamicKnowledgeGraph",
                error=e
            )
            return {"entities": [], "relationships": [], "total_found": 0}

    async def find_spatial_neighbors(
        self,
        entity_id: str,
        max_distance: float = None,
        spatial_relation: Optional[str] = None
    ) -> List[GraphEntity]:
        """Find spatially related entities."""
        try:
            if entity_id not in self.entities:
                return []

            source_entity = self.entities[entity_id]
            neighbors = []

            # If entity has spatial coordinates, use distance calculation
            if source_entity.spatial_coordinates:
                for other_id, other_entity in self.entities.items():
                    if other_id == entity_id or not other_entity.spatial_coordinates:
                        continue

                    # Calculate Euclidean distance
                    distance = np.sqrt(sum(
                        (a - b) ** 2 for a, b in zip(
                            source_entity.spatial_coordinates,
                            other_entity.spatial_coordinates
                        )
                    ))

                    if max_distance is None or distance <= max_distance:
                        neighbors.append(other_entity)

            # Use spatial region matching as fallback
            elif source_entity.spatial_region:
                region_entities = self.spatial_index.get(source_entity.spatial_region, set())
                for other_id in region_entities:
                    if other_id != entity_id:
                        neighbors.append(self.entities[other_id])

            return neighbors[:50]  # Limit results

        except Exception as e:
            logger.error(
                "Failed to find spatial neighbors",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.dynamic_knowledge_graph.DynamicKnowledgeGraph",
                error=e
            )
            return []

    async def consolidate_graph(self) -> Dict[str, Any]:
        """Perform graph consolidation and cleanup."""
        try:
            consolidation_stats = {
                "entities_merged": 0,
                "relationships_pruned": 0,
                "clusters_created": 0,
                "weak_relationships_removed": 0
            }

            # 1. Apply relationship decay
            relationships_to_remove = []
            for rel_id, relationship in self.relationships.items():
                relationship.apply_decay()
                if relationship.strength < self.config["relationship_strength_threshold"]:
                    relationships_to_remove.append(rel_id)

            # Remove weak relationships
            for rel_id in relationships_to_remove:
                del self.relationships[rel_id]
                consolidation_stats["weak_relationships_removed"] += 1

            # 2. Merge similar entities
            entities_to_merge = await self._find_similar_entities()
            for entity_group in entities_to_merge:
                if len(entity_group) > 1:
                    await self._merge_entities(entity_group)
                    consolidation_stats["entities_merged"] += len(entity_group) - 1

            # 3. Create clusters if enabled
            if self.config["enable_automatic_clustering"]:
                new_clusters = await self._create_entity_clusters()
                consolidation_stats["clusters_created"] = len(new_clusters)

            # 4. Update centrality scores
            await self._update_centrality_scores()

            # Update statistics
            self._update_graph_stats()
            self.stats["consolidation_cycles"] += 1

            logger.info(
                "Graph consolidation completed",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.dynamic_knowledge_graph.DynamicKnowledgeGraph",
                data={"agent_id": self.agent_id, "stats": consolidation_stats}
            )

            return consolidation_stats

        except Exception as e:
            logger.error(
                "Graph consolidation failed",
                LogCategory.MEMORY_OPERATIONS,
                "app.memory.dynamic_knowledge_graph.DynamicKnowledgeGraph",
                error=e
            )
            return {"error": str(e)}

    async def _find_similar_entities(self) -> List[List[str]]:
        """Find groups of similar entities that should be merged."""
        similar_groups = []
        processed = set()

        for entity_id, entity in self.entities.items():
            if entity_id in processed:
                continue

            similar_entities = [entity_id]

            # Find entities with similar names or descriptions
            for other_id, other_entity in self.entities.items():
                if other_id == entity_id or other_id in processed:
                    continue

                # Simple similarity check (in production, use embeddings)
                name_similarity = self._calculate_string_similarity(
                    entity.name.lower(), other_entity.name.lower()
                )

                if name_similarity > self.config["entity_similarity_threshold"]:
                    similar_entities.append(other_id)
                    processed.add(other_id)

            if len(similar_entities) > 1:
                similar_groups.append(similar_entities)

            processed.add(entity_id)

        return similar_groups

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        # Simple Jaccard similarity
        set1 = set(str1.split())
        set2 = set(str2.split())

        if not set1 and not set2:
            return 1.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    async def _merge_entities(self, entity_ids: List[str]):
        """Merge similar entities into one."""
        if len(entity_ids) < 2:
            return

        # Use the most frequently accessed entity as the primary
        primary_id = max(entity_ids, key=lambda eid: self.entities[eid].access_frequency)
        primary_entity = self.entities[primary_id]

        # Merge information from other entities
        for entity_id in entity_ids:
            if entity_id == primary_id:
                continue

            entity = self.entities[entity_id]

            # Merge associated memories
            primary_entity.associated_memories.update(entity.associated_memories)

            # Update access frequency
            primary_entity.access_frequency += entity.access_frequency

            # Merge properties
            primary_entity.properties.update(entity.properties)

            # Update relationships to point to primary entity
            for rel_id, relationship in self.relationships.items():
                if relationship.source_entity_id == entity_id:
                    relationship.source_entity_id = primary_id
                elif relationship.target_entity_id == entity_id:
                    relationship.target_entity_id = primary_id

            # Remove merged entity
            del self.entities[entity_id]
            self.stats["total_entities"] -= 1

    async def _create_entity_clusters(self) -> List[GraphCluster]:
        """Create clusters of related entities."""
        clusters = []

        # Simple clustering based on relationship density
        processed_entities = set()

        for entity_id in self.entities:
            if entity_id in processed_entities:
                continue

            # Find highly connected entities
            related = await self.get_related_entities(
                entity_id, max_depth=1, min_strength=0.5
            )

            if len(related["entities"]) >= 3:  # Minimum cluster size
                cluster_id = f"cluster_{len(clusters)}_{self.agent_id}"
                cluster_entities = {entity_id}

                for entity in related["entities"]:
                    cluster_entities.add(entity.entity_id)
                    processed_entities.add(entity.entity_id)

                cluster = GraphCluster(
                    cluster_id=cluster_id,
                    cluster_type="semantic",
                    entities=cluster_entities,
                    centroid_entity=entity_id,
                    coherence_score=0.8  # Default coherence
                )

                clusters.append(cluster)
                self.clusters[cluster_id] = cluster

            processed_entities.add(entity_id)

        return clusters

    async def _update_centrality_scores(self):
        """Update centrality scores for all entities."""
        # Simple degree centrality calculation
        entity_connections = defaultdict(int)

        for relationship in self.relationships.values():
            entity_connections[relationship.source_entity_id] += 1
            entity_connections[relationship.target_entity_id] += 1

        max_connections = max(entity_connections.values()) if entity_connections else 1

        for entity_id, entity in self.entities.items():
            connections = entity_connections.get(entity_id, 0)
            entity.centrality_score = connections / max_connections

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        return {
            **self.stats,
            "entity_types": {
                entity_type: len(entities)
                for entity_type, entities in self.entity_type_index.items()
            },
            "relationship_types": {
                rel_type.value: len(relationships)
                for rel_type, relationships in self.relationship_type_index.items()
            },
            "spatial_regions": {
                region: len(entities)
                for region, entities in self.spatial_index.items()
            }
        }
