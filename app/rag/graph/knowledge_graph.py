"""
Revolutionary Knowledge Graph System for RAG 4.0.

This module provides advanced knowledge graph capabilities including:
- Entity extraction and recognition
- Relationship mapping and inference
- Graph-enhanced search and retrieval
- Semantic reasoning and inference
- Multi-hop relationship discovery
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re

import structlog
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..core.knowledge_base import Document
from ..core.embeddings import EmbeddingManager
from ..core.caching import get_rag_cache, CacheType

logger = structlog.get_logger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    EVENT = "event"
    DOCUMENT = "document"
    TOPIC = "topic"
    KEYWORD = "keyword"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of relationships between entities."""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    CREATED_BY = "created_by"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    MENTIONS = "mentions"
    CONTAINS = "contains"
    REFERENCES = "references"


@dataclass
class Entity:
    """Knowledge graph entity."""
    id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    source_documents: Optional[List[str]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.properties is None:
            self.properties = {}
        if self.source_documents is None:
            self.source_documents = []


@dataclass
class Relationship:
    """Knowledge graph relationship."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    properties: Optional[Dict[str, Any]] = None
    source_documents: Optional[List[str]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.properties is None:
            self.properties = {}
        if self.source_documents is None:
            self.source_documents = []


@dataclass
class GraphPath:
    """Path through the knowledge graph."""
    entities: List[Entity]
    relationships: List[Relationship]
    total_weight: float
    confidence: float
    path_length: int


class EntityExtractor:
    """Advanced entity extraction from text."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.entity_patterns = self._load_entity_patterns()
        self.entity_cache = {}
    
    def _load_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """Load regex patterns for entity recognition."""
        return {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(?:Dr|Mr|Ms|Mrs|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b'  # Title First Last
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b(?:University of|Institute of) [A-Z][a-z]+\b'
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+, [A-Z][A-Z]\b',  # City, State
                r'\b[A-Z][a-z]+ (?:University|College|Institute)\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(?:AI|ML|NLP|API|GPU|CPU|RAM|SSD|HDD)\b',
                r'\b(?:Python|JavaScript|Java|C\+\+|React|Node\.js)\b',
                r'\b(?:TensorFlow|PyTorch|Keras|Scikit-learn)\b'
            ],
            EntityType.CONCEPT: [
                r'\b(?:machine learning|artificial intelligence|deep learning)\b',
                r'\b(?:neural network|algorithm|model|framework)\b'
            ]
        }
    
    async def extract_entities(self, text: str, document_id: str = None) -> List[Entity]:
        """Extract entities from text using multiple methods."""
        entities = []
        
        try:
            # Method 1: Pattern-based extraction
            pattern_entities = await self._extract_by_patterns(text)
            entities.extend(pattern_entities)
            
            # Method 2: Keyword-based extraction
            keyword_entities = await self._extract_keywords(text)
            entities.extend(keyword_entities)
            
            # Method 3: Semantic extraction (simplified)
            semantic_entities = await self._extract_semantic_entities(text)
            entities.extend(semantic_entities)
            
            # Deduplicate and merge similar entities
            entities = await self._deduplicate_entities(entities)
            
            # Add document reference
            if document_id:
                for entity in entities:
                    entity.source_documents.append(document_id)
            
            # Generate embeddings for entities
            for entity in entities:
                if not entity.embedding:
                    entity.embedding = await self.embedding_manager.embed_text(entity.name)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
    
    async def _extract_by_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group().strip()
                    if len(entity_name) > 2:  # Filter out very short matches
                        entity = Entity(
                            id=str(uuid.uuid4()),
                            name=entity_name,
                            entity_type=entity_type,
                            confidence=0.8  # Pattern-based confidence
                        )
                        entities.append(entity)
        
        return entities
    
    async def _extract_keywords(self, text: str) -> List[Entity]:
        """Extract important keywords as entities."""
        # Simple keyword extraction - in production use advanced NLP
        words = text.split()
        keywords = []
        
        # Extract capitalized words (potential proper nouns)
        for word in words:
            if word[0].isupper() and len(word) > 3:
                if word not in ['The', 'This', 'That', 'When', 'Where', 'What', 'How']:
                    keywords.append(word)
        
        # Convert to entities
        entities = []
        for keyword in set(keywords):  # Remove duplicates
            entity = Entity(
                id=str(uuid.uuid4()),
                name=keyword,
                entity_type=EntityType.KEYWORD,
                confidence=0.6  # Lower confidence for keywords
            )
            entities.append(entity)
        
        return entities
    
    async def _extract_semantic_entities(self, text: str) -> List[Entity]:
        """Extract entities using semantic analysis."""
        # Placeholder for advanced semantic entity extraction
        # In production, use spaCy, BERT-based NER, or other advanced models
        entities = []
        
        # Simple implementation: extract noun phrases
        sentences = text.split('.')
        for sentence in sentences[:3]:  # Limit to first 3 sentences
            words = sentence.split()
            if len(words) > 5:  # Only process longer sentences
                # Look for potential concepts (simplified)
                for i, word in enumerate(words):
                    if word.lower() in ['system', 'model', 'algorithm', 'framework', 'architecture']:
                        if i > 0:  # Get preceding word as modifier
                            concept_name = f"{words[i-1]} {word}"
                            entity = Entity(
                                id=str(uuid.uuid4()),
                                name=concept_name,
                                entity_type=EntityType.CONCEPT,
                                confidence=0.7
                            )
                            entities.append(entity)
        
        return entities
    
    async def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate and similar entities."""
        if not entities:
            return []
        
        # Simple deduplication by name similarity
        unique_entities = []
        seen_names = set()
        
        for entity in entities:
            name_lower = entity.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_entities.append(entity)
            else:
                # Merge with existing entity (increase confidence)
                for existing in unique_entities:
                    if existing.name.lower() == name_lower:
                        existing.confidence = max(existing.confidence, entity.confidence)
                        existing.source_documents.extend(entity.source_documents)
                        break
        
        return unique_entities


class RelationshipMapper:
    """Maps relationships between entities."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.relationship_patterns = self._load_relationship_patterns()
    
    def _load_relationship_patterns(self) -> Dict[RelationType, List[str]]:
        """Load patterns for relationship detection."""
        return {
            RelationType.RELATED_TO: [
                r'{entity1}.*(?:related to|associated with|connected to).*{entity2}',
                r'{entity1}.*(?:and|with).*{entity2}'
            ],
            RelationType.PART_OF: [
                r'{entity1}.*(?:part of|component of|element of).*{entity2}',
                r'{entity2}.*(?:contains|includes|comprises).*{entity1}'
            ],
            RelationType.CREATED_BY: [
                r'{entity1}.*(?:created by|developed by|built by).*{entity2}',
                r'{entity2}.*(?:created|developed|built).*{entity1}'
            ],
            RelationType.DEPENDS_ON: [
                r'{entity1}.*(?:depends on|relies on|requires).*{entity2}',
                r'{entity1}.*(?:uses|utilizes|employs).*{entity2}'
            ]
        }
    
    async def extract_relationships(
        self, 
        entities: List[Entity], 
        text: str,
        document_id: str = None
    ) -> List[Relationship]:
        """Extract relationships between entities from text."""
        relationships = []
        
        try:
            # Method 1: Pattern-based relationship extraction
            pattern_relationships = await self._extract_by_patterns(entities, text)
            relationships.extend(pattern_relationships)
            
            # Method 2: Co-occurrence based relationships
            cooccurrence_relationships = await self._extract_by_cooccurrence(entities, text)
            relationships.extend(cooccurrence_relationships)
            
            # Method 3: Semantic similarity relationships
            similarity_relationships = await self._extract_by_similarity(entities)
            relationships.extend(similarity_relationships)
            
            # Add document reference
            if document_id:
                for relationship in relationships:
                    relationship.source_documents.append(document_id)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {str(e)}")
            return []
    
    async def _extract_by_patterns(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Extract relationships using text patterns."""
        relationships = []
        
        # Create entity name mapping
        entity_map = {entity.name.lower(): entity for entity in entities}
        
        for relation_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                # Try all entity pairs
                for entity1 in entities:
                    for entity2 in entities:
                        if entity1.id != entity2.id:
                            # Replace placeholders in pattern
                            filled_pattern = pattern.replace('{entity1}', re.escape(entity1.name))
                            filled_pattern = filled_pattern.replace('{entity2}', re.escape(entity2.name))
                            
                            if re.search(filled_pattern, text, re.IGNORECASE):
                                relationship = Relationship(
                                    id=str(uuid.uuid4()),
                                    source_entity_id=entity1.id,
                                    target_entity_id=entity2.id,
                                    relation_type=relation_type,
                                    confidence=0.8,
                                    weight=1.0
                                )
                                relationships.append(relationship)
        
        return relationships
    
    async def _extract_by_cooccurrence(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence."""
        relationships = []
        
        # Split text into sentences
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_entities = []
            for entity in entities:
                if entity.name.lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # Create relationships between co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    relationship = Relationship(
                        id=str(uuid.uuid4()),
                        source_entity_id=entity1.id,
                        target_entity_id=entity2.id,
                        relation_type=RelationType.RELATED_TO,
                        confidence=0.6,  # Lower confidence for co-occurrence
                        weight=0.5
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _extract_by_similarity(self, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships based on semantic similarity."""
        relationships = []
        
        # Calculate similarity between entity embeddings
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1.embedding and entity2.embedding:
                    # Calculate cosine similarity
                    similarity = np.dot(entity1.embedding, entity2.embedding) / (
                        np.linalg.norm(entity1.embedding) * np.linalg.norm(entity2.embedding)
                    )
                    
                    # Create relationship if similarity is high enough
                    if similarity > 0.7:
                        relationship = Relationship(
                            id=str(uuid.uuid4()),
                            source_entity_id=entity1.id,
                            target_entity_id=entity2.id,
                            relation_type=RelationType.SIMILAR_TO,
                            confidence=similarity,
                            weight=similarity,
                            properties={"similarity_score": similarity}
                        )
                        relationships.append(relationship)
        
        return relationships


class KnowledgeGraph:
    """
    Revolutionary knowledge graph for RAG 4.0.
    
    Features:
    - Entity extraction and management
    - Relationship mapping and inference
    - Graph-based search and reasoning
    - Multi-hop relationship discovery
    - Semantic graph analysis
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.entity_extractor = EntityExtractor(embedding_manager)
        self.relationship_mapper = RelationshipMapper(embedding_manager)
        
        # Graph storage
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Cache
        self.cache = None
        
        # Analytics
        self.graph_stats = {
            "total_entities": 0,
            "total_relationships": 0,
            "entity_types": {},
            "relationship_types": {}
        }
    
    async def initialize(self) -> None:
        """Initialize the knowledge graph."""
        try:
            self.cache = await get_rag_cache()
            logger.info("Knowledge graph initialized")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {str(e)}")
            raise
    
    async def process_document(self, document: Document) -> Dict[str, Any]:
        """Process a document and extract entities and relationships."""
        try:
            # Extract entities
            entities = await self.entity_extractor.extract_entities(
                document.content, 
                document.id
            )
            
            # Extract relationships
            relationships = await self.relationship_mapper.extract_relationships(
                entities, 
                document.content, 
                document.id
            )
            
            # Add to graph
            await self._add_entities(entities)
            await self._add_relationships(relationships)
            
            # Update statistics
            await self._update_stats()
            
            result = {
                "document_id": document.id,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "entities": [asdict(e) for e in entities],
                "relationships": [asdict(r) for r in relationships]
            }
            
            logger.info(f"Processed document {document.id}: {len(entities)} entities, {len(relationships)} relationships")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
    
    async def search_entities(
        self, 
        query: str, 
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10
    ) -> List[Entity]:
        """Search for entities in the graph."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_text(query)
            
            # Calculate similarities
            similarities = []
            for entity in self.entities.values():
                if entity_types and entity.entity_type not in entity_types:
                    continue
                
                if entity.embedding:
                    similarity = np.dot(query_embedding, entity.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entity.embedding)
                    )
                    similarities.append((entity, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [entity for entity, _ in similarities[:limit]]
            
        except Exception as e:
            logger.error(f"Entity search failed: {str(e)}")
            return []
    
    async def find_path(
        self, 
        source_entity_id: str, 
        target_entity_id: str,
        max_hops: int = 3
    ) -> Optional[GraphPath]:
        """Find path between two entities in the graph."""
        try:
            if not self.graph.has_node(source_entity_id) or not self.graph.has_node(target_entity_id):
                return None
            
            # Use NetworkX to find shortest path
            try:
                path_nodes = nx.shortest_path(
                    self.graph, 
                    source_entity_id, 
                    target_entity_id,
                    weight='weight'
                )
                
                if len(path_nodes) > max_hops + 1:
                    return None
                
                # Build GraphPath object
                entities = [self.entities[node_id] for node_id in path_nodes]
                relationships = []
                total_weight = 0
                total_confidence = 0
                
                for i in range(len(path_nodes) - 1):
                    source_id = path_nodes[i]
                    target_id = path_nodes[i + 1]
                    
                    # Find relationship between these nodes
                    edge_data = self.graph.get_edge_data(source_id, target_id)
                    if edge_data:
                        relationship_id = list(edge_data.keys())[0]
                        relationship = self.relationships[relationship_id]
                        relationships.append(relationship)
                        total_weight += relationship.weight
                        total_confidence += relationship.confidence
                
                avg_confidence = total_confidence / len(relationships) if relationships else 0
                
                return GraphPath(
                    entities=entities,
                    relationships=relationships,
                    total_weight=total_weight,
                    confidence=avg_confidence,
                    path_length=len(relationships)
                )
                
            except nx.NetworkXNoPath:
                return None
            
        except Exception as e:
            logger.error(f"Path finding failed: {str(e)}")
            return None
    
    async def get_related_entities(
        self, 
        entity_id: str, 
        relation_types: Optional[List[RelationType]] = None,
        max_distance: int = 2
    ) -> List[Tuple[Entity, float]]:
        """Get entities related to a given entity."""
        try:
            if not self.graph.has_node(entity_id):
                return []
            
            related = []
            
            # Get direct neighbors
            for neighbor_id in self.graph.neighbors(entity_id):
                neighbor_entity = self.entities[neighbor_id]
                
                # Calculate relationship strength
                edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
                if edge_data:
                    max_weight = max(data.get('weight', 0) for data in edge_data.values())
                    related.append((neighbor_entity, max_weight))
            
            # Get second-degree neighbors if requested
            if max_distance > 1:
                for neighbor_id in list(self.graph.neighbors(entity_id)):
                    for second_neighbor_id in self.graph.neighbors(neighbor_id):
                        if second_neighbor_id != entity_id and second_neighbor_id not in [r[0].id for r in related]:
                            second_neighbor_entity = self.entities[second_neighbor_id]
                            # Reduced weight for second-degree connections
                            related.append((second_neighbor_entity, 0.5))
            
            # Sort by relationship strength
            related.sort(key=lambda x: x[1], reverse=True)
            return related
            
        except Exception as e:
            logger.error(f"Related entities search failed: {str(e)}")
            return []
    
    async def _add_entities(self, entities: List[Entity]) -> None:
        """Add entities to the graph."""
        for entity in entities:
            self.entities[entity.id] = entity
            self.graph.add_node(entity.id, **asdict(entity))
    
    async def _add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to the graph."""
        for relationship in relationships:
            self.relationships[relationship.id] = relationship
            self.graph.add_edge(
                relationship.source_entity_id,
                relationship.target_entity_id,
                key=relationship.id,
                weight=relationship.weight,
                **asdict(relationship)
            )
    
    async def _update_stats(self) -> None:
        """Update graph statistics."""
        self.graph_stats["total_entities"] = len(self.entities)
        self.graph_stats["total_relationships"] = len(self.relationships)
        
        # Count entity types
        entity_type_counts = {}
        for entity in self.entities.values():
            entity_type = entity.entity_type.value
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        self.graph_stats["entity_types"] = entity_type_counts
        
        # Count relationship types
        relationship_type_counts = {}
        for relationship in self.relationships.values():
            rel_type = relationship.relation_type.value
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
        self.graph_stats["relationship_types"] = relationship_type_counts
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        await self._update_stats()
        
        # Add NetworkX graph metrics
        if self.graph.number_of_nodes() > 0:
            self.graph_stats.update({
                "graph_density": nx.density(self.graph),
                "connected_components": nx.number_weakly_connected_components(self.graph),
                "average_clustering": nx.average_clustering(self.graph.to_undirected()),
                "graph_diameter": self._safe_diameter()
            })
        
        return self.graph_stats.copy()
    
    def _safe_diameter(self) -> Optional[int]:
        """Safely calculate graph diameter."""
        try:
            if nx.is_weakly_connected(self.graph):
                return nx.diameter(self.graph.to_undirected())
        except:
            pass
        return None


# Global knowledge graph instance
knowledge_graph = None


async def get_knowledge_graph(embedding_manager: EmbeddingManager) -> KnowledgeGraph:
    """Get the global knowledge graph instance."""
    global knowledge_graph
    
    if knowledge_graph is None:
        knowledge_graph = KnowledgeGraph(embedding_manager)
        await knowledge_graph.initialize()
    
    return knowledge_graph
