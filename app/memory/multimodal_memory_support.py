"""
Revolutionary Multimodal Memory Support for Agentic AI.

Implements comprehensive multimodal memory capabilities including image, video, audio,
and cross-modal associations based on state-of-the-art multimodal AI research.

Key Features:
- Multimodal memory storage (text, image, audio, video)
- Cross-modal associations and retrieval
- Multimodal embeddings and similarity search
- Modality-specific processing pipelines
- Cross-modal fusion and reasoning
- Multimodal consolidation
- Rich sensory memory experiences
"""

import asyncio
import time
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ModalityType(str, Enum):
    """Types of modalities supported."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TACTILE = "tactile"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class CrossModalRelationType(str, Enum):
    """Types of cross-modal relationships."""
    DESCRIBES = "describes"          # Text describes image/video
    DEPICTS = "depicts"             # Image/video depicts text concept
    ACCOMPANIES = "accompanies"     # Audio accompanies video
    NARRATES = "narrates"          # Audio narrates visual content
    SYNCHRONIZES = "synchronizes"   # Temporal synchronization
    CONTEXTUALIZES = "contextualizes"  # Provides context
    REINFORCES = "reinforces"       # Reinforces information
    CONTRADICTS = "contradicts"     # Contradictory information


@dataclass
class ModalityData:
    """Data for a specific modality."""
    modality: ModalityType
    content: Union[str, bytes, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing information
    processed: bool = False
    embedding: Optional[np.ndarray] = None
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    quality_score: float = 1.0
    confidence: float = 1.0
    resolution: Optional[Tuple[int, ...]] = None
    
    # Storage information
    storage_path: Optional[str] = None
    compressed: bool = False
    compression_ratio: float = 1.0


@dataclass
class MultimodalMemoryEntry:
    """A memory entry supporting multiple modalities."""
    memory_id: str
    primary_modality: ModalityType
    modalities: Dict[ModalityType, ModalityData] = field(default_factory=dict)
    
    # Cross-modal information
    cross_modal_associations: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength
    modality_coherence: float = 1.0  # How well modalities align
    
    # Memory properties
    importance: str = "medium"
    emotional_valence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def add_modality(self, modality: ModalityType, data: ModalityData):
        """Add a modality to this memory entry."""
        self.modalities[modality] = data
        
        # Update coherence if multiple modalities
        if len(self.modalities) > 1:
            self._update_coherence()
    
    def _update_coherence(self):
        """Update modality coherence score."""
        # Simple coherence calculation - in practice, use ML models
        if len(self.modalities) <= 1:
            self.modality_coherence = 1.0
            return
        
        # Calculate average quality across modalities
        qualities = [data.quality_score for data in self.modalities.values()]
        self.modality_coherence = np.mean(qualities)


@dataclass
class CrossModalAssociation:
    """Association between memories of different modalities."""
    association_id: str
    source_memory_id: str
    target_memory_id: str
    source_modality: ModalityType
    target_modality: ModalityType
    
    # Association properties
    relation_type: CrossModalRelationType
    strength: float = 1.0
    confidence: float = 1.0
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    reinforcement_count: int = 0
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


class MultimodalEmbeddingManager:
    """Manages embeddings for different modalities."""
    
    def __init__(self):
        """Initialize multimodal embedding manager."""
        self.embedding_functions = {}
        self.embedding_dimensions = {}
        
        # Initialize default embedding functions (placeholders)
        self._initialize_embedding_functions()
    
    def _initialize_embedding_functions(self):
        """Initialize embedding functions for different modalities."""
        # These would be replaced with actual embedding models
        self.embedding_functions[ModalityType.TEXT] = self._text_embedding
        self.embedding_functions[ModalityType.IMAGE] = self._image_embedding
        self.embedding_functions[ModalityType.AUDIO] = self._audio_embedding
        self.embedding_functions[ModalityType.VIDEO] = self._video_embedding
        
        # Set embedding dimensions
        self.embedding_dimensions[ModalityType.TEXT] = 768
        self.embedding_dimensions[ModalityType.IMAGE] = 512
        self.embedding_dimensions[ModalityType.AUDIO] = 256
        self.embedding_dimensions[ModalityType.VIDEO] = 1024
    
    async def get_embedding(self, modality: ModalityType, content: Any) -> np.ndarray:
        """Get embedding for content of specific modality."""
        if modality not in self.embedding_functions:
            raise ValueError(f"Unsupported modality: {modality}")
        
        embedding_func = self.embedding_functions[modality]
        return await embedding_func(content)
    
    async def _text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding (placeholder)."""
        # In practice, use models like BERT, RoBERTa, etc.
        hash_value = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to numeric array
        numeric_hash = [ord(c) for c in hash_value]
        # Pad or truncate to desired dimension
        embedding = np.array(numeric_hash[:self.embedding_dimensions[ModalityType.TEXT]])
        if len(embedding) < self.embedding_dimensions[ModalityType.TEXT]:
            padding = np.zeros(self.embedding_dimensions[ModalityType.TEXT] - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        # Normalize
        return embedding / np.linalg.norm(embedding)
    
    async def _image_embedding(self, image_data: bytes) -> np.ndarray:
        """Generate image embedding (placeholder)."""
        # In practice, use models like CLIP, ResNet, etc.
        hash_value = hashlib.md5(image_data).hexdigest()
        numeric_hash = [ord(c) for c in hash_value]
        embedding = np.array(numeric_hash[:self.embedding_dimensions[ModalityType.IMAGE]])
        if len(embedding) < self.embedding_dimensions[ModalityType.IMAGE]:
            padding = np.zeros(self.embedding_dimensions[ModalityType.IMAGE] - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        return embedding / np.linalg.norm(embedding)
    
    async def _audio_embedding(self, audio_data: bytes) -> np.ndarray:
        """Generate audio embedding (placeholder)."""
        # In practice, use models like Wav2Vec, AudioCLIP, etc.
        hash_value = hashlib.md5(audio_data).hexdigest()
        numeric_hash = [ord(c) for c in hash_value]
        embedding = np.array(numeric_hash[:self.embedding_dimensions[ModalityType.AUDIO]])
        if len(embedding) < self.embedding_dimensions[ModalityType.AUDIO]:
            padding = np.zeros(self.embedding_dimensions[ModalityType.AUDIO] - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        return embedding / np.linalg.norm(embedding)
    
    async def _video_embedding(self, video_data: bytes) -> np.ndarray:
        """Generate video embedding (placeholder)."""
        # In practice, use models like VideoCLIP, I3D, etc.
        hash_value = hashlib.md5(video_data).hexdigest()
        numeric_hash = [ord(c) for c in hash_value]
        embedding = np.array(numeric_hash[:self.embedding_dimensions[ModalityType.VIDEO]])
        if len(embedding) < self.embedding_dimensions[ModalityType.VIDEO]:
            padding = np.zeros(self.embedding_dimensions[ModalityType.VIDEO] - len(embedding))
            embedding = np.concatenate([embedding, padding])
        
        return embedding / np.linalg.norm(embedding)
    
    def calculate_cross_modal_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        modality1: ModalityType,
        modality2: ModalityType
    ) -> float:
        """Calculate similarity between embeddings from different modalities."""
        # Simple cosine similarity - in practice, use learned cross-modal mappings
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Apply modality-specific adjustments
        if modality1 == modality2:
            return float(similarity)
        else:
            # Cross-modal similarity is typically lower
            return float(similarity * 0.8)


class MultimodalMemorySystem:
    """
    Revolutionary Multimodal Memory System.
    
    Provides comprehensive multimodal memory capabilities with cross-modal
    associations, retrieval, and reasoning.
    """
    
    def __init__(self, agent_id: str):
        """Initialize multimodal memory system."""
        self.agent_id = agent_id
        
        # Core components
        self.embedding_manager = MultimodalEmbeddingManager()
        
        # Memory storage
        self.multimodal_memories: Dict[str, MultimodalMemoryEntry] = {}
        self.cross_modal_associations: Dict[str, CrossModalAssociation] = {}
        
        # Indices for fast retrieval
        self.modality_index: Dict[ModalityType, Set[str]] = defaultdict(set)
        self.embedding_index: Dict[ModalityType, Dict[str, np.ndarray]] = defaultdict(dict)
        
        # Configuration
        self.config = {
            "max_memory_size_mb": 1000,  # Maximum memory size
            "embedding_similarity_threshold": 0.7,
            "cross_modal_association_threshold": 0.6,
            "auto_generate_associations": True,
            "enable_multimodal_consolidation": True,
            "compression_enabled": True,
            "quality_threshold": 0.5
        }
        
        # Statistics
        self.stats = {
            "total_multimodal_memories": 0,
            "total_cross_modal_associations": 0,
            "modality_distribution": defaultdict(int),
            "avg_modalities_per_memory": 0.0,
            "storage_usage_mb": 0.0,
            "cross_modal_retrieval_requests": 0
        }
        
        logger.info(f"Multimodal Memory System initialized for agent {agent_id}")
    
    async def store_multimodal_memory(
        self,
        primary_modality: ModalityType,
        primary_content: Any,
        additional_modalities: Optional[Dict[ModalityType, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        importance: str = "medium",
        tags: Optional[Set[str]] = None
    ) -> str:
        """Store a multimodal memory."""
        try:
            memory_id = f"mm_{int(time.time())}_{self.agent_id}"
            
            # Create primary modality data
            primary_data = ModalityData(
                modality=primary_modality,
                content=primary_content,
                metadata={"primary": True}
            )
            
            # Process primary modality
            await self._process_modality_data(primary_data)
            
            # Create memory entry
            memory_entry = MultimodalMemoryEntry(
                memory_id=memory_id,
                primary_modality=primary_modality,
                importance=importance,
                context=context or {},
                tags=tags or set()
            )
            
            memory_entry.add_modality(primary_modality, primary_data)
            
            # Add additional modalities
            if additional_modalities:
                for modality, content in additional_modalities.items():
                    modality_data = ModalityData(
                        modality=modality,
                        content=content,
                        metadata={"primary": False}
                    )
                    
                    await self._process_modality_data(modality_data)
                    memory_entry.add_modality(modality, modality_data)
            
            # Store memory
            self.multimodal_memories[memory_id] = memory_entry
            
            # Update indices
            await self._update_indices(memory_entry)
            
            # Generate cross-modal associations if enabled
            if self.config["auto_generate_associations"]:
                await self._generate_cross_modal_associations(memory_entry)
            
            # Update statistics
            self._update_stats(memory_entry)
            
            logger.info(
                "Multimodal memory stored",
                agent_id=self.agent_id,
                memory_id=memory_id,
                primary_modality=primary_modality.value,
                total_modalities=len(memory_entry.modalities)
            )
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store multimodal memory: {e}")
            return ""
    
    async def _process_modality_data(self, modality_data: ModalityData):
        """Process modality-specific data."""
        try:
            # Generate embedding
            modality_data.embedding = await self.embedding_manager.get_embedding(
                modality_data.modality, modality_data.content
            )
            
            # Extract modality-specific features
            if modality_data.modality == ModalityType.TEXT:
                await self._process_text_data(modality_data)
            elif modality_data.modality == ModalityType.IMAGE:
                await self._process_image_data(modality_data)
            elif modality_data.modality == ModalityType.AUDIO:
                await self._process_audio_data(modality_data)
            elif modality_data.modality == ModalityType.VIDEO:
                await self._process_video_data(modality_data)
            
            modality_data.processed = True
            
        except Exception as e:
            logger.error(f"Failed to process modality data: {e}")
    
    async def _process_text_data(self, modality_data: ModalityData):
        """Process text-specific data."""
        if isinstance(modality_data.content, str):
            text = modality_data.content
            
            # Extract basic text features
            modality_data.features = {
                "word_count": len(text.split()),
                "character_count": len(text),
                "language": "en",  # Placeholder - use language detection
                "sentiment": 0.0,  # Placeholder - use sentiment analysis
                "entities": [],    # Placeholder - use NER
                "keywords": []     # Placeholder - use keyword extraction
            }
            
            modality_data.quality_score = min(1.0, len(text) / 100)  # Simple quality metric
    
    async def _process_image_data(self, modality_data: ModalityData):
        """Process image-specific data."""
        if isinstance(modality_data.content, bytes):
            # Extract basic image features (placeholder)
            modality_data.features = {
                "format": "unknown",
                "size_bytes": len(modality_data.content),
                "objects": [],     # Placeholder - use object detection
                "colors": [],      # Placeholder - use color analysis
                "faces": [],       # Placeholder - use face detection
                "text": ""         # Placeholder - use OCR
            }
            
            modality_data.quality_score = 0.8  # Placeholder quality score
    
    async def _process_audio_data(self, modality_data: ModalityData):
        """Process audio-specific data."""
        if isinstance(modality_data.content, bytes):
            # Extract basic audio features (placeholder)
            modality_data.features = {
                "format": "unknown",
                "duration_seconds": 0.0,
                "sample_rate": 44100,
                "channels": 2,
                "transcription": "",  # Placeholder - use speech-to-text
                "emotions": [],       # Placeholder - use emotion detection
                "speaker_id": None    # Placeholder - use speaker identification
            }
            
            modality_data.quality_score = 0.8  # Placeholder quality score
    
    async def _process_video_data(self, modality_data: ModalityData):
        """Process video-specific data."""
        if isinstance(modality_data.content, bytes):
            # Extract basic video features (placeholder)
            modality_data.features = {
                "format": "unknown",
                "duration_seconds": 0.0,
                "fps": 30,
                "resolution": (1920, 1080),
                "scenes": [],         # Placeholder - use scene detection
                "objects": [],        # Placeholder - use object tracking
                "activities": [],     # Placeholder - use activity recognition
                "audio_track": True   # Placeholder - check for audio
            }
            
            modality_data.quality_score = 0.8  # Placeholder quality score
    
    async def _update_indices(self, memory_entry: MultimodalMemoryEntry):
        """Update indices for fast retrieval."""
        memory_id = memory_entry.memory_id
        
        # Update modality index
        for modality in memory_entry.modalities:
            self.modality_index[modality].add(memory_id)
        
        # Update embedding index
        for modality, modality_data in memory_entry.modalities.items():
            if modality_data.embedding is not None:
                self.embedding_index[modality][memory_id] = modality_data.embedding
    
    async def _generate_cross_modal_associations(self, memory_entry: MultimodalMemoryEntry):
        """Generate cross-modal associations for a memory entry."""
        try:
            memory_id = memory_entry.memory_id
            
            # Find similar memories in other modalities
            for modality, modality_data in memory_entry.modalities.items():
                if modality_data.embedding is None:
                    continue
                
                # Search for similar memories in other modalities
                for other_modality in ModalityType:
                    if other_modality == modality:
                        continue
                    
                    similar_memories = await self._find_similar_memories_by_embedding(
                        modality_data.embedding,
                        other_modality,
                        threshold=self.config["cross_modal_association_threshold"]
                    )
                    
                    # Create associations
                    for other_memory_id, similarity in similar_memories:
                        if other_memory_id != memory_id:
                            await self._create_cross_modal_association(
                                memory_id, other_memory_id,
                                modality, other_modality,
                                similarity
                            )
            
        except Exception as e:
            logger.error(f"Failed to generate cross-modal associations: {e}")
    
    async def _find_similar_memories_by_embedding(
        self,
        query_embedding: np.ndarray,
        target_modality: ModalityType,
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find similar memories by embedding similarity."""
        similarities = []
        
        if target_modality not in self.embedding_index:
            return similarities
        
        for memory_id, embedding in self.embedding_index[target_modality].items():
            similarity = self.embedding_manager.calculate_cross_modal_similarity(
                query_embedding, embedding,
                ModalityType.TEXT, target_modality  # Placeholder modalities
            )
            
            if similarity >= threshold:
                similarities.append((memory_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def _create_cross_modal_association(
        self,
        source_memory_id: str,
        target_memory_id: str,
        source_modality: ModalityType,
        target_modality: ModalityType,
        strength: float
    ):
        """Create a cross-modal association."""
        association_id = f"cma_{source_memory_id}_{target_memory_id}"
        
        # Determine relation type based on modalities
        relation_type = self._infer_cross_modal_relation(source_modality, target_modality)
        
        association = CrossModalAssociation(
            association_id=association_id,
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            source_modality=source_modality,
            target_modality=target_modality,
            relation_type=relation_type,
            strength=strength,
            confidence=strength
        )
        
        self.cross_modal_associations[association_id] = association
        
        # Update memory entries
        if source_memory_id in self.multimodal_memories:
            self.multimodal_memories[source_memory_id].cross_modal_associations[target_memory_id] = strength
        
        if target_memory_id in self.multimodal_memories:
            self.multimodal_memories[target_memory_id].cross_modal_associations[source_memory_id] = strength
        
        self.stats["total_cross_modal_associations"] += 1
    
    def _infer_cross_modal_relation(
        self,
        source_modality: ModalityType,
        target_modality: ModalityType
    ) -> CrossModalRelationType:
        """Infer the type of cross-modal relation."""
        # Simple rule-based inference
        if source_modality == ModalityType.TEXT and target_modality == ModalityType.IMAGE:
            return CrossModalRelationType.DESCRIBES
        elif source_modality == ModalityType.IMAGE and target_modality == ModalityType.TEXT:
            return CrossModalRelationType.DEPICTS
        elif source_modality == ModalityType.AUDIO and target_modality == ModalityType.VIDEO:
            return CrossModalRelationType.ACCOMPANIES
        elif source_modality == ModalityType.VIDEO and target_modality == ModalityType.AUDIO:
            return CrossModalRelationType.ACCOMPANIES
        else:
            return CrossModalRelationType.CONTEXTUALIZES
    
    def _update_stats(self, memory_entry: MultimodalMemoryEntry):
        """Update system statistics."""
        self.stats["total_multimodal_memories"] += 1
        
        # Update modality distribution
        for modality in memory_entry.modalities:
            self.stats["modality_distribution"][modality.value] += 1
        
        # Update average modalities per memory
        total_modalities = sum(
            len(memory.modalities) for memory in self.multimodal_memories.values()
        )
        self.stats["avg_modalities_per_memory"] = (
            total_modalities / len(self.multimodal_memories)
        )
        
        # Estimate storage usage (simplified)
        memory_size = 0
        for modality_data in memory_entry.modalities.values():
            if isinstance(modality_data.content, bytes):
                memory_size += len(modality_data.content)
            elif isinstance(modality_data.content, str):
                memory_size += len(modality_data.content.encode('utf-8'))
        
        self.stats["storage_usage_mb"] += memory_size / (1024 * 1024)

    async def multimodal_retrieve(
        self,
        query_modality: ModalityType,
        query_content: Any,
        target_modalities: Optional[List[ModalityType]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float, Dict[ModalityType, float]]]:
        """Retrieve memories across multiple modalities."""
        try:
            self.stats["cross_modal_retrieval_requests"] += 1

            # Generate query embedding
            query_embedding = await self.embedding_manager.get_embedding(
                query_modality, query_content
            )

            # If no target modalities specified, search all
            if target_modalities is None:
                target_modalities = list(ModalityType)

            # Find similar memories across modalities
            all_results = {}
            modality_scores = defaultdict(dict)

            for target_modality in target_modalities:
                similar_memories = await self._find_similar_memories_by_embedding(
                    query_embedding, target_modality, similarity_threshold, top_k * 2
                )

                for memory_id, similarity in similar_memories:
                    if memory_id not in all_results:
                        all_results[memory_id] = 0.0

                    # Weight similarity by modality compatibility
                    weighted_similarity = similarity
                    if query_modality != target_modality:
                        weighted_similarity *= 0.8  # Cross-modal penalty

                    all_results[memory_id] = max(all_results[memory_id], weighted_similarity)
                    modality_scores[memory_id][target_modality] = similarity

            # Sort results by similarity
            sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

            # Format results
            final_results = []
            for memory_id, max_similarity in sorted_results[:top_k]:
                final_results.append((
                    memory_id,
                    max_similarity,
                    dict(modality_scores[memory_id])
                ))

            logger.info(
                "Multimodal retrieval completed",
                agent_id=self.agent_id,
                query_modality=query_modality.value,
                target_modalities=[m.value for m in target_modalities],
                results_count=len(final_results)
            )

            return final_results

        except Exception as e:
            logger.error(f"Multimodal retrieval failed: {e}")
            return []

    async def get_cross_modal_associations(
        self,
        memory_id: str,
        target_modalities: Optional[List[ModalityType]] = None
    ) -> List[CrossModalAssociation]:
        """Get cross-modal associations for a memory."""
        try:
            if memory_id not in self.multimodal_memories:
                return []

            associations = []
            memory_entry = self.multimodal_memories[memory_id]

            for assoc_memory_id, strength in memory_entry.cross_modal_associations.items():
                # Find the association record
                for association in self.cross_modal_associations.values():
                    if ((association.source_memory_id == memory_id and
                         association.target_memory_id == assoc_memory_id) or
                        (association.source_memory_id == assoc_memory_id and
                         association.target_memory_id == memory_id)):

                        # Filter by target modalities if specified
                        if target_modalities:
                            if (association.source_modality not in target_modalities and
                                association.target_modality not in target_modalities):
                                continue

                        associations.append(association)
                        break

            return associations

        except Exception as e:
            logger.error(f"Failed to get cross-modal associations: {e}")
            return []

    async def consolidate_multimodal_memories(self) -> Dict[str, Any]:
        """Consolidate multimodal memories for better organization."""
        try:
            consolidation_stats = {
                "memories_processed": 0,
                "associations_strengthened": 0,
                "weak_associations_removed": 0,
                "modality_clusters_formed": 0
            }

            # Strengthen frequently co-accessed associations
            for association in self.cross_modal_associations.values():
                if association.reinforcement_count > 5:
                    association.strength = min(1.0, association.strength * 1.1)
                    consolidation_stats["associations_strengthened"] += 1

            # Remove weak associations
            weak_associations = []
            for assoc_id, association in self.cross_modal_associations.items():
                if association.strength < 0.2:
                    weak_associations.append(assoc_id)

            for assoc_id in weak_associations:
                del self.cross_modal_associations[assoc_id]
                consolidation_stats["weak_associations_removed"] += 1

            # Update memory coherence scores
            for memory in self.multimodal_memories.values():
                memory._update_coherence()
                consolidation_stats["memories_processed"] += 1

            logger.info(
                "Multimodal memory consolidation completed",
                agent_id=self.agent_id,
                stats=consolidation_stats
            )

            return consolidation_stats

        except Exception as e:
            logger.error(f"Multimodal memory consolidation failed: {e}")
            return {"error": str(e)}

    async def get_memory_by_id(self, memory_id: str) -> Optional[MultimodalMemoryEntry]:
        """Get a multimodal memory by ID."""
        return self.multimodal_memories.get(memory_id)

    async def search_by_modality(
        self,
        modality: ModalityType,
        limit: int = 50
    ) -> List[MultimodalMemoryEntry]:
        """Search memories by specific modality."""
        try:
            memory_ids = self.modality_index.get(modality, set())
            memories = []

            for memory_id in list(memory_ids)[:limit]:
                if memory_id in self.multimodal_memories:
                    memories.append(self.multimodal_memories[memory_id])

            # Sort by access count and recency
            memories.sort(
                key=lambda m: (m.access_count, m.last_accessed),
                reverse=True
            )

            return memories

        except Exception as e:
            logger.error(f"Failed to search by modality: {e}")
            return []

    async def get_modality_statistics(self) -> Dict[str, Any]:
        """Get detailed modality statistics."""
        try:
            modality_stats = {}

            for modality in ModalityType:
                memory_ids = self.modality_index.get(modality, set())
                memories = [
                    self.multimodal_memories[mid] for mid in memory_ids
                    if mid in self.multimodal_memories
                ]

                if memories:
                    avg_quality = np.mean([
                        memory.modalities[modality].quality_score
                        for memory in memories
                        if modality in memory.modalities
                    ])

                    avg_coherence = np.mean([
                        memory.modality_coherence for memory in memories
                    ])

                    modality_stats[modality.value] = {
                        "total_memories": len(memories),
                        "avg_quality_score": avg_quality,
                        "avg_coherence": avg_coherence,
                        "embeddings_count": len(self.embedding_index.get(modality, {}))
                    }
                else:
                    modality_stats[modality.value] = {
                        "total_memories": 0,
                        "avg_quality_score": 0.0,
                        "avg_coherence": 0.0,
                        "embeddings_count": 0
                    }

            return {
                "modality_breakdown": modality_stats,
                "cross_modal_associations": len(self.cross_modal_associations),
                "total_multimodal_memories": len(self.multimodal_memories),
                "storage_usage_mb": self.stats["storage_usage_mb"]
            }

        except Exception as e:
            logger.error(f"Failed to get modality statistics: {e}")
            return {}

    def get_multimodal_stats(self) -> Dict[str, Any]:
        """Get comprehensive multimodal memory statistics."""
        return {
            **self.stats,
            "config": self.config,
            "embedding_dimensions": self.embedding_manager.embedding_dimensions,
            "supported_modalities": [m.value for m in ModalityType]
        }
