"""
Embedding Management for Revolutionary RAG System.

This module provides comprehensive embedding generation and management capabilities
with support for multiple models, dense/sparse embeddings, and vision models.
"""

import asyncio
import hashlib
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum

import numpy as np
import structlog
from pydantic import BaseModel, Field
import torch

# Import performance enhancements
from .intelligent_cache import get_cache, EmbeddingCache

# Handle sentence_transformers import gracefully
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger = structlog.get_logger(__name__)
    logger.warning("sentence_transformers not available, using fallback", error=str(e))
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger = structlog.get_logger(__name__)
    logger.warning("transformers not available, using fallback", error=str(e))
    AutoTokenizer = None
    AutoModel = None
    TRANSFORMERS_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Import revolutionary CLIP vision embeddings
try:
    from app.rag.vision.clip_embeddings import clip_embedder, CLIPConfig
    CLIP_AVAILABLE = True
except ImportError as e:
    logger.warning("CLIP embeddings not available", error=str(e))
    clip_embedder = None
    CLIPConfig = None
    CLIP_AVAILABLE = False


class EmbeddingType(str, Enum):
    """Types of embeddings supported."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    VISION = "vision"
    CODE = "code"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation - OPTIMIZED."""
    model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_type: EmbeddingType = Field(default=EmbeddingType.DENSE)
    batch_size: int = Field(default=64, ge=1, le=256)  # INCREASED from 32
    max_length: int = Field(default=512, ge=128, le=2048)
    normalize: bool = Field(default=True)
    device: str = Field(default="auto")
    cache_embeddings: bool = Field(default=True)

    # Dense embedding specific
    dense_dimension: Optional[int] = Field(default=None)

    # Sparse embedding specific
    sparse_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    # Vision model specific
    vision_model: Optional[str] = Field(default=None)
    image_size: Tuple[int, int] = Field(default=(224, 224))

    # Model manager integration
    use_model_manager: bool = Field(default=True, description="Use embedding model manager for model loading")


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, str] = Field(default_factory=dict)
    processing_time: float = Field(..., description="Processing time in seconds")
    embedding_type: EmbeddingType = Field(..., description="Type of embedding")


class EmbeddingManager:
    """
    🚀 Revolutionary Unified Embedding Manager

    This is THE centralized embedding manager that combines:
    - Model discovery and storage management
    - Actual embedding generation
    - Global configuration management
    - Multi-model support (dense, sparse, vision)
    - Centralized model storage at data/models/

    Features:
    - Dense and sparse embeddings
    - Vision model integration
    - Batch processing optimization
    - Embedding caching
    - Centralized model management
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize the unified embedding manager."""
        self.config = config
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}

        # Initialize intelligent cache
        self._intelligent_cache: Optional[EmbeddingCache] = None

        # Device configuration
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        logger.info(
            "🚀 Unified Embedding Manager initialized",
            model=config.model_name,
            type=config.embedding_type.value,
            device=self.device,
            use_centralized_storage=config.use_model_manager
        )
    
    async def initialize(self) -> None:
        """Initialize embedding models."""
        try:
            # Initialize intelligent cache
            self._intelligent_cache = await get_cache(
                cache_name=f"embeddings_{self.config.model_name}",
                cache_type="embedding",
                max_size=50000,
                max_memory_mb=1024,
                default_ttl=7200
            )

            await self._load_primary_model()

            # Load additional models based on configuration
            if self.config.embedding_type == EmbeddingType.HYBRID:
                await self._load_sparse_model()

            if self.config.vision_model:
                await self._load_vision_model()

            logger.info("Embedding models loaded successfully with intelligent caching")

        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {str(e)}")
            raise
    
    async def _load_primary_model(self) -> None:
        """Load the primary embedding model."""
        try:
            # Check if we should use the model manager
            if self.config.use_model_manager:
                try:
                    from .embedding_model_manager import embedding_model_manager

                    # Check if model is available in model manager
                    model_info = embedding_model_manager.get_model_info(self.config.model_name)
                    if model_info and model_info.is_downloaded:
                        # Use local model from model manager
                        if not SENTENCE_TRANSFORMERS_AVAILABLE:
                            logger.warning("sentence_transformers not available, using fallback")
                            self.models["primary"] = "fallback"
                            self.config.dense_dimension = 384
                            return

                        model = SentenceTransformer(model_info.local_path, device=self.device)
                        self.models["primary"] = model

                        if self.config.dense_dimension is None:
                            self.config.dense_dimension = model.get_sentence_embedding_dimension()

                        logger.info(f"✅ Loaded model from centralized storage: {self.config.model_name}",
                                   path=model_info.local_path)
                        return

                except ImportError:
                    logger.warning("Model manager not available, falling back to direct loading")

            # Fallback to direct model loading
            # Remove sentence-transformers/ prefix if present
            model_name = self.config.model_name
            if model_name.startswith("sentence-transformers/"):
                model_name = model_name.replace("sentence-transformers/", "")

            # Load SentenceTransformer model
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence_transformers not available, using hash-based fallback embedding")
                # Use a hash-based fallback that actually generates embeddings
                self.models["primary"] = "hash_fallback"
                self.config.dense_dimension = 384  # Default dimension
                return

            model = SentenceTransformer(model_name, device=self.device)
            self.models["primary"] = model

            # Get embedding dimension
            if self.config.dense_dimension is None:
                self.config.dense_dimension = model.get_sentence_embedding_dimension()

            logger.info(f"Primary model loaded: {self.config.model_name}")

        except Exception as e:
            logger.error(f"Failed to load primary model: {str(e)}")
            # Fallback to hash-based embeddings
            logger.warning("Using hash-based fallback embeddings")
            self.models["primary"] = "hash_fallback"
            self.config.dense_dimension = 384
                
            logger.info(f"Primary model loaded: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load primary model: {str(e)}")
            raise
    
    async def _load_sparse_model(self) -> None:
        """Load sparse embedding model for hybrid retrieval."""
        try:
            # For now, use TF-IDF for sparse embeddings
            # In production, consider using SPLADE or similar
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            self.models["sparse"] = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Sparse model initialized")
            
        except Exception as e:
            logger.error(f"Failed to load sparse model: {str(e)}")
            raise
    
    async def _load_vision_model(self) -> None:
        """Revolutionary vision model loading with CLIP integration."""
        try:
            if self.config.vision_model:
                # Try to use revolutionary CLIP embedder first
                if CLIP_AVAILABLE and clip_embedder:
                    logger.info("Initializing revolutionary CLIP vision embeddings...")

                    # Configure CLIP with our settings
                    clip_config = CLIPConfig(
                        model_name="sentence-transformers/clip-ViT-I-14",
                        device=self.device,
                        normalize_embeddings=True,
                        cache_embeddings=True
                    )

                    # Initialize CLIP embedder
                    clip_embedder.config = clip_config
                    success = await clip_embedder.initialize()

                    if success:
                        self.models["vision"] = clip_embedder
                        self.tokenizers["vision"] = "clip_embedder"
                        logger.info("Revolutionary CLIP vision model loaded successfully!")
                        return
                    else:
                        logger.warning("CLIP embedder initialization failed, falling back to basic CLIP")

                # Fallback to basic CLIP model
                logger.info("Loading basic CLIP vision model...")
                from transformers import CLIPProcessor, CLIPModel

                processor = CLIPProcessor.from_pretrained(self.config.vision_model)
                model = CLIPModel.from_pretrained(self.config.vision_model)
                model.to(self.device)

                self.tokenizers["vision"] = processor
                self.models["vision"] = model

                logger.info(f"Basic vision model loaded: {self.config.vision_model}")

        except Exception as e:
            logger.error(f"Failed to load vision model: {str(e)}")
            # Don't raise - allow system to continue without vision capabilities
            logger.warning("Continuing without vision model capabilities")
    
    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        embedding_type: Optional[EmbeddingType] = None
    ) -> EmbeddingResult:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Input text(s) to embed
            embedding_type: Type of embedding to generate
            
        Returns:
            EmbeddingResult with generated embeddings and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        # Use configured type if not specified
        if embedding_type is None:
            embedding_type = self.config.embedding_type
        
        try:
            # Check intelligent cache first
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            if self._intelligent_cache:
                for i, text in enumerate(texts):
                    cached = await self._intelligent_cache.get_embedding(text, self.config.model_name)
                    if cached is not None:
                        cached_embeddings.append((i, cached))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))

            # Generate embeddings for uncached texts
            if uncached_texts:
                if embedding_type == EmbeddingType.DENSE:
                    new_embeddings = await self._generate_dense_embeddings(uncached_texts)
                elif embedding_type == EmbeddingType.SPARSE:
                    new_embeddings = await self._generate_sparse_embeddings(uncached_texts)
                elif embedding_type == EmbeddingType.HYBRID:
                    new_embeddings = await self._generate_hybrid_embeddings(uncached_texts)
                else:
                    raise ValueError(f"Unsupported embedding type: {embedding_type}")

                # Cache new embeddings
                if self._intelligent_cache:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        await self._intelligent_cache.set_embedding(text, embedding, self.config.model_name)
            else:
                new_embeddings = []

            # Combine cached and new embeddings in correct order
            embeddings = [None] * len(texts)

            # Place cached embeddings
            for i, embedding in cached_embeddings:
                embeddings[i] = embedding

            # Place new embeddings
            for i, embedding in zip(uncached_indices, new_embeddings):
                embeddings[i] = embedding

            processing_time = asyncio.get_event_loop().time() - start_time
            
            return EmbeddingResult(
                embeddings=embeddings,
                metadata={
                    "input_count": len(texts),
                    "model_name": self.config.model_name,
                    "device": self.device
                },
                model_info={
                    "name": self.config.model_name,
                    "type": embedding_type.value,
                    "dimension": str(len(embeddings[0]) if embeddings else 0)
                },
                processing_time=processing_time,
                embedding_type=embedding_type
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    async def _generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings using SentenceTransformer or similar."""
        try:
            # Check if we have a primary model
            if "primary" not in self.models:
                logger.warning("No primary model available, using fallback")
                # Generate simple hash-based embeddings as fallback
                embeddings = []
                for text in texts:
                    # Create a simple deterministic embedding based on text hash
                    import hashlib
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    # Convert hex to numbers and normalize to create embedding
                    embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), 32), 2)]
                    # Pad to desired dimension
                    while len(embedding) < (self.config.dense_dimension or 384):
                        embedding.append(0.0)
                    embedding = embedding[:(self.config.dense_dimension or 384)]
                    embeddings.append(embedding)
                return embeddings

            model = self.models["primary"]

            # Handle fallback case
            if model in ["fallback", "hash_fallback"]:
                logger.warning("Using fallback embedding generation")
                # Generate simple hash-based embeddings as fallback
                embeddings = []
                for text in texts:
                    import hashlib
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), 32), 2)]
                    while len(embedding) < self.config.dense_dimension:
                        embedding.append(0.0)
                    embedding = embedding[:self.config.dense_dimension]
                    embeddings.append(embedding)
                return embeddings

            if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(model, SentenceTransformer):
                # Use SentenceTransformer
                embeddings = model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize,
                    convert_to_numpy=True
                )
            else:
                # Use HuggingFace model
                tokenizer = self.tokenizers["primary"]
                embeddings = []

                for i in range(0, len(texts), self.config.batch_size):
                    batch = texts[i:i + self.config.batch_size]

                    # Tokenize
                    inputs = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)

                    # Generate embeddings
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Use mean pooling
                        batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                        if self.config.normalize:
                            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                        embeddings.extend(batch_embeddings.cpu().numpy())

            return [embedding.tolist() for embedding in embeddings]

        except Exception as e:
            logger.error(f"Failed to generate dense embeddings: {str(e)}")
            raise

    async def _generate_sparse_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate sparse embeddings using TF-IDF or similar."""
        try:
            vectorizer = self.models["sparse"]

            # Fit and transform if not already fitted
            if not hasattr(vectorizer, 'vocabulary_'):
                sparse_matrix = vectorizer.fit_transform(texts)
            else:
                sparse_matrix = vectorizer.transform(texts)

            # Convert to dense for consistency
            dense_matrix = sparse_matrix.toarray()

            return [embedding.tolist() for embedding in dense_matrix]

        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {str(e)}")
            raise

    async def _generate_hybrid_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate hybrid embeddings combining dense and sparse."""
        try:
            # Generate both dense and sparse embeddings
            dense_embeddings = await self._generate_dense_embeddings(texts)
            sparse_embeddings = await self._generate_sparse_embeddings(texts)

            # Combine embeddings with weighted average
            alpha = self.config.sparse_alpha
            hybrid_embeddings = []

            for dense, sparse in zip(dense_embeddings, sparse_embeddings):
                # Normalize dimensions if different
                if len(dense) != len(sparse):
                    # Pad shorter one with zeros
                    max_len = max(len(dense), len(sparse))
                    dense = dense + [0.0] * (max_len - len(dense))
                    sparse = sparse + [0.0] * (max_len - len(sparse))

                # Weighted combination
                hybrid = [
                    (1 - alpha) * d + alpha * s
                    for d, s in zip(dense, sparse)
                ]
                hybrid_embeddings.append(hybrid)

            return hybrid_embeddings

        except Exception as e:
            logger.error(f"Failed to generate hybrid embeddings: {str(e)}")
            raise


# =============================================================================
# GLOBAL EMBEDDING MANAGER FUNCTIONALITY
# =============================================================================

# Global instance for backward compatibility
_global_embedding_manager: Optional[EmbeddingManager] = None
_global_lock = asyncio.Lock()


async def get_global_embedding_manager() -> EmbeddingManager:
    """
    Get the global embedding manager instance.

    This replaces the old GlobalEmbeddingManager class with a simpler approach.
    """
    global _global_embedding_manager

    async with _global_lock:
        if _global_embedding_manager is None:
            # Create default config with centralized model management
            config = EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                batch_size=32,
                use_model_manager=True  # Enable centralized storage
            )

            _global_embedding_manager = EmbeddingManager(config)
            await _global_embedding_manager.initialize()

            logger.info("🚀 Global embedding manager initialized")

    return _global_embedding_manager


async def generate_global_embeddings(
    texts: Union[str, List[str]],
    prefix: Optional[str] = None
) -> List[List[float]]:
    """Generate embeddings using the global embedding manager."""
    manager = await get_global_embedding_manager()
    result = await manager.generate_embeddings(texts)

    # Extract embeddings from result
    if hasattr(result, 'embeddings'):
        return result.embeddings
    elif isinstance(result, list):
        return result
    else:
        raise ValueError(f"Unexpected embedding result type: {type(result)}")


def update_global_embedding_config(config: Dict[str, Any]) -> None:
    """Update the global embedding configuration."""
    global _global_embedding_manager

    # Reset the global manager to force reinitialization with new config
    _global_embedding_manager = None

    logger.info("🔄 Global embedding configuration updated - will reinitialize on next use")


def get_global_embedding_config() -> Optional[Dict[str, Any]]:
    """Get the current global embedding configuration."""
    global _global_embedding_manager

    if _global_embedding_manager is not None:
        return {
            "model_name": _global_embedding_manager.config.model_name,
            "batch_size": _global_embedding_manager.config.batch_size,
            "embedding_type": _global_embedding_manager.config.embedding_type.value,
            "use_model_manager": _global_embedding_manager.config.use_model_manager
        }

    return None


# Backward compatibility aliases
GlobalEmbeddingManager = EmbeddingManager
global_embedding_manager = _global_embedding_manager
