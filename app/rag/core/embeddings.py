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


class EmbeddingType(str, Enum):
    """Types of embeddings supported."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    VISION = "vision"
    CODE = "code"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_type: EmbeddingType = Field(default=EmbeddingType.DENSE)
    batch_size: int = Field(default=32, ge=1, le=128)
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
    Advanced embedding manager with support for multiple models and types.
    
    Features:
    - Dense and sparse embeddings
    - Vision model integration
    - Batch processing optimization
    - Embedding caching
    - Multi-model support
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding manager."""
        self.config = config
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Device configuration
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
            
        logger.info(
            "Embedding manager initialized",
            model=config.model_name,
            type=config.embedding_type.value,
            device=self.device
        )
    
    async def initialize(self) -> None:
        """Initialize embedding models."""
        try:
            await self._load_primary_model()
            
            # Load additional models based on configuration
            if self.config.embedding_type == EmbeddingType.HYBRID:
                await self._load_sparse_model()
            
            if self.config.vision_model:
                await self._load_vision_model()
                
            logger.info("Embedding models loaded successfully")
            
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

                        logger.info(f"Loaded model from model manager: {self.config.model_name}")
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
                logger.warning("sentence_transformers not available, using fallback embedding")
                # Use a simple fallback embedding
                self.models["primary"] = "fallback"
                self.config.dense_dimension = 384  # Default dimension
                return

            model = SentenceTransformer(model_name, device=self.device)
            self.models["primary"] = model

            # Get embedding dimension
            if self.config.dense_dimension is None:
                self.config.dense_dimension = model.get_sentence_embedding_dimension()

            else:
                # Load HuggingFace model
                if not TRANSFORMERS_AVAILABLE:
                    logger.warning("transformers not available, using fallback embedding")
                    # Use a simple fallback embedding
                    self.models["primary"] = "fallback"
                    self.config.dense_dimension = 384  # Default dimension
                    return

                tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                model = AutoModel.from_pretrained(self.config.model_name)
                model.to(self.device)
                
                self.tokenizers["primary"] = tokenizer
                self.models["primary"] = model
                
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
        """Load vision model for image embeddings."""
        try:
            if self.config.vision_model:
                from transformers import CLIPProcessor, CLIPModel
                
                processor = CLIPProcessor.from_pretrained(self.config.vision_model)
                model = CLIPModel.from_pretrained(self.config.vision_model)
                model.to(self.device)
                
                self.tokenizers["vision"] = processor
                self.models["vision"] = model
                
                logger.info(f"Vision model loaded: {self.config.vision_model}")
                
        except Exception as e:
            logger.error(f"Failed to load vision model: {str(e)}")
            raise
    
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
            if embedding_type == EmbeddingType.DENSE:
                embeddings = await self._generate_dense_embeddings(texts)
            elif embedding_type == EmbeddingType.SPARSE:
                embeddings = await self._generate_sparse_embeddings(texts)
            elif embedding_type == EmbeddingType.HYBRID:
                embeddings = await self._generate_hybrid_embeddings(texts)
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")
            
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
            if model == "fallback":
                logger.warning("Using fallback embedding generation")
                # Generate simple hash-based embeddings as fallback
                embeddings = []
                for text in texts:
                    # Create a simple deterministic embedding based on text hash
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    # Convert hex to numbers and normalize to create embedding
                    embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), 32), 2)]
                    # Pad to desired dimension
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
    
    def get_embedding_cache_key(self, text: str, embedding_type: EmbeddingType) -> str:
        """Generate cache key for embedding."""
        content = f"{text}_{embedding_type.value}_{self.config.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "primary_model": self.config.model_name,
            "embedding_type": self.config.embedding_type.value,
            "device": self.device,
            "dimension": self.config.dense_dimension,
            "loaded_models": list(self.models.keys()),
            "cache_size": len(self.embedding_cache)
        }
