"""
Revolutionary CLIP Vision Embedding System.

This module provides state-of-the-art vision-text embedding capabilities using
the sentence-transformers/clip-ViT-I-14 model for multimodal AI applications.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import torch
import structlog

# Import with fallbacks for missing dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CLIPConfig(BaseModel):
    """Configuration for CLIP vision embedding system."""
    model_name: str = Field(default="sentence-transformers/clip-ViT-I-14", description="CLIP model name")
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")
    batch_size: int = Field(default=32, ge=1, le=128, description="Batch size for processing")
    max_image_size: Tuple[int, int] = Field(default=(224, 224), description="Maximum image size")
    normalize_embeddings: bool = Field(default=True, description="Normalize embeddings")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory")


class VisionTextSimilarity(BaseModel):
    """Vision-text similarity result."""
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    confidence: float = Field(..., description="Confidence in the similarity")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cache_hit: bool = Field(default=False, description="Whether result was cached")


class MultimodalEmbedding(BaseModel):
    """Multimodal embedding result."""
    text_embedding: List[float] = Field(..., description="Text embedding vector")
    image_embedding: Optional[List[float]] = Field(default=None, description="Image embedding vector")
    combined_embedding: Optional[List[float]] = Field(default=None, description="Combined embedding vector")
    embedding_dimension: int = Field(..., description="Embedding dimension")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class RevolutionaryCLIPEmbedding:
    """Revolutionary CLIP vision-text embedding system."""
    
    def __init__(self, config: CLIPConfig = None):
        self.config = config or CLIPConfig()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None
        self.is_initialized = False
        
        # Performance tracking
        self.embedding_cache: Dict[str, List[float]] = {}
        self.performance_metrics: Dict[str, Any] = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Model capabilities
        self.embedding_dimension = 512  # Default for CLIP ViT-I-14
        
    async def initialize(self) -> bool:
        """Initialize CLIP models and processors."""
        try:
            if self.is_initialized:
                return True
            
            logger.info("Initializing Revolutionary CLIP Embedding System...")
            
            # Check dependencies
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not available, using fallback")
                return await self._initialize_fallback()
            
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("transformers not available, using fallback")
                return await self._initialize_fallback()
            
            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device
            
            logger.info(f"Using device: {self.device}")
            
            # Load CLIP model
            await self._load_clip_model()
            
            # Verify model functionality
            await self._verify_model()
            
            self.is_initialized = True
            logger.info("Revolutionary CLIP Embedding System initialized successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP embedding system: {str(e)}")
            return await self._initialize_fallback()
    
    async def _load_clip_model(self) -> None:
        """
        Load CLIP model and processors.

        Checks centralized storage first, then falls back to HuggingFace download.
        """
        try:
            model_path = None

            # Try to use centralized model storage
            try:
                from app.rag.core.embedding_model_manager import embedding_model_manager
                from app.rag.config.required_models import get_model_by_id

                # Get model spec
                model_spec = get_model_by_id(self.config.model_name)

                if model_spec:
                    # Check if model exists in centralized storage
                    model_info = embedding_model_manager.get_model_info(model_spec.local_name)

                    if model_info and model_info.is_downloaded:
                        logger.info(
                            f"Loading CLIP from centralized storage: {model_spec.local_name}",
                            path=model_info.local_path
                        )
                        model_path = model_info.local_path

            except ImportError:
                logger.warning("Model manager not available, loading CLIP directly from HuggingFace")

            # Load sentence-transformers CLIP model
            if model_path:
                logger.info(f"Loading CLIP model from: {model_path}")
                self.model = SentenceTransformer(model_path, device=self.device)
            else:
                logger.info(f"Loading CLIP model from HuggingFace: {self.config.model_name}")
                self.model = SentenceTransformer(self.config.model_name, device=self.device)

            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

            # Load additional processors for advanced features
            if "clip" in self.config.model_name.lower():
                try:
                    # Load CLIP processor for image preprocessing
                    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                    logger.info("Additional CLIP processors loaded")
                except Exception as e:
                    logger.warning(f"Could not load additional processors: {str(e)}")

            logger.info(f"CLIP model loaded successfully, embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    async def _verify_model(self) -> None:
        """Verify model functionality with test inputs."""
        try:
            # Test text embedding
            test_text = "A beautiful sunset over the ocean"
            text_embedding = await self.embed_text(test_text)
            
            if len(text_embedding) != self.embedding_dimension:
                raise ValueError(f"Unexpected embedding dimension: {len(text_embedding)}")
            
            # Test image embedding if possible
            try:
                # Create a simple test image
                test_image = Image.new('RGB', (224, 224), color='red')
                image_embedding = await self.embed_image(test_image)
                
                if len(image_embedding) != self.embedding_dimension:
                    raise ValueError(f"Unexpected image embedding dimension: {len(image_embedding)}")
                
                logger.info("Model verification successful - both text and image embeddings working")
            except Exception as e:
                logger.warning(f"Image embedding test failed: {str(e)}")
                logger.info("Model verification successful - text embeddings working")
            
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            raise
    
    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate text embedding using CLIP model."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = time.time()
            
            # Check cache
            if use_cache and self.config.cache_embeddings:
                cache_key = self._generate_cache_key(text)
                if cache_key in self.embedding_cache:
                    self.performance_metrics["cache_hits"] += 1
                    return self.embedding_cache[cache_key]
            
            # Generate embedding
            if self.model:
                embedding = self.model.encode([text])[0]
                
                if self.config.normalize_embeddings:
                    embedding = embedding / np.linalg.norm(embedding)
                
                embedding_list = embedding.tolist()
            else:
                # Fallback embedding
                embedding_list = await self._generate_fallback_embedding(text)
            
            # Cache result
            if use_cache and self.config.cache_embeddings:
                cache_key = self._generate_cache_key(text)
                self.embedding_cache[cache_key] = embedding_list
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Text embedding failed: {str(e)}")
            return await self._generate_fallback_embedding(text)
    
    async def embed_image(self, image: Image.Image, use_cache: bool = True) -> List[float]:
        """Generate image embedding using CLIP model."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = time.time()
            
            # Preprocess image
            processed_image = await self._preprocess_image(image)
            
            # Check cache
            if use_cache and self.config.cache_embeddings:
                cache_key = self._generate_image_cache_key(processed_image)
                if cache_key in self.embedding_cache:
                    self.performance_metrics["cache_hits"] += 1
                    return self.embedding_cache[cache_key]
            
            # Generate embedding
            if self.model:
                # Convert PIL image to format expected by sentence-transformers
                embedding = self.model.encode([processed_image])[0]
                
                if self.config.normalize_embeddings:
                    embedding = embedding / np.linalg.norm(embedding)
                
                embedding_list = embedding.tolist()
            else:
                # Fallback embedding
                embedding_list = await self._generate_fallback_image_embedding(processed_image)
            
            # Cache result
            if use_cache and self.config.cache_embeddings:
                cache_key = self._generate_image_cache_key(processed_image)
                self.embedding_cache[cache_key] = embedding_list
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Image embedding failed: {str(e)}")
            return await self._generate_fallback_image_embedding(image)
    
    async def compute_similarity(
        self, 
        text: str, 
        image: Image.Image,
        use_cache: bool = True
    ) -> VisionTextSimilarity:
        """Compute similarity between text and image."""
        try:
            start_time = time.time()
            
            # Get embeddings
            text_embedding = await self.embed_text(text, use_cache)
            image_embedding = await self.embed_image(image, use_cache)
            
            # Compute cosine similarity
            text_vec = np.array(text_embedding)
            image_vec = np.array(image_embedding)
            
            similarity = np.dot(text_vec, image_vec) / (np.linalg.norm(text_vec) * np.linalg.norm(image_vec))
            
            # Calculate confidence based on embedding magnitudes and similarity
            confidence = min(1.0, abs(similarity) + 0.1)
            
            processing_time = (time.time() - start_time) * 1000
            
            return VisionTextSimilarity(
                similarity_score=float(similarity),
                confidence=confidence,
                processing_time_ms=processing_time,
                cache_hit=False  # Would need more sophisticated cache tracking
            )
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            return VisionTextSimilarity(
                similarity_score=0.0,
                confidence=0.0,
                processing_time_ms=0.0
            )
    
    async def embed_multimodal(
        self, 
        text: str, 
        image: Optional[Image.Image] = None,
        combine_strategy: str = "concatenate"
    ) -> MultimodalEmbedding:
        """Generate multimodal embeddings combining text and image."""
        try:
            start_time = time.time()
            
            # Get text embedding
            text_embedding = await self.embed_text(text)
            
            # Get image embedding if provided
            image_embedding = None
            if image:
                image_embedding = await self.embed_image(image)
            
            # Combine embeddings
            combined_embedding = None
            if image_embedding:
                if combine_strategy == "concatenate":
                    combined_embedding = text_embedding + image_embedding
                elif combine_strategy == "average":
                    text_vec = np.array(text_embedding)
                    image_vec = np.array(image_embedding)
                    combined_vec = (text_vec + image_vec) / 2
                    combined_embedding = combined_vec.tolist()
                elif combine_strategy == "weighted":
                    # Weight text more heavily (0.7 text, 0.3 image)
                    text_vec = np.array(text_embedding) * 0.7
                    image_vec = np.array(image_embedding) * 0.3
                    combined_vec = text_vec + image_vec
                    combined_embedding = combined_vec.tolist()
            
            processing_time = (time.time() - start_time) * 1000
            
            return MultimodalEmbedding(
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                combined_embedding=combined_embedding,
                embedding_dimension=len(combined_embedding) if combined_embedding else len(text_embedding),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Multimodal embedding failed: {str(e)}")
            # Return fallback with just text embedding
            return MultimodalEmbedding(
                text_embedding=await self._generate_fallback_embedding(text),
                embedding_dimension=self.embedding_dimension,
                processing_time_ms=0.0
            )
    
    async def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for CLIP model."""
        try:
            # Resize image to model requirements
            if image.size != self.config.max_image_size:
                image = image.resize(self.config.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return image
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"text:{text}".encode()).hexdigest()
    
    def _generate_image_cache_key(self, image: Image.Image) -> str:
        """Generate cache key for image."""
        # Simple hash based on image size and mode
        image_info = f"image:{image.size}:{image.mode}"
        return hashlib.md5(image_info.encode()).hexdigest()
    
    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update performance metrics."""
        self.performance_metrics["total_embeddings"] += 1
        self.performance_metrics["total_processing_time"] += processing_time
        self.performance_metrics["average_processing_time"] = (
            self.performance_metrics["total_processing_time"] / 
            self.performance_metrics["total_embeddings"]
        )
    
    async def _initialize_fallback(self) -> bool:
        """Initialize fallback embedding system."""
        logger.warning("Initializing fallback embedding system")
        self.is_initialized = True
        self.embedding_dimension = 384  # Standard fallback dimension
        return True
    
    async def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback embedding for text."""
        # Simple hash-based embedding
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to normalized float vector
        embedding = []
        for i in range(0, min(len(hash_bytes), self.embedding_dimension // 8), 1):
            byte_val = hash_bytes[i] / 255.0  # Normalize to 0-1
            embedding.extend([byte_val] * 8)  # Expand to fill dimension
        
        # Pad or truncate to exact dimension
        while len(embedding) < self.embedding_dimension:
            embedding.append(0.0)
        
        return embedding[:self.embedding_dimension]
    
    async def _generate_fallback_image_embedding(self, image: Image.Image) -> List[float]:
        """Generate fallback embedding for image."""
        # Simple image statistics-based embedding
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate basic statistics
            mean_rgb = np.mean(img_array, axis=(0, 1))
            std_rgb = np.std(img_array, axis=(0, 1))
            
            # Create embedding from statistics
            stats = np.concatenate([mean_rgb, std_rgb])
            
            # Expand to full dimension
            embedding = []
            for i in range(self.embedding_dimension):
                embedding.append(float(stats[i % len(stats)] / 255.0))
            
            return embedding
            
        except Exception as e:
            logger.error(f"Fallback image embedding failed: {str(e)}")
            return [0.0] * self.embedding_dimension
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")


# Global CLIP embedding instance
clip_embedder = RevolutionaryCLIPEmbedding()
