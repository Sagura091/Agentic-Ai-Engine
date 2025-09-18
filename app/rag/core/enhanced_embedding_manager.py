"""
Enhanced Embedding Manager inspired by OpenWebUI.

This module provides a robust embedding management system that handles:
- Multiple embedding engines (local, Ollama, OpenAI)
- Proper model downloading and caching
- Fallback mechanisms for reliability
- Batch processing for efficiency
"""

import os
import asyncio
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from app.http_client import SimpleHTTPClient
from app.rag.config.openwebui_config import get_rag_config
from app.rag.core.simple_embedding import get_fallback_embedding_function

logger = structlog.get_logger(__name__)

class EmbeddingModelInfo(BaseModel):
    """Information about an embedding model."""
    model_id: str
    engine: str = Field(description="Engine type: '', 'ollama', 'openai', 'azure_openai'")
    dimension: int = Field(default=384, description="Embedding dimension")
    max_sequence_length: int = Field(default=512, description="Maximum sequence length")
    is_downloaded: bool = Field(default=False, description="Whether model is downloaded locally")
    local_path: Optional[str] = Field(default=None, description="Local storage path")
    download_date: Optional[datetime] = Field(default=None, description="Download timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")

class EnhancedEmbeddingManager:
    """
    Enhanced embedding manager inspired by OpenWebUI's implementation.
    
    Features:
    - Multiple embedding engines support
    - Proper model downloading and caching
    - Fallback mechanisms for reliability
    - Batch processing for efficiency
    - OpenWebUI-style configuration
    """
    
    def __init__(self, config=None):
        self.config = config or get_rag_config()
        self.models_dir = Path(self.config.config.models_dir) / "embedding"
        self.cache_dir = Path(self.config.config.cache_dir) / "sentence_transformers"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models: Dict[str, EmbeddingModelInfo] = {}
        self.embedding_functions: Dict[str, Any] = {}
        
        # Current active model
        self.current_model: Optional[str] = None
        
        logger.info("✅ Enhanced embedding manager initialized", 
                   models_dir=str(self.models_dir),
                   cache_dir=str(self.cache_dir))
    
    async def initialize(self):
        """Initialize the embedding manager with the configured model."""
        try:
            engine = self.config.config.embedding_engine
            model = self.config.config.embedding_model
            
            await self.load_embedding_model(engine, model)
            logger.info("✅ Embedding manager initialized successfully", 
                       engine=engine, model=model)
            
        except Exception as e:
            logger.error("❌ Failed to initialize embedding manager", error=str(e))
            # Initialize fallback
            await self._initialize_fallback()
    
    async def load_embedding_model(self, engine: str, model: str):
        """Load an embedding model based on engine type."""
        model_key = f"{engine}:{model}"
        
        if model_key in self.embedding_functions:
            self.current_model = model_key
            logger.info("✅ Embedding model already loaded", model=model_key)
            return
        
        try:
            if engine == "":
                # Local sentence-transformers
                await self._load_local_model(model)
            elif engine == "ollama":
                # Ollama embedding
                await self._load_ollama_model(model)
            elif engine in ["openai", "azure_openai"]:
                # OpenAI/Azure embedding
                await self._load_openai_model(engine, model)
            else:
                raise ValueError(f"Unknown embedding engine: {engine}")
            
            self.current_model = model_key
            
            # Update model info
            self.models[model_key] = EmbeddingModelInfo(
                model_id=model,
                engine=engine,
                is_downloaded=(engine == ""),
                local_path=str(self.models_dir / model.replace("/", "_")) if engine == "" else None,
                download_date=datetime.now() if engine == "" else None,
                last_used=datetime.now()
            )
            
            logger.info("✅ Embedding model loaded successfully", model=model_key)
            
        except Exception as e:
            logger.error("❌ Failed to load embedding model", model=model_key, error=str(e))
            raise
    
    async def _load_local_model(self, model: str):
        """Load local sentence-transformers model."""
        try:
            # First, try to download/cache the model properly
            await self._download_sentence_transformer_model(model)
            
            # Import sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            # Try to load from our models directory first
            model_path = self.models_dir / model.replace("/", "_")
            
            if model_path.exists():
                # Load from local cache
                st_model = SentenceTransformer(str(model_path))
                logger.info("✅ Loaded model from local cache", path=str(model_path))
            else:
                # Load from sentence-transformers cache
                st_model = SentenceTransformer(
                    model,
                    cache_folder=str(self.cache_dir)
                )
                logger.info("✅ Loaded model from sentence-transformers cache", model=model)
            
            # Create embedding function
            def embedding_function(texts: Union[str, List[str]], prefix: Optional[str] = None) -> List[List[float]]:
                if isinstance(texts, str):
                    texts = [texts]
                
                # Apply prefix if provided
                if prefix:
                    texts = [f"{prefix}{text}" for text in texts]
                
                # Generate embeddings
                embeddings = st_model.encode(
                    texts, 
                    batch_size=self.config.config.embedding_batch_size,
                    convert_to_tensor=False,
                    normalize_embeddings=True
                )
                
                return embeddings.tolist()
            
            model_key = f":{model}"
            self.embedding_functions[model_key] = embedding_function
            
        except Exception as e:
            logger.error("❌ Failed to load local model", model=model, error=str(e))
            raise
    
    async def _download_sentence_transformer_model(self, model: str):
        """Download sentence transformer model properly."""
        try:
            # Check if we need to download
            model_path = self.models_dir / model.replace("/", "_")
            if model_path.exists():
                logger.info("Model already exists locally", path=str(model_path))
                return
            
            logger.info("📥 Downloading sentence transformer model", model=model)
            
            # Import sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            # Create temporary download directory
            temp_dir = self.cache_dir / "temp_download"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Download model to temporary location
                st_model = SentenceTransformer(
                    model,
                    cache_folder=str(temp_dir)
                )
                
                # Find the downloaded model directory
                downloaded_dirs = list(temp_dir.glob("*"))
                if downloaded_dirs:
                    source_dir = downloaded_dirs[0]
                    
                    # Move to permanent location
                    if model_path.exists():
                        shutil.rmtree(model_path)
                    shutil.move(str(source_dir), str(model_path))
                    
                    logger.info("✅ Model downloaded and cached", 
                               model=model, path=str(model_path))
                else:
                    logger.warning("No downloaded model found in temp directory")
                
            finally:
                # Clean up temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            logger.error("❌ Failed to download sentence transformer model", 
                        model=model, error=str(e))
            # Don't raise here, let the caller handle fallback
    
    async def _load_ollama_model(self, model: str):
        """Load Ollama embedding model."""
        async def ollama_embedding_function(texts: Union[str, List[str]], prefix: Optional[str] = None) -> List[List[float]]:
            if isinstance(texts, str):
                texts = [texts]
            
            # Apply prefix if provided
            if prefix:
                texts = [f"{prefix}{text}" for text in texts]
            
            embeddings = []
            batch_size = self.config.config.embedding_batch_size
            
            # Get Ollama URL from environment or use default
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            
            async with SimpleHTTPClient(
                base_url=ollama_url,
                timeout=60,
                verify_ssl=False
            ) as client:
                
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    response = await client.post(
                        "/api/embed",
                        json={
                            "model": model,
                            "input": batch
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        batch_embeddings = data.get("embeddings", [])
                        embeddings.extend(batch_embeddings)
                    else:
                        raise Exception(f"Ollama embedding failed: {response.text}")
            
            return embeddings
        
        model_key = f"ollama:{model}"
        self.embedding_functions[model_key] = ollama_embedding_function
    
    async def _load_openai_model(self, engine: str, model: str):
        """Load OpenAI/Azure embedding model."""
        async def openai_embedding_function(texts: Union[str, List[str]], prefix: Optional[str] = None) -> List[List[float]]:
            if isinstance(texts, str):
                texts = [texts]
            
            # Apply prefix if provided
            if prefix:
                texts = [f"{prefix}{text}" for text in texts]
            
            embeddings = []
            batch_size = self.config.config.embedding_batch_size
            
            # Get API configuration
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            
            if not api_key:
                raise Exception("OpenAI API key not configured")
            
            async with SimpleHTTPClient(
                base_url=base_url,
                timeout=60
            ) as client:
                
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    response = await client.post(
                        "/embeddings",
                        json={
                            "model": model,
                            "input": batch
                        },
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        embeddings.extend(batch_embeddings)
                    else:
                        raise Exception(f"OpenAI embedding failed: {response.text}")
            
            return embeddings
        
        model_key = f"{engine}:{model}"
        self.embedding_functions[model_key] = openai_embedding_function
    
    async def _initialize_fallback(self):
        """Initialize fallback embedding function."""
        try:
            fallback_function = get_fallback_embedding_function("tfidf", 384)
            
            def embedding_function(texts: Union[str, List[str]], prefix: Optional[str] = None) -> List[List[float]]:
                if isinstance(texts, str):
                    texts = [texts]
                
                # Apply prefix if provided
                if prefix:
                    texts = [f"{prefix}{text}" for text in texts]
                
                return fallback_function(texts)
            
            self.embedding_functions["fallback:tfidf"] = embedding_function
            self.current_model = "fallback:tfidf"
            
            logger.info("✅ Fallback embedding function initialized")
            
        except Exception as e:
            logger.error("❌ Failed to initialize fallback embedding", error=str(e))
            raise
    
    async def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        prefix: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using the current model."""
        if not self.current_model or self.current_model not in self.embedding_functions:
            await self.initialize()
        
        if not self.current_model:
            raise Exception("No embedding model available")
        
        embedding_func = self.embedding_functions[self.current_model]
        
        try:
            if asyncio.iscoroutinefunction(embedding_func):
                return await embedding_func(texts, prefix)
            else:
                return embedding_func(texts, prefix)
        except Exception as e:
            logger.error("❌ Embedding generation failed", model=self.current_model, error=str(e))
            
            # Try fallback if not already using it
            if not self.current_model.startswith("fallback:"):
                logger.info("🔄 Trying fallback embedding")
                await self._initialize_fallback()
                return await self.generate_embeddings(texts, prefix)
            else:
                raise
    
    def get_model_info(self) -> Optional[EmbeddingModelInfo]:
        """Get information about the current model."""
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model]
        return None
    
    def list_available_models(self) -> List[EmbeddingModelInfo]:
        """List all available models."""
        return list(self.models.values())

# Global embedding manager instance
_global_embedding_manager: Optional[EnhancedEmbeddingManager] = None

def get_embedding_manager() -> EnhancedEmbeddingManager:
    """Get the global embedding manager instance."""
    global _global_embedding_manager
    if _global_embedding_manager is None:
        _global_embedding_manager = EnhancedEmbeddingManager()
    return _global_embedding_manager

def reset_embedding_manager():
    """Reset the global embedding manager (useful for testing)."""
    global _global_embedding_manager
    _global_embedding_manager = None
