"""
🚀 Revolutionary Centralized Model Manager

This module provides centralized model management for the RAG system,
ensuring all models are stored and accessed from the data/models/ directory.

Features:
- Centralized model storage at data/models/
- Automatic model detection and registration
- Integration with global configuration system
- Support for embedding and vision models
- Thread-safe operations
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Supported model types."""
    EMBEDDING = "embedding"
    VISION = "vision"
    RERANKING = "reranking"


@dataclass
class ModelInfo:
    """Information about a model."""
    model_id: str
    model_type: ModelType
    local_path: str
    is_downloaded: bool
    size_mb: Optional[float] = None
    description: Optional[str] = None
    config_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "local_path": self.local_path,
            "is_downloaded": self.is_downloaded,
            "size_mb": self.size_mb,
            "description": self.description,
            "config_path": self.config_path
        }


class CentralizedModelManager:
    """
    🚀 Revolutionary Centralized Model Manager
    
    Manages all models from the centralized data/models/ directory.
    Provides seamless integration with the RAG system and global configuration.
    """
    
    def __init__(self):
        """Initialize the centralized model manager."""
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self._models: Dict[str, ModelInfo] = {}
        self._lock = asyncio.Lock()
        
        # Initialize model discovery
        self._discover_models()
        
        logger.info("🚀 Centralized Model Manager initialized", models_dir=str(self.models_dir))
    
    def _discover_models(self) -> None:
        """Discover existing models in the data/models directory."""
        try:
            # Discover embedding models
            embedding_dir = self.models_dir / "embedding"
            if embedding_dir.exists():
                for model_path in embedding_dir.iterdir():
                    if model_path.is_dir() and not self._should_skip_directory(model_path):
                        self._register_model(model_path, ModelType.EMBEDDING)

            # Discover vision models
            vision_dir = self.models_dir / "vision"
            if vision_dir.exists():
                for model_path in vision_dir.iterdir():
                    if model_path.is_dir() and not self._should_skip_directory(model_path):
                        self._register_model(model_path, ModelType.VISION)

            # Discover reranking models
            reranking_dir = self.models_dir / "reranking"
            if reranking_dir.exists():
                for model_path in reranking_dir.iterdir():
                    if model_path.is_dir() and not self._should_skip_directory(model_path):
                        self._register_model(model_path, ModelType.RERANKING)

            logger.info(f"✅ Discovered {len(self._models)} models", models=list(self._models.keys()))

        except Exception as e:
            logger.error(f"❌ Failed to discover models: {str(e)}")

    def _should_skip_directory(self, model_path: Path) -> bool:
        """Check if a directory should be skipped during model discovery."""
        directory_name = model_path.name

        # Skip HuggingFace cache directories
        if directory_name.startswith("models--"):
            return True

        # Skip lock directories
        if directory_name == ".locks":
            return True

        # Skip hidden directories
        if directory_name.startswith("."):
            return True

        # Skip temporary directories
        if directory_name.startswith("tmp") or directory_name.startswith("temp"):
            return True

        return False
    
    def _register_model(self, model_path: Path, model_type: ModelType) -> None:
        """Register a discovered model."""
        try:
            model_id = model_path.name
            
            # Check if model has required files
            is_downloaded = self._validate_model_files(model_path, model_type)
            
            # Calculate size
            size_mb = self._calculate_model_size(model_path) if is_downloaded else None
            
            # Create model info
            model_info = ModelInfo(
                model_id=model_id,
                model_type=model_type,
                local_path=str(model_path),
                is_downloaded=is_downloaded,
                size_mb=size_mb,
                description=f"{model_type.value.title()} model: {model_id}"
            )
            
            self._models[model_id] = model_info
            
            if is_downloaded:
                logger.info(f"✅ Registered {model_type.value} model: {model_id}", path=str(model_path))
            else:
                logger.warning(f"⚠️ Incomplete {model_type.value} model: {model_id}", path=str(model_path))
                
        except Exception as e:
            logger.error(f"❌ Failed to register model {model_path.name}: {str(e)}")
    
    def _validate_model_files(self, model_path: Path, model_type: ModelType) -> bool:
        """Validate that a model has the required files."""
        try:
            if model_type == ModelType.EMBEDDING:
                # Check for sentence-transformers files
                # Accept both modern safetensors and legacy pytorch_model.bin formats
                config_exists = (model_path / "config.json").exists()
                model_exists = (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists()
                return config_exists and model_exists
            
            elif model_type == ModelType.VISION:
                # Check for vision model files
                required_files = ["config.json"]
                return any((model_path / file).exists() for file in required_files)
            
            elif model_type == ModelType.RERANKING:
                # Check for reranking model files
                required_files = ["config.json"]
                return any((model_path / file).exists() for file in required_files)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to validate model files: {str(e)}")
            return False
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate the size of a model in MB."""
        try:
            total_size = 0
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return round(total_size / (1024 * 1024), 2)  # Convert to MB
        except Exception as e:
            logger.error(f"❌ Failed to calculate model size: {str(e)}")
            return 0.0
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self._models.get(model_id)
    
    def get_downloaded_models(self) -> List[ModelInfo]:
        """Get all downloaded models."""
        return [model for model in self._models.values() if model.is_downloaded]
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get all models of a specific type."""
        return [model for model in self._models.values() if model.model_type == model_type]
    
    def get_downloaded_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """Get all downloaded models of a specific type."""
        return [
            model for model in self._models.values() 
            if model.model_type == model_type and model.is_downloaded
        ]
    
    def list_all_models(self) -> Dict[str, ModelInfo]:
        """Get all registered models."""
        return self._models.copy()
    
    async def refresh_models(self) -> None:
        """Refresh the model registry by re-discovering models."""
        async with self._lock:
            self._models.clear()
            self._discover_models()
            logger.info("🔄 Model registry refreshed")
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get the local path for a model."""
        model_info = self.get_model_info(model_id)
        if model_info and model_info.is_downloaded:
            return model_info.local_path
        return None

    async def download_model(self, model_id: str, model_type: Optional[ModelType] = None, force_redownload: bool = False) -> bool:
        """
        Download a model from HuggingFace to the centralized storage.

        Args:
            model_id: HuggingFace model ID (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
            model_type: Type of model (embedding, vision, reranking). If None, will try to infer.
            force_redownload: Force redownload even if model exists

        Returns:
            True if download successful, False otherwise
        """
        try:
            import os

            # Infer model type if not provided
            if model_type is None:
                model_type = self._infer_model_type(model_id)

            # Determine local path
            model_name = model_id.replace("/", "_").replace("-", "_")
            local_path = self.models_dir / model_type.value / model_name

            # Check if already downloaded
            if local_path.exists() and not force_redownload:
                logger.info(f"Model {model_id} already exists at {local_path}")
                return True

            # Create directory
            local_path.mkdir(parents=True, exist_ok=True)

            # Set cache directory to our local path
            os.environ['TRANSFORMERS_CACHE'] = str(local_path.parent)
            os.environ['HF_HOME'] = str(local_path.parent)

            logger.info(f"Downloading {model_type.value} model: {model_id}")

            if model_type == ModelType.EMBEDDING:
                # Try sentence-transformers first, fallback to transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info("Using sentence-transformers for download...")
                    model = SentenceTransformer(model_id, cache_folder=str(local_path.parent))
                    model.save(str(local_path))
                    logger.info(f"✅ Saved embedding model to: {local_path}")
                except ImportError:
                    logger.warning("sentence-transformers not available, using transformers...")
                    from transformers import AutoModel, AutoTokenizer
                    model = AutoModel.from_pretrained(model_id, cache_dir=str(local_path.parent))
                    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(local_path.parent))
                    model.save_pretrained(str(local_path))
                    tokenizer.save_pretrained(str(local_path))
                    logger.info(f"✅ Saved embedding model to: {local_path}")

            elif model_type == ModelType.VISION:
                from transformers import AutoModel, AutoProcessor
                logger.info("Downloading vision model...")
                model = AutoModel.from_pretrained(model_id, cache_dir=str(local_path.parent))
                processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(local_path.parent))
                model.save_pretrained(str(local_path))
                processor.save_pretrained(str(local_path))
                logger.info(f"✅ Saved vision model to: {local_path}")

            elif model_type == ModelType.RERANKING:
                from transformers import AutoModel, AutoTokenizer
                logger.info("Downloading reranking model...")
                model = AutoModel.from_pretrained(model_id, cache_dir=str(local_path.parent))
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(local_path.parent))
                model.save_pretrained(str(local_path))
                tokenizer.save_pretrained(str(local_path))
                logger.info(f"✅ Saved reranking model to: {local_path}")

            # Refresh model registry
            await self.refresh_models()

            return True

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {str(e)}")
            return False

    def _infer_model_type(self, model_id: str) -> ModelType:
        """Infer model type from model ID."""
        model_id_lower = model_id.lower()

        if 'sentence-transformers' in model_id_lower or 'all-minilm' in model_id_lower:
            return ModelType.EMBEDDING
        elif 'clip' in model_id_lower or 'vision' in model_id_lower:
            return ModelType.VISION
        elif 'cross-encoder' in model_id_lower or 'rerank' in model_id_lower:
            return ModelType.RERANKING
        else:
            # Default to embedding
            logger.warning(f"Could not infer model type for {model_id}, defaulting to EMBEDDING")
            return ModelType.EMBEDDING


# Global instance
embedding_model_manager = CentralizedModelManager()


def get_embedding_model_manager() -> CentralizedModelManager:
    """Get the global embedding model manager instance."""
    return embedding_model_manager
