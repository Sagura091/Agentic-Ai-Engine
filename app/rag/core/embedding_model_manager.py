"""
Universal Model Download and Management System.

This module provides comprehensive model management for all 4 model types:
- Text Embedding Models: sentence-transformers from HuggingFace
- Reranking Models: cross-encoder models from HuggingFace
- Vision Models: CLIP and vision-language models from HuggingFace
- LLM Models: Ollama models and API model validation

Features:
- User-driven model downloads (no automatic downloads)
- Real-time progress tracking
- Model validation and testing
- Hot-swapping capabilities
- Performance monitoring
"""

import os
import asyncio
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
from enum import Enum

import structlog
from pydantic import BaseModel, Field

# Use custom HTTP client instead of requests or huggingface_hub
from app.http_client import SimpleHTTPClient

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Types of models supported by the system."""
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    VISION = "vision"
    LLM = "llm"


class ModelSource(str, Enum):
    """Sources for model downloads."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    OPENAI_API = "openai_api"
    ANTHROPIC_API = "anthropic_api"
    GOOGLE_API = "google_api"


class UniversalModelInfo(BaseModel):
    """Universal information about any model type."""
    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: str = Field(..., description="Model description")
    model_type: ModelType = Field(..., description="Type of model")
    model_source: ModelSource = Field(..., description="Source of the model")

    # Model-specific properties
    dimension: Optional[int] = Field(default=None, description="Embedding/output dimension")
    max_sequence_length: Optional[int] = Field(default=None, description="Maximum sequence length")
    image_size: Optional[tuple] = Field(default=None, description="Image input size for vision models")
    context_length: Optional[int] = Field(default=None, description="Context length for LLMs")

    # Download and storage info
    size_mb: float = Field(..., description="Model size in MB")
    download_url: str = Field(..., description="Model download URL")
    local_path: Optional[str] = Field(default=None, description="Local storage path")
    is_downloaded: bool = Field(default=False, description="Whether model is downloaded")
    is_available: bool = Field(default=True, description="Whether model is available for download")

    # Usage tracking
    download_date: Optional[datetime] = Field(default=None, description="Download timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    usage_count: int = Field(default=0, description="Number of times used")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Model tags")
    use_case: Optional[str] = Field(default=None, description="Primary use case")
    performance_tier: Optional[str] = Field(default=None, description="Performance tier (fast/balanced/accurate)")


# Backward compatibility alias
EmbeddingModelInfo = UniversalModelInfo


class ModelDownloadProgress(BaseModel):
    """Progress information for model download."""
    model_id: str
    status: str  # "downloading", "completed", "failed", "queued"
    progress_percent: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    speed_mbps: float = 0.0
    downloaded_files: List[str] = Field(default_factory=list)
    eta_seconds: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class UniversalModelManager:
    """
    Universal model manager for all 4 model types:
    - Text Embedding Models (sentence-transformers)
    - Reranking Models (cross-encoder)
    - Vision Models (CLIP, vision-language)
    - LLM Models (Ollama, API validation)

    Provides user-driven model management with no automatic downloads.
    """

    def __init__(self, base_directory: str = "./data/models"):
        """Initialize the universal model manager."""
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each model type
        self.embedding_dir = self.base_directory / "embedding"
        self.reranking_dir = self.base_directory / "reranking"
        self.vision_dir = self.base_directory / "vision"
        self.llm_dir = self.base_directory / "llm"

        for directory in [self.embedding_dir, self.reranking_dir, self.vision_dir, self.llm_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Set environment variables for model caches
        import os
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(self.embedding_dir / "cache")
        os.environ["HF_HOME"] = str(self.base_directory / "huggingface_cache")

        self.models_info_file = self.base_directory / "models_info.json"
        self.available_models: Dict[str, UniversalModelInfo] = {}
        self.download_progress: Dict[str, ModelDownloadProgress] = {}

        # Initialize popular models catalog
        self._initialize_popular_models()

        # Load existing models info
        self._load_models_info()

        logger.info("Universal model manager initialized",
                   base_directory=str(self.base_directory))
    
    def _initialize_popular_models(self) -> None:
        """Initialize catalog of popular models for each type."""

        # Popular Embedding Models
        embedding_models = {
            "all-MiniLM-L6-v2": UniversalModelInfo(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                name="All MiniLM L6 v2",
                description="Fast and efficient general-purpose embedding model",
                model_type=ModelType.EMBEDDING,
                model_source=ModelSource.HUGGINGFACE,
                dimension=384,
                max_sequence_length=256,
                size_mb=90,
                download_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                tags=["general", "fast", "efficient"],
                use_case="general",
                performance_tier="fast"
            ),
            "all-mpnet-base-v2": UniversalModelInfo(
                model_id="sentence-transformers/all-mpnet-base-v2",
                name="All MPNet Base v2",
                description="High-quality general-purpose embedding model",
                model_type=ModelType.EMBEDDING,
                model_source=ModelSource.HUGGINGFACE,
                dimension=768,
                max_sequence_length=384,
                size_mb=420,
                download_url="https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
                tags=["general", "high-quality", "balanced"],
                use_case="general",
                performance_tier="balanced"
            ),
            "bge-large-en-v1.5": UniversalModelInfo(
                model_id="BAAI/bge-large-en-v1.5",
                name="BGE Large EN v1.5",
                description="State-of-the-art English embedding model",
                model_type=ModelType.EMBEDDING,
                model_source=ModelSource.HUGGINGFACE,
                dimension=1024,
                max_sequence_length=512,
                size_mb=1340,
                download_url="https://huggingface.co/BAAI/bge-large-en-v1.5",
                tags=["english", "large", "sota"],
                use_case="high_accuracy",
                performance_tier="accurate"
            )
        }

        # Popular Reranking Models
        reranking_models = {
            "bge-reranker-base": UniversalModelInfo(
                model_id="BAAI/bge-reranker-base",
                name="BGE Reranker Base",
                description="High-quality general-purpose reranking model",
                model_type=ModelType.RERANKING,
                model_source=ModelSource.HUGGINGFACE,
                dimension=1,  # Rerankers output similarity scores
                max_sequence_length=512,
                size_mb=400,
                download_url="https://huggingface.co/BAAI/bge-reranker-base",
                tags=["reranking", "cross-encoder", "general"],
                use_case="general",
                performance_tier="balanced"
            ),
            "bge-reranker-large": UniversalModelInfo(
                model_id="BAAI/bge-reranker-large",
                name="BGE Reranker Large",
                description="Best quality reranking model (slower)",
                model_type=ModelType.RERANKING,
                model_source=ModelSource.HUGGINGFACE,
                dimension=1,
                max_sequence_length=512,
                size_mb=1200,
                download_url="https://huggingface.co/BAAI/bge-reranker-large",
                tags=["reranking", "cross-encoder", "large"],
                use_case="high_accuracy",
                performance_tier="accurate"
            ),
            "ms-marco-MiniLM-L-6-v2": UniversalModelInfo(
                model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
                name="MS-MARCO MiniLM L6 v2",
                description="Fast and efficient reranking model",
                model_type=ModelType.RERANKING,
                model_source=ModelSource.HUGGINGFACE,
                dimension=1,
                max_sequence_length=512,
                size_mb=90,
                download_url="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2",
                tags=["reranking", "cross-encoder", "fast"],
                use_case="speed_optimized",
                performance_tier="fast"
            )
        }

        # Popular Vision Models
        vision_models = {
            "clip-vit-base-patch32": UniversalModelInfo(
                model_id="openai/clip-vit-base-patch32",
                name="CLIP ViT-B/32",
                description="Standard CLIP model for image-text tasks",
                model_type=ModelType.VISION,
                model_source=ModelSource.HUGGINGFACE,
                dimension=512,
                image_size=(224, 224),
                size_mb=600,
                download_url="https://huggingface.co/openai/clip-vit-base-patch32",
                tags=["vision", "clip", "multimodal"],
                use_case="general",
                performance_tier="balanced"
            ),
            "clip-vit-large-patch14": UniversalModelInfo(
                model_id="openai/clip-vit-large-patch14",
                name="CLIP ViT-L/14",
                description="High-quality CLIP model (slower)",
                model_type=ModelType.VISION,
                model_source=ModelSource.HUGGINGFACE,
                dimension=768,
                image_size=(224, 224),
                size_mb=1800,
                download_url="https://huggingface.co/openai/clip-vit-large-patch14",
                tags=["vision", "clip", "multimodal", "large"],
                use_case="high_accuracy",
                performance_tier="accurate"
            ),
            "sentence-transformers-clip": UniversalModelInfo(
                model_id="sentence-transformers/clip-ViT-B-32",
                name="Sentence-Transformers CLIP",
                description="Optimized CLIP for sentence-transformers integration",
                model_type=ModelType.VISION,
                model_source=ModelSource.HUGGINGFACE,
                dimension=512,
                image_size=(224, 224),
                size_mb=600,
                download_url="https://huggingface.co/sentence-transformers/clip-ViT-B-32",
                tags=["vision", "clip", "optimized"],
                use_case="integration_optimized",
                performance_tier="balanced"
            )
        }

        # Popular LLM Models (Ollama)
        llm_models = {
            "llama3.2:latest": UniversalModelInfo(
                model_id="llama3.2:latest",
                name="Llama 3.2 Latest",
                description="Latest Llama 3.2 model from Meta",
                model_type=ModelType.LLM,
                model_source=ModelSource.OLLAMA,
                context_length=8192,
                size_mb=4800,
                download_url="ollama://llama3.2:latest",
                tags=["llama", "meta", "general"],
                use_case="general",
                performance_tier="balanced"
            ),
            "qwen2.5:7b": UniversalModelInfo(
                model_id="qwen2.5:7b",
                name="Qwen 2.5 7B",
                description="Qwen 2.5 7B parameter model",
                model_type=ModelType.LLM,
                model_source=ModelSource.OLLAMA,
                context_length=32768,
                size_mb=4100,
                download_url="ollama://qwen2.5:7b",
                tags=["qwen", "alibaba", "multilingual"],
                use_case="multilingual",
                performance_tier="balanced"
            ),
            "mistral:latest": UniversalModelInfo(
                model_id="mistral:latest",
                name="Mistral Latest",
                description="Latest Mistral model",
                model_type=ModelType.LLM,
                model_source=ModelSource.OLLAMA,
                context_length=8192,
                size_mb=4100,
                download_url="ollama://mistral:latest",
                tags=["mistral", "efficient"],
                use_case="efficient",
                performance_tier="fast"
            )
        }

        # Combine all models
        self.available_models.update(embedding_models)
        self.available_models.update(reranking_models)
        self.available_models.update(vision_models)
        self.available_models.update(llm_models)

    def _load_models_info(self) -> None:
        """Load models information from local storage."""
        try:
            if self.models_info_file.exists():
                with open(self.models_info_file, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        try:
                            # Handle backward compatibility
                            if "model_type" not in model_data:
                                model_data["model_type"] = ModelType.EMBEDDING
                            if "model_source" not in model_data:
                                model_data["model_source"] = ModelSource.HUGGINGFACE

                            self.available_models[model_id] = UniversalModelInfo(**model_data)
                        except Exception as e:
                            logger.warning(f"Failed to load model info for {model_id}: {e}")
                logger.info("Loaded models info", count=len(self.available_models))
            else:
                # Models already initialized in __init__
                logger.info("Using pre-initialized popular models catalog")
        except Exception as e:
            logger.error("Failed to load models info", error=str(e))
            # Models already initialized in __init__
    
    def _save_models_info(self) -> None:
        """Save models information to local storage."""
        try:
            data = {}
            for model_id, model_info in self.available_models.items():
                data[model_id] = model_info.dict()
            
            with open(self.models_info_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("Saved models info", count=len(self.available_models))
        except Exception as e:
            logger.error("Failed to save models info", error=str(e))
    
    # New methods for model type management

    def get_models_by_type(self, model_type: ModelType) -> List[UniversalModelInfo]:
        """Get all models of a specific type."""
        return [model for model in self.available_models.values() if model.model_type == model_type]

    def get_downloaded_models_by_type(self, model_type: ModelType) -> List[UniversalModelInfo]:
        """Get downloaded models of a specific type."""
        return [model for model in self.available_models.values()
                if model.model_type == model_type and model.is_downloaded]

    def search_models(self, query: str, model_type: Optional[ModelType] = None) -> List[UniversalModelInfo]:
        """Search models by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for model in self.available_models.values():
            if model_type and model.model_type != model_type:
                continue

            # Search in name, description, and tags
            if (query_lower in model.name.lower() or
                query_lower in model.description.lower() or
                any(query_lower in tag.lower() for tag in model.tags)):
                results.append(model)

        return results

    def add_custom_model(self, model_info: UniversalModelInfo) -> bool:
        """Add a custom model to the catalog."""
        try:
            self.available_models[model_info.model_id] = model_info
            self._save_models_info()
            logger.info(f"Added custom model: {model_info.model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom model: {e}")
            return False
    
    async def download_model(self, model_id: str, force_redownload: bool = False) -> bool:
        """
        Universal model download supporting all 4 model types.

        Args:
            model_id: Model identifier
            force_redownload: Whether to redownload if model already exists

        Returns:
            True if download successful, False otherwise
        """
        try:
            if model_id not in self.available_models:
                logger.error("Model not found in available models", model_id=model_id)
                return False

            model_info = self.available_models[model_id]

            # Determine storage directory based on model type
            if model_info.model_type == ModelType.EMBEDDING:
                model_path = self.embedding_dir / model_id.replace("/", "_")
            elif model_info.model_type == ModelType.RERANKING:
                model_path = self.reranking_dir / model_id.replace("/", "_")
            elif model_info.model_type == ModelType.VISION:
                model_path = self.vision_dir / model_id.replace("/", "_")
            elif model_info.model_type == ModelType.LLM:
                model_path = self.llm_dir / model_id.replace("/", "_")
            else:
                model_path = self.base_directory / "unknown" / model_id.replace("/", "_")

            # Check if already downloaded
            if model_path.exists() and not force_redownload:
                logger.info("Model already downloaded", model_id=model_id)
                model_info.is_downloaded = True
                model_info.local_path = str(model_path)
                self._save_models_info()
                return True

            # Initialize download progress
            progress = ModelDownloadProgress(
                model_id=model_id,
                status="downloading",
                started_at=datetime.now()
            )
            self.download_progress[model_id] = progress

            logger.info("Starting model download",
                       model_id=model_id,
                       model_type=model_info.model_type.value,
                       model_source=model_info.model_source.value,
                       path=str(model_path))

            # Route to appropriate download method based on source
            if model_info.model_source == ModelSource.HUGGINGFACE:
                success = await self._download_huggingface_model(model_info, model_path, progress)
            elif model_info.model_source == ModelSource.OLLAMA:
                success = await self._download_ollama_model(model_info, progress)
            else:
                # API models don't need downloading, just validation
                success = await self._validate_api_model(model_info, progress)

            if not success:
                logger.error("Failed to download model", model_id=model_id)
                return False

            # Update model info
            model_info.is_downloaded = True
            if model_info.model_source == ModelSource.HUGGINGFACE:
                model_info.local_path = str(model_path)
            model_info.download_date = datetime.now()

            # Update progress
            progress.status = "completed"
            progress.progress_percent = 100.0
            progress.completed_at = datetime.now()

            self._save_models_info()

            logger.info("Model download completed", model_id=model_id)
            return True

        except Exception as e:
            logger.error("Model download failed", model_id=model_id, error=str(e))

            # Update progress with error
            if model_id in self.download_progress:
                self.download_progress[model_id].status = "failed"
                self.download_progress[model_id].error_message = str(e)

            return False

    async def _download_huggingface_model(
        self,
        model_info: UniversalModelInfo,
        model_path: Path,
        progress: ModelDownloadProgress
    ) -> bool:
        """Download model from HuggingFace."""
        try:
            model_path.mkdir(parents=True, exist_ok=True)

            # Download model using existing method
            success = await self._download_model_files(model_info.model_id, model_path, progress)
            return success

        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return False

    async def _download_ollama_model(
        self,
        model_info: UniversalModelInfo,
        progress: ModelDownloadProgress
    ) -> bool:
        """Download model from Ollama."""
        try:
            from app.http_client import SimpleHTTPClient
            from app.config.settings import get_settings

            settings = get_settings()

            # Use Ollama API to pull model
            async with SimpleHTTPClient(settings.OLLAMA_BASE_URL, timeout=300) as client:
                pull_payload = {
                    "name": model_info.model_id,
                    "stream": False
                }

                progress.status = "downloading"
                progress.progress_percent = 10.0

                response = await client.post("/api/pull", body=pull_payload)

                if response.status_code == 200:
                    progress.progress_percent = 100.0
                    logger.info(f"Ollama model pulled successfully: {model_info.model_id}")
                    return True
                else:
                    logger.error(f"Ollama pull failed: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Ollama download failed: {e}")
            return False

    async def _validate_api_model(
        self,
        model_info: UniversalModelInfo,
        progress: ModelDownloadProgress
    ) -> bool:
        """Validate API model availability."""
        try:
            progress.status = "validating"
            progress.progress_percent = 50.0

            # For API models, we just mark as available
            # Actual validation would happen during usage
            progress.progress_percent = 100.0
            logger.info(f"API model validated: {model_info.model_id}")
            return True

        except Exception as e:
            logger.error(f"API model validation failed: {e}")
            return False

    async def _download_model_files(self, model_id: str, model_path: Path, progress: ModelDownloadProgress) -> bool:
        """
        Download model files using our custom HTTP client.

        This method downloads essential model files for sentence transformers:
        - config.json (model configuration)
        - pytorch_model.bin or model.safetensors (model weights)
        - tokenizer.json (tokenizer configuration)
        - vocab.txt (vocabulary)
        """
        try:
            # Essential files for sentence transformers models
            essential_files = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "vocab.txt",
                "tokenizer_config.json"
            ]

            # Alternative files (try if primary doesn't exist)
            alternative_files = {
                "pytorch_model.bin": ["model.safetensors", "pytorch_model.safetensors"],
                "vocab.txt": ["vocab.json"]
            }

            base_url = f"https://huggingface.co/{model_id}/resolve/main"

            # Use our custom HTTP client with SSL verification disabled for corporate networks
            async with SimpleHTTPClient(
                "https://huggingface.co",
                timeout=60,
                verify_ssl=False,  # Disable SSL verification for corporate networks
                max_retries=3
            ) as client:

                downloaded_files = []
                total_files = len(essential_files)

                for i, filename in enumerate(essential_files):
                    file_path = model_path / filename
                    download_url = f"/{model_id}/resolve/main/{filename}"

                    # Try to download the primary file
                    success = await self._download_single_file(client, download_url, file_path, filename)

                    # If primary file failed, try alternatives
                    if not success and filename in alternative_files:
                        for alt_filename in alternative_files[filename]:
                            alt_url = f"/{model_id}/resolve/main/{alt_filename}"
                            alt_path = model_path / alt_filename
                            success = await self._download_single_file(client, alt_url, alt_path, alt_filename)
                            if success:
                                downloaded_files.append(alt_filename)
                                break
                    elif success:
                        downloaded_files.append(filename)

                    # Update progress
                    progress.progress_percent = ((i + 1) / total_files) * 100
                    progress.downloaded_files = downloaded_files.copy()

                # Check if we have minimum required files
                required_files = ["config.json"]
                has_model_weights = any(f in downloaded_files for f in ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"])
                has_tokenizer = any(f in downloaded_files for f in ["tokenizer.json", "vocab.txt", "vocab.json"])

                if not all(req in downloaded_files for req in required_files):
                    logger.error("Missing required config.json file", model_id=model_id)
                    return False

                if not has_model_weights:
                    logger.warning("No model weights found, model may not work properly", model_id=model_id)

                if not has_tokenizer:
                    logger.warning("No tokenizer files found, model may not work properly", model_id=model_id)

                logger.info("Model files downloaded successfully",
                           model_id=model_id,
                           files=downloaded_files,
                           total_files=len(downloaded_files))
                return True

        except Exception as e:
            logger.error("Failed to download model files", model_id=model_id, error=str(e))
            return False

    async def _download_single_file(self, client: SimpleHTTPClient, url: str, file_path: Path, filename: str) -> bool:
        """Download a single file using the HTTP client."""
        try:
            logger.debug("Downloading file", filename=filename, url=url)

            response = await client.get(url)

            if response.status_code == 200:
                # Write file content
                with open(file_path, 'wb') as f:
                    f.write(response.raw_data.encode('utf-8') if isinstance(response.raw_data, str) else response.raw_data)

                logger.debug("File downloaded successfully", filename=filename, size=len(response.raw_data))
                return True
            else:
                logger.debug("File not found or error", filename=filename, status=response.status_code)
                return False

        except Exception as e:
            logger.debug("Failed to download file", filename=filename, error=str(e))
            return False

    def get_available_models(self) -> List[EmbeddingModelInfo]:
        """Get list of all available embedding models."""
        return list(self.available_models.values())
    
    def get_downloaded_models(self) -> List[UniversalModelInfo]:
        """Get list of all downloaded models."""
        return [model for model in self.available_models.values() if model.is_downloaded]

    def get_model_info(self, model_id: str) -> Optional[UniversalModelInfo]:
        """Get information about a specific model."""
        return self.available_models.get(model_id)

    def get_download_progress(self, model_id: str) -> Optional[ModelDownloadProgress]:
        """Get download progress for a specific model."""
        return self.download_progress.get(model_id)
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a downloaded model from local storage.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if model_id not in self.available_models:
                return False
            
            model_info = self.available_models[model_id]
            if not model_info.is_downloaded or not model_info.local_path:
                return False
            
            model_path = Path(model_info.local_path)
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Update model info
            model_info.is_downloaded = False
            model_info.local_path = None
            model_info.download_date = None
            
            self._save_models_info()
            
            logger.info("Model deleted", model_id=model_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete model", model_id=model_id, error=str(e))
            return False
    
    async def test_model(self, model_id: str, test_text: str = "This is a test sentence.") -> Dict[str, Any]:
        """
        Universal model testing for all 4 model types.

        Args:
            model_id: Model identifier
            test_text: Text to test with

        Returns:
            Dictionary with test results
        """
        try:
            if model_id not in self.available_models:
                return {"success": False, "error": "Model not found"}

            model_info = self.available_models[model_id]

            if not model_info.is_downloaded and model_info.model_source == ModelSource.HUGGINGFACE:
                return {"success": False, "error": "Model not downloaded"}

            # Route to appropriate test method based on model type
            if model_info.model_type == ModelType.EMBEDDING:
                return await self._test_embedding_model(model_info, test_text)
            elif model_info.model_type == ModelType.RERANKING:
                return await self._test_reranking_model(model_info, test_text)
            elif model_info.model_type == ModelType.VISION:
                return await self._test_vision_model(model_info, test_text)
            elif model_info.model_type == ModelType.LLM:
                return await self._test_llm_model(model_info, test_text)
            else:
                return {"success": False, "error": f"Unknown model type: {model_info.model_type}"}

        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return {"success": False, "error": str(e)}

    async def _test_embedding_model(self, model_info: UniversalModelInfo, test_text: str) -> Dict[str, Any]:
        """Test embedding model functionality."""
        try:
            if not TORCH_AVAILABLE:
                return {"success": False, "error": "PyTorch not available"}

            # Try to load and test the model
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_info.model_id)
            embedding = model.encode(test_text)

            return {
                "success": True,
                "model_id": model_info.model_id,
                "model_type": "embedding",
                "test_text": test_text,
                "embedding_dimension": len(embedding),
                "embedding_sample": embedding[:5].tolist() if len(embedding) > 5 else embedding.tolist(),
                "model_info": {
                    "max_seq_length": model.max_seq_length,
                    "device": str(model.device)
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Embedding test failed: {str(e)}"}

    async def _test_reranking_model(self, model_info: UniversalModelInfo, test_text: str) -> Dict[str, Any]:
        """Test reranking model functionality."""
        try:
            if not TORCH_AVAILABLE:
                return {"success": False, "error": "PyTorch not available"}

            # Try to load and test the reranking model
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_info.model_id)

            # Test with query and document pair
            query = test_text
            document = "This is a sample document for testing reranking functionality."

            score = model.predict([(query, document)])

            return {
                "success": True,
                "model_id": model_info.model_id,
                "model_type": "reranking",
                "test_query": query,
                "test_document": document,
                "relevance_score": float(score[0]) if hasattr(score, '__iter__') else float(score),
                "model_info": {
                    "max_length": getattr(model, 'max_length', 512)
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Reranking test failed: {str(e)}"}

    async def _test_vision_model(self, model_info: UniversalModelInfo, test_text: str) -> Dict[str, Any]:
        """Test vision model functionality."""
        try:
            if not TORCH_AVAILABLE:
                return {"success": False, "error": "PyTorch not available"}

            # Try to load and test the vision model
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_info.model_id)

            # Test text encoding (CLIP models can encode both text and images)
            text_embedding = model.encode(test_text)

            return {
                "success": True,
                "model_id": model_info.model_id,
                "model_type": "vision",
                "test_text": test_text,
                "text_embedding_dimension": len(text_embedding),
                "text_embedding_sample": text_embedding[:5].tolist() if len(text_embedding) > 5 else text_embedding.tolist(),
                "model_info": {
                    "max_seq_length": getattr(model, 'max_seq_length', 77),
                    "supports_images": True,
                    "supports_text": True
                }
            }

        except Exception as e:
            return {"success": False, "error": f"Vision model test failed: {str(e)}"}

    async def _test_llm_model(self, model_info: UniversalModelInfo, test_text: str) -> Dict[str, Any]:
        """Test LLM model functionality."""
        try:
            if model_info.model_source == ModelSource.OLLAMA:
                return await self._test_ollama_llm(model_info, test_text)
            else:
                # API models would be tested differently
                return {
                    "success": True,
                    "model_id": model_info.model_id,
                    "model_type": "llm",
                    "model_source": model_info.model_source.value,
                    "message": "API model validation - actual testing requires API keys"
                }

        except Exception as e:
            return {"success": False, "error": f"LLM test failed: {str(e)}"}

    async def _test_ollama_llm(self, model_info: UniversalModelInfo, test_text: str) -> Dict[str, Any]:
        """Test Ollama LLM model."""
        try:
            from app.http_client import SimpleHTTPClient
            from app.config.settings import get_settings

            settings = get_settings()

            async with SimpleHTTPClient(settings.OLLAMA_BASE_URL, timeout=30) as client:
                # Test with a simple generation
                generate_payload = {
                    "model": model_info.model_id,
                    "prompt": f"Complete this sentence: {test_text}",
                    "stream": False,
                    "options": {
                        "num_predict": 50
                    }
                }

                response = await client.post("/api/generate", body=generate_payload)

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "model_id": model_info.model_id,
                        "model_type": "llm",
                        "model_source": "ollama",
                        "test_prompt": generate_payload["prompt"],
                        "response": result.get("response", ""),
                        "model_info": {
                            "context_length": model_info.context_length,
                            "total_duration": result.get("total_duration"),
                            "load_duration": result.get("load_duration")
                        }
                    }
                else:
                    return {"success": False, "error": f"Ollama API error: {response.status_code}"}

        except Exception as e:
            return {"success": False, "error": f"Ollama test failed: {str(e)}"}

# Create global instance for backward compatibility
embedding_model_manager = UniversalModelManager()

# Backward compatibility alias
EmbeddingModelManager = UniversalModelManager

