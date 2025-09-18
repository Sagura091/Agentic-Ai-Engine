"""
Embedding Model Download and Management System.

This module provides comprehensive embedding model management including:
- Downloading models from Hugging Face
- Local model storage and caching
- Model validation and testing
- Multiple model support for different use cases
"""

import os
import asyncio
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib

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


class EmbeddingModelInfo(BaseModel):
    """Information about an embedding model."""
    model_id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: str = Field(..., description="Model description")
    dimension: int = Field(..., description="Embedding dimension")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    size_mb: float = Field(..., description="Model size in MB")
    download_url: str = Field(..., description="Hugging Face model URL")
    local_path: Optional[str] = Field(default=None, description="Local storage path")
    is_downloaded: bool = Field(default=False, description="Whether model is downloaded")
    download_date: Optional[datetime] = Field(default=None, description="Download timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    usage_count: int = Field(default=0, description="Number of times used")
    tags: List[str] = Field(default_factory=list, description="Model tags")


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


class EmbeddingModelManager:
    """
    Comprehensive embedding model manager for downloading and managing
    embedding models from Hugging Face.
    """
    
    def __init__(self, models_directory: str = "./data/cache/embedding"):
        """Initialize the embedding model manager."""
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)

        # Set environment variables for sentence-transformers cache
        import os
        models_cache_dir = str(self.models_directory / "models")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_cache_dir
        os.environ["HF_HOME"] = models_cache_dir
        
        self.models_info_file = self.models_directory / "models_info.json"
        self.available_models: Dict[str, EmbeddingModelInfo] = {}
        self.download_progress: Dict[str, ModelDownloadProgress] = {}
        
        # Load existing models info
        self._load_models_info()
        
        logger.info("Embedding model manager initialized", 
                   models_directory=str(self.models_directory))
    
    def _load_models_info(self) -> None:
        """Load models information from local storage."""
        try:
            if self.models_info_file.exists():
                with open(self.models_info_file, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        self.available_models[model_id] = EmbeddingModelInfo(**model_data)
                logger.info("Loaded models info", count=len(self.available_models))
            else:
                # Initialize with default recommended models
                self._initialize_default_models()
        except Exception as e:
            logger.error("Failed to load models info", error=str(e))
            self._initialize_default_models()
    
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
    
    def _initialize_default_models(self) -> None:
        """Initialize with a curated list of recommended embedding models."""
        default_models = [
            {
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "name": "All MiniLM L6 v2",
                "description": "Fast and efficient general-purpose embedding model",
                "dimension": 384,
                "max_sequence_length": 256,
                "size_mb": 90.9,
                "download_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                "tags": ["general", "fast", "small"]
            },
            {
                "model_id": "sentence-transformers/all-mpnet-base-v2",
                "name": "All MPNet Base v2",
                "description": "High-quality general-purpose embedding model",
                "dimension": 768,
                "max_sequence_length": 384,
                "size_mb": 438.0,
                "download_url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
                "tags": ["general", "high-quality"]
            },
            {
                "model_id": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "name": "Multi QA MiniLM L6",
                "description": "Optimized for question-answering and semantic search",
                "dimension": 384,
                "max_sequence_length": 512,
                "size_mb": 90.9,
                "download_url": "https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "tags": ["qa", "search", "fast"]
            },
            {
                "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "name": "Paraphrase Multilingual MiniLM",
                "description": "Multilingual embedding model for 50+ languages",
                "dimension": 384,
                "max_sequence_length": 128,
                "size_mb": 471.0,
                "download_url": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "tags": ["multilingual", "paraphrase"]
            },
            {
                "model_id": "BAAI/bge-small-en-v1.5",
                "name": "BGE Small English v1.5",
                "description": "High-performance small English embedding model",
                "dimension": 384,
                "max_sequence_length": 512,
                "size_mb": 133.0,
                "download_url": "https://huggingface.co/BAAI/bge-small-en-v1.5",
                "tags": ["english", "high-performance", "small"]
            },
            {
                "model_id": "BAAI/bge-base-en-v1.5",
                "name": "BGE Base English v1.5",
                "description": "High-performance base English embedding model",
                "dimension": 768,
                "max_sequence_length": 512,
                "size_mb": 438.0,
                "download_url": "https://huggingface.co/BAAI/bge-base-en-v1.5",
                "tags": ["english", "high-performance", "base"]
            }
        ]
        
        for model_data in default_models:
            model_info = EmbeddingModelInfo(**model_data)
            self.available_models[model_info.model_id] = model_info
        
        self._save_models_info()
        logger.info("Initialized default embedding models", count=len(default_models))

        # Add a fallback model that doesn't require downloading
        fallback_model = EmbeddingModelInfo(
            model_id="fallback/simple-hash",
            name="Simple Hash Fallback",
            description="Simple hash-based embedding for testing and fallback",
            dimension=384,
            max_sequence_length=512,
            size_mb=0.0,
            download_url="local://fallback",
            local_path="fallback",
            is_downloaded=True,
            download_date=datetime.now(),
            tags=["fallback", "local", "testing"]
        )
        self.available_models[fallback_model.model_id] = fallback_model
    
    async def download_model(self, model_id: str, force_redownload: bool = False) -> bool:
        """
        Download an embedding model from Hugging Face.
        
        Args:
            model_id: Model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            force_redownload: Whether to redownload if model already exists
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            if model_id not in self.available_models:
                logger.error("Model not found in available models", model_id=model_id)
                return False
            
            model_info = self.available_models[model_id]
            model_path = self.models_directory / model_id.replace("/", "_")
            
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
            
            logger.info("Starting model download", model_id=model_id, path=str(model_path))

            # Create model directory
            model_path.mkdir(parents=True, exist_ok=True)

            # Download model using our custom HTTP client
            success = await self._download_model_files(model_id, model_path, progress)

            if not success:
                logger.error("Failed to download model files", model_id=model_id)
                return False

            downloaded_path = str(model_path)
            
            # Update model info
            model_info.is_downloaded = True
            model_info.local_path = str(model_path)
            model_info.download_date = datetime.now()
            
            # Update progress
            progress.status = "completed"
            progress.progress_percent = 100.0
            progress.completed_at = datetime.now()
            
            self._save_models_info()
            
            logger.info("Model download completed", model_id=model_id, path=downloaded_path)
            return True
            
        except Exception as e:
            logger.error("Model download failed", model_id=model_id, error=str(e))
            
            # Update progress with error
            if model_id in self.download_progress:
                self.download_progress[model_id].status = "failed"
                self.download_progress[model_id].error_message = str(e)
            
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
    
    def get_downloaded_models(self) -> List[EmbeddingModelInfo]:
        """Get list of downloaded embedding models."""
        return [model for model in self.available_models.values() if model.is_downloaded]
    
    def get_model_info(self, model_id: str) -> Optional[EmbeddingModelInfo]:
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
        Test an embedding model with sample text.
        
        Args:
            model_id: Model identifier
            test_text: Text to test with
            
        Returns:
            Test results including embedding dimension and performance metrics
        """
        try:
            if model_id not in self.available_models:
                return {"success": False, "error": "Model not found"}

            model_info = self.available_models[model_id]
            if not model_info.is_downloaded:
                return {"success": False, "error": "Model not downloaded"}

            # Handle fallback model
            if model_id == "fallback/simple-hash":
                start_time = datetime.now()
                load_time = (datetime.now() - start_time).total_seconds()

                # Generate simple hash-based embedding
                inference_start = datetime.now()
                text_hash = hashlib.md5(test_text.encode()).hexdigest()
                embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), 32), 2)]
                while len(embedding) < model_info.dimension:
                    embedding.append(0.0)
                embedding = embedding[:model_info.dimension]
                inference_time = (datetime.now() - inference_start).total_seconds()

                return {
                    "success": True,
                    "embedding_dimension": len(embedding),
                    "load_time_seconds": load_time,
                    "inference_time_seconds": inference_time,
                    "embedding_sample": embedding[:5],  # First 5 values
                    "model_type": "fallback"
                }

            # Import here to avoid startup issues
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return {"success": False, "error": "sentence_transformers not available"}

            # Load and test model
            start_time = datetime.now()
            model = SentenceTransformer(model_info.local_path)
            load_time = (datetime.now() - start_time).total_seconds()

            # Generate embedding
            start_time = datetime.now()
            embedding = model.encode([test_text])
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Update usage statistics
            model_info.last_used = datetime.now()
            model_info.usage_count += 1
            self._save_models_info()
            
            return {
                "success": True,
                "model_id": model_id,
                "embedding_dimension": len(embedding[0]),
                "load_time_seconds": load_time,
                "inference_time_seconds": inference_time,
                "test_text": test_text,
                "embedding_sample": embedding[0][:5].tolist()  # First 5 dimensions
            }
            
        except Exception as e:
            logger.error("Model test failed", model_id=model_id, error=str(e))
            return {"success": False, "error": str(e)}

    def get_global_config(self) -> Dict[str, Any]:
        """Get global embedding configuration."""
        config_file = self.models_directory / "global_config.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load global config: {e}")

        # Return default configuration
        return {
            "embedding_engine": "",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_batch_size": 32,
            "openai_url": "https://api.openai.com/v1",
            "openai_key": "",
            "ollama_url": "http://localhost:11434",
            "ollama_key": "",
            "azure_url": "",
            "azure_key": "",
            "azure_version": "2023-05-15"
        }

    def update_global_config(self, config: Dict[str, Any]) -> None:
        """Update global embedding configuration."""
        config_file = self.models_directory / "global_config.json"

        try:
            # Ensure models directory exists
            self.models_directory.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info("Global embedding configuration updated")

            # Notify global embedding manager to reload configuration
            try:
                import asyncio
                from .global_embedding_manager import reload_global_embedding_config

                # Schedule reload in the background
                loop = asyncio.get_event_loop()
                loop.create_task(reload_global_embedding_config())

            except Exception as reload_error:
                logger.warning(f"Failed to reload global embedding config: {reload_error}")

        except Exception as e:
            logger.error(f"Failed to save global config: {e}")
            raise

    async def test_openai_connection(self, url: str, key: str, model: str, test_text: str) -> bool:
        """Test OpenAI embedding connection."""
        try:
            import openai

            client = openai.OpenAI(api_key=key, base_url=url)

            response = client.embeddings.create(
                model=model,
                input=test_text
            )

            return len(response.data) > 0 and len(response.data[0].embedding) > 0

        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False

    async def test_azure_connection(self, url: str, key: str, version: str, model: str, test_text: str) -> bool:
        """Test Azure OpenAI embedding connection."""
        try:
            import openai

            client = openai.AzureOpenAI(
                api_key=key,
                azure_endpoint=url,
                api_version=version
            )

            response = client.embeddings.create(
                model=model,
                input=test_text
            )

            return len(response.data) > 0 and len(response.data[0].embedding) > 0

        except Exception as e:
            logger.error(f"Azure OpenAI connection test failed: {e}")
            return False

    async def test_ollama_connection(self, url: str, key: str, model: str, test_text: str) -> bool:
        """Test Ollama embedding connection."""
        try:
            import httpx

            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{url}/api/embeddings",
                    headers=headers,
                    json={
                        "model": model,
                        "prompt": test_text
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    return "embedding" in data and len(data["embedding"]) > 0

                return False

        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False

    async def test_default_connection(self, model: str, test_text: str) -> bool:
        """Test default sentence transformers connection."""
        try:
            from sentence_transformers import SentenceTransformer

            # Try to load the model
            model_instance = SentenceTransformer(model)

            # Test embedding generation
            embeddings = model_instance.encode([test_text])

            return len(embeddings) > 0 and len(embeddings[0]) > 0

        except Exception as e:
            logger.error(f"Default connection test failed: {e}")
            return False


# Global instance
embedding_model_manager = EmbeddingModelManager()
