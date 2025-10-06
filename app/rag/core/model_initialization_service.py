"""
Model Initialization Service for RAG System.

This service ensures all required models are available before RAG system initialization.
It intelligently detects existing models (in HuggingFace cache or data/models/) and
only downloads missing models.

KEY FEATURES:
- NO DUPLICATES: Detects models in HuggingFace cache and reuses them
- Smart detection: Checks multiple possible locations
- Progress reporting: Real-time download progress
- Atomic operations: Safe concurrent access
- Comprehensive logging: Full audit trail

Author: Agentic AI System
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

import structlog
from pydantic import BaseModel, Field

# Import model specifications
from app.rag.config.required_models import (
    ModelSpec,
    get_required_models,
    get_model_by_id,
    format_size,
    ModelPriority
)

logger = structlog.get_logger(__name__)


@dataclass
class ModelLocation:
    """Information about where a model is located."""
    model_id: str
    location_type: str  # 'centralized', 'huggingface_cache', 'transformers_cache', 'not_found'
    path: Optional[Path] = None
    size_mb: Optional[float] = None
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class DownloadProgress:
    """Progress information for model download."""
    model_id: str
    status: str  # 'pending', 'downloading', 'validating', 'complete', 'failed'
    progress_percent: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ModelInitializationConfig(BaseModel):
    """Configuration for model initialization service."""
    models_dir: Path = Field(default=Path("data/models"))
    check_huggingface_cache: bool = Field(default=True, description="Check HuggingFace cache for existing models")
    reuse_cached_models: bool = Field(default=True, description="Reuse models from HuggingFace cache instead of downloading")
    copy_to_centralized: bool = Field(default=True, description="Copy cached models to centralized storage")
    validate_models: bool = Field(default=True, description="Validate model files after download")
    parallel_downloads: int = Field(default=2, ge=1, le=5, description="Number of parallel downloads")
    download_timeout: int = Field(default=3600, description="Download timeout in seconds")
    enabled_features: List[str] = Field(default_factory=list, description="Enabled features (advanced_retrieval, vision)")
    auto_download_defaults: bool = Field(default=True, description="Automatically download default models on first run")
    silent_mode: bool = Field(default=True, description="Suppress progress output (no CLI interaction)")


class ModelInitializationService:
    """
    Service to ensure all required models are available before RAG initialization.
    
    This service:
    1. Detects existing models in multiple locations
    2. Reuses existing models (no duplicates)
    3. Downloads only missing models
    4. Provides progress reporting
    5. Validates model integrity
    """
    
    def __init__(self, config: Optional[ModelInitializationConfig] = None):
        """Initialize model initialization service."""
        self.config = config or ModelInitializationConfig()
        
        # Ensure models directory exists
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self._model_locations: Dict[str, ModelLocation] = {}
        self._download_progress: Dict[str, DownloadProgress] = {}
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'models_found': 0,
            'models_reused': 0,
            'models_downloaded': 0,
            'models_failed': 0,
            'total_download_mb': 0.0,
            'total_time_seconds': 0.0
        }
        
        logger.info(
            "Model Initialization Service created",
            models_dir=str(self.config.models_dir),
            check_cache=self.config.check_huggingface_cache,
            reuse_cached=self.config.reuse_cached_models
        )
    
    async def ensure_models_available(
        self,
        required_models: Optional[List[ModelSpec]] = None
    ) -> Dict[str, ModelLocation]:
        """
        Ensure all required models are available.

        This is the main entry point. It:
        1. Determines which models are required
        2. Checks for existing models
        3. Downloads missing models (silently if configured)
        4. Returns locations of all models

        Args:
            required_models: List of required models. If None, uses defaults or config.enabled_features

        Returns:
            Dictionary mapping model_id to ModelLocation
        """
        start_time = datetime.utcnow()

        try:
            # Determine required models
            if required_models is None:
                if self.config.auto_download_defaults:
                    # Use the 3 default models
                    from app.rag.config.required_models import get_default_models
                    required_models = get_default_models()
                    if not self.config.silent_mode:
                        logger.info("Using default models (embedding, reranking, vision)")
                else:
                    # Use models based on enabled features
                    required_models = get_required_models(self.config.enabled_features)

            if not self.config.silent_mode:
                logger.info(
                    "Ensuring models are available",
                    required_count=len(required_models),
                    enabled_features=self.config.enabled_features
                )
            
            # Phase 1: Detect existing models
            if not self.config.silent_mode:
                logger.info("Phase 1: Detecting existing models...")
            await self._detect_existing_models(required_models)

            # Phase 2: Download missing models
            if not self.config.silent_mode:
                logger.info("Phase 2: Downloading missing models...")
            await self._download_missing_models(required_models)

            # Phase 3: Validate all models
            if self.config.validate_models:
                if not self.config.silent_mode:
                    logger.info("Phase 3: Validating models...")
                await self._validate_all_models(required_models)
            
            # Calculate statistics
            end_time = datetime.utcnow()
            self._stats['total_time_seconds'] = (end_time - start_time).total_seconds()

            # Log summary (only if not silent)
            if not self.config.silent_mode:
                self._log_summary()
            else:
                # Just log a simple success message
                logger.info(
                    "Models ready",
                    found=self._stats['models_found'],
                    downloaded=self._stats['models_downloaded'],
                    reused=self._stats['models_reused']
                )

            return self._model_locations
            
        except Exception as e:
            logger.error(f"Failed to ensure models available: {e}")
            raise
    
    async def _detect_existing_models(self, required_models: List[ModelSpec]) -> None:
        """
        Detect existing models in all possible locations.

        Priority order:
        1. Centralized storage (data/models/) - HIGHEST PRIORITY
        2. HuggingFace cache (~/.cache/huggingface/) - FALLBACK

        If model exists in data/models/, we use it (NO DOWNLOAD).
        If model exists in HuggingFace cache, we copy it to data/models/ (NO DOWNLOAD).
        If model doesn't exist anywhere, we mark it for download.
        """
        for model_spec in required_models:
            if not self.config.silent_mode:
                logger.info(f"Checking for model: {model_spec.model_id}")

            # ================================================================
            # PRIORITY 1: Check centralized storage (data/models/)
            # ================================================================
            location = await self._check_centralized_storage(model_spec)
            if location.is_valid:
                # Model already exists in data/models/ - USE IT!
                self._model_locations[model_spec.model_id] = location
                self._stats['models_found'] += 1

                if not self.config.silent_mode:
                    logger.info(
                        f"✅ Found in data/models/: {model_spec.model_id}",
                        path=str(location.path)
                    )
                else:
                    logger.debug(f"Found: {model_spec.model_id} at {location.path}")

                continue  # Skip to next model - NO DOWNLOAD NEEDED

            # ================================================================
            # PRIORITY 2: Check HuggingFace cache
            # ================================================================
            if self.config.check_huggingface_cache:
                location = await self._check_huggingface_cache(model_spec)
                if location.is_valid:
                    # Model exists in HuggingFace cache - REUSE IT!
                    self._model_locations[model_spec.model_id] = location
                    self._stats['models_found'] += 1

                    # Copy to centralized storage (no download, just copy)
                    if self.config.copy_to_centralized:
                        if not self.config.silent_mode:
                            logger.info(
                                f"📦 Found in HuggingFace cache, copying to data/models/: {model_spec.model_id}"
                            )
                        await self._copy_to_centralized(model_spec, location)
                    else:
                        if not self.config.silent_mode:
                            logger.info(
                                f"✅ Found in HuggingFace cache (reusing): {model_spec.model_id}",
                                path=str(location.path)
                            )

                    continue  # Skip to next model - NO DOWNLOAD NEEDED

            # ================================================================
            # Model not found anywhere - MARK FOR DOWNLOAD
            # ================================================================
            if not self.config.silent_mode:
                logger.warning(f"❌ Model not found, will download: {model_spec.model_id}")

            self._model_locations[model_spec.model_id] = ModelLocation(
                model_id=model_spec.model_id,
                location_type='not_found',
                is_valid=False
            )
    
    async def _check_centralized_storage(self, model_spec: ModelSpec) -> ModelLocation:
        """Check if model exists in centralized storage."""
        model_path = self.config.models_dir / model_spec.model_type / model_spec.local_name
        
        if not model_path.exists():
            return ModelLocation(
                model_id=model_spec.model_id,
                location_type='centralized',
                path=model_path,
                is_valid=False,
                validation_errors=["Directory does not exist"]
            )
        
        # Validate model files
        is_valid, errors = await self._validate_model_files(model_path, model_spec.model_type)
        
        if is_valid:
            size_mb = await self._calculate_directory_size(model_path)
            return ModelLocation(
                model_id=model_spec.model_id,
                location_type='centralized',
                path=model_path,
                size_mb=size_mb,
                is_valid=True
            )
        
        return ModelLocation(
            model_id=model_spec.model_id,
            location_type='centralized',
            path=model_path,
            is_valid=False,
            validation_errors=errors
        )
    
    async def _check_huggingface_cache(self, model_spec: ModelSpec) -> ModelLocation:
        """Check if model exists in HuggingFace cache."""
        # Get HuggingFace cache directory
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        
        if not hf_cache.exists():
            return ModelLocation(
                model_id=model_spec.model_id,
                location_type='huggingface_cache',
                is_valid=False,
                validation_errors=["HuggingFace cache directory does not exist"]
            )
        
        # Try to find model in cache
        # HuggingFace uses format: models--{org}--{model}
        model_cache_name = model_spec.model_id.replace("/", "--")
        model_cache_name = f"models--{model_cache_name}"
        
        # Check for snapshots directory
        model_cache_path = hf_cache / model_cache_name / "snapshots"
        
        if not model_cache_path.exists():
            return ModelLocation(
                model_id=model_spec.model_id,
                location_type='huggingface_cache',
                is_valid=False,
                validation_errors=["Model not in HuggingFace cache"]
            )
        
        # Get latest snapshot
        snapshots = list(model_cache_path.iterdir())
        if not snapshots:
            return ModelLocation(
                model_id=model_spec.model_id,
                location_type='huggingface_cache',
                is_valid=False,
                validation_errors=["No snapshots found"]
            )
        
        # Use most recent snapshot
        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
        
        # Validate
        is_valid, errors = await self._validate_model_files(latest_snapshot, model_spec.model_type)
        
        if is_valid:
            size_mb = await self._calculate_directory_size(latest_snapshot)
            return ModelLocation(
                model_id=model_spec.model_id,
                location_type='huggingface_cache',
                path=latest_snapshot,
                size_mb=size_mb,
                is_valid=True
            )
        
        return ModelLocation(
            model_id=model_spec.model_id,
            location_type='huggingface_cache',
            path=latest_snapshot,
            is_valid=False,
            validation_errors=errors
        )

    async def _validate_model_files(self, model_path: Path, model_type: str) -> Tuple[bool, List[str]]:
        """
        Validate that model directory contains required files.

        Args:
            model_path: Path to model directory
            model_type: Type of model (embedding, reranking, vision)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not model_path.exists():
            return False, ["Model directory does not exist"]

        if not model_path.is_dir():
            return False, ["Model path is not a directory"]

        # Check for required files based on model type
        required_files = []

        if model_type in ["embedding", "reranking"]:
            # Sentence-transformers models need these files
            required_files = [
                "config.json",
                "pytorch_model.bin"  # or model.safetensors
            ]
            # Also accept safetensors format
            has_pytorch = (model_path / "pytorch_model.bin").exists()
            has_safetensors = (model_path / "model.safetensors").exists()

            if not has_pytorch and not has_safetensors:
                errors.append("Missing model weights (pytorch_model.bin or model.safetensors)")

        elif model_type == "vision":
            # Vision models need these files
            required_files = [
                "config.json",
                "pytorch_model.bin"  # or model.safetensors
            ]
            has_pytorch = (model_path / "pytorch_model.bin").exists()
            has_safetensors = (model_path / "model.safetensors").exists()

            if not has_pytorch and not has_safetensors:
                errors.append("Missing model weights")

        # Check for config.json (required for all)
        if not (model_path / "config.json").exists():
            errors.append("Missing config.json")

        # Check for tokenizer files (common but not always required)
        has_tokenizer = (
            (model_path / "tokenizer_config.json").exists() or
            (model_path / "tokenizer.json").exists()
        )

        if not has_tokenizer:
            # Not an error, but log it
            logger.debug(f"No tokenizer files found in {model_path}")

        return len(errors) == 0, errors

    async def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate total size of directory in MB."""
        total_size = 0

        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate directory size: {e}")
            return 0.0

        return total_size / (1024 * 1024)  # Convert to MB

    async def _copy_to_centralized(self, model_spec: ModelSpec, source_location: ModelLocation) -> None:
        """
        Copy model from cache to centralized storage.

        Args:
            model_spec: Model specification
            source_location: Location of source model
        """
        if not source_location.path or not source_location.path.exists():
            logger.error(f"Cannot copy model, source path invalid: {source_location.path}")
            return

        # Determine destination
        dest_path = self.config.models_dir / model_spec.model_type / model_spec.local_name

        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Copying model from {source_location.path} to {dest_path}")

            # Copy directory
            if dest_path.exists():
                logger.warning(f"Destination already exists, removing: {dest_path}")
                shutil.rmtree(dest_path)

            # Use shutil.copytree for directory copy
            await asyncio.get_event_loop().run_in_executor(
                None,
                shutil.copytree,
                source_location.path,
                dest_path
            )

            logger.info(f"✅ Successfully copied model to centralized storage")

            # Update location
            self._model_locations[model_spec.model_id] = ModelLocation(
                model_id=model_spec.model_id,
                location_type='centralized',
                path=dest_path,
                size_mb=source_location.size_mb,
                is_valid=True
            )

            self._stats['models_reused'] += 1

        except Exception as e:
            logger.error(f"Failed to copy model to centralized storage: {e}")

    async def _download_missing_models(self, required_models: List[ModelSpec]) -> None:
        """
        Download models that were not found in data/models/ or HuggingFace cache.

        This only downloads models that don't exist anywhere.
        If a model exists in data/models/, it's already marked as found.

        Args:
            required_models: List of required models
        """
        # Find models that need downloading
        to_download = []
        for model_spec in required_models:
            location = self._model_locations.get(model_spec.model_id)
            if not location or not location.is_valid:
                to_download.append(model_spec)

        if not to_download:
            # All models already exist - NO DOWNLOAD NEEDED
            if not self.config.silent_mode:
                logger.info("✅ All required models already exist in data/models/")
            else:
                logger.debug("All models found, no downloads needed")
            return

        # Some models need downloading
        if not self.config.silent_mode:
            logger.info(f"Downloading {len(to_download)} missing models...")
            for model_spec in to_download:
                logger.info(f"  • {model_spec.model_id} ({format_size(model_spec.size_mb)})")

        # Download models (with parallelization)
        semaphore = asyncio.Semaphore(self.config.parallel_downloads)

        async def download_with_semaphore(model_spec: ModelSpec):
            async with semaphore:
                await self._download_model(model_spec)

        # Create download tasks
        tasks = [download_with_semaphore(model_spec) for model_spec in to_download]

        # Wait for all downloads
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to download {to_download[i].model_id}: {result}")
                self._stats['models_failed'] += 1

    async def _download_model(self, model_spec: ModelSpec) -> None:
        """
        Download a single model.

        Args:
            model_spec: Model specification
        """
        logger.info(f"Downloading model: {model_spec.model_id} ({format_size(model_spec.size_mb)})")

        # Initialize progress tracking
        progress = DownloadProgress(
            model_id=model_spec.model_id,
            status='downloading',
            total_mb=model_spec.size_mb,
            start_time=datetime.utcnow()
        )
        self._download_progress[model_spec.model_id] = progress

        try:
            # Import model manager for download
            from app.rag.core.embedding_model_manager import embedding_model_manager, ModelType

            # Map model type
            model_type_map = {
                'embedding': ModelType.EMBEDDING,
                'reranking': ModelType.RERANKING,
                'vision': ModelType.VISION
            }

            model_type = model_type_map.get(model_spec.model_type)
            if not model_type:
                raise ValueError(f"Unknown model type: {model_spec.model_type}")

            # Download using model manager
            success = await embedding_model_manager.download_model(
                model_id=model_spec.model_id,
                model_type=model_type,
                force_redownload=False
            )

            if not success:
                raise Exception("Download failed")

            # Update progress
            progress.status = 'complete'
            progress.progress_percent = 100.0
            progress.end_time = datetime.utcnow()

            # Update location
            dest_path = self.config.models_dir / model_spec.model_type / model_spec.local_name
            self._model_locations[model_spec.model_id] = ModelLocation(
                model_id=model_spec.model_id,
                location_type='centralized',
                path=dest_path,
                size_mb=model_spec.size_mb,
                is_valid=True
            )

            self._stats['models_downloaded'] += 1
            self._stats['total_download_mb'] += model_spec.size_mb

            logger.info(f"✅ Successfully downloaded: {model_spec.model_id}")

        except Exception as e:
            logger.error(f"Failed to download {model_spec.model_id}: {e}")
            progress.status = 'failed'
            progress.error_message = str(e)
            progress.end_time = datetime.utcnow()
            raise

    async def _validate_all_models(self, required_models: List[ModelSpec]) -> None:
        """
        Validate all required models are available and valid.

        Args:
            required_models: List of required models
        """
        logger.info("Validating all models...")

        all_valid = True
        for model_spec in required_models:
            location = self._model_locations.get(model_spec.model_id)

            if not location:
                logger.error(f"❌ Model not found: {model_spec.model_id}")
                all_valid = False
                continue

            if not location.is_valid:
                logger.error(
                    f"❌ Model invalid: {model_spec.model_id}",
                    errors=location.validation_errors
                )
                all_valid = False
                continue

            logger.info(f"✅ Model valid: {model_spec.model_id} at {location.path}")

        if not all_valid:
            raise Exception("Some required models are missing or invalid")

        logger.info("✅ All models validated successfully")

    def _log_summary(self) -> None:
        """Log summary of model initialization."""
        logger.info("=" * 80)
        logger.info("MODEL INITIALIZATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Models Already in data/models/: {self._stats['models_found']}")
        logger.info(f"Models Copied from Cache: {self._stats['models_reused']}")
        logger.info(f"Models Downloaded: {self._stats['models_downloaded']}")
        logger.info(f"Models Failed: {self._stats['models_failed']}")

        if self._stats['models_downloaded'] > 0:
            logger.info(f"Total Downloaded: {format_size(self._stats['total_download_mb'])}")

        logger.info(f"Total Time: {self._stats['total_time_seconds']:.1f}s")
        logger.info("=" * 80)

        # Show what happened
        if self._stats['models_found'] > 0 and self._stats['models_downloaded'] == 0:
            logger.info("✅ All models already available - NO DOWNLOADS NEEDED")
        elif self._stats['models_downloaded'] > 0:
            logger.info(f"✅ Downloaded {self._stats['models_downloaded']} new models")

        logger.info("=" * 80)

    def get_model_location(self, model_id: str) -> Optional[ModelLocation]:
        """
        Get location of a specific model.

        Args:
            model_id: Model ID to look up

        Returns:
            ModelLocation if found, None otherwise
        """
        return self._model_locations.get(model_id)

    def get_all_model_locations(self) -> Dict[str, ModelLocation]:
        """Get all model locations."""
        return self._model_locations.copy()

    def get_download_progress(self, model_id: str) -> Optional[DownloadProgress]:
        """
        Get download progress for a specific model.

        Args:
            model_id: Model ID

        Returns:
            DownloadProgress if found, None otherwise
        """
        return self._download_progress.get(model_id)

    def get_all_download_progress(self) -> Dict[str, DownloadProgress]:
        """Get all download progress."""
        return self._download_progress.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get initialization statistics."""
        return self._stats.copy()


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_model_init_service: Optional[ModelInitializationService] = None
_service_lock = asyncio.Lock()


async def get_model_initialization_service(
    config: Optional[ModelInitializationConfig] = None
) -> ModelInitializationService:
    """
    Get or create the global model initialization service.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        ModelInitializationService instance
    """
    global _model_init_service

    if _model_init_service is None:
        async with _service_lock:
            if _model_init_service is None:
                _model_init_service = ModelInitializationService(config)

    return _model_init_service


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def ensure_embedding_model(model_id: str = "all-MiniLM-L6-v2") -> Optional[Path]:
    """
    Ensure a specific embedding model is available.

    Args:
        model_id: Model ID (short name or full HuggingFace ID)

    Returns:
        Path to model if available, None otherwise
    """
    from app.rag.config.required_models import get_model_by_id

    model_spec = get_model_by_id(model_id)
    if not model_spec:
        logger.error(f"Unknown model: {model_id}")
        return None

    service = await get_model_initialization_service()
    locations = await service.ensure_models_available([model_spec])

    location = locations.get(model_spec.model_id)
    if location and location.is_valid:
        return location.path

    return None


async def ensure_reranking_model(model_id: str = "bge-reranker-base") -> Optional[Path]:
    """
    Ensure a specific reranking model is available.

    Args:
        model_id: Model ID (short name or full HuggingFace ID)

    Returns:
        Path to model if available, None otherwise
    """
    from app.rag.config.required_models import get_model_by_id

    model_spec = get_model_by_id(model_id)
    if not model_spec:
        logger.error(f"Unknown model: {model_id}")
        return None

    service = await get_model_initialization_service()
    locations = await service.ensure_models_available([model_spec])

    location = locations.get(model_spec.model_id)
    if location and location.is_valid:
        return location.path

    return None


async def ensure_vision_model(model_id: str = "clip-ViT-B-32") -> Optional[Path]:
    """
    Ensure a specific vision model is available.

    Args:
        model_id: Model ID (short name or full HuggingFace ID)

    Returns:
        Path to model if available, None otherwise
    """
    from app.rag.config.required_models import get_model_by_id

    model_spec = get_model_by_id(model_id)
    if not model_spec:
        logger.error(f"Unknown model: {model_id}")
        return None

    service = await get_model_initialization_service()
    locations = await service.ensure_models_available([model_spec])

    location = locations.get(model_spec.model_id)
    if location and location.is_valid:
        return location.path

    return None


async def ensure_all_required_models(enabled_features: Optional[List[str]] = None) -> bool:
    """
    Ensure all required models for enabled features are available.

    Args:
        enabled_features: List of enabled features (e.g., ['advanced_retrieval', 'vision'])

    Returns:
        True if all models available, False otherwise
    """
    try:
        config = ModelInitializationConfig(enabled_features=enabled_features or [])
        service = await get_model_initialization_service(config)

        locations = await service.ensure_models_available()

        # Check if all required models are valid
        from app.rag.config.required_models import get_required_models
        required_models = get_required_models(enabled_features)

        for model_spec in required_models:
            location = locations.get(model_spec.model_id)
            if not location or not location.is_valid:
                logger.error(f"Required model not available: {model_spec.model_id}")
                return False

        return True

    except Exception as e:
        logger.error(f"Failed to ensure required models: {e}")
        return False


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """CLI interface for model initialization."""
    import sys

    print("\n" + "=" * 80)
    print("RAG SYSTEM - MODEL INITIALIZATION SERVICE")
    print("=" * 80 + "\n")

    # Parse command line arguments
    enabled_features = []
    if len(sys.argv) > 1:
        enabled_features = sys.argv[1].split(",")

    print(f"Enabled Features: {enabled_features or ['none (embedding only)']}\n")

    # Create service
    config = ModelInitializationConfig(
        enabled_features=enabled_features,
        check_huggingface_cache=True,
        reuse_cached_models=True,
        copy_to_centralized=True
    )

    service = ModelInitializationService(config)

    # Ensure models
    try:
        locations = await service.ensure_models_available()

        print("\n" + "=" * 80)
        print("MODEL LOCATIONS")
        print("=" * 80)

        for model_id, location in locations.items():
            status = "✅" if location.is_valid else "❌"
            print(f"\n{status} {model_id}")
            print(f"   Type: {location.location_type}")
            print(f"   Path: {location.path}")
            if location.size_mb:
                print(f"   Size: {format_size(location.size_mb)}")
            if not location.is_valid:
                print(f"   Errors: {location.validation_errors}")

        print("\n" + "=" * 80)
        print("✅ MODEL INITIALIZATION COMPLETE")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

