"""
Required Models Configuration for RAG System.

This module defines all models required by the RAG system and their configurations.
Models are downloaded to data/models/ and reused across all components.

NO DUPLICATES: If a model exists anywhere (HuggingFace cache or data/models/),
we detect it and point to the existing location.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class ModelPriority(str, Enum):
    """Model priority levels."""
    REQUIRED = "required"  # Must be downloaded for system to work
    RECOMMENDED = "recommended"  # Should be downloaded for full functionality
    OPTIONAL = "optional"  # Only download if explicitly enabled


@dataclass
class ModelSpec:
    """Specification for a model."""
    model_id: str  # HuggingFace model ID
    model_type: str  # embedding, reranking, vision
    priority: ModelPriority
    size_mb: float
    description: str
    local_name: str  # Directory name in data/models/{type}/
    dimension: Optional[int] = None
    requires_feature: Optional[str] = None  # Feature flag that enables this model
    alternative_ids: List[str] = None  # Alternative HuggingFace IDs to check
    
    def __post_init__(self):
        if self.alternative_ids is None:
            self.alternative_ids = []


# ============================================================================
# EMBEDDING MODELS
# ============================================================================

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": ModelSpec(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        model_type="embedding",
        priority=ModelPriority.REQUIRED,
        size_mb=90,
        description="Default text embedding model - fast and efficient",
        local_name="all_MiniLM_L6_v2",
        dimension=384,
        alternative_ids=[
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
    ),
    "bge-base-en-v1.5": ModelSpec(
        model_id="BAAI/bge-base-en-v1.5",
        model_type="embedding",
        priority=ModelPriority.OPTIONAL,
        size_mb=440,
        description="High-quality embedding model for better retrieval",
        local_name="bge_base_en_v1_5",
        dimension=768,
        alternative_ids=[
            "bge-base-en-v1.5",
            "BAAI/bge-base-en-v1.5"
        ]
    ),
    "bge-large-en-v1.5": ModelSpec(
        model_id="BAAI/bge-large-en-v1.5",
        model_type="embedding",
        priority=ModelPriority.OPTIONAL,
        size_mb=1340,
        description="Highest quality embedding model",
        local_name="bge_large_en_v1_5",
        dimension=1024,
        alternative_ids=[
            "bge-large-en-v1.5",
            "BAAI/bge-large-en-v1.5"
        ]
    )
}


# ============================================================================
# RERANKING MODELS
# ============================================================================

RERANKING_MODELS = {
    "bge-reranker-base": ModelSpec(
        model_id="BAAI/bge-reranker-base",
        model_type="reranking",
        priority=ModelPriority.RECOMMENDED,
        size_mb=280,
        description="Default cross-encoder reranking model",
        local_name="bge_reranker_base",
        requires_feature="advanced_retrieval",
        alternative_ids=[
            "bge-reranker-base",
            "BAAI/bge-reranker-base"
        ]
    ),
    "bge-reranker-large": ModelSpec(
        model_id="BAAI/bge-reranker-large",
        model_type="reranking",
        priority=ModelPriority.OPTIONAL,
        size_mb=560,
        description="High-quality reranking model",
        local_name="bge_reranker_large",
        requires_feature="advanced_retrieval",
        alternative_ids=[
            "bge-reranker-large",
            "BAAI/bge-reranker-large"
        ]
    ),
    "ms-marco-MiniLM-L-6-v2": ModelSpec(
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_type="reranking",
        priority=ModelPriority.OPTIONAL,
        size_mb=90,
        description="Fast reranking model",
        local_name="ms_marco_MiniLM_L_6_v2",
        requires_feature="advanced_retrieval",
        alternative_ids=[
            "ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ]
    ),
    "ms-marco-MiniLM-L-12-v2": ModelSpec(
        model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
        model_type="reranking",
        priority=ModelPriority.OPTIONAL,
        size_mb=130,
        description="Balanced reranking model",
        local_name="ms_marco_MiniLM_L_12_v2",
        requires_feature="advanced_retrieval",
        alternative_ids=[
            "ms-marco-MiniLM-L-12-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2"
        ]
    )
}


# ============================================================================
# VISION MODELS
# ============================================================================

VISION_MODELS = {
    "clip-ViT-B-32": ModelSpec(
        model_id="openai/clip-vit-base-patch32",
        model_type="vision",
        priority=ModelPriority.OPTIONAL,
        size_mb=600,
        description="CLIP vision-text model - base version",
        local_name="clip_vit_base_patch32",
        dimension=512,
        requires_feature="vision",
        alternative_ids=[
            "clip-vit-base-patch32",
            "openai/clip-vit-base-patch32"
        ]
    ),
    "clip-ViT-L-14": ModelSpec(
        model_id="sentence-transformers/clip-ViT-L-14",
        model_type="vision",
        priority=ModelPriority.OPTIONAL,
        size_mb=1700,
        description="CLIP vision-text model - large version",
        local_name="clip_ViT_L_14",
        dimension=768,
        requires_feature="vision",
        alternative_ids=[
            "clip-ViT-L-14",
            "sentence-transformers/clip-ViT-L-14"
        ]
    )
}


# ============================================================================
# COMBINED REGISTRY
# ============================================================================

ALL_MODELS: Dict[str, ModelSpec] = {
    **EMBEDDING_MODELS,
    **RERANKING_MODELS,
    **VISION_MODELS
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_required_models(enabled_features: Optional[List[str]] = None, include_optional: bool = False) -> List[ModelSpec]:
    """
    Get list of required models based on enabled features.

    Args:
        enabled_features: List of enabled features (e.g., ['advanced_retrieval', 'vision'])
        include_optional: If True, include OPTIONAL models for enabled features

    Returns:
        List of ModelSpec objects that should be downloaded
    """
    if enabled_features is None:
        enabled_features = []

    required = []

    for model_spec in ALL_MODELS.values():
        # Always include REQUIRED models
        if model_spec.priority == ModelPriority.REQUIRED:
            required.append(model_spec)
            continue

        # Include RECOMMENDED models if their feature is enabled
        if model_spec.priority == ModelPriority.RECOMMENDED:
            if model_spec.requires_feature is None or model_spec.requires_feature in enabled_features:
                required.append(model_spec)
                continue

        # Include OPTIONAL models only if explicitly requested
        if model_spec.priority == ModelPriority.OPTIONAL and include_optional:
            if model_spec.requires_feature and model_spec.requires_feature in enabled_features:
                required.append(model_spec)

    return required


def get_default_models() -> List[ModelSpec]:
    """
    Get the 3 default models that are automatically downloaded.

    Returns:
        List containing:
        1. Default embedding model (all-MiniLM-L6-v2)
        2. Default reranking model (bge-reranker-base)
        3. Default vision model (clip-ViT-B-32)
    """
    return [
        EMBEDDING_MODELS["all-MiniLM-L6-v2"],  # Required
        RERANKING_MODELS["bge-reranker-base"],  # Recommended for advanced retrieval
        VISION_MODELS["clip-ViT-B-32"]  # Optional but included in defaults
    ]


def get_model_by_id(model_id: str) -> Optional[ModelSpec]:
    """
    Get model spec by any of its IDs (including alternatives).
    
    Args:
        model_id: Model ID to search for
        
    Returns:
        ModelSpec if found, None otherwise
    """
    # Direct lookup
    if model_id in ALL_MODELS:
        return ALL_MODELS[model_id]
    
    # Search in alternative IDs
    for model_spec in ALL_MODELS.values():
        if model_id in model_spec.alternative_ids or model_id == model_spec.model_id:
            return model_spec
    
    return None


def get_models_by_type(model_type: str) -> List[ModelSpec]:
    """
    Get all models of a specific type.
    
    Args:
        model_type: Type of models to retrieve (embedding, reranking, vision)
        
    Returns:
        List of ModelSpec objects
    """
    return [spec for spec in ALL_MODELS.values() if spec.model_type == model_type]


def get_default_model(model_type: str) -> Optional[ModelSpec]:
    """
    Get the default (first REQUIRED or RECOMMENDED) model for a type.
    
    Args:
        model_type: Type of model (embedding, reranking, vision)
        
    Returns:
        ModelSpec of default model, or None if no default
    """
    models = get_models_by_type(model_type)
    
    # First try REQUIRED
    for model in models:
        if model.priority == ModelPriority.REQUIRED:
            return model
    
    # Then try RECOMMENDED
    for model in models:
        if model.priority == ModelPriority.RECOMMENDED:
            return model
    
    # Finally try OPTIONAL
    if models:
        return models[0]
    
    return None


def get_total_download_size(models: List[ModelSpec]) -> float:
    """
    Calculate total download size for a list of models.
    
    Args:
        models: List of ModelSpec objects
        
    Returns:
        Total size in MB
    """
    return sum(model.size_mb for model in models)


def format_size(size_mb: float) -> str:
    """
    Format size in human-readable format.
    
    Args:
        size_mb: Size in megabytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "250 MB")
    """
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.0f} MB"


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_model_summary():
    """Print summary of all available models."""
    print("\n" + "=" * 80)
    print("RAG SYSTEM - AVAILABLE MODELS")
    print("=" * 80)
    
    for model_type in ["embedding", "reranking", "vision"]:
        models = get_models_by_type(model_type)
        if not models:
            continue
        
        print(f"\n{model_type.upper()} MODELS ({len(models)}):")
        print("-" * 80)
        
        for model in models:
            priority_icon = {
                ModelPriority.REQUIRED: "🔴",
                ModelPriority.RECOMMENDED: "🟡",
                ModelPriority.OPTIONAL: "🟢"
            }[model.priority]
            
            print(f"{priority_icon} {model.local_name}")
            print(f"   ID: {model.model_id}")
            print(f"   Size: {format_size(model.size_mb)}")
            print(f"   Priority: {model.priority.value}")
            if model.dimension:
                print(f"   Dimension: {model.dimension}")
            if model.requires_feature:
                print(f"   Requires: {model.requires_feature}")
            print(f"   Description: {model.description}")
            print()
    
    print("=" * 80)
    print(f"Total Models: {len(ALL_MODELS)}")
    print(f"Required: {len([m for m in ALL_MODELS.values() if m.priority == ModelPriority.REQUIRED])}")
    print(f"Recommended: {len([m for m in ALL_MODELS.values() if m.priority == ModelPriority.RECOMMENDED])}")
    print(f"Optional: {len([m for m in ALL_MODELS.values() if m.priority == ModelPriority.OPTIONAL])}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Print summary when run directly
    print_model_summary()

