"""
RAG Configuration Module.

This module provides comprehensive configuration management for the RAG system,
including OpenWebUI-inspired configuration patterns and model specifications.
"""

from .openwebui_config import (
    PersistentConfig,
    RAGConfig,
    DataDirectoryConfig,
    get_rag_config,
    get_data_directory_config
)

from .required_models import (
    ModelPriority,
    ModelSpec,
    EMBEDDING_MODELS,
    RERANKING_MODELS,
    VISION_MODELS,
    get_all_required_models,
    get_models_by_priority,
    get_model_spec
)

__all__ = [
    # OpenWebUI Configuration
    "PersistentConfig",
    "RAGConfig",
    "DataDirectoryConfig",
    "get_rag_config",
    "get_data_directory_config",
    
    # Model Specifications
    "ModelPriority",
    "ModelSpec",
    "EMBEDDING_MODELS",
    "RERANKING_MODELS",
    "VISION_MODELS",
    "get_all_required_models",
    "get_models_by_priority",
    "get_model_spec"
]

