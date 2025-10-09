"""
Vision and Multimodal Components for RAG.

This module provides state-of-the-art vision-text embedding capabilities
using CLIP models for multimodal AI applications.
"""

from .clip_embeddings import (
    RevolutionaryCLIPEmbedding,
    CLIPConfig,
    VisionTextSimilarity,
    MultimodalEmbedding
)

__all__ = [
    "RevolutionaryCLIPEmbedding",
    "CLIPConfig",
    "VisionTextSimilarity",
    "MultimodalEmbedding"
]

