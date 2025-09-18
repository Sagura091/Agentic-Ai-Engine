"""
Revolutionary Document Ingestion Pipeline for Advanced RAG System.

This module provides comprehensive document processing capabilities including
multi-modal processing, text extraction, chunking, metadata enrichment, and batch processing.
"""

from .pipeline import RevolutionaryIngestionPipeline, RevolutionaryIngestionConfig
from .processors import DocumentProcessor, RevolutionaryProcessorRegistry

# Legacy compatibility
IngestionPipeline = RevolutionaryIngestionPipeline
IngestionConfig = RevolutionaryIngestionConfig
ProcessorRegistry = RevolutionaryProcessorRegistry

__all__ = [
    "RevolutionaryIngestionPipeline",
    "RevolutionaryIngestionConfig",
    "DocumentProcessor",
    "RevolutionaryProcessorRegistry",
    # Legacy compatibility
    "IngestionPipeline",
    "IngestionConfig",
    "ProcessorRegistry"
]
