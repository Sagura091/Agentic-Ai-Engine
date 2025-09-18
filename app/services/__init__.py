"""
Services module for business logic.

This module contains the remaining services that support the unified
multi-agent architecture, including document processing, LLM management,
and specialized ingestion capabilities.

Note: Core RAG, memory, and monitoring services have been moved to
the unified architecture components in app/rag/core/, app/communication/,
and app/optimization/ respectively.
"""

# Import remaining services
from .document_service import DocumentService
from .llm_service import LLMService
from .revolutionary_ingestion_engine import RevolutionaryIngestionEngine

__all__ = [
    "DocumentService",
    "LLMService",
    "RevolutionaryIngestionEngine"
]
