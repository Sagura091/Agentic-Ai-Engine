"""
RAG Integration Module - Complete RAG System Integration

This module provides the complete integration between the RAG system and agents,
implementing both model-level RAG (automatic context injection) and agent-level
RAG (explicit tools) for the best of both worlds.
"""

from .hybrid_rag_integration import (
    HybridRAGIntegration,
    get_hybrid_rag_integration,
    initialize_hybrid_rag_system
)

__all__ = [
    "HybridRAGIntegration",
    "get_hybrid_rag_integration", 
    "initialize_hybrid_rag_system"
]
