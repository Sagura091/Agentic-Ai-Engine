"""
🚀 Revolutionary Configuration Observers

This package contains observer implementations that respond to configuration
changes and update their respective systems in real-time.

Observers implement the observer pattern to provide loose coupling between
the global configuration manager and individual system components.

COMPLETE OBSERVER COVERAGE:
- LLM Providers: Real-time model and provider configuration updates
- RAG System: Embedding, vision, and retrieval configuration updates
- Memory System: Memory type, storage, and retention policy updates
- Database: Connection pool, query optimization, and performance updates
- Storage: Backend, compression, and backup configuration updates
- Performance: CPU, memory, caching, and load balancing updates
"""

from .rag_observer import RAGConfigurationObserver
from .llm_observer import LLMConfigurationObserver
from .memory_observer import MemoryConfigurationObserver
from .database_observer import DatabaseConfigurationObserver
from .storage_observer import StorageConfigurationObserver
from .performance_observer import PerformanceConfigurationObserver

__all__ = [
    "RAGConfigurationObserver",
    "LLMConfigurationObserver",
    "MemoryConfigurationObserver",
    "DatabaseConfigurationObserver",
    "StorageConfigurationObserver",
    "PerformanceConfigurationObserver"
]
