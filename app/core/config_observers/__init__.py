"""
🚀 Revolutionary Configuration Observers

This package contains observer implementations that respond to configuration
changes and update their respective systems in real-time.

Observers implement the observer pattern to provide loose coupling between
the global configuration manager and individual system components.
"""

from .rag_observer import RAGConfigurationObserver

__all__ = [
    "RAGConfigurationObserver"
]
