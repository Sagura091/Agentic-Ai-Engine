"""
Global Embedding Manager for RAG System.

This module provides a singleton embedding manager that uses the global
embedding configuration and is shared across all knowledge bases.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import structlog
from pydantic import BaseModel

from .embedding_model_manager import embedding_model_manager
from .embeddings import EmbeddingManager, EmbeddingConfig

logger = structlog.get_logger(__name__)


class GlobalEmbeddingManager:
    """
    Global embedding manager that uses the global embedding configuration
    and is shared across all knowledge bases.
    """
    
    def __init__(self):
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.current_config: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the global embedding manager with current configuration."""
        async with self._lock:
            if self.is_initialized:
                return
                
            try:
                # Get global configuration
                config = embedding_model_manager.get_global_config()
                
                # Create embedding config from global config
                embedding_config = EmbeddingConfig(
                    model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
                    batch_size=config.get("embedding_batch_size", 32),
                    use_model_manager=True  # Use the embedding model manager
                )
                
                # Initialize embedding manager
                self.embedding_manager = EmbeddingManager(embedding_config)
                await self.embedding_manager.initialize()
                
                self.current_config = config
                self.is_initialized = True
                
                logger.info(
                    "Global embedding manager initialized",
                    engine=config.get("embedding_engine", ""),
                    model=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                )
                
            except Exception as e:
                logger.error(f"Failed to initialize global embedding manager: {e}")
                # Initialize with fallback
                await self._initialize_fallback()
    
    async def _initialize_fallback(self) -> None:
        """Initialize with fallback configuration."""
        try:
            fallback_config = EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=32,
                use_model_manager=False
            )
            
            self.embedding_manager = EmbeddingManager(fallback_config)
            await self.embedding_manager.initialize()
            
            self.current_config = {
                "embedding_engine": "",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_batch_size": 32
            }
            self.is_initialized = True
            
            logger.info("Global embedding manager initialized with fallback")
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback embedding manager: {e}")
            raise
    
    async def reload_configuration(self) -> None:
        """Reload the global embedding configuration."""
        async with self._lock:
            try:
                # Get updated configuration
                new_config = embedding_model_manager.get_global_config()
                
                # Check if configuration changed
                if self.current_config and self._config_changed(new_config):
                    logger.info("Global embedding configuration changed, reloading...")
                    
                    # Reinitialize with new configuration
                    embedding_config = EmbeddingConfig(
                        model_name=new_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                        batch_size=new_config.get("embedding_batch_size", 32),
                        use_model_manager=True
                    )
                    
                    # Create new embedding manager
                    new_embedding_manager = EmbeddingManager(embedding_config)
                    await new_embedding_manager.initialize()
                    
                    # Replace old manager
                    self.embedding_manager = new_embedding_manager
                    self.current_config = new_config
                    
                    logger.info(
                        "Global embedding configuration reloaded",
                        engine=new_config.get("embedding_engine", ""),
                        model=new_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                    )
                
            except Exception as e:
                logger.error(f"Failed to reload global embedding configuration: {e}")
                raise
    
    def _config_changed(self, new_config: Dict[str, Any]) -> bool:
        """Check if the configuration has changed."""
        if not self.current_config:
            return True
            
        key_fields = ["embedding_engine", "embedding_model", "embedding_batch_size"]
        
        for field in key_fields:
            if self.current_config.get(field) != new_config.get(field):
                return True
                
        return False
    
    async def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        prefix: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using the global configuration."""
        if not self.is_initialized:
            await self.initialize()
            
        if not self.embedding_manager:
            raise RuntimeError("Global embedding manager not initialized")
            
        try:
            # Generate embeddings
            result = await self.embedding_manager.generate_embeddings(texts)
            
            # Extract embeddings from result
            if hasattr(result, 'embeddings'):
                return result.embeddings
            elif isinstance(result, list):
                return result
            else:
                raise ValueError(f"Unexpected embedding result type: {type(result)}")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get the current global embedding configuration."""
        return self.current_config.copy() if self.current_config else None
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current embedding model."""
        if not self.embedding_manager:
            return None
            
        try:
            model_info = self.embedding_manager.get_model_info()
            if model_info:
                return {
                    "model_name": model_info.model_name if hasattr(model_info, 'model_name') else "unknown",
                    "embedding_type": model_info.embedding_type.value if hasattr(model_info, 'embedding_type') else "dense",
                    "device": self.embedding_manager.device if hasattr(self.embedding_manager, 'device') else "unknown"
                }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            
        return None

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the global embedding configuration."""
        try:
            # Update the embedding model manager's global config
            embedding_model_manager.update_global_config(config)

            # Update current config
            self.current_config = config.copy()

            # Reset initialization to force reload
            self.is_initialized = False
            self.embedding_manager = None

            logger.info("Global embedding configuration updated")

        except Exception as e:
            logger.error(f"Failed to update global embedding configuration: {e}")
            raise


# Global instance
_global_embedding_manager: Optional[GlobalEmbeddingManager] = None


async def get_global_embedding_manager() -> GlobalEmbeddingManager:
    """Get the global embedding manager instance."""
    global _global_embedding_manager
    
    if _global_embedding_manager is None:
        _global_embedding_manager = GlobalEmbeddingManager()
        await _global_embedding_manager.initialize()
    
    return _global_embedding_manager


async def reload_global_embedding_config() -> None:
    """Reload the global embedding configuration."""
    global _global_embedding_manager
    
    if _global_embedding_manager is not None:
        await _global_embedding_manager.reload_configuration()


def get_global_embedding_config() -> Optional[Dict[str, Any]]:
    """Get the current global embedding configuration (sync)."""
    global _global_embedding_manager

    if _global_embedding_manager is not None:
        return _global_embedding_manager.get_current_config()

    return None


def set_global_embedding_config(config: Dict[str, Any]) -> None:
    """Set the global embedding configuration (sync)."""
    global _global_embedding_manager

    if _global_embedding_manager is None:
        _global_embedding_manager = GlobalEmbeddingManager()

    # Update the configuration
    _global_embedding_manager.update_config(config)
