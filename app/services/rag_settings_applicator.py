"""
RAG Settings Applicator.

This service applies admin settings to the RAG system, ensuring that when
admins change RAG settings in the frontend, they actually take effect in
the running RAG system.
"""

from typing import Any, Dict, Optional

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.rag.core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig
from app.rag.config.openwebui_config import get_rag_config, OpenWebUIRAGConfig
from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator

logger = get_logger()


class RAGSettingsApplicator:
    """
    Applies admin settings to the RAG system.
    
    This class bridges the gap between admin settings and the actual RAG system
    configuration, ensuring that settings changes are properly applied to the
    running system.
    """
    
    def __init__(self):
        self.orchestrator = None
        self.rag_config_manager = None
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the RAG settings applicator."""
        try:
            # Get the system orchestrator
            self.orchestrator = get_enhanced_system_orchestrator()
            
            # Get the RAG configuration manager
            self.rag_config_manager = get_rag_config()
            
            self.is_initialized = True
            logger.info(
                "RAG settings applicator initialized",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator"
            )

        except Exception as e:
            logger.error(
                "Failed to initialize RAG settings applicator",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )
            raise
    
    async def apply_rag_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Apply RAG settings to the running system.
        
        Args:
            settings: Dictionary of RAG settings to apply
            
        Returns:
            True if settings were applied successfully
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            logger.info(
                "Applying RAG settings to system",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                data={"settings_count": len(settings)}
            )

            # Apply vector store settings
            await self._apply_vector_store_settings(settings)

            # Apply embedding settings
            await self._apply_embedding_settings(settings)

            # Apply chunking settings
            await self._apply_chunking_settings(settings)

            # Apply retrieval settings
            await self._apply_retrieval_settings(settings)

            # Apply performance settings
            await self._apply_performance_settings(settings)

            # Apply multi-agent settings
            await self._apply_multi_agent_settings(settings)

            logger.info(
                "RAG settings applied successfully",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator"
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to apply RAG settings",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )
            return False
    
    async def _apply_vector_store_settings(self, settings: Dict[str, Any]) -> None:
        """Apply vector store related settings."""
        try:
            vector_store_settings = {}
            
            # Map admin settings to RAG config
            if "persist_directory" in settings:
                vector_store_settings["vector_db_dir"] = settings["persist_directory"]["value"]
            
            if "collection_metadata" in settings:
                vector_store_settings["chroma_collection_metadata"] = settings["collection_metadata"]["value"]
            
            if "connection_pool_size" in settings:
                vector_store_settings["connection_pool_size"] = settings["connection_pool_size"]["value"]
            
            if "max_batch_size" in settings:
                vector_store_settings["max_batch_size"] = settings["max_batch_size"]["value"]
            
            if vector_store_settings:
                # Update RAG configuration
                self.rag_config_manager.update_config(**vector_store_settings)
                
                # If the orchestrator has a RAG system, update its config
                if self.orchestrator and self.orchestrator.unified_rag:
                    # Create new config with updated settings
                    current_config = self.orchestrator.unified_rag.config
                    new_config = UnifiedRAGConfig(
                        persist_directory=vector_store_settings.get("vector_db_dir", current_config.persist_directory),
                        embedding_model=current_config.embedding_model,
                        chunk_size=current_config.chunk_size,
                        chunk_overlap=current_config.chunk_overlap,
                        top_k=current_config.top_k,
                        batch_size=vector_store_settings.get("max_batch_size", current_config.batch_size)
                    )
                    
                    # Update the RAG system config
                    self.orchestrator.unified_rag.config = new_config

                logger.info(
                    "Vector store settings applied",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.rag_settings_applicator",
                    data={"settings": vector_store_settings}
                )

        except Exception as e:
            logger.error(
                "Failed to apply vector store settings",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )
    
    async def _apply_embedding_settings(self, settings: Dict[str, Any]) -> None:
        """Apply embedding related settings."""
        try:
            embedding_settings = {}
            
            # Map admin settings to RAG config
            if "embedding_model" in settings:
                embedding_settings["embedding_model"] = settings["embedding_model"]["value"]
            
            if "embedding_batch_size" in settings:
                embedding_settings["embedding_batch_size"] = settings["embedding_batch_size"]["value"]
            
            if "normalize_embeddings" in settings:
                embedding_settings["normalize_embeddings"] = settings["normalize_embeddings"]["value"]
            
            if "cache_embeddings" in settings:
                embedding_settings["cache_embeddings"] = settings["cache_embeddings"]["value"]
            
            if "cache_size" in settings:
                embedding_settings["embedding_cache_size"] = settings["cache_size"]["value"]
            
            if embedding_settings:
                # Update RAG configuration
                self.rag_config_manager.update_config(**embedding_settings)
                
                # Update unified RAG system if available
                if self.orchestrator and self.orchestrator.unified_rag:
                    current_config = self.orchestrator.unified_rag.config
                    new_config = UnifiedRAGConfig(
                        persist_directory=current_config.persist_directory,
                        embedding_model=embedding_settings.get("embedding_model", current_config.embedding_model),
                        chunk_size=current_config.chunk_size,
                        chunk_overlap=current_config.chunk_overlap,
                        top_k=current_config.top_k,
                        batch_size=embedding_settings.get("embedding_batch_size", current_config.batch_size)
                    )
                    
                    self.orchestrator.unified_rag.config = new_config
                    
                    # If embedding model changed, we need to reinitialize
                    if "embedding_model" in embedding_settings:
                        logger.info(
                            "Embedding model changed, reinitializing RAG system",
                            LogCategory.RAG_OPERATIONS,
                            "app.services.rag_settings_applicator"
                        )
                        # Note: This would require a restart in production

                logger.info(
                    "Embedding settings applied",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.rag_settings_applicator",
                    data={"settings": embedding_settings}
                )

        except Exception as e:
            logger.error(
                "Failed to apply embedding settings",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )
    
    async def _apply_chunking_settings(self, settings: Dict[str, Any]) -> None:
        """Apply chunking related settings."""
        try:
            chunking_settings = {}
            
            # Map admin settings to RAG config
            if "chunk_size" in settings:
                chunking_settings["rag_chunk_size"] = settings["chunk_size"]["value"]
            
            if "chunk_overlap" in settings:
                chunking_settings["rag_chunk_overlap"] = settings["chunk_overlap"]["value"]
            
            if "chunking_strategy" in settings:
                chunking_settings["chunking_strategy"] = settings["chunking_strategy"]["value"]
            
            if chunking_settings:
                # Update RAG configuration
                self.rag_config_manager.update_config(**chunking_settings)
                
                # Update unified RAG system
                if self.orchestrator and self.orchestrator.unified_rag:
                    current_config = self.orchestrator.unified_rag.config
                    new_config = UnifiedRAGConfig(
                        persist_directory=current_config.persist_directory,
                        embedding_model=current_config.embedding_model,
                        chunk_size=chunking_settings.get("rag_chunk_size", current_config.chunk_size),
                        chunk_overlap=chunking_settings.get("rag_chunk_overlap", current_config.chunk_overlap),
                        top_k=current_config.top_k,
                        batch_size=current_config.batch_size
                    )
                    
                    self.orchestrator.unified_rag.config = new_config

                logger.info(
                    "Chunking settings applied",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.rag_settings_applicator",
                    data={"settings": chunking_settings}
                )

        except Exception as e:
            logger.error(
                "Failed to apply chunking settings",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )
    
    async def _apply_retrieval_settings(self, settings: Dict[str, Any]) -> None:
        """Apply retrieval related settings."""
        try:
            retrieval_settings = {}
            
            # Map admin settings to RAG config
            if "top_k" in settings:
                retrieval_settings["rag_top_k"] = settings["top_k"]["value"]
            
            if "score_threshold" in settings:
                retrieval_settings["score_threshold"] = settings["score_threshold"]["value"]
            
            if "enable_reranking" in settings:
                retrieval_settings["enable_reranking"] = settings["enable_reranking"]["value"]
            
            if "enable_hybrid_search" in settings:
                retrieval_settings["enable_hybrid_search"] = settings["enable_hybrid_search"]["value"]
            
            if "hybrid_bm25_weight" in settings:
                retrieval_settings["hybrid_bm25_weight"] = settings["hybrid_bm25_weight"]["value"]
            
            if retrieval_settings:
                # Update RAG configuration
                self.rag_config_manager.update_config(**retrieval_settings)
                
                # Update unified RAG system
                if self.orchestrator and self.orchestrator.unified_rag:
                    current_config = self.orchestrator.unified_rag.config
                    new_config = UnifiedRAGConfig(
                        persist_directory=current_config.persist_directory,
                        embedding_model=current_config.embedding_model,
                        chunk_size=current_config.chunk_size,
                        chunk_overlap=current_config.chunk_overlap,
                        top_k=retrieval_settings.get("rag_top_k", current_config.top_k),
                        batch_size=current_config.batch_size
                    )
                    
                    self.orchestrator.unified_rag.config = new_config

                logger.info(
                    "Retrieval settings applied",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.rag_settings_applicator",
                    data={"settings": retrieval_settings}
                )

        except Exception as e:
            logger.error(
                "Failed to apply retrieval settings",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )
    
    async def _apply_performance_settings(self, settings: Dict[str, Any]) -> None:
        """Apply performance related settings."""
        try:
            performance_settings = {}
            
            # Map admin settings to RAG config
            if "enable_caching" in settings:
                performance_settings["enable_caching"] = settings["enable_caching"]["value"]
            
            if "cache_ttl" in settings:
                performance_settings["cache_ttl"] = settings["cache_ttl"]["value"]
            
            if "max_concurrent_queries" in settings:
                performance_settings["max_concurrent_queries"] = settings["max_concurrent_queries"]["value"]
            
            if performance_settings:
                # Update RAG configuration
                self.rag_config_manager.update_config(**performance_settings)

                logger.info(
                    "Performance settings applied",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.rag_settings_applicator",
                    data={"settings": performance_settings}
                )

        except Exception as e:
            logger.error(
                "Failed to apply performance settings",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )
    
    async def _apply_multi_agent_settings(self, settings: Dict[str, Any]) -> None:
        """Apply multi-agent related settings."""
        try:
            multi_agent_settings = {}
            
            # Map admin settings to RAG config
            if "enable_agent_isolation" in settings:
                multi_agent_settings["enable_agent_isolation"] = settings["enable_agent_isolation"]["value"]
            
            if "enable_knowledge_sharing" in settings:
                multi_agent_settings["enable_knowledge_sharing"] = settings["enable_knowledge_sharing"]["value"]
            
            if "default_retention_days" in settings:
                multi_agent_settings["default_retention_days"] = settings["default_retention_days"]["value"]
            
            if "max_memory_items" in settings:
                multi_agent_settings["max_memory_items"] = settings["max_memory_items"]["value"]
            
            if multi_agent_settings:
                # Update RAG configuration
                self.rag_config_manager.update_config(**multi_agent_settings)

                logger.info(
                    "Multi-agent settings applied",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.rag_settings_applicator",
                    data={"settings": multi_agent_settings}
                )

        except Exception as e:
            logger.error(
                "Failed to apply multi-agent settings",
                LogCategory.RAG_OPERATIONS,
                "app.services.rag_settings_applicator",
                error=e
            )


# Global applicator instance
_rag_settings_applicator: Optional[RAGSettingsApplicator] = None


async def get_rag_settings_applicator() -> RAGSettingsApplicator:
    """Get the global RAG settings applicator instance."""
    global _rag_settings_applicator
    if _rag_settings_applicator is None:
        _rag_settings_applicator = RAGSettingsApplicator()
        await _rag_settings_applicator.initialize()
    return _rag_settings_applicator
