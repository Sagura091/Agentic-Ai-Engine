"""
🚀 Revolutionary RAG Section Manager

Manages real-time configuration updates for the RAG system with
advanced validation, application, and rollback capabilities.
"""

from typing import Any, Dict, List, Optional
import structlog

from ..global_config_manager import ConfigurationSection
from .base_section_manager import BaseConfigurationSectionManager

logger = structlog.get_logger(__name__)


class RAGSectionManager(BaseConfigurationSectionManager):
    """
    🚀 Revolutionary RAG Configuration Section Manager
    
    Handles all RAG-related configuration updates with real-time application
    to the UnifiedRAGSystem without requiring server restarts.
    """
    
    def __init__(self):
        """Initialize the RAG section manager."""
        super().__init__()
        self._rag_system = None
        self._rag_config_manager = None
        logger.info("🚀 RAG Section Manager initialized")
    
    @property
    def section_name(self) -> ConfigurationSection:
        """The configuration section this manager handles."""
        return ConfigurationSection.RAG_CONFIGURATION
    
    def set_rag_system(self, rag_system) -> None:
        """Set the RAG system instance to manage."""
        self._rag_system = rag_system
        logger.info("✅ RAG system instance registered with RAG section manager")
    
    def set_rag_config_manager(self, rag_config_manager) -> None:
        """Set the RAG configuration manager instance."""
        self._rag_config_manager = rag_config_manager
        logger.info("✅ RAG configuration manager registered with RAG section manager")
    
    async def _load_initial_configuration(self) -> None:
        """Load the initial RAG configuration."""
        try:
            # Load from RAG system if available
            if self._rag_system and hasattr(self._rag_system, 'config'):
                config = self._rag_system.config
                self._current_config = {
                    "persist_directory": config.persist_directory,
                    "embedding_model": config.embedding_model,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "top_k": config.top_k,
                    "batch_size": config.batch_size,
                    "auto_create_collections": config.auto_create_collections,
                    "strict_isolation": config.strict_isolation,
                    "short_term_ttl_hours": config.short_term_ttl_hours,
                    "long_term_max_items": config.long_term_max_items
                }
                logger.info("✅ Loaded initial RAG configuration from RAG system")
            else:
                # Default configuration
                self._current_config = {
                    "persist_directory": "./data/chroma",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "top_k": 10,
                    "batch_size": 32,
                    "auto_create_collections": True,
                    "strict_isolation": True,
                    "short_term_ttl_hours": 24,
                    "long_term_max_items": 10000
                }
                logger.info("✅ Loaded default RAG configuration")
                
        except Exception as e:
            logger.error(f"❌ Failed to load initial RAG configuration: {str(e)}")
            # Use safe defaults
            self._current_config = {}
    
    async def _setup_validation_rules(self) -> None:
        """Setup validation rules for RAG configuration."""
        self._validation_rules = {
            "field_types": {
                "persist_directory": str,
                "embedding_model": str,
                "chunk_size": int,
                "chunk_overlap": int,
                "top_k": int,
                "batch_size": int,
                "auto_create_collections": bool,
                "strict_isolation": bool,
                "short_term_ttl_hours": int,
                "long_term_max_items": int
            },
            "field_ranges": {
                "chunk_size": (100, 4000),
                "chunk_overlap": (0, 1000),
                "top_k": (1, 100),
                "batch_size": (1, 128),
                "short_term_ttl_hours": (1, 168),  # 1 hour to 1 week
                "long_term_max_items": (100, 100000)
            },
            "required_fields": [],  # No required fields for updates
            "valid_embedding_models": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "text-embedding-ada-002",  # OpenAI
                "text-embedding-3-small",  # OpenAI
                "text-embedding-3-large"   # OpenAI
            ]
        }
        logger.info("✅ RAG validation rules configured")
    
    async def _validate_custom_rules(self, config: Dict[str, Any]) -> List[str]:
        """Validate RAG-specific business rules."""
        errors = []
        
        try:
            # Validate embedding model
            if "embedding_model" in config:
                model = config["embedding_model"]
                valid_models = self._validation_rules["valid_embedding_models"]
                if model not in valid_models:
                    errors.append(f"Embedding model '{model}' is not supported. Valid models: {valid_models}")
            
            # Validate chunk overlap vs chunk size
            chunk_size = config.get("chunk_size", self._current_config.get("chunk_size", 1000))
            chunk_overlap = config.get("chunk_overlap", self._current_config.get("chunk_overlap", 200))
            
            if chunk_overlap >= chunk_size:
                errors.append("Chunk overlap must be less than chunk size")
            
            # Validate persist directory
            if "persist_directory" in config:
                persist_dir = config["persist_directory"]
                if not persist_dir or not isinstance(persist_dir, str):
                    errors.append("Persist directory must be a non-empty string")
            
            # Validate memory settings consistency
            if "short_term_ttl_hours" in config and config["short_term_ttl_hours"] <= 0:
                errors.append("Short-term TTL must be positive")
            
            if "long_term_max_items" in config and config["long_term_max_items"] <= 0:
                errors.append("Long-term max items must be positive")
                
        except Exception as e:
            errors.append(f"Custom validation error: {str(e)}")
        
        return errors
    
    async def _apply_configuration_changes(
        self, 
        config: Dict[str, Any], 
        previous_config: Dict[str, Any]
    ) -> List[str]:
        """Apply RAG configuration changes to the RAG system."""
        warnings = []
        
        try:
            logger.info("🔄 Applying RAG configuration changes", changes=list(config.keys()))
            
            # Apply through RAG configuration manager if available
            if self._rag_config_manager:
                result = await self._rag_config_manager.update_rag_configuration(config)
                
                if result.get("success"):
                    logger.info("✅ RAG configuration applied via RAG configuration manager")
                    if result.get("warnings"):
                        warnings.extend(result["warnings"])
                else:
                    raise Exception(f"RAG configuration manager failed: {result.get('error')}")
            
            # Apply directly to RAG system if available
            elif self._rag_system:
                if hasattr(self._rag_system, 'update_configuration'):
                    result = await self._rag_system.update_configuration(config)
                    
                    if result.get("success"):
                        logger.info("✅ RAG configuration applied directly to RAG system")
                        if result.get("warnings"):
                            warnings.extend(result["warnings"])
                    else:
                        raise Exception(f"RAG system update failed: {result.get('error')}")
                else:
                    warnings.append("RAG system does not support dynamic configuration updates")
            
            else:
                warnings.append("No RAG system or configuration manager available - configuration stored but not applied")
            
            # Log specific changes
            for key, value in config.items():
                old_value = previous_config.get(key, "not set")
                logger.info(f"🔧 RAG config changed: {key} = {value} (was: {old_value})")
            
            return warnings
            
        except Exception as e:
            error_msg = f"Failed to apply RAG configuration: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def _perform_rollback(self, rollback_data: Dict[str, Any]) -> bool:
        """Perform rollback of RAG configuration changes."""
        try:
            logger.info("🔄 Rolling back RAG configuration changes")
            
            # Apply rollback through RAG configuration manager if available
            if self._rag_config_manager:
                result = await self._rag_config_manager.update_rag_configuration(rollback_data)
                success = result.get("success", False)
                
                if success:
                    logger.info("✅ RAG configuration rollback successful via RAG configuration manager")
                else:
                    logger.error(f"❌ RAG configuration rollback failed: {result.get('error')}")
                
                return success
            
            # Apply rollback directly to RAG system if available
            elif self._rag_system and hasattr(self._rag_system, 'update_configuration'):
                result = await self._rag_system.update_configuration(rollback_data)
                success = result.get("success", False)
                
                if success:
                    logger.info("✅ RAG configuration rollback successful via RAG system")
                else:
                    logger.error(f"❌ RAG configuration rollback failed: {result.get('error')}")
                
                return success
            
            else:
                logger.warning("⚠️ No RAG system available for rollback - configuration not restored")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during RAG configuration rollback: {str(e)}")
            return False
