"""
🚀 Revolutionary Dynamic RAG Configuration Manager

This module provides real-time configuration management for the RAG system,
allowing settings to be updated without server restarts.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.rag.core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


class RAGConfigurationManager:
    """
    🚀 Revolutionary RAG Configuration Manager
    
    Manages real-time configuration updates for the RAG system without
    requiring server restarts.
    """
    
    def __init__(self):
        self.rag_system: Optional[UnifiedRAGSystem] = None
        self._update_lock = asyncio.Lock()
        self._update_history: List[Dict[str, Any]] = []
        logger.info(
            "🚀 RAG Configuration Manager initialized",
            LogCategory.SYSTEM_OPERATIONS,
            "app.rag.core.dynamic_config_manager.RAGConfigurationManager"
        )

    def set_rag_system(self, rag_system: UnifiedRAGSystem) -> None:
        """Set the RAG system instance to manage."""
        self.rag_system = rag_system
        logger.info(
            "✅ RAG system instance registered with configuration manager",
            LogCategory.SYSTEM_OPERATIONS,
            "app.rag.core.dynamic_config_manager.RAGConfigurationManager"
        )
    
    async def update_rag_configuration(self, settings_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        🚀 Update RAG configuration in real-time.
        
        Args:
            settings_updates: Dictionary of setting key-value pairs to update
            
        Returns:
            Dictionary with update results
        """
        async with self._update_lock:
            try:
                if not self.rag_system:
                    return {
                        "success": False,
                        "error": "RAG system not initialized",
                        "updates_applied": [],
                        "warnings": []
                    }

                logger.info(
                    f"🔄 Processing RAG configuration update with {len(settings_updates)} settings...",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.rag.core.dynamic_config_manager.RAGConfigurationManager",
                    data={"settings_count": len(settings_updates)}
                )

                # Convert enhanced admin settings keys to RAG config keys
                rag_config_updates = self._convert_admin_settings_to_rag_config(settings_updates)
                
                if not rag_config_updates:
                    return {
                        "success": True,
                        "message": "No RAG-specific settings to update",
                        "updates_applied": [],
                        "warnings": []
                    }
                
                # Apply updates to RAG system
                result = await self.rag_system.update_configuration(rag_config_updates)
                
                # Record update in history
                self._update_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "settings_updated": list(settings_updates.keys()),
                    "rag_config_updates": rag_config_updates,
                    "success": result.get("success", False),
                    "updates_applied": result.get("updates_applied", [])
                })
                
                # Keep only last 100 updates in history
                if len(self._update_history) > 100:
                    self._update_history = self._update_history[-100:]

                logger.info(
                    f"✅ RAG configuration update completed: {result.get('message', 'Unknown result')}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.rag.core.dynamic_config_manager.RAGConfigurationManager",
                    data={"result": result.get('message', 'Unknown result')}
                )

                return result

            except Exception as e:
                logger.error(
                    "❌ Failed to update RAG configuration",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.rag.core.dynamic_config_manager.RAGConfigurationManager",
                    error=e
                )
                return {
                    "success": False,
                    "error": f"Configuration update failed: {str(e)}",
                    "updates_applied": [],
                    "warnings": []
                }
    
    def _convert_admin_settings_to_rag_config(self, admin_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert enhanced admin settings keys to RAG configuration keys.
        
        Maps settings like 'rag_configuration.embedding_model' to 'embedding_model'
        """
        rag_config = {}
        
        # Mapping from admin settings keys to RAG config keys
        key_mappings = {
            # Core settings
            "rag_configuration.persist_directory": "persist_directory",
            "rag_configuration.embedding_model": "embedding_model",
            
            # Performance settings
            "rag_configuration.chunk_size": "chunk_size",
            "rag_configuration.chunk_overlap": "chunk_overlap",
            "rag_configuration.top_k": "top_k",
            "rag_configuration.batch_size": "batch_size",
            
            # Agent isolation
            "rag_configuration.auto_create_collections": "auto_create_collections",
            "rag_configuration.strict_isolation": "strict_isolation",
            "rag_configuration.enable_agent_isolation": "strict_isolation",
            
            # Memory settings
            "rag_configuration.short_term_ttl_hours": "short_term_ttl_hours",
            "rag_configuration.long_term_max_items": "long_term_max_items",
        }
        
        for admin_key, value in admin_settings.items():
            if admin_key in key_mappings:
                rag_key = key_mappings[admin_key]
                rag_config[rag_key] = value
                logger.debug(
                    f"Mapped {admin_key} → {rag_key} = {value}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.rag.core.dynamic_config_manager.RAGConfigurationManager",
                    data={"admin_key": admin_key, "rag_key": rag_key, "value": value}
                )
        
        return rag_config
    
    async def get_rag_status(self) -> Dict[str, Any]:
        """Get current RAG system status and configuration."""
        try:
            if not self.rag_system:
                return {
                    "initialized": False,
                    "error": "RAG system not available"
                }
            
            config_info = await self.rag_system.get_current_configuration()
            
            return {
                "initialized": self.rag_system.is_initialized,
                "configuration": config_info["config"],
                "stats": config_info["stats"],
                "last_update": self._update_history[-1] if self._update_history else None,
                "total_updates": len(self._update_history),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(
                "❌ Failed to get RAG status",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.core.dynamic_config_manager.RAGConfigurationManager",
                error=e
            )
            return {
                "initialized": False,
                "error": f"Failed to get status: {str(e)}"
            }
    
    async def reload_rag_system(self) -> Dict[str, Any]:
        """Reload the entire RAG system (for major configuration changes)."""
        try:
            if not self.rag_system:
                return {
                    "success": False,
                    "error": "RAG system not available"
                }
            
            result = await self.rag_system.reload_system()
            
            # Record reload in history
            self._update_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "system_reload",
                "success": result.get("success", False),
                "message": result.get("message", "System reload attempted")
            })
            
            return result

        except Exception as e:
            logger.error(
                "❌ Failed to reload RAG system",
                LogCategory.SYSTEM_OPERATIONS,
                "app.rag.core.dynamic_config_manager.RAGConfigurationManager",
                error=e
            )
            return {
                "success": False,
                "error": f"System reload failed: {str(e)}"
            }
    
    def get_update_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent configuration update history."""
        return self._update_history[-limit:] if self._update_history else []


# Global instance
rag_config_manager = RAGConfigurationManager()


async def initialize_rag_config_manager(rag_system: UnifiedRAGSystem) -> None:
    """Initialize the global RAG configuration manager with a RAG system instance."""
    rag_config_manager.set_rag_system(rag_system)
    logger.info(
        "🚀 Global RAG configuration manager initialized",
        LogCategory.SYSTEM_OPERATIONS,
        "app.rag.core.dynamic_config_manager"
    )


async def update_rag_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    🚀 Global function to update RAG settings in real-time.
    
    This function can be called from anywhere in the application to update
    RAG configuration without server restart.
    """
    return await rag_config_manager.update_rag_configuration(settings)


async def get_rag_system_status() -> Dict[str, Any]:
    """Get current RAG system status."""
    return await rag_config_manager.get_rag_status()


async def reload_rag_system() -> Dict[str, Any]:
    """Reload the RAG system completely."""
    return await rag_config_manager.reload_rag_system()
