"""
🚀 Revolutionary RAG Configuration Observer

Observes RAG configuration changes and applies them to the RAG system
in real-time without requiring server restarts.
"""

from typing import Any, Dict, Set
import structlog

from ..global_config_manager import ConfigurationObserver, ConfigurationSection, UpdateResult

logger = structlog.get_logger(__name__)


class RAGConfigurationObserver(ConfigurationObserver):
    """
    🚀 Revolutionary RAG Configuration Observer
    
    Monitors RAG configuration changes and applies them to the UnifiedRAGSystem
    in real-time, ensuring zero-downtime configuration updates.
    """
    
    def __init__(self, rag_system=None, rag_config_manager=None):
        """
        Initialize the RAG configuration observer.
        
        Args:
            rag_system: UnifiedRAGSystem instance to update
            rag_config_manager: RAGConfigurationManager instance to use
        """
        self._rag_system = rag_system
        self._rag_config_manager = rag_config_manager
        logger.info("🚀 RAG Configuration Observer initialized")
    
    @property
    def observer_name(self) -> str:
        """Name of the observer for logging and identification."""
        return "RAGConfigurationObserver"
    
    @property
    def observed_sections(self) -> Set[ConfigurationSection]:
        """Set of configuration sections this observer is interested in."""
        return {ConfigurationSection.RAG_CONFIGURATION}
    
    def set_rag_system(self, rag_system) -> None:
        """Set the RAG system instance to update."""
        self._rag_system = rag_system
        logger.info("✅ RAG system instance registered with RAG observer")
    
    def set_rag_config_manager(self, rag_config_manager) -> None:
        """Set the RAG configuration manager instance."""
        self._rag_config_manager = rag_config_manager
        logger.info("✅ RAG configuration manager registered with RAG observer")
    
    async def on_configuration_changed(
        self, 
        section: ConfigurationSection, 
        changes: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """
        Called when RAG configuration changes.
        
        Args:
            section: The configuration section that changed (should be RAG_CONFIGURATION)
            changes: Dictionary of changed configuration values
            previous_config: Previous configuration values for rollback
            
        Returns:
            UpdateResult indicating success/failure and any issues
        """
        if section != ConfigurationSection.RAG_CONFIGURATION:
            logger.warning(f"RAG observer received unexpected section: {section.value}")
            return UpdateResult(
                success=False,
                section=section.value,
                errors=[f"RAG observer only handles RAG configuration, got {section.value}"]
            )
        
        logger.info("🔄 RAG configuration changed, applying updates", changes=list(changes.keys()))
        
        try:
            warnings = []
            
            # Apply changes through RAG configuration manager if available
            if self._rag_config_manager:
                result = await self._apply_via_config_manager(changes, previous_config)
                if result.warnings:
                    warnings.extend(result.warnings)
                if not result.success:
                    return result
            
            # Apply changes directly to RAG system if available
            elif self._rag_system:
                result = await self._apply_via_rag_system(changes, previous_config)
                if result.warnings:
                    warnings.extend(result.warnings)
                if not result.success:
                    return result
            
            else:
                warnings.append("No RAG system or configuration manager available")
                logger.warning("⚠️ No RAG system or configuration manager available for configuration update")
            
            # Log successful changes
            for key, value in changes.items():
                old_value = previous_config.get(key, "not set")
                logger.info(f"✅ RAG config updated: {key} = {value} (was: {old_value})")
            
            return UpdateResult(
                success=True,
                section=section.value,
                changes_applied=changes,
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = f"Failed to apply RAG configuration changes: {str(e)}"
            logger.error(error_msg, changes=changes)
            
            return UpdateResult(
                success=False,
                section=section.value,
                errors=[error_msg]
            )
    
    async def _apply_via_config_manager(
        self, 
        changes: Dict[str, Any], 
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """Apply changes through the RAG configuration manager."""
        try:
            logger.info("🔄 Applying RAG configuration via RAG configuration manager")
            
            # Use the existing RAG configuration manager
            result = await self._rag_config_manager.update_rag_configuration(changes)
            
            if result.get("success"):
                logger.info("✅ RAG configuration applied successfully via configuration manager")
                return UpdateResult(
                    success=True,
                    section=ConfigurationSection.RAG_CONFIGURATION.value,
                    changes_applied=changes,
                    warnings=result.get("warnings", [])
                )
            else:
                error_msg = result.get("error", "Unknown error from RAG configuration manager")
                logger.error(f"❌ RAG configuration manager failed: {error_msg}")
                return UpdateResult(
                    success=False,
                    section=ConfigurationSection.RAG_CONFIGURATION.value,
                    errors=[error_msg]
                )
                
        except Exception as e:
            error_msg = f"Error applying RAG configuration via configuration manager: {str(e)}"
            logger.error(error_msg)
            return UpdateResult(
                success=False,
                section=ConfigurationSection.RAG_CONFIGURATION.value,
                errors=[error_msg]
            )
    
    async def _apply_via_rag_system(
        self, 
        changes: Dict[str, Any], 
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """Apply changes directly to the RAG system."""
        try:
            logger.info("🔄 Applying RAG configuration directly to RAG system")
            
            # Check if RAG system supports dynamic configuration
            if not hasattr(self._rag_system, 'update_configuration'):
                warning_msg = "RAG system does not support dynamic configuration updates"
                logger.warning(f"⚠️ {warning_msg}")
                return UpdateResult(
                    success=True,
                    section=ConfigurationSection.RAG_CONFIGURATION.value,
                    changes_applied={},
                    warnings=[warning_msg]
                )
            
            # Apply configuration to RAG system
            result = await self._rag_system.update_configuration(changes)
            
            if result.get("success"):
                logger.info("✅ RAG configuration applied successfully to RAG system")
                return UpdateResult(
                    success=True,
                    section=ConfigurationSection.RAG_CONFIGURATION.value,
                    changes_applied=changes,
                    warnings=result.get("warnings", [])
                )
            else:
                error_msg = result.get("error", "Unknown error from RAG system")
                logger.error(f"❌ RAG system configuration update failed: {error_msg}")
                return UpdateResult(
                    success=False,
                    section=ConfigurationSection.RAG_CONFIGURATION.value,
                    errors=[error_msg]
                )
                
        except Exception as e:
            error_msg = f"Error applying RAG configuration to RAG system: {str(e)}"
            logger.error(error_msg)
            return UpdateResult(
                success=False,
                section=ConfigurationSection.RAG_CONFIGURATION.value,
                errors=[error_msg]
            )
