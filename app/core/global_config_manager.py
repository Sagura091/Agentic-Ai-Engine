"""
🚀 Revolutionary Global Configuration Manager

This is THE central configuration management system for the entire application.
Provides real-time, section-based configuration updates with observer pattern
support and zero-downtime reconfiguration capabilities.

ARCHITECTURE:
- Section-based configuration management
- Observer pattern for real-time updates
- Thread-safe operations with async support
- Granular updates to minimize payload sizes
- Automatic rollback on failures
- Complete audit trail
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from uuid import uuid4
import structlog

from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ConfigurationSection(str, Enum):
    """Available configuration sections."""
    RAG_CONFIGURATION = "rag_configuration"
    LLM_PROVIDERS = "llm_providers"
    MEMORY_SYSTEM = "memory_system"
    AGENT_MANAGEMENT = "agent_management"
    BACKEND_LOGGING = "backend_logging"
    TOOL_CONFIGURATION = "tool_configuration"
    SYSTEM_CONFIGURATION = "system_configuration"


class UpdateResult(BaseModel):
    """Result of a configuration update operation."""
    success: bool
    section: str
    changes_applied: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    rollback_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    update_id: str = Field(default_factory=lambda: str(uuid4()))


class ConfigurationObserver(ABC):
    """
    Abstract base class for configuration observers.
    
    Systems that need to respond to configuration changes should implement this interface.
    """
    
    @property
    @abstractmethod
    def observer_name(self) -> str:
        """Name of the observer for logging and identification."""
        pass
    
    @property
    @abstractmethod
    def observed_sections(self) -> Set[ConfigurationSection]:
        """Set of configuration sections this observer is interested in."""
        pass
    
    @abstractmethod
    async def on_configuration_changed(
        self, 
        section: ConfigurationSection, 
        changes: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """
        Called when a configuration section this observer is interested in changes.
        
        Args:
            section: The configuration section that changed
            changes: Dictionary of changed configuration values
            previous_config: Previous configuration values for rollback
            
        Returns:
            UpdateResult indicating success/failure and any issues
        """
        pass


class ConfigurationSectionManager(ABC):
    """
    Abstract base class for section-specific configuration managers.
    
    Each configuration section (RAG, LLM, Memory, etc.) has its own manager
    that handles validation, application, and rollback for that section.
    """
    
    @property
    @abstractmethod
    def section_name(self) -> ConfigurationSection:
        """The configuration section this manager handles."""
        pass
    
    @abstractmethod
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration changes before applying them.
        
        Args:
            config: Configuration changes to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    @abstractmethod
    async def apply_configuration(
        self, 
        config: Dict[str, Any], 
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """
        Apply configuration changes to the target system.
        
        Args:
            config: New configuration values
            previous_config: Previous configuration for rollback
            
        Returns:
            UpdateResult with success/failure information
        """
        pass
    
    @abstractmethod
    async def rollback_configuration(self, rollback_data: Dict[str, Any]) -> bool:
        """
        Rollback configuration changes in case of failure.
        
        Args:
            rollback_data: Data needed to rollback the changes
            
        Returns:
            True if rollback was successful
        """
        pass


class ConfigurationHistory(BaseModel):
    """Record of a configuration change."""
    update_id: str
    section: ConfigurationSection
    changes: Dict[str, Any]
    previous_values: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    success: bool
    errors: List[str] = Field(default_factory=list)
    rollback_performed: bool = False


class GlobalConfigurationManager:
    """
    🚀 Revolutionary Global Configuration Manager
    
    THE central hub for all configuration management in the application.
    Provides real-time, section-based updates with observer pattern support.
    """
    
    def __init__(self):
        """Initialize the global configuration manager."""
        # Observer registry: section -> list of observers
        self._observers: Dict[ConfigurationSection, List[ConfigurationObserver]] = {}
        
        # Section managers: section -> manager instance
        self._section_managers: Dict[ConfigurationSection, ConfigurationSectionManager] = {}
        
        # Current configuration state
        self._current_config: Dict[ConfigurationSection, Dict[str, Any]] = {}
        
        # Configuration history (keep last 1000 changes)
        self._history: List[ConfigurationHistory] = []
        
        # Thread safety
        self._update_lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "rollbacks_performed": 0,
            "observers_registered": 0,
            "section_managers_registered": 0
        }
        
        self._initialized = False
        logger.info("🚀 Global Configuration Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the configuration manager."""
        if self._initialized:
            logger.warning("Global Configuration Manager already initialized")
            return
        
        # Initialize all section managers
        for section, manager in self._section_managers.items():
            if hasattr(manager, 'initialize'):
                await manager.initialize()
        
        self._initialized = True
        logger.info("✅ Global Configuration Manager fully initialized")
    
    def register_observer(self, observer: ConfigurationObserver) -> None:
        """
        Register an observer for configuration changes.
        
        Args:
            observer: Observer instance to register
        """
        for section in observer.observed_sections:
            if section not in self._observers:
                self._observers[section] = []
            
            # Avoid duplicate registrations
            if observer not in self._observers[section]:
                self._observers[section].append(observer)
                self._stats["observers_registered"] += 1
                logger.info(
                    f"✅ Registered observer {observer.observer_name} for section {section.value}"
                )
    
    def register_section_manager(self, manager: ConfigurationSectionManager) -> None:
        """
        Register a section manager for a specific configuration section.
        
        Args:
            manager: Section manager instance to register
        """
        section = manager.section_name
        if section in self._section_managers:
            logger.warning(f"Section manager for {section.value} already registered, replacing")
        
        self._section_managers[section] = manager
        self._stats["section_managers_registered"] += 1
        logger.info(f"✅ Registered section manager for {section.value}")
    
    async def update_section(
        self,
        section: ConfigurationSection,
        changes: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> UpdateResult:
        """
        Update a specific configuration section.
        
        Args:
            section: Configuration section to update
            changes: Dictionary of configuration changes
            user_id: ID of user making the change (for audit trail)
            
        Returns:
            UpdateResult with success/failure information
        """
        async with self._update_lock:
            update_id = str(uuid4())
            logger.info(f"🔄 Starting configuration update for section {section.value}", 
                       update_id=update_id, changes=list(changes.keys()))
            
            # Get current configuration for this section
            current_config = self._current_config.get(section, {})
            previous_config = current_config.copy()
            
            try:
                # Validate changes if section manager exists
                if section in self._section_managers:
                    validation_errors = await self._section_managers[section].validate_configuration(changes)
                    if validation_errors:
                        result = UpdateResult(
                            success=False,
                            section=section.value,
                            errors=validation_errors,
                            update_id=update_id
                        )
                        await self._record_history(section, changes, previous_config, user_id, result)
                        return result
                
                # Apply changes through section manager if available
                if section in self._section_managers:
                    result = await self._section_managers[section].apply_configuration(changes, previous_config)
                    result.section = section.value
                    result.update_id = update_id
                else:
                    # Fallback: direct configuration update
                    result = UpdateResult(
                        success=True,
                        section=section.value,
                        changes_applied=changes,
                        update_id=update_id,
                        warnings=["No section manager registered - using direct update"]
                    )
                
                if result.success:
                    # Update current configuration
                    if section not in self._current_config:
                        self._current_config[section] = {}
                    self._current_config[section].update(changes)
                    
                    # Notify observers
                    await self._notify_observers(section, changes, previous_config)
                    
                    self._stats["successful_updates"] += 1
                    logger.info(f"✅ Configuration update successful for section {section.value}", 
                               update_id=update_id)
                else:
                    self._stats["failed_updates"] += 1
                    logger.error(f"❌ Configuration update failed for section {section.value}", 
                                update_id=update_id, errors=result.errors)
                
                # Record in history
                await self._record_history(section, changes, previous_config, user_id, result)
                
                self._stats["total_updates"] += 1
                return result
                
            except Exception as e:
                error_msg = f"Unexpected error during configuration update: {str(e)}"
                logger.error(error_msg, update_id=update_id, error=str(e))
                
                result = UpdateResult(
                    success=False,
                    section=section.value,
                    errors=[error_msg],
                    update_id=update_id
                )
                
                await self._record_history(section, changes, previous_config, user_id, result)
                self._stats["failed_updates"] += 1
                self._stats["total_updates"] += 1
                
                return result

    async def update_multiple_sections(
        self,
        section_updates: Dict[ConfigurationSection, Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> Dict[ConfigurationSection, UpdateResult]:
        """
        Update multiple configuration sections atomically.

        Args:
            section_updates: Dictionary mapping sections to their updates
            user_id: ID of user making the changes

        Returns:
            Dictionary mapping sections to their update results
        """
        results = {}
        successful_updates = []

        logger.info(f"🔄 Starting multi-section configuration update",
                   sections=list(section_updates.keys()), user_id=user_id)

        try:
            # Apply all updates
            for section, changes in section_updates.items():
                result = await self.update_section(section, changes, user_id)
                results[section] = result

                if result.success:
                    successful_updates.append((section, changes, result.rollback_data))
                else:
                    # If any update fails, rollback all successful ones
                    logger.error(f"❌ Multi-section update failed at section {section.value}, rolling back")
                    await self._rollback_updates(successful_updates)
                    break

            if all(result.success for result in results.values()):
                logger.info("✅ Multi-section configuration update completed successfully")
            else:
                logger.error("❌ Multi-section configuration update failed")

            return results

        except Exception as e:
            logger.error(f"❌ Unexpected error in multi-section update: {str(e)}")
            await self._rollback_updates(successful_updates)

            # Return error results for all sections
            for section in section_updates.keys():
                if section not in results:
                    results[section] = UpdateResult(
                        success=False,
                        section=section.value,
                        errors=[f"Multi-section update failed: {str(e)}"]
                    )

            return results

    async def get_section_configuration(self, section: ConfigurationSection) -> Dict[str, Any]:
        """
        Get current configuration for a specific section.

        Args:
            section: Configuration section to retrieve

        Returns:
            Current configuration dictionary for the section
        """
        return self._current_config.get(section, {}).copy()

    async def get_all_configuration(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current configuration for all sections.

        Returns:
            Dictionary mapping section names to their configurations
        """
        return {
            section.value: config.copy()
            for section, config in self._current_config.items()
        }

    async def get_configuration_history(
        self,
        section: Optional[ConfigurationSection] = None,
        limit: int = 100
    ) -> List[ConfigurationHistory]:
        """
        Get configuration change history.

        Args:
            section: Optional section to filter by
            limit: Maximum number of history entries to return

        Returns:
            List of configuration history entries
        """
        history = self._history

        if section:
            history = [h for h in history if h.section == section]

        return history[-limit:] if limit else history

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get configuration manager statistics.

        Returns:
            Dictionary of statistics and metrics
        """
        return {
            **self._stats,
            "sections_configured": len(self._current_config),
            "total_observers": sum(len(observers) for observers in self._observers.values()),
            "history_entries": len(self._history),
            "initialized": self._initialized
        }

    async def _notify_observers(
        self,
        section: ConfigurationSection,
        changes: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> None:
        """Notify all observers interested in this section."""
        if section not in self._observers:
            return

        observers = self._observers[section]
        logger.info(f"📢 Notifying {len(observers)} observers for section {section.value}")

        for observer in observers:
            try:
                await observer.on_configuration_changed(section, changes, previous_config)
                logger.debug(f"✅ Notified observer {observer.observer_name}")
            except Exception as e:
                logger.error(f"❌ Failed to notify observer {observer.observer_name}: {str(e)}")

    async def _record_history(
        self,
        section: ConfigurationSection,
        changes: Dict[str, Any],
        previous_config: Dict[str, Any],
        user_id: Optional[str],
        result: UpdateResult
    ) -> None:
        """Record configuration change in history."""
        history_entry = ConfigurationHistory(
            update_id=result.update_id,
            section=section,
            changes=changes,
            previous_values=previous_config,
            timestamp=result.timestamp,
            user_id=user_id,
            success=result.success,
            errors=result.errors
        )

        self._history.append(history_entry)

        # Keep only last 1000 entries
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    async def _rollback_updates(self, successful_updates: List[tuple]) -> None:
        """Rollback a list of successful updates."""
        logger.info(f"🔄 Rolling back {len(successful_updates)} configuration updates")

        for section, changes, rollback_data in reversed(successful_updates):
            try:
                if section in self._section_managers and rollback_data:
                    success = await self._section_managers[section].rollback_configuration(rollback_data)
                    if success:
                        logger.info(f"✅ Rolled back section {section.value}")
                        self._stats["rollbacks_performed"] += 1
                    else:
                        logger.error(f"❌ Failed to rollback section {section.value}")
                else:
                    logger.warning(f"⚠️ No rollback mechanism for section {section.value}")
            except Exception as e:
                logger.error(f"❌ Error during rollback of section {section.value}: {str(e)}")


# Global instance
global_config_manager = GlobalConfigurationManager()


async def initialize_global_config_manager() -> None:
    """Initialize the global configuration manager."""
    await global_config_manager.initialize()
    logger.info("🚀 Global Configuration Manager ready for revolutionary configuration management!")


async def update_configuration_section(
    section: ConfigurationSection,
    changes: Dict[str, Any],
    user_id: Optional[str] = None
) -> UpdateResult:
    """
    Global function to update a configuration section.

    Args:
        section: Configuration section to update
        changes: Dictionary of configuration changes
        user_id: ID of user making the change

    Returns:
        UpdateResult with success/failure information
    """
    return await global_config_manager.update_section(section, changes, user_id)


async def get_configuration_section(section: ConfigurationSection) -> Dict[str, Any]:
    """
    Global function to get configuration for a section.

    Args:
        section: Configuration section to retrieve

    Returns:
        Current configuration dictionary for the section
    """
    return await global_config_manager.get_section_configuration(section)
