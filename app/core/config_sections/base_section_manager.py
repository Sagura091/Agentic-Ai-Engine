"""
🚀 Base Configuration Section Manager

Provides a base implementation for configuration section managers with
common functionality and patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import structlog

from ..global_config_manager import ConfigurationSection, ConfigurationSectionManager, UpdateResult

logger = structlog.get_logger(__name__)


class BaseConfigurationSectionManager(ConfigurationSectionManager):
    """
    Base implementation for configuration section managers.
    
    Provides common functionality and patterns that can be reused
    across different section managers.
    """
    
    def __init__(self):
        """Initialize the base section manager."""
        self._initialized = False
        self._validation_rules: Dict[str, Any] = {}
        self._current_config: Dict[str, Any] = {}
        logger.info(f"🔧 {self.__class__.__name__} initialized")
    
    async def initialize(self) -> None:
        """Initialize the section manager."""
        if self._initialized:
            return
        
        await self._load_initial_configuration()
        await self._setup_validation_rules()
        self._initialized = True
        logger.info(f"✅ {self.__class__.__name__} fully initialized")
    
    @property
    @abstractmethod
    def section_name(self) -> ConfigurationSection:
        """The configuration section this manager handles."""
        pass
    
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration changes using common validation patterns.
        
        Args:
            config: Configuration changes to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Basic validation
            errors.extend(await self._validate_required_fields(config))
            errors.extend(await self._validate_field_types(config))
            errors.extend(await self._validate_field_ranges(config))
            errors.extend(await self._validate_field_formats(config))
            
            # Custom validation
            errors.extend(await self._validate_custom_rules(config))
            
            if errors:
                logger.warning(f"Configuration validation failed for {self.section_name.value}", 
                             errors=errors)
            else:
                logger.info(f"Configuration validation passed for {self.section_name.value}")
            
            return errors
            
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg, section=self.section_name.value)
            return [error_msg]
    
    async def apply_configuration(
        self, 
        config: Dict[str, Any], 
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """
        Apply configuration changes with common error handling.
        
        Args:
            config: New configuration values
            previous_config: Previous configuration for rollback
            
        Returns:
            UpdateResult with success/failure information
        """
        try:
            logger.info(f"🔄 Applying configuration for {self.section_name.value}", 
                       changes=list(config.keys()))
            
            # Create rollback data
            rollback_data = await self._create_rollback_data(config, previous_config)
            
            # Apply the configuration
            warnings = await self._apply_configuration_changes(config, previous_config)
            
            # Update current configuration
            self._current_config.update(config)
            
            logger.info(f"✅ Configuration applied successfully for {self.section_name.value}")
            
            return UpdateResult(
                success=True,
                section=self.section_name.value,
                changes_applied=config,
                warnings=warnings,
                rollback_data=rollback_data
            )
            
        except Exception as e:
            error_msg = f"Failed to apply configuration: {str(e)}"
            logger.error(error_msg, section=self.section_name.value)
            
            return UpdateResult(
                success=False,
                section=self.section_name.value,
                errors=[error_msg]
            )
    
    async def rollback_configuration(self, rollback_data: Dict[str, Any]) -> bool:
        """
        Rollback configuration changes with common error handling.
        
        Args:
            rollback_data: Data needed to rollback the changes
            
        Returns:
            True if rollback was successful
        """
        try:
            logger.info(f"🔄 Rolling back configuration for {self.section_name.value}")
            
            success = await self._perform_rollback(rollback_data)
            
            if success:
                logger.info(f"✅ Configuration rollback successful for {self.section_name.value}")
            else:
                logger.error(f"❌ Configuration rollback failed for {self.section_name.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error during rollback for {self.section_name.value}: {str(e)}")
            return False
    
    # Abstract methods for subclasses to implement
    
    @abstractmethod
    async def _load_initial_configuration(self) -> None:
        """Load the initial configuration for this section."""
        pass
    
    @abstractmethod
    async def _setup_validation_rules(self) -> None:
        """Setup validation rules for this section."""
        pass
    
    @abstractmethod
    async def _apply_configuration_changes(
        self, 
        config: Dict[str, Any], 
        previous_config: Dict[str, Any]
    ) -> List[str]:
        """
        Apply the actual configuration changes to the target system.
        
        Returns:
            List of warnings (empty if no warnings)
        """
        pass
    
    @abstractmethod
    async def _perform_rollback(self, rollback_data: Dict[str, Any]) -> bool:
        """
        Perform the actual rollback operation.
        
        Returns:
            True if rollback was successful
        """
        pass
    
    # Helper methods for common validation patterns
    
    async def _validate_required_fields(self, config: Dict[str, Any]) -> List[str]:
        """Validate that required fields are present."""
        errors = []
        required_fields = self._validation_rules.get("required_fields", [])
        
        for field in required_fields:
            if field not in config or config[field] is None:
                errors.append(f"Required field '{field}' is missing or null")
        
        return errors
    
    async def _validate_field_types(self, config: Dict[str, Any]) -> List[str]:
        """Validate field types."""
        errors = []
        field_types = self._validation_rules.get("field_types", {})
        
        for field, expected_type in field_types.items():
            if field in config and not isinstance(config[field], expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
        
        return errors
    
    async def _validate_field_ranges(self, config: Dict[str, Any]) -> List[str]:
        """Validate numeric field ranges."""
        errors = []
        field_ranges = self._validation_rules.get("field_ranges", {})
        
        for field, (min_val, max_val) in field_ranges.items():
            if field in config:
                value = config[field]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        errors.append(f"Field '{field}' must be between {min_val} and {max_val}")
        
        return errors
    
    async def _validate_field_formats(self, config: Dict[str, Any]) -> List[str]:
        """Validate field formats using regex or custom validators."""
        errors = []
        # Subclasses can override this for specific format validation
        return errors
    
    async def _validate_custom_rules(self, config: Dict[str, Any]) -> List[str]:
        """Validate custom business rules."""
        errors = []
        # Subclasses can override this for custom validation
        return errors
    
    async def _create_rollback_data(
        self, 
        config: Dict[str, Any], 
        previous_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create rollback data for the configuration changes."""
        # Default implementation: store previous values for changed fields
        rollback_data = {}
        for key in config.keys():
            if key in previous_config:
                rollback_data[key] = previous_config[key]
            else:
                rollback_data[key] = None  # Field was added, so remove it on rollback
        
        return rollback_data
