"""
Admin Settings Service.

This service handles the storage, retrieval, and application of admin settings
to the running system components. It bridges the gap between the admin interface
and the actual system configuration.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.models.database.base import get_database_session
from app.models.admin_settings import AdminSetting, AdminSettingHistory, SystemConfigurationCache
from app.config.settings import get_settings

logger = get_logger()


class AdminSettingsService:
    """
    Service for managing admin settings with database persistence and system application.
    
    This service provides:
    - Database storage and retrieval of settings
    - Setting validation and type conversion
    - System configuration cache management
    - Automatic application of settings to running systems
    - Complete audit trail of all changes
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False
    
    async def initialize(self) -> None:
        """Initialize the settings service and load cache."""
        try:
            await self._load_settings_cache()
            logger.info(
                "Admin settings service initialized",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.admin_settings_service"
            )
        except Exception as e:
            logger.error(
                "Failed to initialize admin settings service",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.admin_settings_service",
                error=e
            )
            raise
    
    async def get_setting(
        self, 
        category: str, 
        key: str, 
        default: Any = None
    ) -> Any:
        """
        Get a single setting value.
        
        Args:
            category: Setting category
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        try:
            async for session in get_database_session():
                result = await session.execute(
                    select(AdminSetting).where(
                        AdminSetting.category == category,
                        AdminSetting.key == key,
                        AdminSetting.is_active == True
                    )
                )
                setting = result.scalar_one_or_none()
                
                if setting:
                    return setting.value
                return default

        except Exception as e:
            logger.error(
                "Failed to get setting",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.admin_settings_service",
                data={"category": category, "key": key},
                error=e
            )
            return default
    
    async def set_setting(
        self,
        category: str,
        key: str,
        value: Any,
        user_id: UUID,
        setting_type: str = "string",
        description: Optional[str] = None,
        requires_restart: bool = False,
        validation_rules: Optional[Dict[str, Any]] = None,
        enum_values: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Set a setting value with full audit trail.
        
        Args:
            category: Setting category
            key: Setting key
            value: New value
            user_id: User making the change
            setting_type: Type of the setting
            description: Setting description
            requires_restart: Whether setting requires restart
            validation_rules: Validation rules
            enum_values: Enum values if applicable
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            async for session in get_database_session():
                # Check if setting exists
                result = await session.execute(
                    select(AdminSetting).where(
                        AdminSetting.category == category,
                        AdminSetting.key == key
                    )
                )
                existing_setting = result.scalar_one_or_none()
                
                old_value = None
                if existing_setting:
                    # Update existing setting
                    old_value = existing_setting.value
                    existing_setting.value = value
                    existing_setting.updated_at = datetime.utcnow()
                    existing_setting.updated_by = user_id
                    
                    if description:
                        existing_setting.description = description
                    if validation_rules:
                        existing_setting.validation_rules = validation_rules
                    if enum_values:
                        existing_setting.enum_values = enum_values
                    
                    setting_id = existing_setting.id
                else:
                    # Create new setting
                    new_setting = AdminSetting(
                        category=category,
                        key=key,
                        value=value,
                        default_value=value,  # First value becomes default
                        setting_type=setting_type,
                        description=description or f"{category}.{key}",
                        requires_restart=requires_restart,
                        validation_rules=validation_rules or {},
                        enum_values=enum_values,
                        updated_by=user_id
                    )
                    session.add(new_setting)
                    await session.flush()  # Get the ID
                    setting_id = new_setting.id
                
                # Create history record
                history = AdminSettingHistory(
                    setting_id=setting_id,
                    category=category,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    changed_by=user_id,
                    system_restart_required=requires_restart
                )
                session.add(history)
                
                await session.commit()
                
                # Update system configuration cache
                await self._update_configuration_cache(category)
                
                # Apply setting to running system
                await self._apply_setting_to_system(category, key, value)

                logger.info(
                    "Setting updated successfully",
                    LogCategory.CONFIGURATION_MANAGEMENT,
                    "app.services.admin_settings_service",
                    data={
                        "category": category,
                        "key": key,
                        "old_value": old_value,
                        "new_value": value,
                        "user_id": str(user_id),
                        "requires_restart": requires_restart
                    }
                )

                return True, None

        except Exception as e:
            logger.error(
                "Failed to set setting",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "app.services.admin_settings_service",
                data={"category": category, "key": key},
                error=e
            )
            return False, str(e)
    
    async def get_category_settings(self, category: str) -> Dict[str, Any]:
        """
        Get all settings for a category.
        
        Args:
            category: Setting category
            
        Returns:
            Dictionary of settings
        """
        try:
            async for session in get_database_session():
                result = await session.execute(
                    select(AdminSetting).where(
                        AdminSetting.category == category,
                        AdminSetting.is_active == True
                    )
                )
                settings = result.scalars().all()
                
                return {
                    setting.key: {
                        "value": setting.value,
                        "default_value": setting.default_value,
                        "type": setting.setting_type,
                        "description": setting.description,
                        "requires_restart": setting.requires_restart,
                        "validation_rules": setting.validation_rules,
                        "enum_values": setting.enum_values,
                        "updated_at": setting.updated_at.isoformat() if setting.updated_at else None
                    }
                    for setting in settings
                }

        except Exception as e:
            logger.error(
                "Failed to get category settings",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.admin_settings_service",
                data={"category": category},
                error=e
            )
            return {}
    
    async def _load_settings_cache(self) -> None:
        """Load all settings into memory cache for fast access."""
        try:
            async for session in get_database_session():
                result = await session.execute(
                    select(AdminSetting).where(AdminSetting.is_active == True)
                )
                settings = result.scalars().all()
                
                self._cache = {}
                for setting in settings:
                    if setting.category not in self._cache:
                        self._cache[setting.category] = {}
                    self._cache[setting.category][setting.key] = setting.value

                self._cache_loaded = True
                logger.info(
                    "Settings cache loaded",
                    LogCategory.SERVICE_OPERATIONS,
                    "app.services.admin_settings_service",
                    data={"categories": len(self._cache)}
                )

        except Exception as e:
            logger.error(
                "Failed to load settings cache",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.admin_settings_service",
                error=e
            )
            raise
    
    async def _update_configuration_cache(self, category: str) -> None:
        """Update the system configuration cache for a category."""
        try:
            # Get all settings for the category
            category_settings = await self.get_category_settings(category)
            
            # Create configuration hash for change detection
            config_json = json.dumps(category_settings, sort_keys=True)
            config_hash = hashlib.sha256(config_json.encode()).hexdigest()
            
            async for session in get_database_session():
                # Check if cache entry exists
                result = await session.execute(
                    select(SystemConfigurationCache).where(
                        SystemConfigurationCache.component_name == category
                    )
                )
                cache_entry = result.scalar_one_or_none()
                
                if cache_entry:
                    # Update existing cache
                    cache_entry.configuration = category_settings
                    cache_entry.configuration_hash = config_hash
                    cache_entry.updated_at = datetime.utcnow()
                else:
                    # Create new cache entry
                    cache_entry = SystemConfigurationCache(
                        component_name=category,
                        configuration=category_settings,
                        configuration_hash=config_hash
                    )
                    session.add(cache_entry)
                
                await session.commit()

                logger.info(
                    "Configuration cache updated",
                    LogCategory.CONFIGURATION_MANAGEMENT,
                    "app.services.admin_settings_service",
                    data={"category": category, "hash": config_hash[:8]}
                )

        except Exception as e:
            logger.error(
                "Failed to update configuration cache",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "app.services.admin_settings_service",
                data={"category": category},
                error=e
            )
    
    async def _apply_setting_to_system(self, category: str, key: str, value: Any) -> None:
        """Apply a setting change to the running system."""
        try:
            logger.info(
                "Applying setting to system",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "app.services.admin_settings_service",
                data={"category": category, "key": key, "value": value}
            )

            # Apply settings based on category
            if category == "rag_configuration":
                await self._apply_rag_setting(key, value)
            elif category == "agent_management":
                await self._apply_agent_setting(key, value)
            elif category == "llm_providers":
                await self._apply_llm_setting(key, value)
            elif category == "system_configuration":
                await self._apply_system_setting(key, value)
            else:
                logger.info(
                    "No specific application handler for category",
                    LogCategory.CONFIGURATION_MANAGEMENT,
                    "app.services.admin_settings_service",
                    data={"category": category}
                )

        except Exception as e:
            logger.error(
                "Failed to apply setting to system",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "app.services.admin_settings_service",
                data={"category": category, "key": key},
                error=e
            )

    async def _apply_rag_setting(self, key: str, value: Any) -> None:
        """Apply RAG-specific settings to the RAG system."""
        try:
            from app.services.rag_settings_applicator import get_rag_settings_applicator

            # Get all RAG settings to apply as a batch
            rag_settings = await self.get_category_settings("rag_configuration")

            # Apply to RAG system
            applicator = await get_rag_settings_applicator()
            success = await applicator.apply_rag_settings(rag_settings)

            if success:
                logger.info(
                    "RAG setting applied successfully",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.admin_settings_service",
                    data={"key": key, "value": value}
                )
            else:
                logger.error(
                    "Failed to apply RAG setting",
                    LogCategory.RAG_OPERATIONS,
                    "app.services.admin_settings_service",
                    data={"key": key, "value": value}
                )

        except Exception as e:
            logger.error(
                "Failed to apply RAG setting",
                LogCategory.RAG_OPERATIONS,
                "app.services.admin_settings_service",
                data={"key": key},
                error=e
            )

    async def _apply_agent_setting(self, key: str, value: Any) -> None:
        """Apply agent management settings."""
        try:
            # TODO: Implement agent setting application
            logger.info(
                "Agent setting would be applied",
                LogCategory.AGENT_OPERATIONS,
                "app.services.admin_settings_service",
                data={"key": key, "value": value}
            )
        except Exception as e:
            logger.error(
                "Failed to apply agent setting",
                LogCategory.AGENT_OPERATIONS,
                "app.services.admin_settings_service",
                data={"key": key},
                error=e
            )

    async def _apply_llm_setting(self, key: str, value: Any) -> None:
        """Apply LLM provider settings."""
        try:
            # TODO: Implement LLM setting application
            logger.info(
                "LLM setting would be applied",
                LogCategory.LLM_OPERATIONS,
                "app.services.admin_settings_service",
                data={"key": key, "value": value}
            )
        except Exception as e:
            logger.error(
                "Failed to apply LLM setting",
                LogCategory.LLM_OPERATIONS,
                "app.services.admin_settings_service",
                data={"key": key},
                error=e
            )

    async def _apply_system_setting(self, key: str, value: Any) -> None:
        """Apply system configuration settings."""
        try:
            # TODO: Implement system setting application
            logger.info(
                "System setting would be applied",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "app.services.admin_settings_service",
                data={"key": key, "value": value}
            )
        except Exception as e:
            logger.error(
                "Failed to apply system setting",
                LogCategory.CONFIGURATION_MANAGEMENT,
                "app.services.admin_settings_service",
                data={"key": key},
                error=e
            )


# Global service instance
_admin_settings_service: Optional[AdminSettingsService] = None


async def get_admin_settings_service() -> AdminSettingsService:
    """Get the global admin settings service instance."""
    global _admin_settings_service
    if _admin_settings_service is None:
        _admin_settings_service = AdminSettingsService()
        await _admin_settings_service.initialize()
    return _admin_settings_service
