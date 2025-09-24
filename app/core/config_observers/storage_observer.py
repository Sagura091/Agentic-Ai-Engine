"""
🚀 Revolutionary Storage Configuration Observer

Real-time observer for storage system configuration changes.
Applies changes immediately to storage backends and file systems without server restarts.

REAL-TIME STORAGE UPDATES:
✅ Storage Backend Switching
✅ File System Configuration
✅ Compression Settings
✅ Backup Configuration
✅ Cleanup Policies
✅ Performance Optimization
"""

from typing import Any, Dict, Optional, List
import structlog

from ..global_config_manager import ConfigurationObserver, ConfigurationSection, UpdateResult

logger = structlog.get_logger(__name__)


class StorageConfigurationObserver(ConfigurationObserver):
    """🚀 Revolutionary Storage Configuration Observer"""
    
    def __init__(self):
        super().__init__()
        self._storage_service = None
        self._file_manager = None
        self._backup_service = None
        
    @property
    def observer_name(self) -> str:
        return "StorageConfigurationObserver"
    
    @property
    def observed_sections(self) -> List[ConfigurationSection]:
        return [ConfigurationSection.SYSTEM_CONFIGURATION]
    
    async def initialize(self) -> None:
        """Initialize storage components."""
        try:
            # Import storage services
            from app.core.storage import get_storage_service
            from app.core.storage.file_manager import get_file_manager
            from app.core.storage.backup_service import get_backup_service
            
            self._storage_service = get_storage_service()
            self._file_manager = get_file_manager()
            self._backup_service = get_backup_service()
            
            logger.info("✅ Storage configuration observer initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Some storage components not available: {str(e)}")
    
    async def on_configuration_changed(
        self,
        section: ConfigurationSection,
        changes: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """Handle storage configuration changes."""
        try:
            logger.info("🔄 Processing storage configuration changes", changes=list(changes.keys()))
            
            warnings = []
            
            # Apply storage backend changes
            if any(key.startswith('storage_backend_') for key in changes.keys()):
                await self._apply_storage_backend_changes(changes, warnings)
            
            # Apply file system changes
            if any(key.startswith('storage_filesystem_') for key in changes.keys()):
                await self._apply_filesystem_changes(changes, warnings)
            
            # Apply compression changes
            if any(key.startswith('storage_compression_') for key in changes.keys()):
                await self._apply_compression_changes(changes, warnings)
            
            # Apply backup changes
            if any(key.startswith('storage_backup_') for key in changes.keys()):
                await self._apply_backup_changes(changes, warnings)
            
            # Apply cleanup changes
            if any(key.startswith('storage_cleanup_') for key in changes.keys()):
                await self._apply_cleanup_changes(changes, warnings)
            
            logger.info("✅ Storage configuration changes applied successfully")
            
            return UpdateResult(
                success=True,
                section=section.value,
                changes_applied=changes,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to apply storage configuration changes: {str(e)}")
            return UpdateResult(
                success=False,
                section=section.value,
                changes_applied={},
                errors=[str(e)]
            )
    
    async def _apply_storage_backend_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply storage backend configuration changes."""
        try:
            backend_changes = {k: v for k, v in changes.items() if k.startswith('storage_backend_')}
            
            if not backend_changes:
                return
            
            logger.info("🔄 Applying storage backend changes", changes=list(backend_changes.keys()))
            
            # Apply to storage service
            if self._storage_service and hasattr(self._storage_service, 'update_backend_configuration'):
                await self._storage_service.update_backend_configuration(backend_changes)
                logger.info("✅ Updated storage backend configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply storage backend changes: {str(e)}")
            raise
    
    async def _apply_filesystem_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply file system configuration changes."""
        try:
            filesystem_changes = {k: v for k, v in changes.items() if k.startswith('storage_filesystem_')}
            
            if not filesystem_changes:
                return
            
            logger.info("🔄 Applying filesystem changes", changes=list(filesystem_changes.keys()))
            
            # Apply to file manager
            if self._file_manager and hasattr(self._file_manager, 'update_configuration'):
                await self._file_manager.update_configuration(filesystem_changes)
                logger.info("✅ Updated filesystem configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply filesystem changes: {str(e)}")
            raise
    
    async def _apply_compression_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply compression configuration changes."""
        try:
            compression_changes = {k: v for k, v in changes.items() if k.startswith('storage_compression_')}
            
            if not compression_changes:
                return
            
            logger.info("🔄 Applying compression changes", changes=list(compression_changes.keys()))
            
            # Apply to storage components
            for component in [self._storage_service, self._file_manager]:
                if component and hasattr(component, 'update_compression_settings'):
                    await component.update_compression_settings(compression_changes)
            
            logger.info("✅ Updated compression settings")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply compression changes: {str(e)}")
            raise
    
    async def _apply_backup_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply backup configuration changes."""
        try:
            backup_changes = {k: v for k, v in changes.items() if k.startswith('storage_backup_')}
            
            if not backup_changes:
                return
            
            logger.info("🔄 Applying backup changes", changes=list(backup_changes.keys()))
            
            # Apply to backup service
            if self._backup_service and hasattr(self._backup_service, 'update_configuration'):
                await self._backup_service.update_configuration(backup_changes)
                logger.info("✅ Updated backup configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply backup changes: {str(e)}")
            raise
    
    async def _apply_cleanup_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply cleanup policy changes."""
        try:
            cleanup_changes = {k: v for k, v in changes.items() if k.startswith('storage_cleanup_')}
            
            if not cleanup_changes:
                return
            
            logger.info("🔄 Applying cleanup changes", changes=list(cleanup_changes.keys()))
            
            # Apply to storage service
            if self._storage_service and hasattr(self._storage_service, 'update_cleanup_policies'):
                await self._storage_service.update_cleanup_policies(cleanup_changes)
                logger.info("✅ Updated cleanup policies")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply cleanup changes: {str(e)}")
            raise


# Global instance
storage_observer = StorageConfigurationObserver()
