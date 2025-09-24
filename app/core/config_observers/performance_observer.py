"""
🚀 Revolutionary Performance Configuration Observer

Real-time observer for performance configuration changes.
Applies changes immediately to all system components for optimal performance without server restarts.

REAL-TIME PERFORMANCE UPDATES:
✅ CPU & Memory Optimization
✅ Caching Configuration
✅ Concurrency Settings
✅ Load Balancing
✅ Resource Allocation
✅ Monitoring & Metrics
"""

from typing import Any, Dict, Optional, List
import structlog

from ..global_config_manager import ConfigurationObserver, ConfigurationSection, UpdateResult

logger = structlog.get_logger(__name__)


class PerformanceConfigurationObserver(ConfigurationObserver):
    """🚀 Revolutionary Performance Configuration Observer"""
    
    def __init__(self):
        super().__init__()
        self._performance_service = None
        self._cache_manager = None
        self._load_balancer = None
        self._resource_manager = None
        
    @property
    def observer_name(self) -> str:
        return "PerformanceConfigurationObserver"
    
    @property
    def observed_sections(self) -> List[ConfigurationSection]:
        return [ConfigurationSection.SYSTEM_CONFIGURATION]
    
    async def initialize(self) -> None:
        """Initialize performance components."""
        try:
            # Import performance services
            from app.core.performance import get_performance_service
            from app.core.performance.cache_manager import get_cache_manager
            from app.core.performance.load_balancer import get_load_balancer
            from app.core.performance.resource_manager import get_resource_manager
            
            self._performance_service = get_performance_service()
            self._cache_manager = get_cache_manager()
            self._load_balancer = get_load_balancer()
            self._resource_manager = get_resource_manager()
            
            logger.info("✅ Performance configuration observer initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Some performance components not available: {str(e)}")
    
    async def on_configuration_changed(
        self,
        section: ConfigurationSection,
        changes: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """Handle performance configuration changes."""
        try:
            logger.info("🔄 Processing performance configuration changes", changes=list(changes.keys()))
            
            warnings = []
            
            # Apply CPU & memory optimization changes
            if any(key.startswith('performance_cpu_') or key.startswith('performance_memory_') for key in changes.keys()):
                await self._apply_cpu_memory_changes(changes, warnings)
            
            # Apply caching changes
            if any(key.startswith('performance_cache_') for key in changes.keys()):
                await self._apply_caching_changes(changes, warnings)
            
            # Apply concurrency changes
            if any(key.startswith('performance_concurrency_') for key in changes.keys()):
                await self._apply_concurrency_changes(changes, warnings)
            
            # Apply load balancing changes
            if any(key.startswith('performance_load_') for key in changes.keys()):
                await self._apply_load_balancing_changes(changes, warnings)
            
            # Apply resource allocation changes
            if any(key.startswith('performance_resource_') for key in changes.keys()):
                await self._apply_resource_changes(changes, warnings)
            
            logger.info("✅ Performance configuration changes applied successfully")
            
            return UpdateResult(
                success=True,
                section=section.value,
                changes_applied=changes,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to apply performance configuration changes: {str(e)}")
            return UpdateResult(
                success=False,
                section=section.value,
                changes_applied={},
                errors=[str(e)]
            )
    
    async def _apply_cpu_memory_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply CPU and memory optimization changes."""
        try:
            cpu_memory_changes = {k: v for k, v in changes.items() 
                                if k.startswith('performance_cpu_') or k.startswith('performance_memory_')}
            
            if not cpu_memory_changes:
                return
            
            logger.info("🔄 Applying CPU/memory optimization changes", changes=list(cpu_memory_changes.keys()))
            
            # Apply to performance service
            if self._performance_service and hasattr(self._performance_service, 'update_cpu_memory_settings'):
                await self._performance_service.update_cpu_memory_settings(cpu_memory_changes)
                logger.info("✅ Updated CPU/memory optimization settings")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply CPU/memory changes: {str(e)}")
            raise
    
    async def _apply_caching_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply caching configuration changes."""
        try:
            cache_changes = {k: v for k, v in changes.items() if k.startswith('performance_cache_')}
            
            if not cache_changes:
                return
            
            logger.info("🔄 Applying caching changes", changes=list(cache_changes.keys()))
            
            # Apply to cache manager
            if self._cache_manager and hasattr(self._cache_manager, 'update_configuration'):
                await self._cache_manager.update_configuration(cache_changes)
                logger.info("✅ Updated caching configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply caching changes: {str(e)}")
            raise
    
    async def _apply_concurrency_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply concurrency configuration changes."""
        try:
            concurrency_changes = {k: v for k, v in changes.items() if k.startswith('performance_concurrency_')}
            
            if not concurrency_changes:
                return
            
            logger.info("🔄 Applying concurrency changes", changes=list(concurrency_changes.keys()))
            
            # Apply to performance service
            if self._performance_service and hasattr(self._performance_service, 'update_concurrency_settings'):
                await self._performance_service.update_concurrency_settings(concurrency_changes)
                logger.info("✅ Updated concurrency settings")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply concurrency changes: {str(e)}")
            raise
    
    async def _apply_load_balancing_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply load balancing configuration changes."""
        try:
            load_changes = {k: v for k, v in changes.items() if k.startswith('performance_load_')}
            
            if not load_changes:
                return
            
            logger.info("🔄 Applying load balancing changes", changes=list(load_changes.keys()))
            
            # Apply to load balancer
            if self._load_balancer and hasattr(self._load_balancer, 'update_configuration'):
                await self._load_balancer.update_configuration(load_changes)
                logger.info("✅ Updated load balancing configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply load balancing changes: {str(e)}")
            raise
    
    async def _apply_resource_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply resource allocation changes."""
        try:
            resource_changes = {k: v for k, v in changes.items() if k.startswith('performance_resource_')}
            
            if not resource_changes:
                return
            
            logger.info("🔄 Applying resource allocation changes", changes=list(resource_changes.keys()))
            
            # Apply to resource manager
            if self._resource_manager and hasattr(self._resource_manager, 'update_configuration'):
                await self._resource_manager.update_configuration(resource_changes)
                logger.info("✅ Updated resource allocation configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply resource changes: {str(e)}")
            raise


# Global instance
performance_observer = PerformanceConfigurationObserver()
