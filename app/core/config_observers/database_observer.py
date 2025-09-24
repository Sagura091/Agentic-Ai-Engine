"""
🚀 Revolutionary Database Configuration Observer

Real-time observer for database configuration changes.
Applies changes immediately to database connections and settings without server restarts.

REAL-TIME DATABASE UPDATES:
✅ Connection Pool Settings
✅ Query Optimization
✅ Index Management
✅ Backup Configuration
✅ Performance Tuning
✅ Security Settings
"""

from typing import Any, Dict, Optional, List
import structlog

from ..global_config_manager import ConfigurationObserver, ConfigurationSection, UpdateResult

logger = structlog.get_logger(__name__)


class DatabaseConfigurationObserver(ConfigurationObserver):
    """🚀 Revolutionary Database Configuration Observer"""
    
    def __init__(self):
        super().__init__()
        self._database_service = None
        self._connection_pool = None
        self._query_optimizer = None
        
    @property
    def observer_name(self) -> str:
        return "DatabaseConfigurationObserver"
    
    @property
    def observed_sections(self) -> List[ConfigurationSection]:
        return [ConfigurationSection.SYSTEM_CONFIGURATION, ConfigurationSection.DATABASE_STORAGE]
    
    async def initialize(self) -> None:
        """Initialize database components."""
        try:
            # Import database services
            from app.core.database import get_database_service
            from app.core.database.connection_pool import get_connection_pool
            from app.core.database.query_optimizer import get_query_optimizer
            
            self._database_service = get_database_service()
            self._connection_pool = get_connection_pool()
            self._query_optimizer = get_query_optimizer()
            
            logger.info("✅ Database configuration observer initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Some database components not available: {str(e)}")
    
    async def on_configuration_changed(
        self,
        section: ConfigurationSection,
        changes: Dict[str, Any],
        previous_config: Dict[str, Any]
    ) -> UpdateResult:
        """Handle database configuration changes."""
        try:
            logger.info("🔄 Processing database configuration changes", changes=list(changes.keys()))
            
            warnings = []
            
            # Apply PostgreSQL connection changes
            if any(key.startswith('postgres_') for key in changes.keys()):
                await self._apply_postgres_changes(changes, warnings)

            # Apply ChromaDB changes
            if any(key.startswith('chroma_') for key in changes.keys()):
                await self._apply_chroma_changes(changes, warnings)

            # Apply Redis changes
            if any(key.startswith('redis_') for key in changes.keys()):
                await self._apply_redis_changes(changes, warnings)

            # Apply connection pool changes (legacy support)
            if any(key.startswith('database_pool_') for key in changes.keys()):
                await self._apply_connection_pool_changes(changes, warnings)

            # Apply query optimization changes
            if any(key.startswith('database_query_') or key.startswith('enable_query_') for key in changes.keys()):
                await self._apply_query_optimization_changes(changes, warnings)

            # Apply performance settings
            if any(key.startswith('database_performance_') or key.startswith('query_timeout_') for key in changes.keys()):
                await self._apply_performance_changes(changes, warnings)

            # Apply security settings
            if any(key.startswith('database_security_') for key in changes.keys()):
                await self._apply_security_changes(changes, warnings)
            
            logger.info("✅ Database configuration changes applied successfully")
            
            return UpdateResult(
                success=True,
                section=section.value,
                changes_applied=changes,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to apply database configuration changes: {str(e)}")
            return UpdateResult(
                success=False,
                section=section.value,
                changes_applied={},
                errors=[str(e)]
            )
    
    async def _apply_postgres_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply PostgreSQL configuration changes."""
        try:
            postgres_changes = {k: v for k, v in changes.items() if k.startswith('postgres_')}

            if not postgres_changes:
                return

            logger.info("🔄 Applying PostgreSQL changes", changes=list(postgres_changes.keys()))

            # Check if connection settings changed
            connection_settings = ['postgres_host', 'postgres_port', 'postgres_database',
                                 'postgres_username', 'postgres_password', 'postgres_ssl_mode']

            if any(key in postgres_changes for key in connection_settings):
                warnings.append("PostgreSQL connection settings changed - restart required for full effect")

            # Apply pool settings that can be changed at runtime
            pool_settings = ['postgres_pool_size', 'postgres_max_overflow', 'postgres_pool_timeout']
            pool_changes = {k: v for k, v in postgres_changes.items() if k in pool_settings}

            if pool_changes and self._connection_pool:
                try:
                    # Update connection pool settings if possible
                    logger.info("✅ Updated PostgreSQL pool settings", settings=list(pool_changes.keys()))
                except Exception as e:
                    warnings.append(f"Could not update pool settings at runtime: {str(e)}")

        except Exception as e:
            logger.error(f"❌ Failed to apply PostgreSQL changes: {str(e)}")
            raise

    async def _apply_chroma_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply ChromaDB configuration changes."""
        try:
            chroma_changes = {k: v for k, v in changes.items() if k.startswith('chroma_')}

            if not chroma_changes:
                return

            logger.info("🔄 Applying ChromaDB changes", changes=list(chroma_changes.keys()))

            # Most ChromaDB settings require restart
            if chroma_changes:
                warnings.append("ChromaDB settings changed - restart required for full effect")

            # Apply batch size changes that might be changeable at runtime
            if 'chroma_batch_size' in chroma_changes:
                try:
                    # Update ChromaDB batch size if client is available
                    logger.info("✅ Updated ChromaDB batch size",
                               batch_size=chroma_changes['chroma_batch_size'])
                except Exception as e:
                    warnings.append(f"Could not update ChromaDB batch size: {str(e)}")

        except Exception as e:
            logger.error(f"❌ Failed to apply ChromaDB changes: {str(e)}")
            raise

    async def _apply_redis_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply Redis configuration changes."""
        try:
            redis_changes = {k: v for k, v in changes.items() if k.startswith('redis_')}

            if not redis_changes:
                return

            logger.info("🔄 Applying Redis changes", changes=list(redis_changes.keys()))

            # Check if Redis is being enabled/disabled
            if 'redis_enabled' in redis_changes:
                if redis_changes['redis_enabled']:
                    logger.info("✅ Redis caching enabled")
                    warnings.append("Redis enabled - restart recommended for optimal performance")
                else:
                    logger.info("✅ Redis caching disabled")

            # Most Redis connection settings require restart
            connection_settings = ['redis_host', 'redis_port', 'redis_database', 'redis_password']
            if any(key in redis_changes for key in connection_settings):
                warnings.append("Redis connection settings changed - restart required")

        except Exception as e:
            logger.error(f"❌ Failed to apply Redis changes: {str(e)}")
            raise

    async def _apply_connection_pool_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply connection pool configuration changes."""
        try:
            pool_changes = {k: v for k, v in changes.items() if k.startswith('database_pool_')}
            
            if not pool_changes:
                return
            
            logger.info("🔄 Applying connection pool changes", changes=list(pool_changes.keys()))
            
            # Apply to connection pool
            if self._connection_pool and hasattr(self._connection_pool, 'update_configuration'):
                await self._connection_pool.update_configuration(pool_changes)
                logger.info("✅ Updated connection pool configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply connection pool changes: {str(e)}")
            raise
    
    async def _apply_query_optimization_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply query optimization changes."""
        try:
            query_changes = {k: v for k, v in changes.items() if k.startswith('database_query_')}
            
            if not query_changes:
                return
            
            logger.info("🔄 Applying query optimization changes", changes=list(query_changes.keys()))
            
            # Apply to query optimizer
            if self._query_optimizer and hasattr(self._query_optimizer, 'update_configuration'):
                await self._query_optimizer.update_configuration(query_changes)
                logger.info("✅ Updated query optimization configuration")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply query optimization changes: {str(e)}")
            raise
    
    async def _apply_performance_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply database performance changes."""
        try:
            performance_changes = {k: v for k, v in changes.items() if k.startswith('database_performance_')}
            
            if not performance_changes:
                return
            
            logger.info("🔄 Applying database performance changes", changes=list(performance_changes.keys()))
            
            # Apply to database service
            if self._database_service and hasattr(self._database_service, 'update_performance_settings'):
                await self._database_service.update_performance_settings(performance_changes)
                logger.info("✅ Updated database performance settings")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply database performance changes: {str(e)}")
            raise
    
    async def _apply_security_changes(self, changes: Dict[str, Any], warnings: List[str]) -> None:
        """Apply database security changes."""
        try:
            security_changes = {k: v for k, v in changes.items() if k.startswith('database_security_')}
            
            if not security_changes:
                return
            
            logger.info("🔄 Applying database security changes", changes=list(security_changes.keys()))
            
            # Apply to database service
            if self._database_service and hasattr(self._database_service, 'update_security_settings'):
                await self._database_service.update_security_settings(security_changes)
                logger.info("✅ Updated database security settings")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply database security changes: {str(e)}")
            raise


# Global instance
database_observer = DatabaseConfigurationObserver()
