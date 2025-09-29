"""
Database Storage Configuration Section Manager

Handles configuration for PostgreSQL database, ChromaDB vector storage,
Redis caching, and other storage-related settings.
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.core.global_config_manager import ConfigurationSection, UpdateResult
from .base_section_manager import BaseConfigurationSectionManager

logger = structlog.get_logger(__name__)


class DatabaseSectionManager(BaseConfigurationSectionManager):
    """
    Manages database and storage configuration settings.
    
    Handles:
    - PostgreSQL database settings
    - ChromaDB vector storage configuration
    - Redis caching settings
    - Connection pooling and performance tuning
    - Backup and maintenance settings
    """
    
    def __init__(self):
        super().__init__()
        self._default_config = {
            # PostgreSQL Database Settings
            "postgres_host": "localhost",
            "postgres_port": 5432,
            "postgres_database": "agentic_ai",
            "postgres_username": "postgres",
            "postgres_password": "",
            "postgres_ssl_mode": "prefer",
            "postgres_pool_size": 50,    # INCREASED from 10
            "postgres_max_overflow": 50,  # INCREASED from 20
            "postgres_pool_timeout": 30,
            "postgres_pool_recycle": 3600,
            "postgres_echo_sql": False,
            "postgres_docker_enabled": True,
            "postgres_docker_image": "postgres:16-alpine",
            "postgres_docker_port": 5432,
            "postgres_docker_host": "localhost",
            "postgres_docker_volumes": True,
            "postgres_backup_enabled": False,
            "postgres_backup_schedule": "0 2 * * *",  # Daily at 2 AM
            
            # Vector Database Settings (ChromaDB & PgVector)
            "vector_db_type": "auto",  # auto, chromadb, pgvector
            "vector_db_auto_detect": True,

            # ChromaDB Settings
            "chroma_persist_directory": "data/chroma",
            "chroma_collection_name": "agentic_documents",
            "chroma_distance_function": "cosine",
            "chroma_batch_size": 100,
            "chroma_max_batch_size": 5000,
            "chroma_enable_persistence": True,
            "chroma_anonymized_telemetry": False,
            "chroma_allow_reset": False,
            "chroma_docker_enabled": True,
            "chroma_docker_image": "chromadb/chroma:latest",
            "chroma_docker_port": 8000,
            "chroma_docker_host": "localhost",

            # PgVector Settings
            "pgvector_enabled": False,
            "pgvector_table_name": "embeddings",
            "pgvector_dimension": 1536,
            "pgvector_distance_function": "cosine",  # cosine, l2, inner_product
            "pgvector_index_type": "ivfflat",  # ivfflat, hnsw
            "pgvector_index_lists": 100,
            "pgvector_docker_enabled": True,
            "pgvector_docker_image": "pgvector/pgvector:pg16",
            "pgvector_docker_port": 5433,
            
            # Redis Cache Settings (if enabled)
            "redis_enabled": False,
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_database": 0,
            "redis_password": "",
            "redis_ssl": False,
            "redis_connection_pool_size": 10,
            "redis_socket_timeout": 5,
            "redis_socket_connect_timeout": 5,
            "redis_retry_on_timeout": True,
            "redis_docker_enabled": True,
            "redis_docker_image": "redis:7-alpine",
            "redis_docker_port": 6379,
            "redis_docker_host": "localhost",
            "redis_persistence_enabled": True,
            "redis_memory_policy": "allkeys-lru",
            
            # Performance and Optimization
            "enable_query_optimization": True,
            "enable_connection_pooling": True,
            "enable_prepared_statements": True,
            "query_timeout_seconds": 30,
            "slow_query_threshold_ms": 1000,
            "enable_query_logging": False,
            "enable_slow_query_logging": True,
            "enable_query_cache": True,
            "query_cache_size_mb": 256,
            "enable_index_optimization": True,
            "auto_analyze_enabled": True,
            "connection_pool_monitoring": True,
            
            # Backup and Maintenance
            "enable_auto_backup": False,
            "backup_retention_days": 30,
            "backup_schedule": "0 2 * * *",  # Daily at 2 AM
            "backup_compression": True,
            "backup_encryption": False,
            "enable_auto_vacuum": True,
            "vacuum_schedule": "0 3 * * 0",  # Weekly on Sunday at 3 AM
            "enable_auto_reindex": False,
            "reindex_schedule": "0 4 * * 0",  # Weekly on Sunday at 4 AM
            "enable_statistics_update": True,
            "statistics_update_schedule": "0 1 * * *",  # Daily at 1 AM
            "enable_log_rotation": True,
            "log_retention_days": 7,
            
            # Storage Limits and Monitoring
            "max_database_size_gb": 100,
            "max_vector_storage_gb": 50,
            "enable_storage_monitoring": True,
            "storage_warning_threshold_percent": 80,
            "storage_critical_threshold_percent": 95,
            
            # Migration and Schema Management
            "enable_auto_migrations": True,
            "migration_timeout_seconds": 300,
            "enable_schema_validation": True,
            "enable_foreign_key_checks": True,
        }
    
    @property
    def section_name(self) -> ConfigurationSection:
        """The configuration section this manager handles."""
        return ConfigurationSection.DATABASE_STORAGE
    
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate database storage configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Validate PostgreSQL settings
            if "postgres_port" in config:
                port = config["postgres_port"]
                if not isinstance(port, int) or port < 1 or port > 65535:
                    errors.append("PostgreSQL port must be between 1 and 65535")
            
            if "postgres_pool_size" in config:
                pool_size = config["postgres_pool_size"]
                if not isinstance(pool_size, int) or pool_size < 1 or pool_size > 100:
                    errors.append("PostgreSQL pool size must be between 1 and 100")
            
            if "postgres_max_overflow" in config:
                max_overflow = config["postgres_max_overflow"]
                if not isinstance(max_overflow, int) or max_overflow < 0 or max_overflow > 200:
                    errors.append("PostgreSQL max overflow must be between 0 and 200")
            
            # Validate ChromaDB settings
            if "chroma_persist_directory" in config:
                persist_dir = config["chroma_persist_directory"]
                if not isinstance(persist_dir, str) or len(persist_dir.strip()) == 0:
                    errors.append("ChromaDB persist directory cannot be empty")
            
            if "chroma_batch_size" in config:
                batch_size = config["chroma_batch_size"]
                if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 10000:
                    errors.append("ChromaDB batch size must be between 1 and 10000")
            
            if "chroma_distance_function" in config:
                distance_func = config["chroma_distance_function"]
                valid_functions = ["cosine", "euclidean", "manhattan", "dot"]
                if distance_func not in valid_functions:
                    errors.append(f"ChromaDB distance function must be one of: {', '.join(valid_functions)}")
            
            # Validate Redis settings (if enabled)
            if config.get("redis_enabled", False):
                if "redis_port" in config:
                    redis_port = config["redis_port"]
                    if not isinstance(redis_port, int) or redis_port < 1 or redis_port > 65535:
                        errors.append("Redis port must be between 1 and 65535")
                
                if "redis_database" in config:
                    redis_db = config["redis_database"]
                    if not isinstance(redis_db, int) or redis_db < 0 or redis_db > 15:
                        errors.append("Redis database must be between 0 and 15")
            
            # Validate performance settings
            if "query_timeout_seconds" in config:
                timeout = config["query_timeout_seconds"]
                if not isinstance(timeout, int) or timeout < 1 or timeout > 300:
                    errors.append("Query timeout must be between 1 and 300 seconds")
            
            # Validate storage limits
            if "max_database_size_gb" in config:
                max_size = config["max_database_size_gb"]
                if not isinstance(max_size, int) or max_size < 1 or max_size > 1000:
                    errors.append("Max database size must be between 1 and 1000 GB")
            
            if "storage_warning_threshold_percent" in config:
                threshold = config["storage_warning_threshold_percent"]
                if not isinstance(threshold, int) or threshold < 50 or threshold > 99:
                    errors.append("Storage warning threshold must be between 50 and 99 percent")
            
        except Exception as e:
            logger.error("Error validating database storage configuration", error=str(e))
            errors.append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    async def apply_configuration(self, config: Dict[str, Any], previous_config: Dict[str, Any]) -> UpdateResult:
        """
        Apply database storage configuration changes.
        
        Args:
            config: New configuration to apply
            previous_config: Previous configuration for rollback
            
        Returns:
            UpdateResult indicating success/failure and any issues
        """
        try:
            # Validate configuration first
            validation_errors = await self.validate_configuration(config)
            if validation_errors:
                return UpdateResult(
                    success=False,
                    section=self.section_name.value,
                    message=f"Configuration validation failed: {'; '.join(validation_errors)}"
                )
            
            # Apply configuration changes
            warnings = await self._apply_configuration_changes(config, previous_config)
            
            # Update current configuration
            self._current_config.update(config)
            
            message = "Database storage configuration updated successfully"
            if warnings:
                message += f" (warnings: {'; '.join(warnings)})"
            
            logger.info("Database storage configuration applied", 
                       section=self.section_name.value, 
                       changes=list(config.keys()))
            
            return UpdateResult(
                success=True,
                section=self.section_name.value,
                message=message
            )
            
        except Exception as e:
            logger.error("Failed to apply database storage configuration", 
                        section=self.section_name.value, 
                        error=str(e))
            return UpdateResult(
                success=False,
                section=self.section_name.value,
                message=f"Failed to apply configuration: {str(e)}"
            )
    
    async def _apply_configuration_changes(self, config: Dict[str, Any], previous_config: Dict[str, Any]) -> List[str]:
        """
        Apply the actual configuration changes to database and storage systems.
        
        Args:
            config: New configuration
            previous_config: Previous configuration
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        try:
            # Check if database connection settings changed
            db_settings_changed = any(
                key.startswith("postgres_") and config.get(key) != previous_config.get(key)
                for key in config.keys()
            )
            
            if db_settings_changed:
                warnings.append("Database connection settings changed - restart required for full effect")
                # Note: In a production system, you might want to recreate connection pools here
            
            # Check if ChromaDB settings changed
            chroma_settings_changed = any(
                key.startswith("chroma_") and config.get(key) != previous_config.get(key)
                for key in config.keys()
            )
            
            if chroma_settings_changed:
                warnings.append("ChromaDB settings changed - some changes require restart")
                # Apply ChromaDB configuration changes
                await self._apply_chroma_config_changes(config)
            
            # Check if Redis settings changed
            redis_settings_changed = any(
                key.startswith("redis_") and config.get(key) != previous_config.get(key)
                for key in config.keys()
            )
            
            if redis_settings_changed:
                if config.get("redis_enabled", False):
                    warnings.append("Redis settings changed - restart required for full effect")
                    # Note: In production, you might want to recreate Redis connections here
            
            # Apply performance settings
            await self._apply_performance_settings(config)
            
            # Apply monitoring settings
            await self._apply_monitoring_settings(config)
            
        except Exception as e:
            logger.error("Error applying database storage configuration changes", error=str(e))
            warnings.append(f"Some configuration changes may not have been applied: {str(e)}")
        
        return warnings
    
    async def _apply_chroma_config_changes(self, config: Dict[str, Any]):
        """Apply ChromaDB configuration changes."""
        try:
            # Update ChromaDB settings if the service is available
            # This is a placeholder - in production you'd update the actual ChromaDB client
            logger.info("ChromaDB configuration updated", 
                       persist_directory=config.get("chroma_persist_directory"),
                       batch_size=config.get("chroma_batch_size"))
        except Exception as e:
            logger.warning("Failed to apply ChromaDB configuration changes", error=str(e))
    
    async def _apply_performance_settings(self, config: Dict[str, Any]):
        """Apply performance-related settings."""
        try:
            # Update query optimization settings
            if "enable_query_optimization" in config:
                logger.info("Query optimization setting updated", 
                           enabled=config["enable_query_optimization"])
            
            # Update connection pooling settings
            if "enable_connection_pooling" in config:
                logger.info("Connection pooling setting updated", 
                           enabled=config["enable_connection_pooling"])
                
        except Exception as e:
            logger.warning("Failed to apply performance settings", error=str(e))
    
    async def _apply_monitoring_settings(self, config: Dict[str, Any]):
        """Apply monitoring and alerting settings."""
        try:
            # Update storage monitoring settings
            if "enable_storage_monitoring" in config:
                logger.info("Storage monitoring setting updated", 
                           enabled=config["enable_storage_monitoring"])
            
            # Update backup settings
            if "enable_auto_backup" in config:
                logger.info("Auto backup setting updated", 
                           enabled=config["enable_auto_backup"])
                
        except Exception as e:
            logger.warning("Failed to apply monitoring settings", error=str(e))
    
    async def get_current_configuration(self) -> Dict[str, Any]:
        """Get the current configuration for this section."""
        return {**self._default_config, **self._current_config}
    
    async def reset_to_defaults(self) -> UpdateResult:
        """Reset configuration to default values."""
        try:
            self._current_config = {}
            logger.info("Database storage configuration reset to defaults")
            return UpdateResult(
                success=True,
                section=self.section_name.value,
                message="Configuration reset to defaults successfully"
            )
        except Exception as e:
            logger.error("Failed to reset database storage configuration", error=str(e))
            return UpdateResult(
                success=False,
                section=self.section_name.value,
                message=f"Failed to reset configuration: {str(e)}"
            )

    async def rollback_configuration(self, rollback_data: Dict[str, Any]) -> bool:
        """
        Rollback database storage configuration to previous state.

        Args:
            rollback_data: Previous configuration state to restore

        Returns:
            bool: True if rollback was successful, False otherwise
        """
        try:
            logger.info("Rolling back database storage configuration",
                       rollback_keys=list(rollback_data.keys()))

            # Restore previous configuration
            self._current_config.update(rollback_data)

            # Apply the rolled-back configuration
            warnings = await self._apply_configuration_changes(rollback_data, {})

            if warnings:
                logger.warning("Database storage rollback completed with warnings",
                              warnings=warnings)
            else:
                logger.info("Database storage configuration rollback successful")

            return True

        except Exception as e:
            logger.error("Failed to rollback database storage configuration", error=str(e))
            return False

    async def _load_initial_configuration(self) -> None:
        """Load the initial configuration for database storage section."""
        try:
            # Load any persisted configuration or use defaults
            logger.info("Loading initial database storage configuration")
            # The current config is already initialized with defaults in __init__
            # Any persisted config will be loaded by the global config manager
        except Exception as e:
            logger.error("Failed to load initial database storage configuration", error=str(e))

    async def _setup_validation_rules(self) -> None:
        """Setup validation rules for database storage section."""
        try:
            # Validation rules are already implemented in validate_configuration method
            logger.info("Database storage validation rules setup complete")
        except Exception as e:
            logger.error("Failed to setup database storage validation rules", error=str(e))

    async def _perform_rollback(self, rollback_data: Dict[str, Any]) -> bool:
        """
        Perform the actual rollback operation for database storage.

        Args:
            rollback_data: Previous configuration state to restore

        Returns:
            True if rollback was successful
        """
        try:
            logger.info("Performing database storage configuration rollback",
                       rollback_keys=list(rollback_data.keys()))

            # Restore previous configuration
            self._current_config.update(rollback_data)

            # Apply the rolled-back configuration
            warnings = await self._apply_configuration_changes(rollback_data, {})

            if warnings:
                logger.warning("Database storage rollback completed with warnings",
                              warnings=warnings)

            return True

        except Exception as e:
            logger.error("Failed to perform database storage rollback", error=str(e))
            return False

    async def detect_available_vector_databases(self) -> Dict[str, bool]:
        """
        Detect which vector databases are available.

        Returns:
            Dict with availability status for each vector database
        """
        availability = {
            "chromadb": False,
            "pgvector": False
        }

        try:
            # Check ChromaDB availability
            try:
                import chromadb
                availability["chromadb"] = True
                logger.info("ChromaDB is available")
            except ImportError:
                logger.info("ChromaDB is not available (not installed)")

            # Check PgVector availability (check if pgvector extension is available in PostgreSQL)
            try:
                # This would require a database connection to check
                # For now, we'll assume it's available if PostgreSQL is configured
                postgres_host = self._current_config.get("postgres_host", "localhost")
                if postgres_host:
                    availability["pgvector"] = True
                    logger.info("PgVector is potentially available (PostgreSQL configured)")
            except Exception:
                logger.info("PgVector availability check failed")

        except Exception as e:
            logger.error("Error detecting vector database availability", error=str(e))

        return availability

    async def get_recommended_vector_db(self) -> str:
        """
        Get recommended vector database based on availability and configuration.

        Returns:
            Recommended vector database type
        """
        try:
            availability = await self.detect_available_vector_databases()

            # If auto-detect is enabled, choose based on availability
            if self._current_config.get("vector_db_auto_detect", True):
                if availability["chromadb"]:
                    return "chromadb"
                elif availability["pgvector"]:
                    return "pgvector"
                else:
                    return "chromadb"  # Default fallback
            else:
                # Use configured type
                return self._current_config.get("vector_db_type", "chromadb")

        except Exception as e:
            logger.error("Error getting recommended vector database", error=str(e))
            return "chromadb"  # Safe fallback
