"""
🚀 Revolutionary Memory Configuration Observer

Real-time observer for memory system configuration changes.
Applies changes immediately to memory systems without server restarts.

REAL-TIME MEMORY UPDATES:
✅ Memory Limits and Thresholds
✅ TTL and Retention Policies
✅ Cleanup Configuration
✅ Performance Settings
✅ Security and Access Control
✅ Backup and Recovery Settings
✅ Analytics and Monitoring
"""

from typing import Any, Dict, Optional, List
import structlog

from ..global_config_manager import ConfigurationObserver, ConfigurationSection

logger = structlog.get_logger(__name__)


class MemoryConfigurationObserver(ConfigurationObserver):
    """
    🚀 Revolutionary Memory Configuration Observer
    
    Observes memory system configuration changes and applies them
    in real-time to the unified memory system and RAG system.
    """
    
    def __init__(self):
        """Initialize the memory configuration observer."""
        self._unified_memory_system = None
        self._unified_rag_system = None
        self._memory_manager = None
        logger.info("🚀 Memory Configuration Observer initialized")

    @property
    def observer_name(self) -> str:
        """Name of this observer."""
        return "MemoryConfigurationObserver"

    @property
    def observed_sections(self) -> List[str]:
        """Configuration sections this observer watches."""
        return [ConfigurationSection.MEMORY_SYSTEM]
    
    def set_unified_memory_system(self, memory_system) -> None:
        """Set the unified memory system instance to update."""
        self._unified_memory_system = memory_system
        logger.info("✅ Unified memory system registered with memory observer")
    
    def set_unified_rag_system(self, rag_system) -> None:
        """Set the unified RAG system instance to update."""
        self._unified_rag_system = rag_system
        logger.info("✅ Unified RAG system registered with memory observer")
    
    def set_memory_manager(self, memory_manager) -> None:
        """Set the memory manager instance to update."""
        self._memory_manager = memory_manager
        logger.info("✅ Memory manager registered with memory observer")
    
    async def on_configuration_changed(self, section: str, changes: Dict[str, Any]) -> bool:
        """Handle memory system configuration changes in real-time."""
        try:
            if section != ConfigurationSection.MEMORY_SYSTEM:
                return True  # Not our section, ignore
            
            logger.info("🔄 Processing memory configuration changes", 
                       section=section, changes=list(changes.keys()))
            
            # Apply memory limit changes
            await self._apply_memory_limit_changes(changes)
            
            # Apply TTL and retention changes
            await self._apply_ttl_retention_changes(changes)
            
            # Apply cleanup policy changes
            await self._apply_cleanup_policy_changes(changes)
            
            # Apply performance setting changes
            await self._apply_performance_changes(changes)
            
            # Apply security setting changes
            await self._apply_security_changes(changes)
            
            # Apply backup and recovery changes
            await self._apply_backup_recovery_changes(changes)
            
            # Apply analytics and monitoring changes
            await self._apply_analytics_monitoring_changes(changes)
            
            logger.info("✅ Memory configuration changes applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply memory configuration changes: {str(e)}")
            return False
    
    async def _apply_memory_limit_changes(self, changes: Dict[str, Any]) -> None:
        """Apply memory limit and threshold changes."""
        try:
            limit_changes = {}
            
            # Memory limits
            if "max_short_term_memories" in changes:
                limit_changes["max_short_term"] = changes["max_short_term_memories"]
            if "max_long_term_memories" in changes:
                limit_changes["max_long_term"] = changes["max_long_term_memories"]
            if "max_memory_per_agent_mb" in changes:
                limit_changes["max_per_agent_mb"] = changes["max_memory_per_agent_mb"]
            if "max_total_memory_mb" in changes:
                limit_changes["max_total_mb"] = changes["max_total_memory_mb"]
            
            # Usage thresholds
            if "memory_usage_warning_threshold" in changes:
                limit_changes["warning_threshold"] = changes["memory_usage_warning_threshold"]
            if "memory_usage_critical_threshold" in changes:
                limit_changes["critical_threshold"] = changes["memory_usage_critical_threshold"]
            
            # Monitoring settings
            if "enable_memory_usage_alerts" in changes:
                limit_changes["enable_alerts"] = changes["enable_memory_usage_alerts"]
            if "memory_usage_check_interval" in changes:
                limit_changes["check_interval"] = changes["memory_usage_check_interval"]
            
            if not limit_changes:
                return
            
            logger.info("🔄 Applying memory limit changes", changes=list(limit_changes.keys()))
            
            # Apply to unified memory system
            if self._unified_memory_system:
                if hasattr(self._unified_memory_system, 'update_memory_limits'):
                    await self._unified_memory_system.update_memory_limits(limit_changes)
                    logger.info("✅ Updated memory limits in unified memory system")
                
                # Update configuration directly if available
                if hasattr(self._unified_memory_system, 'config'):
                    for key, value in limit_changes.items():
                        if key == "max_short_term":
                            self._unified_memory_system.config["max_short_term_memories"] = value
                        elif key == "max_long_term":
                            self._unified_memory_system.config["max_long_term_memories"] = value
            
            # Apply to memory manager
            if self._memory_manager and hasattr(self._memory_manager, 'update_memory_limits'):
                await self._memory_manager.update_memory_limits(limit_changes)
                logger.info("✅ Updated memory limits in memory manager")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply memory limit changes: {str(e)}")
            raise
    
    async def _apply_ttl_retention_changes(self, changes: Dict[str, Any]) -> None:
        """Apply TTL and retention policy changes."""
        try:
            ttl_changes = {}
            
            # TTL settings
            if "short_term_ttl_hours" in changes:
                ttl_changes["short_term_ttl"] = changes["short_term_ttl_hours"]
            if "long_term_retention_days" in changes:
                ttl_changes["long_term_retention"] = changes["long_term_retention_days"]
            
            # Cleanup intervals
            if "short_term_cleanup_interval_hours" in changes:
                ttl_changes["short_term_cleanup_interval"] = changes["short_term_cleanup_interval_hours"]
            if "long_term_cleanup_interval_hours" in changes:
                ttl_changes["long_term_cleanup_interval"] = changes["long_term_cleanup_interval_hours"]
            
            # Auto-cleanup settings
            if "short_term_auto_cleanup" in changes:
                ttl_changes["short_term_auto_cleanup"] = changes["short_term_auto_cleanup"]
            if "long_term_auto_cleanup" in changes:
                ttl_changes["long_term_auto_cleanup"] = changes["long_term_auto_cleanup"]
            
            if not ttl_changes:
                return
            
            logger.info("🔄 Applying TTL and retention changes", changes=list(ttl_changes.keys()))
            
            # Apply to unified memory system
            if self._unified_memory_system:
                if hasattr(self._unified_memory_system, 'update_ttl_settings'):
                    await self._unified_memory_system.update_ttl_settings(ttl_changes)
                    logger.info("✅ Updated TTL settings in unified memory system")
                
                # Update configuration directly
                if hasattr(self._unified_memory_system, 'config'):
                    if "short_term_ttl" in ttl_changes:
                        self._unified_memory_system.config["short_term_ttl_hours"] = ttl_changes["short_term_ttl"]
                    if "short_term_cleanup_interval" in ttl_changes:
                        self._unified_memory_system.config["cleanup_interval_hours"] = ttl_changes["short_term_cleanup_interval"]
            
            # Apply to RAG system memory settings
            if self._unified_rag_system and hasattr(self._unified_rag_system, 'update_memory_ttl'):
                rag_ttl_updates = {}
                if "short_term_ttl" in ttl_changes:
                    rag_ttl_updates["short_term_ttl_hours"] = ttl_changes["short_term_ttl"]
                if "long_term_retention" in ttl_changes:
                    rag_ttl_updates["long_term_max_items"] = changes.get("max_long_term_memories", 10000)
                
                if rag_ttl_updates:
                    await self._unified_rag_system.update_memory_ttl(rag_ttl_updates)
                    logger.info("✅ Updated memory TTL in RAG system")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply TTL and retention changes: {str(e)}")
            raise
    
    async def _apply_cleanup_policy_changes(self, changes: Dict[str, Any]) -> None:
        """Apply cleanup policy and strategy changes."""
        try:
            cleanup_changes = {}
            
            # Cleanup strategy and settings
            if "cleanup_strategy" in changes:
                cleanup_changes["strategy"] = changes["cleanup_strategy"]
            if "cleanup_batch_size" in changes:
                cleanup_changes["batch_size"] = changes["cleanup_batch_size"]
            if "cleanup_max_duration_seconds" in changes:
                cleanup_changes["max_duration"] = changes["cleanup_max_duration_seconds"]
            if "enable_auto_cleanup" in changes:
                cleanup_changes["enable_auto_cleanup"] = changes["enable_auto_cleanup"]
            if "cleanup_dry_run_mode" in changes:
                cleanup_changes["dry_run_mode"] = changes["cleanup_dry_run_mode"]
            if "enable_cleanup_logging" in changes:
                cleanup_changes["enable_logging"] = changes["enable_cleanup_logging"]
            
            if not cleanup_changes:
                return
            
            logger.info("🔄 Applying cleanup policy changes", changes=list(cleanup_changes.keys()))
            
            # Apply to unified memory system
            if self._unified_memory_system and hasattr(self._unified_memory_system, 'update_cleanup_policies'):
                await self._unified_memory_system.update_cleanup_policies(cleanup_changes)
                logger.info("✅ Updated cleanup policies in unified memory system")
            
            # Apply to memory manager
            if self._memory_manager and hasattr(self._memory_manager, 'update_cleanup_policies'):
                await self._memory_manager.update_cleanup_policies(cleanup_changes)
                logger.info("✅ Updated cleanup policies in memory manager")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply cleanup policy changes: {str(e)}")
            raise
    
    async def _apply_performance_changes(self, changes: Dict[str, Any]) -> None:
        """Apply memory performance and optimization changes."""
        try:
            performance_changes = {}
            
            # Caching settings
            if "enable_memory_caching" in changes:
                performance_changes["enable_caching"] = changes["enable_memory_caching"]
            if "memory_cache_ttl" in changes:
                performance_changes["cache_ttl"] = changes["memory_cache_ttl"]
            if "memory_cache_size_mb" in changes:
                performance_changes["cache_size_mb"] = changes["memory_cache_size_mb"]
            
            # Indexing settings
            if "enable_memory_indexing" in changes:
                performance_changes["enable_indexing"] = changes["enable_memory_indexing"]
            if "memory_index_rebuild_hours" in changes:
                performance_changes["index_rebuild_hours"] = changes["memory_index_rebuild_hours"]
            
            # Compression settings
            if "enable_memory_compression" in changes:
                performance_changes["enable_compression"] = changes["enable_memory_compression"]
            if "compression_algorithm" in changes:
                performance_changes["compression_algorithm"] = changes["compression_algorithm"]
            if "compression_level" in changes:
                performance_changes["compression_level"] = changes["compression_level"]
            
            # Search and retrieval settings
            if "enable_semantic_search" in changes:
                performance_changes["enable_semantic_search"] = changes["enable_semantic_search"]
            if "semantic_search_top_k" in changes:
                performance_changes["search_top_k"] = changes["semantic_search_top_k"]
            if "semantic_search_threshold" in changes:
                performance_changes["search_threshold"] = changes["semantic_search_threshold"]
            
            # Deduplication settings
            if "enable_memory_deduplication" in changes:
                performance_changes["enable_deduplication"] = changes["enable_memory_deduplication"]
            if "deduplication_similarity_threshold" in changes:
                performance_changes["deduplication_threshold"] = changes["deduplication_similarity_threshold"]
            
            if not performance_changes:
                return
            
            logger.info("🔄 Applying memory performance changes", changes=list(performance_changes.keys()))
            
            # Apply to unified memory system
            if self._unified_memory_system and hasattr(self._unified_memory_system, 'update_performance_settings'):
                await self._unified_memory_system.update_performance_settings(performance_changes)
                logger.info("✅ Updated performance settings in unified memory system")
            
            # Apply to RAG system
            if self._unified_rag_system and hasattr(self._unified_rag_system, 'update_memory_performance'):
                await self._unified_rag_system.update_memory_performance(performance_changes)
                logger.info("✅ Updated memory performance in RAG system")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply performance changes: {str(e)}")
            raise
    
    async def _apply_security_changes(self, changes: Dict[str, Any]) -> None:
        """Apply memory security and access control changes."""
        try:
            security_changes = {}
            
            # Access control
            if "enable_memory_access_control" in changes:
                security_changes["enable_access_control"] = changes["enable_memory_access_control"]
            if "memory_access_logging" in changes:
                security_changes["access_logging"] = changes["memory_access_logging"]
            
            # Audit trail
            if "enable_memory_audit_trail" in changes:
                security_changes["enable_audit_trail"] = changes["enable_memory_audit_trail"]
            if "audit_retention_days" in changes:
                security_changes["audit_retention_days"] = changes["audit_retention_days"]
            
            # Data sanitization
            if "enable_memory_sanitization" in changes:
                security_changes["enable_sanitization"] = changes["enable_memory_sanitization"]
            if "sanitization_rules" in changes:
                security_changes["sanitization_rules"] = changes["sanitization_rules"]
            
            # Encryption
            if "enable_memory_encryption" in changes:
                security_changes["enable_encryption"] = changes["enable_memory_encryption"]
            if "encryption_algorithm" in changes:
                security_changes["encryption_algorithm"] = changes["encryption_algorithm"]
            
            if not security_changes:
                return
            
            logger.info("🔄 Applying memory security changes", changes=list(security_changes.keys()))
            
            # Apply to unified memory system
            if self._unified_memory_system and hasattr(self._unified_memory_system, 'update_security_settings'):
                await self._unified_memory_system.update_security_settings(security_changes)
                logger.info("✅ Updated security settings in unified memory system")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply security changes: {str(e)}")
            raise
    
    async def _apply_backup_recovery_changes(self, changes: Dict[str, Any]) -> None:
        """Apply backup and recovery setting changes."""
        try:
            backup_changes = {}
            
            # Backup settings
            if "enable_memory_backup" in changes:
                backup_changes["enable_backup"] = changes["enable_memory_backup"]
            if "backup_interval_hours" in changes:
                backup_changes["backup_interval"] = changes["backup_interval_hours"]
            if "backup_retention_days" in changes:
                backup_changes["backup_retention"] = changes["backup_retention_days"]
            if "backup_compression" in changes:
                backup_changes["backup_compression"] = changes["backup_compression"]
            if "enable_incremental_backup" in changes:
                backup_changes["incremental_backup"] = changes["enable_incremental_backup"]
            if "backup_verification" in changes:
                backup_changes["backup_verification"] = changes["backup_verification"]
            
            if not backup_changes:
                return
            
            logger.info("🔄 Applying backup and recovery changes", changes=list(backup_changes.keys()))
            
            # Apply to unified memory system
            if self._unified_memory_system and hasattr(self._unified_memory_system, 'update_backup_settings'):
                await self._unified_memory_system.update_backup_settings(backup_changes)
                logger.info("✅ Updated backup settings in unified memory system")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply backup and recovery changes: {str(e)}")
            raise
    
    async def _apply_analytics_monitoring_changes(self, changes: Dict[str, Any]) -> None:
        """Apply analytics and monitoring setting changes."""
        try:
            analytics_changes = {}
            
            # Analytics settings
            if "enable_memory_analytics" in changes:
                analytics_changes["enable_analytics"] = changes["enable_memory_analytics"]
            if "analytics_retention_days" in changes:
                analytics_changes["analytics_retention"] = changes["analytics_retention_days"]
            
            # Metrics settings
            if "enable_memory_metrics" in changes:
                analytics_changes["enable_metrics"] = changes["enable_memory_metrics"]
            if "metrics_collection_interval" in changes:
                analytics_changes["metrics_interval"] = changes["metrics_collection_interval"]
            
            # Profiling settings
            if "enable_memory_profiling" in changes:
                analytics_changes["enable_profiling"] = changes["enable_memory_profiling"]
            if "profiling_sample_rate" in changes:
                analytics_changes["profiling_sample_rate"] = changes["profiling_sample_rate"]
            
            if not analytics_changes:
                return
            
            logger.info("🔄 Applying analytics and monitoring changes", changes=list(analytics_changes.keys()))
            
            # Apply to unified memory system
            if self._unified_memory_system and hasattr(self._unified_memory_system, 'update_analytics_settings'):
                await self._unified_memory_system.update_analytics_settings(analytics_changes)
                logger.info("✅ Updated analytics settings in unified memory system")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply analytics and monitoring changes: {str(e)}")
            raise
