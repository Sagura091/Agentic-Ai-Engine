"""
🚀 Revolutionary LLM Configuration Observer

Real-time observer for LLM provider configuration changes.
Applies changes immediately to LLM services and providers without server restarts.

REAL-TIME LLM UPDATES:
✅ Provider Enablement/Disablement
✅ API Key Updates
✅ Model Switching
✅ Parameter Changes
✅ Performance Settings
✅ Provider Failover
✅ Load Balancing Updates
"""

from typing import Any, Dict, Optional, List
import structlog

from ..global_config_manager import ConfigurationObserver, ConfigurationSection
from ..configuration_broadcaster import configuration_broadcaster, BroadcastLevel, NotificationType

logger = structlog.get_logger(__name__)


class LLMConfigurationObserver(ConfigurationObserver):
    """
    🚀 Revolutionary LLM Configuration Observer
    
    Observes LLM provider configuration changes and applies them
    in real-time to the LLM service and provider manager.
    """
    
    def __init__(self):
        """Initialize the LLM configuration observer."""
        self._llm_service = None
        self._llm_manager = None
        self._llm_provider_manager = None
        logger.info("🚀 LLM Configuration Observer initialized")

    @property
    def observer_name(self) -> str:
        """Name of this observer."""
        return "LLMConfigurationObserver"

    @property
    def observed_sections(self) -> List[str]:
        """Configuration sections this observer watches."""
        return [ConfigurationSection.LLM_PROVIDERS]
    
    def set_llm_service(self, llm_service) -> None:
        """Set the LLM service instance to update."""
        self._llm_service = llm_service
        logger.info("✅ LLM service registered with LLM observer")
    
    def set_llm_manager(self, llm_manager) -> None:
        """Set the LLM manager instance to update."""
        self._llm_manager = llm_manager
        logger.info("✅ LLM manager registered with LLM observer")
    
    def set_llm_provider_manager(self, provider_manager) -> None:
        """Set the LLM provider manager instance to update."""
        self._llm_provider_manager = provider_manager
        logger.info("✅ LLM provider manager registered with LLM observer")
    
    async def on_configuration_changed(self, section: str, changes: Dict[str, Any], previous_config: Dict[str, Any]) -> bool:
        """Handle LLM provider configuration changes in real-time with broadcasting."""
        try:
            if section != ConfigurationSection.LLM_PROVIDERS:
                return True  # Not our section, ignore

            logger.info("🔄 Processing LLM configuration changes",
                       section=section, changes=list(changes.keys()))

            # Apply provider enablement changes
            await self._apply_provider_enablement_changes(changes)

            # Apply credential changes
            await self._apply_credential_changes(changes)

            # Apply model configuration changes
            await self._apply_model_changes(changes)

            # Apply performance setting changes
            await self._apply_performance_changes(changes)

            # Apply advanced feature changes
            await self._apply_advanced_feature_changes(changes)

            # 🚀 REVOLUTIONARY: Broadcast changes to users
            await self._broadcast_llm_changes(section, changes)

            logger.info("✅ LLM configuration changes applied successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to apply LLM configuration changes: {str(e)}")
            return False

    async def _broadcast_llm_changes(self, section: str, changes: Dict[str, Any]) -> None:
        """🚀 Broadcast LLM configuration changes to users."""
        try:
            # Determine broadcast level and notification type
            broadcast_level = BroadcastLevel.PUBLIC  # Default to public for LLM changes
            notification_type = NotificationType.LLM_UPDATES

            # Check for model availability changes (high priority)
            if any(key.startswith("available_models") for key in changes.keys()):
                notification_type = NotificationType.MODEL_UPDATES
                broadcast_level = BroadcastLevel.PUBLIC

            # Check for credential changes (admin only)
            elif any("api_key" in key.lower() or "secret" in key.lower() for key in changes.keys()):
                broadcast_level = BroadcastLevel.ADMIN_ONLY
                notification_type = NotificationType.SECURITY_UPDATES

            # Broadcast the changes
            for setting_key, value in changes.items():
                await configuration_broadcaster.broadcast_configuration_change(
                    section=section,
                    setting_key=setting_key,
                    changes={setting_key: value},
                    broadcast_level=broadcast_level,
                    admin_user_id="system",  # Will be updated with actual admin ID
                    notification_type=notification_type
                )

            logger.info(f"📢 Broadcasted LLM changes: {list(changes.keys())}")

        except Exception as e:
            logger.error(f"❌ Failed to broadcast LLM changes: {str(e)}")
    
    async def _apply_provider_enablement_changes(self, changes: Dict[str, Any]) -> None:
        """Apply provider enablement/disablement changes."""
        try:
            provider_changes = {}
            
            # Check for provider enablement changes
            if "enable_ollama" in changes:
                provider_changes["ollama"] = changes["enable_ollama"]
            if "enable_openai" in changes:
                provider_changes["openai"] = changes["enable_openai"]
            if "enable_anthropic" in changes:
                provider_changes["anthropic"] = changes["enable_anthropic"]
            if "enable_google" in changes:
                provider_changes["google"] = changes["enable_google"]
            
            if not provider_changes:
                return
            
            logger.info("🔄 Applying provider enablement changes", changes=provider_changes)
            
            # Apply to LLM service
            if self._llm_service:
                for provider, enabled in provider_changes.items():
                    if hasattr(self._llm_service, 'set_provider_enabled'):
                        await self._llm_service.set_provider_enabled(provider, enabled)
                        logger.info(f"✅ {'Enabled' if enabled else 'Disabled'} {provider} provider in LLM service")
            
            # Apply to LLM manager
            if self._llm_manager:
                for provider, enabled in provider_changes.items():
                    if hasattr(self._llm_manager, 'set_provider_enabled'):
                        await self._llm_manager.set_provider_enabled(provider, enabled)
                        logger.info(f"✅ {'Enabled' if enabled else 'Disabled'} {provider} provider in LLM manager")
            
            # Apply to provider manager
            if self._llm_provider_manager:
                for provider, enabled in provider_changes.items():
                    if hasattr(self._llm_provider_manager, 'enable_provider'):
                        if enabled:
                            await self._llm_provider_manager.enable_provider(provider)
                        else:
                            await self._llm_provider_manager.disable_provider(provider)
                        logger.info(f"✅ {'Enabled' if enabled else 'Disabled'} {provider} in provider manager")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply provider enablement changes: {str(e)}")
            raise
    
    async def _apply_credential_changes(self, changes: Dict[str, Any]) -> None:
        """Apply API key and credential changes."""
        try:
            credential_changes = {}
            
            # OpenAI credentials
            if any(key.startswith("openai_") for key in changes.keys()):
                openai_creds = {}
                if "openai_api_key" in changes:
                    openai_creds["api_key"] = changes["openai_api_key"]
                if "openai_base_url" in changes:
                    openai_creds["base_url"] = changes["openai_base_url"]
                if "openai_organization" in changes:
                    openai_creds["organization"] = changes["openai_organization"]
                if "openai_project" in changes:
                    openai_creds["project"] = changes["openai_project"]
                
                if openai_creds:
                    credential_changes["openai"] = openai_creds
            
            # Anthropic credentials
            if any(key.startswith("anthropic_") for key in changes.keys()):
                anthropic_creds = {}
                if "anthropic_api_key" in changes:
                    anthropic_creds["api_key"] = changes["anthropic_api_key"]
                if "anthropic_base_url" in changes:
                    anthropic_creds["base_url"] = changes["anthropic_base_url"]
                
                if anthropic_creds:
                    credential_changes["anthropic"] = anthropic_creds
            
            # Google credentials
            if any(key.startswith("google_") for key in changes.keys()):
                google_creds = {}
                if "google_api_key" in changes:
                    google_creds["api_key"] = changes["google_api_key"]
                if "google_base_url" in changes:
                    google_creds["base_url"] = changes["google_base_url"]
                
                if google_creds:
                    credential_changes["google"] = google_creds
            
            # Ollama configuration
            if any(key.startswith("ollama_") for key in changes.keys()):
                ollama_config = {}
                if "ollama_base_url" in changes:
                    ollama_config["base_url"] = changes["ollama_base_url"]
                if "ollama_timeout" in changes:
                    ollama_config["timeout"] = changes["ollama_timeout"]
                
                if ollama_config:
                    credential_changes["ollama"] = ollama_config
            
            if not credential_changes:
                return
            
            logger.info("🔄 Applying credential changes", providers=list(credential_changes.keys()))
            
            # Apply credential changes to LLM service
            if self._llm_service:
                for provider, creds in credential_changes.items():
                    if hasattr(self._llm_service, 'register_provider_credentials'):
                        await self._llm_service.register_provider_credentials(provider, creds)
                        logger.info(f"✅ Updated {provider} credentials in LLM service")
            
            # Apply to provider manager
            if self._llm_provider_manager:
                for provider, creds in credential_changes.items():
                    if hasattr(self._llm_provider_manager, 'update_provider_credentials'):
                        await self._llm_provider_manager.update_provider_credentials(provider, creds)
                        logger.info(f"✅ Updated {provider} credentials in provider manager")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply credential changes: {str(e)}")
            raise
    
    async def _apply_model_changes(self, changes: Dict[str, Any]) -> None:
        """Apply model configuration and availability changes."""
        try:
            model_changes = {}
            
            # Default model changes
            if "default_agent_model" in changes:
                model_changes["default_model"] = changes["default_agent_model"]
            if "default_agent_provider" in changes:
                model_changes["default_provider"] = changes["default_agent_provider"]
            if "backup_agent_model" in changes:
                model_changes["backup_model"] = changes["backup_agent_model"]
            if "backup_agent_provider" in changes:
                model_changes["backup_provider"] = changes["backup_agent_provider"]
            
            # Available model lists
            if "available_ollama_models" in changes:
                model_changes["ollama_models"] = changes["available_ollama_models"]
            if "available_openai_models" in changes:
                model_changes["openai_models"] = changes["available_openai_models"]
            if "available_anthropic_models" in changes:
                model_changes["anthropic_models"] = changes["available_anthropic_models"]
            if "available_google_models" in changes:
                model_changes["google_models"] = changes["available_google_models"]
            
            # Generation parameter defaults
            generation_params = {}
            if "default_temperature" in changes:
                generation_params["temperature"] = changes["default_temperature"]
            if "default_max_tokens" in changes:
                generation_params["max_tokens"] = changes["default_max_tokens"]
            if "default_top_p" in changes:
                generation_params["top_p"] = changes["default_top_p"]
            if "default_top_k" in changes:
                generation_params["top_k"] = changes["default_top_k"]
            
            if generation_params:
                model_changes["generation_params"] = generation_params
            
            if not model_changes:
                return
            
            logger.info("🔄 Applying model configuration changes", changes=list(model_changes.keys()))
            
            # Apply to LLM service
            if self._llm_service and hasattr(self._llm_service, 'update_model_configuration'):
                await self._llm_service.update_model_configuration(model_changes)
                logger.info("✅ Updated model configuration in LLM service")
            
            # Apply to LLM manager
            if self._llm_manager and hasattr(self._llm_manager, 'update_model_configuration'):
                await self._llm_manager.update_model_configuration(model_changes)
                logger.info("✅ Updated model configuration in LLM manager")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply model changes: {str(e)}")
            raise
    
    async def _apply_performance_changes(self, changes: Dict[str, Any]) -> None:
        """Apply performance and concurrency setting changes."""
        try:
            performance_changes = {}
            
            # Concurrency settings
            if "max_concurrent_agents" in changes:
                performance_changes["max_concurrent_agents"] = changes["max_concurrent_agents"]
            if "agent_timeout_seconds" in changes:
                performance_changes["agent_timeout"] = changes["agent_timeout_seconds"]
            
            # Provider-specific concurrency
            provider_concurrency = {}
            if "ollama_max_concurrent_requests" in changes:
                provider_concurrency["ollama"] = changes["ollama_max_concurrent_requests"]
            if "openai_max_concurrent_requests" in changes:
                provider_concurrency["openai"] = changes["openai_max_concurrent_requests"]
            if "anthropic_max_concurrent_requests" in changes:
                provider_concurrency["anthropic"] = changes["anthropic_max_concurrent_requests"]
            if "google_max_concurrent_requests" in changes:
                provider_concurrency["google"] = changes["google_max_concurrent_requests"]
            
            if provider_concurrency:
                performance_changes["provider_concurrency"] = provider_concurrency
            
            # Caching and optimization
            if "enable_model_caching" in changes:
                performance_changes["enable_caching"] = changes["enable_model_caching"]
            if "model_cache_ttl" in changes:
                performance_changes["cache_ttl"] = changes["model_cache_ttl"]
            if "enable_response_streaming" in changes:
                performance_changes["enable_streaming"] = changes["enable_response_streaming"]
            
            if not performance_changes:
                return
            
            logger.info("🔄 Applying performance changes", changes=list(performance_changes.keys()))
            
            # Apply to systems
            if self._llm_service and hasattr(self._llm_service, 'update_performance_settings'):
                await self._llm_service.update_performance_settings(performance_changes)
                logger.info("✅ Updated performance settings in LLM service")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply performance changes: {str(e)}")
            raise
    
    async def _apply_advanced_feature_changes(self, changes: Dict[str, Any]) -> None:
        """Apply advanced feature configuration changes."""
        try:
            advanced_changes = {}
            
            # Load balancing and failover
            if "enable_load_balancing" in changes:
                advanced_changes["enable_load_balancing"] = changes["enable_load_balancing"]
            if "load_balancing_strategy" in changes:
                advanced_changes["load_balancing_strategy"] = changes["load_balancing_strategy"]
            if "enable_failover" in changes:
                advanced_changes["enable_failover"] = changes["enable_failover"]
            if "failover_timeout" in changes:
                advanced_changes["failover_timeout"] = changes["failover_timeout"]
            
            # Health checks and monitoring
            if "enable_health_checks" in changes:
                advanced_changes["enable_health_checks"] = changes["enable_health_checks"]
            if "health_check_interval" in changes:
                advanced_changes["health_check_interval"] = changes["health_check_interval"]
            if "enable_metrics_collection" in changes:
                advanced_changes["enable_metrics"] = changes["enable_metrics_collection"]
            
            # Cost management
            if "enable_cost_tracking" in changes:
                advanced_changes["enable_cost_tracking"] = changes["enable_cost_tracking"]
            if "cost_limit_daily" in changes:
                advanced_changes["cost_limit_daily"] = changes["cost_limit_daily"]
            if "cost_limit_monthly" in changes:
                advanced_changes["cost_limit_monthly"] = changes["cost_limit_monthly"]
            
            if not advanced_changes:
                return
            
            logger.info("🔄 Applying advanced feature changes", changes=list(advanced_changes.keys()))
            
            # Apply to systems
            if self._llm_service and hasattr(self._llm_service, 'update_advanced_features'):
                await self._llm_service.update_advanced_features(advanced_changes)
                logger.info("✅ Updated advanced features in LLM service")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply advanced feature changes: {str(e)}")
            raise
