"""
LLM Provider Service.

This service provides a high-level interface for managing LLM providers,
handling agent creation with LLM configurations, and providing unified
access to multiple LLM providers.
"""

from typing import Dict, List, Optional, Any, Union
from langchain_core.language_models import BaseLanguageModel

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory
from app.config.settings import get_settings
from app.llm.manager import get_llm_manager, initialize_llm_manager
from app.llm.models import (
    ProviderType,
    LLMConfig,
    ModelInfo,
    ProviderCredentials,
    ProviderStatus
)

logger = get_logger()


class LLMService:
    """High-level service for LLM provider management."""
    
    def __init__(self):
        self.settings = get_settings()
        self._manager = None
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the LLM service."""
        try:
            # Get provider credentials from settings
            credentials = self.settings.get_provider_credentials()
            
            # Initialize the LLM manager
            self._manager = await initialize_llm_manager(credentials)
            self._is_initialized = True

            logger.info(
                "LLM Service initialized",
                LogCategory.LLM_OPERATIONS,
                "app.services.llm_service",
                data={"enabled_providers": self.settings.get_enabled_providers()}
            )

        except Exception as e:
            logger.error(
                "Failed to initialize LLM Service",
                LogCategory.LLM_OPERATIONS,
                "app.services.llm_service",
                error=e
            )
            raise
    
    def _ensure_initialized(self) -> None:
        """Ensure the service is initialized."""
        if not self._is_initialized or not self._manager:
            raise RuntimeError("LLM Service not initialized. Call initialize() first.")
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        self._ensure_initialized()
        providers = self._manager.get_available_providers()
        return [p.value for p in providers]
    
    async def get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available models from all providers."""
        self._ensure_initialized()
        
        models_by_provider = await self._manager.get_all_models()
        
        # Convert to serializable format
        result = {}
        for provider_type, models in models_by_provider.items():
            result[provider_type.value] = [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider.value,
                    "description": model.description,
                    "capabilities": [cap.value for cap in model.capabilities],
                    "max_tokens": model.max_tokens,
                    "context_length": model.context_length,
                    "status": model.status,
                    "parameters": model.parameters
                }
                for model in models
            ]
        
        return result
    
    async def get_models_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Get models from a specific provider."""
        self._ensure_initialized()
        
        try:
            provider_type = ProviderType(provider)
            models = await self._manager.get_models_by_provider(provider_type)
            
            return [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider.value,
                    "description": model.description,
                    "capabilities": [cap.value for cap in model.capabilities],
                    "max_tokens": model.max_tokens,
                    "context_length": model.context_length,
                    "status": model.status,
                    "parameters": model.parameters
                }
                for model in models
            ]
        except ValueError:
            logger.error(
                "Invalid provider type",
                LogCategory.LLM_OPERATIONS,
                "app.services.llm_service",
                data={"provider": provider}
            )
            return []

    async def validate_model_config(self, provider: str, model_id: str) -> bool:
        """Validate if a model configuration is valid."""
        self._ensure_initialized()

        try:
            provider_type = ProviderType(provider)
            return await self._manager.validate_model(provider_type, model_id)
        except ValueError:
            logger.error(
                "Invalid provider type",
                LogCategory.LLM_OPERATIONS,
                "app.services.llm_service",
                data={"provider": provider}
            )
            return False
    
    async def create_llm_instance(self, config: Union[Dict[str, Any], "LLMConfig"]) -> BaseLanguageModel:
        """Create an LLM instance from configuration."""
        self._ensure_initialized()

        try:
            # Handle both dict and LLMConfig inputs
            if isinstance(config, dict):
                # Convert config dict to LLMConfig
                llm_config = LLMConfig(
                    provider=ProviderType(config["provider"]),
                    model_id=config["model_id"],
                    model_name=config.get("model_name"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 2048),
                    top_p=config.get("top_p"),
                    top_k=config.get("top_k"),
                    frequency_penalty=config.get("frequency_penalty"),
                    presence_penalty=config.get("presence_penalty"),
                    additional_params=config.get("additional_params", {})
                )
            else:
                # Already an LLMConfig object
                llm_config = config

            return await self._manager.create_llm_instance(llm_config)

        except Exception as e:
            logger.error(
                "Failed to create LLM instance",
                LogCategory.LLM_OPERATIONS,
                "app.services.llm_service",
                data={"config": str(config)},
                error=e
            )
            raise
    
    async def test_provider_connection(self, provider: str) -> Dict[str, Any]:
        """Test connection to a specific provider."""
        self._ensure_initialized()
        
        try:
            provider_type = ProviderType(provider)
            status = await self._manager.test_provider(provider_type)
            
            if status:
                return {
                    "provider": status.provider.value,
                    "is_available": status.is_available,
                    "is_authenticated": status.is_authenticated,
                    "error_message": status.error_message,
                    "available_models": status.available_models,
                    "response_time_ms": status.response_time_ms,
                    "last_checked": status.last_checked.isoformat()
                }
            else:
                return {
                    "provider": provider,
                    "is_available": False,
                    "error_message": "Provider not found"
                }
                
        except ValueError:
            return {
                "provider": provider,
                "is_available": False,
                "error_message": "Invalid provider type"
            }
    
    async def test_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """Test connection to all providers."""
        self._ensure_initialized()
        
        results = await self._manager.test_all_providers()
        
        return {
            provider_type.value: {
                "provider": status.provider.value,
                "is_available": status.is_available,
                "is_authenticated": status.is_authenticated,
                "error_message": status.error_message,
                "available_models": status.available_models,
                "response_time_ms": status.response_time_ms,
                "last_checked": status.last_checked.isoformat()
            }
            for provider_type, status in results.items()
        }
    
    async def register_provider_credentials(self, provider: str, credentials: Dict[str, Any]) -> bool:
        """Register or update provider credentials."""
        self._ensure_initialized()
        
        try:
            provider_type = ProviderType(provider)
            provider_credentials = ProviderCredentials(
                provider=provider_type,
                api_key=credentials.get("api_key"),
                base_url=credentials.get("base_url"),
                organization=credentials.get("organization"),
                project=credentials.get("project"),
                additional_headers=credentials.get("additional_headers")
            )
            
            return await self._manager.register_provider(provider_type, provider_credentials)

        except ValueError:
            logger.error(
                "Invalid provider type",
                LogCategory.LLM_OPERATIONS,
                "app.services.llm_service",
                data={"provider": provider}
            )
            return False
    
    async def get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            "provider": self.settings.DEFAULT_AGENT_PROVIDER,
            "model_id": self.settings.DEFAULT_AGENT_MODEL,
            "temperature": 0.7,
            "max_tokens": 2048
        }
    
    async def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the LLM service and providers."""
        if not self._is_initialized:
            return {
                "initialized": False,
                "error": "Service not initialized"
            }
        
        manager_info = self._manager.get_provider_info()
        
        return {
            "initialized": True,
            "service_info": manager_info,
            "settings": {
                "default_provider": self.settings.DEFAULT_AGENT_PROVIDER,
                "default_model": self.settings.DEFAULT_AGENT_MODEL,
                "enabled_providers": self.settings.get_enabled_providers(),
                "backup_provider": self.settings.BACKUP_AGENT_PROVIDER,
                "backup_model": self.settings.BACKUP_AGENT_MODEL
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup the LLM service."""
        if self._manager:
            await self._manager.cleanup()
        self._is_initialized = False
        logger.info(
            "LLM Service cleaned up",
            LogCategory.LLM_OPERATIONS,
            "app.services.llm_service"
        )


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def initialize_llm_service() -> LLMService:
    """Initialize the global LLM service."""
    service = get_llm_service()
    if not service._is_initialized:
        await service.initialize()
    return service
