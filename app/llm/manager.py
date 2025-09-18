"""
LLM Provider Manager.

This module provides a centralized manager for all LLM providers,
handling provider registration, selection, and lifecycle management.
"""

import asyncio
from typing import Dict, List, Optional, Type, Any
import structlog
from langchain_core.language_models import BaseLanguageModel

from .models import (
    ProviderType,
    LLMConfig,
    ModelInfo,
    ProviderCredentials,
    ProviderStatus
)
from .providers import (
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider
)

logger = structlog.get_logger(__name__)


class LLMProviderManager:
    """Centralized manager for LLM providers."""
    
    def __init__(self):
        self._providers: Dict[ProviderType, LLMProvider] = {}
        self._provider_classes: Dict[ProviderType, Type[LLMProvider]] = {
            ProviderType.OLLAMA: OllamaProvider,
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.ANTHROPIC: AnthropicProvider,
            ProviderType.GOOGLE: GoogleProvider
        }
        self._is_initialized = False
    
    async def initialize(self, provider_credentials: Optional[Dict[ProviderType, ProviderCredentials]] = None) -> None:
        """Initialize all providers with their credentials."""
        try:
            credentials_map = provider_credentials or {}
            
            for provider_type, provider_class in self._provider_classes.items():
                try:
                    credentials = credentials_map.get(provider_type)
                    provider = provider_class(credentials)
                    
                    # Initialize provider
                    if await provider.initialize():
                        self._providers[provider_type] = provider
                        logger.info("Provider initialized successfully", provider=provider_type.value)
                    else:
                        logger.warning("Provider initialization failed", provider=provider_type.value)
                        
                except Exception as e:
                    logger.error("Failed to initialize provider", provider=provider_type.value, error=str(e))
            
            self._is_initialized = True
            logger.info("LLM Provider Manager initialized", active_providers=list(self._providers.keys()))
            
        except Exception as e:
            logger.error("Failed to initialize LLM Provider Manager", error=str(e))
            raise
    
    async def register_provider(self, provider_type: ProviderType, credentials: Optional[ProviderCredentials] = None) -> bool:
        """Register a new provider or update existing one."""
        try:
            if provider_type not in self._provider_classes:
                logger.error("Unknown provider type", provider=provider_type.value)
                return False
            
            provider_class = self._provider_classes[provider_type]
            provider = provider_class(credentials)
            
            if await provider.initialize():
                # Cleanup old provider if exists
                if provider_type in self._providers:
                    await self._providers[provider_type].cleanup()
                
                self._providers[provider_type] = provider
                logger.info("Provider registered successfully", provider=provider_type.value)
                return True
            else:
                logger.warning("Provider registration failed", provider=provider_type.value)
                return False
                
        except Exception as e:
            logger.error("Failed to register provider", provider=provider_type.value, error=str(e))
            return False
    
    async def unregister_provider(self, provider_type: ProviderType) -> bool:
        """Unregister a provider."""
        try:
            if provider_type in self._providers:
                await self._providers[provider_type].cleanup()
                del self._providers[provider_type]
                logger.info("Provider unregistered", provider=provider_type.value)
                return True
            return False
        except Exception as e:
            logger.error("Failed to unregister provider", provider=provider_type.value, error=str(e))
            return False
    
    def get_provider(self, provider_type: ProviderType) -> Optional[LLMProvider]:
        """Get a registered provider."""
        return self._providers.get(provider_type)
    
    def get_available_providers(self) -> List[ProviderType]:
        """Get list of available providers."""
        return list(self._providers.keys())
    
    async def get_all_models(self) -> Dict[ProviderType, List[ModelInfo]]:
        """Get all available models from all providers."""
        all_models = {}
        
        for provider_type, provider in self._providers.items():
            try:
                models = await provider.get_available_models()
                all_models[provider_type] = models
            except Exception as e:
                logger.error("Failed to get models from provider", provider=provider_type.value, error=str(e))
                all_models[provider_type] = []
        
        return all_models
    
    async def get_models_by_provider(self, provider_type: ProviderType) -> List[ModelInfo]:
        """Get models from a specific provider."""
        provider = self.get_provider(provider_type)
        if not provider:
            logger.warning("Provider not available", provider=provider_type.value)
            return []
        
        try:
            return await provider.get_available_models()
        except Exception as e:
            logger.error("Failed to get models from provider", provider=provider_type.value, error=str(e))
            return []
    
    async def validate_model(self, provider_type: ProviderType, model_id: str) -> bool:
        """Validate if a model is available from a provider."""
        provider = self.get_provider(provider_type)
        if not provider:
            return False
        
        try:
            return await provider.validate_model(model_id)
        except Exception as e:
            logger.error("Failed to validate model", provider=provider_type.value, model=model_id, error=str(e))
            return False
    
    async def create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """Create an LLM instance using the specified configuration."""
        provider = self.get_provider(config.provider)
        if not provider:
            raise ValueError(f"Provider {config.provider.value} is not available")
        
        # Validate model
        if not await provider.validate_model(config.model_id):
            raise ValueError(f"Model {config.model_id} is not available from {config.provider.value}")
        
        try:
            return await provider.create_llm_instance(config)
        except Exception as e:
            logger.error("Failed to create LLM instance", provider=config.provider.value, model=config.model_id, error=str(e))
            raise
    
    async def test_all_providers(self) -> Dict[ProviderType, ProviderStatus]:
        """Test connection to all providers."""
        results = {}
        
        for provider_type, provider in self._providers.items():
            try:
                status = await provider.test_connection()
                results[provider_type] = status
            except Exception as e:
                results[provider_type] = ProviderStatus(
                    provider=provider_type,
                    error_message=str(e)
                )
        
        return results
    
    async def test_provider(self, provider_type: ProviderType) -> Optional[ProviderStatus]:
        """Test connection to a specific provider."""
        provider = self.get_provider(provider_type)
        if not provider:
            return None
        
        try:
            return await provider.test_connection()
        except Exception as e:
            return ProviderStatus(
                provider=provider_type,
                error_message=str(e)
            )
    
    async def cleanup(self) -> None:
        """Cleanup all providers."""
        for provider in self._providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.error("Failed to cleanup provider", error=str(e))
        
        self._providers.clear()
        self._is_initialized = False
        logger.info("LLM Provider Manager cleaned up")
    
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._is_initialized
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all providers."""
        return {
            "initialized": self._is_initialized,
            "available_providers": [p.value for p in self._providers.keys()],
            "total_providers": len(self._providers),
            "supported_providers": [p.value for p in self._provider_classes.keys()]
        }


# Global instance
_llm_manager: Optional[LLMProviderManager] = None


def get_llm_manager() -> LLMProviderManager:
    """Get the global LLM provider manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMProviderManager()
    return _llm_manager


async def initialize_llm_manager(provider_credentials: Optional[Dict[ProviderType, ProviderCredentials]] = None) -> LLMProviderManager:
    """Initialize the global LLM provider manager."""
    manager = get_llm_manager()
    if not manager.is_initialized():
        await manager.initialize(provider_credentials)
    return manager
