"""
LLM Provider Manager.

This module provides a centralized manager for all LLM providers,
handling provider registration, selection, and lifecycle management.
"""

import asyncio
from typing import Dict, List, Optional, Type, Any
from datetime import datetime
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


# ============================================================================
# AGENT BUILDER PLATFORM ENHANCEMENTS
# ============================================================================

class EnhancedLLMProviderManager(LLMProviderManager):
    """
    Enhanced LLM Provider Manager for the Agent Builder Platform.

    Provides additional capabilities for agent-specific LLM management,
    model switching, performance monitoring, and cost optimization.
    """

    def __init__(self):
        super().__init__()
        self._model_performance_cache: Dict[str, Dict[str, Any]] = {}
        self._agent_model_assignments: Dict[str, LLMConfig] = {}
        self._model_usage_stats: Dict[str, Dict[str, int]] = {}

    async def get_optimal_model_for_task(self, task_type: str, requirements: Optional[Dict[str, Any]] = None) -> LLMConfig:
        """
        Get the optimal model configuration for a specific task type.

        Args:
            task_type: Type of task (reasoning, creative, analytical, etc.)
            requirements: Additional requirements (speed, accuracy, cost, etc.)

        Returns:
            Optimal LLM configuration
        """
        requirements = requirements or {}

        # Task-specific model recommendations
        task_model_map = {
            "reasoning": {
                "provider": ProviderType.OLLAMA,
                "model": "llama3.2:latest",
                "temperature": 0.1,
                "max_tokens": 4096
            },
            "creative": {
                "provider": ProviderType.OLLAMA,
                "model": "llama3.2:latest",
                "temperature": 0.8,
                "max_tokens": 4096
            },
            "analytical": {
                "provider": ProviderType.OLLAMA,
                "model": "llama3.2:latest",
                "temperature": 0.2,
                "max_tokens": 3072
            },
            "conversational": {
                "provider": ProviderType.OLLAMA,
                "model": "llama3.2:latest",
                "temperature": 0.7,
                "max_tokens": 2048
            },
            "code": {
                "provider": ProviderType.OLLAMA,
                "model": "llama3.2:latest",
                "temperature": 0.1,
                "max_tokens": 4096
            }
        }

        # Get base configuration
        base_config = task_model_map.get(task_type, task_model_map["conversational"])

        # Apply requirements-based adjustments
        if requirements.get("prefer_speed"):
            base_config["max_tokens"] = min(base_config["max_tokens"], 2048)

        if requirements.get("prefer_accuracy"):
            base_config["temperature"] = max(0.1, base_config["temperature"] - 0.2)

        if requirements.get("prefer_creativity"):
            base_config["temperature"] = min(1.0, base_config["temperature"] + 0.2)

        # Check if OpenAI is available and preferred
        if requirements.get("prefer_cloud") and ProviderType.OPENAI in self._providers:
            base_config["provider"] = ProviderType.OPENAI
            base_config["model"] = "gpt-4o-mini"

        return LLMConfig(
            provider=base_config["provider"],
            model_id=base_config["model"],
            temperature=base_config["temperature"],
            max_tokens=base_config["max_tokens"]
        )

    async def get_model_for_agent(self, config: 'AgentBuilderConfig') -> LLMConfig:
        """
        Get model configuration for an agent, supporting both manual and automatic selection.

        Args:
            config: Agent builder configuration

        Returns:
            LLMConfig for the agent (manual selection takes precedence)
        """
        try:
            # Check if manual selection is enabled
            if hasattr(config.llm_config, 'manual_selection') and config.llm_config.manual_selection:
                logger.info(f"Using manual model selection: {config.llm_config.model_id}")
                return config.llm_config

            # Use automatic optimization based on agent type
            agent_type_task_mapping = {
                "react": "reasoning",
                "autonomous": "reasoning",
                "rag": "analytical",
                "knowledge_search": "analytical",
                "workflow": "reasoning",
                "multimodal": "analytical",
                "composite": "reasoning"
            }

            task_type = agent_type_task_mapping.get(config.agent_type.value.lower(), "reasoning")

            # Get optimal model and merge with user preferences
            optimal_config = await self.get_optimal_model_for_task(task_type)

            # Override with user-specified values if present
            if config.llm_config:
                if config.llm_config.provider:
                    optimal_config.provider = config.llm_config.provider
                if config.llm_config.model_id:
                    optimal_config.model_id = config.llm_config.model_id
                if config.llm_config.temperature is not None:
                    optimal_config.temperature = config.llm_config.temperature
                if config.llm_config.max_tokens:
                    optimal_config.max_tokens = config.llm_config.max_tokens

            logger.info(f"Selected optimal model for {config.agent_type.value}: {optimal_config.model_id}")
            return optimal_config

        except Exception as e:
            logger.error(f"Failed to get model for agent: {str(e)}")
            return config.llm_config or LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.7,
                max_tokens=2048
            )

    def get_available_models_by_provider(self) -> Dict[str, List[str]]:
        """
        Get all available models grouped by provider for manual selection.

        Returns:
            Dictionary mapping provider names to available model lists
        """
        try:
            available_models = {}

            # Ollama models
            if ProviderType.OLLAMA in self._providers:
                available_models["ollama"] = [
                    "llama3.2:latest", "llama3.2:3b", "llama3.2:1b",
                    "codellama:latest", "codellama:13b", "codellama:7b",
                    "mistral:latest", "mistral:7b",
                    "phi3:latest", "phi3:mini",
                    "qwen2:latest", "qwen2:7b"
                ]

            # OpenAI models
            if ProviderType.OPENAI in self._providers:
                available_models["openai"] = [
                    "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
                    "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
                ]

            # Anthropic models
            if ProviderType.ANTHROPIC in self._providers:
                available_models["anthropic"] = [
                    "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
                ]

            # Google models
            if ProviderType.GOOGLE in self._providers:
                available_models["google"] = [
                    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"
                ]

            return available_models

        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return {}

    def get_model_recommendations(self, agent_type: str) -> Dict[str, Any]:
        """
        Get model recommendations for a specific agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Dictionary with recommended models and reasoning
        """
        try:
            recommendations = {
                "react": {
                    "primary": {"provider": "ollama", "model": "llama3.2:latest", "reason": "Excellent reasoning capabilities"},
                    "alternatives": [
                        {"provider": "openai", "model": "gpt-4", "reason": "Superior reasoning but higher cost"},
                        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "reason": "Strong analytical skills"}
                    ]
                },
                "autonomous": {
                    "primary": {"provider": "ollama", "model": "llama3.2:latest", "reason": "Good for autonomous decision making"},
                    "alternatives": [
                        {"provider": "openai", "model": "gpt-4", "reason": "Advanced reasoning for complex decisions"},
                        {"provider": "anthropic", "model": "claude-3-opus-20240229", "reason": "Excellent for complex autonomous tasks"}
                    ]
                },
                "rag": {
                    "primary": {"provider": "ollama", "model": "llama3.2:latest", "reason": "Good context understanding"},
                    "alternatives": [
                        {"provider": "openai", "model": "gpt-4-turbo", "reason": "Large context window for RAG"},
                        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "reason": "Excellent document analysis"}
                    ]
                },
                "multimodal": {
                    "primary": {"provider": "openai", "model": "gpt-4o", "reason": "Native vision capabilities"},
                    "alternatives": [
                        {"provider": "google", "model": "gemini-1.5-pro", "reason": "Strong multimodal understanding"},
                        {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "reason": "Good vision analysis"}
                    ]
                }
            }

            return recommendations.get(agent_type.lower(), recommendations["react"])

        except Exception as e:
            logger.error(f"Failed to get model recommendations: {str(e)}")
            return {}

    async def assign_model_to_agent(self, agent_id: str, llm_config: LLMConfig) -> bool:
        """
        Assign a specific model configuration to an agent.

        Args:
            agent_id: Agent identifier
            llm_config: LLM configuration to assign

        Returns:
            Success status
        """
        try:
            # Validate the configuration
            provider = self.get_provider(llm_config.provider)
            if not provider:
                logger.error(f"Provider {llm_config.provider.value} not available")
                return False

            if not await provider.validate_model(llm_config.model_id):
                logger.error(f"Model {llm_config.model_id} not available from {llm_config.provider.value}")
                return False

            # Store assignment
            self._agent_model_assignments[agent_id] = llm_config

            # Initialize usage stats
            model_key = f"{llm_config.provider.value}:{llm_config.model_id}"
            if model_key not in self._model_usage_stats:
                self._model_usage_stats[model_key] = {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "avg_response_time": 0.0,
                    "error_count": 0
                }

            logger.info(f"Model assigned to agent {agent_id}: {model_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to assign model to agent {agent_id}: {str(e)}")
            return False

    def get_agent_model_config(self, agent_id: str) -> Optional[LLMConfig]:
        """Get the model configuration assigned to an agent."""
        return self._agent_model_assignments.get(agent_id)

    async def switch_agent_model(self, agent_id: str, new_config: LLMConfig) -> bool:
        """
        Switch an agent to a different model configuration.

        Args:
            agent_id: Agent identifier
            new_config: New LLM configuration

        Returns:
            Success status
        """
        try:
            # Validate new configuration
            if not await self.assign_model_to_agent(agent_id, new_config):
                return False

            logger.info(f"Agent {agent_id} switched to model: {new_config.provider.value}:{new_config.model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch model for agent {agent_id}: {str(e)}")
            return False

    def record_model_usage(self, agent_id: str, tokens_used: int, response_time: float, success: bool = True):
        """
        Record model usage statistics for monitoring and optimization.

        Args:
            agent_id: Agent identifier
            tokens_used: Number of tokens used
            response_time: Response time in seconds
            success: Whether the request was successful
        """
        config = self._agent_model_assignments.get(agent_id)
        if not config:
            return

        model_key = f"{config.provider.value}:{config.model_id}"
        stats = self._model_usage_stats.get(model_key)
        if not stats:
            return

        # Update statistics
        stats["total_requests"] += 1
        stats["total_tokens"] += tokens_used

        # Update average response time
        current_avg = stats["avg_response_time"]
        total_requests = stats["total_requests"]
        stats["avg_response_time"] = ((current_avg * (total_requests - 1)) + response_time) / total_requests

        if not success:
            stats["error_count"] += 1

    def get_model_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive model usage statistics."""
        return self._model_usage_stats.copy()

    def get_provider_health_status(self) -> Dict[ProviderType, Dict[str, Any]]:
        """Get health status for all providers."""
        health_status = {}

        for provider_type, provider in self._providers.items():
            try:
                # Basic health check
                status = {
                    "available": True,
                    "status": provider.status.value if hasattr(provider, 'status') else "unknown",
                    "models_available": len(getattr(provider, 'available_models', [])),
                    "last_check": datetime.utcnow().isoformat()
                }

                # Add usage statistics if available
                provider_stats = {k: v for k, v in self._model_usage_stats.items()
                                if k.startswith(provider_type.value)}
                if provider_stats:
                    total_requests = sum(stats["total_requests"] for stats in provider_stats.values())
                    total_errors = sum(stats["error_count"] for stats in provider_stats.values())
                    status["total_requests"] = total_requests
                    status["error_rate"] = (total_errors / total_requests) if total_requests > 0 else 0.0

                health_status[provider_type] = status

            except Exception as e:
                health_status[provider_type] = {
                    "available": False,
                    "error": str(e),
                    "last_check": datetime.utcnow().isoformat()
                }

        return health_status


# Global enhanced manager instance
_enhanced_llm_manager: Optional[EnhancedLLMProviderManager] = None


def get_enhanced_llm_manager() -> EnhancedLLMProviderManager:
    """Get the global enhanced LLM provider manager instance."""
    global _enhanced_llm_manager
    if _enhanced_llm_manager is None:
        _enhanced_llm_manager = EnhancedLLMProviderManager()
    return _enhanced_llm_manager
