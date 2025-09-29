"""
LLM Provider Implementations.

This module contains the abstract base class and concrete implementations
for different LLM providers including Ollama, OpenAI, Anthropic, and Google.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
import structlog
from langchain_core.language_models import BaseLanguageModel

# Import our custom HTTP client
from app.http_client import SimpleHTTPClient, HTTPResponse, HTTPError

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from .models import (
    ProviderType,
    LLMConfig,
    ModelInfo,
    ProviderCredentials,
    ProviderStatus,
    LLMResponse,
    StreamingLLMResponse,
    ModelCapability,
    DEFAULT_MODELS,
    PROVIDER_DEFAULTS
)

logger = structlog.get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, credentials: Optional[ProviderCredentials] = None):
        self.credentials = credentials
        self.provider_type = self._get_provider_type()
        self._client = None
        self._is_initialized = False
    
    @abstractmethod
    def _get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def validate_model(self, model_id: str) -> bool:
        """Validate if a model is available."""
        pass
    
    @abstractmethod
    async def create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """Create a LangChain LLM instance."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> ProviderStatus:
        """Test provider connection and authentication."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        if self._client and hasattr(self._client, 'aclose'):
            await self._client.aclose()


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""
    
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA
    
    async def initialize(self) -> bool:
        """Initialize Ollama provider."""
        try:
            base_url = self.credentials.base_url if self.credentials else PROVIDER_DEFAULTS[ProviderType.OLLAMA]["base_url"]

            # Always use our custom HTTP client for Ollama to bypass proxy issues
            self._client = SimpleHTTPClient(base_url, timeout=60)
            self._use_custom_client = True
            self._is_initialized = True

            logger.info("Ollama provider initialized", base_url=base_url, custom_client=True)
            return True
        except Exception as e:
            logger.error("Failed to initialize Ollama provider", error=str(e))
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Ollama models."""
        if not self._is_initialized:
            await self.initialize()

        models = []
        try:
            # Use our custom HTTP client (now async)
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                for model in data.get("models", []):
                    models.append(ModelInfo(
                        id=model["name"],
                        name=model["name"],
                        provider=ProviderType.OLLAMA,
                        description=f"Ollama model: {model['name']}",
                        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION],
                        parameters={
                            "size": model.get("size", 0),
                            "modified_at": model.get("modified_at", ""),
                            "digest": model.get("digest", "")
                        }
                    ))
        except Exception as e:
            logger.error("Failed to get Ollama models", error=str(e))
            # Return default models as fallback
            for model_id in DEFAULT_MODELS[ProviderType.OLLAMA]:
                models.append(ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider=ProviderType.OLLAMA,
                    description=f"Ollama model: {model_id}",
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION],
                    status="unknown"
                ))

        return models
    
    async def validate_model(self, model_id: str) -> bool:
        """Validate Ollama model availability."""
        try:
            models = await self.get_available_models()
            return any(model.id == model_id for model in models)
        except Exception:
            return model_id in DEFAULT_MODELS[ProviderType.OLLAMA]
    
    async def create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """Create Ollama LLM instance using our custom HTTP client."""
        if ChatOllama is None:
            raise ImportError("langchain_ollama is required for Ollama provider")

        # Use production-ready provider system
        from .production_providers import ProductionOllamaProvider

        production_provider = ProductionOllamaProvider(self.credentials)
        await production_provider.initialize()

        return await production_provider.create_llm_instance(config)
    
    async def test_connection(self) -> ProviderStatus:
        """Test Ollama connection using production provider."""
        status = ProviderStatus(provider=ProviderType.OLLAMA)

        try:
            from .production_providers import ProductionOllamaProvider

            production_provider = ProductionOllamaProvider(self.credentials)

            import time
            start_time = time.time()

            # Test basic connectivity
            await production_provider._test_basic_connectivity()

            # Get available models with error handling
            try:
                models = await production_provider.get_available_models()
                status.available_models = [model.id for model in models]
            except Exception as model_error:
                logger.warning(f"Failed to get Ollama models during connection test: {str(model_error)}")
                # Still mark as available if basic connectivity works
                status.available_models = []

            response_time = (time.time() - start_time) * 1000

            status.is_available = True
            status.is_authenticated = True  # Ollama doesn't require auth
            status.response_time_ms = response_time

        except Exception as e:
            status.error_message = str(e)
            logger.error("Ollama connection test failed", error=str(e))

        return status


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.OPENAI
    
    async def initialize(self) -> bool:
        """Initialize OpenAI provider."""
        try:
            if not self.credentials or not self.credentials.api_key:
                logger.warning("OpenAI API key not provided")
                return False
            
            self._is_initialized = True
            logger.info("OpenAI provider initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize OpenAI provider", error=str(e))
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available OpenAI models."""
        models = []
        
        # Return default models (in production, could fetch from OpenAI API)
        for model_id in DEFAULT_MODELS[ProviderType.OPENAI]:
            capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION]
            if "gpt-4" in model_id:
                capabilities.extend([ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING])
            
            models.append(ModelInfo(
                id=model_id,
                name=model_id.upper().replace("-", " "),
                provider=ProviderType.OPENAI,
                description=f"OpenAI {model_id} model",
                capabilities=capabilities,
                max_tokens=8192 if "gpt-4" in model_id else 4096,
                context_length=128000 if "gpt-4" in model_id else 16385
            ))
        
        return models
    
    async def validate_model(self, model_id: str) -> bool:
        """Validate OpenAI model availability."""
        return model_id in DEFAULT_MODELS[ProviderType.OPENAI]
    
    async def create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """Create OpenAI LLM instance using production provider."""
        from .production_providers import ProductionOpenAIProvider

        production_provider = ProductionOpenAIProvider(self.credentials)
        await production_provider.initialize()

        return await production_provider.create_llm_instance(config)
    
    async def test_connection(self) -> ProviderStatus:
        """Test OpenAI connection using production provider."""
        status = ProviderStatus(provider=ProviderType.OPENAI)

        try:
            from .production_providers import ProductionOpenAIProvider

            production_provider = ProductionOpenAIProvider(self.credentials)

            import time
            start_time = time.time()

            # Test basic connectivity
            await production_provider._test_basic_connectivity()

            response_time = (time.time() - start_time) * 1000

            status.is_available = True
            status.is_authenticated = True
            status.available_models = DEFAULT_MODELS[ProviderType.OPENAI]
            status.response_time_ms = response_time

        except Exception as e:
            status.error_message = str(e)
            logger.error("OpenAI connection test failed", error=str(e))

        return status


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC
    
    async def initialize(self) -> bool:
        """Initialize Anthropic provider."""
        try:
            if not self.credentials or not self.credentials.api_key:
                logger.warning("Anthropic API key not provided")
                return False
            
            self._is_initialized = True
            logger.info("Anthropic provider initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize Anthropic provider", error=str(e))
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Anthropic models."""
        models = []
        
        for model_id in DEFAULT_MODELS[ProviderType.ANTHROPIC]:
            models.append(ModelInfo(
                id=model_id,
                name=f"Claude {model_id.split('-')[1].title()}",
                provider=ProviderType.ANTHROPIC,
                description=f"Anthropic {model_id} model",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CONVERSATION,
                    ModelCapability.REASONING,
                    ModelCapability.ANALYSIS
                ],
                max_tokens=4096,
                context_length=200000
            ))
        
        return models
    
    async def validate_model(self, model_id: str) -> bool:
        """Validate Anthropic model availability."""
        return model_id in DEFAULT_MODELS[ProviderType.ANTHROPIC]
    
    async def create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """Create Anthropic LLM instance using production provider."""
        from .production_providers import ProductionAnthropicProvider

        production_provider = ProductionAnthropicProvider(self.credentials)
        await production_provider.initialize()

        return await production_provider.create_llm_instance(config)
    
    async def test_connection(self) -> ProviderStatus:
        """Test Anthropic connection using production provider."""
        status = ProviderStatus(provider=ProviderType.ANTHROPIC)

        try:
            from .production_providers import ProductionAnthropicProvider

            production_provider = ProductionAnthropicProvider(self.credentials)

            import time
            start_time = time.time()

            # Test basic connectivity
            await production_provider._test_basic_connectivity()

            response_time = (time.time() - start_time) * 1000

            status.is_available = True
            status.is_authenticated = True
            status.available_models = DEFAULT_MODELS[ProviderType.ANTHROPIC]
            status.response_time_ms = response_time

        except Exception as e:
            status.error_message = str(e)
            logger.error("Anthropic connection test failed", error=str(e))

        return status


class GoogleProvider(LLMProvider):
    """Google provider implementation."""
    
    def _get_provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE
    
    async def initialize(self) -> bool:
        """Initialize Google provider."""
        try:
            if not self.credentials or not self.credentials.api_key:
                logger.warning("Google API key not provided")
                return False
            
            self._is_initialized = True
            logger.info("Google provider initialized")
            return True
        except Exception as e:
            logger.error("Failed to initialize Google provider", error=str(e))
            return False
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Google models."""
        models = []
        
        for model_id in DEFAULT_MODELS[ProviderType.GOOGLE]:
            capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION]
            if "vision" in model_id:
                capabilities.append(ModelCapability.MULTIMODAL)
            
            models.append(ModelInfo(
                id=model_id,
                name=f"Gemini {model_id.split('-')[1].title()}",
                provider=ProviderType.GOOGLE,
                description=f"Google {model_id} model",
                capabilities=capabilities,
                max_tokens=8192,
                context_length=32768
            ))
        
        return models
    
    async def validate_model(self, model_id: str) -> bool:
        """Validate Google model availability."""
        return model_id in DEFAULT_MODELS[ProviderType.GOOGLE]
    
    async def create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """Create Google LLM instance."""
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain_google_genai is required for Google provider")

        if not self.credentials or not self.credentials.api_key:
            raise ValueError("Google API key is required")

        return ChatGoogleGenerativeAI(
            model=config.model_id,
            google_api_key=self.credentials.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            **config.additional_params
        )
    
    async def test_connection(self) -> ProviderStatus:
        """Test Google connection."""
        status = ProviderStatus(provider=ProviderType.GOOGLE)
        
        try:
            if not self.credentials or not self.credentials.api_key:
                status.error_message = "API key not provided"
                return status
            
            status.is_available = True
            status.is_authenticated = True
            status.available_models = DEFAULT_MODELS[ProviderType.GOOGLE]
            
        except Exception as e:
            status.error_message = str(e)
            logger.error("Google connection test failed", error=str(e))
        
        return status
