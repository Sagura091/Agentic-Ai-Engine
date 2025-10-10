"""
LLM Provider Implementations.

This module contains the abstract base class and concrete implementations
for different LLM providers including Ollama, OpenAI, Anthropic, and Google.

PHASE 1 & 2 IMPROVEMENTS:
- Removed unused HTTPClient dependency
- Merged Provider and ProductionProvider layers
- Added comprehensive logging
- Implemented streaming support
- Added dependency injection support
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from datetime import datetime
from langchain_core.language_models import BaseLanguageModel

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

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogCategory, LogLevel

# Get backend logger instance
logger = get_logger()


# ============================================================================
# PRODUCTION-READY INFRASTRUCTURE COMPONENTS
# ============================================================================

class ConnectionPool:
    """Production-ready connection pool for LLM providers."""

    def __init__(self, max_connections: int = 10, max_idle_time: int = 300):
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.connection_stats = {
            "total_created": 0,
            "total_reused": 0,
            "total_expired": 0,
            "active_connections": 0
        }
        logger.debug(
            "ConnectionPool initialized",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.ConnectionPool",
            data={"max_connections": max_connections, "max_idle_time": max_idle_time}
        )

    async def get_connection(self, provider_key: str, factory_func) -> Any:
        """Get or create a connection from the pool."""
        now = datetime.now()

        # Check if we have a valid connection
        if provider_key in self.connections:
            conn_info = self.connections[provider_key]

            # Check if connection is still valid
            if (now - conn_info["created_at"]).seconds < self.max_idle_time:
                self.connection_stats["total_reused"] += 1
                logger.debug(
                    "Reusing connection from pool",
                    LogCategory.LLM_OPERATIONS,
                    "app.llm.providers.ConnectionPool",
                    data={"provider_key": provider_key, "age_seconds": (now - conn_info["created_at"]).seconds}
                )
                return conn_info["connection"]
            else:
                # Connection expired
                del self.connections[provider_key]
                self.connection_stats["total_expired"] += 1
                logger.debug(
                    "Connection expired, removing from pool",
                    LogCategory.LLM_OPERATIONS,
                    "app.llm.providers.ConnectionPool",
                    data={"provider_key": provider_key}
                )

        # Create new connection
        if len(self.connections) >= self.max_connections:
            # Remove oldest connection
            oldest_key = min(self.connections.keys(),
                           key=lambda k: self.connections[k]["created_at"])
            del self.connections[oldest_key]
            logger.debug(
                "Pool full, removed oldest connection",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.ConnectionPool",
                data={"removed_key": oldest_key}
            )

        # Create new connection
        logger.info(
            "Creating new connection",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.ConnectionPool",
            data={"provider_key": provider_key}
        )
        connection = await factory_func()
        self.connections[provider_key] = {
            "connection": connection,
            "created_at": now,
            "last_used": now
        }

        self.connection_stats["total_created"] += 1
        self.connection_stats["active_connections"] = len(self.connections)

        logger.info(
            "New connection created and added to pool",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.ConnectionPool",
            data={
                "provider_key": provider_key,
                "pool_size": len(self.connections),
                "stats": self.connection_stats
            }
        )

        return connection

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self.connection_stats,
            "pool_size": len(self.connections),
            "max_connections": self.max_connections
        }


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        logger.debug(
            "CircuitBreaker initialized",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.CircuitBreaker",
            data={"failure_threshold": failure_threshold, "recovery_timeout": recovery_timeout}
        )

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info(
                    "Circuit breaker transitioning to HALF_OPEN state",
                    LogCategory.LLM_OPERATIONS,
                    "app.llm.providers.CircuitBreaker"
                )
                return True
            logger.warn(
                "Circuit breaker is OPEN, blocking execution",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.CircuitBreaker",
                data={"failure_count": self.failure_count, "last_failure": str(self.last_failure_time)}
            )
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record successful execution."""
        old_state = self.state
        self.failure_count = 0
        self.state = "CLOSED"
        if old_state != "CLOSED":
            logger.info(
                "Circuit breaker reset to CLOSED state after successful execution",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.CircuitBreaker"
            )

    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            old_state = self.state
            self.state = "OPEN"
            if old_state != "OPEN":
                logger.error(
                    "Circuit breaker opened due to failures",
                    LogCategory.LLM_OPERATIONS,
                    "app.llm.providers.CircuitBreaker",
                    data={"failure_count": self.failure_count, "threshold": self.failure_threshold}
                )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True

        time_since_failure = (datetime.now() - self.last_failure_time).seconds
        return time_since_failure >= self.recovery_timeout

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class ProviderMetrics:
    """Performance metrics for LLM providers."""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_execution_time = 0.0
        self.recent_errors = []
        self.max_recent_errors = 10
        logger.debug(
            "ProviderMetrics initialized",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.ProviderMetrics"
        )

    def record_success(self, execution_time: float) -> None:
        """Record successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_execution_time += execution_time
        logger.debug(
            "Request success recorded",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.ProviderMetrics",
            data={"execution_time": execution_time, "total_requests": self.total_requests}
        )

    def record_failure(self, execution_time: float, error_message: str) -> None:
        """Record failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.total_execution_time += execution_time

        # Keep recent errors
        self.recent_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "execution_time": execution_time
        })

        # Limit recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]

        logger.error(
            "Request failure recorded",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.ProviderMetrics",
            data={"error_message": error_message, "execution_time": execution_time, "total_failures": self.failed_requests}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_execution_time = (
            self.total_execution_time / self.total_requests
            if self.total_requests > 0 else 0
        )

        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0 else 0
        )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "recent_errors": self.recent_errors
        }


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers with production-ready features.

    PHASE 2 IMPROVEMENTS:
    - Merged with ProductionProvider layer
    - Built-in connection pooling
    - Circuit breaker pattern
    - Performance metrics
    - Comprehensive logging
    - Streaming support
    """

    def __init__(self, credentials: Optional[ProviderCredentials] = None):
        self.credentials = credentials
        self.provider_type = self._get_provider_type()
        self._is_initialized = False

        # Production-ready infrastructure
        self._connection_pool = ConnectionPool()
        self._circuit_breaker = CircuitBreaker()
        self._metrics = ProviderMetrics()

        logger.info(
            "LLM Provider initialized",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.LLMProvider",
            data={"provider": self.provider_type.value, "has_credentials": credentials is not None}
        )

    @abstractmethod
    def _get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        pass

    @abstractmethod
    async def _create_llm_connection(self, config: LLMConfig) -> BaseLanguageModel:
        """Create a new LLM connection (provider-specific implementation)."""
        pass

    @abstractmethod
    async def _test_basic_connectivity(self) -> None:
        """Test basic provider connectivity (provider-specific implementation)."""
        pass

    async def initialize(self) -> bool:
        """Initialize the provider."""
        try:
            if self._is_initialized:
                logger.debug(
                    "Provider already initialized",
                    LogCategory.LLM_OPERATIONS,
                    "app.llm.providers.LLMProvider",
                    data={"provider": self.provider_type.value}
                )
                return True

            logger.info(
                "Initializing provider",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                data={"provider": self.provider_type.value}
            )

            # Test basic connectivity
            await self._test_basic_connectivity()

            self._is_initialized = True
            logger.info(
                "Provider initialized successfully",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                data={"provider": self.provider_type.value}
            )
            return True

        except Exception as e:
            logger.error(
                "Provider initialization failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                error=e,
                data={"provider": self.provider_type.value, "error_type": type(e).__name__}
            )
            return False

    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        pass

    @abstractmethod
    async def validate_model(self, model_id: str) -> bool:
        """Validate if a model is available."""
        pass

    async def create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """
        Create LLM instance with production-ready features.

        Features:
        - Connection pooling
        - Circuit breaker protection
        - Performance metrics
        - Comprehensive logging
        """
        if not self._is_initialized:
            logger.info(
                "Provider not initialized, initializing now",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                data={"provider": self.provider_type.value}
            )
            await self.initialize()

        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            error_msg = "Circuit breaker is open - too many recent failures"
            logger.error(
                error_msg,
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                data={"provider": self.provider_type.value, "circuit_breaker_stats": self._circuit_breaker.get_stats()}
            )
            raise ConnectionError(error_msg)

        provider_key = f"{self.provider_type.value}_{config.model_id}"

        try:
            start_time = time.time()
            logger.info(
                "Creating LLM instance",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                data={
                    "provider": self.provider_type.value,
                    "model": config.model_id,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens
                }
            )

            # Get connection from pool
            llm = await self._connection_pool.get_connection(
                provider_key,
                lambda: self._create_llm_connection(config)
            )

            # Test the connection (skip for phi4 to avoid timeout issues)
            if config.model_id != "phi4:latest":
                await self._test_llm_instance(llm)
            else:
                logger.info(
                    "Skipping LLM test for phi4:latest to avoid timeout",
                    LogCategory.LLM_OPERATIONS,
                    "app.llm.providers.LLMProvider",
                    data={"model": config.model_id}
                )

            # Record success
            execution_time = time.time() - start_time
            self._circuit_breaker.record_success()
            self._metrics.record_success(execution_time)

            logger.info(
                "LLM instance created successfully",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                data={
                    "provider": self.provider_type.value,
                    "model": config.model_id,
                    "execution_time": execution_time,
                    "pool_stats": self._connection_pool.get_stats()
                }
            )

            return llm

        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._circuit_breaker.record_failure()
            self._metrics.record_failure(execution_time, str(e))

            logger.error(
                "Failed to create LLM instance",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                error=e,
                data={
                    "provider": self.provider_type.value,
                    "model": config.model_id,
                    "error_type": type(e).__name__,
                    "execution_time": execution_time,
                    "metrics": self._metrics.get_stats()
                }
            )
            raise

    async def create_streaming_llm(self, config: LLMConfig) -> AsyncGenerator[str, None]:
        """
        Create streaming LLM instance for long responses.

        PHASE 2 IMPROVEMENT: New streaming support
        """
        llm = await self.create_llm_instance(config)

        logger.info(
            "Creating streaming LLM",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.LLMProvider",
            data={"provider": self.provider_type.value, "model": config.model_id}
        )

        # Return the LLM instance configured for streaming
        # The actual streaming happens when the LLM is invoked
        return llm

    async def _test_llm_instance(self, llm: BaseLanguageModel) -> None:
        """Test LLM instance with a simple request."""
        try:
            logger.debug(
                "Testing LLM instance with simple prompt",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider"
            )
            # Use simple string instead of HumanMessage for better compatibility
            test_prompt = "Test"

            # Test with timeout - use string prompt (longer timeout for large models)
            response = await asyncio.wait_for(
                llm.ainvoke(test_prompt),
                timeout=60.0
            )

            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid response from LLM")

            logger.debug(
                "LLM instance test successful",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                data={"response_length": len(str(response.content))}
            )

        except asyncio.TimeoutError:
            logger.error(
                "LLM test timed out after 60 seconds",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider"
            )
            raise ConnectionError("LLM test timed out")
        except Exception as e:
            logger.error(
                "LLM test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.LLMProvider",
                error=e,
                data={"error_type": type(e).__name__}
            )
            raise ConnectionError(f"LLM test failed: {e}")

    @abstractmethod
    async def test_connection(self) -> ProviderStatus:
        """Test provider connection and authentication."""
        pass

    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        logger.info(
            "Cleaning up provider resources",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.LLMProvider",
            data={"provider": self.provider_type.value}
        )
        # Connection pool cleanup is handled automatically
        self._is_initialized = False
        logger.info(
            "Provider cleanup complete",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.LLMProvider",
            data={"provider": self.provider_type.value}
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get provider metrics."""
        return {
            "provider": self.provider_type.value,
            "connection_pool": self._connection_pool.get_stats(),
            "circuit_breaker": self._circuit_breaker.get_stats(),
            "performance": self._metrics.get_stats()
        }


# ============================================================================
# OLLAMA PROVIDER
# ============================================================================

class OllamaProvider(LLMProvider):
    """
    Ollama provider implementation with production-ready features.

    PHASE 2 IMPROVEMENTS:
    - Merged with ProductionOllamaProvider
    - Removed HTTPClient dependency
    - Direct aiohttp usage for API calls
    - Built-in connection pooling via base class
    """

    def _get_provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    def _get_keep_alive_setting(self) -> str:
        """Get the keep_alive setting from configuration."""
        try:
            from app.config.settings import get_settings
            settings = get_settings()
            return settings.OLLAMA_KEEP_ALIVE
        except Exception:
            return "30m"  # Default fallback

    async def _test_basic_connectivity(self) -> None:
        """Test Ollama server connectivity."""
        import aiohttp

        base_url = (
            self.credentials.base_url if self.credentials
            else PROVIDER_DEFAULTS[ProviderType.OLLAMA]["base_url"]
        )

        try:
            logger.debug(
                "Testing Ollama connectivity",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                data={"base_url": base_url}
            )
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status != 200:
                        raise ConnectionError(f"Ollama server returned {response.status}")

                    data = await response.json()
                    models = data.get("models", [])
                    logger.info(
                        "Ollama connectivity test passed",
                        LogCategory.LLM_OPERATIONS,
                        "app.llm.providers.OllamaProvider",
                        data={"base_url": base_url, "available_models": len(models)}
                    )

        except Exception as e:
            logger.error(
                "Ollama connectivity test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                error=e,
                data={"base_url": base_url, "error_type": type(e).__name__}
            )
            raise ConnectionError(f"Ollama connectivity test failed: {e}")

    async def _create_llm_connection(self, config: LLMConfig) -> BaseLanguageModel:
        """Create production-ready Ollama LLM connection."""
        if ChatOllama is None:
            raise ImportError("langchain_ollama is required for Ollama provider")

        base_url = (
            self.credentials.base_url if self.credentials
            else PROVIDER_DEFAULTS[ProviderType.OLLAMA]["base_url"]
        )

        logger.debug(
            "Creating Ollama LLM connection",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.OllamaProvider",
            data={
                "model": config.model_id,
                "base_url": base_url,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        )

        # Production configuration
        ollama_config = {
            "model": config.model_id,
            "base_url": base_url,
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
            "timeout": PROVIDER_DEFAULTS[ProviderType.OLLAMA]["timeout"],
            "num_ctx": config.additional_params.get("num_ctx", 4096),
            "repeat_penalty": config.additional_params.get("repeat_penalty", 1.1),
            "top_k": config.top_k or 40,
            "top_p": config.top_p or 0.9,
            # Production settings
            "num_thread": config.additional_params.get("num_thread", 8),
            "num_gpu": config.additional_params.get("num_gpu", 1),
            "main_gpu": config.additional_params.get("main_gpu", 0),
            # Keep model loaded in memory to avoid reload delays
            "keep_alive": config.additional_params.get("keep_alive", self._get_keep_alive_setting()),
        }

        # Add any additional parameters
        ollama_config.update({
            k: v for k, v in config.additional_params.items()
            if k not in ollama_config
        })

        try:
            llm = ChatOllama(**ollama_config)
            logger.info(
                "Ollama LLM created successfully",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                data={
                    "model": config.model_id,
                    "base_url": base_url,
                    "config_keys": list(ollama_config.keys())
                }
            )
            return llm

        except Exception as e:
            logger.error(
                "Failed to create Ollama LLM",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                error=e,
                data={"model": config.model_id, "error_type": type(e).__name__}
            )
            raise

    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Ollama models."""
        import aiohttp

        base_url = (
            self.credentials.base_url if self.credentials
            else PROVIDER_DEFAULTS[ProviderType.OLLAMA]["base_url"]
        )

        try:
            logger.debug(
                "Fetching available Ollama models",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                data={"base_url": base_url}
            )
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url}/api/tags") as response:
                    response.raise_for_status()
                    data = await response.json()

                    models = []
                    for model_data in data.get("models", []):
                        # Parse context_length properly
                        context_length = None
                        try:
                            param_size = model_data.get("details", {}).get("parameter_size", "Unknown")
                            if param_size and param_size != "Unknown":
                                # Handle formats like "108.6B", "7B", "13B", etc.
                                if isinstance(param_size, str):
                                    param_size = param_size.upper().replace("B", "").replace("M", "")
                                    if "." in param_size:
                                        # Convert "108.6" to 108600 (millions of parameters)
                                        context_length = int(float(param_size) * 1000)
                                    else:
                                        # Convert "7" to 7000 (millions of parameters)
                                        context_length = int(param_size) * 1000
                                elif isinstance(param_size, (int, float)):
                                    context_length = int(param_size)
                        except (ValueError, TypeError):
                            # If parsing fails, leave as None
                            context_length = None

                        model_info = ModelInfo(
                            id=model_data["name"],
                            name=model_data["name"],
                            provider=ProviderType.OLLAMA,
                            context_length=context_length,
                            description=f"Ollama model: {model_data['name']}",
                            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION]
                        )
                        models.append(model_info)

                    logger.info(
                        "Successfully fetched Ollama models",
                        LogCategory.LLM_OPERATIONS,
                        "app.llm.providers.OllamaProvider",
                        data={"count": len(models), "models": [m.id for m in models]}
                    )
                    return models

        except Exception as e:
            logger.error(
                "Failed to get Ollama models",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                error=e,
                data={"base_url": base_url, "error_type": type(e).__name__}
            )
            # Return default models as fallback
            logger.info(
                "Returning default Ollama models as fallback",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider"
            )
            return [
                ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider=ProviderType.OLLAMA,
                    context_length=None,
                    description=f"Default Ollama model: {model_id}",
                    capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION]
                )
                for model_id in DEFAULT_MODELS[ProviderType.OLLAMA]
            ]

    async def validate_model(self, model_id: str) -> bool:
        """Validate Ollama model availability."""
        try:
            logger.debug(
                "Validating Ollama model",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                data={"model_id": model_id}
            )
            models = await self.get_available_models()
            is_valid = any(model.id == model_id for model in models)
            logger.debug(
                "Model validation result",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                data={"model_id": model_id, "is_valid": is_valid}
            )
            return is_valid
        except Exception as e:
            logger.warn(
                "Model validation failed, checking defaults",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                data={"model_id": model_id, "error": str(e)}
            )
            return model_id in DEFAULT_MODELS[ProviderType.OLLAMA]

    async def test_connection(self) -> ProviderStatus:
        """Test Ollama connection."""
        status = ProviderStatus(provider=ProviderType.OLLAMA)

        try:
            logger.info(
                "Testing Ollama connection",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider"
            )
            start_time = time.time()

            # Test basic connectivity
            await self._test_basic_connectivity()

            # Get available models with error handling
            try:
                models = await self.get_available_models()
                status.available_models = [model.id for model in models]
            except Exception as model_error:
                logger.warn(
                    "Failed to get Ollama models during connection test",
                    LogCategory.LLM_OPERATIONS,
                    "app.llm.providers.OllamaProvider",
                    data={"error": str(model_error)}
                )
                # Still mark as available if basic connectivity works
                status.available_models = []

            response_time = (time.time() - start_time) * 1000

            status.is_available = True
            status.is_authenticated = True  # Ollama doesn't require auth
            status.response_time_ms = response_time

            logger.info(
                "Ollama connection test successful",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                data={"response_time_ms": response_time, "models_count": len(status.available_models)}
            )

        except Exception as e:
            status.error_message = str(e)
            logger.error(
                "Ollama connection test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OllamaProvider",
                error=e,
                data={"error_type": type(e).__name__}
            )

        return status


# ============================================================================
# OPENAI PROVIDER
# ============================================================================

class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation with production-ready features.

    PHASE 2 IMPROVEMENTS:
    - Merged with ProductionOpenAIProvider
    - Built-in connection pooling via base class
    - Enhanced error handling and logging
    """

    def _get_provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    async def _test_basic_connectivity(self) -> None:
        """Test OpenAI API connectivity."""
        if not self.credentials or not self.credentials.api_key:
            raise ValueError("OpenAI API key is required")

        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.credentials.api_key}",
            "Content-Type": "application/json"
        }

        base_url = self.credentials.base_url or "https://api.openai.com/v1"

        try:
            logger.debug(
                "Testing OpenAI connectivity",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OpenAIProvider",
                data={"base_url": base_url}
            )
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url}/models", headers=headers) as response:
                    if response.status == 401:
                        raise ConnectionError("Invalid OpenAI API key")
                    elif response.status != 200:
                        raise ConnectionError(f"OpenAI API returned {response.status}")

                    data = await response.json()
                    models = data.get("data", [])
                    logger.info(
                        "OpenAI connectivity test passed",
                        LogCategory.LLM_OPERATIONS,
                        "app.llm.providers.OpenAIProvider",
                        data={"base_url": base_url, "available_models": len(models)}
                    )

        except Exception as e:
            logger.error(
                "OpenAI connectivity test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OpenAIProvider",
                error=e,
                data={"base_url": base_url, "error_type": type(e).__name__}
            )
            raise ConnectionError(f"OpenAI connectivity test failed: {e}")

    async def _create_llm_connection(self, config: LLMConfig) -> BaseLanguageModel:
        """Create production-ready OpenAI LLM connection."""
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for OpenAI provider")

        if not self.credentials or not self.credentials.api_key:
            raise ValueError("OpenAI API key is required")

        logger.debug(
            "Creating OpenAI LLM connection",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.OpenAIProvider",
            data={
                "model": config.model_id,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        )

        # Production configuration
        openai_config = {
            "model": config.model_id,
            "api_key": self.credentials.api_key,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.additional_params.get("frequency_penalty", 0),
            "presence_penalty": config.additional_params.get("presence_penalty", 0),
            "timeout": PROVIDER_DEFAULTS[ProviderType.OPENAI]["timeout"],
            "max_retries": 3,
            "request_timeout": 60,
        }

        # Add base URL if provided
        if self.credentials.base_url:
            openai_config["base_url"] = self.credentials.base_url

        # Add organization if provided
        if hasattr(self.credentials, 'organization') and self.credentials.organization:
            openai_config["organization"] = self.credentials.organization

        try:
            llm = ChatOpenAI(**openai_config)
            logger.info(
                "OpenAI LLM created successfully",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OpenAIProvider",
                data={
                    "model": config.model_id,
                    "has_base_url": bool(self.credentials.base_url),
                    "has_organization": bool(getattr(self.credentials, 'organization', None))
                }
            )
            return llm

        except Exception as e:
            logger.error(
                "Failed to create OpenAI LLM",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OpenAIProvider",
                error=e,
                data={"model": config.model_id, "error_type": type(e).__name__}
            )
            raise

    async def get_available_models(self) -> List[ModelInfo]:
        """Get available OpenAI models."""
        logger.debug(
            "Fetching available OpenAI models",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.OpenAIProvider"
        )
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

        logger.info(
            "Successfully fetched OpenAI models",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.OpenAIProvider",
            data={"count": len(models), "models": [m.id for m in models]}
        )
        return models

    async def validate_model(self, model_id: str) -> bool:
        """Validate OpenAI model availability."""
        logger.debug(
            "Validating OpenAI model",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.OpenAIProvider",
            data={"model_id": model_id}
        )
        is_valid = model_id in DEFAULT_MODELS[ProviderType.OPENAI]
        logger.debug(
            "Model validation result",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.OpenAIProvider",
            data={"model_id": model_id, "is_valid": is_valid}
        )
        return is_valid

    async def test_connection(self) -> ProviderStatus:
        """Test OpenAI connection."""
        status = ProviderStatus(provider=ProviderType.OPENAI)

        try:
            logger.info(
                "Testing OpenAI connection",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OpenAIProvider"
            )
            start_time = time.time()

            # Test basic connectivity
            await self._test_basic_connectivity()

            response_time = (time.time() - start_time) * 1000

            status.is_available = True
            status.is_authenticated = True
            status.available_models = DEFAULT_MODELS[ProviderType.OPENAI]
            status.response_time_ms = response_time

            logger.info(
                "OpenAI connection test successful",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OpenAIProvider",
                data={"response_time_ms": response_time, "models_count": len(status.available_models)}
            )

        except Exception as e:
            status.error_message = str(e)
            logger.error(
                "OpenAI connection test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.OpenAIProvider",
                error=e,
                data={"error_type": type(e).__name__}
            )

        return status


# ============================================================================
# ANTHROPIC PROVIDER
# ============================================================================

class AnthropicProvider(LLMProvider):
    """
    Anthropic provider implementation with production-ready features.

    PHASE 2 IMPROVEMENTS:
    - Merged with ProductionAnthropicProvider
    - Built-in connection pooling via base class
    - Enhanced error handling and logging
    """

    def _get_provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    async def _test_basic_connectivity(self) -> None:
        """Test Anthropic API connectivity."""
        if not self.credentials or not self.credentials.api_key:
            raise ValueError("Anthropic API key is required")

        # For Anthropic, we'll test with a simple message
        try:
            logger.debug(
                "Testing Anthropic connectivity",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider"
            )
            from langchain_core.messages import HumanMessage

            # Create a minimal test instance
            test_llm = ChatAnthropic(
                model="claude-3-haiku-20240307",  # Use fastest model for test
                api_key=self.credentials.api_key,
                max_tokens=10,
                timeout=10
            )

            # Simple test
            response = await asyncio.wait_for(
                test_llm.ainvoke([HumanMessage(content="Hi")]),
                timeout=15.0
            )

            if not response:
                raise ConnectionError("No response from Anthropic API")

            logger.info(
                "Anthropic connectivity test passed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider"
            )

        except Exception as e:
            logger.error(
                "Anthropic connectivity test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider",
                error=e,
                data={"error_type": type(e).__name__}
            )
            raise ConnectionError(f"Anthropic connectivity test failed: {e}")

    async def _create_llm_connection(self, config: LLMConfig) -> BaseLanguageModel:
        """Create production-ready Anthropic LLM connection."""
        if ChatAnthropic is None:
            raise ImportError("langchain_anthropic is required for Anthropic provider")

        if not self.credentials or not self.credentials.api_key:
            raise ValueError("Anthropic API key is required")

        logger.debug(
            "Creating Anthropic LLM connection",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.AnthropicProvider",
            data={
                "model": config.model_id,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        )

        # Production configuration
        anthropic_config = {
            "model": config.model_id,
            "api_key": self.credentials.api_key,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "timeout": PROVIDER_DEFAULTS[ProviderType.ANTHROPIC]["timeout"],
            "max_retries": 3,
        }

        # Add base URL if provided
        if self.credentials.base_url:
            anthropic_config["base_url"] = self.credentials.base_url

        try:
            llm = ChatAnthropic(**anthropic_config)
            logger.info(
                "Anthropic LLM created successfully",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider",
                data={"model": config.model_id, "has_base_url": bool(self.credentials.base_url)}
            )
            return llm

        except Exception as e:
            logger.error(
                "Failed to create Anthropic LLM",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider",
                error=e,
                data={"model": config.model_id, "error_type": type(e).__name__}
            )
            raise

    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Anthropic models."""
        logger.debug(
            "Fetching available Anthropic models",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.AnthropicProvider"
        )
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

        logger.info(
            "Successfully fetched Anthropic models",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.AnthropicProvider",
            data={"count": len(models), "models": [m.id for m in models]}
        )
        return models

    async def validate_model(self, model_id: str) -> bool:
        """Validate Anthropic model availability."""
        logger.debug(
            "Validating Anthropic model",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.AnthropicProvider",
            data={"model_id": model_id}
        )
        is_valid = model_id in DEFAULT_MODELS[ProviderType.ANTHROPIC]
        logger.debug(
            "Model validation result",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.AnthropicProvider",
            data={"model_id": model_id, "is_valid": is_valid}
        )
        return is_valid

    async def test_connection(self) -> ProviderStatus:
        """Test Anthropic connection."""
        status = ProviderStatus(provider=ProviderType.ANTHROPIC)

        try:
            logger.info(
                "Testing Anthropic connection",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider"
            )
            start_time = time.time()

            # Test basic connectivity
            await self._test_basic_connectivity()

            response_time = (time.time() - start_time) * 1000

            status.is_available = True
            status.is_authenticated = True
            status.available_models = DEFAULT_MODELS[ProviderType.ANTHROPIC]
            status.response_time_ms = response_time

            logger.info(
                "Anthropic connection test successful",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider",
                data={"response_time_ms": response_time, "models_count": len(status.available_models)}
            )

        except Exception as e:
            status.error_message = str(e)
            logger.error(
                "Anthropic connection test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.AnthropicProvider",
                error=e,
                data={"error_type": type(e).__name__}
            )

        return status


# ============================================================================
# GOOGLE PROVIDER
# ============================================================================

class GoogleProvider(LLMProvider):
    """
    Google Generative AI provider implementation with production-ready features.

    PHASE 2 IMPROVEMENTS:
    - NEW: ProductionGoogleProvider implementation
    - Built-in connection pooling via base class
    - Circuit breaker protection
    - Performance metrics
    - Enhanced error handling and logging
    """

    def _get_provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    async def _test_basic_connectivity(self) -> None:
        """Test Google API connectivity."""
        if not self.credentials or not self.credentials.api_key:
            raise ValueError("Google API key is required")

        try:
            logger.debug(
                "Testing Google connectivity",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider"
            )
            from langchain_core.messages import HumanMessage

            # Create a minimal test instance
            test_llm = ChatGoogleGenerativeAI(
                model="gemini-pro",  # Use basic model for test
                google_api_key=self.credentials.api_key,
                max_output_tokens=10,
                timeout=10
            )

            # Simple test
            response = await asyncio.wait_for(
                test_llm.ainvoke([HumanMessage(content="Hi")]),
                timeout=15.0
            )

            if not response:
                raise ConnectionError("No response from Google API")

            logger.info(
                "Google connectivity test passed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider"
            )

        except Exception as e:
            logger.error(
                "Google connectivity test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider",
                error=e,
                data={"error_type": type(e).__name__}
            )
            raise ConnectionError(f"Google connectivity test failed: {e}")

    async def _create_llm_connection(self, config: LLMConfig) -> BaseLanguageModel:
        """Create production-ready Google LLM connection."""
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain_google_genai is required for Google provider")

        if not self.credentials or not self.credentials.api_key:
            raise ValueError("Google API key is required")

        logger.debug(
            "Creating Google LLM connection",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.GoogleProvider",
            data={
                "model": config.model_id,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
        )

        # Production configuration
        google_config = {
            "model": config.model_id,
            "google_api_key": self.credentials.api_key,
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
            "top_p": config.top_p,
            "top_k": config.top_k or 40,
            "timeout": PROVIDER_DEFAULTS[ProviderType.GOOGLE]["timeout"],
            "max_retries": 3,
        }

        # Add safety settings if provided
        if "safety_settings" in config.additional_params:
            google_config["safety_settings"] = config.additional_params["safety_settings"]

        # Add candidate count if provided
        if "candidate_count" in config.additional_params:
            google_config["candidate_count"] = config.additional_params["candidate_count"]

        try:
            llm = ChatGoogleGenerativeAI(**google_config)
            logger.info(
                "Google LLM created successfully",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider",
                data={
                    "model": config.model_id,
                    "has_safety_settings": bool("safety_settings" in config.additional_params)
                }
            )
            return llm

        except Exception as e:
            logger.error(
                "Failed to create Google LLM",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider",
                error=e,
                data={"model": config.model_id, "error_type": type(e).__name__}
            )
            raise

    async def get_available_models(self) -> List[ModelInfo]:
        """Get available Google models."""
        logger.debug(
            "Fetching available Google models",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.GoogleProvider"
        )
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

        logger.info(
            "Successfully fetched Google models",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.GoogleProvider",
            data={"count": len(models), "models": [m.id for m in models]}
        )
        return models

    async def validate_model(self, model_id: str) -> bool:
        """Validate Google model availability."""
        logger.debug(
            "Validating Google model",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.GoogleProvider",
            data={"model_id": model_id}
        )
        is_valid = model_id in DEFAULT_MODELS[ProviderType.GOOGLE]
        logger.debug(
            "Model validation result",
            LogCategory.LLM_OPERATIONS,
            "app.llm.providers.GoogleProvider",
            data={"model_id": model_id, "is_valid": is_valid}
        )
        return is_valid

    async def test_connection(self) -> ProviderStatus:
        """Test Google connection."""
        status = ProviderStatus(provider=ProviderType.GOOGLE)

        try:
            logger.info(
                "Testing Google connection",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider"
            )
            start_time = time.time()

            # Test basic connectivity
            await self._test_basic_connectivity()

            response_time = (time.time() - start_time) * 1000

            status.is_available = True
            status.is_authenticated = True
            status.available_models = DEFAULT_MODELS[ProviderType.GOOGLE]
            status.response_time_ms = response_time

            logger.info(
                "Google connection test successful",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider",
                data={"response_time_ms": response_time, "models_count": len(status.available_models)}
            )

        except Exception as e:
            status.error_message = str(e)
            logger.error(
                "Google connection test failed",
                LogCategory.LLM_OPERATIONS,
                "app.llm.providers.GoogleProvider",
                error=e,
                data={"error_type": type(e).__name__}
            )

        return status
