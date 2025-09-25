"""
Configuration Management System for AI Agents.

This module provides centralized configuration management with:
- Layered configuration (defaults → environment → user overrides)
- Smart defaults with validation
- Environment variable support
- Configuration validation and constraints
- Hot reloading capabilities
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

# Integration with Global Config Manager
try:
    from app.core.global_config_manager import global_config_manager, ConfigurationSection
    GLOBAL_CONFIG_AVAILABLE = True
    logger.info("Global Config Manager integration enabled")
except ImportError:
    logger.warning("Global Config Manager not available - using standalone mode")
    GLOBAL_CONFIG_AVAILABLE = False
    global_config_manager = None
    ConfigurationSection = None


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


class ConfigLayer(Enum):
    """Configuration layer priorities."""
    DEFAULTS = 1
    ENVIRONMENT = 2
    USER_OVERRIDE = 3
    RUNTIME_OVERRIDE = 4


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_path: str
    rule_type: str  # "range", "choices", "type", "required"
    constraint: Any
    error_message: str


class AgentConfigurationManager:
    """
    Centralized configuration management system for AI agents.
    
    Provides layered configuration with smart defaults, validation,
    and environment variable support. Configuration files are stored
    in the data/ directory while Python code is in app/.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            data_dir: Path to data directory containing config files
        """
        # Default to data/ directory relative to project root
        if data_dir is None:
            # Get project root (assuming app/config/agent_config_manager.py structure)
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
        
        self.data_dir = data_dir
        self.config_dir = data_dir / "config"
        self._config_cache: Dict[str, Any] = {}
        self._validation_rules: List[ConfigValidationRule] = []
        self._watchers: Dict[str, Any] = {}
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration layers
        self._load_configuration()
        self._setup_validation_rules()
        
        logger.info("Agent configuration manager initialized", 
                   config_dir=str(self.config_dir),
                   data_dir=str(self.data_dir))
    
    def _load_configuration(self):
        """Load configuration from all layers."""
        try:
            # Layer 1: Load defaults from data/config/agent_defaults.yaml
            defaults_file = self.config_dir / "agent_defaults.yaml"
            if defaults_file.exists():
                with open(defaults_file, 'r', encoding='utf-8') as f:
                    defaults = yaml.safe_load(f)
                self._config_cache = defaults.copy()
                logger.info("Loaded default configuration", file=str(defaults_file))
            else:
                logger.warning("Default configuration file not found", file=str(defaults_file))
                self._config_cache = self._get_fallback_config()
            
            # Layer 2: Load environment overrides
            self._apply_environment_overrides()
            
            # Layer 3: Load user overrides from data/config/user_config.yaml
            user_config_file = self.config_dir / "user_config.yaml"
            if user_config_file.exists():
                with open(user_config_file, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                self._merge_config(self._config_cache, user_config)
                logger.info("Loaded user configuration", file=str(user_config_file))
            
            # Layer 4: Load global config (legacy support) from data/config/global_config.json
            global_config_file = self.config_dir / "global_config.json"
            if global_config_file.exists():
                with open(global_config_file, 'r', encoding='utf-8') as f:
                    global_config = json.load(f)
                self._merge_config(self._config_cache, global_config)
                logger.info("Loaded global configuration", file=str(global_config_file))

            # Layer 5: Integrate with Global Config Manager (runtime settings)
            self._integrate_with_global_config()

            # Add generation timestamp
            self._config_cache["generated_at"] = datetime.now().isoformat()

        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration if no config files are found."""
        return {
            "llm_providers": {
                "default_provider": "ollama",
                "ollama": {
                    "default_model": "llama3.2:latest",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "timeout_seconds": 300
                }
            },
            "agent_types": {
                "react": {
                    "framework": "react",
                    "default_temperature": 0.7,
                    "max_iterations": 50,
                    "timeout_seconds": 300,
                    "enable_memory": True,
                    "memory_type": "simple"
                }
            },
            "performance": {
                "max_execution_time_seconds": 3600,
                "max_iterations_hard_limit": 200,
                "max_memory_per_agent_mb": 1024,
                "max_concurrent_agents": 50,
                "default_decision_threshold": 0.6
            },
            "infrastructure": {
                "health_check_interval_seconds": 60,
                "cache_ttl_seconds": 300,
                "connection_pool_size": 10
            },
            "security": {
                "requests_per_minute": 60,
                "max_file_size_mb": 100
            },
            "logging": {
                "default_level": "INFO",
                "enable_metrics": True
            }
        }
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            # LLM Provider settings
            "AGENT_DEFAULT_PROVIDER": "llm_providers.default_provider",
            "AGENT_DEFAULT_MODEL": "llm_providers.ollama.default_model",
            "AGENT_DEFAULT_TEMPERATURE": "llm_providers.ollama.temperature",
            "AGENT_MAX_TOKENS": "llm_providers.ollama.max_tokens",
            "AGENT_TIMEOUT": "llm_providers.ollama.timeout_seconds",
            
            # Performance settings
            "AGENT_MAX_ITERATIONS": "performance.max_iterations_hard_limit",
            "AGENT_MAX_EXECUTION_TIME": "performance.max_execution_time_seconds",
            "AGENT_MAX_MEMORY_MB": "performance.max_memory_per_agent_mb",
            "AGENT_MAX_CONCURRENT": "performance.max_concurrent_agents",
            "AGENT_DECISION_THRESHOLD": "performance.default_decision_threshold",
            
            # Infrastructure settings
            "AGENT_HEALTH_CHECK_INTERVAL": "infrastructure.health_check_interval_seconds",
            "AGENT_CACHE_TTL": "infrastructure.cache_ttl_seconds",
            "AGENT_CONNECTION_POOL_SIZE": "infrastructure.connection_pool_size",
            
            # Security settings
            "AGENT_RATE_LIMIT_PER_MINUTE": "security.requests_per_minute",
            "AGENT_MAX_FILE_SIZE_MB": "security.max_file_size_mb",
            
            # Logging settings
            "AGENT_LOG_LEVEL": "logging.default_level",
            "AGENT_ENABLE_METRICS": "logging.enable_metrics",
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                self._set_nested_value(self._config_cache, config_path, converted_value)
                logger.debug("Applied environment override", 
                           env_var=env_var, config_path=config_path, value=converted_value)
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def _setup_validation_rules(self):
        """Setup configuration validation rules."""
        self._validation_rules = [
            # LLM Provider validation
            ConfigValidationRule(
                "llm_providers.default_provider",
                "choices",
                ["ollama", "openai", "anthropic", "google"],
                "Default provider must be one of: ollama, openai, anthropic, google"
            ),

            # Performance validation
            ConfigValidationRule(
                "performance.max_execution_time_seconds",
                "range",
                (30, 7200),  # 30 seconds to 2 hours
                "Max execution time must be between 30 and 7200 seconds"
            ),
            ConfigValidationRule(
                "performance.max_iterations_hard_limit",
                "range",
                (1, 1000),
                "Max iterations must be between 1 and 1000"
            ),
            ConfigValidationRule(
                "performance.max_memory_per_agent_mb",
                "range",
                (64, 8192),  # 64MB to 8GB
                "Max memory per agent must be between 64 and 8192 MB"
            ),
            ConfigValidationRule(
                "performance.default_decision_threshold",
                "range",
                (0.1, 0.9),
                "Decision threshold must be between 0.1 and 0.9"
            ),

            # Temperature validation for all providers
            ConfigValidationRule(
                "llm_providers.ollama.temperature",
                "range",
                (0.0, 2.0),
                "Temperature must be between 0.0 and 2.0"
            ),

            # Security validation
            ConfigValidationRule(
                "security.requests_per_minute",
                "range",
                (1, 10000),
                "Requests per minute must be between 1 and 10000"
            ),
            ConfigValidationRule(
                "security.max_file_size_mb",
                "range",
                (1, 1000),
                "Max file size must be between 1 and 1000 MB"
            ),
        ]

    def validate_configuration(self) -> List[str]:
        """
        Validate current configuration against rules.

        Returns:
            List of validation error messages
        """
        errors = []

        for rule in self._validation_rules:
            value = self._get_nested_value(self._config_cache, rule.field_path)

            if value is None and rule.rule_type == "required":
                errors.append(f"Required field missing: {rule.field_path}")
                continue

            if value is None:
                continue  # Skip validation for optional missing fields

            if rule.rule_type == "range":
                min_val, max_val = rule.constraint
                if not (min_val <= value <= max_val):
                    errors.append(f"{rule.field_path}: {rule.error_message} (got {value})")

            elif rule.rule_type == "choices":
                if value not in rule.constraint:
                    errors.append(f"{rule.field_path}: {rule.error_message} (got {value})")

            elif rule.rule_type == "type":
                if not isinstance(value, rule.constraint):
                    errors.append(f"{rule.field_path}: must be of type {rule.constraint.__name__} (got {type(value).__name__})")

        return errors

    def _integrate_with_global_config(self):
        """Integrate with Global Config Manager for runtime settings."""
        if not GLOBAL_CONFIG_AVAILABLE or not global_config_manager:
            return

        try:
            # Get LLM provider settings from Global Config Manager
            llm_config = global_config_manager._current_config.get(ConfigurationSection.LLM_PROVIDERS, {})
            if llm_config:
                # Merge LLM settings from global config
                if "default_provider" in llm_config:
                    self._set_nested_value(self._config_cache, "llm_providers.default_provider", llm_config["default_provider"])

                # Merge provider-specific settings
                for provider, settings in llm_config.items():
                    if isinstance(settings, dict) and provider != "default_provider":
                        for key, value in settings.items():
                            self._set_nested_value(self._config_cache, f"llm_providers.{provider}.{key}", value)

            # Get agent management settings
            agent_config = global_config_manager._current_config.get(ConfigurationSection.AGENT_MANAGEMENT, {})
            if agent_config:
                # Merge agent management settings
                for key, value in agent_config.items():
                    if key in ["max_concurrent_agents", "default_timeout", "health_check_interval"]:
                        target_path = f"performance.{key}" if key.startswith("max_") else f"infrastructure.{key}"
                        self._set_nested_value(self._config_cache, target_path, value)

            logger.info("Successfully integrated with Global Config Manager")

        except Exception as e:
            logger.warning(f"Failed to integrate with Global Config Manager: {str(e)}")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path.

        Args:
            path: Dot-separated configuration path
            default: Default value if not found

        Returns:
            Configuration value
        """
        return self._get_nested_value(self._config_cache, path, default)

    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Agent configuration dictionary
        """
        base_config = self.get(f"agent_types.{agent_type}", {})
        if not base_config:
            logger.warning("No configuration found for agent type", agent_type=agent_type)
            # Return basic defaults
            base_config = {
                "framework": "basic",
                "default_temperature": 0.7,
                "max_iterations": 50,
                "timeout_seconds": 300,
                "enable_memory": True,
                "memory_type": "simple"
            }

        return base_config

    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get LLM configuration for a provider.

        Args:
            provider: LLM provider name (defaults to default_provider)

        Returns:
            LLM configuration dictionary
        """
        if not provider:
            provider = self.get("llm_providers.default_provider", "ollama")

        return self.get(f"llm_providers.{provider}", {})

    def get_system_prompt(self, template_name: str = "base_template", **kwargs) -> str:
        """
        Get system prompt template with formatting.

        Args:
            template_name: Name of the prompt template
            **kwargs: Template formatting arguments

        Returns:
            Formatted system prompt
        """
        template = self.get(f"system_prompts.{template_name}", "")
        if not template:
            template = self.get("system_prompts.base_template", "You are a helpful AI assistant.")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning("Missing template variable", template=template_name, missing_var=str(e))
            return template

    def get_performance_limits(self) -> Dict[str, Any]:
        """Get performance and safety limits."""
        return {
            "max_execution_time": self.get("performance.max_execution_time_seconds", 3600),
            "max_iterations": self.get("performance.max_iterations_hard_limit", 200),
            "max_memory_mb": self.get("performance.max_memory_per_agent_mb", 1024),
            "max_concurrent": self.get("performance.max_concurrent_agents", 50),
            "decision_threshold": self.get("performance.default_decision_threshold", 0.6),
        }

    def get_security_limits(self) -> Dict[str, Any]:
        """Get security and rate limiting configuration."""
        return {
            "requests_per_minute": self.get("security.requests_per_minute", 60),
            "requests_per_hour": self.get("security.requests_per_hour", 1000),
            "max_file_size_mb": self.get("security.max_file_size_mb", 100),
            "allowed_file_types": self.get("security.allowed_file_types", [".txt", ".pdf", ".md"]),
        }

    def reload_configuration(self):
        """Reload configuration from files."""
        logger.info("Reloading configuration")
        self._config_cache.clear()
        self._load_configuration()

        # Validate after reload
        errors = self.validate_configuration()
        if errors:
            logger.error("Configuration validation failed after reload", errors=errors)
            raise ConfigurationError(f"Configuration validation failed: {errors}")

    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._config_cache.copy()


# Global configuration manager instance
_config_manager: Optional[AgentConfigurationManager] = None


def get_agent_config_manager() -> AgentConfigurationManager:
    """Get the global agent configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = AgentConfigurationManager()
    return _config_manager


def initialize_agent_config_manager(data_dir: Optional[Path] = None) -> AgentConfigurationManager:
    """Initialize the global agent configuration manager."""
    global _config_manager
    _config_manager = AgentConfigurationManager(data_dir)
    return _config_manager
