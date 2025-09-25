"""
Agent Configuration Integration Module.

This module provides integration functions to replace hardcoded values
throughout the agent system with configuration-driven values.
"""

from typing import Dict, Any, Optional, List
import structlog
from pathlib import Path

from app.config.agent_config_manager import get_agent_config_manager, AgentConfigurationManager
from app.agents.base.agent import AgentConfig, AgentCapability
from app.agents.factory import AgentBuilderConfig, AgentType, MemoryType
from app.llm.models import LLMConfig, ProviderType

logger = structlog.get_logger(__name__)


class ConfigIntegration:
    """
    Integration layer between the configuration system and agent components.
    
    This class provides methods to create agent configurations using
    the centralized configuration system instead of hardcoded values.
    """
    
    def __init__(self, config_manager: Optional[AgentConfigurationManager] = None):
        """
        Initialize configuration integration.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or get_agent_config_manager()
        logger.info("Configuration integration initialized")
    
    def create_agent_config(
        self,
        name: str,
        description: str,
        agent_type: str = "react",
        capabilities: Optional[List[AgentCapability]] = None,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **overrides
    ) -> AgentConfig:
        """
        Create an AgentConfig using configuration system instead of hardcoded values.
        
        Args:
            name: Agent name
            description: Agent description
            agent_type: Type of agent
            capabilities: Agent capabilities (optional)
            tools: Tool list (optional)
            system_prompt: Custom system prompt (optional)
            **overrides: Configuration overrides
            
        Returns:
            Configured AgentConfig instance
        """
        # Get agent type configuration
        agent_config = self.config_manager.get_agent_config(agent_type)
        
        # Get LLM configuration
        llm_config = self.config_manager.get_llm_config()
        
        # Get performance limits
        performance = self.config_manager.get_performance_limits()
        
        # Use configured values instead of hardcoded ones
        config = AgentConfig(
            name=name,
            description=description,
            agent_type=agent_type,
            framework=agent_config.get("framework", "basic"),
            
            # LLM configuration from config system
            model_name=overrides.get("model_name") or llm_config.get("default_model", "llama3.2:latest"),
            model_provider=overrides.get("model_provider") or self.config_manager.get("llm_providers.default_provider", "ollama"),
            temperature=overrides.get("temperature") or agent_config.get("default_temperature", 0.7),
            max_tokens=overrides.get("max_tokens") or llm_config.get("max_tokens", 2048),
            
            # Performance configuration from config system
            max_iterations=overrides.get("max_iterations") or agent_config.get("max_iterations", 50),
            timeout_seconds=overrides.get("timeout_seconds") or agent_config.get("timeout_seconds", 300),
            
            # System prompt from config system
            system_prompt=system_prompt or self._get_system_prompt(agent_type, tools or []),
            
            # Capabilities and tools
            capabilities=capabilities or self._get_default_capabilities(agent_type),
            tools=tools or self._get_default_tools(agent_type),
            
            # Memory configuration
            memory_enabled=overrides.get("memory_enabled", agent_config.get("enable_memory", True)),
            memory_window=overrides.get("memory_window", 10),
            
            # Collaboration
            can_delegate=overrides.get("can_delegate", False),
            can_be_delegated_to=overrides.get("can_be_delegated_to", True),
            
            # Custom configuration
            custom_config=overrides.get("custom_config", {})
        )
        
        logger.info("Created agent config from configuration system",
                   name=name, agent_type=agent_type, 
                   model=config.model_name, provider=config.model_provider)
        
        return config
    
    def create_builder_config(
        self,
        name: str,
        description: str,
        agent_type: AgentType,
        capabilities: Optional[List[AgentCapability]] = None,
        tools: Optional[List[str]] = None,
        **overrides
    ) -> AgentBuilderConfig:
        """
        Create an AgentBuilderConfig using configuration system.
        
        Args:
            name: Agent name
            description: Agent description
            agent_type: Agent type enum
            capabilities: Agent capabilities
            tools: Tool list
            **overrides: Configuration overrides
            
        Returns:
            Configured AgentBuilderConfig instance
        """
        # Get configurations
        agent_config = self.config_manager.get_agent_config(agent_type.value)
        llm_config = self.config_manager.get_llm_config()
        performance = self.config_manager.get_performance_limits()
        
        # Create LLM configuration
        provider_name = overrides.get("provider") or self.config_manager.get("llm_providers.default_provider", "ollama")
        provider_type = self._get_provider_type(provider_name)
        
        llm_config_obj = LLMConfig(
            provider=provider_type,
            model_id=overrides.get("model_id") or llm_config.get("default_model", "llama3.2:latest"),
            temperature=overrides.get("temperature") or agent_config.get("default_temperature", 0.7),
            max_tokens=overrides.get("max_tokens") or llm_config.get("max_tokens", 2048),
            manual_selection=overrides.get("manual_selection", False)
        )
        
        # Determine memory type
        memory_type_str = overrides.get("memory_type") or agent_config.get("memory_type", "simple")
        memory_type = self._get_memory_type(memory_type_str)
        
        config = AgentBuilderConfig(
            name=name,
            description=description,
            agent_type=agent_type,
            llm_config=llm_config_obj,
            capabilities=capabilities or self._get_default_capabilities(agent_type.value),
            tools=tools or self._get_default_tools(agent_type.value),
            
            # Configuration-driven values
            system_prompt=overrides.get("system_prompt") or self._get_system_prompt(agent_type.value, tools or []),
            max_iterations=overrides.get("max_iterations") or agent_config.get("max_iterations", 50),
            timeout_seconds=overrides.get("timeout_seconds") or agent_config.get("timeout_seconds", 300),
            enable_memory=overrides.get("enable_memory", agent_config.get("enable_memory", True)),
            enable_learning=overrides.get("enable_learning", False),
            enable_collaboration=overrides.get("enable_collaboration", False),
            memory_type=memory_type,
            memory_config=overrides.get("memory_config"),
            custom_config=overrides.get("custom_config", {})
        )
        
        logger.info("Created builder config from configuration system",
                   name=name, agent_type=agent_type.value,
                   provider=provider_name, model=llm_config_obj.model_id)
        
        return config
    
    def _get_system_prompt(self, agent_type: str, tools: List[str]) -> str:
        """Get system prompt for agent type."""
        # Try agent-specific template first
        template_name = f"{agent_type}_template"
        prompt = self.config_manager.get_system_prompt(template_name, tools=", ".join(tools))
        
        # Fall back to base template if agent-specific not found
        if not prompt or prompt == self.config_manager.get("system_prompts.base_template", ""):
            prompt = self.config_manager.get_system_prompt("base_template", tools=", ".join(tools))
        
        return prompt
    
    def _get_default_capabilities(self, agent_type: str) -> List[AgentCapability]:
        """Get default capabilities for agent type."""
        capability_map = {
            "react": [AgentCapability.REASONING, AgentCapability.TOOL_USE],
            "knowledge_search": [AgentCapability.REASONING, AgentCapability.TOOL_USE, AgentCapability.MEMORY],
            "rag": [AgentCapability.REASONING, AgentCapability.TOOL_USE, AgentCapability.MEMORY],
            "workflow": [AgentCapability.REASONING, AgentCapability.TOOL_USE, AgentCapability.PLANNING],
            "multimodal": [AgentCapability.REASONING, AgentCapability.TOOL_USE, AgentCapability.VISION, AgentCapability.MULTIMODAL],
            "composite": [AgentCapability.REASONING, AgentCapability.TOOL_USE, AgentCapability.COORDINATION],
            "autonomous": [AgentCapability.REASONING, AgentCapability.TOOL_USE, AgentCapability.MEMORY, 
                          AgentCapability.PLANNING, AgentCapability.LEARNING]
        }
        
        return capability_map.get(agent_type, [AgentCapability.REASONING, AgentCapability.TOOL_USE])
    
    def _get_default_tools(self, agent_type: str) -> List[str]:
        """Get default tools for agent type."""
        return self.config_manager.get(f"tool_defaults.{agent_type}", [])
    
    def _get_provider_type(self, provider_name: str) -> ProviderType:
        """Convert provider name to ProviderType enum."""
        provider_map = {
            "ollama": ProviderType.OLLAMA,
            "openai": ProviderType.OPENAI,
            "anthropic": ProviderType.ANTHROPIC,
            "google": ProviderType.GOOGLE,
        }
        
        return provider_map.get(provider_name.lower(), ProviderType.OLLAMA)
    
    def _get_memory_type(self, memory_type_str: str) -> MemoryType:
        """Convert memory type string to MemoryType enum."""
        memory_map = {
            "none": MemoryType.NONE,
            "simple": MemoryType.SIMPLE,
            "advanced": MemoryType.ADVANCED,
            "auto": MemoryType.AUTO,
        }
        
        return memory_map.get(memory_type_str.lower(), MemoryType.SIMPLE)
    
    def get_validation_constraints(self) -> Dict[str, Any]:
        """Get validation constraints for agent parameters."""
        performance = self.config_manager.get_performance_limits()
        security = self.config_manager.get_security_limits()
        
        return {
            "max_execution_time": performance["max_execution_time"],
            "max_iterations": performance["max_iterations"],
            "max_memory_mb": performance["max_memory_mb"],
            "decision_threshold_range": (0.1, 0.9),
            "temperature_range": (0.0, 2.0),
            "max_file_size_mb": security["max_file_size_mb"],
            "rate_limit_per_minute": security["requests_per_minute"],
        }
    
    def validate_agent_config(self, config: AgentConfig) -> List[str]:
        """
        Validate agent configuration against constraints.
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        constraints = self.get_validation_constraints()
        
        # Validate timeout
        if config.timeout_seconds > constraints["max_execution_time"]:
            errors.append(f"Timeout {config.timeout_seconds}s exceeds maximum {constraints['max_execution_time']}s")
        
        # Validate iterations
        if config.max_iterations > constraints["max_iterations"]:
            errors.append(f"Max iterations {config.max_iterations} exceeds maximum {constraints['max_iterations']}")
        
        # Validate temperature
        temp_min, temp_max = constraints["temperature_range"]
        if not (temp_min <= config.temperature <= temp_max):
            errors.append(f"Temperature {config.temperature} must be between {temp_min} and {temp_max}")
        
        return errors


# Global configuration integration instance
_config_integration: Optional[ConfigIntegration] = None


def get_config_integration() -> ConfigIntegration:
    """Get the global configuration integration instance."""
    global _config_integration
    if _config_integration is None:
        _config_integration = ConfigIntegration()
    return _config_integration


def initialize_config_integration(config_manager: Optional[AgentConfigurationManager] = None) -> ConfigIntegration:
    """Initialize the global configuration integration."""
    global _config_integration
    _config_integration = ConfigIntegration(config_manager)
    return _config_integration
