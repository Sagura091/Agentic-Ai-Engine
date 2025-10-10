"""
Agent Configuration Integration Module.

This module provides integration functions to replace hardcoded values
throughout the agent system with configuration-driven values.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

from app.config.agent_config_manager import get_agent_config_manager, AgentConfigurationManager
from app.agents.base.agent import AgentConfig, AgentCapability
from app.agents.factory import AgentBuilderConfig, AgentType, MemoryType
from app.llm.models import LLMConfig, ProviderType

_backend_logger = get_backend_logger()


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
        _backend_logger.info(
            "Configuration integration initialized",
            LogCategory.AGENT_OPERATIONS,
            "app.config.agent_config_integration"
        )
    
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
        
        _backend_logger.info(
            "Created agent config from configuration system",
            LogCategory.AGENT_OPERATIONS,
            "app.config.agent_config_integration",
            data={
                "name": name,
                "agent_type": agent_type,
                "model": config.model_name,
                "provider": config.model_provider
            }
        )

        return config

    def create_builder_config_from_yaml(
        self,
        agent_id: str,
        **overrides
    ) -> AgentBuilderConfig:
        """
        Create an AgentBuilderConfig from individual agent YAML configuration.

        Args:
            agent_id: Unique agent identifier
            **overrides: Configuration overrides

        Returns:
            Configured AgentBuilderConfig instance
        """
        try:
            # Load individual agent configuration
            agent_config = self.config_manager.get_individual_agent_config(agent_id)
            if not agent_config:
                raise ValueError(f"No YAML configuration found for agent: {agent_id}")

            # Apply overrides
            if overrides:
                agent_config = self._deep_merge_configs(agent_config, overrides)

            # Extract basic configuration
            name = agent_config.get("name", f"Agent {agent_id}")
            description = agent_config.get("description", f"Agent created from YAML configuration")

            # Extract agent type and framework
            agent_type_str = agent_config.get("agent_type", "react")
            framework = agent_config.get("framework", "basic")

            # Convert agent type string to enum
            try:
                agent_type = AgentType(agent_type_str.lower())
            except ValueError:
                _backend_logger.warn(
                    f"Unknown agent type '{agent_type_str}', defaulting to REACT",
                    LogCategory.AGENT_OPERATIONS,
                    "app.config.agent_config_integration"
                )
                agent_type = AgentType.REACT

            # Extract LLM configuration
            llm_config_data = agent_config.get("llm_config", {})
            llm_config = self._create_llm_config_from_yaml(llm_config_data)

            # Extract capabilities
            capabilities = self._extract_capabilities(agent_config)

            # Extract tools from use_cases
            tools = self._extract_tools_from_use_cases(agent_config)

            # Extract memory configuration
            memory_config = agent_config.get("memory_config", {})
            memory_type = self._extract_memory_type(memory_config)
            enable_memory = memory_config.get("enable_short_term", True) or memory_config.get("enable_long_term", True)

            # Extract system prompt and personality
            system_prompt = self._build_system_prompt(agent_config)

            # Extract performance settings
            performance_config = agent_config.get("performance", {})
            timeout_seconds = performance_config.get("timeout_seconds", 300)
            max_iterations = performance_config.get("max_iterations", 50)

            # Extract autonomy settings for autonomous agents
            autonomy_config = self._extract_autonomy_config(agent_config)

            # Create AgentBuilderConfig
            builder_config = AgentBuilderConfig(
                name=name,
                description=description,
                agent_type=agent_type,
                llm_config=llm_config,
                capabilities=capabilities,
                tools=tools,
                system_prompt=system_prompt,
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations,
                enable_memory=enable_memory,
                memory_type=memory_type,
                enable_learning=agent_config.get("enable_learning", False),
                custom_config=autonomy_config
            )

            _backend_logger.info(
                f"Created AgentBuilderConfig from YAML for agent: {agent_id}",
                LogCategory.AGENT_OPERATIONS,
                "app.config.agent_config_integration"
            )
            return builder_config

        except Exception as e:
            _backend_logger.error(
                f"Failed to create builder config from YAML for agent {agent_id}: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "app.config.agent_config_integration"
            )
            raise

    def _create_llm_config_from_yaml(self, llm_config_data: Dict[str, Any]) -> LLMConfig:
        """Create LLMConfig from YAML data."""
        try:
            provider_str = llm_config_data.get("provider", "ollama")
            provider = ProviderType(provider_str.upper())
        except ValueError:
            _backend_logger.warn(
                f"Unknown provider '{provider_str}', defaulting to OLLAMA",
                LogCategory.AGENT_OPERATIONS,
                "app.config.agent_config_integration"
            )
            provider = ProviderType.OLLAMA

        return LLMConfig(
            provider=provider,
            model_id=llm_config_data.get("model_id", "llama3.2:latest"),
            model_name=llm_config_data.get("model_name", "Default Model"),
            temperature=llm_config_data.get("temperature", 0.7),
            max_tokens=llm_config_data.get("max_tokens", 2048),
            top_p=llm_config_data.get("top_p", 0.95),
            frequency_penalty=llm_config_data.get("frequency_penalty", 0.0),
            presence_penalty=llm_config_data.get("presence_penalty", 0.0),
            timeout_seconds=llm_config_data.get("timeout_seconds", 300),
            max_retries=llm_config_data.get("max_retries", 3),
            retry_delay_seconds=llm_config_data.get("retry_delay_seconds", 2),
            manual_selection=llm_config_data.get("manual_selection", False)
        )

    def _extract_capabilities(self, agent_config: Dict[str, Any]) -> List[AgentCapability]:
        """Extract capabilities from agent configuration."""
        capabilities = []

        # Add capabilities based on agent type
        agent_type = agent_config.get("agent_type", "react")
        if agent_type in ["react", "autonomous"]:
            capabilities.extend([AgentCapability.REASONING, AgentCapability.TOOL_USE])

        # Add memory capability if memory is enabled
        memory_config = agent_config.get("memory_config", {})
        if any(memory_config.get(key, False) for key in ["enable_short_term", "enable_long_term", "enable_episodic"]):
            capabilities.append(AgentCapability.MEMORY)

        # Add learning capability if enabled
        if agent_config.get("enable_learning", False):
            capabilities.append(AgentCapability.LEARNING)

        # Add planning capability for autonomous agents
        if agent_type == "autonomous":
            capabilities.append(AgentCapability.PLANNING)

        # Add collaboration capability if enabled
        if agent_config.get("enable_collaboration", False):
            capabilities.append(AgentCapability.COLLABORATION)

        return capabilities

    def _extract_tools_from_use_cases(self, agent_config: Dict[str, Any]) -> List[str]:
        """Extract tools based on use cases OR direct tools specification."""
        # First check for direct tools specification (preferred method)
        direct_tools = agent_config.get("tools", [])
        if direct_tools:
            _backend_logger.info(
                f"Found direct tools specification: {direct_tools}",
                LogCategory.AGENT_OPERATIONS,
                "app.config.agent_config_integration"
            )
            return direct_tools

        # Fall back to use_cases mapping for backward compatibility
        use_cases = agent_config.get("use_cases", [])
        if not use_cases:
            _backend_logger.warn(
                "No tools or use_cases found in agent configuration",
                LogCategory.AGENT_OPERATIONS,
                "app.config.agent_config_integration"
            )
            return []

        # Map use cases to tools (this would be enhanced with actual tool repository integration)
        tool_mapping = {
            "business_analysis": ["business_intelligence", "revolutionary_web_scraper"],
            "document_generation": ["revolutionary_document_intelligence"],
            "excel_processing": ["revolutionary_document_intelligence", "business_intelligence"],
            "financial_analysis": ["business_intelligence", "calculator"],
            "data_generation": ["business_intelligence"],
            "web_research": ["revolutionary_web_scraper", "web_search"],
            "text_processing": ["text_processing_nlp"],
            "social_media": ["social_media_orchestrator"],
            "music_composition": ["ai_music_composition"],
            "image_analysis": ["screenshot_analysis", "image_processing"],
            "file_operations": ["file_system_operations"],
            "api_integration": ["api_integration_tool"]
        }

        tools = set()
        for use_case in use_cases:
            if use_case in tool_mapping:
                tools.update(tool_mapping[use_case])

        _backend_logger.info(
            f"Extracted tools from use_cases {use_cases}: {list(tools)}",
            LogCategory.AGENT_OPERATIONS,
            "app.config.agent_config_integration"
        )
        return list(tools)

    def _extract_memory_type(self, memory_config: Dict[str, Any]) -> MemoryType:
        """Extract memory type from configuration."""
        memory_type_str = memory_config.get("memory_type", "simple")

        try:
            return MemoryType(memory_type_str.upper())
        except ValueError:
            _backend_logger.warn(
                f"Unknown memory type '{memory_type_str}', defaulting to SIMPLE",
                LogCategory.AGENT_OPERATIONS,
                "app.config.agent_config_integration"
            )
            return MemoryType.SIMPLE

    def _build_system_prompt(self, agent_config: Dict[str, Any]) -> str:
        """Build system prompt from configuration."""
        # Get base system prompt
        base_prompt = agent_config.get("system_prompt", "")

        if base_prompt:
            return base_prompt

        # Build system prompt from personality and expertise
        personality = agent_config.get("personality", {})
        name = agent_config.get("name", "AI Agent")
        description = agent_config.get("description", "An AI assistant")

        expertise_areas = personality.get("expertise_areas", [])
        communication_style = personality.get("communication_style", "professional")
        creativity_level = personality.get("creativity_level", "balanced")

        prompt_parts = [
            f"You are {name}, {description}.",
            "",
            "Your capabilities include:"
        ]

        if expertise_areas:
            for area in expertise_areas:
                prompt_parts.append(f"- Expert knowledge in {area}")

        prompt_parts.extend([
            "",
            f"Communication style: {communication_style}",
            f"Creativity level: {creativity_level}",
            "",
            "Always provide helpful, accurate, and relevant responses while maintaining your specified personality and expertise."
        ])

        return "\n".join(prompt_parts)

    def _extract_autonomy_config(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract autonomy configuration for autonomous agents."""
        autonomy_config = {}

        # Extract autonomy level
        autonomy_level = agent_config.get("autonomy_level", "reactive")
        autonomy_config["autonomy_level"] = autonomy_level

        # Extract decision settings
        autonomy_config["decision_threshold"] = agent_config.get("decision_threshold", 0.6)
        autonomy_config["decision_confidence"] = agent_config.get("decision_confidence", "medium")

        # Extract learning settings
        autonomy_config["learning_mode"] = agent_config.get("learning_mode", "passive")

        # Extract proactive behavior settings
        autonomy_config["enable_proactive_behavior"] = agent_config.get("enable_proactive_behavior", False)
        autonomy_config["enable_goal_setting"] = agent_config.get("enable_goal_setting", False)
        autonomy_config["enable_self_improvement"] = agent_config.get("enable_self_improvement", False)

        # Extract RAG configuration
        rag_config = agent_config.get("rag_config", {})
        autonomy_config["rag_config"] = rag_config

        return autonomy_config

    def _deep_merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value

        return result
        agent_config = self.config_manager.get_individual_agent_config(agent_id)

        # Extract required fields
        name = agent_config.get("name", agent_id)
        description = agent_config.get("description", f"Agent {agent_id}")
        agent_type_str = agent_config.get("agent_type", "autonomous")

        # Convert agent type string to enum
        try:
            from app.agents.factory import AgentType
            agent_type = AgentType(agent_type_str)
        except ValueError:
            _backend_logger.warn(
                f"Unknown agent type '{agent_type_str}', defaulting to AUTONOMOUS",
                LogCategory.AGENT_OPERATIONS,
                "app.config.agent_config_integration"
            )
            agent_type = AgentType.AUTONOMOUS

        # Extract LLM configuration
        llm_config_dict = agent_config.get("llm_config", {})
        provider_name = llm_config_dict.get("provider", "ollama")
        provider_type = self._get_provider_type(provider_name)

        llm_config_obj = LLMConfig(
            provider=provider_type,
            model_id=llm_config_dict.get("model_id", "llama3.2:latest"),
            temperature=llm_config_dict.get("temperature", 0.7),
            max_tokens=llm_config_dict.get("max_tokens", 2048),
            timeout_seconds=llm_config_dict.get("timeout_seconds", 300)
        )

        # Extract capabilities
        capabilities_list = agent_config.get("capabilities", [])
        capabilities = [AgentCapability(cap) for cap in capabilities_list if hasattr(AgentCapability, cap.upper())]

        # Extract tools (use_cases for dynamic tool selection)
        use_cases = agent_config.get("use_cases", [])
        tools = agent_config.get("tools", [])

        # Extract memory configuration
        memory_config_dict = agent_config.get("memory_config", {})
        memory_type_str = memory_config_dict.get("memory_type", "simple")
        memory_type = self._get_memory_type(memory_type_str)

        # Extract execution settings
        execution_config = agent_config.get("execution", {})

        # Create AgentBuilderConfig
        config = AgentBuilderConfig(
            name=name,
            description=description,
            agent_type=agent_type,
            llm_config=llm_config_obj,
            capabilities=capabilities,
            tools=tools,
            system_prompt=agent_config.get("system_prompt", ""),
            max_iterations=execution_config.get("max_iterations", 50),
            timeout_seconds=execution_config.get("timeout_seconds", 300),
            enable_memory=memory_config_dict.get("enable_short_term", True),
            enable_learning=agent_config.get("enable_continuous_learning", False),
            enable_collaboration=agent_config.get("enable_peer_learning", False),
            memory_type=memory_type,
            memory_config=memory_config_dict,
            custom_config={
                "use_cases": use_cases,
                "autonomy_level": agent_config.get("autonomy_level", "adaptive"),
                "decision_threshold": agent_config.get("decision_threshold", 0.6),
                "learning_mode": agent_config.get("learning_mode", "active"),
                "rag_config": agent_config.get("rag_config", {}),
                "personality": agent_config.get("personality", {}),
                "safety_constraints": agent_config.get("safety_constraints", []),
                "ethical_guidelines": agent_config.get("ethical_guidelines", []),
                **overrides.get("custom_config", {})
            }
        )

        _backend_logger.info(
            "Created builder config from individual agent YAML",
            LogCategory.AGENT_OPERATIONS,
            "app.config.agent_config_integration",
            data={
                "agent_id": agent_id,
                "name": name,
                "agent_type": agent_type.value,
                "provider": provider_name,
                "model": llm_config_obj.model_id
            }
        )

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
        
        _backend_logger.info(
            "Created builder config from configuration system",
            LogCategory.AGENT_OPERATIONS,
            "app.config.agent_config_integration",
            data={
                "name": name,
                "agent_type": agent_type.value,
                "provider": provider_name,
                "model": llm_config_obj.model_id
            }
        )

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
