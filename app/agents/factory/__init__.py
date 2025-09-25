"""
AI Agent Builder Factory - Core Agent Creation and Management System.

This module provides the foundational agent builder framework that supports
creating multiple agent types with different capabilities and configurations.

SUPPORTED AGENT TYPES:
- ReactAgent: Reasoning and Acting agents with tool use
- KnowledgeSearchAgent: RAG-focused agents for knowledge retrieval
- WorkflowAgent: Custom process automation agents
- MultiModalAgent: Vision + Text + Audio capable agents
- CompositeAgent: Multi-agent coordination systems

DESIGN PRINCIPLES:
- Provider-agnostic LLM support (Ollama, OpenAI, vLLM, etc.)
- Template-based agent creation
- Extensible capability system
- Enterprise-grade configuration management
"""

from typing import Dict, List, Optional, Any, Type, Union
from enum import Enum
from dataclasses import dataclass
import structlog

from app.agents.base.agent import LangGraphAgent, AgentConfig, AgentCapability
from app.agents.autonomous.autonomous_agent import AutonomousLangGraphAgent, AutonomousAgentConfig
from app.llm.models import LLMConfig, ProviderType
from app.llm.manager import LLMProviderManager
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.agents.autonomous.persistent_memory import PersistentMemorySystem

logger = structlog.get_logger(__name__)


class MemoryType(Enum):
    """Types of memory systems available for agents."""
    NONE = "none"           # No memory system
    SIMPLE = "simple"       # Short-term + Long-term memory (UnifiedMemorySystem)
    ADVANCED = "advanced"   # Episodic + Semantic + Procedural + Working memory (PersistentMemorySystem)
    AUTO = "auto"          # Automatically determine based on agent type and capabilities


class AgentType(Enum):
    """Supported agent types in the builder platform."""
    REACT = "react"
    KNOWLEDGE_SEARCH = "knowledge_search"
    RAG = "rag"
    WORKFLOW = "workflow"
    MULTIMODAL = "multimodal"
    COMPOSITE = "composite"
    AUTONOMOUS = "autonomous"


class AgentTemplate(Enum):
    """Pre-built agent templates for common use cases."""
    RESEARCH_ASSISTANT = "research_assistant"
    CUSTOMER_SUPPORT = "customer_support"
    DATA_ANALYST = "data_analyst"
    CONTENT_CREATOR = "content_creator"
    CODE_REVIEWER = "code_reviewer"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    DOCUMENT_PROCESSOR = "document_processor"
    MULTI_AGENT_COORDINATOR = "multi_agent_coordinator"


@dataclass
class AgentBuilderConfig:
    """Configuration for building agents through the platform."""
    
    # Basic agent information (required fields first)
    name: str
    description: str
    agent_type: AgentType

    # LLM configuration (required)
    llm_config: LLMConfig

    # Capabilities and tools (required)
    capabilities: List[AgentCapability]
    tools: List[str]

    # Optional fields with defaults
    template: Optional[AgentTemplate] = None
    system_prompt: Optional[str] = None
    max_iterations: int = 50
    timeout_seconds: int = 300
    enable_memory: bool = True
    enable_learning: bool = False
    enable_collaboration: bool = False
    memory_type: MemoryType = MemoryType.AUTO
    memory_config: Optional[Dict[str, Any]] = None

    # Custom configuration
    custom_config: Optional[Dict[str, Any]] = None


class AgentBuilderFactory:
    """
    Core factory for building different types of AI agents.
    
    This factory provides a unified interface for creating agents with
    different capabilities, configurations, and LLM providers.
    """
    
    def __init__(self, llm_manager: LLMProviderManager, unified_memory_system: Optional[UnifiedMemorySystem] = None):
        self.llm_manager = llm_manager
        self.unified_memory_system = unified_memory_system
        self._agent_builders: Dict[AgentType, callable] = {
            AgentType.REACT: self._build_react_agent,
            AgentType.KNOWLEDGE_SEARCH: self._build_knowledge_search_agent,
            AgentType.RAG: self._build_rag_agent,
            AgentType.WORKFLOW: self._build_workflow_agent,
            AgentType.MULTIMODAL: self._build_multimodal_agent,
            AgentType.COMPOSITE: self._build_composite_agent,
            AgentType.AUTONOMOUS: self._build_autonomous_agent,
        }

        self._templates: Dict[AgentTemplate, AgentBuilderConfig] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize pre-built agent templates."""
        # Research Assistant Template
        self._templates[AgentTemplate.RESEARCH_ASSISTANT] = AgentBuilderConfig(
            name="Research Assistant",
            description="Autonomous research agent with web search and analysis capabilities",
            agent_type=AgentType.AUTONOMOUS,
            llm_config=LLMConfig(provider=ProviderType.OLLAMA, model_id="llama3.2:latest"),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.PLANNING
            ],
            tools=["web_search", "calculator", "document_analyzer"],
            system_prompt="You are an expert research assistant. Conduct thorough research, analyze information critically, and provide comprehensive insights.",
            enable_memory=True,
            enable_learning=True,
            memory_type=MemoryType.ADVANCED  # Advanced memory for autonomous research
        )
        
        # Customer Support Template
        self._templates[AgentTemplate.CUSTOMER_SUPPORT] = AgentBuilderConfig(
            name="Customer Support Agent",
            description="Intelligent customer support agent with knowledge base access",
            agent_type=AgentType.KNOWLEDGE_SEARCH,
            llm_config=LLMConfig(provider=ProviderType.OLLAMA, model_id="llama3.2:latest"),
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY
            ],
            tools=["knowledge_search", "ticket_system", "escalation_manager"],
            system_prompt="You are a helpful customer support agent. Provide accurate, empathetic assistance using the knowledge base.",
            enable_memory=True,
            enable_collaboration=True,
            memory_type=MemoryType.SIMPLE  # Simple memory for customer support
        )
    
    async def build_agent(self, config: AgentBuilderConfig) -> Union[LangGraphAgent, AutonomousLangGraphAgent]:
        """
        Build an agent based on the provided configuration.
        
        Args:
            config: Agent builder configuration
            
        Returns:
            Configured agent instance
        """
        try:
            logger.info("Building agent", agent_type=config.agent_type.value, name=config.name)
            
            # Get the appropriate builder function
            builder_func = self._agent_builders.get(config.agent_type)
            if not builder_func:
                raise ValueError(f"Unsupported agent type: {config.agent_type}")

            # Get optimal LLM configuration (supports manual and automatic selection)
            optimal_llm_config = await self.llm_manager.get_model_for_agent(config)

            # Create LLM instance with optimal configuration
            llm = await self.llm_manager.create_llm_instance(optimal_llm_config)
            
            # Build the agent
            agent = await builder_func(config, llm)

            # Automatic memory assignment
            if config.enable_memory:
                await self._assign_memory_system(agent, config)

            logger.info("Agent built successfully",
                       agent_type=config.agent_type.value,
                       name=config.name,
                       memory_enabled=config.enable_memory,
                       memory_type=config.memory_type.value if config.enable_memory else "none")
            return agent
            
        except Exception as e:
            logger.error("Failed to build agent", error=str(e), config=config)
            raise
    
    async def build_from_template(self, template: AgentTemplate, overrides: Optional[Dict[str, Any]] = None) -> Union[LangGraphAgent, AutonomousLangGraphAgent]:
        """
        Build an agent from a pre-defined template.
        
        Args:
            template: Agent template to use
            overrides: Configuration overrides
            
        Returns:
            Configured agent instance
        """
        if template not in self._templates:
            raise ValueError(f"Unknown template: {template}")
        
        config = self._templates[template]
        
        # Apply overrides if provided
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return await self.build_agent(config)

    def _determine_memory_type(self, config: AgentBuilderConfig) -> MemoryType:
        """
        Determine the appropriate memory type for an agent.

        Priority order:
        1. User explicit selection (config.memory_type != AUTO)
        2. Template-based selection (if template specifies memory type)
        3. Agent type and capabilities-based selection

        Args:
            config: Agent builder configuration

        Returns:
            Determined memory type
        """
        # 1. User explicit selection
        if config.memory_type != MemoryType.AUTO:
            logger.info("Using user-selected memory type",
                       memory_type=config.memory_type.value,
                       agent_name=config.name)
            return config.memory_type

        # 2. Template-based selection (already handled in template initialization)
        # Templates set their preferred memory_type, so AUTO means we need to determine

        # 3. Agent type and capabilities-based selection
        if config.agent_type == AgentType.AUTONOMOUS:
            return MemoryType.ADVANCED

        if AgentCapability.LEARNING in config.capabilities:
            return MemoryType.ADVANCED

        if AgentCapability.MEMORY in config.capabilities:
            return MemoryType.SIMPLE

        # Default to simple memory if memory is enabled
        return MemoryType.SIMPLE if config.enable_memory else MemoryType.NONE

    async def _assign_memory_system(self, agent: Union[LangGraphAgent, AutonomousLangGraphAgent], config: AgentBuilderConfig):
        """
        Assign appropriate memory system to an agent.

        Args:
            agent: The agent instance
            config: Agent builder configuration
        """
        try:
            memory_type = self._determine_memory_type(config)

            if memory_type == MemoryType.NONE:
                logger.info("No memory system assigned", agent_id=agent.agent_id)
                return

            if memory_type == MemoryType.SIMPLE:
                await self._assign_simple_memory(agent, config)
            elif memory_type == MemoryType.ADVANCED:
                await self._assign_advanced_memory(agent, config)

            logger.info("Memory system assigned successfully",
                       agent_id=agent.agent_id,
                       memory_type=memory_type.value,
                       agent_name=config.name)

        except Exception as e:
            logger.error("Failed to assign memory system",
                        agent_id=agent.agent_id,
                        error=str(e))
            # Don't fail agent creation if memory assignment fails

    async def _assign_simple_memory(self, agent: Union[LangGraphAgent, AutonomousLangGraphAgent], config: AgentBuilderConfig):
        """Assign UnifiedMemorySystem (simple memory) to an agent."""
        if not self.unified_memory_system:
            logger.warning("UnifiedMemorySystem not available, skipping simple memory assignment")
            return

        # Create agent memory collection
        memory_collection = await self.unified_memory_system.create_agent_memory(agent.agent_id)

        # Store reference in agent for easy access
        agent.memory_system = self.unified_memory_system
        agent.memory_collection = memory_collection
        agent.memory_type = "simple"

        logger.info("Simple memory system assigned",
                   agent_id=agent.agent_id,
                   collection_id=memory_collection.agent_id)

    async def _assign_advanced_memory(self, agent: Union[LangGraphAgent, AutonomousLangGraphAgent], config: AgentBuilderConfig):
        """Assign PersistentMemorySystem (advanced memory) to an agent."""
        # For autonomous agents, they already have PersistentMemorySystem
        if isinstance(agent, AutonomousLangGraphAgent):
            # Ensure the autonomous agent's memory system is properly initialized
            if hasattr(agent, 'memory_system') and agent.memory_system:
                # Wait for the autonomous components to initialize (they run async)
                import asyncio
                await asyncio.sleep(0.1)  # Give time for async initialization

                # Verify memory system is initialized
                if not agent.memory_system.is_initialized:
                    logger.info("Initializing autonomous agent's memory system",
                               agent_id=agent.agent_id)
                    await agent.memory_system.initialize()

                agent.memory_type = "advanced"
                logger.info("Autonomous agent memory system verified and ready",
                           agent_id=agent.agent_id,
                           episodic_count=len(agent.memory_system.episodic_memory),
                           semantic_count=len(agent.memory_system.semantic_memory))
            else:
                logger.warning("Autonomous agent missing memory system", agent_id=agent.agent_id)
            return

        # For regular agents, create and assign PersistentMemorySystem
        memory_config = config.memory_config or {}

        persistent_memory = PersistentMemorySystem(
            agent_id=agent.agent_id,
            llm=agent.llm,
            max_working_memory=memory_config.get("max_working_memory", 20),
            max_episodic_memory=memory_config.get("max_episodic_memory", 10000),
            max_semantic_memory=memory_config.get("max_semantic_memory", 5000),
            consolidation_threshold=memory_config.get("consolidation_threshold", 5)
        )

        # Initialize the memory system
        await persistent_memory.initialize()

        # Store reference in agent
        agent.memory_system = persistent_memory
        agent.memory_type = "advanced"

        logger.info("Advanced memory system assigned",
                   agent_id=agent.agent_id,
                   episodic_count=len(persistent_memory.episodic_memory),
                   semantic_count=len(persistent_memory.semantic_memory))

    def get_available_templates(self) -> List[AgentTemplate]:
        """Get list of available agent templates."""
        return list(self._templates.keys())
    
    def get_template_config(self, template: AgentTemplate) -> AgentBuilderConfig:
        """Get configuration for a specific template."""
        if template not in self._templates:
            raise ValueError(f"Unknown template: {template}")
        return self._templates[template]

    # Agent Builder Methods
    async def _build_react_agent(self, config: AgentBuilderConfig, llm) -> LangGraphAgent:
        """Build a React (Reasoning + Acting) agent."""
        agent_config = AgentConfig(
            name=config.name,
            description=config.description,
            agent_type="react",
            framework="react",
            system_prompt=config.system_prompt or "You are a reasoning and acting agent. Think step by step and use tools when needed.",
            capabilities=config.capabilities,
            tools=config.tools,
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            model_name=config.llm_config.model_id,
            model_provider=config.llm_config.provider.value
        )
        # Get tools from the unified tool repository
        tools = await self._get_agent_tools(config.tools)
        return LangGraphAgent(config=agent_config, llm=llm, tools=tools)

    async def _build_knowledge_search_agent(self, config: AgentBuilderConfig, llm) -> LangGraphAgent:
        """Build a knowledge search agent focused on RAG operations."""
        agent_config = AgentConfig(
            name=config.name,
            description=config.description,
            agent_type="knowledge_search",
            framework="basic",
            system_prompt=config.system_prompt or "You are a knowledge search agent. Use RAG tools to find and synthesize information.",
            capabilities=config.capabilities,
            tools=config.tools + ["knowledge_search", "document_retrieval"],
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            model_name=config.llm_config.model_id,
            model_provider=config.llm_config.provider.value
        )
        # Get tools from the unified tool repository
        tools = await self._get_agent_tools(config.tools + ["knowledge_search", "document_retrieval"])
        return LangGraphAgent(config=agent_config, llm=llm, tools=tools)

    async def _build_rag_agent(self, config: AgentBuilderConfig, llm) -> LangGraphAgent:
        """Build a RAG (Retrieval-Augmented Generation) agent."""
        agent_config = AgentConfig(
            name=config.name,
            description=config.description,
            agent_type="rag",
            framework="basic",
            system_prompt=config.system_prompt or "You are a RAG agent. Retrieve relevant information and generate comprehensive responses.",
            capabilities=config.capabilities,
            tools=config.tools + ["rag_search", "document_analysis", "knowledge_synthesis"],
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            model_name=config.llm_config.model_id,
            model_provider=config.llm_config.provider.value
        )
        # Get tools from the unified tool repository
        tools = await self._get_agent_tools(config.tools + ["rag_search", "document_analysis", "knowledge_synthesis"])
        return LangGraphAgent(config=agent_config, llm=llm, tools=tools)

    async def _build_workflow_agent(self, config: AgentBuilderConfig, llm) -> LangGraphAgent:
        """Build a workflow automation agent."""
        agent_config = AgentConfig(
            name=config.name,
            description=config.description,
            agent_type="workflow",
            framework="basic",
            system_prompt=config.system_prompt or "You are a workflow automation agent. Execute processes step by step efficiently.",
            capabilities=config.capabilities,
            tools=config.tools + ["workflow_executor", "task_manager"],
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            model_name=config.llm_config.model_id,
            model_provider=config.llm_config.provider.value
        )
        # Get tools from the unified tool repository
        tools = await self._get_agent_tools(config.tools + ["workflow_executor", "task_manager"])
        return LangGraphAgent(config=agent_config, llm=llm, tools=tools)

    async def _build_multimodal_agent(self, config: AgentBuilderConfig, llm) -> LangGraphAgent:
        """Build a multi-modal agent (text, vision, audio)."""
        agent_config = AgentConfig(
            name=config.name,
            description=config.description,
            agent_type="multimodal",
            framework="basic",
            system_prompt=config.system_prompt or "You are a multi-modal agent. Process text, images, and audio inputs effectively.",
            capabilities=config.capabilities + [AgentCapability.VISION, AgentCapability.AUDIO],
            tools=config.tools + ["image_analysis", "audio_processing", "multimodal_synthesis"],
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            model_name=config.llm_config.model_id,
            model_provider=config.llm_config.provider.value
        )
        # Get tools from the unified tool repository
        tools = await self._get_agent_tools(config.tools + ["image_analysis", "audio_processing", "multimodal_synthesis"])
        return LangGraphAgent(config=agent_config, llm=llm, tools=tools)

    async def _build_composite_agent(self, config: AgentBuilderConfig, llm) -> LangGraphAgent:
        """Build a composite agent for multi-agent coordination."""
        agent_config = AgentConfig(
            name=config.name,
            description=config.description,
            agent_type="composite",
            framework="basic",
            system_prompt=config.system_prompt or "You are a composite agent coordinator. Manage and coordinate multiple sub-agents.",
            capabilities=config.capabilities + [AgentCapability.COORDINATION],
            tools=config.tools + ["agent_coordinator", "task_distributor", "result_aggregator"],
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            model_name=config.llm_config.model_id,
            model_provider=config.llm_config.provider.value
        )
        # Get tools from the unified tool repository
        tools = await self._get_agent_tools(config.tools + ["agent_coordinator", "task_distributor", "result_aggregator"])
        return LangGraphAgent(config=agent_config, llm=llm, tools=tools)

    async def _build_autonomous_agent(self, config: AgentBuilderConfig, llm) -> AutonomousLangGraphAgent:
        """Build an autonomous agent with BDI architecture."""
        from app.agents.autonomous import AutonomyLevel, LearningMode

        autonomous_config = AutonomousAgentConfig(
            name=config.name,
            description=config.description,
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            learning_mode=LearningMode.ACTIVE if config.enable_learning else LearningMode.PASSIVE,
            capabilities=config.capabilities,
            enable_proactive_behavior=True,
            enable_goal_setting=True,
            enable_self_modification=config.enable_learning,
            safety_constraints=["verify_actions", "respect_boundaries", "maintain_ethics"]
        )
        # Get tools from the unified tool repository
        tools = await self._get_agent_tools(config.tools)
        return AutonomousLangGraphAgent(config=autonomous_config, llm=llm, tools=tools)

    async def _get_agent_tools(self, tool_names: List[str]) -> List:
        """
        Get actual tool instances from the unified tool repository.

        Args:
            tool_names: List of tool names/IDs to retrieve

        Returns:
            List of tool instances
        """
        tools = []
        try:
            # Get the unified system orchestrator to access tool repository
            from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
            orchestrator = get_enhanced_system_orchestrator()

            logger.info(f"🔍 Looking for tools: {tool_names}")
            logger.info(f"🔧 Orchestrator available: {orchestrator is not None}")
            logger.info(f"🛠️ Tool repository available: {orchestrator.tool_repository is not None if orchestrator else False}")

            if orchestrator and orchestrator.tool_repository:
                available_tools = list(orchestrator.tool_repository.tools.keys())
                logger.info(f"📋 Available tools in repository: {available_tools}")

                for tool_name in tool_names:
                    # Try to get tool by ID
                    if tool_name in orchestrator.tool_repository.tools:
                        tool_instance = orchestrator.tool_repository.tools[tool_name]
                        tools.append(tool_instance)
                        logger.info(f"✅ Found tool: {tool_name}")
                    else:
                        logger.warning(f"⚠️ Tool not found: {tool_name}")
            else:
                logger.warning("Tool repository not available")

        except Exception as e:
            logger.error(f"Failed to get agent tools: {e}")

        logger.info(f"Retrieved {len(tools)} tools for agent: {[t.name for t in tools]}")
        return tools


__all__ = [
    "AgentType",
    "AgentTemplate",
    "AgentBuilderConfig",
    "AgentBuilderFactory"
]
