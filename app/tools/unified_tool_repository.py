"""
Unified Tool Repository - THE Tool System for Multi-Agent Architecture.

This is THE ONLY tool repository in the entire application.
All tool operations flow through this unified repository.

CORE ARCHITECTURE:
- Centralized tool registry with agent-specific access
- Dynamic tool assignment based on use cases
- RAG-enabled and non-RAG tools
- Simple, clean, fast operations

DESIGN PRINCIPLES:
- One tool repository to rule them all
- Use case driven tool access
- Agent-specific tool permissions
- No complexity unless absolutely necessary

PHASE 2 ENHANCEMENT:
✅ Integration with UnifiedRAGSystem
✅ Dynamic tool assignment
✅ Agent-specific tool access
✅ Use case based tool selection
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

logger = structlog.get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools in the repository - SIMPLIFIED."""
    RAG_ENABLED = "rag_enabled"       # Tools that use RAG system
    COMPUTATION = "computation"       # Calculator, math tools
    COMMUNICATION = "communication"   # Agent communication tools
    RESEARCH = "research"             # Web search, research tools
    BUSINESS = "business"             # Business analysis tools
    UTILITY = "utility"               # File operations, utilities


class ToolAccessLevel(str, Enum):
    """Access levels for tools - SIMPLIFIED."""
    PUBLIC = "public"                 # Available to all agents
    PRIVATE = "private"               # Agent-specific tools
    CONDITIONAL = "conditional"       # Based on agent configuration


@dataclass
class ToolMetadata:
    """Simple metadata for a tool in the repository - ENHANCED."""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    access_level: ToolAccessLevel
    requires_rag: bool = False           # NEW: Does this tool need RAG access?
    use_cases: Set[str] = field(default_factory=set)  # NEW: Use cases for this tool
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    is_active: bool = True


@dataclass
class AgentToolProfile:
    """Simple tool profile for an agent - ENHANCED."""
    agent_id: str
    assigned_tools: Set[str] = field(default_factory=set)
    usage_stats: Dict[str, int] = field(default_factory=dict)
    rag_enabled: bool = True             # NEW: Does this agent have RAG access?
    allowed_categories: Set[ToolCategory] = field(default_factory=set)  # NEW: Allowed tool categories
    last_updated: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, agent_id: str, rag_enabled: bool = True) -> "AgentToolProfile":
        """Create a new agent tool profile."""
        return cls(
            agent_id=agent_id,
            rag_enabled=rag_enabled,
            allowed_categories=set(ToolCategory)  # Allow all categories by default
        )


class UnifiedToolRepository:
    """
    Unified Tool Repository - THE Tool System.

    SIMPLIFIED ARCHITECTURE:
    - Dynamic tool assignment based on use cases
    - RAG-enabled vs non-RAG tools
    - Agent-specific tool access
    - Use case driven tool selection
    """

    def __init__(self, unified_rag=None, isolation_manager=None):
        """Initialize THE unified tool repository."""
        self.unified_rag = unified_rag
        self.isolation_manager = isolation_manager

        # Tool registry - ENHANCED
        self.tools: Dict[str, BaseTool] = {}                    # tool_id -> tool_instance
        self.tool_metadata: Dict[str, ToolMetadata] = {}        # tool_id -> metadata

        # Agent profiles - ENHANCED
        self.agent_profiles: Dict[str, AgentToolProfile] = {}   # agent_id -> profile

        # Use case mapping - NEW
        self.use_case_tools: Dict[str, Set[str]] = {}           # use_case -> tool_ids

        # Simple stats
        self.stats = {
            "total_tools": 0,
            "total_agents": 0,
            "total_tool_calls": 0,
            "rag_enabled_tools": 0,
            "tools_by_category": {}
        }

        self.is_initialized = False
        logger.info("THE Unified tool repository created")
    
    async def initialize(self) -> None:
        """Initialize the tool repository."""
        try:
            if self.is_initialized:
                return

            # Initialize category stats
            for category in ToolCategory:
                self.stats["tools_by_category"][category.value] = 0

            self.is_initialized = True
            logger.info("Tool repository initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize tool repository: {str(e)}")
            raise

    async def register_tool(
        self,
        tool_instance: BaseTool,
        metadata: ToolMetadata
    ) -> str:
        """
        Register a new tool in the repository - ENHANCED.

        Args:
            tool_instance: Tool instance to register
            metadata: Tool metadata

        Returns:
            Tool ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            tool_id = metadata.tool_id

            if tool_id in self.tools:
                logger.warning(f"Tool {tool_id} already registered")
                return tool_id

            # Register tool
            self.tools[tool_id] = tool_instance
            self.tool_metadata[tool_id] = metadata

            # Update use case mapping - NEW
            for use_case in metadata.use_cases:
                if use_case not in self.use_case_tools:
                    self.use_case_tools[use_case] = set()
                self.use_case_tools[use_case].add(tool_id)

            # Update stats
            self.stats["total_tools"] += 1
            self.stats["tools_by_category"][metadata.category.value] += 1
            if metadata.requires_rag:
                self.stats["rag_enabled_tools"] += 1

            logger.info(f"Registered tool: {tool_id} ({metadata.category.value}, RAG: {metadata.requires_rag})")
            return tool_id

        except Exception as e:
            logger.error(f"Failed to register tool {metadata.tool_id}: {str(e)}")
            raise
    
    async def get_tools_for_use_case(
        self,
        agent_id: str,
        use_cases: List[str],
        include_rag_tools: bool = True
    ) -> List[BaseTool]:
        """
        Get tools for an agent based on use cases - DYNAMIC TOOL SELECTION.

        Args:
            agent_id: Agent requesting tools
            use_cases: List of use cases (e.g., ["knowledge_search", "calculation"])
            include_rag_tools: Whether to include RAG-enabled tools

        Returns:
            List of tools for the agent
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get or create agent profile
            if agent_id not in self.agent_profiles:
                await self.create_agent_profile(agent_id)

            profile = self.agent_profiles[agent_id]
            selected_tools = []

            # Get tools for each use case
            for use_case in use_cases:
                if use_case in self.use_case_tools:
                    for tool_id in self.use_case_tools[use_case]:
                        metadata = self.tool_metadata[tool_id]

                        # Check if agent can access this tool
                        if not self._can_agent_access_tool(agent_id, tool_id):
                            continue

                        # Check RAG requirements
                        if metadata.requires_rag and not include_rag_tools:
                            continue

                        if metadata.requires_rag and not profile.rag_enabled:
                            continue

                        # Add tool if not already added
                        tool = self.tools[tool_id]
                        if tool not in selected_tools:
                            selected_tools.append(tool)

            logger.debug(f"Selected {len(selected_tools)} tools for agent {agent_id} with use cases: {use_cases}")
            return selected_tools

        except Exception as e:
            logger.error(f"Failed to get tools for agent {agent_id}: {str(e)}")
            return []

    def _can_agent_access_tool(self, agent_id: str, tool_id: str) -> bool:
        """Check if an agent can access a specific tool."""
        metadata = self.tool_metadata[tool_id]

        # Public tools are always accessible
        if metadata.access_level == ToolAccessLevel.PUBLIC:
            return True

        # Private tools need explicit assignment
        if metadata.access_level == ToolAccessLevel.PRIVATE:
            profile = self.agent_profiles.get(agent_id)
            return profile and tool_id in profile.assigned_tools

        # Conditional tools based on agent configuration
        if metadata.access_level == ToolAccessLevel.CONDITIONAL:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                return False
            return metadata.category in profile.allowed_categories

        return False

    async def create_agent_profile(
        self,
        agent_id: str,
        rag_enabled: bool = True,
        allowed_categories: Optional[Set[ToolCategory]] = None
    ) -> AgentToolProfile:
        """
        Create a tool profile for an agent - ENHANCED.

        Args:
            agent_id: Agent identifier
            rag_enabled: Whether agent has RAG access
            allowed_categories: Allowed tool categories

        Returns:
            Agent tool profile
        """
        try:
            if agent_id in self.agent_profiles:
                logger.warning(f"Tool profile already exists for agent {agent_id}")
                return self.agent_profiles[agent_id]

            # Create profile
            profile = AgentToolProfile.create(agent_id)
            self.agent_profiles[agent_id] = profile

            # Assign public tools by default
            for tool_id, metadata in self.tool_metadata.items():
                if metadata.access_level == ToolAccessLevel.PUBLIC:
                    profile.assigned_tools.add(tool_id)

            self.stats["total_agents"] += 1

            logger.info(f"Created tool profile for agent {agent_id}")
            return profile

        except Exception as e:
            logger.error(f"Failed to create tool profile for agent {agent_id}: {str(e)}")
            raise

    async def get_agent_tools(self, agent_id: str) -> List[BaseTool]:
        """
        Get all tools available to an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of available tools
        """
        try:
            if agent_id not in self.agent_profiles:
                await self.create_agent_profile(agent_id)

            profile = self.agent_profiles[agent_id]
            available_tools = []

            for tool_id in profile.assigned_tools:
                if tool_id in self.tools:
                    available_tools.append(self.tools[tool_id])

            return available_tools

        except Exception as e:
            logger.error(f"Failed to get tools for agent {agent_id}: {str(e)}")
            return []
    async def assign_tool_to_agent(
        self,
        agent_id: str,
        tool_id: str
    ) -> bool:
        """
        Assign a specific tool to an agent.

        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier

        Returns:
            True if assignment successful
        """
        try:
            if tool_id not in self.tools:
                logger.error(f"Tool {tool_id} not found")
                return False

            if agent_id not in self.agent_profiles:
                await self.create_agent_profile(agent_id)

            profile = self.agent_profiles[agent_id]

            # Assign tool
            profile.assigned_tools.add(tool_id)
            profile.last_updated = datetime.now()

            logger.info(f"Assigned tool {tool_id} to agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to assign tool {tool_id} to agent {agent_id}: {str(e)}")
            return False
    async def record_tool_usage(
        self,
        agent_id: str,
        tool_id: str
    ) -> None:
        """
        Record tool usage for analytics.

        Args:
            agent_id: Agent that used the tool
            tool_id: Tool that was used
        """
        try:
            # Update agent profile
            if agent_id in self.agent_profiles:
                profile = self.agent_profiles[agent_id]
                if tool_id not in profile.usage_stats:
                    profile.usage_stats[tool_id] = 0
                profile.usage_stats[tool_id] += 1

            # Update tool metadata
            if tool_id in self.tool_metadata:
                metadata = self.tool_metadata[tool_id]
                metadata.usage_count += 1

            # Update global stats
            self.stats["total_tool_calls"] += 1

        except Exception as e:
            logger.error(f"Failed to record tool usage: {str(e)}")

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Get a tool by ID."""
        return self.tools.get(tool_id)

    def get_tool_metadata(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get tool metadata by ID."""
        return self.tool_metadata.get(tool_id)

    def get_agent_profile(self, agent_id: str) -> Optional[AgentToolProfile]:
        """Get agent tool profile."""
        return self.agent_profiles.get(agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "tools_count": len(self.tools),
            "agent_profiles_count": len(self.agent_profiles)
        }

