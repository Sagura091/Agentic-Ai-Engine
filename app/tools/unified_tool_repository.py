"""
Unified Tool Repository for Multi-Agent Architecture.

This module provides a centralized tool repository that manages all tools
across the platform with agent-specific access controls and capabilities.

Features:
- Centralized tool registry
- Agent-specific tool access
- Dynamic tool assignment based on capabilities
- Tool usage tracking and analytics
- Performance optimization
- Security and access control
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set, Callable, Type
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import structlog
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from app.rag.core.agent_isolation_manager import AgentIsolationManager, ResourceType
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools in the repository."""
    KNOWLEDGE = "knowledge"           # RAG and knowledge-based tools
    COMPUTATION = "computation"       # Mathematical and analytical tools
    COMMUNICATION = "communication"   # External communication tools
    RESEARCH = "research"             # Web research and data gathering
    BUSINESS = "business"             # Business intelligence and analysis
    CREATIVE = "creative"             # Content creation and design
    SYSTEM = "system"                 # System administration and control
    UTILITY = "utility"               # General utility tools


class ToolAccessLevel(str, Enum):
    """Access levels for tools."""
    PUBLIC = "public"                 # Available to all agents
    RESTRICTED = "restricted"         # Requires specific permissions
    PRIVATE = "private"               # Agent-specific tools
    ADMIN = "admin"                   # Administrative tools only


@dataclass
class ToolCapability:
    """Represents a capability required to use a tool."""
    name: str
    level: int = 1                    # Skill level required (1-10)
    description: str = ""


@dataclass
class ToolMetadata:
    """Metadata for a tool in the repository."""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    access_level: ToolAccessLevel
    required_capabilities: List[ToolCapability] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    author: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    performance_score: float = 1.0
    is_active: bool = True


@dataclass
class AgentToolProfile:
    """Tool profile for an agent."""
    agent_id: str
    capabilities: Dict[str, int] = field(default_factory=dict)  # capability_name -> level
    assigned_tools: Set[str] = field(default_factory=set)       # tool_ids
    restricted_tools: Set[str] = field(default_factory=set)     # tool_ids
    usage_stats: Dict[str, int] = field(default_factory=dict)   # tool_id -> usage_count
    last_updated: datetime = field(default_factory=datetime.utcnow)


class UnifiedToolRepository:
    """
    Unified Tool Repository for Multi-Agent Architecture.
    
    Manages all tools across the platform with intelligent assignment,
    access control, and performance optimization.
    """
    
    def __init__(self, isolation_manager: AgentIsolationManager):
        """Initialize the unified tool repository."""
        self.isolation_manager = isolation_manager
        
        # Tool registry
        self.tools: Dict[str, BaseTool] = {}                    # tool_id -> tool_instance
        self.tool_metadata: Dict[str, ToolMetadata] = {}        # tool_id -> metadata
        self.tool_classes: Dict[str, Type[BaseTool]] = {}       # tool_id -> tool_class
        
        # Agent profiles
        self.agent_profiles: Dict[str, AgentToolProfile] = {}   # agent_id -> profile
        
        # Access control
        self.tool_permissions: Dict[str, Set[str]] = {}         # tool_id -> agent_ids
        
        # Performance tracking
        self.stats = {
            "total_tools": 0,
            "total_agents": 0,
            "total_tool_calls": 0,
            "avg_tool_performance": 0.0,
            "tools_by_category": {}
        }
        
        self.is_initialized = False
        logger.info("Unified tool repository created")
    
    async def initialize(self) -> None:
        """Initialize the tool repository."""
        try:
            if self.is_initialized:
                logger.warning("Tool repository already initialized")
                return
            
            logger.info("Initializing unified tool repository...")
            
            # Register built-in tools
            await self._register_builtin_tools()
            
            # Initialize category stats
            for category in ToolCategory:
                self.stats["tools_by_category"][category.value] = 0
            
            self.is_initialized = True
            logger.info("Unified tool repository initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tool repository: {str(e)}")
            raise
    
    async def register_tool(
        self,
        tool_class: Type[BaseTool],
        metadata: ToolMetadata,
        auto_assign: bool = True
    ) -> str:
        """
        Register a new tool in the repository.
        
        Args:
            tool_class: Tool class to register
            metadata: Tool metadata
            auto_assign: Whether to auto-assign to compatible agents
            
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
            
            # Create tool instance
            tool_instance = tool_class()
            
            # Register tool
            self.tools[tool_id] = tool_instance
            self.tool_metadata[tool_id] = metadata
            self.tool_classes[tool_id] = tool_class
            
            # Initialize permissions
            self.tool_permissions[tool_id] = set()
            
            # Update stats
            self.stats["total_tools"] += 1
            self.stats["tools_by_category"][metadata.category.value] += 1
            
            # Auto-assign to compatible agents if requested
            if auto_assign:
                await self._auto_assign_tool(tool_id)
            
            logger.info(f"Registered tool: {tool_id} ({metadata.category.value})")
            return tool_id
            
        except Exception as e:
            logger.error(f"Failed to register tool {metadata.tool_id}: {str(e)}")
            raise
    
    async def create_agent_profile(
        self,
        agent_id: str,
        capabilities: Optional[Dict[str, int]] = None
    ) -> AgentToolProfile:
        """
        Create a tool profile for an agent.
        
        Args:
            agent_id: Agent identifier
            capabilities: Agent capabilities
            
        Returns:
            Agent tool profile
        """
        try:
            if agent_id in self.agent_profiles:
                logger.warning(f"Tool profile already exists for agent {agent_id}")
                return self.agent_profiles[agent_id]
            
            # Create profile
            profile = AgentToolProfile(
                agent_id=agent_id,
                capabilities=capabilities or {}
            )
            
            self.agent_profiles[agent_id] = profile
            
            # Assign compatible tools
            await self._assign_compatible_tools(agent_id)
            
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
                if tool_id in self.tools and tool_id not in profile.restricted_tools:
                    # Check access permissions
                    if await self._check_tool_access(agent_id, tool_id):
                        available_tools.append(self.tools[tool_id])
            
            return available_tools
            
        except Exception as e:
            logger.error(f"Failed to get tools for agent {agent_id}: {str(e)}")
            raise
    
    async def assign_tool_to_agent(
        self,
        agent_id: str,
        tool_id: str,
        force: bool = False
    ) -> bool:
        """
        Assign a specific tool to an agent.
        
        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier
            force: Force assignment even if capabilities don't match
            
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
            metadata = self.tool_metadata[tool_id]
            
            # Check capabilities unless forced
            if not force and not await self._check_tool_compatibility(agent_id, tool_id):
                logger.warning(f"Agent {agent_id} lacks capabilities for tool {tool_id}")
                return False
            
            # Check access level
            if not await self._check_tool_access(agent_id, tool_id):
                logger.warning(f"Agent {agent_id} lacks access to tool {tool_id}")
                return False
            
            # Assign tool
            profile.assigned_tools.add(tool_id)
            self.tool_permissions[tool_id].add(agent_id)
            profile.last_updated = datetime.utcnow()
            
            logger.info(f"Assigned tool {tool_id} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign tool {tool_id} to agent {agent_id}: {str(e)}")
            return False
    
    async def record_tool_usage(
        self,
        agent_id: str,
        tool_id: str,
        success: bool = True,
        execution_time: float = 0.0
    ) -> None:
        """
        Record tool usage for analytics.
        
        Args:
            agent_id: Agent that used the tool
            tool_id: Tool that was used
            success: Whether the tool call was successful
            execution_time: Tool execution time
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
                
                # Update performance score based on success rate
                if success:
                    metadata.performance_score = min(metadata.performance_score + 0.01, 1.0)
                else:
                    metadata.performance_score = max(metadata.performance_score - 0.05, 0.0)
            
            # Update global stats
            self.stats["total_tool_calls"] += 1
            
            # Update resource usage
            await self.isolation_manager.update_resource_usage(agent_id, "tool_calls", 1)
            
        except Exception as e:
            logger.error(f"Failed to record tool usage: {str(e)}")
    
    async def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        try:
            # Import and register existing tools
            from .calculator_tool import CalculatorTool
            from .web_research_tool import WebResearchTool
            from app.rag.tools.knowledge_tools import KnowledgeSearchTool
            
            # Register calculator tool
            calc_metadata = ToolMetadata(
                tool_id="calculator",
                name="Calculator",
                description="Basic mathematical calculations",
                category=ToolCategory.COMPUTATION,
                access_level=ToolAccessLevel.PUBLIC,
                required_capabilities=[
                    ToolCapability("mathematics", 1, "Basic math skills")
                ],
                tags={"math", "calculation", "utility"}
            )
            await self.register_tool(CalculatorTool, calc_metadata, auto_assign=True)
            
            # Register web research tool
            web_metadata = ToolMetadata(
                tool_id="web_research",
                name="Web Research",
                description="Web search and research capabilities",
                category=ToolCategory.RESEARCH,
                access_level=ToolAccessLevel.RESTRICTED,
                required_capabilities=[
                    ToolCapability("research", 3, "Research and analysis skills")
                ],
                tags={"web", "search", "research"}
            )
            await self.register_tool(WebResearchTool, web_metadata, auto_assign=False)
            
            # Register knowledge search tool
            knowledge_metadata = ToolMetadata(
                tool_id="knowledge_search",
                name="Knowledge Search",
                description="Search knowledge bases and memory",
                category=ToolCategory.KNOWLEDGE,
                access_level=ToolAccessLevel.PUBLIC,
                required_capabilities=[
                    ToolCapability("knowledge_retrieval", 2, "Knowledge retrieval skills")
                ],
                tags={"knowledge", "search", "rag"}
            )
            await self.register_tool(KnowledgeSearchTool, knowledge_metadata, auto_assign=True)
            
        except Exception as e:
            logger.error(f"Failed to register built-in tools: {str(e)}")
    
    async def _auto_assign_tool(self, tool_id: str) -> None:
        """Auto-assign a tool to compatible agents."""
        try:
            for agent_id in self.agent_profiles:
                if await self._check_tool_compatibility(agent_id, tool_id):
                    await self.assign_tool_to_agent(agent_id, tool_id)
                    
        except Exception as e:
            logger.error(f"Failed to auto-assign tool {tool_id}: {str(e)}")
    
    async def _assign_compatible_tools(self, agent_id: str) -> None:
        """Assign all compatible tools to an agent."""
        try:
            for tool_id in self.tools:
                if await self._check_tool_compatibility(agent_id, tool_id):
                    await self.assign_tool_to_agent(agent_id, tool_id)
                    
        except Exception as e:
            logger.error(f"Failed to assign compatible tools to agent {agent_id}: {str(e)}")
    
    async def _check_tool_compatibility(self, agent_id: str, tool_id: str) -> bool:
        """Check if an agent has the capabilities to use a tool."""
        try:
            if agent_id not in self.agent_profiles or tool_id not in self.tool_metadata:
                return False
            
            profile = self.agent_profiles[agent_id]
            metadata = self.tool_metadata[tool_id]
            
            # Check required capabilities
            for capability in metadata.required_capabilities:
                agent_level = profile.capabilities.get(capability.name, 0)
                if agent_level < capability.level:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check tool compatibility: {str(e)}")
            return False
    
    async def _check_tool_access(self, agent_id: str, tool_id: str) -> bool:
        """Check if an agent has access to a tool."""
        try:
            if tool_id not in self.tool_metadata:
                return False
            
            metadata = self.tool_metadata[tool_id]
            
            # Public tools are available to all
            if metadata.access_level == ToolAccessLevel.PUBLIC:
                return True
            
            # Check explicit permissions
            return agent_id in self.tool_permissions.get(tool_id, set())
            
        except Exception as e:
            logger.error(f"Failed to check tool access: {str(e)}")
            return False
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return {
            **self.stats,
            "tools_count": len(self.tools),
            "agent_profiles_count": len(self.agent_profiles)
        }
