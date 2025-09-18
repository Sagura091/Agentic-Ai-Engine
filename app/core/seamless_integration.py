"""
Seamless Integration Module for Revolutionary Agentic AI System.

This module ensures all components work together seamlessly, providing
unlimited agent creation, dynamic tool management, and true agentic AI
capabilities that integrate perfectly with the existing codebase.
"""

import asyncio
import sys
from typing import Dict, Any, List, Optional

import structlog
from fastapi import FastAPI

from app.config.settings import get_settings
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator
from app.tools.production_tool_system import production_tool_registry
from app.agents.autonomous import (
    AutonomousLangGraphAgent,
    create_autonomous_agent,
    create_research_agent,
    create_creative_agent,
    create_optimization_agent
)

logger = structlog.get_logger(__name__)


class SeamlessIntegrationManager:
    """
    Manager for seamless integration of all system components.
    
    This manager ensures that:
    - Enhanced orchestrator replaces the original seamlessly
    - All existing APIs continue to work
    - New capabilities are available throughout the system
    - Tool system integrates with all agents
    - Performance is optimized across all components
    """
    
    def __init__(self):
        """Initialize the seamless integration manager."""
        self.settings = get_settings()
        self.is_initialized = False
        self.integration_status = {}
        
        logger.info("Seamless integration manager created")
    
    async def initialize_complete_system(self) -> None:
        """
        Initialize the complete integrated system.
        
        This method ensures all components are properly initialized
        and integrated with each other.
        """
        try:
            logger.info("Starting complete system initialization")
            
            # Step 1: Initialize enhanced orchestrator
            await self._initialize_enhanced_orchestrator()
            
            # Step 2: Initialize production tool system
            await self._initialize_production_tools()
            
            # Step 3: Integrate orchestrator with tool system
            await self._integrate_orchestrator_tools()
            
            # Step 4: Replace global orchestrator references
            await self._replace_global_orchestrator()
            
            # Step 5: Initialize autonomous agents
            await self._initialize_autonomous_agents()
            
            # Step 6: Verify all integrations
            await self._verify_integrations()
            
            self.is_initialized = True
            logger.info("Complete system initialization successful")
            
        except Exception as e:
            logger.error("Complete system initialization failed", error=str(e))
            raise
    
    async def _initialize_enhanced_orchestrator(self) -> None:
        """Initialize the enhanced orchestrator."""
        try:
            await enhanced_orchestrator.initialize()
            self.integration_status["enhanced_orchestrator"] = "initialized"
            logger.info("Enhanced orchestrator initialized")
        except Exception as e:
            logger.error("Enhanced orchestrator initialization failed", error=str(e))
            raise
    
    async def _initialize_production_tools(self) -> None:
        """Initialize the production tool system."""
        try:
            # Initialize with orchestrator integration
            await production_tool_registry.initialize_with_orchestrator(enhanced_orchestrator)
            self.integration_status["production_tools"] = "initialized"
            logger.info("Production tool system initialized")
        except Exception as e:
            logger.error("Production tool system initialization failed", error=str(e))
            raise
    
    async def _integrate_orchestrator_tools(self) -> None:
        """Integrate orchestrator with tool system."""
        try:
            # Ensure all tools are available to all agents
            for agent_id, agent in enhanced_orchestrator.agents.items():
                # Add all production tools to each agent
                for tool_name, tool in production_tool_registry.registered_tools.items():
                    if tool_name not in agent.tools:
                        await agent.add_tool(tool)
            
            self.integration_status["orchestrator_tools"] = "integrated"
            logger.info("Orchestrator-tools integration completed")
        except Exception as e:
            logger.error("Orchestrator-tools integration failed", error=str(e))
            raise
    
    async def _replace_global_orchestrator(self) -> None:
        """Replace global orchestrator references throughout the system."""
        try:
            # Replace in orchestrator module
            import app.orchestration.orchestrator as orchestrator_module
            orchestrator_module.orchestrator = enhanced_orchestrator
            
            # Replace in API endpoints
            import app.api.v1.endpoints.agents as agents_module
            import app.api.v1.endpoints.standalone as standalone_module
            import app.api.v1.endpoints.workflows as workflows_module
            
            # Update orchestrator references
            agents_module.orchestrator = enhanced_orchestrator
            standalone_module.orchestrator = enhanced_orchestrator
            if hasattr(workflows_module, 'orchestrator'):
                workflows_module.orchestrator = enhanced_orchestrator
            
            # Update dependencies
            from app.core.dependencies import get_orchestrator
            
            # Create a new dependency function that returns enhanced orchestrator
            async def get_enhanced_orchestrator():
                return enhanced_orchestrator
            
            # Replace the dependency
            import app.core.dependencies as deps_module
            deps_module.get_orchestrator = get_enhanced_orchestrator
            
            self.integration_status["global_orchestrator"] = "replaced"
            logger.info("Global orchestrator references replaced")
        except Exception as e:
            logger.error("Global orchestrator replacement failed", error=str(e))
            raise
    
    async def _initialize_autonomous_agents(self) -> None:
        """Initialize autonomous agents for immediate use."""
        try:
            # Create a set of default autonomous agents
            default_agents = [
                {
                    "type": "research",
                    "name": "Research Assistant",
                    "description": "Autonomous research agent for information gathering and analysis"
                },
                {
                    "type": "creative",
                    "name": "Creative Problem Solver",
                    "description": "Creative autonomous agent for innovative solutions"
                },
                {
                    "type": "optimization",
                    "name": "System Optimizer",
                    "description": "Optimization agent for performance enhancement"
                }
            ]
            
            created_agents = []
            for agent_spec in default_agents:
                try:
                    if agent_spec["type"] == "research":
                        agent = create_research_agent(
                            llm=enhanced_orchestrator.llm,
                            tools=list(production_tool_registry.registered_tools.values())
                        )
                    elif agent_spec["type"] == "creative":
                        agent = create_creative_agent(
                            llm=enhanced_orchestrator.llm,
                            tools=list(production_tool_registry.registered_tools.values())
                        )
                    elif agent_spec["type"] == "optimization":
                        agent = create_optimization_agent(
                            llm=enhanced_orchestrator.llm,
                            tools=list(production_tool_registry.registered_tools.values())
                        )
                    
                    # Register with enhanced orchestrator
                    agent_id = f"default_{agent_spec['type']}_agent"
                    enhanced_orchestrator.agents[agent_id] = agent
                    enhanced_orchestrator.agent_registry[agent_id] = {
                        "type": agent_spec["type"],
                        "name": agent_spec["name"],
                        "description": agent_spec["description"],
                        "config": {},
                        "created_at": asyncio.get_event_loop().time(),
                        "status": "active"
                    }
                    
                    created_agents.append(agent_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to create default {agent_spec['type']} agent", error=str(e))
            
            self.integration_status["autonomous_agents"] = f"initialized_{len(created_agents)}"
            logger.info(f"Autonomous agents initialized: {len(created_agents)} agents")
        except Exception as e:
            logger.error("Autonomous agents initialization failed", error=str(e))
            raise
    
    async def _verify_integrations(self) -> None:
        """Verify all integrations are working correctly."""
        try:
            verification_results = {}
            
            # Verify orchestrator
            verification_results["orchestrator"] = {
                "initialized": enhanced_orchestrator.is_initialized,
                "agents_count": len(enhanced_orchestrator.agents),
                "tools_count": len(enhanced_orchestrator.global_tools)
            }
            
            # Verify tool system
            verification_results["tool_system"] = {
                "registered_tools": len(production_tool_registry.registered_tools),
                "factory_tools": len(production_tool_registry.tool_factory.registered_tools),
                "integration": production_tool_registry.orchestrator_integration
            }
            
            # Verify agent capabilities
            agent_capabilities = {}
            for agent_id, agent in enhanced_orchestrator.agents.items():
                agent_capabilities[agent_id] = {
                    "tools_count": len(agent.tools),
                    "is_autonomous": isinstance(agent, AutonomousLangGraphAgent),
                    "capabilities": getattr(agent, 'capabilities', [])
                }
            
            verification_results["agents"] = agent_capabilities
            
            # Test basic functionality
            if enhanced_orchestrator.agents:
                test_agent_id = list(enhanced_orchestrator.agents.keys())[0]
                try:
                    test_result = await enhanced_orchestrator.execute_agent_task(
                        agent_id=test_agent_id,
                        task="Test integration - respond with 'Integration successful'",
                        context={"test": True}
                    )
                    verification_results["functionality_test"] = {
                        "status": "success",
                        "result": test_result.get("status", "unknown")
                    }
                except Exception as e:
                    verification_results["functionality_test"] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            self.integration_status["verification"] = verification_results
            logger.info("Integration verification completed", results=verification_results)
            
        except Exception as e:
            logger.error("Integration verification failed", error=str(e))
            raise
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            "is_initialized": self.is_initialized,
            "integration_status": self.integration_status,
            "system_metrics": {
                "orchestrator": {
                    "agents": len(enhanced_orchestrator.agents),
                    "tools": len(enhanced_orchestrator.global_tools),
                    "workflows": len(enhanced_orchestrator.workflows),
                    "execution_metrics": enhanced_orchestrator.execution_metrics
                },
                "tool_system": production_tool_registry.get_system_metrics(),
                "autonomous_agents": len([
                    agent for agent in enhanced_orchestrator.agents.values()
                    if isinstance(agent, AutonomousLangGraphAgent)
                ])
            }
        }
    
    async def create_unlimited_agent(
        self,
        agent_type: str,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None
    ) -> str:
        """
        Create unlimited agents with full integration.
        
        Args:
            agent_type: Type of agent
            name: Agent name
            description: Agent description
            config: Optional configuration
            tools: Optional tool names
            
        Returns:
            Agent ID
        """
        from app.orchestration.enhanced_orchestrator import AgentType
        
        # Map string to enum
        type_mapping = {
            "basic": AgentType.BASIC,
            "autonomous": AgentType.AUTONOMOUS,
            "research": AgentType.RESEARCH,
            "creative": AgentType.CREATIVE,
            "optimization": AgentType.OPTIMIZATION,
            "custom": AgentType.CUSTOM
        }
        
        agent_type_enum = type_mapping.get(agent_type, AgentType.BASIC)
        
        return await enhanced_orchestrator.create_agent_unlimited(
            agent_type=agent_type_enum,
            name=name,
            description=description,
            config=config or {},
            tools=tools or []
        )
    
    async def create_unlimited_tool(
        self,
        name: str,
        description: str,
        functionality_description: str,
        assign_to_agent: Optional[str] = None,
        make_global: bool = True
    ) -> str:
        """
        Create unlimited tools with full integration.
        
        Args:
            name: Tool name
            description: Tool description
            functionality_description: What the tool should do
            assign_to_agent: Optional agent to assign to
            make_global: Whether to make tool globally available
            
        Returns:
            Tool name
        """
        # Create the tool
        tool = await production_tool_registry.create_production_tool_from_description(
            name=name,
            description=description,
            functionality_description=functionality_description
        )
        
        # Make global if requested
        if make_global:
            enhanced_orchestrator.global_tools[tool.name] = tool
        
        # Assign to specific agent if requested
        if assign_to_agent and assign_to_agent in enhanced_orchestrator.agents:
            await enhanced_orchestrator.assign_tools_to_agent(
                agent_id=assign_to_agent,
                tool_names=[tool.name]
            )
        
        return tool.name


# Global seamless integration manager
seamless_integration = SeamlessIntegrationManager()
