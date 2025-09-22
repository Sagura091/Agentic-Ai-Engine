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
from app.core.unified_system_orchestrator import UnifiedSystemOrchestrator, get_system_orchestrator
from app.tools.unified_tool_repository import UnifiedToolRepository
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
    - Unified system orchestrator manages all components
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
        self.unified_orchestrator = None

        logger.info("Seamless integration manager created")
    
    async def initialize_complete_system(self) -> None:
        """
        Initialize the complete integrated system.

        This method ensures all components are properly initialized
        and integrated with each other.
        """
        try:
            logger.info("Starting complete system initialization")

            # Step 1: Initialize unified system orchestrator
            await self._initialize_unified_orchestrator()

            # Step 2: Verify all integrations
            await self._verify_integrations()

            self.is_initialized = True
            logger.info("Complete system initialization successful")

        except Exception as e:
            logger.error("Complete system initialization failed", error=str(e))
            raise
    
    async def _initialize_unified_orchestrator(self) -> None:
        """Initialize the enhanced unified system orchestrator."""
        try:
            from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
            self.unified_orchestrator = get_enhanced_system_orchestrator()

            # Initialize the enhanced orchestrator if not already initialized
            if not self.unified_orchestrator.is_initialized:
                await self.unified_orchestrator.initialize()

            self.integration_status["unified_orchestrator"] = "initialized"
            logger.info("Enhanced unified system orchestrator initialized")
        except Exception as e:
            logger.error("Enhanced unified system orchestrator initialization failed", error=str(e))
            raise
    
    async def _verify_integrations(self) -> None:
        """Verify all system integrations."""
        try:
            if not self.unified_orchestrator:
                raise RuntimeError("Unified orchestrator not initialized")

            # Verify orchestrator is running
            if not self.unified_orchestrator.status.is_running:
                raise RuntimeError("Unified orchestrator not running")

            # Verify core components
            required_components = [
                "unified_rag", "kb_manager", "isolation_manager",
                "memory_system", "tool_repository"
            ]

            for component in required_components:
                if not self.unified_orchestrator.status.components_status.get(component, False):
                    raise RuntimeError(f"Required component {component} not initialized")

            self.integration_status["system_verification"] = "passed"
            logger.info("System integration verification completed")

        except Exception as e:
            logger.error("System integration verification failed", error=str(e))
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            if not self.unified_orchestrator:
                return {"status": "not_initialized", "error": "Unified orchestrator not available"}

            return {
                "status": "operational" if self.unified_orchestrator.status.is_running else "stopped",
                "components": self.unified_orchestrator.status.components_status,
                "integration_status": self.integration_status,
                "system_info": {
                    "start_time": self.unified_orchestrator.status.start_time.isoformat() if self.unified_orchestrator.status.start_time else None,
                    "is_initialized": self.unified_orchestrator.status.is_initialized,
                    "is_running": self.unified_orchestrator.status.is_running
                }
            }
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {"status": "error", "error": str(e)}
    
    # This method was already replaced above - removing duplicate
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            "is_initialized": self.is_initialized,
            "integration_status": self.integration_status,
            "unified_orchestrator": {
                "status": "operational" if self.unified_orchestrator and self.unified_orchestrator.status.is_running else "stopped",
                "components": self.unified_orchestrator.status.components_status if self.unified_orchestrator else {}
            }
        }
    
    async def create_agent_ecosystem(
        self,
        agent_id: str,
        agent_type: str = "general",
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create agent ecosystem using unified system.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            config: Optional configuration

        Returns:
            Agent ID
        """
        if not self.unified_orchestrator:
            raise RuntimeError("Unified orchestrator not initialized")

        # Create agent ecosystem through unified system
        await self.unified_orchestrator.unified_rag.create_agent_ecosystem(agent_id)
        await self.unified_orchestrator.isolation_manager.create_agent_profile(agent_id)
        await self.unified_orchestrator.memory_system.create_agent_memory(agent_id)
        await self.unified_orchestrator.tool_repository.create_agent_profile(agent_id)

        logger.info(f"Created agent ecosystem for {agent_id}")
        return agent_id
    
    async def shutdown(self) -> None:
        """Shutdown the seamless integration system."""
        try:
            if self.unified_orchestrator:
                await self.unified_orchestrator.shutdown()

            self.is_initialized = False
            logger.info("Seamless integration system shutdown completed")

        except Exception as e:
            logger.error("Failed to shutdown seamless integration system", error=str(e))


# Global seamless integration manager
seamless_integration = SeamlessIntegrationManager()
