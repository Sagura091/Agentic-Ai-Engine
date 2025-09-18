"""
Agent Isolation Manager - THE Agent Isolation System.

This is THE ONLY agent isolation system in the entire application.
All agent isolation and access control flows through this manager.

CORE ARCHITECTURE:
- Strict isolation by default (agents cannot access each other's data)
- Optional sharing through explicit permissions
- Simple resource quotas and tracking
- Clean, fast access control

DESIGN PRINCIPLES:
- Isolation first, sharing second
- Simple, clean, fast operations
- No complexity unless absolutely necessary
- Security through isolation
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class IsolationLevel(str, Enum):
    """Levels of agent isolation - SIMPLIFIED."""
    STRICT = "strict"         # Complete isolation, no sharing (DEFAULT)
    SHARED = "shared"         # Allow sharing with explicit permissions


@dataclass
class ResourceQuota:
    """Simple resource quota for an agent - REASONABLE DEFAULTS."""
    max_documents: int = 10000
    max_memory_items: int = 5000
    max_queries_per_hour: int = 1000
    max_storage_mb: int = 1000


@dataclass
class ResourceUsage:
    """Current resource usage tracking - SIMPLIFIED."""
    documents_created: int = 0
    memory_items_created: int = 0
    queries_this_hour: int = 0
    storage_used_mb: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)


@dataclass
class AgentIsolationProfile:
    """Simple isolation profile for an agent - SIMPLIFIED."""
    agent_id: str
    isolation_level: IsolationLevel
    resource_quota: ResourceQuota
    resource_usage: ResourceUsage
    allowed_agents: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def can_access_agent(self, target_agent_id: str) -> bool:
        """Check if this agent can access another agent's resources."""
        if self.isolation_level == IsolationLevel.STRICT:
            return target_agent_id == self.agent_id  # Only self-access
        else:
            return target_agent_id == self.agent_id or target_agent_id in self.allowed_agents


class AgentIsolationManager:
    """
    Agent Isolation Manager - THE Agent Isolation System.

    SIMPLIFIED ARCHITECTURE:
    - Strict isolation by default
    - Simple resource quotas
    - Clean access control
    - Fast permission checks
    """

    def __init__(self, unified_rag):
        """Initialize THE agent isolation manager."""
        self.unified_rag = unified_rag
        self.is_initialized = False

        # Agent isolation profiles - SIMPLIFIED
        self.isolation_profiles: Dict[str, AgentIsolationProfile] = {}

        # Agent permissions - SIMPLIFIED
        self.agent_permissions: Dict[str, Set[str]] = {}  # agent_id -> allowed_agents

        # Simple stats
        self.stats = {
            "total_agents": 0,
            "strict_agents": 0,
            "shared_agents": 0,
            "total_permissions": 0
        }

        logger.info("THE Agent isolation manager initialized")

    async def initialize(self) -> None:
        """Initialize THE isolation manager."""
        try:
            if self.is_initialized:
                return

            # Ensure unified RAG is initialized
            if not self.unified_rag.is_initialized:
                await self.unified_rag.initialize()

            self.is_initialized = True
            logger.info("THE Isolation manager initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize THE isolation manager: {str(e)}")
            raise

    async def create_agent_isolation(
        self,
        agent_id: str,
        isolation_level: IsolationLevel = IsolationLevel.STRICT,
        resource_quota: Optional[ResourceQuota] = None,
        allowed_agents: Optional[Set[str]] = None
    ) -> AgentIsolationProfile:
        """
        Create isolation profile for a new agent.

        Args:
            agent_id: Agent identifier
            isolation_level: Level of isolation
            resource_quota: Resource limits for the agent
            allowed_agents: Set of allowed agent IDs for sharing

        Returns:
            Agent isolation profile
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            if agent_id in self.isolation_profiles:
                logger.warning(f"Isolation profile already exists for agent {agent_id}")
                return self.isolation_profiles[agent_id]

            # Create default resource quota if not provided
            if resource_quota is None:
                resource_quota = ResourceQuota()

            # Create isolation profile
            profile = AgentIsolationProfile(
                agent_id=agent_id,
                isolation_level=isolation_level,
                resource_quota=resource_quota,
                resource_usage=ResourceUsage(),
                allowed_agents=allowed_agents or set(),
                created_at=datetime.now(),
                last_updated=datetime.now()
            )

            self.isolation_profiles[agent_id] = profile

            # Initialize permissions
            self.agent_permissions[agent_id] = allowed_agents or set()

            # Ensure agent ecosystem exists in unified RAG
            await self.unified_rag.create_agent_ecosystem(agent_id)

            # Update stats
            self.stats["total_agents"] += 1

            logger.info(f"Created isolation profile for agent {agent_id}")
            return profile

        except Exception as e:
            logger.error(f"Failed to create isolation for agent {agent_id}: {str(e)}")
            raise
    
    async def validate_access(
        self,
        requesting_agent_id: str,
        target_agent_id: str
    ) -> bool:
        """
        Validate if an agent can access another agent's resources.

        Args:
            requesting_agent_id: Agent requesting access
            target_agent_id: Agent whose resources are being accessed

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Self-access is always allowed
            if requesting_agent_id == target_agent_id:
                return True

            # Check if requesting agent exists
            if requesting_agent_id not in self.isolation_profiles:
                return False

            # Check isolation level
            profile = self.isolation_profiles[requesting_agent_id]
            if profile.isolation_level == IsolationLevel.STRICT:
                return False

            # Check if target agent is in allowed list
            return target_agent_id in profile.allowed_agents

        except Exception as e:
            logger.error(f"Error validating access: {str(e)}")
            return False

    def get_agent_profile(self, agent_id: str) -> Optional[AgentIsolationProfile]:
        """Get isolation profile for an agent."""
        return self.isolation_profiles.get(agent_id)

    async def grant_access(
        self,
        granting_agent_id: str,
        requesting_agent_id: str
    ) -> bool:
        """
        Grant access permissions to another agent.

        Args:
            granting_agent_id: Agent granting access
            requesting_agent_id: Agent receiving access

        Returns:
            True if access was granted successfully
        """
        try:
            # Validate both agents exist
            if (granting_agent_id not in self.isolation_profiles or
                requesting_agent_id not in self.isolation_profiles):
                return False

            # Add to allowed agents
            profile = self.isolation_profiles[granting_agent_id]
            profile.allowed_agents.add(requesting_agent_id)
            profile.last_updated = datetime.now()

            # Update permissions
            if granting_agent_id not in self.agent_permissions:
                self.agent_permissions[granting_agent_id] = set()
            self.agent_permissions[granting_agent_id].add(requesting_agent_id)

            logger.info(f"Granted access from {granting_agent_id} to {requesting_agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to grant access: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get isolation manager statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "isolation_levels": {
                level.value: sum(1 for p in self.isolation_profiles.values()
                               if p.isolation_level == level)
                for level in IsolationLevel
            }
        }

