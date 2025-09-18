"""
Agent Isolation Manager.

This module provides comprehensive agent isolation capabilities ensuring
that each agent operates in its own secure, isolated environment while
maintaining the ability to share resources when explicitly permitted.

Features:
- Collection-based data isolation
- Access control and permissions
- Resource quotas and limits
- Security validation
- Audit logging
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .unified_rag_system import UnifiedRAGSystem, AgentCollections

logger = structlog.get_logger(__name__)


class IsolationLevel(str, Enum):
    """Levels of agent isolation."""
    STRICT = "strict"         # Complete isolation, no sharing
    CONTROLLED = "controlled" # Sharing with explicit permissions
    COLLABORATIVE = "collaborative"  # Open sharing within groups


class ResourceType(str, Enum):
    """Types of resources that can be isolated."""
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    TOOLS = "tools"
    COMMUNICATION = "communication"


@dataclass
class ResourceQuota:
    """Resource quota for an agent."""
    max_documents: int = 10000
    max_memory_items: int = 5000
    max_storage_mb: int = 1000
    max_queries_per_hour: int = 1000
    max_tool_calls_per_hour: int = 500


@dataclass
class AgentIsolationProfile:
    """Isolation profile for an agent."""
    agent_id: str
    isolation_level: IsolationLevel
    resource_quota: ResourceQuota
    allowed_shared_resources: Set[str]
    blocked_agents: Set[str]
    created_at: datetime
    last_updated: datetime


class AgentIsolationManager:
    """
    Agent Isolation Manager.
    
    Manages agent isolation, access controls, and resource quotas
    to ensure secure and efficient multi-agent operations.
    """
    
    def __init__(self, unified_rag: UnifiedRAGSystem):
        """Initialize the agent isolation manager."""
        self.unified_rag = unified_rag
        
        # Agent isolation profiles
        self.isolation_profiles: Dict[str, AgentIsolationProfile] = {}
        
        # Access control matrices
        self.access_permissions: Dict[str, Dict[str, Set[ResourceType]]] = {}  # agent_id -> {target_agent_id -> permissions}
        self.shared_resource_access: Dict[str, Set[str]] = {}  # resource_id -> agent_ids
        
        # Resource usage tracking
        self.resource_usage: Dict[str, Dict[str, Any]] = {}  # agent_id -> usage_stats
        
        # Security audit log
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info("Agent isolation manager initialized")
    
    async def create_agent_isolation(
        self,
        agent_id: str,
        isolation_level: IsolationLevel = IsolationLevel.CONTROLLED,
        resource_quota: Optional[ResourceQuota] = None
    ) -> AgentIsolationProfile:
        """
        Create isolation profile for a new agent.
        
        Args:
            agent_id: Agent identifier
            isolation_level: Level of isolation
            resource_quota: Resource limits for the agent
            
        Returns:
            Agent isolation profile
        """
        try:
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
                allowed_shared_resources=set(),
                blocked_agents=set(),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.isolation_profiles[agent_id] = profile
            
            # Initialize access permissions
            self.access_permissions[agent_id] = {}
            
            # Initialize resource usage tracking
            self.resource_usage[agent_id] = {
                "documents_created": 0,
                "memory_items_created": 0,
                "storage_used_mb": 0,
                "queries_today": 0,
                "tool_calls_today": 0,
                "last_reset": datetime.utcnow()
            }
            
            # Ensure agent ecosystem exists in unified RAG
            await self.unified_rag.create_agent_ecosystem(agent_id)
            
            # Log creation
            await self._audit_log(
                agent_id=agent_id,
                action="create_isolation",
                details={"isolation_level": isolation_level.value}
            )
            
            logger.info(f"Created isolation profile for agent {agent_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to create isolation for agent {agent_id}: {str(e)}")
            raise
    
    async def validate_access(
        self,
        requesting_agent_id: str,
        target_agent_id: str,
        resource_type: ResourceType,
        operation: str = "read"
    ) -> bool:
        """
        Validate if an agent can access another agent's resources.
        
        Args:
            requesting_agent_id: Agent requesting access
            target_agent_id: Agent whose resources are being accessed
            resource_type: Type of resource being accessed
            operation: Type of operation (read, write, delete)
            
        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Self-access is always allowed
            if requesting_agent_id == target_agent_id:
                return True
            
            # Check if requesting agent exists
            if requesting_agent_id not in self.isolation_profiles:
                logger.warning(f"Unknown agent {requesting_agent_id} requesting access")
                return False
            
            # Check if target agent exists
            if target_agent_id not in self.isolation_profiles:
                logger.warning(f"Target agent {target_agent_id} does not exist")
                return False
            
            requesting_profile = self.isolation_profiles[requesting_agent_id]
            target_profile = self.isolation_profiles[target_agent_id]
            
            # Check if target agent is blocked
            if requesting_agent_id in target_profile.blocked_agents:
                await self._audit_log(
                    agent_id=requesting_agent_id,
                    action="access_denied_blocked",
                    details={
                        "target_agent": target_agent_id,
                        "resource_type": resource_type.value,
                        "operation": operation
                    }
                )
                return False
            
            # Check isolation level
            if target_profile.isolation_level == IsolationLevel.STRICT:
                # Strict isolation - no external access
                return False
            
            # Check explicit permissions
            agent_permissions = self.access_permissions.get(target_agent_id, {})
            requesting_permissions = agent_permissions.get(requesting_agent_id, set())
            
            if resource_type in requesting_permissions:
                await self._audit_log(
                    agent_id=requesting_agent_id,
                    action="access_granted",
                    details={
                        "target_agent": target_agent_id,
                        "resource_type": resource_type.value,
                        "operation": operation
                    }
                )
                return True
            
            # For collaborative isolation, allow access to shared resources
            if target_profile.isolation_level == IsolationLevel.COLLABORATIVE:
                # Check if resource is in shared resources
                resource_key = f"{target_agent_id}_{resource_type.value}"
                if resource_key in self.shared_resource_access:
                    shared_agents = self.shared_resource_access[resource_key]
                    if requesting_agent_id in shared_agents:
                        return True
            
            # Default deny
            await self._audit_log(
                agent_id=requesting_agent_id,
                action="access_denied",
                details={
                    "target_agent": target_agent_id,
                    "resource_type": resource_type.value,
                    "operation": operation,
                    "reason": "no_permission"
                }
            )
            return False
            
        except Exception as e:
            logger.error(f"Failed to validate access: {str(e)}")
            return False
    
    async def grant_access(
        self,
        granting_agent_id: str,
        requesting_agent_id: str,
        resource_types: List[ResourceType]
    ) -> bool:
        """
        Grant access permissions to another agent.
        
        Args:
            granting_agent_id: Agent granting access
            requesting_agent_id: Agent receiving access
            resource_types: Types of resources to grant access to
            
        Returns:
            True if access was granted successfully
        """
        try:
            # Validate both agents exist
            if (granting_agent_id not in self.isolation_profiles or 
                requesting_agent_id not in self.isolation_profiles):
                return False
            
            # Initialize permissions if not exists
            if granting_agent_id not in self.access_permissions:
                self.access_permissions[granting_agent_id] = {}
            
            if requesting_agent_id not in self.access_permissions[granting_agent_id]:
                self.access_permissions[granting_agent_id][requesting_agent_id] = set()
            
            # Grant permissions
            for resource_type in resource_types:
                self.access_permissions[granting_agent_id][requesting_agent_id].add(resource_type)
            
            # Update profile
            self.isolation_profiles[granting_agent_id].last_updated = datetime.utcnow()
            
            # Log the permission grant
            await self._audit_log(
                agent_id=granting_agent_id,
                action="grant_access",
                details={
                    "target_agent": requesting_agent_id,
                    "resource_types": [rt.value for rt in resource_types]
                }
            )
            
            logger.info(f"Granted access from {granting_agent_id} to {requesting_agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to grant access: {str(e)}")
            return False
    
    async def check_resource_quota(
        self,
        agent_id: str,
        resource_type: str,
        amount: int = 1
    ) -> bool:
        """
        Check if agent can use additional resources without exceeding quota.
        
        Args:
            agent_id: Agent identifier
            resource_type: Type of resource
            amount: Amount of resource to check
            
        Returns:
            True if within quota, False otherwise
        """
        try:
            if agent_id not in self.isolation_profiles:
                return False
            
            profile = self.isolation_profiles[agent_id]
            usage = self.resource_usage.get(agent_id, {})
            
            # Check specific resource limits
            if resource_type == "documents":
                current = usage.get("documents_created", 0)
                return current + amount <= profile.resource_quota.max_documents
            
            elif resource_type == "memory_items":
                current = usage.get("memory_items_created", 0)
                return current + amount <= profile.resource_quota.max_memory_items
            
            elif resource_type == "storage_mb":
                current = usage.get("storage_used_mb", 0)
                return current + amount <= profile.resource_quota.max_storage_mb
            
            elif resource_type == "queries":
                current = usage.get("queries_today", 0)
                return current + amount <= profile.resource_quota.max_queries_per_hour
            
            elif resource_type == "tool_calls":
                current = usage.get("tool_calls_today", 0)
                return current + amount <= profile.resource_quota.max_tool_calls_per_hour
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check resource quota: {str(e)}")
            return False
    
    async def update_resource_usage(
        self,
        agent_id: str,
        resource_type: str,
        amount: int = 1
    ) -> None:
        """Update resource usage for an agent."""
        try:
            if agent_id not in self.resource_usage:
                self.resource_usage[agent_id] = {}
            
            usage = self.resource_usage[agent_id]
            current = usage.get(resource_type, 0)
            usage[resource_type] = current + amount
            
        except Exception as e:
            logger.error(f"Failed to update resource usage: {str(e)}")
    
    async def _audit_log(
        self,
        agent_id: str,
        action: str,
        details: Dict[str, Any]
    ) -> None:
        """Add entry to audit log."""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "action": action,
                "details": details
            }
            
            self.audit_log.append(log_entry)
            
            # Keep only last 10000 entries
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-10000:]
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {str(e)}")
    
    def get_agent_profile(self, agent_id: str) -> Optional[AgentIsolationProfile]:
        """Get isolation profile for an agent."""
        return self.isolation_profiles.get(agent_id)
    
    def get_isolation_stats(self) -> Dict[str, Any]:
        """Get isolation manager statistics."""
        return {
            "total_agents": len(self.isolation_profiles),
            "isolation_levels": {
                level.value: sum(1 for p in self.isolation_profiles.values() 
                               if p.isolation_level == level)
                for level in IsolationLevel
            },
            "total_permissions": sum(len(perms) for perms in self.access_permissions.values()),
            "audit_log_entries": len(self.audit_log)
        }
