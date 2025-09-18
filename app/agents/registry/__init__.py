"""
AI Agent Registry - Agent Management and Discovery System.

This module provides comprehensive agent registration, discovery, and lifecycle
management capabilities for the AI Agent Builder platform.

CORE FEATURES:
- Agent registration and discovery
- Lifecycle management (create, start, stop, destroy)
- Performance monitoring and metrics
- Agent health checking and recovery
- Multi-tenant agent isolation
- Agent collaboration and communication

DESIGN PRINCIPLES:
- Centralized agent management
- Enterprise-grade monitoring
- Scalable multi-agent coordination
- Production-ready reliability
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import structlog

from app.agents.base.agent import LangGraphAgent
from app.agents.autonomous.autonomous_agent import AutonomousLangGraphAgent
from app.agents.factory import AgentType, AgentTemplate, AgentBuilderFactory, AgentBuilderConfig
from app.core.unified_system_orchestrator import UnifiedSystemOrchestrator

logger = structlog.get_logger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DESTROYED = "destroyed"


class AgentHealth(Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentMetrics:
    """Agent performance and usage metrics."""
    
    # Execution metrics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Tool usage metrics
    tool_calls: int = 0
    successful_tool_calls: int = 0
    
    # Collaboration metrics
    messages_sent: int = 0
    messages_received: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    last_health_check: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegisteredAgent:
    """Registered agent with metadata and management information."""
    
    # Basic information
    agent_id: str
    name: str
    description: str
    agent_type: AgentType
    template: Optional[AgentTemplate]
    
    # Agent instance
    agent: Union[LangGraphAgent, AutonomousLangGraphAgent]
    
    # Status and health
    status: AgentStatus = AgentStatus.CREATED
    health: AgentHealth = AgentHealth.UNKNOWN
    
    # Configuration
    config: AgentBuilderConfig
    
    # Metrics and monitoring
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    
    # Collaboration
    collaborators: Set[str] = field(default_factory=set)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    tenant_id: Optional[str] = None


class AgentRegistry:
    """
    Centralized registry for managing AI agents in the platform.
    
    This registry provides comprehensive agent lifecycle management,
    monitoring, and coordination capabilities.
    """
    
    def __init__(self, agent_factory: AgentBuilderFactory, system_orchestrator: Optional[UnifiedSystemOrchestrator] = None):
        self.agent_factory = agent_factory
        self.system_orchestrator = system_orchestrator
        
        # Agent storage
        self._agents: Dict[str, RegisteredAgent] = {}
        self._agents_by_type: Dict[AgentType, Set[str]] = {agent_type: set() for agent_type in AgentType}
        self._agents_by_template: Dict[AgentTemplate, Set[str]] = {template: set() for template in AgentTemplate}
        self._agents_by_tenant: Dict[str, Set[str]] = {}
        
        # Monitoring
        self._health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Collaboration
        self._collaboration_groups: Dict[str, Set[str]] = {}

        # Distributed architecture components
        self._distributed_mode = False
        self._node_id = None
        self._cluster_name = None
        self._redis_client = None
        self._postgres_pool = None

        logger.info("Agent registry initialized")

        # Initialize distributed components if enabled
        self._initialize_distributed_components()
    
    async def register_agent(
        self,
        config: AgentBuilderConfig,
        agent_id: Optional[str] = None,
        owner: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a new agent in the platform.
        
        Args:
            config: Agent builder configuration
            agent_id: Optional custom agent ID
            owner: Agent owner identifier
            tenant_id: Tenant identifier for multi-tenancy
            tags: Optional tags for categorization
            
        Returns:
            Agent ID
        """
        try:
            # Generate agent ID if not provided
            if not agent_id:
                agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            
            # Check if agent ID already exists
            if agent_id in self._agents:
                raise ValueError(f"Agent with ID {agent_id} already exists")
            
            # Build the agent
            agent = await self.agent_factory.build_agent(config)
            
            # Create registered agent
            registered_agent = RegisteredAgent(
                agent_id=agent_id,
                name=config.name,
                description=config.description,
                agent_type=config.agent_type,
                template=config.template,
                agent=agent,
                config=config,
                tags=tags or [],
                owner=owner,
                tenant_id=tenant_id
            )
            
            # Store agent
            self._agents[agent_id] = registered_agent
            self._agents_by_type[config.agent_type].add(agent_id)
            
            if config.template:
                self._agents_by_template[config.template].add(agent_id)
            
            if tenant_id:
                if tenant_id not in self._agents_by_tenant:
                    self._agents_by_tenant[tenant_id] = set()
                self._agents_by_tenant[tenant_id].add(agent_id)
            
            # Start health monitoring if this is the first agent
            if len(self._agents) == 1 and not self._health_check_task:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("Agent registered successfully", agent_id=agent_id, name=config.name, type=config.agent_type.value)
            return agent_id
            
        except Exception as e:
            logger.error("Failed to register agent", error=str(e), config=config)
            raise
    
    async def register_from_template(
        self,
        template: AgentTemplate,
        overrides: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        owner: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Register an agent from a template."""
        agent = await self.agent_factory.build_from_template(template, overrides)
        config = self.agent_factory.get_template_config(template)
        
        # Apply overrides to config if provided
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return await self.register_agent(config, agent_id, owner, tenant_id, tags)
    
    def get_agent(self, agent_id: str) -> Optional[RegisteredAgent]:
        """Get a registered agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(
        self,
        agent_type: Optional[AgentType] = None,
        template: Optional[AgentTemplate] = None,
        tenant_id: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[RegisteredAgent]:
        """List agents with optional filtering."""
        agents = list(self._agents.values())
        
        # Apply filters
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        
        if template:
            agents = [a for a in agents if a.template == template]
        
        if tenant_id:
            agents = [a for a in agents if a.tenant_id == tenant_id]
        
        if status:
            agents = [a for a in agents if a.status == status]
        
        if tags:
            agents = [a for a in agents if any(tag in a.tags for tag in tags)]
        
        return agents
    
    async def start_agent(self, agent_id: str) -> bool:
        """Start an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        
        try:
            agent.status = AgentStatus.STARTING
            # Agent-specific startup logic would go here
            agent.status = AgentStatus.RUNNING
            agent.metrics.last_activity = datetime.utcnow()
            
            logger.info("Agent started", agent_id=agent_id)
            return True
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            logger.error("Failed to start agent", agent_id=agent_id, error=str(e))
            return False
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        
        try:
            agent.status = AgentStatus.STOPPING
            # Agent-specific shutdown logic would go here
            agent.status = AgentStatus.STOPPED
            
            logger.info("Agent stopped", agent_id=agent_id)
            return True
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            logger.error("Failed to stop agent", agent_id=agent_id, error=str(e))
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister and destroy an agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        try:
            # Stop agent if running
            if agent.status == AgentStatus.RUNNING:
                await self.stop_agent(agent_id)

            # Remove from all indexes
            self._agents_by_type[agent.agent_type].discard(agent_id)

            if agent.template:
                self._agents_by_template[agent.template].discard(agent_id)

            if agent.tenant_id:
                self._agents_by_tenant.get(agent.tenant_id, set()).discard(agent_id)

            # Remove from collaboration groups
            for group_agents in self._collaboration_groups.values():
                group_agents.discard(agent_id)

            # Remove from registry
            del self._agents[agent_id]

            # Stop health monitoring if no agents left
            if not self._agents and self._health_check_task:
                self._health_check_task.cancel()
                self._health_check_task = None

            logger.info("Agent unregistered", agent_id=agent_id)
            return True

        except Exception as e:
            logger.error("Failed to unregister agent", agent_id=agent_id, error=str(e))
            return False

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent."""
        agent = self._agents.get(agent_id)
        return agent.metrics if agent else None

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        total_agents = len(self._agents)
        agents_by_status = {}
        agents_by_type = {}
        agents_by_health = {}

        for agent in self._agents.values():
            # Count by status
            status_key = agent.status.value
            agents_by_status[status_key] = agents_by_status.get(status_key, 0) + 1

            # Count by type
            type_key = agent.agent_type.value
            agents_by_type[type_key] = agents_by_type.get(type_key, 0) + 1

            # Count by health
            health_key = agent.health.value
            agents_by_health[health_key] = agents_by_health.get(health_key, 0) + 1

        return {
            "total_agents": total_agents,
            "agents_by_status": agents_by_status,
            "agents_by_type": agents_by_type,
            "agents_by_health": agents_by_health,
            "collaboration_groups": len(self._collaboration_groups),
            "tenants": len(self._agents_by_tenant)
        }

    async def create_collaboration_group(self, group_id: str, agent_ids: List[str]) -> bool:
        """Create a collaboration group for agents."""
        try:
            # Validate all agents exist
            for agent_id in agent_ids:
                if agent_id not in self._agents:
                    raise ValueError(f"Agent {agent_id} not found")

            # Create group
            self._collaboration_groups[group_id] = set(agent_ids)

            # Update agent collaborators
            for agent_id in agent_ids:
                agent = self._agents[agent_id]
                agent.collaborators.update(set(agent_ids) - {agent_id})

            logger.info("Collaboration group created", group_id=group_id, agents=agent_ids)
            return True

        except Exception as e:
            logger.error("Failed to create collaboration group", group_id=group_id, error=str(e))
            return False

    async def _health_check_loop(self):
        """Background task for agent health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))

    async def _perform_health_checks(self):
        """Perform health checks on all agents."""
        for agent_id, agent in self._agents.items():
            try:
                # Simple health check - can be enhanced
                if agent.status == AgentStatus.RUNNING:
                    # Check if agent has been active recently
                    time_since_activity = datetime.utcnow() - agent.metrics.last_activity

                    if time_since_activity > timedelta(minutes=30):
                        agent.health = AgentHealth.DEGRADED
                    else:
                        agent.health = AgentHealth.HEALTHY
                else:
                    agent.health = AgentHealth.UNKNOWN

                agent.metrics.last_health_check = datetime.utcnow()

            except Exception as e:
                agent.health = AgentHealth.UNHEALTHY
                logger.error("Health check failed", agent_id=agent_id, error=str(e))


    def _initialize_distributed_components(self):
        """Initialize distributed architecture components if enabled."""
        try:
            from app.config.settings import get_settings
            settings = get_settings()

            if hasattr(settings, 'ENABLE_DISTRIBUTED_MODE') and settings.ENABLE_DISTRIBUTED_MODE:
                self._distributed_mode = True
                self._node_id = getattr(settings, 'NODE_ID', f"node-{id(self)}")
                self._cluster_name = getattr(settings, 'CLUSTER_NAME', 'agent_builder_cluster')

                logger.info("Distributed mode enabled", node_id=self._node_id, cluster=self._cluster_name)

                # Initialize Redis for distributed state
                self._initialize_redis()

                # Initialize PostgreSQL for persistence
                self._initialize_postgres()

        except Exception as e:
            logger.warning("Failed to initialize distributed components", error=str(e))

    def _initialize_redis(self):
        """Initialize Redis client for distributed state management."""
        try:
            import redis
            from app.config.settings import get_settings
            settings = get_settings()

            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            self._redis_client = redis.from_url(redis_url, decode_responses=True)

            # Test connection
            self._redis_client.ping()
            logger.info("Redis connection established", url=redis_url)

        except Exception as e:
            logger.error("Failed to initialize Redis", error=str(e))
            self._redis_client = None

    def _initialize_postgres(self):
        """Initialize PostgreSQL connection pool for persistence."""
        try:
            import asyncpg
            from app.config.settings import get_settings
            settings = get_settings()

            # This would be initialized with actual connection pool
            # For now, we'll just log the intent
            postgres_url = getattr(settings, 'DATABASE_URL', 'postgresql://localhost/agent_builder')
            logger.info("PostgreSQL persistence configured", url=postgres_url)

        except Exception as e:
            logger.error("Failed to initialize PostgreSQL", error=str(e))
            self._postgres_pool = None

    async def sync_agent_state(self, agent_id: str):
        """Sync agent state across distributed nodes."""
        if not self._distributed_mode or not self._redis_client:
            return

        try:
            agent = self._agents.get(agent_id)
            if not agent:
                return

            # Serialize agent state
            agent_state = {
                "agent_id": agent_id,
                "status": agent.status.value,
                "node_id": self._node_id,
                "last_updated": agent.last_updated.isoformat(),
                "config": {
                    "name": agent.config.name,
                    "agent_type": agent.config.agent_type.value,
                    "description": agent.config.description
                }
            }

            # Store in Redis with expiration
            key = f"agent:{self._cluster_name}:{agent_id}"
            self._redis_client.setex(key, 3600, str(agent_state))  # 1 hour expiration

            logger.debug("Agent state synced", agent_id=agent_id, node_id=self._node_id)

        except Exception as e:
            logger.error("Failed to sync agent state", agent_id=agent_id, error=str(e))

    async def discover_cluster_agents(self) -> Dict[str, Any]:
        """Discover agents across the cluster."""
        if not self._distributed_mode or not self._redis_client:
            return {}

        try:
            pattern = f"agent:{self._cluster_name}:*"
            keys = self._redis_client.keys(pattern)

            cluster_agents = {}
            for key in keys:
                agent_data = self._redis_client.get(key)
                if agent_data:
                    # Parse agent state (simplified - would use proper JSON)
                    agent_id = key.split(":")[-1]
                    cluster_agents[agent_id] = agent_data

            logger.info("Discovered cluster agents", count=len(cluster_agents))
            return cluster_agents

        except Exception as e:
            logger.error("Failed to discover cluster agents", error=str(e))
            return {}

    def get_distributed_status(self) -> Dict[str, Any]:
        """Get distributed architecture status."""
        return {
            "distributed_mode": self._distributed_mode,
            "node_id": self._node_id,
            "cluster_name": self._cluster_name,
            "redis_connected": self._redis_client is not None,
            "postgres_connected": self._postgres_pool is not None
        }


# Global registry instance
_global_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> Optional[AgentRegistry]:
    """Get the global agent registry instance."""
    return _global_registry


def initialize_agent_registry(agent_factory: AgentBuilderFactory, system_orchestrator: Optional[UnifiedSystemOrchestrator] = None) -> AgentRegistry:
    """Initialize the global agent registry."""
    global _global_registry
    _global_registry = AgentRegistry(agent_factory, system_orchestrator)
    return _global_registry


__all__ = [
    "AgentStatus",
    "AgentHealth",
    "AgentMetrics",
    "RegisteredAgent",
    "AgentRegistry",
    "get_agent_registry",
    "initialize_agent_registry"
]
