"""
Background Memory Consolidation Service

CRITICAL FIX: Automatic memory consolidation for all agents.

This service runs periodic consolidation to:
- Promote important short-term memories to long-term storage
- Forget low-value expired memories
- Optimize memory storage and retrieval
- Ensure agents learn and adapt over time

The service runs as a background task and consolidates memories
for all agents at configurable intervals.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory

logger = get_logger()


class MemoryConsolidationService:
    """
    Background service for automatic memory consolidation.
    
    This service ensures that agent memories are continuously optimized:
    - Important memories are promoted to long-term storage
    - Low-value memories are forgotten
    - Memory associations are strengthened
    - Learning patterns are identified and reinforced
    """
    
    def __init__(
        self,
        memory_system,
        interval_hours: int = 6,
        consolidation_threshold: int = 100,
        max_agents_per_cycle: int = 50
    ):
        """
        Initialize the consolidation service.
        
        Args:
            memory_system: The UnifiedMemorySystem instance
            interval_hours: Hours between consolidation cycles (default: 6)
            consolidation_threshold: Minimum memories before consolidation (default: 100)
            max_agents_per_cycle: Maximum agents to consolidate per cycle (default: 50)
        """
        self.memory_system = memory_system
        self.interval_hours = interval_hours
        self.consolidation_threshold = consolidation_threshold
        self.max_agents_per_cycle = max_agents_per_cycle
        
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._last_consolidation: Optional[datetime] = None
        self._consolidation_stats: Dict[str, Any] = {
            "total_cycles": 0,
            "total_agents_processed": 0,
            "total_memories_consolidated": 0,
            "total_memories_promoted": 0,
            "total_memories_forgotten": 0,
            "last_cycle_duration_ms": 0
        }
    
    async def start(self):
        """Start the consolidation service."""
        if self.is_running:
            logger.warn(
                "Memory consolidation service already running",
                LogCategory.MEMORY_OPERATIONS,
                "app.services.memory_consolidation_service"
            )
            return

        self.is_running = True
        self._task = asyncio.create_task(self._consolidation_loop())

        logger.info(
            "Memory consolidation service started",
            LogCategory.MEMORY_OPERATIONS,
            "app.services.memory_consolidation_service",
            data={
                "interval_hours": self.interval_hours,
                "consolidation_threshold": self.consolidation_threshold,
                "max_agents_per_cycle": self.max_agents_per_cycle
            }
        )

    async def stop(self):
        """Stop the consolidation service."""
        if not self.is_running:
            return

        self.is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(
            "Memory consolidation service stopped",
            LogCategory.MEMORY_OPERATIONS,
            "app.services.memory_consolidation_service",
            data={"stats": self._consolidation_stats}
        )
    
    async def _consolidation_loop(self):
        """Main consolidation loop."""
        while self.is_running:
            try:
                # Wait for the configured interval
                await asyncio.sleep(self.interval_hours * 3600)
                
                # Run consolidation cycle
                await self._run_consolidation_cycle()
                
            except asyncio.CancelledError:
                logger.info(
                    "Consolidation loop cancelled",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.services.memory_consolidation_service"
                )
                break
            except Exception as e:
                logger.error(
                    f"Consolidation loop error: {e}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.services.memory_consolidation_service",
                    error=e
                )
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _run_consolidation_cycle(self):
        """Run a single consolidation cycle for all agents."""
        cycle_start = datetime.utcnow()
        
        try:
            # Get all agent IDs with memory collections
            agent_ids = list(self.memory_system.agent_memories.keys())
            
            if not agent_ids:
                logger.debug(
                    "No agents with memories to consolidate",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.services.memory_consolidation_service"
                )
                return

            logger.info(
                "Starting memory consolidation cycle",
                LogCategory.MEMORY_OPERATIONS,
                "app.services.memory_consolidation_service",
                data={
                    "total_agents": len(agent_ids),
                    "max_agents_per_cycle": self.max_agents_per_cycle
                }
            )
            
            # Limit agents per cycle to avoid overload
            agents_to_process = agent_ids[:self.max_agents_per_cycle]
            
            cycle_stats = {
                "agents_processed": 0,
                "memories_consolidated": 0,
                "memories_promoted": 0,
                "memories_forgotten": 0,
                "errors": 0
            }
            
            # Process each agent
            for agent_id in agents_to_process:
                try:
                    # Check if agent has enough memories to consolidate
                    collection = self.memory_system.agent_memories.get(agent_id)
                    if not collection:
                        continue
                    
                    total_memories = len(collection.memories)
                    if total_memories < self.consolidation_threshold:
                        logger.debug(
                            f"Agent {agent_id} has insufficient memories for consolidation",
                            LogCategory.MEMORY_OPERATIONS,
                            "app.services.memory_consolidation_service",
                            data={
                                "agent_id": agent_id,
                                "total_memories": total_memories,
                                "threshold": self.consolidation_threshold
                            }
                        )
                        continue

                    # Run consolidation for this agent
                    result = await self.memory_system.run_consolidation_for_agent(agent_id)

                    # Update stats
                    cycle_stats["agents_processed"] += 1
                    cycle_stats["memories_consolidated"] += result.get("memories_processed", 0)
                    cycle_stats["memories_promoted"] += result.get("memories_promoted", 0)
                    cycle_stats["memories_forgotten"] += result.get("memories_forgotten", 0)

                    logger.info(
                        f"Consolidation completed for agent {agent_id}",
                        LogCategory.MEMORY_OPERATIONS,
                        "app.services.memory_consolidation_service",
                        data={
                            "agent_id": agent_id,
                            "result": result
                        }
                    )

                except Exception as e:
                    cycle_stats["errors"] += 1
                    logger.error(
                        f"Consolidation failed for agent {agent_id}: {e}",
                        LogCategory.MEMORY_OPERATIONS,
                        "app.services.memory_consolidation_service",
                        data={"agent_id": agent_id},
                        error=e
                    )
                    continue
            
            # Calculate cycle duration
            cycle_duration = (datetime.utcnow() - cycle_start).total_seconds() * 1000
            
            # Update service stats
            self._consolidation_stats["total_cycles"] += 1
            self._consolidation_stats["total_agents_processed"] += cycle_stats["agents_processed"]
            self._consolidation_stats["total_memories_consolidated"] += cycle_stats["memories_consolidated"]
            self._consolidation_stats["total_memories_promoted"] += cycle_stats["memories_promoted"]
            self._consolidation_stats["total_memories_forgotten"] += cycle_stats["memories_forgotten"]
            self._consolidation_stats["last_cycle_duration_ms"] = cycle_duration
            self._last_consolidation = cycle_start
            
            logger.info(
                "Memory consolidation cycle completed",
                LogCategory.MEMORY_OPERATIONS,
                "app.services.memory_consolidation_service",
                data={
                    "cycle_stats": cycle_stats,
                    "duration_ms": cycle_duration,
                    "total_stats": self._consolidation_stats
                }
            )

        except Exception as e:
            logger.error(
                f"Failed to run consolidation cycle: {e}",
                LogCategory.MEMORY_OPERATIONS,
                "app.services.memory_consolidation_service",
                error=e
            )
    
    async def trigger_consolidation(self, agent_id: Optional[str] = None):
        """
        Manually trigger consolidation for a specific agent or all agents.
        
        Args:
            agent_id: Optional agent ID. If None, consolidates all agents.
        """
        try:
            if agent_id:
                # Consolidate specific agent
                result = await self.memory_system.run_consolidation_for_agent(agent_id)
                logger.info(
                    f"Manual consolidation completed for agent {agent_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.services.memory_consolidation_service",
                    data={
                        "agent_id": agent_id,
                        "result": result
                    }
                )
                return result
            else:
                # Consolidate all agents
                await self._run_consolidation_cycle()
                logger.info(
                    "Manual consolidation cycle completed",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.services.memory_consolidation_service"
                )
                return self._consolidation_stats

        except Exception as e:
            logger.error(
                "Manual consolidation failed",
                LogCategory.MEMORY_OPERATIONS,
                "app.services.memory_consolidation_service",
                error=e
            )
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation service statistics."""
        return {
            **self._consolidation_stats,
            "is_running": self.is_running,
            "interval_hours": self.interval_hours,
            "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
            "next_consolidation": (
                (self._last_consolidation + timedelta(hours=self.interval_hours)).isoformat()
                if self._last_consolidation else None
            )
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "is_running": self.is_running,
            "interval_hours": self.interval_hours,
            "consolidation_threshold": self.consolidation_threshold,
            "max_agents_per_cycle": self.max_agents_per_cycle,
            "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
            "total_cycles": self._consolidation_stats["total_cycles"],
            "total_agents_processed": self._consolidation_stats["total_agents_processed"]
        }

