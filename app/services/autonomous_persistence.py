#!/usr/bin/env python3
"""
Autonomous Agent Persistence Service

This module provides persistence capabilities for autonomous agents,
including goal management, decision history, and learning data storage.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path

from app.backend_logging import get_logger
from app.backend_logging.models import LogCategory

logger = get_logger()


@dataclass
class AutonomousGoal:
    """Represents an autonomous goal for an agent."""
    goal_id: str
    agent_id: str
    description: str
    priority: float  # 0.0 to 1.0
    status: str  # "active", "completed", "paused", "cancelled"
    created_at: datetime
    target_completion: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_goals: List[str] = field(default_factory=list)  # List of goal_ids
    dependencies: List[str] = field(default_factory=list)  # List of goal_ids


@dataclass
class DecisionRecord:
    """Records an autonomous decision made by an agent."""
    decision_id: str
    agent_id: str
    context: str
    decision: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    outcome: Optional[str] = None
    success: Optional[bool] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningData:
    """Stores learning data for autonomous agents."""
    learning_id: str
    agent_id: str
    experience_type: str  # "success", "failure", "observation", "feedback"
    context: str
    action_taken: str
    result: str
    lesson_learned: str
    confidence_change: float  # How much this changed agent's confidence
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutonomousPersistenceService:
    """
    Persistence service for autonomous agents.
    
    Provides storage and retrieval of:
    - Goals and goal hierarchies
    - Decision history and reasoning
    - Learning experiences and patterns
    - Agent state and memory
    """
    
    def __init__(self, data_dir: str = "./data/autonomous"):
        """Initialize the persistence service."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches for performance
        self.goals_cache: Dict[str, Dict[str, AutonomousGoal]] = {}  # agent_id -> goal_id -> goal
        self.decisions_cache: Dict[str, List[DecisionRecord]] = {}  # agent_id -> decisions
        self.learning_cache: Dict[str, List[LearningData]] = {}  # agent_id -> learning_data
        
        # File paths
        self.goals_file = self.data_dir / "goals.json"
        self.decisions_file = self.data_dir / "decisions.json"
        self.learning_file = self.data_dir / "learning.json"

        logger.info(
            "Autonomous persistence service initialized",
            LogCategory.SERVICE_OPERATIONS,
            "app.services.autonomous_persistence",
            data={"data_dir": str(self.data_dir)}
        )

    async def initialize(self) -> None:
        """Initialize the persistence service and load existing data."""
        try:
            await self._load_all_data()
            logger.info(
                "Autonomous persistence service loaded successfully",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.autonomous_persistence"
            )
        except Exception as e:
            logger.error(
                "Failed to initialize autonomous persistence",
                LogCategory.SERVICE_OPERATIONS,
                "app.services.autonomous_persistence",
                error=e
            )
            raise
    
    # Goal Management
    async def create_goal(self, agent_id: str, description: str, priority: float = 0.5,
                         target_completion: Optional[datetime] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> AutonomousGoal:
        """Create a new goal for an agent."""
        goal = AutonomousGoal(
            goal_id=str(uuid.uuid4()),
            agent_id=agent_id,
            description=description,
            priority=priority,
            status="active",
            created_at=datetime.now(),
            target_completion=target_completion,
            metadata=metadata or {}
        )
        
        if agent_id not in self.goals_cache:
            self.goals_cache[agent_id] = {}
        
        self.goals_cache[agent_id][goal.goal_id] = goal
        await self._save_goals()

        logger.info(
            f"Created goal for agent {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "app.services.autonomous_persistence",
            data={"agent_id": agent_id, "goal_id": goal.goal_id, "description": description}
        )
        return goal
    
    async def get_agent_goals(self, agent_id: str, status: Optional[str] = None) -> List[AutonomousGoal]:
        """Get all goals for an agent, optionally filtered by status."""
        agent_goals = self.goals_cache.get(agent_id, {})
        goals = list(agent_goals.values())
        
        if status:
            goals = [g for g in goals if g.status == status]
        
        return sorted(goals, key=lambda g: g.priority, reverse=True)
    
    async def update_goal_progress(self, goal_id: str, progress: float, 
                                  status: Optional[str] = None) -> bool:
        """Update goal progress and optionally status."""
        for agent_goals in self.goals_cache.values():
            if goal_id in agent_goals:
                goal = agent_goals[goal_id]
                goal.progress = max(0.0, min(1.0, progress))
                if status:
                    goal.status = status
                await self._save_goals()
                logger.info(
                    "Updated goal progress",
                    LogCategory.AGENT_OPERATIONS,
                    "app.services.autonomous_persistence",
                    data={"goal_id": goal_id, "progress": progress, "status": status}
                )
                return True
        return False
    
    # Decision History
    async def record_decision(self, agent_id: str, context: str, decision: str,
                            confidence: float, reasoning: str,
                            metadata: Optional[Dict[str, Any]] = None) -> DecisionRecord:
        """Record a decision made by an autonomous agent."""
        record = DecisionRecord(
            decision_id=str(uuid.uuid4()),
            agent_id=agent_id,
            context=context,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata or {}
        )
        
        if agent_id not in self.decisions_cache:
            self.decisions_cache[agent_id] = []
        
        self.decisions_cache[agent_id].append(record)
        await self._save_decisions()

        logger.info(
            f"Recorded decision for agent {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "app.services.autonomous_persistence",
            data={"agent_id": agent_id, "decision_id": record.decision_id}
        )
        return record
    
    async def get_decision_history(self, agent_id: str, limit: int = 100) -> List[DecisionRecord]:
        """Get decision history for an agent."""
        decisions = self.decisions_cache.get(agent_id, [])
        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)[:limit]
    
    # Learning Data
    async def record_learning(self, agent_id: str, experience_type: str, context: str,
                            action_taken: str, result: str, lesson_learned: str,
                            confidence_change: float = 0.0,
                            metadata: Optional[Dict[str, Any]] = None) -> LearningData:
        """Record a learning experience for an autonomous agent."""
        learning = LearningData(
            learning_id=str(uuid.uuid4()),
            agent_id=agent_id,
            experience_type=experience_type,
            context=context,
            action_taken=action_taken,
            result=result,
            lesson_learned=lesson_learned,
            confidence_change=confidence_change,
            metadata=metadata or {}
        )
        
        if agent_id not in self.learning_cache:
            self.learning_cache[agent_id] = []
        
        self.learning_cache[agent_id].append(learning)
        await self._save_learning()

        logger.info(
            f"Recorded learning for agent {agent_id}",
            LogCategory.AGENT_OPERATIONS,
            "app.services.autonomous_persistence",
            data={"agent_id": agent_id, "learning_id": learning.learning_id}
        )
        return learning
    
    async def get_learning_history(self, agent_id: str, experience_type: Optional[str] = None,
                                  limit: int = 100) -> List[LearningData]:
        """Get learning history for an agent."""
        learning_data = self.learning_cache.get(agent_id, [])

        if experience_type:
            learning_data = [l for l in learning_data if l.experience_type == experience_type]

        return sorted(learning_data, key=lambda l: l.timestamp, reverse=True)[:limit]

    async def load_agent_memories(self, agent_id: str) -> List[Dict[str, Any]]:
        """Load agent memories from persistence storage."""
        try:
            from app.models.database.base import get_database_session
            from app.models.autonomous import AutonomousAgentState, AgentMemoryDB
            from sqlalchemy import select

            memories = []

            # Convert agent_id to UUID if needed
            import uuid as uuid_module
            if isinstance(agent_id, str):
                try:
                    agent_uuid = uuid_module.UUID(agent_id)
                except ValueError:
                    logger.warn(
                        f"Invalid agent_id format: {agent_id}",
                        LogCategory.AGENT_OPERATIONS,
                        "app.services.autonomous_persistence",
                        data={"agent_id": agent_id}
                    )
                    return []
            else:
                agent_uuid = agent_id

            async for session in get_database_session():
                # Get agent state
                agent_state = await session.execute(
                    select(AutonomousAgentState).where(AutonomousAgentState.agent_id == agent_uuid)
                )
                agent_state_record = agent_state.scalar_one_or_none()

                if not agent_state_record:
                    logger.debug(
                        f"No agent state found for {agent_uuid}",
                        LogCategory.AGENT_OPERATIONS,
                        "app.services.autonomous_persistence",
                        data={"agent_uuid": str(agent_uuid)}
                    )
                    return []

                # Load memories for this agent
                memories_query = await session.execute(
                    select(AgentMemoryDB)
                    .where(AgentMemoryDB.agent_state_id == agent_state_record.id)
                    .order_by(AgentMemoryDB.created_at.desc())
                    .limit(1000)  # Load last 1000 memories
                )
                memory_records = memories_query.scalars().all()

                for memory_record in memory_records:
                    memory_data = {
                        "memory_id": memory_record.memory_id,
                        "content": memory_record.content,
                        "memory_type": memory_record.memory_type,
                        "context": memory_record.context or {},
                        "importance": memory_record.importance,
                        "emotional_valence": memory_record.emotional_valence,
                        "tags": memory_record.tags or [],
                        "created_at": memory_record.created_at,
                        "last_accessed": memory_record.last_accessed,
                        "access_count": memory_record.access_count,
                        "session_id": memory_record.session_id,
                        "expires_at": memory_record.expires_at
                    }
                    memories.append(memory_data)

                logger.info(
                    f"Loaded {len(memories)} memories from database for agent {agent_id}",
                    LogCategory.MEMORY_OPERATIONS,
                    "app.services.autonomous_persistence",
                    data={"agent_id": agent_id, "memory_count": len(memories)}
                )
                return memories

        except Exception as e:
            logger.error(
                f"Failed to load agent memories for {agent_id}",
                LogCategory.MEMORY_OPERATIONS,
                "app.services.autonomous_persistence",
                data={"agent_id": agent_id},
                error=e
            )
            return []

    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent state from persistence storage."""
        try:
            from app.models.database.base import get_database_session
            from app.models.autonomous import AutonomousAgentState
            from sqlalchemy import select

            # Convert agent_id to UUID if needed
            import uuid as uuid_module
            if isinstance(agent_id, str):
                try:
                    agent_uuid = uuid_module.UUID(agent_id)
                except ValueError:
                    logger.warn(
                        f"Invalid agent_id format: {agent_id}",
                        LogCategory.AGENT_OPERATIONS,
                        "app.services.autonomous_persistence",
                        data={"agent_id": agent_id}
                    )
                    return None
            else:
                agent_uuid = agent_id

            async for session in get_database_session():
                # Get agent state
                agent_state = await session.execute(
                    select(AutonomousAgentState).where(AutonomousAgentState.agent_id == agent_uuid)
                )
                agent_state_record = agent_state.scalar_one_or_none()

                if not agent_state_record:
                    logger.debug(
                        f"No agent state found for {agent_uuid}",
                        LogCategory.AGENT_OPERATIONS,
                        "app.services.autonomous_persistence",
                        data={"agent_uuid": str(agent_uuid)}
                    )
                    return None

                # Convert to dictionary
                state_data = {
                    "agent_id": str(agent_state_record.agent_id),
                    "session_id": agent_state_record.session_id,
                    "autonomy_level": agent_state_record.autonomy_level,
                    "decision_confidence": agent_state_record.decision_confidence,
                    "learning_enabled": agent_state_record.learning_enabled,
                    "current_task": agent_state_record.current_task,
                    "tools_available": agent_state_record.tools_available or [],
                    "outputs": agent_state_record.outputs or {},
                    "errors": agent_state_record.errors or [],
                    "iteration_count": agent_state_record.iteration_count,
                    "max_iterations": agent_state_record.max_iterations,
                    "custom_state": agent_state_record.custom_state or {},
                    "goal_stack": agent_state_record.goal_stack or [],
                    "context_memory": agent_state_record.context_memory or {},
                    "performance_metrics": agent_state_record.performance_metrics or {},
                    "self_initiated_tasks": agent_state_record.self_initiated_tasks or [],
                    "decision_history": agent_state_record.decision_history or [],
                    "adaptation_data": agent_state_record.adaptation_data or {}
                }

                logger.info(
                    f"Loaded agent state from database for agent {agent_id}",
                    LogCategory.AGENT_OPERATIONS,
                    "app.services.autonomous_persistence",
                    data={"agent_id": agent_id}
                )
                return state_data

        except Exception as e:
            logger.error(
                f"Failed to load agent state for {agent_id}",
                LogCategory.AGENT_OPERATIONS,
                "app.services.autonomous_persistence",
                data={"agent_id": agent_id},
                error=e
            )
            return None
    
    # Private methods for data persistence
    async def _load_all_data(self) -> None:
        """Load all data from files."""
        await asyncio.gather(
            self._load_goals(),
            self._load_decisions(),
            self._load_learning()
        )
    
    async def _load_goals(self) -> None:
        """Load goals from file."""
        if self.goals_file.exists():
            try:
                with open(self.goals_file, 'r') as f:
                    data = json.load(f)
                
                for agent_id, goals_data in data.items():
                    self.goals_cache[agent_id] = {}
                    for goal_data in goals_data:
                        # Convert datetime strings back to datetime objects
                        goal_data['created_at'] = datetime.fromisoformat(goal_data['created_at'])
                        if goal_data.get('target_completion'):
                            goal_data['target_completion'] = datetime.fromisoformat(goal_data['target_completion'])
                        
                        goal = AutonomousGoal(**goal_data)
                        self.goals_cache[agent_id][goal.goal_id] = goal
                        
            except Exception as e:
                logger.error(
                    "Failed to load goals",
                    LogCategory.AGENT_OPERATIONS,
                    "app.services.autonomous_persistence",
                    error=e
                )
    
    async def _save_goals(self) -> None:
        """Save goals to file."""
        try:
            data = {}
            for agent_id, goals in self.goals_cache.items():
                data[agent_id] = []
                for goal in goals.values():
                    goal_dict = asdict(goal)
                    # Convert datetime objects to strings
                    goal_dict['created_at'] = goal.created_at.isoformat()
                    if goal.target_completion:
                        goal_dict['target_completion'] = goal.target_completion.isoformat()
                    data[agent_id].append(goal_dict)
            
            with open(self.goals_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(
                "Failed to save goals",
                LogCategory.AGENT_OPERATIONS,
                "app.services.autonomous_persistence",
                error=e
            )
    
    async def _load_decisions(self) -> None:
        """Load decisions from file."""
        if self.decisions_file.exists():
            try:
                with open(self.decisions_file, 'r') as f:
                    data = json.load(f)
                
                for agent_id, decisions_data in data.items():
                    self.decisions_cache[agent_id] = []
                    for decision_data in decisions_data:
                        decision_data['timestamp'] = datetime.fromisoformat(decision_data['timestamp'])
                        decision = DecisionRecord(**decision_data)
                        self.decisions_cache[agent_id].append(decision)
                        
            except Exception as e:
                logger.error(
                    "Failed to load decisions",
                    LogCategory.AGENT_OPERATIONS,
                    "app.services.autonomous_persistence",
                    error=e
                )
    
    async def _save_decisions(self) -> None:
        """Save decisions to file."""
        try:
            data = {}
            for agent_id, decisions in self.decisions_cache.items():
                data[agent_id] = []
                for decision in decisions:
                    decision_dict = asdict(decision)
                    decision_dict['timestamp'] = decision.timestamp.isoformat()
                    data[agent_id].append(decision_dict)
            
            with open(self.decisions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(
                "Failed to save decisions",
                LogCategory.AGENT_OPERATIONS,
                "app.services.autonomous_persistence",
                error=e
            )
    
    async def _load_learning(self) -> None:
        """Load learning data from file."""
        if self.learning_file.exists():
            try:
                with open(self.learning_file, 'r') as f:
                    data = json.load(f)
                
                for agent_id, learning_data in data.items():
                    self.learning_cache[agent_id] = []
                    for learning_item in learning_data:
                        learning_item['timestamp'] = datetime.fromisoformat(learning_item['timestamp'])
                        learning = LearningData(**learning_item)
                        self.learning_cache[agent_id].append(learning)
                        
            except Exception as e:
                logger.error(
                    "Failed to load learning data",
                    LogCategory.AGENT_OPERATIONS,
                    "app.services.autonomous_persistence",
                    error=e
                )
    
    async def _save_learning(self) -> None:
        """Save learning data to file."""
        try:
            data = {}
            for agent_id, learning_data in self.learning_cache.items():
                data[agent_id] = []
                for learning in learning_data:
                    learning_dict = asdict(learning)
                    learning_dict['timestamp'] = learning.timestamp.isoformat()
                    data[agent_id].append(learning_dict)
            
            with open(self.learning_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(
                "Failed to save learning data",
                LogCategory.AGENT_OPERATIONS,
                "app.services.autonomous_persistence",
                error=e
            )


# Global instance
_autonomous_persistence_service = None

def get_autonomous_persistence_service() -> AutonomousPersistenceService:
    """Get the global autonomous persistence service instance."""
    global _autonomous_persistence_service
    if _autonomous_persistence_service is None:
        _autonomous_persistence_service = AutonomousPersistenceService()
    return _autonomous_persistence_service

# Create the global instance that other modules expect
autonomous_persistence = get_autonomous_persistence_service()
