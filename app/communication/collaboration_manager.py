"""
Collaboration Manager - Streamlined Foundation.

This module provides simple collaboration capabilities for agents
including basic task coordination and role management.

Features:
- Simple task coordination
- Basic role management
- Progress tracking
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import structlog

from .agent_communication_system import AgentCommunicationSystem, MessageType

logger = structlog.get_logger(__name__)


class TaskStatus(str, Enum):
    """Status of collaboration tasks."""
    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CollaborationRole(str, Enum):
    """Roles in collaboration sessions."""
    COORDINATOR = "coordinator"       # Manages the collaboration
    CONTRIBUTOR = "contributor"       # Contributes to tasks
    OBSERVER = "observer"             # Observes but doesn't actively participate


class SessionStatus(str, Enum):
    """Status of collaboration sessions."""
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class CollaborationTask:
    """Simple collaboration task."""
    task_id: str
    session_id: str
    title: str
    description: str
    assigned_to: Optional[str]
    created_by: str
    status: TaskStatus
    created_at: datetime

    @classmethod
    def create(
        cls,
        session_id: str,
        title: str,
        description: str,
        created_by: str,
        assigned_to: Optional[str] = None
    ) -> "CollaborationTask":
        """Create a new collaboration task."""
        return cls(
            task_id=str(uuid.uuid4()),
            session_id=session_id,
            title=title,
            description=description,
            assigned_to=assigned_to,
            created_by=created_by,
            status=TaskStatus.CREATED,
            created_at=datetime.now()
        )


@dataclass
class CollaborationSession:
    """Simple collaboration session managing multiple agents and tasks."""
    session_id: str
    name: str
    description: str
    coordinator_id: str
    participants: List[str]
    status: SessionStatus
    created_at: datetime

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        coordinator_id: str
    ) -> "CollaborationSession":
        """Create a new collaboration session."""
        return cls(
            session_id=str(uuid.uuid4()),
            name=name,
            description=description,
            coordinator_id=coordinator_id,
            participants=[coordinator_id],
            status=SessionStatus.ACTIVE,
            created_at=datetime.now()
        )


class CollaborationManager:
    """
    Collaboration Manager - Streamlined Foundation.

    Manages simple collaborative workflows and task coordination.
    """

    def __init__(self, communication_system: AgentCommunicationSystem):
        """Initialize the collaboration manager."""
        self.communication_system = communication_system
        self.is_initialized = False

        # Session management
        self.sessions: Dict[str, CollaborationSession] = {}
        self.tasks: Dict[str, CollaborationTask] = {}

        # Simple stats
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "completed_sessions": 0,
            "total_tasks": 0,
            "completed_tasks": 0
        }

        logger.info("Collaboration manager initialized")

    async def initialize(self) -> None:
        """Initialize the collaboration manager."""
        try:
            if self.is_initialized:
                return

            self.is_initialized = True
            logger.info("Collaboration manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize collaboration manager: {str(e)}")
            raise
    async def create_collaboration_session(
        self,
        coordinator_id: str,
        name: str,
        description: str = ""
    ) -> str:
        """
        Create a new collaboration session.

        Args:
            coordinator_id: Agent coordinating the collaboration
            name: Session name
            description: Session description

        Returns:
            Session ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Create session
            session = CollaborationSession.create(
                name=name,
                description=description,
                coordinator_id=coordinator_id
            )

            # Store session
            self.sessions[session.session_id] = session

            # Send notification to coordinator
            await self.communication_system.send_message(
                sender_id="system",
                recipient_id=coordinator_id,
                content=f"Collaboration session '{name}' created successfully",
                message_type=MessageType.SYSTEM
            )

            self.stats["total_sessions"] += 1
            self.stats["active_sessions"] += 1

            logger.info(f"Created collaboration session: {session.session_id}")
            return session.session_id

        except Exception as e:
            logger.error(f"Failed to create collaboration session: {str(e)}")
            raise
    async def add_participant(
        self,
        session_id: str,
        agent_id: str
    ) -> bool:
        """
        Add a participant to a collaboration session.

        Args:
            session_id: Session to add participant to
            agent_id: Agent to add

        Returns:
            True if successful
        """
        try:
            # Get session
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            session = self.sessions[session_id]

            # Add participant
            if agent_id not in session.participants:
                session.participants.append(agent_id)

            # Send notification
            await self.communication_system.send_message(
                sender_id="system",
                recipient_id=agent_id,
                content=f"You have been added to collaboration session: {session.name}",
                message_type=MessageType.SYSTEM
            )

            logger.info(f"Added participant {agent_id} to session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add participant: {str(e)}")
            return False
    async def create_task(
        self,
        session_id: str,
        title: str,
        description: str,
        created_by: str,
        assigned_to: Optional[str] = None
    ) -> str:
        """
        Create a task within a collaboration session.

        Args:
            session_id: Session to create task in
            title: Task title
            description: Task description
            created_by: Agent creating the task
            assigned_to: Agent to assign task to

        Returns:
            Task ID
        """
        try:
            # Get session
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")

            # Create task
            task = CollaborationTask.create(
                session_id=session_id,
                title=title,
                description=description,
                created_by=created_by,
                assigned_to=assigned_to
            )

            # Store task
            self.tasks[task.task_id] = task

            # Update assignment status
            if assigned_to:
                task.status = TaskStatus.ASSIGNED

                # Notify assigned agent
                await self.communication_system.send_message(
                    sender_id="system",
                    recipient_id=assigned_to,
                    content=f"You have been assigned task: {title}",
                    message_type=MessageType.SYSTEM
                )

            self.stats["total_tasks"] += 1

            logger.info(f"Created task {task.task_id} in session {session_id}")
            return task.task_id

        except Exception as e:
            logger.error(f"Failed to create task: {str(e)}")
            raise
    async def update_task_status(
        self,
        task_id: str,
        new_status: TaskStatus,
        updated_by: str
    ) -> bool:
        """
        Update the status of a task.

        Args:
            task_id: Task to update
            new_status: New status for the task
            updated_by: Agent updating the task

        Returns:
            True if successful
        """
        try:
            # Get task
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")

            task = self.tasks[task_id]

            # Update task
            old_status = task.status
            task.status = new_status

            # Update timing
            if new_status == TaskStatus.COMPLETED:
                self.stats["completed_tasks"] += 1

            logger.info(f"Updated task {task_id} status: {old_status} -> {new_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to update task status: {str(e)}")
            return False
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a collaboration session by ID."""
        return self.sessions.get(session_id)

    def get_task(self, task_id: str) -> Optional[CollaborationTask]:
        """Get a collaboration task by ID."""
        return self.tasks.get(task_id)

    def get_agent_sessions(self, agent_id: str) -> List[CollaborationSession]:
        """Get all sessions for an agent."""
        sessions = []
        for session in self.sessions.values():
            if agent_id in session.participants:
                sessions.append(session)
        return sessions

    def get_agent_tasks(self, agent_id: str) -> List[CollaborationTask]:
        """Get all tasks assigned to an agent."""
        tasks = []
        for task in self.tasks.values():
            if task.assigned_to == agent_id:
                tasks.append(task)
        return tasks

    def get_stats(self) -> Dict[str, Any]:
        """Get collaboration manager statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "total_stored_sessions": len(self.sessions),
            "total_stored_tasks": len(self.tasks)
        }
