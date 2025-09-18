"""
Collaboration Manager for Multi-Agent Systems.

This module provides comprehensive collaboration capabilities for agents
including task coordination, collaborative workflows, role management,
and progress tracking.

Features:
- Multi-agent task coordination
- Collaborative workflow management
- Role-based collaboration
- Progress tracking and reporting
- Resource sharing and allocation
- Conflict resolution
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .agent_communication_system import AgentCommunicationSystem, MessageType, MessagePriority
from app.rag.core.agent_isolation_manager import AgentIsolationManager, ResourceType

logger = structlog.get_logger(__name__)


class TaskStatus(str, Enum):
    """Status of collaboration tasks."""
    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CollaborationRole(str, Enum):
    """Roles in collaboration sessions."""
    COORDINATOR = "coordinator"       # Manages the collaboration
    CONTRIBUTOR = "contributor"       # Contributes to tasks
    REVIEWER = "reviewer"             # Reviews work and provides feedback
    OBSERVER = "observer"             # Observes but doesn't actively participate
    SPECIALIST = "specialist"         # Domain expert for specific areas


class SessionStatus(str, Enum):
    """Status of collaboration sessions."""
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class CollaborationTask:
    """Individual task within a collaboration."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    
    # Task details
    title: str = ""
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    
    # Assignment
    assigned_to: Optional[str] = None
    created_by: str = ""
    
    # Status and timing
    status: TaskStatus = TaskStatus.CREATED
    priority: int = 1  # 1-10, 10 being highest
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    
    # Dates
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    
    # Dependencies
    depends_on: Set[str] = field(default_factory=set)  # task_ids
    blocks: Set[str] = field(default_factory=set)      # task_ids
    
    # Progress and feedback
    progress_percentage: int = 0
    notes: List[str] = field(default_factory=list)
    feedback: List[str] = field(default_factory=list)
    
    # Resources
    required_resources: List[str] = field(default_factory=list)
    shared_resources: List[str] = field(default_factory=list)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationSession:
    """Collaboration session managing multiple agents and tasks."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Session details
    title: str = ""
    description: str = ""
    objectives: List[str] = field(default_factory=list)
    
    # Participants
    coordinator_id: str = ""
    participants: Dict[str, CollaborationRole] = field(default_factory=dict)  # agent_id -> role
    
    # Status and timing
    status: SessionStatus = SessionStatus.PLANNING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Tasks
    tasks: Dict[str, CollaborationTask] = field(default_factory=dict)  # task_id -> task
    
    # Communication
    communication_channel: Optional[str] = None
    meeting_schedule: List[datetime] = field(default_factory=list)
    
    # Progress tracking
    overall_progress: int = 0
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resources and constraints
    budget: Optional[float] = None
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborationManager:
    """
    Collaboration Manager for Multi-Agent Systems.
    
    Manages collaborative workflows, task coordination, and
    multi-agent project execution with role-based access control.
    """
    
    def __init__(
        self,
        communication_system: AgentCommunicationSystem,
        isolation_manager: AgentIsolationManager
    ):
        """Initialize the collaboration manager."""
        self.communication_system = communication_system
        self.isolation_manager = isolation_manager
        
        # Session management
        self.sessions: Dict[str, CollaborationSession] = {}
        
        # Agent participation tracking
        self.agent_sessions: Dict[str, Set[str]] = {}  # agent_id -> session_ids
        
        # Performance tracking
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "completed_sessions": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "total_participants": 0
        }
        
        logger.info("Collaboration manager initialized")
    
    async def create_collaboration_session(
        self,
        coordinator_id: str,
        title: str,
        description: str = "",
        objectives: Optional[List[str]] = None,
        deadline: Optional[datetime] = None,
        initial_participants: Optional[Dict[str, CollaborationRole]] = None
    ) -> str:
        """
        Create a new collaboration session.
        
        Args:
            coordinator_id: Agent coordinating the collaboration
            title: Session title
            description: Session description
            objectives: List of objectives
            deadline: Session deadline
            initial_participants: Initial participants and their roles
            
        Returns:
            Session ID
        """
        try:
            # Validate coordinator exists
            if not await self._validate_agent_exists(coordinator_id):
                raise ValueError(f"Coordinator agent {coordinator_id} not found")
            
            # Create session
            session = CollaborationSession(
                title=title,
                description=description,
                objectives=objectives or [],
                coordinator_id=coordinator_id,
                deadline=deadline
            )
            
            # Add coordinator as participant
            session.participants[coordinator_id] = CollaborationRole.COORDINATOR
            
            # Add initial participants
            if initial_participants:
                for agent_id, role in initial_participants.items():
                    if await self._validate_agent_exists(agent_id):
                        session.participants[agent_id] = role
                    else:
                        logger.warning(f"Agent {agent_id} not found, skipping")
            
            # Create communication channel
            channel_id = await self.communication_system.create_channel(
                creator_id=coordinator_id,
                name=f"Collaboration: {title}",
                description=f"Communication channel for collaboration session: {title}",
                is_public=False
            )
            session.communication_channel = channel_id
            
            # Store session
            self.sessions[session.session_id] = session
            
            # Update agent participation tracking
            for agent_id in session.participants:
                if agent_id not in self.agent_sessions:
                    self.agent_sessions[agent_id] = set()
                self.agent_sessions[agent_id].add(session.session_id)
            
            # Send notifications to participants
            await self._notify_participants(
                session,
                f"You have been invited to collaborate on: {title}",
                MessageType.TASK_COORDINATION
            )
            
            self.stats["total_sessions"] += 1
            
            logger.info(f"Created collaboration session: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            logger.error(f"Failed to create collaboration session: {str(e)}")
            raise
    
    async def add_participant(
        self,
        session_id: str,
        agent_id: str,
        role: CollaborationRole,
        added_by: str
    ) -> bool:
        """
        Add a participant to a collaboration session.
        
        Args:
            session_id: Session to add participant to
            agent_id: Agent to add
            role: Role for the agent
            added_by: Agent adding the participant
            
        Returns:
            True if successful
        """
        try:
            # Get session
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            # Check permissions
            if not await self._check_session_permission(added_by, session, "manage_participants"):
                raise PermissionError(f"Agent {added_by} cannot add participants to session {session_id}")
            
            # Validate new participant
            if not await self._validate_agent_exists(agent_id):
                raise ValueError(f"Agent {agent_id} not found")
            
            # Add participant
            session.participants[agent_id] = role
            
            # Update tracking
            if agent_id not in self.agent_sessions:
                self.agent_sessions[agent_id] = set()
            self.agent_sessions[agent_id].add(session_id)
            
            # Send notification
            await self.communication_system.send_message(
                sender_id="system",
                recipient_id=agent_id,
                content=f"You have been added to collaboration session: {session.title}",
                message_type=MessageType.TASK_COORDINATION,
                subject="Collaboration Invitation",
                metadata={
                    "session_id": session_id,
                    "role": role.value,
                    "added_by": added_by
                }
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
        assigned_to: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        due_date: Optional[datetime] = None,
        priority: int = 1,
        depends_on: Optional[Set[str]] = None
    ) -> str:
        """
        Create a task within a collaboration session.
        
        Args:
            session_id: Session to create task in
            title: Task title
            description: Task description
            created_by: Agent creating the task
            assigned_to: Agent to assign task to
            requirements: Task requirements
            due_date: Task due date
            priority: Task priority (1-10)
            depends_on: Task dependencies
            
        Returns:
            Task ID
        """
        try:
            # Get session
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            # Check permissions
            if not await self._check_session_permission(created_by, session, "create_tasks"):
                raise PermissionError(f"Agent {created_by} cannot create tasks in session {session_id}")
            
            # Validate assigned agent
            if assigned_to and assigned_to not in session.participants:
                raise ValueError(f"Agent {assigned_to} is not a participant in session {session_id}")
            
            # Create task
            task = CollaborationTask(
                session_id=session_id,
                title=title,
                description=description,
                created_by=created_by,
                assigned_to=assigned_to,
                requirements=requirements or [],
                due_date=due_date,
                priority=priority,
                depends_on=depends_on or set()
            )
            
            # Store task
            session.tasks[task.task_id] = task
            
            # Update assignment status
            if assigned_to:
                task.status = TaskStatus.ASSIGNED
                task.assigned_at = datetime.utcnow()
                
                # Notify assigned agent
                await self.communication_system.send_message(
                    sender_id="system",
                    recipient_id=assigned_to,
                    content=f"You have been assigned task: {title}",
                    message_type=MessageType.TASK_COORDINATION,
                    priority=MessagePriority.HIGH,
                    subject="Task Assignment",
                    metadata={
                        "session_id": session_id,
                        "task_id": task.task_id,
                        "due_date": due_date.isoformat() if due_date else None
                    }
                )
            
            # Update session progress
            await self._update_session_progress(session_id)
            
            self.stats["total_tasks"] += 1
            
            logger.info(f"Created task {task.task_id} in session {session_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Failed to create task: {str(e)}")
            raise
    
    async def update_task_status(
        self,
        session_id: str,
        task_id: str,
        new_status: TaskStatus,
        updated_by: str,
        progress_percentage: Optional[int] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update the status of a task.
        
        Args:
            session_id: Session containing the task
            task_id: Task to update
            new_status: New status for the task
            updated_by: Agent updating the task
            progress_percentage: Progress percentage (0-100)
            notes: Additional notes
            
        Returns:
            True if successful
        """
        try:
            # Get session and task
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            if task_id not in session.tasks:
                raise ValueError(f"Task {task_id} not found in session {session_id}")
            
            task = session.tasks[task_id]
            
            # Check permissions
            if not await self._check_task_permission(updated_by, session, task, "update"):
                raise PermissionError(f"Agent {updated_by} cannot update task {task_id}")
            
            # Update task
            old_status = task.status
            task.status = new_status
            
            if progress_percentage is not None:
                task.progress_percentage = max(0, min(100, progress_percentage))
            
            if notes:
                task.notes.append(f"{datetime.utcnow().isoformat()}: {notes}")
            
            # Update timing
            if new_status == TaskStatus.IN_PROGRESS and old_status != TaskStatus.IN_PROGRESS:
                task.started_at = datetime.utcnow()
            elif new_status == TaskStatus.COMPLETED:
                task.completed_at = datetime.utcnow()
                self.stats["completed_tasks"] += 1
            
            # Notify relevant participants
            await self._notify_task_update(session, task, updated_by, old_status, new_status)
            
            # Update session progress
            await self._update_session_progress(session_id)
            
            logger.info(f"Updated task {task_id} status: {old_status} -> {new_status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update task status: {str(e)}")
            return False
    
    async def start_session(self, session_id: str, started_by: str) -> bool:
        """Start a collaboration session."""
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            
            # Check permissions
            if not await self._check_session_permission(started_by, session, "manage_session"):
                raise PermissionError(f"Agent {started_by} cannot start session {session_id}")
            
            # Update session status
            session.status = SessionStatus.ACTIVE
            session.started_at = datetime.utcnow()
            
            # Notify participants
            await self._notify_participants(
                session,
                f"Collaboration session '{session.title}' has started",
                MessageType.TASK_COORDINATION
            )
            
            self.stats["active_sessions"] += 1
            
            logger.info(f"Started collaboration session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start session: {str(e)}")
            return False
    
    async def _update_session_progress(self, session_id: str) -> None:
        """Update overall session progress."""
        try:
            session = self.sessions[session_id]
            
            if not session.tasks:
                session.overall_progress = 0
                return
            
            # Calculate progress based on task completion
            total_tasks = len(session.tasks)
            completed_tasks = sum(1 for task in session.tasks.values() if task.status == TaskStatus.COMPLETED)
            
            session.overall_progress = int((completed_tasks / total_tasks) * 100)
            
            # Check if session is complete
            if session.overall_progress == 100 and session.status == SessionStatus.ACTIVE:
                session.status = SessionStatus.COMPLETED
                session.completed_at = datetime.utcnow()
                self.stats["completed_sessions"] += 1
                self.stats["active_sessions"] -= 1
                
                await self._notify_participants(
                    session,
                    f"Collaboration session '{session.title}' has been completed!",
                    MessageType.TASK_COORDINATION
                )
            
        except Exception as e:
            logger.error(f"Failed to update session progress: {str(e)}")
    
    async def _check_session_permission(self, agent_id: str, session: CollaborationSession, action: str) -> bool:
        """Check if agent has permission for session action."""
        try:
            if agent_id not in session.participants:
                return False
            
            role = session.participants[agent_id]
            
            # Coordinators can do everything
            if role == CollaborationRole.COORDINATOR:
                return True
            
            # Action-specific permissions
            if action == "create_tasks":
                return role in [CollaborationRole.CONTRIBUTOR, CollaborationRole.SPECIALIST]
            elif action == "manage_participants":
                return role == CollaborationRole.COORDINATOR
            elif action == "manage_session":
                return role == CollaborationRole.COORDINATOR
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check session permission: {str(e)}")
            return False
    
    async def _check_task_permission(self, agent_id: str, session: CollaborationSession, task: CollaborationTask, action: str) -> bool:
        """Check if agent has permission for task action."""
        try:
            # Task assignee can always update their own tasks
            if task.assigned_to == agent_id:
                return True
            
            # Check session-level permissions
            return await self._check_session_permission(agent_id, session, "create_tasks")
            
        except Exception as e:
            logger.error(f"Failed to check task permission: {str(e)}")
            return False
    
    async def _notify_participants(self, session: CollaborationSession, message: str, message_type: MessageType) -> None:
        """Notify all session participants."""
        try:
            for participant_id in session.participants:
                await self.communication_system.send_message(
                    sender_id="system",
                    recipient_id=participant_id,
                    content=message,
                    message_type=message_type,
                    subject=f"Collaboration: {session.title}",
                    metadata={"session_id": session.session_id}
                )
                
        except Exception as e:
            logger.error(f"Failed to notify participants: {str(e)}")
    
    async def _notify_task_update(self, session: CollaborationSession, task: CollaborationTask, updated_by: str, old_status: TaskStatus, new_status: TaskStatus) -> None:
        """Notify relevant participants about task updates."""
        try:
            message = f"Task '{task.title}' status changed from {old_status.value} to {new_status.value}"
            
            # Notify coordinator and task assignee
            recipients = {session.coordinator_id}
            if task.assigned_to:
                recipients.add(task.assigned_to)
            
            # Remove the updater from notifications
            recipients.discard(updated_by)
            
            for recipient_id in recipients:
                await self.communication_system.send_message(
                    sender_id="system",
                    recipient_id=recipient_id,
                    content=message,
                    message_type=MessageType.TASK_COORDINATION,
                    subject=f"Task Update: {task.title}",
                    metadata={
                        "session_id": session.session_id,
                        "task_id": task.task_id,
                        "old_status": old_status.value,
                        "new_status": new_status.value,
                        "updated_by": updated_by
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to notify task update: {str(e)}")
    
    async def _validate_agent_exists(self, agent_id: str) -> bool:
        """Validate that an agent exists in the system."""
        try:
            profile = self.isolation_manager.get_agent_profile(agent_id)
            return profile is not None
            
        except Exception as e:
            logger.error(f"Failed to validate agent existence: {str(e)}")
            return False
    
    def get_agent_sessions(self, agent_id: str) -> List[CollaborationSession]:
        """Get all sessions for an agent."""
        sessions = []
        session_ids = self.agent_sessions.get(agent_id, set())
        
        for session_id in session_ids:
            if session_id in self.sessions:
                sessions.append(self.sessions[session_id])
        
        return sessions
    
    def get_session_tasks(self, session_id: str, agent_id: Optional[str] = None) -> List[CollaborationTask]:
        """Get tasks for a session, optionally filtered by agent."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        tasks = list(session.tasks.values())
        
        if agent_id:
            tasks = [task for task in tasks if task.assigned_to == agent_id]
        
        return tasks
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration manager statistics."""
        return {
            **self.stats,
            "sessions_by_status": {
                status.value: sum(1 for s in self.sessions.values() if s.status == status)
                for status in SessionStatus
            },
            "tasks_by_status": {
                status.value: sum(1 for s in self.sessions.values() 
                                for t in s.tasks.values() if t.status == status)
                for status in TaskStatus
            }
        }
