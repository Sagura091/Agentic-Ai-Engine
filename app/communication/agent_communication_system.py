"""
Agent Communication System - THE Communication Hub for Multi-Agent Architecture.

This is THE ONLY communication system in the entire application.
All agent-to-agent communication flows through this unified hub.

CORE ARCHITECTURE:
- Direct agent-to-agent messaging
- Knowledge sharing protocols
- Memory sharing capabilities
- Optional communication (case-by-case basis)

DESIGN PRINCIPLES:
- Agents are isolated by default
- Communication is optional and explicit
- Simple, clean, fast messaging
- No complexity unless absolutely necessary

PHASE 3 ENHANCEMENT:
✅ Integration with UnifiedRAGSystem
✅ Knowledge sharing protocols
✅ Memory sharing capabilities
✅ Collaboration mechanisms
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import structlog

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """Types of messages in the communication system - ENHANCED."""
    DIRECT = "direct"                 # Direct agent-to-agent message
    BROADCAST = "broadcast"           # Message to all agents
    SYSTEM = "system"                 # System notifications
    KNOWLEDGE_SHARE = "knowledge_share"  # NEW: Knowledge sharing request
    MEMORY_SHARE = "memory_share"     # NEW: Memory sharing request
    COLLABORATION = "collaboration"   # NEW: Collaboration request


class MessagePriority(str, Enum):
    """Priority levels for messages - SIMPLIFIED."""
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"                 # NEW: For critical communications


class MessageStatus(str, Enum):
    """Status of messages - ENHANCED."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    PROCESSED = "processed"           # NEW: Message was processed successfully


@dataclass
class Message:
    """Communication message structure - ENHANCED."""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    content: str
    message_type: MessageType
    priority: MessagePriority
    created_at: datetime
    status: MessageStatus = MessageStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)  # NEW: Additional data
    response_to: Optional[str] = None  # NEW: Response to message ID

    @classmethod
    def create(
        cls,
        sender_id: str,
        content: str,
        recipient_id: Optional[str] = None,
        message_type: MessageType = MessageType.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        response_to: Optional[str] = None
    ) -> "Message":
        """Create a new message."""
        return cls(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            message_type=message_type,
            priority=priority,
            created_at=datetime.now(),
            metadata=metadata or {},
            response_to=response_to
        )


@dataclass
class AgentCommunicationProfile:
    """Communication profile for an agent - ENHANCED."""
    agent_id: str
    message_queue: deque = field(default_factory=lambda: deque(maxlen=1000))
    allowed_senders: Set[str] = field(default_factory=set)
    communication_enabled: bool = True  # NEW: Can this agent communicate?
    knowledge_sharing_enabled: bool = False  # NEW: Can share knowledge?
    memory_sharing_enabled: bool = False  # NEW: Can share memory?
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        agent_id: str,
        communication_enabled: bool = True,
        knowledge_sharing_enabled: bool = False,
        memory_sharing_enabled: bool = False
    ) -> "AgentCommunicationProfile":
        """Create a new communication profile."""
        return cls(
            agent_id=agent_id,
            communication_enabled=communication_enabled,
            knowledge_sharing_enabled=knowledge_sharing_enabled,
            memory_sharing_enabled=memory_sharing_enabled
        )


class AgentCommunicationSystem:
    """
    Agent Communication System - THE Communication Hub.

    SIMPLIFIED ARCHITECTURE:
    - Optional communication (disabled by default)
    - Knowledge sharing protocols
    - Memory sharing capabilities
    - Case-by-case collaboration
    """

    def __init__(self, unified_rag=None, unified_memory=None, isolation_manager=None):
        """Initialize THE agent communication system."""
        self.unified_rag = unified_rag
        self.unified_memory = unified_memory
        self.isolation_manager = isolation_manager
        self.is_initialized = False

        # Agent management - ENHANCED
        self.agent_profiles: Dict[str, AgentCommunicationProfile] = {}

        # Message management
        self.messages: Dict[str, Message] = {}  # message_id -> message
        self.message_handlers: Dict[MessageType, Callable] = {}  # NEW: Message handlers

        # Simple stats
        self.stats = {
            "total_agents": 0,
            "total_messages": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "knowledge_shares": 0,
            "memory_shares": 0,
            "collaborations": 0
        }

        logger.info("THE Agent communication system created")

    async def initialize(self) -> None:
        """Initialize the communication system."""
        try:
            if self.is_initialized:
                return

            self.is_initialized = True
            logger.info("Agent communication system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize communication system: {str(e)}")
            raise

    async def register_agent(self, agent_id: str) -> AgentCommunicationProfile:
        """
        Register an agent for communication.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent communication profile
        """
        try:
            if agent_id in self.agent_profiles:
                logger.warning(f"Agent {agent_id} already registered for communication")
                return self.agent_profiles[agent_id]

            # Create communication profile
            profile = AgentCommunicationProfile.create(agent_id)
            self.agent_profiles[agent_id] = profile
            self.stats["total_agents"] += 1

            logger.info(f"Registered agent {agent_id} for communication")
            return profile

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {str(e)}")
            raise
    async def send_message(
        self,
        sender_id: str,
        content: str,
        recipient_id: Optional[str] = None,
        message_type: MessageType = MessageType.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """
        Send a message to another agent.

        Args:
            sender_id: Sending agent ID
            content: Message content
            recipient_id: Receiving agent ID (None for broadcast)
            message_type: Type of message
            priority: Message priority

        Returns:
            Message ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Ensure sender is registered
            if sender_id not in self.agent_profiles:
                await self.register_agent(sender_id)

            # Create message
            message = Message.create(
                sender_id=sender_id,
                content=content,
                recipient_id=recipient_id,
                message_type=message_type,
                priority=priority
            )

            # Store message
            self.messages[message.message_id] = message

            # Deliver message
            await self._deliver_message(message)

            # Update stats
            self.stats["total_messages"] += 1

            logger.info(f"Message sent: {message.message_id}")
            return message.message_id

        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            raise
    async def get_messages(
        self,
        agent_id: str,
        limit: int = 50
    ) -> List[Message]:
        """
        Get messages for an agent.

        Args:
            agent_id: Agent ID
            limit: Maximum number of messages

        Returns:
            List of messages
        """
        try:
            if agent_id not in self.agent_profiles:
                return []

            profile = self.agent_profiles[agent_id]
            messages = []

            # Get messages from queue (most recent first)
            for _ in range(min(limit, len(profile.message_queue))):
                if profile.message_queue:
                    message_id = profile.message_queue.popleft()
                    if message_id in self.messages:
                        messages.append(self.messages[message_id])

            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for agent {agent_id}: {str(e)}")
            return []
    async def _deliver_message(self, message: Message) -> None:
        """Deliver a message to recipients."""
        try:
            if message.message_type == MessageType.DIRECT and message.recipient_id:
                # Direct message
                if message.recipient_id in self.agent_profiles:
                    profile = self.agent_profiles[message.recipient_id]
                    profile.message_queue.append(message.message_id)
                    message.status = MessageStatus.DELIVERED
                    self.stats["messages_delivered"] += 1
                else:
                    message.status = MessageStatus.FAILED
                    self.stats["messages_failed"] += 1

            elif message.message_type == MessageType.BROADCAST:
                # Broadcast to all agents
                delivered = 0
                for agent_id in self.agent_profiles:
                    if agent_id != message.sender_id:  # Don't send to sender
                        profile = self.agent_profiles[agent_id]
                        profile.message_queue.append(message.message_id)
                        delivered += 1

                if delivered > 0:
                    message.status = MessageStatus.DELIVERED
                    self.stats["messages_delivered"] += 1
                else:
                    message.status = MessageStatus.FAILED
                    self.stats["messages_failed"] += 1

        except Exception as e:
            logger.error(f"Failed to deliver message: {str(e)}")
            message.status = MessageStatus.FAILED
            self.stats["messages_failed"] += 1

    def get_agent_profile(self, agent_id: str) -> Optional[AgentCommunicationProfile]:
        """Get agent communication profile."""
        return self.agent_profiles.get(agent_id)

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get a message by ID."""
        return self.messages.get(message_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get communication system statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "total_agents": len(self.agent_profiles),
            "total_stored_messages": len(self.messages)
        }
