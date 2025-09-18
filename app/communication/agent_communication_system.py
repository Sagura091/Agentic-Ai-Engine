"""
Agent Communication System.

This module provides the core communication infrastructure for multi-agent
systems, enabling secure, efficient, and intelligent communication between
agents with proper access controls and message routing.

Features:
- Direct and broadcast messaging
- Message queuing and delivery
- Communication channels and routing
- Access control and security
- Message persistence and history
- Performance optimization
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

import structlog
from pydantic import BaseModel, Field

from app.rag.core.agent_isolation_manager import AgentIsolationManager, ResourceType
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """Types of messages in the communication system."""
    DIRECT = "direct"                 # Direct agent-to-agent message
    BROADCAST = "broadcast"           # Message to all agents
    MULTICAST = "multicast"           # Message to specific group
    KNOWLEDGE_SHARE = "knowledge_share"  # Knowledge sharing request
    TASK_COORDINATION = "task_coordination"  # Task collaboration
    SYSTEM = "system"                 # System notifications
    EMERGENCY = "emergency"           # Emergency communications


class MessagePriority(str, Enum):
    """Priority levels for messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class MessageStatus(str, Enum):
    """Status of messages."""
    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Message:
    """Communication message structure."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    recipient_group: Optional[str] = None  # For multicast
    
    # Message content
    content: str = ""
    message_type: MessageType = MessageType.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Metadata
    subject: str = ""
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    # Status
    status: MessageStatus = MessageStatus.PENDING
    delivery_attempts: int = 0
    max_delivery_attempts: int = 3
    
    # Security
    requires_acknowledgment: bool = False
    is_encrypted: bool = False
    access_level: str = "normal"


@dataclass
class CommunicationChannel:
    """Communication channel for organizing messages."""
    channel_id: str
    name: str
    description: str
    participants: Set[str] = field(default_factory=set)
    admins: Set[str] = field(default_factory=set)
    is_public: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    message_history: List[str] = field(default_factory=list)  # message_ids


@dataclass
class AgentCommunicationProfile:
    """Communication profile for an agent."""
    agent_id: str
    display_name: str = ""
    status: str = "online"  # online, offline, busy, away
    
    # Communication preferences
    max_messages_per_hour: int = 100
    auto_respond: bool = False
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    
    # Statistics
    messages_sent: int = 0
    messages_received: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Channels and groups
    subscribed_channels: Set[str] = field(default_factory=set)
    blocked_agents: Set[str] = field(default_factory=set)


class CommunicationConfig(BaseModel):
    """Configuration for the communication system."""
    # Message limits
    max_message_size: int = Field(default=10000, description="Maximum message size in characters")
    max_queue_size: int = Field(default=1000, description="Maximum messages in queue per agent")
    
    # Delivery settings
    delivery_timeout: int = Field(default=30, description="Message delivery timeout in seconds")
    retry_interval: int = Field(default=5, description="Retry interval in seconds")
    
    # Persistence settings
    enable_message_history: bool = Field(default=True, description="Enable message history")
    history_retention_days: int = Field(default=30, description="Message history retention in days")
    
    # Performance settings
    enable_message_compression: bool = Field(default=True, description="Enable message compression")
    batch_delivery: bool = Field(default=True, description="Enable batch message delivery")
    
    # Security settings
    enable_encryption: bool = Field(default=False, description="Enable message encryption")
    require_authentication: bool = Field(default=True, description="Require agent authentication")


class AgentCommunicationSystem:
    """
    Agent Communication System.
    
    Provides comprehensive communication infrastructure for multi-agent
    systems with security, efficiency, and scalability.
    """
    
    def __init__(
        self,
        isolation_manager: AgentIsolationManager,
        config: Optional[CommunicationConfig] = None
    ):
        """Initialize the agent communication system."""
        self.isolation_manager = isolation_manager
        self.config = config or CommunicationConfig()
        
        # Agent management
        self.agent_profiles: Dict[str, AgentCommunicationProfile] = {}
        
        # Message management
        self.messages: Dict[str, Message] = {}  # message_id -> message
        self.agent_inboxes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_queue_size))
        self.agent_outboxes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_queue_size))
        
        # Channel management
        self.channels: Dict[str, CommunicationChannel] = {}
        
        # Message routing
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.stats = {
            "total_messages": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "active_agents": 0,
            "active_channels": 0
        }
        
        self.is_initialized = False
        logger.info("Agent communication system created", config=self.config.dict())
    
    async def initialize(self) -> None:
        """Initialize the communication system."""
        try:
            if self.is_initialized:
                logger.warning("Communication system already initialized")
                return
            
            logger.info("Initializing agent communication system...")
            
            # Start background tasks
            asyncio.create_task(self._message_delivery_worker())
            asyncio.create_task(self._cleanup_expired_messages())
            
            self.is_initialized = True
            logger.info("Agent communication system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize communication system: {str(e)}")
            raise
    
    async def register_agent(
        self,
        agent_id: str,
        display_name: Optional[str] = None
    ) -> AgentCommunicationProfile:
        """
        Register an agent for communication.
        
        Args:
            agent_id: Agent identifier
            display_name: Display name for the agent
            
        Returns:
            Agent communication profile
        """
        try:
            if agent_id in self.agent_profiles:
                logger.warning(f"Agent {agent_id} already registered for communication")
                return self.agent_profiles[agent_id]
            
            # Create communication profile
            profile = AgentCommunicationProfile(
                agent_id=agent_id,
                display_name=display_name or agent_id,
                notification_preferences={
                    "direct_messages": True,
                    "broadcast_messages": True,
                    "knowledge_shares": True,
                    "task_coordination": True
                }
            )
            
            self.agent_profiles[agent_id] = profile
            self.stats["active_agents"] += 1
            
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
        recipient_group: Optional[str] = None,
        message_type: MessageType = MessageType.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL,
        subject: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        expires_in_hours: Optional[int] = None
    ) -> str:
        """
        Send a message to another agent or group.
        
        Args:
            sender_id: Sending agent ID
            content: Message content
            recipient_id: Receiving agent ID (for direct messages)
            recipient_group: Group ID (for multicast)
            message_type: Type of message
            priority: Message priority
            subject: Message subject
            metadata: Additional metadata
            expires_in_hours: Message expiration time
            
        Returns:
            Message ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Validate sender
            if sender_id not in self.agent_profiles:
                await self.register_agent(sender_id)
            
            # Check rate limits
            if not await self._check_rate_limit(sender_id):
                raise ValueError(f"Agent {sender_id} has exceeded message rate limit")
            
            # Validate message size
            if len(content) > self.config.max_message_size:
                raise ValueError(f"Message exceeds maximum size of {self.config.max_message_size} characters")
            
            # Create message
            message = Message(
                sender_id=sender_id,
                recipient_id=recipient_id,
                recipient_group=recipient_group,
                content=content,
                message_type=message_type,
                priority=priority,
                subject=subject,
                metadata=metadata or {}
            )
            
            # Set expiration
            if expires_in_hours:
                message.expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            # Store message
            self.messages[message.message_id] = message
            
            # Queue for delivery
            await self._queue_message_for_delivery(message)
            
            # Update statistics
            self.agent_profiles[sender_id].messages_sent += 1
            self.agent_profiles[sender_id].last_activity = datetime.utcnow()
            self.stats["total_messages"] += 1
            
            logger.info(f"Message queued for delivery: {message.message_id}")
            return message.message_id
            
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            raise
    
    async def get_messages(
        self,
        agent_id: str,
        message_type: Optional[MessageType] = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Message]:
        """
        Get messages for an agent.
        
        Args:
            agent_id: Agent ID
            message_type: Filter by message type
            unread_only: Only return unread messages
            limit: Maximum number of messages
            
        Returns:
            List of messages
        """
        try:
            if agent_id not in self.agent_profiles:
                return []
            
            inbox = self.agent_inboxes[agent_id]
            messages = []
            
            for message_id in list(inbox):
                if message_id in self.messages:
                    message = self.messages[message_id]
                    
                    # Apply filters
                    if message_type and message.message_type != message_type:
                        continue
                    
                    if unread_only and message.status == MessageStatus.READ:
                        continue
                    
                    messages.append(message)
                    
                    if len(messages) >= limit:
                        break
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages for agent {agent_id}: {str(e)}")
            return []
    
    async def mark_message_read(self, agent_id: str, message_id: str) -> bool:
        """Mark a message as read."""
        try:
            if message_id not in self.messages:
                return False
            
            message = self.messages[message_id]
            
            # Verify agent can read this message
            if (message.recipient_id != agent_id and 
                agent_id not in self._get_message_recipients(message)):
                return False
            
            message.status = MessageStatus.READ
            message.read_at = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark message as read: {str(e)}")
            return False
    
    async def create_channel(
        self,
        creator_id: str,
        name: str,
        description: str = "",
        is_public: bool = False
    ) -> str:
        """
        Create a communication channel.
        
        Args:
            creator_id: Agent creating the channel
            name: Channel name
            description: Channel description
            is_public: Whether the channel is public
            
        Returns:
            Channel ID
        """
        try:
            channel_id = f"channel_{name.lower().replace(' ', '_')}_{creator_id}"
            
            if channel_id in self.channels:
                raise ValueError(f"Channel '{name}' already exists")
            
            # Create channel
            channel = CommunicationChannel(
                channel_id=channel_id,
                name=name,
                description=description,
                participants={creator_id},
                admins={creator_id},
                is_public=is_public
            )
            
            self.channels[channel_id] = channel
            self.stats["active_channels"] += 1
            
            # Subscribe creator to channel
            if creator_id in self.agent_profiles:
                self.agent_profiles[creator_id].subscribed_channels.add(channel_id)
            
            logger.info(f"Created channel: {channel_id}")
            return channel_id
            
        except Exception as e:
            logger.error(f"Failed to create channel: {str(e)}")
            raise
    
    async def _queue_message_for_delivery(self, message: Message) -> None:
        """Queue a message for delivery."""
        try:
            recipients = self._get_message_recipients(message)
            
            for recipient_id in recipients:
                # Check if recipient exists and can receive messages
                if recipient_id in self.agent_profiles:
                    # Check access permissions
                    if await self.isolation_manager.validate_access(
                        message.sender_id,
                        recipient_id,
                        ResourceType.COMMUNICATION,
                        "write"
                    ):
                        self.agent_inboxes[recipient_id].append(message.message_id)
                        logger.debug(f"Queued message {message.message_id} for {recipient_id}")
                    else:
                        logger.warning(f"Access denied: {message.sender_id} -> {recipient_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue message for delivery: {str(e)}")
    
    def _get_message_recipients(self, message: Message) -> Set[str]:
        """Get all recipients for a message."""
        recipients = set()
        
        if message.message_type == MessageType.DIRECT and message.recipient_id:
            recipients.add(message.recipient_id)
        
        elif message.message_type == MessageType.BROADCAST:
            recipients.update(self.agent_profiles.keys())
        
        elif message.message_type == MessageType.MULTICAST and message.recipient_group:
            # Get channel participants
            if message.recipient_group in self.channels:
                recipients.update(self.channels[message.recipient_group].participants)
        
        return recipients
    
    async def _check_rate_limit(self, agent_id: str) -> bool:
        """Check if agent is within rate limits."""
        try:
            if agent_id not in self.agent_profiles:
                return True
            
            profile = self.agent_profiles[agent_id]
            
            # Simple rate limiting - could be enhanced with sliding window
            current_hour = datetime.utcnow().hour
            if profile.last_activity.hour != current_hour:
                # Reset counter for new hour
                profile.messages_sent = 0
            
            return profile.messages_sent < profile.max_messages_per_hour
            
        except Exception as e:
            logger.error(f"Failed to check rate limit: {str(e)}")
            return True
    
    async def _message_delivery_worker(self) -> None:
        """Background worker for message delivery."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Process pending messages
                for message_id, message in list(self.messages.items()):
                    if message.status == MessageStatus.PENDING:
                        # Check if message has expired
                        if message.expires_at and message.expires_at < datetime.utcnow():
                            message.status = MessageStatus.EXPIRED
                            continue
                        
                        # Attempt delivery
                        if message.delivery_attempts < message.max_delivery_attempts:
                            message.status = MessageStatus.DELIVERED
                            message.delivered_at = datetime.utcnow()
                            self.stats["messages_delivered"] += 1
                        else:
                            message.status = MessageStatus.FAILED
                            self.stats["messages_failed"] += 1
                
            except Exception as e:
                logger.error(f"Message delivery worker error: {str(e)}")
    
    async def _cleanup_expired_messages(self) -> None:
        """Background cleanup of expired messages."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                cutoff_date = datetime.utcnow() - timedelta(days=self.config.history_retention_days)
                expired_messages = []
                
                for message_id, message in self.messages.items():
                    if message.created_at < cutoff_date:
                        expired_messages.append(message_id)
                
                for message_id in expired_messages:
                    del self.messages[message_id]
                
                if expired_messages:
                    logger.info(f"Cleaned up {len(expired_messages)} expired messages")
                
            except Exception as e:
                logger.error(f"Message cleanup error: {str(e)}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get communication system statistics."""
        return {
            **self.stats,
            "total_agents": len(self.agent_profiles),
            "total_channels": len(self.channels),
            "pending_messages": sum(1 for m in self.messages.values() if m.status == MessageStatus.PENDING)
        }
