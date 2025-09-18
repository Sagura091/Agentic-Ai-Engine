"""
Agent Communication System for Multi-Agent Architecture.

This module provides comprehensive communication capabilities for agents
including direct messaging, broadcast communication, knowledge sharing,
and collaborative workflows.

Features:
- Direct agent-to-agent communication
- Broadcast and multicast messaging
- Knowledge sharing protocols
- Collaborative task coordination
- Message routing and delivery
- Communication security and access control
"""

from .agent_communication_system import (
    AgentCommunicationSystem,
    CommunicationConfig,
    Message,
    MessageType,
    MessagePriority,
    CommunicationChannel,
    AgentCommunicationProfile
)

from .knowledge_sharing_protocols import (
    KnowledgeSharingProtocol,
    SharingRequest,
    SharingResponse,
    SharingPermission,
    KnowledgeShareType
)

from .collaboration_manager import (
    CollaborationManager,
    CollaborationSession,
    CollaborationTask,
    CollaborationRole,
    TaskStatus
)

__all__ = [
    # Core communication
    "AgentCommunicationSystem",
    "CommunicationConfig",
    "Message",
    "MessageType",
    "MessagePriority",
    "CommunicationChannel",
    "AgentCommunicationProfile",
    
    # Knowledge sharing
    "KnowledgeSharingProtocol",
    "SharingRequest",
    "SharingResponse",
    "SharingPermission",
    "KnowledgeShareType",
    
    # Collaboration
    "CollaborationManager",
    "CollaborationSession",
    "CollaborationTask",
    "CollaborationRole",
    "TaskStatus"
]

__version__ = "1.0.0"
__author__ = "Multi-Agent Team"
__description__ = "Comprehensive agent communication and collaboration system"
