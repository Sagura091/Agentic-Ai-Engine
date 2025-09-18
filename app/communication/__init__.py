"""
Agent Communication System - THE Communication Hub for Multi-Agent Architecture.

This is THE ONLY communication system in the entire application.
All agent-to-agent communication flows through this unified hub.

CORE ARCHITECTURE:
✅ AgentCommunicationSystem - THE communication hub
✅ Optional communication (disabled by default)
✅ Knowledge sharing protocols
✅ Memory sharing capabilities
✅ Case-by-case collaboration

DESIGN PRINCIPLES:
- Agents are isolated by default
- Communication is optional and explicit
- Simple, clean, fast messaging
- No complexity unless absolutely necessary

PHASE 3 COMPLETE:
✅ Agent communication layer
✅ Knowledge sharing protocols
✅ Collaboration mechanisms
"""

from .agent_communication_system import (
    AgentCommunicationSystem,
    Message,
    MessageType,
    MessagePriority,
    MessageStatus,
    AgentCommunicationProfile
)

# Optional components (will be imported if available)
try:
    from .knowledge_sharing_protocols import KnowledgeSharingProtocol
except ImportError:
    KnowledgeSharingProtocol = None

try:
    from .collaboration_manager import CollaborationManager
except ImportError:
    CollaborationManager = None

__all__ = [
    # THE Communication System
    "AgentCommunicationSystem",
    "Message",
    "MessageType",
    "MessagePriority",
    "MessageStatus",
    "AgentCommunicationProfile"
]

# Add optional components if available
if KnowledgeSharingProtocol:
    __all__.append("KnowledgeSharingProtocol")
if CollaborationManager:
    __all__.append("CollaborationManager")

__version__ = "2.0.0"  # Updated for unified system
__description__ = "THE Communication Hub for Multi-Agent Architecture"
