"""
Agent Communication System - THE Communication Hub for Multi-Agent Architecture.

This is THE ONLY communication system in the entire application.
All agent-to-agent communication flows through this unified hub.

CORE ARCHITECTURE:
✅ AgentCommunicationSystem - THE communication hub
✅ Optional communication (disabled by default)
✅ Direct agent-to-agent messaging
✅ Broadcast messaging
✅ Message queuing and delivery

DESIGN PRINCIPLES:
- Agents are isolated by default
- Communication is optional and explicit
- Simple, clean, fast messaging
- No complexity unless absolutely necessary

IMPLEMENTATION STATUS:
✅ Agent communication layer - COMPLETE
✅ Message routing and delivery - COMPLETE
✅ Agent registration and profiles - COMPLETE
"""

from .agent_communication_system import (
    AgentCommunicationSystem,
    Message,
    MessageType,
    MessagePriority,
    MessageStatus,
    AgentCommunicationProfile
)

__all__ = [
    "AgentCommunicationSystem",
    "Message",
    "MessageType",
    "MessagePriority",
    "MessageStatus",
    "AgentCommunicationProfile"
]

__version__ = "1.0.0"
__description__ = "THE Communication Hub for Multi-Agent Architecture"
