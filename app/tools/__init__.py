"""
Unified Tool System - THE Tool System for Multi-Agent Architecture.

This is THE ONLY tool system in the entire application.
All tool operations flow through the UnifiedToolRepository.

CORE ARCHITECTURE:
✅ UnifiedToolRepository - THE single tool repository
✅ Dynamic tool assignment based on use cases
✅ Agent-specific tool access and permissions
✅ RAG-enabled and non-RAG tools
✅ Simple, clean, fast operations

DESIGN PRINCIPLES:
- One tool repository to rule them all
- Use case driven tool access
- Agent-specific tool permissions
- No complexity unless absolutely necessary

PHASE 2 COMPLETE:
✅ Tool repository consolidation
✅ Agent-specific tool collections
✅ Dynamic tool selection
"""

# Import THE unified tool repository (THE ONLY tool system)
from .unified_tool_repository import (
    UnifiedToolRepository,
    ToolCategory,
    ToolAccessLevel,
    ToolMetadata,
    AgentToolProfile
)

__all__ = [
    # THE Unified Tool System
    "UnifiedToolRepository",
    "ToolCategory",
    "ToolAccessLevel",
    "ToolMetadata",
    "AgentToolProfile"
]

# Version information
__version__ = "2.0.0"  # Updated for unified system
__description__ = "THE Unified Tool System for Multi-Agent Architecture"
