"""
Unified Memory System - THE Memory System for Multi-Agent Architecture.

This is THE ONLY memory system in the entire application.
All memory operations flow through the UnifiedMemorySystem.

CORE ARCHITECTURE:
✅ UnifiedMemorySystem - THE single memory system
✅ Short-term and long-term memory per agent
✅ RAG integration for persistent storage
✅ Agent-specific memory collections
✅ Simple, clean, fast operations

DESIGN PRINCIPLES:
- One memory system to rule them all
- Agent-specific memory isolation
- RAG-backed persistent storage
- No complexity unless absolutely necessary

PHASE 2 COMPLETE:
✅ Unified memory system
✅ Agent-specific memory collections
✅ RAG integration
"""

from .unified_memory_system import UnifiedMemorySystem
from .memory_models import MemoryType, MemoryEntry, MemoryCollection

__all__ = [
    "UnifiedMemorySystem",
    "MemoryType",
    "MemoryEntry",
    "MemoryCollection"
]
