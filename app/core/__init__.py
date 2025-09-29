"""
Core utilities and components for the Agentic AI Microservice.

THE UNIFIED MULTI-AGENT SYSTEM - ALL PHASES COMPLETE:
✅ PHASE 1: Foundation (UnifiedRAGSystem, CollectionBasedKBManager, AgentIsolationManager)
✅ PHASE 2: Memory & Tools (UnifiedMemorySystem, UnifiedToolRepository)
✅ PHASE 3: Communication (AgentCommunicationSystem)
✅ PHASE 4: Optimization (PerformanceOptimizer)

SINGLE ENTRY POINT: UnifiedSystemOrchestrator
"""

from .unified_system_orchestrator import UnifiedSystemOrchestrator, SystemConfig, SystemStatus

__all__ = [
    "UnifiedSystemOrchestrator",
    "SystemConfig",
    "SystemStatus"
]
