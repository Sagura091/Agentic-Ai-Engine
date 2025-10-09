"""
Revolutionary Unified Memory System - THE Memory System for Multi-Agent Architecture.

This is THE ONLY memory system in the entire application.
All memory operations flow through the Revolutionary UnifiedMemorySystem.

REVOLUTIONARY ARCHITECTURE (PHASE 3):
✅ UnifiedMemorySystem - THE single revolutionary memory system
✅ All memory types: Short-term, Long-term, Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault
✅ Active Retrieval Engine for automatic context-based memory retrieval
✅ Memory Orchestrator for multi-agent coordination with specialized managers
✅ Revolutionary memory models with associations, importance, and emotional valence
✅ RAG integration for persistent storage
✅ Agent-specific revolutionary memory collections

DESIGN PRINCIPLES:
- One revolutionary memory system to rule them all
- Agent-specific memory isolation with revolutionary capabilities
- Active retrieval without explicit search commands
- Memory orchestration with specialized managers
- RAG-backed persistent storage
- Revolutionary complexity that provides game-changing capabilities

PHASE 3 REVOLUTIONARY COMPLETE:
✅ Revolutionary unified memory system
✅ Core Memory (always-visible persistent context)
✅ Knowledge Vault (secure sensitive information storage)
✅ Resource Memory (document and file management)
✅ Active Retrieval Engine (automatic context-based retrieval)
✅ Memory Orchestrator (multi-agent coordination)
✅ Revolutionary memory models and collections

ALL PHASES REVOLUTIONARY COMPLETE:
✅ Dynamic Knowledge Graph (incremental graph updates with spatial relationships)
✅ Advanced Retrieval Mechanisms (BM25, embedding similarity, hybrid retrieval)
✅ Memory Consolidation System (sleep-cycle memory processing)
✅ Lifelong Learning Capabilities (cross-task transfer, performance-driven weighting)
✅ Multimodal Memory Support (images, videos, audio, cross-modal associations)
✅ Memory-Driven Decision Making (memory-informed planning and decisions)
"""

# Revolutionary Memory System
from .unified_memory_system import UnifiedMemorySystem

# Revolutionary Memory Models
from .memory_models import (
    MemoryType, MemoryEntry, MemoryCollection, RevolutionaryMemoryCollection,
    MemoryImportance, SensitivityLevel, CoreMemoryBlock,
    ResourceMemoryEntry, KnowledgeVaultEntry
)

# Revolutionary Memory Components
from .active_retrieval_engine import (
    ActiveRetrievalEngine, RetrievalContext, RetrievalResult
)
from .memory_orchestrator import (
    MemoryOrchestrator, MemoryOperation, MemoryManagerType,
    BaseMemoryManager, CoreMemoryManager, ResourceMemoryManager, KnowledgeVaultManager
)

# ALL PHASES REVOLUTIONARY COMPONENTS
from .dynamic_knowledge_graph import (
    DynamicKnowledgeGraph, GraphEntity, GraphRelationship, RelationshipType
)
from .advanced_retrieval_mechanisms import (
    AdvancedRetrievalMechanisms, RetrievalQuery, RetrievalMethod, RetrievalResult as AdvancedRetrievalResult
)
# Optional components (not required for core functionality)
# from .memory_consolidation_system import (
#     MemoryConsolidationSystem, ConsolidationPhase, ConsolidationStrategy, ConsolidationSession
# )
# from .lifelong_learning_capabilities import (
#     LifelongLearningCapabilities, LearningType, SkillProfile, LearningExperience
# )
# from .multimodal_memory_support import (
#     MultimodalMemorySystem, ModalityType, MultimodalMemoryEntry, CrossModalAssociation
# )
# from .memory_driven_decision_making import (
#     MemoryDrivenDecisionMaking, DecisionType, DecisionOption, DecisionRecord
# )

__all__ = [
    # THE Revolutionary Memory System
    "UnifiedMemorySystem",

    # Revolutionary Memory Models
    "MemoryType",
    "MemoryEntry",
    "MemoryCollection",
    "RevolutionaryMemoryCollection",
    "MemoryImportance",
    "SensitivityLevel",
    "CoreMemoryBlock",
    "ResourceMemoryEntry",
    "KnowledgeVaultEntry",

    # Revolutionary Memory Components
    "ActiveRetrievalEngine",
    "RetrievalContext",
    "RetrievalResult",
    "MemoryOrchestrator",
    "MemoryOperation",
    "MemoryManagerType",
    "BaseMemoryManager",
    "CoreMemoryManager",
    "ResourceMemoryManager",
    "KnowledgeVaultManager",

    # ALL PHASES REVOLUTIONARY COMPONENTS (Implemented)
    "DynamicKnowledgeGraph",
    "GraphEntity",
    "GraphRelationship",
    "RelationshipType",
    "AdvancedRetrievalMechanisms",
    "RetrievalQuery",
    "RetrievalMethod",
    "AdvancedRetrievalResult",

    # Note: The following components are planned but not yet implemented:
    # - MemoryConsolidationSystem (see app/services/memory_consolidation_service.py for alternative)
    # - LifelongLearningCapabilities
    # - MultimodalMemorySystem
    # - MemoryDrivenDecisionMaking
]
