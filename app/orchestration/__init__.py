"""
Agent orchestration module.

This module contains legacy orchestration components. The main system
orchestration has been moved to the unified system orchestrator at
app/core/unified_system_orchestrator.py.

Remaining components:
- Subgraphs: LangGraph subgraph definitions for agent workflows
"""

# Import remaining orchestration components
from .subgraphs import (
    HierarchicalWorkflowOrchestrator,
    LangGraphSubgraph,
    ResearchTeamSubgraph,
    DocumentTeamSubgraph,
    AutonomyLevel,
    DecisionStrategy,
    CoordinationPattern,
    SubgraphState,
    HierarchicalWorkflowState,
    ResearchTeamState,
    DocumentTeamState,
    AutonomousDecisionState
)

__all__ = [
    # Main orchestrator
    "HierarchicalWorkflowOrchestrator",

    # Subgraph classes
    "LangGraphSubgraph",
    "ResearchTeamSubgraph",
    "DocumentTeamSubgraph",

    # Enums
    "AutonomyLevel",
    "DecisionStrategy",
    "CoordinationPattern",

    # State types
    "SubgraphState",
    "HierarchicalWorkflowState",
    "ResearchTeamState",
    "DocumentTeamState",
    "AutonomousDecisionState"
]
