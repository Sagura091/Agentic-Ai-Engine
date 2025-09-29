"""
Autonomous Agent System for Revolutionary Agentic AI.

This module provides advanced autonomous agents with true agentic capabilities
including self-directed decision-making, adaptive learning, emergent intelligence,
and autonomous goal management.

Key Features:
- Autonomous Decision Making: Agents can make independent decisions based on context
- Adaptive Learning: Continuous learning from experience and adaptation of behavior
- Goal Management: Self-directed goal setting, planning, and pursuit
- Emergent Intelligence: Development of emergent behaviors and intelligence patterns
- Collaboration: Multi-agent coordination and knowledge sharing
- Safety & Ethics: Built-in safety constraints and ethical guidelines

Components:
- AutonomousLangGraphAgent: Main autonomous agent class
- AutonomousDecisionEngine: Advanced decision-making system
- AdaptiveLearningSystem: Continuous learning and adaptation
- AutonomousGoalManager: Goal setting and management
- CollaborationManager: Multi-agent collaboration
- PerformanceTracker: Performance monitoring and optimization
"""

from typing import Dict, List, Any, Optional

from .autonomous_agent import (
    AutonomousLangGraphAgent,
    AutonomousAgentConfig,
    AutonomousAgentState,
    AutonomousDecision,
    AutonomyLevel,
    DecisionConfidence,
    LearningMode
)

from .decision_engine import (
    AutonomousDecisionEngine,
    DecisionCriteria,
    DecisionOption
)

from .learning_system import (
    AdaptiveLearningSystem,
    LearningInsight,
    PerformancePattern,
    AdaptationSuggestion
)

from .goal_manager import (
    AutonomousGoalManager,
    AutonomousGoal,
    GoalPlan,
    GoalType,
    GoalPriority,
    GoalStatus,
    GoalMetrics
)

__all__ = [
    # Main agent classes
    "AutonomousLangGraphAgent",
    "AutonomousAgentConfig",
    "AutonomousAgentState",
    "AutonomousDecision",
    
    # Enums and types
    "AutonomyLevel",
    "DecisionConfidence", 
    "LearningMode",
    "GoalType",
    "GoalPriority",
    "GoalStatus",
    
    # Decision making
    "AutonomousDecisionEngine",
    "DecisionCriteria",
    "DecisionOption",
    
    # Learning system
    "AdaptiveLearningSystem",
    "LearningInsight",
    "PerformancePattern",
    "AdaptationSuggestion",
    
    # Goal management
    "AutonomousGoalManager",
    "AutonomousGoal",
    "GoalPlan",
    "GoalMetrics"
]

# Version information
__version__ = "1.0.0"
__author__ = "Agentic AI Team"
__description__ = "Revolutionary autonomous agent system with true agentic capabilities"

# Module metadata
AUTONOMOUS_AGENT_FEATURES = [
    "autonomous_decision_making",
    "adaptive_learning",
    "goal_management",
    "emergent_intelligence",
    "multi_agent_collaboration",
    "safety_constraints",
    "ethical_guidelines",
    "performance_optimization"
]

SUPPORTED_AUTONOMY_LEVELS = [
    "reactive",      # Responds to direct instructions only
    "proactive",     # Can initiate actions based on context
    "adaptive",      # Learns and adapts behavior patterns
    "autonomous",    # Full self-directed operation
    "emergent"       # Develops emergent intelligence
]

LEARNING_CAPABILITIES = [
    "experience_based_learning",
    "pattern_recognition",
    "behavioral_adaptation",
    "meta_learning",
    "reinforcement_learning",
    "collaborative_learning"
]

# Configuration defaults
DEFAULT_AUTONOMOUS_CONFIG = {
    "autonomy_level": "adaptive",
    "decision_threshold": 0.6,
    "learning_mode": "active",
    "enable_proactive_behavior": True,
    "enable_goal_setting": True,
    "enable_self_improvement": True,
    "enable_peer_learning": True,
    "enable_knowledge_sharing": True,
    "safety_constraints": [
        "no_harmful_actions",
        "respect_resource_limits",
        "maintain_ethical_guidelines"
    ],
    "ethical_guidelines": [
        "transparency_in_decision_making",
        "respect_for_human_oversight",
        "beneficial_outcomes_priority"
    ]
}

def create_autonomous_agent(
    name: str,
    description: str,
    llm,
    tools=None,
    autonomy_level: str = "adaptive",
    learning_mode: str = "active",
    **kwargs
) -> AutonomousLangGraphAgent:
    """
    Factory function to create an autonomous agent with sensible defaults.
    
    Args:
        name: Agent name
        description: Agent description
        llm: LangChain language model
        tools: List of LangChain tools
        autonomy_level: Level of autonomy
        learning_mode: Learning mode
        **kwargs: Additional configuration options
        
    Returns:
        Configured AutonomousLangGraphAgent
    """
    # Extract agent_id from kwargs if provided
    agent_id = kwargs.pop('agent_id', None)

    # Create configuration
    config_dict = DEFAULT_AUTONOMOUS_CONFIG.copy()
    config_dict.update({
        "name": name,
        "description": description,
        "autonomy_level": AutonomyLevel(autonomy_level),
        "learning_mode": LearningMode(learning_mode),
        **kwargs
    })

    config = AutonomousAgentConfig(**config_dict)

    # Create and return agent with optional agent_id
    return AutonomousLangGraphAgent(
        config=config,
        llm=llm,
        tools=tools or [],
        agent_id=agent_id
    )

def create_research_agent(llm, tools=None) -> AutonomousLangGraphAgent:
    """Create a specialized autonomous research agent."""
    return create_autonomous_agent(
        name="Autonomous Research Agent",
        description="Specialized agent for autonomous research and analysis tasks",
        llm=llm,
        tools=tools,
        autonomy_level="autonomous",
        learning_mode="active",
        capabilities=["reasoning", "tool_use", "memory", "planning"],
        enable_proactive_behavior=True,
        enable_goal_setting=True,
        safety_constraints=[
            "verify_information_sources",
            "maintain_research_ethics",
            "respect_intellectual_property"
        ]
    )

def create_creative_agent(llm, tools=None) -> AutonomousLangGraphAgent:
    """Create a specialized autonomous creative agent."""
    return create_autonomous_agent(
        name="Autonomous Creative Agent",
        description="Specialized agent for creative problem-solving and innovation",
        llm=llm,
        tools=tools,
        autonomy_level="emergent",
        learning_mode="active",
        capabilities=["reasoning", "tool_use", "planning"],
        enable_proactive_behavior=True,
        enable_goal_setting=True,
        decision_threshold=0.4,  # Lower threshold for creative exploration
        safety_constraints=[
            "maintain_creative_integrity",
            "respect_originality",
            "ensure_constructive_outcomes"
        ]
    )

def create_optimization_agent(llm, tools=None) -> AutonomousLangGraphAgent:
    """Create a specialized autonomous optimization agent."""
    return create_autonomous_agent(
        name="Autonomous Optimization Agent",
        description="Specialized agent for performance optimization and efficiency improvement",
        llm=llm,
        tools=tools,
        autonomy_level="adaptive",
        learning_mode="reinforcement",
        capabilities=["reasoning", "tool_use", "learning"],
        enable_self_improvement=True,
        decision_threshold=0.7,  # Higher threshold for optimization decisions
        safety_constraints=[
            "maintain_system_stability",
            "preserve_existing_functionality",
            "ensure_measurable_improvements"
        ]
    )

# Utility functions for agent management
def get_agent_capabilities(agent: AutonomousLangGraphAgent) -> Dict[str, Any]:
    """Get comprehensive information about an agent's capabilities."""
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "autonomy_level": agent.autonomous_config.autonomy_level,
        "learning_mode": agent.autonomous_config.learning_mode,
        "capabilities": agent.capabilities,
        "tools_available": list(agent.tools.keys()),
        "decision_threshold": agent.autonomous_config.decision_threshold,
        "safety_constraints": agent.autonomous_config.safety_constraints,
        "ethical_guidelines": agent.autonomous_config.ethical_guidelines,
        "performance_metrics": getattr(agent, 'performance_tracker', {})
    }

def validate_autonomous_config(config: dict) -> bool:
    """Validate autonomous agent configuration."""
    required_fields = ["name", "description", "autonomy_level"]
    
    for field in required_fields:
        if field not in config:
            return False
    
    if config["autonomy_level"] not in SUPPORTED_AUTONOMY_LEVELS:
        return False
    
    return True

# Export module information
__module_info__ = {
    "name": "autonomous",
    "version": __version__,
    "description": __description__,
    "features": AUTONOMOUS_AGENT_FEATURES,
    "autonomy_levels": SUPPORTED_AUTONOMY_LEVELS,
    "learning_capabilities": LEARNING_CAPABILITIES
}
