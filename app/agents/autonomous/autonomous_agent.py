"""
Enhanced Autonomous Agent System with True Agentic Capabilities.

This module implements revolutionary autonomous agents with self-directed
decision-making, adaptive learning, and emergent intelligence capabilities
built on LangChain/LangGraph foundation.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

import structlog
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from app.agents.base.agent import LangGraphAgent, AgentConfig, AgentCapability
from app.core.exceptions import AgentExecutionError

logger = structlog.get_logger(__name__)


# PHASE 1 ENHANCEMENT: Master Orchestrator Agent Capabilities
class TaskComplexity(str, Enum):
    """Task complexity levels for orchestration decisions."""
    SIMPLE = "simple"           # Single tool, straightforward execution
    MODERATE = "moderate"       # Multiple tools, sequential execution
    COMPLEX = "complex"         # Multiple tools, parallel execution, coordination needed
    REVOLUTIONARY = "revolutionary"  # Multi-agent collaboration, advanced orchestration


class OrchestrationStrategy(str, Enum):
    """Strategies for orchestrating multi-agent workflows."""
    SEQUENTIAL = "sequential"   # Execute agents one after another
    PARALLEL = "parallel"       # Execute agents simultaneously
    HIERARCHICAL = "hierarchical"  # Master-worker pattern
    COLLABORATIVE = "collaborative"  # Peer-to-peer collaboration


class TaskAnalysis(BaseModel):
    """Analysis of task requirements for orchestration."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_task: str = Field(..., description="Original task description")
    complexity: TaskComplexity = Field(..., description="Task complexity level")
    required_capabilities: List[str] = Field(..., description="Required agent capabilities")
    required_tools: List[str] = Field(..., description="Required tools")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")
    orchestration_strategy: OrchestrationStrategy = Field(..., description="Recommended orchestration strategy")
    subtasks: List[Dict[str, Any]] = Field(default_factory=list, description="Broken down subtasks")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")


class AgentSelection(BaseModel):
    """Selection of specialized agents for task execution."""
    selection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_analysis: TaskAnalysis = Field(..., description="Task analysis that led to this selection")
    selected_agents: List[Dict[str, Any]] = Field(..., description="Selected specialized agents")
    selection_reasoning: List[str] = Field(..., description="Reasoning for agent selection")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in selection")
    fallback_agents: List[Dict[str, Any]] = Field(default_factory=list, description="Fallback agents if primary fails")


class WorkflowExecution(BaseModel):
    """Execution plan for multi-agent workflow."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_selection: AgentSelection = Field(..., description="Agent selection for this workflow")
    execution_steps: List[Dict[str, Any]] = Field(..., description="Ordered execution steps")
    coordination_points: List[Dict[str, Any]] = Field(default_factory=list, description="Points where coordination is needed")
    context_sharing_plan: Dict[str, Any] = Field(default_factory=dict, description="How context will be shared between agents")
    monitoring_checkpoints: List[Dict[str, Any]] = Field(default_factory=list, description="Monitoring and validation points")
    rollback_strategy: Dict[str, Any] = Field(default_factory=dict, description="Strategy if execution fails")


class AutonomyLevel(str, Enum):
    """Levels of agent autonomy for decision-making."""
    REACTIVE = "reactive"       # Responds to direct instructions only
    PROACTIVE = "proactive"     # Can initiate actions based on context
    ADAPTIVE = "adaptive"       # Learns and adapts behavior patterns
    AUTONOMOUS = "autonomous"   # Full self-directed operation
    EMERGENT = "emergent"       # Develops emergent intelligence


class DecisionConfidence(str, Enum):
    """Confidence levels for autonomous decisions."""
    VERY_LOW = "very_low"       # 0-20% confidence
    LOW = "low"                 # 20-40% confidence
    MEDIUM = "medium"           # 40-60% confidence
    HIGH = "high"               # 60-80% confidence
    VERY_HIGH = "very_high"     # 80-100% confidence


class LearningMode(str, Enum):
    """Learning modes for adaptive behavior."""
    DISABLED = "disabled"       # No learning
    PASSIVE = "passive"         # Observes but doesn't adapt
    ACTIVE = "active"           # Actively learns and adapts
    REINFORCEMENT = "reinforcement"  # Reinforcement learning
    EMERGENT = "emergent"       # Emergent learning patterns


class AutonomousAgentState(TypedDict):
    """Enhanced state for autonomous agent workflows."""
    # Base agent state
    messages: Annotated[List[BaseMessage], add_messages]
    current_task: str
    agent_id: str
    session_id: str
    tools_available: List[str]
    tool_calls: List[Dict[str, Any]]
    outputs: Dict[str, Any]
    errors: List[str]
    iteration_count: int
    max_iterations: int
    custom_state: Dict[str, Any]
    
    # Autonomous capabilities
    autonomy_level: str
    decision_confidence: float
    learning_enabled: bool
    adaptation_history: List[Dict[str, Any]]
    decision_tree: Dict[str, Any]
    goal_stack: List[Dict[str, Any]]
    context_memory: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    # Self-direction capabilities
    self_initiated_tasks: List[Dict[str, Any]]
    proactive_actions: List[Dict[str, Any]]
    emergent_behaviors: List[Dict[str, Any]]
    collaboration_state: Dict[str, Any]


class AutonomousDecision(BaseModel):
    """Model for autonomous agent decisions."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision_type: str = Field(..., description="Type of decision made")
    context: Dict[str, Any] = Field(..., description="Decision context")
    options_considered: List[Dict[str, Any]] = Field(..., description="Options evaluated")
    chosen_option: Dict[str, Any] = Field(..., description="Selected option")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    reasoning: List[str] = Field(..., description="Reasoning chain")
    expected_outcome: Dict[str, Any] = Field(..., description="Expected results")
    actual_outcome: Optional[Dict[str, Any]] = Field(default=None, description="Actual results")
    learning_value: float = Field(default=0.0, description="Learning value from decision")


class AutonomousAgentConfig(AgentConfig):
    """Enhanced configuration for autonomous agents."""
    
    # Autonomy settings
    autonomy_level: AutonomyLevel = Field(default=AutonomyLevel.ADAPTIVE, description="Agent autonomy level")
    decision_threshold: float = Field(default=0.6, description="Minimum confidence for autonomous decisions")
    learning_mode: LearningMode = Field(default=LearningMode.ACTIVE, description="Learning mode")
    
    # Self-direction settings
    enable_proactive_behavior: bool = Field(default=True, description="Enable proactive actions")
    enable_goal_setting: bool = Field(default=True, description="Enable autonomous goal setting")
    enable_self_improvement: bool = Field(default=True, description="Enable self-improvement")
    
    # Collaboration settings
    enable_peer_learning: bool = Field(default=True, description="Enable learning from other agents")
    enable_knowledge_sharing: bool = Field(default=True, description="Enable knowledge sharing")
    
    # Safety and constraints
    safety_constraints: List[str] = Field(default_factory=list, description="Safety constraints")
    ethical_guidelines: List[str] = Field(default_factory=list, description="Ethical guidelines")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource usage limits")


class AutonomousLangGraphAgent(LangGraphAgent):
    """
    Revolutionary autonomous agent with true agentic capabilities.
    
    This agent can make independent decisions, learn from experience,
    adapt its behavior, and exhibit emergent intelligence patterns.
    """
    
    def __init__(
        self,
        config: AutonomousAgentConfig,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        checkpoint_saver: Optional[BaseCheckpointSaver] = None,
        agent_id: Optional[str] = None,
        full_yaml_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the autonomous agent."""
        # Initialize base agent
        super().__init__(config, llm, tools, checkpoint_saver)

        # Set agent ID
        self.agent_id = agent_id or str(uuid.uuid4())

        # Store full YAML configuration for metadata-driven decision engine
        self.full_yaml_config = full_yaml_config or {}

        # Enhanced autonomous capabilities
        self.autonomous_config = config
        self.decision_engine = AutonomousDecisionEngine(config, llm, self.full_yaml_config)
        self.learning_system = AdaptiveLearningSystem(config)

        # Import the real goal manager
        from app.agents.autonomous.goal_manager import AutonomousGoalManager as RealGoalManager
        self.goal_manager = RealGoalManager(config)
        self.collaboration_manager = CollaborationManager(config)

        # New Phase 1 components
        from app.agents.autonomous.bdi_planning_engine import BDIPlanningEngine
        from app.agents.autonomous.persistent_memory import PersistentMemorySystem
        from app.agents.autonomous.proactive_behavior import ProactiveBehaviorSystem
        from app.services.autonomous_persistence import autonomous_persistence

        self.bdi_engine = BDIPlanningEngine(self.agent_id, llm)
        self.persistence_service = autonomous_persistence
        self.memory_system = PersistentMemorySystem(self.agent_id, llm, persistence_service=self.persistence_service)
        self.proactive_system = ProactiveBehaviorSystem(self.agent_id, llm)

        # PHASE 1 ENHANCEMENT: Master Orchestrator Capabilities
        self.task_analyzer = TaskAnalysisEngine(self.agent_id, llm)
        self.agent_selector = AgentSelectionIntelligence(self.agent_id, llm)
        self.workflow_orchestrator = WorkflowOrchestrator(self.agent_id, llm)
        self.context_manager = CrossAgentContextManager(self.agent_id)

        # Orchestration state
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.specialized_agents: Dict[str, Any] = {}  # Cache of specialized agents
        self.orchestration_history: List[Dict[str, Any]] = []
        
        # State tracking
        self.decision_history: List[AutonomousDecision] = []
        self.adaptation_data: Dict[str, Any] = {}
        self.performance_tracker = PerformanceTracker()
        
        # Rebuild graph with autonomous capabilities
        self._build_autonomous_graph()

        # Initialize Phase 1 components asynchronously
        asyncio.create_task(self._initialize_autonomous_components())

        logger.info(
            "Autonomous LangGraph agent initialized",
            agent_id=self.agent_id,
            autonomy_level=config.autonomy_level,
            learning_mode=config.learning_mode,
            capabilities=config.capabilities
        )

    async def _initialize_autonomous_components(self):
        """Initialize all autonomous components."""
        try:
            # Initialize memory system
            await self.memory_system.initialize()

            # Load persistent state if available
            await self._load_persistent_state()

            logger.info(
                "Autonomous components initialized",
                agent_id=self.agent_id,
                memory_initialized=self.memory_system.is_initialized
            )

        except Exception as e:
            logger.error(
                "Failed to initialize autonomous components",
                agent_id=self.agent_id,
                error=str(e)
            )

    async def _load_persistent_state(self):
        """Load persistent state from database."""
        try:
            if not self.persistence_service:
                logger.debug("No persistence service available", agent_id=self.agent_id)
                return

            # Load agent state
            agent_state = await self.persistence_service.load_agent_state(self.agent_id)
            if agent_state:
                # Restore decision history
                self.decision_history = agent_state.get("decision_history", [])
                self.adaptation_data = agent_state.get("adaptation_data", {})

                logger.info("Persistent state loaded",
                           agent_id=self.agent_id,
                           decisions_loaded=len(self.decision_history))

        except Exception as e:
            logger.error("Failed to load persistent state",
                        agent_id=self.agent_id,
                        error=str(e))

    def _build_autonomous_graph(self) -> None:
        """Build enhanced LangGraph workflow with autonomous capabilities."""
        # Create enhanced state graph
        self.graph = StateGraph(AutonomousAgentState)
        
        # Core autonomous nodes
        self.graph.add_node("autonomous_planning", self._autonomous_planning_node)
        self.graph.add_node("decision_making", self._decision_making_node)
        self.graph.add_node("action_execution", self._action_execution_node)
        self.graph.add_node("learning_reflection", self._learning_reflection_node)
        self.graph.add_node("adaptation", self._adaptation_node)
        self.graph.add_node("goal_management", self._goal_management_node)
        
        # Enhanced workflow edges
        self.graph.add_edge(START, "autonomous_planning")
        self.graph.add_edge("autonomous_planning", "goal_management")
        
        # Conditional routing based on autonomy level
        self.graph.add_conditional_edges(
            "goal_management",
            self._route_based_on_autonomy,
            {
                "make_decision": "decision_making",
                "execute_directly": "action_execution",
                "learn_first": "learning_reflection",
                "end": END
            }
        )
        
        self.graph.add_edge("decision_making", "action_execution")
        self.graph.add_edge("action_execution", "learning_reflection")
        
        # Adaptive loop
        self.graph.add_conditional_edges(
            "learning_reflection",
            self._should_adapt,
            {
                "adapt": "adaptation",
                "continue": "autonomous_planning",
                "end": END
            }
        )
        
        self.graph.add_edge("adaptation", "autonomous_planning")
        
        # Compile with enhanced checkpoint system
        if self.checkpoint_saver:
            self.compiled_graph = self.graph.compile(
                checkpointer=self.checkpoint_saver,
                interrupt_before=["decision_making", "adaptation"]  # Allow human intervention
            )
        else:
            self.compiled_graph = self.graph.compile()
        
        logger.info(
            "Autonomous LangGraph workflow built",
            agent_id=self.agent_id,
            nodes=["autonomous_planning", "decision_making", "action_execution",
                   "learning_reflection", "adaptation", "goal_management"],
            autonomy_features=["decision_engine", "learning_system", "goal_manager"]
        )

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute task with autonomous agent state initialization."""
        import time
        import uuid
        from app.backend_logging.backend_logger import get_logger
        from app.backend_logging.models import LogCategory
        from app.backend_logging.context import CorrelationContext, AgentContext
        from app.core.clean_logging import get_conversation_logger
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableConfig

        backend_logger = get_logger()
        conversation_logger = get_conversation_logger(self.name)
        start_time = time.time()
        session_id = str(uuid.uuid4())

        # Set up agent context for logging
        agent_context = AgentContext(self.agent_id, self.config.agent_type)
        CorrelationContext.update_context(
            agent_id=self.agent_id,
            session_id=session_id,
            component="AutonomousLangGraphAgent",
            operation="execute_task"
        )

        # Log task initiation (backend)
        backend_logger.info(
            f"Agent task execution started: {task[:100]}...",
            LogCategory.AGENT_OPERATIONS,
            "LangGraphAgent",
            data={
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "agent_type": self.config.agent_type,
                "task": task,
                "session_id": session_id,
                "tools_available": list(self.tools.keys()),
                "max_iterations": self.config.max_iterations,
                "timeout_seconds": self.config.timeout_seconds
            }
        )

        # Log task initiation (conversation - user-facing)
        conversation_logger.user_query(task)
        conversation_logger.agent_goal(f"I'll work autonomously to complete this task with {self.config.autonomy_level.value} autonomy level")

        async with self._execution_lock:
            try:
                # Create autonomous initial state
                initial_state = AutonomousAgentState(
                    messages=[HumanMessage(content=task)],
                    current_task=task,
                    agent_id=self.agent_id,
                    session_id=session_id,
                    tools_available=list(self.tools.keys()),
                    tool_calls=[],
                    outputs={},
                    errors=[],
                    iteration_count=0,
                    max_iterations=self.config.max_iterations,
                    custom_state=context or {},
                    # Autonomous-specific fields
                    autonomy_level=self.config.autonomy_level.value,  # Convert enum to string
                    decision_confidence=0.0,
                    learning_enabled=self.config.learning_mode == LearningMode.ACTIVE,
                    adaptation_history=[],  # Initialize empty list
                    decision_tree={},
                    goal_stack=[],
                    context_memory={},
                    performance_metrics={}
                )

                # Log workflow initialization
                backend_logger.info(
                    "Starting LangGraph workflow execution",
                    LogCategory.AGENT_OPERATIONS,
                    "LangGraphAgent",
                    data={
                        "agent_id": self.agent_id,
                        "session_id": session_id,
                        "workflow_config": {
                            "thread_id": session_id,
                            "checkpoint_ns": self.config.checkpoint_namespace
                        }
                    }
                )

                # Execute the LangGraph workflow
                config = RunnableConfig(
                    configurable={
                        "thread_id": session_id,
                        "checkpoint_ns": self.config.checkpoint_namespace
                    }
                )

                # Run the autonomous workflow
                result = await self.compiled_graph.ainvoke(initial_state, config)

                # Log successful completion (backend)
                execution_time = time.time() - start_time
                backend_logger.info(
                    "Agent task execution completed successfully",
                    LogCategory.AGENT_OPERATIONS,
                    "LangGraphAgent",
                    data={
                        "agent_id": self.agent_id,
                        "session_id": session_id,
                        "task": task,
                        "execution_time_seconds": execution_time,
                        "iterations_completed": result.get("iteration_count", 0),
                        "tools_used": len(result.get("tool_calls", [])),
                        "errors_encountered": len(result.get("errors", [])),
                        "final_state_keys": list(result.keys())
                    }
                )

                # Log completion (conversation - user-facing)
                # Extract final response from messages
                final_response = ""
                if result.get("messages"):
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        final_response = str(last_msg.content)

                if final_response:
                    conversation_logger.agent_response(final_response)
                else:
                    conversation_logger.success(f"Autonomous task completed successfully in {execution_time:.1f}s!")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                backend_logger.error(
                    f"Agent task execution failed: {str(e)}",
                    LogCategory.AGENT_OPERATIONS,
                    "LangGraphAgent",
                    data={
                        "agent_id": self.agent_id,
                        "session_id": session_id,
                        "task": task,
                        "execution_time_seconds": execution_time,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                raise
    
    async def _autonomous_planning_node(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Autonomous planning node with self-directed goal setting."""
        try:
            print(f"\n📋 **AUTONOMOUS PLANNING PHASE** - Creating strategic plan...")
            print(f"🎯 **CURRENT TASK**: {state['current_task'][:100]}{'...' if len(state['current_task']) > 100 else ''}")
            print(f"🔄 **ITERATION**: {state.get('iteration_count', 0)}")

            # Analyze current context and goals
            current_context = {
                "task": state["current_task"],
                "available_tools": state["tools_available"],
                "previous_outputs": state["outputs"],
                "performance_metrics": state.get("performance_metrics", {}),
                "goal_stack": state.get("goal_stack", [])
            }

            print(f"🔍 **CONTEXT ANALYSIS**:")
            print(f"   • Available Tools: {len(current_context['available_tools'])}")
            print(f"   • Previous Outputs: {len(current_context['previous_outputs'])}")
            print(f"   • Current Goals: {len(current_context['goal_stack'])}")

            # Generate autonomous plan
            print(f"🎯 **PLAN GENERATION** - Creating autonomous execution plan...")
            plan = await self.goal_manager.create_autonomous_plan(
                context=current_context,
                autonomy_level=state["autonomy_level"]
            )

            print(f"✅ **PLAN CREATED**: {plan.get('summary', 'Strategic plan generated')}")
            
            # Update state with plan
            updated_state = state.copy()
            updated_state["goal_stack"] = plan.get("goals", [])
            updated_state["custom_state"]["autonomous_plan"] = plan
            updated_state["messages"].append(
                AIMessage(content=f"Autonomous planning completed: {plan.get('summary', 'Plan created')}")
            )

            # CRITICAL FIX: Increment iteration count to track progress
            updated_state["iteration_count"] = state.get("iteration_count", 0) + 1

            logger.debug(
                "Autonomous planning completed",
                agent_id=self.agent_id,
                goals_count=len(plan.get("goals", [])),
                plan_complexity=plan.get("complexity", "unknown"),
                iteration=updated_state["iteration_count"]
            )

            return updated_state
            
        except Exception as e:
            logger.error("Autonomous planning failed", agent_id=self.agent_id, error=str(e))
            updated_state = state.copy()
            updated_state["errors"].append(f"Planning failed: {str(e)}")
            return updated_state
    
    async def _decision_making_node(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Advanced decision-making node with autonomous reasoning."""
        try:
            print(f"\n🧠 **AUTONOMOUS REASONING PHASE** - Agent is thinking...")

            # AUTONOMOUS THINKING: Let the agent reason about the current situation using its personality
            reasoning_prompt = f"""
            CURRENT SITUATION:
            - TASK: {state.get('current_task', 'Unknown')[:200]}...
            - ACTIVE GOALS: {len(state.get('goal_stack', []))} goals
            - TOOLS AVAILABLE: {state.get('tools_available', [])}
            - ITERATION: {state.get('iteration_count', 0)}
            - PREVIOUS OUTPUTS: {len(state.get('outputs', {}))} outputs generated

            I need to:
            1. Explain what I'm currently thinking about (in my personality/character)
            2. Describe my strategy and approach
            3. Explain which tools I plan to use and why
            4. Share my confidence level and reasoning
            5. Communicate any insights or observations

            Please respond as if you're talking directly to the user, explaining your thought process and next steps.
            Be conversational and informative while staying true to your personality and character.
            Start with something like "I'm analyzing the situation..." or "Let me think about this..."
            """

            autonomous_thoughts = await self._autonomous_reasoning(reasoning_prompt, state)

            # Debug: Check what we actually got back
            print(f"\n💭 **AGENT COMMUNICATION**:")
            if autonomous_thoughts and len(autonomous_thoughts.strip()) > 0:
                print(f"{autonomous_thoughts}")
            else:
                print(f"   [DEBUG: Empty or None response - autonomous_thoughts = '{autonomous_thoughts}']")
            print(f"   (The agent is using its LLM to reason and communicate with us)")

            # Extract decision context
            available_actions = self._get_available_actions(state)
            decision_context = {
                "current_goals": state.get("goal_stack", []),
                "available_actions": available_actions,
                "tools_available": [tool["name"] for tool in available_actions.get("tools", [])],  # Add this for MetadataDrivenDecisionEngine
                "current_task": state.get("current_task", ""),
                "context_memory": state.get("context_memory", {}),
                "performance_history": state.get("performance_metrics", {}),
                "constraints": self.autonomous_config.safety_constraints,
                "autonomous_reasoning": autonomous_thoughts
            }

            print(f"⚖️ **DECISION ANALYSIS** - Evaluating {len(decision_context['available_actions'].get('tools', []))} available tools...")

            # Make autonomous decision
            decision = await self.decision_engine.make_autonomous_decision(
                context=decision_context,
                confidence_threshold=self.autonomous_config.decision_threshold
            )

            # SHOW DECISION REASONING
            chosen_action = decision.chosen_option.get('action', 'Unknown')
            confidence = decision.confidence
            print(f"✅ **DECISION MADE**: {chosen_action}")
            print(f"🎯 **CONFIDENCE LEVEL**: {confidence:.1%}")
            print(f"🔍 **REASONING**: {decision.reasoning[:2] if decision.reasoning else ['No reasoning provided']}")

            # Record decision
            self.decision_history.append(decision)

            # Update state with decision
            updated_state = state.copy()
            updated_state["decision_confidence"] = decision.confidence
            updated_state["custom_state"]["current_decision"] = decision.dict()
            updated_state["messages"].append(
                AIMessage(content=f"Decision made: {decision.chosen_option.get('action', 'Unknown')} "
                                f"(confidence: {decision.confidence:.2f})")
            )
            
            logger.info(
                "Autonomous decision made",
                agent_id=self.agent_id,
                decision_type=decision.decision_type,
                confidence=decision.confidence,
                chosen_action=decision.chosen_option.get("action", "unknown")
            )
            
            return updated_state
            
        except Exception as e:
            logger.error("Decision making failed", agent_id=self.agent_id, error=str(e))
            updated_state = state.copy()
            updated_state["errors"].append(f"Decision making failed: {str(e)}")
            return updated_state

    async def _action_execution_node(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Enhanced action execution with autonomous tool selection."""
        try:
            print(f"\n🚀 **ACTION EXECUTION PHASE** - Implementing my decision...")

            # Get current decision
            current_decision = state["custom_state"].get("current_decision")
            if not current_decision:
                print("❌ **NO DECISION FOUND** - Cannot execute action without a decision")
                return state

            # Execute chosen action
            action = current_decision.get("chosen_option", {})
            action_type = action.get("type", "unknown")

            print(f"🎯 **EXECUTING ACTION**: {action.get('action', 'Unknown')} (Type: {action_type})")

            updated_state = state.copy()

            if action_type == "tool_use":
                # CRITICAL FIX: Extract tool parameters correctly from decision structure
                parameters = action.get("parameters", {})
                tool_name = parameters.get("tool")
                tool_args = parameters.get("args", {})

                print(f"🔧 **TOOL SELECTED**: {tool_name}")
                print(f"📋 **PARAMETERS**: {tool_args}")
                logger.debug(f"Attempting to execute tool: {tool_name} with args: {tool_args}")
                logger.debug(f"Available tools: {list(self.tools.keys())}")

                if tool_name and tool_name in self.tools:
                    tool = self.tools[tool_name]
                    print(f"⚡ **EXECUTING TOOL**: {tool_name} - Starting execution...")
                    logger.info(f"Executing autonomous tool: {tool_name}")

                    try:
                        result = await tool.ainvoke(tool_args)
                        print(f"✅ **TOOL EXECUTION COMPLETE**: {tool_name}")
                        print(f"📊 **RESULTS SUMMARY**: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")
                        logger.info(f"Tool execution successful: {tool_name} -> {result}")

                        # AGENT COMMUNICATION: Let the agent analyze and communicate the results
                        analysis_prompt = f"""
                        I just executed the {tool_name} tool with the query: {tool_args.get('query', 'N/A')}

                        RESULTS RECEIVED:
                        {str(result)[:500]}...

                        As an Apple Stock Monitor Agent, I need to:
                        1. Analyze what this data tells me about Apple stock
                        2. Explain the significance of these findings
                        3. Describe what I plan to do next
                        4. Share any insights or concerns

                        Please communicate directly with the user about what you found and what it means.
                        Be specific about Apple stock insights if any were found.
                        """

                        analysis_response = await self._autonomous_reasoning(analysis_prompt, updated_state)

                        # Debug: Check what we actually got back
                        print(f"\n🔍 **AGENT ANALYSIS**:")
                        if analysis_response and len(analysis_response.strip()) > 0:
                            print(f"{analysis_response}")
                        else:
                            print(f"   [DEBUG: Empty or None response - analysis_response = '{analysis_response}']")

                        # Record tool execution
                        updated_state["tool_calls"].append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": result,
                            "autonomous": True,
                            "confidence": current_decision.get("confidence", 0.0),
                            "agent_analysis": analysis_response
                        })

                        # Add result to outputs
                        updated_state["outputs"][f"autonomous_tool_{len(updated_state['tool_calls'])}"] = result

                        updated_state["messages"].append(
                            AIMessage(content=f"Autonomous tool execution: {tool_name} -> {result}")
                        )

                    except Exception as tool_error:
                        logger.error(f"Tool execution failed: {tool_name} - {str(tool_error)}")
                        updated_state["errors"].append(f"Tool execution failed: {tool_name} - {str(tool_error)}")

                else:
                    error_msg = f"Tool not found or invalid: {tool_name}. Available: {list(self.tools.keys())}"
                    logger.error(error_msg)
                    updated_state["errors"].append(error_msg)

            elif action_type == "reasoning":
                # Perform autonomous reasoning
                reasoning_prompt = action.get("prompt", "")
                reasoning_result = await self._autonomous_reasoning(reasoning_prompt, state)

                updated_state["outputs"]["autonomous_reasoning"] = reasoning_result
                updated_state["messages"].append(
                    AIMessage(content=f"Autonomous reasoning: {reasoning_result}")
                )

            elif action_type == "goal_pursuit":
                # Pursue autonomous goal
                goal = action.get("goal", {})
                goal_result = await self.goal_manager.pursue_goal(goal, state)

                updated_state["outputs"]["goal_pursuit"] = goal_result
                updated_state["messages"].append(
                    AIMessage(content=f"Goal pursuit: {goal_result.get('status', 'unknown')}")
                )

            logger.debug(
                "Autonomous action executed",
                agent_id=self.agent_id,
                action_type=action_type,
                success=len(updated_state["errors"]) == len(state["errors"])
            )

            return updated_state

        except Exception as e:
            logger.error("Action execution failed", agent_id=self.agent_id, error=str(e))
            updated_state = state.copy()
            updated_state["errors"].append(f"Action execution failed: {str(e)}")
            return updated_state

    async def _learning_reflection_node(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Learning and reflection node for continuous improvement."""
        try:
            print(f"\n🧠 **LEARNING & REFLECTION PHASE** - Analyzing my performance...")

            if not state.get("learning_enabled", True):
                print("📚 **LEARNING DISABLED** - Skipping reflection phase")
                return state

            # Analyze performance and outcomes
            performance_data = {
                "iteration": state["iteration_count"],
                "decisions_made": len([d for d in self.decision_history if d.timestamp > datetime.utcnow() - timedelta(hours=1)]),
                "tools_used": len(state["tool_calls"]),
                "errors_encountered": len(state["errors"]),
                "outputs_generated": len(state["outputs"]),
                "confidence_levels": [d.confidence for d in self.decision_history[-5:]]  # Last 5 decisions
            }

            print(f"📊 **PERFORMANCE ANALYSIS**:")
            print(f"   • Iteration: {performance_data['iteration']}")
            print(f"   • Tools Used: {performance_data['tools_used']}")
            print(f"   • Decisions Made: {performance_data['decisions_made']}")
            print(f"   • Errors: {performance_data['errors_encountered']}")
            print(f"   • Outputs Generated: {performance_data['outputs_generated']}")

            if performance_data['confidence_levels']:
                avg_confidence = sum(performance_data['confidence_levels']) / len(performance_data['confidence_levels'])
                print(f"   • Average Confidence: {avg_confidence:.1%}")

            print(f"🔍 **LEARNING ANALYSIS** - Extracting insights from recent experience...")

            # Learn from recent experience
            learning_insights = await self.learning_system.analyze_and_learn(
                performance_data=performance_data,
                decision_history=self.decision_history[-10:],  # Last 10 decisions
                context=state["custom_state"]
            )

            # Update adaptation data
            updated_state = state.copy()
            updated_state["adaptation_history"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "performance_data": performance_data,
                "learning_insights": learning_insights,
                "adaptation_suggestions": learning_insights.get("adaptations", [])
            })

            # Update performance metrics
            updated_state["performance_metrics"] = {
                **updated_state.get("performance_metrics", {}),
                **performance_data,
                "learning_rate": learning_insights.get("learning_rate", 0.0),
                "adaptation_score": learning_insights.get("adaptation_score", 0.0)
            }

            updated_state["messages"].append(
                AIMessage(content=f"Learning reflection completed: {learning_insights.get('summary', 'Insights gained')}")
            )

            logger.info(
                "Learning reflection completed",
                agent_id=self.agent_id,
                insights_count=len(learning_insights.get("insights", [])),
                adaptation_score=learning_insights.get("adaptation_score", 0.0)
            )

            return updated_state

        except Exception as e:
            logger.error("Learning reflection failed", agent_id=self.agent_id, error=str(e))
            updated_state = state.copy()
            updated_state["errors"].append(f"Learning reflection failed: {str(e)}")
            return updated_state

    async def _adaptation_node(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Adaptation node for behavioral evolution."""
        try:
            # Get latest adaptation suggestions
            latest_adaptation = state["adaptation_history"][-1] if state["adaptation_history"] else {}
            adaptations = latest_adaptation.get("adaptation_suggestions", [])

            if not adaptations:
                return state

            # Apply adaptations
            updated_state = state.copy()
            applied_adaptations = []

            for adaptation in adaptations:
                adaptation_type = adaptation.get("type")

                if adaptation_type == "decision_threshold":
                    # Adjust decision confidence threshold
                    new_threshold = adaptation.get("value", self.autonomous_config.decision_threshold)
                    self.autonomous_config.decision_threshold = max(0.1, min(0.9, new_threshold))
                    applied_adaptations.append(f"Decision threshold -> {new_threshold:.2f}")

                elif adaptation_type == "tool_preference":
                    # Update tool usage preferences
                    tool_preferences = adaptation.get("preferences", {})
                    updated_state["custom_state"]["tool_preferences"] = tool_preferences
                    applied_adaptations.append(f"Tool preferences updated: {list(tool_preferences.keys())}")

                elif adaptation_type == "goal_strategy":
                    # Adapt goal pursuit strategies
                    strategy_updates = adaptation.get("strategy", {})
                    await self.goal_manager.update_strategies(strategy_updates)
                    applied_adaptations.append(f"Goal strategies updated: {list(strategy_updates.keys())}")

                elif adaptation_type == "learning_rate":
                    # Adjust learning parameters
                    learning_rate = adaptation.get("rate", 0.1)
                    await self.learning_system.adjust_learning_rate(learning_rate)
                    applied_adaptations.append(f"Learning rate -> {learning_rate:.3f}")

            # Record adaptation
            updated_state["adaptation_data"] = {
                **updated_state.get("adaptation_data", {}),
                "last_adaptation": datetime.utcnow().isoformat(),
                "adaptations_applied": applied_adaptations,
                "adaptation_count": len(applied_adaptations)
            }

            updated_state["messages"].append(
                AIMessage(content=f"Behavioral adaptation completed: {len(applied_adaptations)} changes applied")
            )

            logger.info(
                "Behavioral adaptation completed",
                agent_id=self.agent_id,
                adaptations_applied=len(applied_adaptations),
                changes=applied_adaptations
            )

            return updated_state

        except Exception as e:
            logger.error("Adaptation failed", agent_id=self.agent_id, error=str(e))
            updated_state = state.copy()
            updated_state["errors"].append(f"Adaptation failed: {str(e)}")
            return updated_state

    async def _goal_management_node(self, state: AutonomousAgentState) -> AutonomousAgentState:
        """Goal management node for autonomous goal setting and tracking."""
        try:
            # Update goal stack based on current context
            current_goals = state.get("goal_stack", [])
            context = {
                "current_task": state["current_task"],
                "performance_metrics": state.get("performance_metrics", {}),
                "available_tools": state["tools_available"],
                "recent_outputs": state["outputs"]
            }

            # Let goal manager update goals
            updated_goals = await self.goal_manager.update_goal_stack(current_goals, context)

            updated_state = state.copy()
            updated_state["goal_stack"] = updated_goals
            updated_state["messages"].append(
                AIMessage(content=f"Goal management: {len(updated_goals)} active goals")
            )

            logger.debug(
                "Goal management completed",
                agent_id=self.agent_id,
                active_goals=len(updated_goals)
            )

            return updated_state

        except Exception as e:
            logger.error("Goal management failed", agent_id=self.agent_id, error=str(e))
            updated_state = state.copy()
            updated_state["errors"].append(f"Goal management failed: {str(e)}")
            return updated_state

    def _route_based_on_autonomy(self, state: AutonomousAgentState) -> str:
        """Route workflow based on autonomy level and current state."""
        autonomy_level = state.get("autonomy_level", "medium")
        goal_stack = state.get("goal_stack", [])
        errors = state.get("errors", [])
        iteration_count = state.get("iteration_count", 0)

        # Safety check - end if too many errors
        if len(errors) > 5:
            logger.warning("Too many errors, ending execution", agent_id=self.agent_id)
            return "end"

        # CRITICAL FIX: Always prioritize action execution for the first few iterations
        # to ensure the agent actually uses tools instead of getting stuck in learning loops
        if iteration_count < 3:
            logger.debug(f"Early iteration {iteration_count}, routing to decision making for action execution")
            return "make_decision"

        # Route based on autonomy level
        if autonomy_level in ["low", "reactive"]:
            # Low autonomy - execute directly without complex decision making
            return "execute_directly"
        elif autonomy_level in ["medium", "proactive"]:
            # Medium autonomy - make decisions but don't learn extensively
            if goal_stack:
                return "make_decision"
            else:
                return "execute_directly"
        elif autonomy_level in ["high", "adaptive", "autonomous"]:
            # High autonomy - full decision making and learning
            # FIXED: Only learn after we've had some action execution iterations
            if state.get("learning_enabled", True) and iteration_count > 5 and iteration_count % 4 == 0:
                return "learn_first"  # Learn every 4th iteration after iteration 5
            else:
                return "make_decision"
        elif autonomy_level == "emergent":
            # Emergent autonomy - learn but not immediately
            if iteration_count > 3:
                return "learn_first"
            else:
                return "make_decision"
        else:
            # Default to decision making
            return "make_decision"

    def _should_adapt(self, state: AutonomousAgentState) -> str:
        """Determine if the agent should adapt its behavior."""
        adaptation_history = state.get("adaptation_history", [])
        performance_metrics = state.get("performance_metrics", {})
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 50)
        tools_used = len(state.get("tool_calls", []))

        # ENHANCED TASK COMPLETION LOGIC: Check if task is truly complete
        if tools_used > 0 and state.get("outputs"):
            task_complete = self._is_task_truly_complete(state)
            if task_complete:
                logger.info(f"Task completed successfully with {tools_used} tools used, ending execution")
                return "end"
            else:
                logger.debug(f"Task partially complete ({tools_used} tools used), continuing execution")
                # Continue execution to complete remaining requirements
                return "continue"

        # CRITICAL FIX: Prevent infinite loops - limit total iterations
        if iteration_count >= max_iterations:
            logger.warning(f"Maximum iterations ({max_iterations}) reached, ending execution")
            return "end"

        # CRITICAL FIX: End if we've been running too long without tool usage
        if iteration_count > 10 and tools_used == 0:
            logger.warning(f"No tools used after {iteration_count} iterations, ending execution")
            return "end"

        # Check if adaptation is needed
        recent_adaptations = [
            a for a in adaptation_history
            if datetime.fromisoformat(a["timestamp"]) > datetime.utcnow() - timedelta(hours=1)
        ]

        # Don't adapt too frequently
        if len(recent_adaptations) > 2:
            logger.debug("Too many recent adaptations, ending execution")
            return "end"

        # Adapt if performance is declining
        error_rate = performance_metrics.get("errors_encountered", 0) / max(1, performance_metrics.get("outputs_generated", 1))
        if error_rate > 0.3:
            return "adapt"

        # Adapt if learning suggests improvements
        if adaptation_history:
            latest_adaptation = adaptation_history[-1]
            adaptation_suggestions = latest_adaptation.get("adaptation_suggestions", [])
            if adaptation_suggestions:
                return "adapt"

        # CRITICAL FIX: Default to ending execution instead of continuing indefinitely
        # Only continue if we haven't used tools yet and haven't exceeded reasonable limits
        goal_stack = state.get("goal_stack", [])
        if goal_stack and tools_used == 0 and iteration_count < 5:
            logger.debug(f"Goals remain and no tools used yet (iteration {iteration_count}), continuing")
            return "continue"

        # End execution by default to prevent infinite loops
        logger.info(f"Ending execution after {iteration_count} iterations with {tools_used} tools used")
        return "end"

    def _is_task_truly_complete(self, state: AutonomousAgentState) -> bool:
        """Check if the task is truly complete based on requirements analysis."""
        try:
            current_task = state.get("current_task", "").lower()
            tool_calls = state.get("tool_calls", [])
            outputs = state.get("outputs", {})

            # Extract tools that have been used
            tools_used = [call.get("tool") for call in tool_calls if call.get("tool")]

            # Check for multi-step tasks that require both analysis AND generation
            requires_analysis = any(keyword in current_task for keyword in [
                "analyze", "analysis", "metrics", "insights", "intelligence", "review"
            ])

            requires_generation = any(keyword in current_task for keyword in [
                "excel", "spreadsheet", "document", "file", "generate", "create", "build", "produce"
            ])

            # If task requires both analysis and generation
            if requires_analysis and requires_generation:
                has_analysis = any(tool in tools_used for tool in [
                    "business_intelligence", "knowledge_search", "document_analysis"
                ])
                has_generation = any(tool in tools_used for tool in [
                    "revolutionary_document_intelligence", "file_system"
                ])

                if has_analysis and not has_generation:
                    logger.debug("Task requires both analysis and generation - analysis complete, generation pending")
                    return False
                elif has_analysis and has_generation:
                    logger.debug("Task requires both analysis and generation - both complete")
                    return True
                else:
                    logger.debug("Task requires both analysis and generation - analysis pending")
                    return False

            # For single-requirement tasks, check if requirement is met
            if requires_generation:
                has_generation = any(tool in tools_used for tool in [
                    "revolutionary_document_intelligence", "file_system"
                ])
                return has_generation

            if requires_analysis:
                has_analysis = any(tool in tools_used for tool in [
                    "business_intelligence", "knowledge_search", "document_analysis"
                ])
                return has_analysis

            # For other tasks, consider complete if we have outputs
            return bool(outputs)

        except Exception as e:
            logger.warning(f"Task completion check failed: {e}")
            # Default to incomplete to be safe
            return False

    def _get_available_actions(self, state: AutonomousAgentState) -> Dict[str, Any]:
        """Get available actions based on current state - dynamically generate tool parameters."""
        # Extract context for dynamic parameter generation
        current_task = state.get("current_task", "")
        goal_stack = state.get("goal_stack", [])

        # Use metadata-driven parameter generation
        from app.tools.metadata import get_global_registry
        from app.tools.metadata.parameter_generator import ParameterGenerator

        metadata_registry = get_global_registry()
        param_generator = ParameterGenerator()

        tools_with_params = []
        for tool_name in state["tools_available"]:
            # Get tool metadata
            tool_metadata = metadata_registry.get_tool_metadata(tool_name)

            if tool_metadata:
                # Generate parameters using metadata
                context = {
                    "current_task": current_task,
                    "goal_stack": goal_stack,
                    "state": state,
                    "chaos_mode": getattr(self.config, 'chaos_mode', 'normal'),
                    "creativity_level": getattr(self.config, 'creativity_level', 'medium')
                }
                suggested_params = param_generator.generate_parameters(tool_metadata, context)

                # Calculate confidence using metadata
                tool_confidence = metadata_registry.calculate_tool_confidence(tool_name, context)
            else:
                # Fallback for tools without metadata
                suggested_params = {}
                tool_confidence = 0.5

            tools_with_params.append({
                "name": tool_name,
                "suggested_params": suggested_params,
                "confidence": tool_confidence
            })

        # Calculate reasoning confidence based on recent iterations
        reasoning_confidence = self._calculate_reasoning_confidence(state)

        return {
            "tools": tools_with_params,
            "reasoning": {"available": True, "confidence": reasoning_confidence},
            "goal_actions": [{"action": "pursue_goal", "goal": goal} for goal in goal_stack],
            "exploration": {"available": self.config.learning_mode != "disabled", "risk_level": 0.5}
        }

    # Removed hardcoded _generate_tool_parameters method - now using metadata-driven parameter generation






    # Removed hardcoded _calculate_tool_confidence method - now using metadata-driven confidence calculation

    def _calculate_reasoning_confidence(self, state: AutonomousAgentState) -> float:
        """Calculate reasoning confidence based on recent iterations."""
        # Count recent reasoning actions
        reasoning_iterations = self._count_recent_reasoning_actions(state)

        # Start with base confidence
        base_confidence = 0.7

        # Reduce confidence after multiple reasoning iterations
        confidence_reduction = reasoning_iterations * 0.15

        # Minimum confidence to prevent complete elimination
        final_confidence = max(0.3, base_confidence - confidence_reduction)

        return final_confidence

    def _count_recent_reasoning_actions(self, state: AutonomousAgentState) -> int:
        """Count recent reasoning actions to prevent infinite loops."""
        # Look at recent outputs for reasoning actions
        outputs = state.get("outputs", {})
        reasoning_count = 0

        # Count autonomous_reasoning outputs
        if "autonomous_reasoning" in outputs:
            reasoning_count += 1

        # Check iteration count as proxy for reasoning loops
        iteration_count = state.get("iteration_count", 0)
        if iteration_count > 2:  # After 2 iterations, start reducing reasoning confidence
            reasoning_count += max(0, iteration_count - 2)

        return reasoning_count

    def _count_recent_tool_failures(self, tool_name: str, state: AutonomousAgentState) -> int:
        """Count recent failures for a specific tool."""
        # This is a placeholder - in a full implementation, this would check
        # the agent's memory or execution history for tool failures
        errors = state.get("errors", [])
        tool_failures = sum(1 for error in errors if tool_name in str(error))
        return tool_failures

    def _extract_key_terms(self, text: str) -> list:
        """Extract key terms from text for search queries."""
        # Simple keyword extraction - remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "you", "must", "use", "your", "now"}
        words = text.lower().split()
        key_terms = [word.strip(".,!?:;") for word in words if word.strip(".,!?:;") not in stop_words and len(word) > 2]
        return key_terms

    async def _autonomous_reasoning(self, prompt: str, state: AutonomousAgentState) -> str:
        """Perform autonomous reasoning using the LLM."""
        try:
            # CRITICAL FIX: Use the configured system prompt (contains personality) instead of hardcoded generic one
            tools_description = ", ".join(self.tools.keys()) if self.tools else "None"
            configured_system_prompt = self.config.system_prompt.format(tools=tools_description)

            # Enhance the configured system prompt with autonomous capabilities
            enhanced_system_prompt = f"""{configured_system_prompt}

AUTONOMOUS CAPABILITIES:
You have access to tools that you MUST use when appropriate:
- Use tools for calculations, analysis, and data processing
- Always call tools when the task requires specialized functionality
- Use function calling format to invoke tools

Make autonomous decisions about which tools to use and when to use them while maintaining your personality and character."""

            reasoning_prompt = ChatPromptTemplate.from_messages([
                ("system", enhanced_system_prompt),
                ("human", "{prompt}\n\nContext: {context}\nCurrent task: {task}")
            ])

            # CRITICAL: Use regular LLM for reasoning (not tool-bound) since we just want text responses
            chain = reasoning_prompt | self.llm

            # CRITICAL FIX: Convert datetime objects to strings for JSON serialization
            def serialize_for_json(obj):
                from datetime import datetime, date, time
                import uuid

                if isinstance(obj, (datetime, date, time)):
                    return obj.isoformat()
                elif isinstance(obj, uuid.UUID):
                    return str(obj)
                elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):  # Pydantic models
                    try:
                        return serialize_for_json(obj.dict())
                    except:
                        return str(obj)
                elif hasattr(obj, '__dict__'):  # Other objects with attributes
                    try:
                        return serialize_for_json(obj.__dict__)
                    except:
                        return str(obj)
                elif isinstance(obj, list):
                    return [serialize_for_json(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: serialize_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    # Fallback for any other type
                    try:
                        return str(obj)
                    except:
                        return f"<{type(obj).__name__} object>"

            context_info = {
                "available_tools": state["tools_available"],
                "current_outputs": serialize_for_json(state["outputs"]),
                "goal_stack": serialize_for_json(state.get("goal_stack", [])),
                "performance_metrics": serialize_for_json(state.get("performance_metrics", {})),
                "iteration": state.get("iteration_count", 0)
            }

            # Debug: Try to serialize and catch any remaining issues
            try:
                context_json = json.dumps(context_info, indent=2)
            except Exception as e:
                logger.error(f"JSON serialization still failing: {e}")
                # Fallback: serialize everything as strings
                context_info = {k: str(v) for k, v in context_info.items()}
                context_json = json.dumps(context_info, indent=2)

            # Debug: Log what we're sending to the LLM
            logger.debug(f"Sending to LLM - Prompt: {prompt[:100]}...")
            logger.debug(f"Context length: {len(context_json)}")

            response = await chain.ainvoke({
                "prompt": prompt,
                "context": context_json,
                "task": state["current_task"]
            })

            # Debug: Log what we got back
            logger.debug(f"LLM response type: {type(response)}")
            logger.debug(f"LLM response: {response}")

            if hasattr(response, 'content'):
                result = response.content
                logger.debug(f"Response content: '{result}'")
            else:
                result = str(response)
                logger.debug(f"Response as string: '{result}'")

            return result

        except Exception as e:
            logger.error("Autonomous reasoning failed", agent_id=self.agent_id, error=str(e))
            return f"Reasoning failed: {str(e)}"

    async def execute_orchestrated_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        PHASE 1 ENHANCEMENT: Execute task using master orchestration capabilities.

        This method analyzes complex tasks, selects optimal specialized agents,
        and orchestrates multi-agent workflows for revolutionary results.

        Args:
            task: The complex task to execute
            context: Additional context for task execution

        Returns:
            Dict containing orchestrated execution results
        """
        try:
            logger.info(f"Starting orchestrated task execution: {task[:100]}...")
            start_time = time.time()

            # Initialize context
            execution_context = context or {}
            execution_context.update({
                "orchestrator_agent_id": self.agent_id,
                "execution_start_time": datetime.utcnow(),
                "orchestration_mode": "master_orchestrator"
            })

            # STEP 1: Analyze the task
            logger.info("🧠 Analyzing task complexity and requirements...")
            task_analysis = await self.task_analyzer.analyze_task(task, execution_context)

            logger.info(f"📊 Task Analysis Complete:")
            logger.info(f"   - Complexity: {task_analysis.complexity}")
            logger.info(f"   - Required Tools: {task_analysis.required_tools}")
            logger.info(f"   - Strategy: {task_analysis.orchestration_strategy}")

            # STEP 2: Select optimal agents
            logger.info("🎯 Selecting specialized agents...")
            agent_selection = await self.agent_selector.select_agents(task_analysis, execution_context)

            logger.info(f"🤖 Agent Selection Complete:")
            logger.info(f"   - Selected Agents: {len(agent_selection.selected_agents)}")
            logger.info(f"   - Confidence: {agent_selection.confidence_score:.2f}")

            # STEP 3: Create orchestrated workflow
            logger.info("🎭 Creating orchestrated workflow...")
            workflow = await self.workflow_orchestrator.create_workflow(agent_selection, execution_context)

            logger.info(f"🔄 Workflow Created:")
            logger.info(f"   - Execution Steps: {len(workflow.execution_steps)}")
            logger.info(f"   - Coordination Points: {len(workflow.coordination_points)}")

            # STEP 4: Create shared context for multi-agent coordination
            logger.info("🌐 Setting up cross-agent context...")
            context_id = await self.context_manager.create_shared_context(workflow.execution_id, execution_context)

            # STEP 5: Execute the orchestrated workflow
            logger.info("🚀 Executing orchestrated workflow...")
            workflow_results = await self.workflow_orchestrator.execute_workflow(workflow, execution_context)

            # STEP 6: Compile final results
            execution_time = time.time() - start_time

            final_results = {
                "status": "success",
                "orchestration_type": "master_orchestrated",
                "task_analysis": {
                    "original_task": task,
                    "complexity": task_analysis.complexity,
                    "required_tools": task_analysis.required_tools,
                    "orchestration_strategy": task_analysis.orchestration_strategy
                },
                "agent_selection": {
                    "selected_agents": [agent["agent_type"] for agent in agent_selection.selected_agents],
                    "confidence_score": agent_selection.confidence_score
                },
                "workflow_execution": workflow_results,
                "execution_metrics": {
                    "total_execution_time": execution_time,
                    "agents_coordinated": len(agent_selection.selected_agents),
                    "coordination_points": len(workflow.coordination_points),
                    "context_id": context_id
                },
                "orchestrator_agent_id": self.agent_id
            }

            # STEP 7: Clean up context
            await self.context_manager.cleanup_context(context_id)

            # Update orchestration history
            self.orchestration_history.append({
                "timestamp": datetime.utcnow(),
                "task": task,
                "results": final_results,
                "execution_time": execution_time
            })

            logger.info(f"✅ Orchestrated execution completed in {execution_time:.2f} seconds")
            logger.info(f"🎉 Coordinated {len(agent_selection.selected_agents)} specialized agents successfully!")

            return final_results

        except Exception as e:
            logger.error(f"Orchestrated execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "orchestration_type": "master_orchestrated",
                "task": task,
                "orchestrator_agent_id": self.agent_id,
                "execution_time": time.time() - start_time if 'start_time' in locals() else 0
            }


# Supporting classes that need to be imported
class AutonomousDecisionEngine:
    """Placeholder for decision engine - implemented in decision_engine.py"""
    def __init__(self, config, llm, full_yaml_config=None):
        self.config = config
        self.llm = llm
        self.full_yaml_config = full_yaml_config or {}

    async def make_autonomous_decision(self, context, confidence_threshold):
        # Use metadata-driven decision engine
        from app.agents.autonomous.metadata_driven_decision_engine import MetadataDrivenDecisionEngine
        # Use full YAML config which contains decision_patterns and behavioral_rules
        metadata_engine = MetadataDrivenDecisionEngine(self.full_yaml_config)
        decision_result = await metadata_engine.make_decision(context)

        # Convert DecisionResult to AutonomousDecision format expected by the agent
        autonomous_decision = AutonomousDecision(
            decision_type="metadata_driven",
            context=context,
            options_considered=[option.model_dump() if hasattr(option, 'model_dump') else option.dict() for option in decision_result.all_options],
            chosen_option=decision_result.selected_option.model_dump() if hasattr(decision_result.selected_option, 'model_dump') else decision_result.selected_option.dict(),
            confidence=decision_result.confidence,
            reasoning=decision_result.reasoning,
            expected_outcome=decision_result.expected_outcome
        )

        return autonomous_decision


class AdaptiveLearningSystem:
    """Placeholder for learning system - implemented in learning_system.py"""
    def __init__(self, config):
        self.config = config

    async def analyze_and_learn(self, performance_data, decision_history, context):
        # This will be replaced by the actual implementation
        from app.agents.autonomous.learning_system import AdaptiveLearningSystem as RealSystem
        real_system = RealSystem(self.config)
        return await real_system.analyze_and_learn(performance_data, decision_history, context)

    async def adjust_learning_rate(self, learning_rate):
        pass





class CollaborationManager:
    """Placeholder for collaboration manager - to be implemented"""
    def __init__(self, config):
        self.config = config


class PerformanceTracker:
    """Placeholder for performance tracker - to be implemented"""
    def __init__(self):
        pass


# PHASE 1 ENHANCEMENT: Master Orchestrator Engine Classes
class TaskAnalysisEngine:
    """
    Analyzes complex tasks and determines orchestration requirements.

    This engine breaks down complex tasks into subtasks, identifies required
    capabilities and tools, and recommends orchestration strategies.
    """

    def __init__(self, agent_id: str, llm: BaseLanguageModel):
        self.agent_id = agent_id
        self.llm = llm
        self.analysis_history: List[TaskAnalysis] = []

    async def analyze_task(self, task: str, context: Dict[str, Any]) -> TaskAnalysis:
        """Analyze a task and determine orchestration requirements."""
        try:
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze this task for multi-agent orchestration:

            TASK: {task}
            CONTEXT: {json.dumps(context, indent=2)}

            Determine:
            1. Task complexity (simple/moderate/complex/revolutionary)
            2. Required capabilities and tools
            3. Recommended orchestration strategy
            4. Subtask breakdown if needed
            5. Dependencies and success criteria

            Respond with structured analysis focusing on practical orchestration needs.
            """

            # Get LLM analysis
            from langchain_core.prompts import ChatPromptTemplate
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a task analysis expert for multi-agent orchestration."),
                ("human", analysis_prompt)
            ])

            chain = prompt_template | self.llm
            response = await chain.ainvoke({})

            # Parse response and create TaskAnalysis
            analysis = self._parse_analysis_response(task, response.content, context)
            self.analysis_history.append(analysis)

            logger.info(f"Task analysis completed: {analysis.complexity} complexity, {len(analysis.required_tools)} tools needed")
            return analysis

        except Exception as e:
            logger.error(f"Task analysis failed: {str(e)}")
            # Return fallback analysis
            return TaskAnalysis(
                original_task=task,
                complexity=TaskComplexity.MODERATE,
                required_capabilities=["general"],
                required_tools=["business_intelligence"],
                estimated_duration=300,
                orchestration_strategy=OrchestrationStrategy.SEQUENTIAL,
                subtasks=[{"task": task, "priority": 1}],
                success_criteria=["Task completed successfully"]
            )

    def _parse_analysis_response(self, task: str, response: str, context: Dict[str, Any]) -> TaskAnalysis:
        """Parse LLM response into TaskAnalysis object."""
        # Simple parsing logic - in production this would be more sophisticated
        complexity = TaskComplexity.MODERATE
        if "revolutionary" in response.lower() or "complex" in response.lower():
            complexity = TaskComplexity.COMPLEX
        elif "simple" in response.lower():
            complexity = TaskComplexity.SIMPLE

        # Extract required tools from context and response
        required_tools = []
        if "business" in task.lower() or "financial" in task.lower():
            required_tools.extend(["business_intelligence", "revolutionary_document_intelligence"])
        if "music" in task.lower() or "audio" in task.lower():
            required_tools.append("ai_music_composition")
        if "social" in task.lower() or "media" in task.lower():
            required_tools.append("social_media_orchestrator")
        if "document" in task.lower() or "report" in task.lower():
            required_tools.append("revolutionary_document_intelligence")

        # Default to business intelligence if no specific tools identified
        if not required_tools:
            required_tools = ["business_intelligence"]

        return TaskAnalysis(
            original_task=task,
            complexity=complexity,
            required_capabilities=["analysis", "generation", "orchestration"],
            required_tools=required_tools,
            estimated_duration=300 if complexity == TaskComplexity.MODERATE else 600,
            orchestration_strategy=OrchestrationStrategy.SEQUENTIAL if len(required_tools) <= 2 else OrchestrationStrategy.PARALLEL,
            subtasks=[{"task": task, "priority": 1, "tools": required_tools}],
            success_criteria=["Task completed successfully", "All required outputs generated"]
        )


class AgentSelectionIntelligence:
    """
    Selects optimal specialized agents for task execution.

    This intelligence system analyzes task requirements and selects the best
    combination of specialized agents to handle the work.
    """

    def __init__(self, agent_id: str, llm: BaseLanguageModel):
        self.agent_id = agent_id
        self.llm = llm
        self.selection_history: List[AgentSelection] = []

        # Tool-to-agent mapping (will be enhanced in Phase 2)
        self.tool_agent_mapping = {
            "business_intelligence": "BusinessIntelligenceAgent",
            "revolutionary_document_intelligence": "DocumentIntelligenceAgent",
            "ai_music_composition": "MusicCompositionAgent",
            "social_media_orchestrator": "SocialMediaEmpireAgent",
            "revolutionary_web_scraper": "WebScrapingAgent",
            "browser_automation": "BrowserAutomationAgent"
        }

    async def select_agents(self, task_analysis: TaskAnalysis, context: Dict[str, Any]) -> AgentSelection:
        """Select optimal agents for the analyzed task."""
        try:
            selected_agents = []
            selection_reasoning = []

            # Select agents based on required tools
            for tool_name in task_analysis.required_tools:
                if tool_name in self.tool_agent_mapping:
                    agent_type = self.tool_agent_mapping[tool_name]
                    agent_config = {
                        "agent_type": agent_type,
                        "tool_name": tool_name,
                        "specialization": self._get_agent_specialization(tool_name),
                        "autonomy_level": self._determine_autonomy_level(tool_name, task_analysis.complexity),
                        "capabilities": self._get_agent_capabilities(tool_name)
                    }
                    selected_agents.append(agent_config)
                    selection_reasoning.append(f"Selected {agent_type} for {tool_name} based on task requirements")

            # Calculate confidence based on agent-tool matching
            confidence = min(0.9, 0.6 + (len(selected_agents) * 0.1))

            selection = AgentSelection(
                task_analysis=task_analysis,
                selected_agents=selected_agents,
                selection_reasoning=selection_reasoning,
                confidence_score=confidence,
                fallback_agents=self._get_fallback_agents(selected_agents)
            )

            self.selection_history.append(selection)
            logger.info(f"Agent selection completed: {len(selected_agents)} agents selected with {confidence:.2f} confidence")

            return selection

        except Exception as e:
            logger.error(f"Agent selection failed: {str(e)}")
            # Return fallback selection
            return AgentSelection(
                task_analysis=task_analysis,
                selected_agents=[{
                    "agent_type": "GeneralAgent",
                    "tool_name": "business_intelligence",
                    "specialization": "general",
                    "autonomy_level": "autonomous",
                    "capabilities": ["analysis", "generation"]
                }],
                selection_reasoning=["Fallback to general agent due to selection error"],
                confidence_score=0.5
            )

    def _get_agent_specialization(self, tool_name: str) -> str:
        """Get specialization level for tool-specific agent."""
        specializations = {
            "business_intelligence": "financial_analysis",
            "revolutionary_document_intelligence": "document_processing",
            "ai_music_composition": "creative_composition",
            "social_media_orchestrator": "multi_platform_management",
            "revolutionary_web_scraper": "data_harvesting"
        }
        return specializations.get(tool_name, "general")

    def _determine_autonomy_level(self, tool_name: str, complexity: TaskComplexity) -> str:
        """Determine appropriate autonomy level for agent."""
        if complexity == TaskComplexity.REVOLUTIONARY:
            return "autonomous"
        elif complexity == TaskComplexity.COMPLEX:
            return "proactive"
        else:
            return "reactive"

    def _get_agent_capabilities(self, tool_name: str) -> List[str]:
        """Get capabilities for tool-specific agent."""
        capabilities_map = {
            "business_intelligence": ["analysis", "calculation", "reporting"],
            "revolutionary_document_intelligence": ["document_generation", "formatting", "export"],
            "ai_music_composition": ["creative_generation", "audio_processing", "composition"],
            "social_media_orchestrator": ["multi_platform", "content_optimization", "analytics"],
            "revolutionary_web_scraper": ["data_extraction", "web_navigation", "content_analysis"]
        }
        return capabilities_map.get(tool_name, ["general"])

    def _get_fallback_agents(self, primary_agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get fallback agents in case primary agents fail."""
        return [{
            "agent_type": "GeneralAgent",
            "tool_name": "business_intelligence",
            "specialization": "general",
            "autonomy_level": "autonomous",
            "capabilities": ["analysis", "generation"]
        }]


class WorkflowOrchestrator:
    """
    Orchestrates multi-agent workflows and coordinates execution.

    This orchestrator manages the execution of complex workflows involving
    multiple specialized agents working together.
    """

    def __init__(self, agent_id: str, llm: BaseLanguageModel):
        self.agent_id = agent_id
        self.llm = llm
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []

    async def create_workflow(self, agent_selection: AgentSelection, context: Dict[str, Any]) -> WorkflowExecution:
        """Create an execution workflow from agent selection."""
        try:
            execution_steps = []
            coordination_points = []

            # Create execution steps based on orchestration strategy
            strategy = agent_selection.task_analysis.orchestration_strategy

            if strategy == OrchestrationStrategy.SEQUENTIAL:
                # Sequential execution - one agent after another
                for i, agent_config in enumerate(agent_selection.selected_agents):
                    step = {
                        "step_id": f"step_{i+1}",
                        "agent_config": agent_config,
                        "dependencies": [f"step_{i}"] if i > 0 else [],
                        "expected_duration": 120,
                        "success_criteria": ["Agent execution completed successfully"]
                    }
                    execution_steps.append(step)

                    # Add coordination point between steps
                    if i > 0:
                        coordination_points.append({
                            "point_id": f"coord_{i}",
                            "type": "result_handoff",
                            "from_step": f"step_{i}",
                            "to_step": f"step_{i+1}",
                            "data_transfer": "execution_results"
                        })

            elif strategy == OrchestrationStrategy.PARALLEL:
                # Parallel execution - all agents simultaneously
                for i, agent_config in enumerate(agent_selection.selected_agents):
                    step = {
                        "step_id": f"parallel_step_{i+1}",
                        "agent_config": agent_config,
                        "dependencies": [],
                        "expected_duration": 180,
                        "success_criteria": ["Agent execution completed successfully"]
                    }
                    execution_steps.append(step)

                # Add final coordination point to combine results
                coordination_points.append({
                    "point_id": "final_coordination",
                    "type": "result_aggregation",
                    "from_steps": [step["step_id"] for step in execution_steps],
                    "data_transfer": "combined_results"
                })

            # Create context sharing plan
            context_sharing_plan = {
                "shared_context": {
                    "task_description": agent_selection.task_analysis.original_task,
                    "success_criteria": agent_selection.task_analysis.success_criteria,
                    "execution_context": context
                },
                "result_sharing": "all_agents_receive_previous_results",
                "coordination_method": "context_manager"
            }

            # Create monitoring checkpoints
            monitoring_checkpoints = [
                {
                    "checkpoint_id": "start",
                    "type": "execution_start",
                    "validation": "all_agents_ready"
                },
                {
                    "checkpoint_id": "midpoint",
                    "type": "progress_check",
                    "validation": "agents_progressing_normally"
                },
                {
                    "checkpoint_id": "completion",
                    "type": "execution_complete",
                    "validation": "all_success_criteria_met"
                }
            ]

            workflow = WorkflowExecution(
                agent_selection=agent_selection,
                execution_steps=execution_steps,
                coordination_points=coordination_points,
                context_sharing_plan=context_sharing_plan,
                monitoring_checkpoints=monitoring_checkpoints,
                rollback_strategy={
                    "strategy": "step_by_step_rollback",
                    "fallback_agent": "GeneralAgent",
                    "max_retries": 3
                }
            )

            self.active_executions[workflow.execution_id] = workflow
            logger.info(f"Workflow created: {workflow.execution_id} with {len(execution_steps)} steps")

            return workflow

        except Exception as e:
            logger.error(f"Workflow creation failed: {str(e)}")
            raise AgentExecutionError(f"Failed to create workflow: {str(e)}")

    async def execute_workflow(self, workflow: WorkflowExecution, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the orchestrated workflow."""
        try:
            logger.info(f"Starting workflow execution: {workflow.execution_id}")

            # Initialize execution context
            execution_context = {
                "workflow_id": workflow.execution_id,
                "start_time": datetime.utcnow(),
                "shared_context": workflow.context_sharing_plan["shared_context"],
                "step_results": {},
                "coordination_data": {}
            }

            # Execute based on orchestration strategy
            strategy = workflow.agent_selection.task_analysis.orchestration_strategy

            if strategy == OrchestrationStrategy.SEQUENTIAL:
                results = await self._execute_sequential_workflow(workflow, execution_context)
            elif strategy == OrchestrationStrategy.PARALLEL:
                results = await self._execute_parallel_workflow(workflow, execution_context)
            else:
                # Default to sequential
                results = await self._execute_sequential_workflow(workflow, execution_context)

            # Move to history
            self.execution_history.append(workflow)
            if workflow.execution_id in self.active_executions:
                del self.active_executions[workflow.execution_id]

            logger.info(f"Workflow execution completed: {workflow.execution_id}")
            return results

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "workflow_id": workflow.execution_id
            }

    async def _execute_sequential_workflow(self, workflow: WorkflowExecution, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps sequentially."""
        results = {"status": "success", "step_results": {}, "final_result": None}

        for step in workflow.execution_steps:
            try:
                logger.info(f"Executing step: {step['step_id']}")

                # For now, simulate agent execution (Phase 2 will implement actual agents)
                step_result = await self._simulate_agent_execution(step, execution_context)
                results["step_results"][step["step_id"]] = step_result

                # Update execution context with results
                execution_context["step_results"][step["step_id"]] = step_result

                logger.info(f"Step completed: {step['step_id']}")

            except Exception as e:
                logger.error(f"Step execution failed: {step['step_id']} - {str(e)}")
                results["status"] = "partial_failure"
                results["failed_step"] = step["step_id"]
                results["error"] = str(e)
                break

        # Combine results for final output
        if results["status"] == "success":
            results["final_result"] = self._combine_step_results(results["step_results"])

        return results

    async def _execute_parallel_workflow(self, workflow: WorkflowExecution, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps in parallel."""
        results = {"status": "success", "step_results": {}, "final_result": None}

        # Create tasks for parallel execution
        tasks = []
        for step in workflow.execution_steps:
            task = asyncio.create_task(self._simulate_agent_execution(step, execution_context))
            tasks.append((step["step_id"], task))

        # Wait for all tasks to complete
        for step_id, task in tasks:
            try:
                step_result = await task
                results["step_results"][step_id] = step_result
                logger.info(f"Parallel step completed: {step_id}")
            except Exception as e:
                logger.error(f"Parallel step failed: {step_id} - {str(e)}")
                results["status"] = "partial_failure"
                results[f"error_{step_id}"] = str(e)

        # Combine results for final output
        if results["step_results"]:
            results["final_result"] = self._combine_step_results(results["step_results"])

        return results

    async def _simulate_agent_execution(self, step: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent execution (will be replaced with real agents in Phase 2)."""
        agent_config = step["agent_config"]

        # Simulate execution based on agent type
        await asyncio.sleep(1)  # Simulate processing time

        return {
            "agent_type": agent_config["agent_type"],
            "tool_used": agent_config["tool_name"],
            "execution_status": "completed",
            "output": f"Simulated output from {agent_config['agent_type']}",
            "execution_time": 1.0,
            "success": True
        }

    def _combine_step_results(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple steps into final output."""
        combined = {
            "execution_summary": f"Completed {len(step_results)} steps successfully",
            "agents_used": [result.get("agent_type", "Unknown") for result in step_results.values()],
            "tools_used": [result.get("tool_used", "Unknown") for result in step_results.values()],
            "total_execution_time": sum(result.get("execution_time", 0) for result in step_results.values()),
            "all_outputs": [result.get("output", "") for result in step_results.values()]
        }
        return combined


class CrossAgentContextManager:
    """
    Manages context sharing and coordination between specialized agents.

    This manager ensures that agents can share context, results, and
    coordinate their activities effectively.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.shared_contexts: Dict[str, Dict[str, Any]] = {}
        self.agent_communications: List[Dict[str, Any]] = []
        self.coordination_events: List[Dict[str, Any]] = []

    async def create_shared_context(self, workflow_id: str, initial_context: Dict[str, Any]) -> str:
        """Create a shared context for multi-agent workflow."""
        context_id = f"context_{workflow_id}_{uuid.uuid4().hex[:8]}"

        self.shared_contexts[context_id] = {
            "workflow_id": workflow_id,
            "created_at": datetime.utcnow(),
            "initial_context": initial_context,
            "shared_data": {},
            "agent_contributions": {},
            "coordination_state": "active"
        }

        logger.info(f"Shared context created: {context_id} for workflow {workflow_id}")
        return context_id

    async def update_shared_context(self, context_id: str, agent_id: str, data: Dict[str, Any]) -> bool:
        """Update shared context with agent contribution."""
        try:
            if context_id not in self.shared_contexts:
                logger.error(f"Shared context not found: {context_id}")
                return False

            context = self.shared_contexts[context_id]

            # Add agent contribution
            if agent_id not in context["agent_contributions"]:
                context["agent_contributions"][agent_id] = []

            contribution = {
                "timestamp": datetime.utcnow(),
                "data": data,
                "contribution_type": data.get("type", "result")
            }

            context["agent_contributions"][agent_id].append(contribution)

            # Update shared data
            if "results" not in context["shared_data"]:
                context["shared_data"]["results"] = {}

            context["shared_data"]["results"][agent_id] = data

            logger.info(f"Context updated by agent {agent_id} in context {context_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update shared context: {str(e)}")
            return False

    async def get_shared_context(self, context_id: str, requesting_agent_id: str) -> Optional[Dict[str, Any]]:
        """Get shared context for an agent."""
        try:
            if context_id not in self.shared_contexts:
                return None

            context = self.shared_contexts[context_id]

            # Return context relevant to the requesting agent
            return {
                "workflow_id": context["workflow_id"],
                "initial_context": context["initial_context"],
                "shared_results": context["shared_data"].get("results", {}),
                "other_agent_contributions": {
                    aid: contributions for aid, contributions in context["agent_contributions"].items()
                    if aid != requesting_agent_id
                },
                "coordination_state": context["coordination_state"]
            }

        except Exception as e:
            logger.error(f"Failed to get shared context: {str(e)}")
            return None

    async def coordinate_agents(self, context_id: str, coordination_type: str, data: Dict[str, Any]) -> bool:
        """Coordinate between agents at specific points."""
        try:
            coordination_event = {
                "event_id": str(uuid.uuid4()),
                "context_id": context_id,
                "coordination_type": coordination_type,
                "timestamp": datetime.utcnow(),
                "data": data,
                "status": "active"
            }

            self.coordination_events.append(coordination_event)

            # Handle different coordination types
            if coordination_type == "result_handoff":
                await self._handle_result_handoff(context_id, data)
            elif coordination_type == "result_aggregation":
                await self._handle_result_aggregation(context_id, data)
            elif coordination_type == "synchronization":
                await self._handle_synchronization(context_id, data)

            logger.info(f"Coordination event processed: {coordination_type} for context {context_id}")
            return True

        except Exception as e:
            logger.error(f"Coordination failed: {str(e)}")
            return False

    async def _handle_result_handoff(self, context_id: str, data: Dict[str, Any]):
        """Handle result handoff between sequential agents."""
        # Update shared context with handoff data
        if context_id in self.shared_contexts:
            context = self.shared_contexts[context_id]
            if "handoffs" not in context["shared_data"]:
                context["shared_data"]["handoffs"] = []

            context["shared_data"]["handoffs"].append({
                "timestamp": datetime.utcnow(),
                "from_agent": data.get("from_agent"),
                "to_agent": data.get("to_agent"),
                "data": data.get("handoff_data", {})
            })

    async def _handle_result_aggregation(self, context_id: str, data: Dict[str, Any]):
        """Handle aggregation of results from parallel agents."""
        if context_id in self.shared_contexts:
            context = self.shared_contexts[context_id]

            # Collect all agent results
            all_results = context["shared_data"].get("results", {})

            # Create aggregated result
            aggregated = {
                "aggregation_timestamp": datetime.utcnow(),
                "participating_agents": list(all_results.keys()),
                "combined_results": all_results,
                "aggregation_summary": f"Combined results from {len(all_results)} agents"
            }

            context["shared_data"]["aggregated_result"] = aggregated

    async def _handle_synchronization(self, context_id: str, data: Dict[str, Any]):
        """Handle synchronization between agents."""
        if context_id in self.shared_contexts:
            context = self.shared_contexts[context_id]

            # Update synchronization state
            if "synchronization" not in context["shared_data"]:
                context["shared_data"]["synchronization"] = []

            context["shared_data"]["synchronization"].append({
                "timestamp": datetime.utcnow(),
                "sync_type": data.get("sync_type", "general"),
                "participating_agents": data.get("agents", []),
                "sync_data": data.get("sync_data", {})
            })

    async def cleanup_context(self, context_id: str):
        """Clean up shared context after workflow completion."""
        try:
            if context_id in self.shared_contexts:
                context = self.shared_contexts[context_id]
                context["coordination_state"] = "completed"
                context["completed_at"] = datetime.utcnow()

                # Archive context (in production, this might go to persistent storage)
                logger.info(f"Context archived: {context_id}")

                # Remove from active contexts after a delay
                await asyncio.sleep(60)  # Keep for 1 minute for any final access
                if context_id in self.shared_contexts:
                    del self.shared_contexts[context_id]
                    logger.info(f"Context cleaned up: {context_id}")

        except Exception as e:
            logger.error(f"Context cleanup failed: {str(e)}")
