"""
Enhanced Autonomous Agent System with True Agentic Capabilities.

This module implements revolutionary autonomous agents with self-directed
decision-making, adaptive learning, and emergent intelligence capabilities
built on LangChain/LangGraph foundation.
"""

import asyncio
import json
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
        agent_id: Optional[str] = None
    ):
        """Initialize the autonomous agent."""
        # Initialize base agent
        super().__init__(config, llm, tools, checkpoint_saver)

        # Set agent ID
        self.agent_id = agent_id or str(uuid.uuid4())

        # Enhanced autonomous capabilities
        self.autonomous_config = config
        self.decision_engine = AutonomousDecisionEngine(config, llm)
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
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnableConfig

        backend_logger = get_logger()
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

        # Log task initiation
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

                # Log successful completion
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
            decision_context = {
                "current_goals": state.get("goal_stack", []),
                "available_actions": self._get_available_actions(state),
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

        # CRITICAL FIX: End execution if we've completed the task successfully
        # Check if we've used tools and have outputs - indicates successful execution
        if tools_used > 0 and state.get("outputs"):
            logger.info(f"Task completed successfully with {tools_used} tools used, ending execution")
            return "end"

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

    def _get_available_actions(self, state: AutonomousAgentState) -> Dict[str, Any]:
        """Get available actions based on current state - dynamically generate tool parameters."""
        # Extract context for dynamic parameter generation
        current_task = state.get("current_task", "")
        goal_stack = state.get("goal_stack", [])

        # Dynamically generate tool parameters based on context
        tools_with_params = []
        for tool_name in state["tools_available"]:
            suggested_params = self._generate_tool_parameters(tool_name, current_task, goal_stack, state)
            tools_with_params.append({
                "name": tool_name,
                "suggested_params": suggested_params,
                "confidence": 0.7 if suggested_params else 0.3  # Higher confidence if we have good params
            })

        return {
            "tools": tools_with_params,
            "reasoning": {"available": True, "confidence": 0.8},
            "goal_actions": [{"action": "pursue_goal", "goal": goal} for goal in goal_stack],
            "exploration": {"available": self.config.learning_mode != "disabled", "risk_level": 0.5}
        }

    def _generate_tool_parameters(self, tool_name: str, current_task: str, goal_stack: list, state: AutonomousAgentState) -> Dict[str, Any]:
        """Dynamically generate appropriate parameters for tools based on current context."""
        params = {}

        # Extract key information from task and goals
        task_lower = current_task.lower()

        if tool_name == "web_research":
            # Generate search queries based on task content
            if "apple" in task_lower and "stock" in task_lower:
                params = {
                    "action": "search",
                    "query": "Apple stock price AAPL current market analysis",
                    "max_results": 5
                }
            elif "monitor" in task_lower:
                params = {
                    "action": "search",
                    "query": "real-time stock monitoring Apple AAPL",
                    "max_results": 3
                }
            else:
                # Extract key terms from task for generic search
                key_terms = self._extract_key_terms(current_task)
                if key_terms:
                    params = {
                        "action": "search",
                        "query": " ".join(key_terms[:5]),  # Use top 5 key terms
                        "max_results": 5
                    }

        elif tool_name == "business_intelligence":
            # Generate business analysis parameters
            if "apple" in task_lower:
                params = {
                    "action": "analyze_company",
                    "company": "Apple Inc",
                    "analysis_type": "stock_performance"
                }
            elif "stock" in task_lower:
                params = {
                    "action": "market_analysis",
                    "focus": "stock_market_trends"
                }

        # Add goal-based parameters if available
        if goal_stack:
            for goal in goal_stack:
                if isinstance(goal, dict):
                    goal_title = goal.get("title", "").lower()
                    if "apple" in goal_title and tool_name == "web_research":
                        params.update({
                            "context": "Apple stock monitoring for investment timing",
                            "priority": "high"
                        })

        return params

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


# Supporting classes that need to be imported
class AutonomousDecisionEngine:
    """Placeholder for decision engine - implemented in decision_engine.py"""
    def __init__(self, config, llm):
        self.config = config
        self.llm = llm

    async def make_autonomous_decision(self, context, confidence_threshold):
        # This will be replaced by the actual implementation
        from app.agents.autonomous.decision_engine import AutonomousDecisionEngine as RealEngine
        real_engine = RealEngine(self.config, self.llm)
        return await real_engine.make_autonomous_decision(context, confidence_threshold)


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
