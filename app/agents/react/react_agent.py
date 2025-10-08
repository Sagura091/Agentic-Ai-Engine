"""
🚀 REVOLUTIONARY TRUE REACT AGENT IMPLEMENTATION

This module implements authentic ReAct (Reasoning and Acting) agents that follow
the TRUE ReAct pattern with AUTONOMOUS INTELLIGENCE:

1. THOUGHT: Pure reasoning using regular LLM with PERSISTENT MEMORY
2. DECISION: Metadata-driven decision engine (from autonomous agents)
3. ACTION: Execute tools OR provide conversational response based on decision
4. LEARNING: Adaptive learning from every interaction

This is user-driven agentic AI with autonomous intelligence - responds to user
queries with intelligent reasoning, learns from experience, and remembers context.

LEVERAGES AUTONOMOUS AGENT CAPABILITIES:
- MetadataDrivenDecisionEngine: Sophisticated decision-making
- AdaptiveLearningSystem: Learning from experience
- PersistentMemorySystem: Remembering context and preferences
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

from app.agents.base.agent import LangGraphAgent, AgentConfig, AgentGraphState
from app.core.exceptions import AgentExecutionError

# Import autonomous agent capabilities
from app.agents.autonomous.metadata_driven_decision_engine import MetadataDrivenDecisionEngine, DecisionPattern
from app.agents.autonomous.learning_system import AdaptiveLearningSystem
from app.agents.autonomous.persistent_memory import PersistentMemorySystem, MemoryType, MemoryImportance, MemoryTrace
from app.agents.autonomous.decision_engine import DecisionOption, DecisionResult

# Import backend logging system
from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

# Get backend logger instance
_backend_logger = get_backend_logger()


# ============================================================================
# REACT AGENT STATE
# ============================================================================

class ReActAgentState(AgentGraphState):
    """
    Enhanced state for ReAct agents with thought/decision/action tracking.
    """
    # ReAct-specific state
    current_thought: str  # Current reasoning/thought
    current_decision: Dict[str, Any]  # Current decision result
    react_history: List[Dict[str, Any]]  # History of thought/decision/action cycles


# ============================================================================
# REACT AGENT CONFIGURATION
# ============================================================================

class ReActAgentConfig(AgentConfig):
    """Enhanced configuration for ReAct agents."""
    
    # ReAct-specific settings
    enable_thought_logging: bool = Field(default=True, description="Log thought process to user")
    enable_decision_logging: bool = Field(default=True, description="Log decisions to user")
    decision_confidence_threshold: float = Field(default=0.6, description="Minimum confidence for decisions")
    max_thought_iterations: int = Field(default=3, description="Max iterations for thought refinement")
    
    # Override default system prompt with ReAct-specific one
    system_prompt: str = Field(
        default="""You are a ReAct (Reasoning and Acting) AI agent - a revolutionary system that thinks before acting.

🧠 REACT PHILOSOPHY:
You follow the Thought → Decision → Action cycle:
1. THOUGHT: Analyze the user's request deeply
2. DECISION: Decide if tools are needed or if you can respond directly
3. ACTION: Either use tools OR provide a conversational response

AVAILABLE TOOLS: {tools}

🎯 WHEN TO USE TOOLS:
Use tools when the task requires specialized capabilities:
- Document creation/manipulation (Excel, Word, PDF)
- Complex calculations or data processing
- Web research for current information
- File operations or system tasks
- Specialized domain-specific operations

💬 WHEN TO RESPOND CONVERSATIONALLY:
Respond directly (NO tools) for:
- Greetings and casual conversation ("hi", "hello", "how are you")
- Simple questions you can answer from your knowledge
- Clarifications or follow-up questions
- General discussion or explanations
- Acknowledgments ("thanks", "ok", "got it")

🚀 YOUR SUPERPOWER:
You THINK before you act. You analyze whether a tool is truly needed, or if a simple conversation is better.
You are user-driven - you respond to what the user asks, not what you think they might want.

Be helpful, thoughtful, and choose the right approach for each situation.""",
        description="System prompt for the ReAct agent"
    )


# ============================================================================
# REACT LANGGRAPH AGENT
# ============================================================================

class ReActLangGraphAgent(LangGraphAgent):
    """
    🚀 Revolutionary TRUE ReAct Agent with Autonomous Intelligence

    Implements authentic Reasoning and Acting pattern with:
    - Separate thought step using regular LLM (no tool bias) + PERSISTENT MEMORY
    - Metadata-driven decision-making (from autonomous agents)
    - Action execution based on reasoned decisions
    - ADAPTIVE LEARNING from every interaction
    - Full conversation support without forced tool usage

    This is user-driven agentic AI that thinks before it acts, learns from experience,
    and remembers context - combining ReAct pattern with autonomous intelligence.
    """

    def __init__(
        self,
        config: ReActAgentConfig,
        llm: BaseLanguageModel,
        tools: Optional[List] = None,
        checkpoint_saver: Optional[BaseCheckpointSaver] = None
    ):
        """
        Initialize the ReAct agent with autonomous capabilities.

        Args:
            config: ReAct agent configuration
            llm: Language model instance
            tools: List of tools available to the agent
            checkpoint_saver: Optional checkpoint saver for state persistence
        """
        # Initialize base agent
        super().__init__(config, llm, tools, checkpoint_saver)

        self.react_config = config

        # 🚀 INITIALIZE AUTONOMOUS CAPABILITIES

        # 1. Persistent Memory System - Remember context and user preferences
        self.memory_system = PersistentMemorySystem(
            agent_id=self.agent_id,
            llm=llm,
            max_working_memory=20,
            max_episodic_memory=100,
            max_semantic_memory=50
        )

        # 2. Metadata-Driven Decision Engine - Sophisticated decision-making
        decision_config = {
            "decision_threshold": config.decision_confidence_threshold,
            "reasoning_penalty_per_iteration": 0.15,
            "tool_boost_for_execution_tasks": 0.4
        }
        self.decision_engine = MetadataDrivenDecisionEngine(decision_config)

        # 3. Adaptive Learning System - Learn from experience
        # Create a minimal autonomous config for the learning system
        from app.agents.autonomous.autonomous_agent import AutonomousAgentConfig
        learning_config = AutonomousAgentConfig(
            name=config.name,
            description=config.description,
            learning_mode="active",  # Enable active learning
            autonomy_level="reactive"  # User-driven, not proactive
        )
        self.learning_system = AdaptiveLearningSystem(learning_config)

        # Override the graph with ReAct-specific workflow
        self._build_react_graph()

        _backend_logger.info(
            "🚀 Revolutionary ReAct agent initialized with autonomous intelligence",
            LogCategory.AGENT_OPERATIONS,
            "app.agents.react.react_agent",
            data={
                "agent_id": self.agent_id,
                "tools_available": len(self.tools),
                "thought_logging": config.enable_thought_logging,
                "decision_logging": config.enable_decision_logging,
                "memory_enabled": True,
                "learning_enabled": True,
                "decision_engine": "metadata_driven"
            }
        )
    
    def _build_react_graph(self) -> None:
        """
        Build the ReAct-specific LangGraph workflow.
        
        Graph structure:
        START → thought → decision → action → END
                                  ↓
                              (loop back to thought if needed)
        """
        # Create the state graph with ReAct state
        self.graph = StateGraph(AgentGraphState)
        
        # Add ReAct nodes
        self.graph.add_node("thought", self._thought_node)
        self.graph.add_node("decision", self._decision_node)
        self.graph.add_node("action", self._action_node)
        self.graph.add_node("tool_execution", self._tool_execution_node)
        
        # Build the ReAct workflow
        self.graph.add_edge(START, "thought")
        self.graph.add_edge("thought", "decision")
        
        # Decision routes to either action or tool_execution
        self.graph.add_conditional_edges(
            "decision",
            self._route_after_decision,
            {
                "action": "action",
                "tool_execution": "tool_execution",
                "end": END
            }
        )
        
        # After action, check if we need to continue
        self.graph.add_conditional_edges(
            "action",
            self._should_continue_react,
            {
                "continue": "thought",
                "end": END
            }
        )
        
        # After tool execution, go back to thought for observation
        self.graph.add_edge("tool_execution", "thought")
        
        # Compile the graph
        if self.checkpoint_saver:
            self.compiled_graph = self.graph.compile(checkpointer=self.checkpoint_saver)
        else:
            self.compiled_graph = self.graph.compile()

        _backend_logger.info(
            "ReAct LangGraph workflow built",
            LogCategory.AGENT_OPERATIONS,
            "app.agents.react.react_agent",
            data={
                "agent_id": self.agent_id,
                "nodes": ["thought", "decision", "action", "tool_execution"]
            }
        )
    
    async def _thought_node(self, state: AgentGraphState) -> AgentGraphState:
        """
        🧠 THOUGHT NODE - Pure reasoning with PERSISTENT MEMORY

        The agent thinks about the task using:
        1. Regular LLM (NOT tool-bound) for unbiased reasoning
        2. PERSISTENT MEMORY to recall relevant past interactions
        3. Context from conversation history

        Args:
            state: Current graph state

        Returns:
            Updated state with thought
        """
        from app.core.clean_logging import get_conversation_logger

        conversation_logger = get_conversation_logger(self.name)
        thought_start_time = time.time()

        try:
            # 🧠 RETRIEVE RELEVANT MEMORIES
            relevant_memories = await self.memory_system.retrieve_relevant_memories(
                query=state['current_task'],
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
                limit=5
            )

            memory_context = ""
            if relevant_memories:
                memory_context = "RELEVANT PAST EXPERIENCES:\n"
                for mem in relevant_memories:
                    memory_context += f"- {mem.content}\n"

            # Build context from conversation history
            conversation_context = ""
            if state["messages"]:
                recent_messages = state["messages"][-5:]  # Last 5 messages for context
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        conversation_context += f"User: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        if hasattr(msg, 'content') and msg.content:
                            conversation_context += f"Assistant: {msg.content}\n"

            # Create thought prompt with memory
            tools_list = "\n".join([f"- {name}: {tool.description if hasattr(tool, 'description') else 'No description'}"
                                    for name, tool in self.tools.items()]) if self.tools else "No tools available"

            thought_prompt = f"""You are analyzing a user request to determine the best course of action.

CURRENT TASK: {state['current_task']}

{memory_context}

CONVERSATION HISTORY:
{conversation_context if conversation_context else "No previous conversation"}

AVAILABLE TOOLS:
{tools_list}

Think step-by-step about this task:
1. What is the user actually asking for?
2. Based on past experiences, what approach worked well?
3. Is this a simple conversational request (greeting, question, clarification) or does it require specialized tools?
4. If tools are needed, which specific tool(s) would be most appropriate and why?
5. If no tools are needed, what would be an appropriate conversational response?

Provide your reasoning in a clear, structured way. Be honest about what you can and cannot do."""

            # Use REGULAR LLM (not tool-bound) for pure reasoning
            reasoning_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a thoughtful AI assistant analyzing tasks to determine the best approach. You learn from past experiences."),
                ("human", "{prompt}")
            ])

            chain = reasoning_prompt | self.llm  # ← CRITICAL: Regular LLM, NOT self.llm_with_tools

            _backend_logger.debug(
                "Invoking LLM for THOUGHT step (pure reasoning with memory)",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": state["agent_id"],
                    "task": state["current_task"][:100],
                    "memories_retrieved": len(relevant_memories)
                }
            )

            response = await chain.ainvoke({"prompt": thought_prompt})

            # Extract text content
            if hasattr(response, 'content'):
                thought_text = response.content
            else:
                thought_text = str(response)

            thought_time_ms = (time.time() - thought_start_time) * 1000

            _backend_logger.info(
                "THOUGHT step completed with memory",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": state["agent_id"],
                    "thought_length": len(thought_text),
                    "thought_time_ms": thought_time_ms,
                    "memories_used": len(relevant_memories)
                }
            )

            # Log thought to user if enabled
            if self.react_config.enable_thought_logging:
                conversation_logger.agent_thinking(f"💭 Thinking: {thought_text[:200]}...")

            # 💾 STORE THIS THOUGHT AS WORKING MEMORY
            await self.memory_system.store_memory(
                MemoryTrace(
                    content=f"Thought about: {state['current_task'][:100]} - {thought_text[:200]}",
                    memory_type=MemoryType.WORKING,
                    importance=MemoryImportance.TEMPORARY,
                    tags={"thought", "reasoning"},
                    session_id=state.get("session_id")
                )
            )

            # Update state
            updated_state = state.copy()
            updated_state["custom_state"]["current_thought"] = thought_text

            return updated_state

        except Exception as e:
            _backend_logger.error(
                "THOUGHT node failed",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": self.agent_id,
                    "error": str(e)
                }
            )
            updated_state = state.copy()
            updated_state["errors"].append(f"Thought failed: {str(e)}")
            return updated_state

    async def _decision_node(self, state: AgentGraphState) -> AgentGraphState:
        """
        🎯 DECISION NODE - Metadata-driven decision making

        Uses the MetadataDrivenDecisionEngine from autonomous agents for
        sophisticated, configuration-driven decision making.

        Decides whether to:
        - RESPOND: Provide a conversational response (no tools needed)
        - USE_TOOL: Use one or more tools to accomplish the task
        - CLARIFY: Ask for clarification from the user

        Args:
            state: Current graph state

        Returns:
            Updated state with decision
        """
        from app.core.clean_logging import get_conversation_logger

        conversation_logger = get_conversation_logger(self.name)
        decision_start_time = time.time()

        try:
            thought = state["custom_state"].get("current_thought", "")

            # 🚀 USE METADATA-DRIVEN DECISION ENGINE
            decision_context = {
                "task": state["current_task"],
                "thought": thought,
                "messages": state["messages"],
                "iteration": state["iteration_count"],
                "tools_available": list(self.tools.keys()),
                "session_id": state.get("session_id"),
                "agent_id": state["agent_id"]
            }

            _backend_logger.debug(
                "Using MetadataDrivenDecisionEngine for DECISION step",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": state["agent_id"],
                    "task": state["current_task"][:100],
                    "tools_available": len(self.tools)
                }
            )

            # Get decision from metadata-driven engine
            engine_decision: DecisionResult = await self.decision_engine.make_decision(decision_context)

            # Translate autonomous decision to ReAct decision format
            selected_option = engine_decision.selected_option

            # Determine decision type based on selected option
            # Handle different attribute names (action_type vs action)
            action_str = ""
            if hasattr(selected_option, 'action_type'):
                action_str = selected_option.action_type.lower()
            elif hasattr(selected_option, 'action'):
                action_str = selected_option.action.lower()
            elif hasattr(selected_option, 'name'):
                action_str = selected_option.name.lower()

            if "respond" in action_str or "conversation" in action_str or "greeting" in action_str:
                decision_type = "RESPOND"
                selected_tools = []
            elif "tool" in action_str or "execute" in action_str or "use" in action_str:
                decision_type = "USE_TOOL"
                # Extract tool names from the decision
                selected_tools = []
                if hasattr(selected_option, 'parameters') and isinstance(selected_option.parameters, dict):
                    selected_tools = selected_option.parameters.get("tools", [])
                if not selected_tools and hasattr(selected_option, 'tool_name'):
                    selected_tools = [selected_option.tool_name]
            else:
                decision_type = "CLARIFY"
                selected_tools = []

            # Validate tools if USE_TOOL decision
            if decision_type == "USE_TOOL" and selected_tools:
                # Filter to only valid tools
                valid_tools = [t for t in selected_tools if t in self.tools]
                if not valid_tools:
                    # No valid tools selected, fall back to RESPOND
                    decision_type = "RESPOND"
                    selected_tools = []

            decision_result = {
                "decision": decision_type,
                "confidence": engine_decision.confidence,
                "reasoning": engine_decision.reasoning,
                "selected_tools": selected_tools,
                "raw_response": str(engine_decision.selected_option)
            }

            decision_time_ms = (time.time() - decision_start_time) * 1000

            _backend_logger.info(
                "DECISION step completed",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": state["agent_id"],
                    "decision": decision_type,
                    "confidence": engine_decision.confidence,
                    "tools_selected": len(selected_tools),
                    "decision_time_ms": decision_time_ms
                }
            )

            # Log decision to user if enabled
            if self.react_config.enable_decision_logging:
                if decision_type == "RESPOND":
                    conversation_logger.agent_thinking(f"💬 Decision: Responding conversationally")
                elif decision_type == "USE_TOOL":
                    tools_str = ", ".join(selected_tools)
                    conversation_logger.agent_thinking(f"🔧 Decision: Using tools - {tools_str}")
                else:
                    conversation_logger.agent_thinking(f"❓ Decision: Requesting clarification")

            # Update state
            updated_state = state.copy()
            updated_state["custom_state"]["current_decision"] = decision_result

            return updated_state

        except Exception as e:
            _backend_logger.error(
                "DECISION node failed",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": self.agent_id,
                    "error": str(e)
                }
            )
            updated_state = state.copy()
            updated_state["errors"].append(f"Decision failed: {str(e)}")
            # Default to RESPOND on error
            updated_state["custom_state"]["current_decision"] = {
                "decision": "RESPOND",
                "confidence": 0.5,
                "reasoning": f"Error in decision making: {str(e)}",
                "selected_tools": []
            }
            return updated_state

    async def _action_node(self, state: AgentGraphState) -> AgentGraphState:
        """
        ⚡ ACTION NODE - Execute conversational response or clarification

        This node handles RESPOND and CLARIFY decisions.
        USE_TOOL decisions are routed to tool_execution_node instead.

        Args:
            state: Current graph state

        Returns:
            Updated state with action response
        """
        action_start_time = time.time()

        try:
            decision = state["custom_state"].get("current_decision", {})
            thought = state["custom_state"].get("current_thought", "")
            decision_type = decision.get("decision", "RESPOND")

            _backend_logger.debug(
                f"Executing ACTION: {decision_type}",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": state["agent_id"],
                    "decision": decision_type
                }
            )

            if decision_type == "RESPOND":
                # Generate conversational response using regular LLM
                response_prompt = f"""Based on your analysis, provide a helpful conversational response to the user.

YOUR REASONING:
{thought}

USER'S REQUEST: {state['current_task']}

Provide a natural, helpful response. Be conversational and friendly."""

                response_chain = ChatPromptTemplate.from_messages([
                    ("system", self.config.system_prompt.format(tools="None - conversational mode")),
                    MessagesPlaceholder(variable_name="messages"),
                    ("human", "{prompt}")
                ]) | self.llm  # Regular LLM for conversational response

                response = await response_chain.ainvoke({
                    "messages": state["messages"],
                    "prompt": response_prompt
                })

                action_time_ms = (time.time() - action_start_time) * 1000

                _backend_logger.info(
                    "ACTION: Conversational response generated",
                    LogCategory.AGENT_OPERATIONS,
                    "app.agents.react.react_agent",
                    data={
                        "agent_id": state["agent_id"],
                        "response_length": len(response.content) if hasattr(response, 'content') else 0,
                        "action_time_ms": action_time_ms
                    }
                )

            else:  # CLARIFY
                # Generate clarification request
                clarify_prompt = f"""Based on your analysis, ask the user for clarification.

YOUR REASONING:
{thought}

USER'S REQUEST: {state['current_task']}

Ask specific questions to clarify what the user needs."""

                clarify_chain = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant asking for clarification."),
                    MessagesPlaceholder(variable_name="messages"),
                    ("human", "{prompt}")
                ]) | self.llm

                response = await clarify_chain.ainvoke({
                    "messages": state["messages"],
                    "prompt": clarify_prompt
                })

                action_time_ms = (time.time() - action_start_time) * 1000

                _backend_logger.info(
                    "ACTION: Clarification request generated",
                    LogCategory.AGENT_OPERATIONS,
                    "app.agents.react.react_agent",
                    data={
                        "agent_id": state["agent_id"],
                        "response_length": len(response.content) if hasattr(response, 'content') else 0,
                        "action_time_ms": action_time_ms
                    }
                )

            # Update state
            updated_state = state.copy()
            updated_state["messages"] = state["messages"] + [response]
            updated_state["iteration_count"] += 1

            # 🎓 LEARN FROM THIS INTERACTION
            await self._learn_from_interaction(
                task=state["current_task"],
                thought=state["custom_state"].get("current_thought", ""),
                decision=decision,
                action_type=decision_type,
                response=response,
                success=True
            )

            # 💾 STORE THIS INTERACTION AS EPISODIC MEMORY
            await self.memory_system.store_memory(
                MemoryTrace(
                    content=f"User asked: {state['current_task'][:100]} | Decision: {decision_type} | Response: {response.content[:100] if hasattr(response, 'content') else 'N/A'}",
                    memory_type=MemoryType.EPISODIC,
                    importance=MemoryImportance.MEDIUM,
                    tags={"interaction", decision_type.lower()},
                    session_id=state.get("session_id"),
                    emotional_valence=0.5  # Positive interaction
                )
            )

            return updated_state

        except Exception as e:
            _backend_logger.error(
                "ACTION node failed",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": self.agent_id,
                    "error": str(e)
                }
            )

            # Learn from failure
            await self._learn_from_interaction(
                task=state.get("current_task", ""),
                thought=state["custom_state"].get("current_thought", ""),
                decision=state["custom_state"].get("current_decision", {}),
                action_type="ERROR",
                response=None,
                success=False,
                error=str(e)
            )

            updated_state = state.copy()
            updated_state["errors"].append(f"Action failed: {str(e)}")
            return updated_state

    async def _learn_from_interaction(
        self,
        task: str,
        thought: str,
        decision: Dict[str, Any],
        action_type: str,
        response: Optional[AIMessage],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        🎓 LEARN FROM INTERACTION - Adaptive learning

        Uses the AdaptiveLearningSystem to learn from each interaction.

        Args:
            task: The user's task
            thought: The agent's thought
            decision: The decision made
            action_type: Type of action taken
            response: The response generated
            success: Whether the interaction was successful
            error: Error message if failed
        """
        try:
            # Create experience record
            experience = {
                "task": task,
                "thought": thought,
                "decision": decision,
                "action_type": action_type,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "error": error
            }

            # Record the experience in the learning system
            await self.learning_system.record_experience(
                experience_type="react_interaction",
                context=experience,
                outcome="success" if success else "failure",
                performance_metrics={
                    "decision_confidence": decision.get("confidence", 0.5),
                    "action_type": action_type
                }
            )

            # Periodically analyze and adapt (every 10 interactions)
            if self.learning_system.learning_stats["total_experiences"] % 10 == 0:
                insights = await self.learning_system.analyze_and_learn()
                if insights:
                    _backend_logger.info(
                        "Learning insights generated",
                        LogCategory.AGENT_OPERATIONS,
                        "app.agents.react.react_agent",
                        data={
                            "agent_id": self.agent_id,
                            "insights_count": len(insights)
                        }
                    )

        except Exception as e:
            _backend_logger.warn(
                "Learning from interaction failed",
                LogCategory.AGENT_OPERATIONS,
                "app.agents.react.react_agent",
                data={
                    "agent_id": self.agent_id,
                    "error": str(e)
                }
            )

    def _route_after_decision(self, state: AgentGraphState) -> str:
        """
        Route after decision node based on decision type.

        Args:
            state: Current graph state

        Returns:
            Next node name: "action", "tool_execution", or "end"
        """
        decision = state["custom_state"].get("current_decision", {})
        decision_type = decision.get("decision", "RESPOND")

        if decision_type == "USE_TOOL":
            return "tool_execution"
        elif decision_type in ["RESPOND", "CLARIFY"]:
            return "action"
        else:
            return "end"

    def _should_continue_react(self, state: AgentGraphState) -> str:
        """
        Determine if ReAct cycle should continue or end.

        Args:
            state: Current graph state

        Returns:
            "continue" or "end"
        """
        # Check iteration limit
        if state["iteration_count"] >= self.config.max_iterations:
            return "end"

        # Check if there are errors
        if state["errors"]:
            return "end"

        # Check if last message has tool calls (need to observe results)
        if state["messages"]:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "continue"  # Continue to observe tool results

        # Otherwise, end
        return "end"

