"""
LangChain/LangGraph-based agent system for the Agentic AI Microservice.

This module provides the foundational classes that integrate deeply with LangChain
and LangGraph, ensuring all agents are built on the proper LangChain foundation
with full LangGraph workflow capabilities.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Sequence

import structlog
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from app.core.exceptions import AgentExecutionError, AgentTimeoutError

logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    """Agent execution status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AgentCapability(str, Enum):
    """Agent capability enumeration."""
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    PLANNING = "planning"
    COLLABORATION = "collaboration"
    LEARNING = "learning"
    MULTIMODAL = "multimodal"


# LangGraph State Definition
class AgentGraphState(TypedDict):
    """
    LangGraph state definition for agent workflows.

    This defines the state structure that flows through the LangGraph nodes.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
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


class AgentDNA(BaseModel):
    """Agent DNA configuration for personality and behavior."""
    identity: Dict[str, Any] = Field(default_factory=dict, description="Identity configuration")
    cognition: Dict[str, Any] = Field(default_factory=dict, description="Cognitive configuration")
    behavior: Dict[str, Any] = Field(default_factory=dict, description="Behavioral configuration")


class FrameworkConfig(BaseModel):
    """Framework-specific configuration."""
    framework_id: str = Field(..., description="Framework identifier")
    components: List[Dict[str, Any]] = Field(default_factory=list, description="Framework components")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Framework-specific settings")


class AgentConfig(BaseModel):
    """Enhanced LangChain/LangGraph-based agent configuration with multi-framework support."""

    # Basic configuration
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    version: str = Field(default="1.0.0", description="Agent version")
    agent_type: str = Field(default="basic", description="Agent type")
    framework: str = Field(default="basic", description="Agent framework (basic, react, bdi, crewai, autogen, swarm)")

    # Agent DNA configuration
    agent_dna: Optional[AgentDNA] = Field(default=None, description="Agent DNA configuration")
    framework_config: Optional[FrameworkConfig] = Field(default=None, description="Framework configuration")

    # LangChain LLM configuration
    model_name: str = Field(default="llama3.2:latest", description="LLM model to use")
    model_provider: str = Field(default="ollama", description="LLM provider (ollama, openai, anthropic, google)")
    temperature: float = Field(default=0.7, description="Model temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, description="Maximum tokens", gt=0)

    # Additional LLM parameters
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter", gt=0)
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty", ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty", ge=-2.0, le=2.0)

    # Provider-specific credentials (for runtime override)
    llm_credentials: Optional[Dict[str, Any]] = Field(default=None, description="LLM provider credentials")

    # LangChain prompt configuration
    system_prompt: str = Field(
        default="""You are an intelligent AI agent with access to powerful tools to help solve problems.

AVAILABLE TOOLS: {tools}

You have access to these tools and should use them naturally when they would be helpful for the task. Analyze the user's request and determine which tools would be most appropriate to provide accurate, comprehensive answers.

Guidelines:
- Use tools when they would provide better, more accurate, or more current information than your training data
- For research tasks, web search tools can provide the latest information
- For calculations, use computational tools for accuracy
- Choose tools based on what would best serve the user's needs

Think through the task and use the most appropriate tools to provide the best possible response.""",
        description="System prompt for the agent"
    )

    # Capabilities
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")

    # LangChain tools configuration
    tools: List[str] = Field(default_factory=list, description="Available tool names")
    max_tool_calls: int = Field(default=10, description="Maximum tool calls per execution")

    # LangGraph execution configuration
    timeout_seconds: int = Field(default=300, description="Execution timeout")
    max_iterations: int = Field(default=50, description="Maximum reasoning iterations")

    # LangGraph checkpoint configuration
    enable_checkpoints: bool = Field(default=True, description="Enable LangGraph checkpoints")
    checkpoint_namespace: str = Field(default="agent", description="Checkpoint namespace")

    # Memory configuration
    memory_enabled: bool = Field(default=True, description="Enable memory persistence")
    memory_window: int = Field(default=10, description="Memory window size")

    # Collaboration configuration
    can_delegate: bool = Field(default=False, description="Can delegate to other agents")
    can_be_delegated_to: bool = Field(default=True, description="Can receive delegated tasks")

    # Custom configuration
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration")


class AgentState(BaseModel):
    """Agent state model for persistence and recovery."""
    
    # Identity
    agent_id: str = Field(..., description="Unique agent identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # Status
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # Execution context
    current_task: Optional[str] = Field(default=None, description="Current task description")
    execution_step: int = Field(default=0, description="Current execution step")
    
    # Messages and history
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Message history")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool call history")
    
    # Results and outputs
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Agent outputs")
    errors: List[str] = Field(default_factory=list, description="Error history")
    
    # Metrics
    execution_time_seconds: float = Field(default=0.0, description="Total execution time")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    
    # Custom state
    custom_state: Dict[str, Any] = Field(default_factory=dict, description="Custom state data")


class LangGraphAgent(ABC):
    """
    LangChain/LangGraph-based agent implementation.

    This is the core agent class that integrates deeply with LangChain and LangGraph,
    providing a proper foundation for agentic AI workflows with state management,
    tool integration, and graph-based execution.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        checkpoint_saver: Optional[BaseCheckpointSaver] = None
    ):
        """
        Initialize the LangGraph agent.

        Args:
            config: Agent configuration
            llm: LangChain language model
            tools: List of LangChain tools
            checkpoint_saver: LangGraph checkpoint saver for state persistence
        """
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.checkpoint_saver = checkpoint_saver

        # Initialize LangGraph components
        self.graph: Optional[StateGraph] = None
        self.compiled_graph: Optional[Runnable] = None
        self.tool_node: Optional[ToolNode] = None

        # CRITICAL: Handle tool calling - prefer manual approach for compatibility
        # Many models don't support function calling, so we use text-based tool calling
        if self.tools:
            # Always use manual tool calling for maximum compatibility
            self.llm_with_tools = self.llm
            self.supports_tool_calling = False
            logger.info(
                "Using manual tool calling for maximum compatibility",
                agent_id=self.agent_id,
                tools_available=list(self.tools.keys()),
                llm_type=type(self.llm).__name__
            )
        else:
            self.llm_with_tools = self.llm
            self.supports_tool_calling = False

        # State management
        self.current_state: Optional[AgentGraphState] = None
        self._execution_lock = asyncio.Lock()

        # Memory system (assigned by AgentBuilderFactory)
        self.memory_system: Optional[Any] = None
        self.memory_collection: Optional[Any] = None
        self.memory_type: Optional[str] = None

        # Initialize the agent graph
        self._build_agent_graph()

        logger.info(
            "LangGraph agent initialized",
            agent_id=self.agent_id,
            name=config.name,
            capabilities=config.capabilities,
            tools_count=len(self.tools),
        )
    
    @property
    def name(self) -> str:
        """Get agent name."""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Get agent description."""
        return self.config.description
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        return self.config.capabilities
    
    @property
    def is_busy(self) -> bool:
        """Check if agent is currently executing."""
        return self.state.status == AgentStatus.RUNNING

    @property
    def has_memory(self) -> bool:
        """Check if agent has a memory system assigned."""
        return self.memory_system is not None

    async def add_memory(self, content: str, memory_type: str = "short_term", metadata: Optional[Dict[str, Any]] = None):
        """
        Add a memory to the agent's memory system.

        Args:
            content: Memory content
            memory_type: Type of memory (short_term, long_term, episodic, semantic, etc.)
            metadata: Additional metadata
        """
        if not self.has_memory:
            logger.warning("No memory system assigned to agent", agent_id=self.agent_id)
            return

        try:
            if self.memory_type == "simple":
                # UnifiedMemorySystem
                from app.memory.memory_models import MemoryType as UnifiedMemoryType
                unified_type = UnifiedMemoryType.SHORT_TERM if memory_type == "short_term" else UnifiedMemoryType.LONG_TERM
                await self.memory_system.add_memory(self.agent_id, unified_type, content, metadata)
            elif self.memory_type == "advanced":
                # PersistentMemorySystem
                from app.agents.autonomous.persistent_memory import MemoryType as PersistentMemoryType, MemoryImportance
                persistent_type = getattr(PersistentMemoryType, memory_type.upper(), PersistentMemoryType.EPISODIC)
                await self.memory_system.store_memory(content, persistent_type, MemoryImportance.MEDIUM, metadata=metadata)

            logger.debug("Memory added successfully", agent_id=self.agent_id, memory_type=memory_type)
        except Exception as e:
            logger.error("Failed to add memory", agent_id=self.agent_id, error=str(e))

    async def retrieve_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve memories from the agent's memory system.

        Args:
            query: Search query
            limit: Maximum number of memories to retrieve

        Returns:
            List of memory entries
        """
        if not self.has_memory:
            logger.warning("No memory system assigned to agent", agent_id=self.agent_id)
            return []

        try:
            if self.memory_type == "simple":
                # UnifiedMemorySystem retrieval
                memories = await self.memory_system.search_memories(self.agent_id, query, limit)
                return [{"content": m.content, "metadata": m.metadata, "created_at": m.created_at} for m in memories]
            elif self.memory_type == "advanced":
                # PersistentMemorySystem retrieval
                memories = await self.memory_system.retrieve_memories(query, limit)
                return [{"content": m.content, "metadata": m.metadata, "created_at": m.created_at} for m in memories]

        except Exception as e:
            logger.error("Failed to retrieve memories", agent_id=self.agent_id, error=str(e))
            return []
    
    def _build_agent_graph(self) -> None:
        """
        Build the LangGraph workflow for the agent.

        This creates the core LangGraph structure with nodes for reasoning,
        tool execution, and decision making.
        """
        # Create the state graph
        self.graph = StateGraph(AgentGraphState)

        # Add nodes
        self.graph.add_node("reasoning", self._reasoning_node)
        self.graph.add_node("tool_execution", self._tool_execution_node)
        self.graph.add_node("decision", self._decision_node)

        # Add edges
        self.graph.add_edge("reasoning", "decision")
        self.graph.add_conditional_edges(
            "decision",
            self._should_continue,
            {
                "continue": "tool_execution",
                "end": END
            }
        )
        self.graph.add_edge("tool_execution", "reasoning")

        # Set entry point
        self.graph.set_entry_point("reasoning")

        # Compile the graph with checkpoint saver if available
        if self.checkpoint_saver:
            self.compiled_graph = self.graph.compile(
                checkpointer=self.checkpoint_saver
            )
        else:
            self.compiled_graph = self.graph.compile()

        # Initialize tool node
        if self.tools:
            self.tool_node = ToolNode(list(self.tools.values()))

        logger.info(
            "LangGraph workflow built",
            agent_id=self.agent_id,
            nodes=["reasoning", "tool_execution", "decision"],
            tools_available=len(self.tools)
        )

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a task using the LangGraph workflow with comprehensive logging.

        Args:
            task: Task description or instruction
            context: Additional context for execution
            **kwargs: Additional execution parameters

        Returns:
            Execution results from LangGraph

        Raises:
            AgentExecutionError: If execution fails
            AgentTimeoutError: If execution times out
        """
        # Import backend logger and time
        import time
        from app.backend_logging.backend_logger import get_logger
        from app.backend_logging.models import LogCategory, AgentMetrics, PerformanceMetrics
        from app.backend_logging.context import CorrelationContext, AgentContext

        backend_logger = get_logger()
        start_time = time.time()
        session_id = str(uuid.uuid4())

        # Set up agent context for logging
        agent_context = AgentContext(self.agent_id, self.config.agent_type)
        CorrelationContext.update_context(
            agent_id=self.agent_id,
            session_id=session_id,
            component="LangGraphAgent",
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
                # Create initial state
                initial_state = AgentGraphState(
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
                    custom_state=context or {}
                )

                # Log workflow initialization
                backend_logger.debug(
                    "LangGraph workflow initialized",
                    LogCategory.AGENT_OPERATIONS,
                    "LangGraphAgent",
                    data={
                        "agent_id": self.agent_id,
                        "session_id": session_id,
                        "initial_state_keys": list(initial_state.keys())
                    }
                )

                # Execute the LangGraph workflow
                config = RunnableConfig(
                    configurable={
                        "thread_id": session_id,
                        "checkpoint_ns": self.config.checkpoint_namespace
                    }
                )

                # Log workflow execution start
                backend_logger.info(
                    "Starting LangGraph workflow execution",
                    LogCategory.AGENT_OPERATIONS,
                    "LangGraphAgent",
                    data={
                        "agent_id": self.agent_id,
                        "session_id": session_id,
                        "workflow_config": config.get("configurable", {})
                    }
                )

                result = await self._execute_with_timeout(
                    self.compiled_graph.ainvoke(initial_state, config),
                    self.config.timeout_seconds
                )

                # Calculate execution metrics
                execution_time_ms = (time.time() - start_time) * 1000

                # Create agent metrics
                agent_metrics = AgentMetrics(
                    agent_type=self.config.agent_type,
                    agent_state="completed",
                    tools_used=list(set(call.get("tool", "") for call in result.get("tool_calls", []))),
                    tasks_completed=1,
                    execution_time_ms=execution_time_ms,
                    tokens_consumed=self._estimate_tokens(result.get("messages", []))
                )

                # Create performance metrics
                performance_metrics = PerformanceMetrics(
                    duration_ms=execution_time_ms,
                    memory_usage_mb=self._get_memory_usage(),
                    cpu_usage_percent=0.0,  # Could be enhanced with actual CPU monitoring
                    iterations_count=result.get("iteration_count", 0),
                    tools_called=len(result.get("tool_calls", []))
                )

                # Log successful completion
                backend_logger.info(
                    f"Agent task execution completed successfully",
                    LogCategory.AGENT_OPERATIONS,
                    "LangGraphAgent",
                    data={
                        "agent_id": self.agent_id,
                        "session_id": session_id,
                        "task": task,
                        "execution_time_ms": execution_time_ms,
                        "iterations": result.get("iteration_count", 0),
                        "tools_used": len(result.get("tool_calls", [])),
                        "messages_generated": len(result.get("messages", [])),
                        "outputs_keys": list(result.get("outputs", {}).keys()),
                        "errors_count": len(result.get("errors", []))
                    },
                    agent_metrics=agent_metrics,
                    performance=performance_metrics
                )

                # Extract results
                execution_result = {
                    "agent_id": self.agent_id,
                    "task": task,
                    "status": "completed",
                    "messages": result.get("messages", []),
                    "outputs": result.get("outputs", {}),
                    "tool_calls": result.get("tool_calls", []),
                    "iteration_count": result.get("iteration_count", 0),
                    "errors": result.get("errors", []),
                    "execution_time_ms": execution_time_ms,
                    "session_id": session_id
                }

                return execution_result

            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000

                # Create error agent metrics
                agent_metrics = AgentMetrics(
                    agent_type=self.config.agent_type,
                    agent_state="failed",
                    tools_used=[],
                    tasks_completed=0,
                    execution_time_ms=execution_time_ms,
                    tokens_consumed=0
                )

                # Log execution failure
                backend_logger.error(
                    f"Agent task execution failed: {str(e)}",
                    LogCategory.AGENT_OPERATIONS,
                    "LangGraphAgent",
                    data={
                        "agent_id": self.agent_id,
                        "session_id": session_id,
                        "task": task,
                        "execution_time_ms": execution_time_ms,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    agent_metrics=agent_metrics
                )

                logger.error(
                    "Agent execution failed",
                    agent_id=self.agent_id,
                    task=task,
                    error=str(e)
                )
                raise AgentExecutionError(self.agent_id, str(e))

    async def _reasoning_node(self, state: AgentGraphState) -> AgentGraphState:
        """
        LangGraph reasoning node - where the agent thinks and plans.

        Args:
            state: Current graph state

        Returns:
            Updated graph state
        """
        import time
        from app.backend_logging.backend_logger import get_logger
        from app.backend_logging.models import LogCategory

        backend_logger = get_logger()
        reasoning_start_time = time.time()

        try:
            # Log reasoning start
            backend_logger.debug(
                f"Agent reasoning iteration {state.get('iteration_count', 0) + 1} started",
                LogCategory.AGENT_OPERATIONS,
                "LangGraphAgent",
                data={
                    "agent_id": state["agent_id"],
                    "session_id": state.get("session_id"),
                    "iteration": state.get("iteration_count", 0) + 1,
                    "current_task": state["current_task"],
                    "tools_available": list(self.tools.keys()),
                    "messages_count": len(state["messages"])
                }
            )

            # Create the prompt template with tools information
            tools_description = ", ".join(self.tools.keys()) if self.tools else "None"
            formatted_system_prompt = self.config.system_prompt.format(tools=tools_description)

            prompt = ChatPromptTemplate.from_messages([
                ("system", formatted_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "Current task: {task}")
            ])

            # CRITICAL: Use the tool-bound LLM for proper tool calling
            chain = prompt | self.llm_with_tools

            # Log LLM invocation
            backend_logger.debug(
                "Invoking LLM for reasoning",
                LogCategory.AGENT_OPERATIONS,
                "LangGraphAgent",
                data={
                    "agent_id": state["agent_id"],
                    "session_id": state.get("session_id"),
                    "tools_description": tools_description,
                    "messages_in_context": len(state["messages"])
                }
            )

            # Get response from LLM
            response = await chain.ainvoke({
                "messages": state["messages"],
                "task": state["current_task"]
            })

            # If LLM doesn't support tool calling, manually parse for tool usage requests
            if not self.supports_tool_calling and self.tools:
                response = await self._handle_manual_tool_calling(response, state)

            # Update state
            updated_state = state.copy()
            updated_state["messages"] = state["messages"] + [response]
            updated_state["iteration_count"] += 1

            reasoning_time_ms = (time.time() - reasoning_start_time) * 1000

            # Log reasoning completion
            backend_logger.info(
                f"Agent reasoning iteration {updated_state['iteration_count']} completed",
                LogCategory.AGENT_OPERATIONS,
                "LangGraphAgent",
                data={
                    "agent_id": state["agent_id"],
                    "session_id": state.get("session_id"),
                    "iteration": updated_state["iteration_count"],
                    "reasoning_time_ms": reasoning_time_ms,
                    "response_type": type(response).__name__,
                    "response_length": len(str(response.content)) if hasattr(response, 'content') else 0,
                    "has_tool_calls": hasattr(response, 'tool_calls') and bool(response.tool_calls)
                }
            )

            logger.debug(
                "Reasoning node completed",
                agent_id=self.agent_id,
                iteration=updated_state["iteration_count"]
            )

            return updated_state

        except Exception as e:
            reasoning_time_ms = (time.time() - reasoning_start_time) * 1000

            # Log reasoning failure
            backend_logger.error(
                f"Agent reasoning iteration failed: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "LangGraphAgent",
                data={
                    "agent_id": state["agent_id"],
                    "session_id": state.get("session_id"),
                    "iteration": state.get("iteration_count", 0) + 1,
                    "reasoning_time_ms": reasoning_time_ms,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )

            logger.error(
                "Reasoning node failed",
                agent_id=self.agent_id,
                error=str(e)
            )
            updated_state = state.copy()
            updated_state["errors"].append(f"Reasoning failed: {str(e)}")
            return updated_state

    async def _tool_execution_node(self, state: AgentGraphState) -> AgentGraphState:
        """
        LangGraph tool execution node - where tools are called.

        Args:
            state: Current graph state

        Returns:
            Updated graph state
        """
        import time
        from app.backend_logging.backend_logger import get_logger
        from app.backend_logging.models import LogCategory

        backend_logger = get_logger()
        tool_execution_start_time = time.time()

        try:
            updated_state = state.copy()

            # Extract tool calls from the last message
            last_message = state["messages"][-1] if state["messages"] else None
            tools_executed = []

            # Log tool execution start
            backend_logger.debug(
                "Tool execution node started",
                LogCategory.AGENT_OPERATIONS,
                "LangGraphAgent",
                data={
                    "agent_id": state["agent_id"],
                    "session_id": state.get("session_id"),
                    "iteration": state.get("iteration_count", 0),
                    "has_tool_calls": bool(last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls),
                    "available_tools": list(self.tools.keys())
                }
            )

            if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Execute tool calls
                for tool_call in last_message.tool_calls:
                    tool_start_time = time.time()
                    tool_name = tool_call["name"]

                    if tool_name in self.tools:
                        tool = self.tools[tool_name]

                        # Log individual tool execution start
                        backend_logger.info(
                            f"Executing tool: {tool_name}",
                            LogCategory.AGENT_OPERATIONS,
                            "LangGraphAgent",
                            data={
                                "agent_id": state["agent_id"],
                                "session_id": state.get("session_id"),
                                "tool_name": tool_name,
                                "tool_args": tool_call["args"],
                                "iteration": state.get("iteration_count", 0)
                            }
                        )

                        try:
                            # Execute the tool
                            tool_result = await tool.ainvoke(tool_call["args"])
                            tool_execution_time_ms = (time.time() - tool_start_time) * 1000

                            # Record the tool call
                            updated_state["tool_calls"].append({
                                "tool": tool_name,
                                "args": tool_call["args"],
                                "result": tool_result
                            })

                            # Add tool result to messages
                            tool_message = AIMessage(
                                content=f"Tool {tool_name} result: {tool_result}"
                            )
                            updated_state["messages"].append(tool_message)

                            tools_executed.append(tool_name)

                            # Log successful tool execution
                            backend_logger.info(
                                f"Tool {tool_name} executed successfully",
                                LogCategory.AGENT_OPERATIONS,
                                "LangGraphAgent",
                                data={
                                    "agent_id": state["agent_id"],
                                    "session_id": state.get("session_id"),
                                    "tool_name": tool_name,
                                    "execution_time_ms": tool_execution_time_ms,
                                    "result_length": len(str(tool_result)),
                                    "iteration": state.get("iteration_count", 0)
                                }
                            )

                        except Exception as tool_error:
                            tool_execution_time_ms = (time.time() - tool_start_time) * 1000

                            # Log tool execution failure
                            backend_logger.error(
                                f"Tool {tool_name} execution failed: {str(tool_error)}",
                                LogCategory.AGENT_OPERATIONS,
                                "LangGraphAgent",
                                data={
                                    "agent_id": state["agent_id"],
                                    "session_id": state.get("session_id"),
                                    "tool_name": tool_name,
                                    "execution_time_ms": tool_execution_time_ms,
                                    "error_type": type(tool_error).__name__,
                                    "error_message": str(tool_error),
                                    "iteration": state.get("iteration_count", 0)
                                }
                            )

                            # Add error to state
                            updated_state["errors"].append(f"Tool {tool_name} failed: {str(tool_error)}")

            total_execution_time_ms = (time.time() - tool_execution_start_time) * 1000

            # Log tool execution completion
            backend_logger.info(
                f"Tool execution node completed - {len(tools_executed)} tools executed",
                LogCategory.AGENT_OPERATIONS,
                "LangGraphAgent",
                data={
                    "agent_id": state["agent_id"],
                    "session_id": state.get("session_id"),
                    "tools_executed": tools_executed,
                    "total_execution_time_ms": total_execution_time_ms,
                    "total_tool_calls": len(updated_state["tool_calls"]),
                    "iteration": state.get("iteration_count", 0)
                }
            )

            logger.debug(
                "Tool execution node completed",
                agent_id=self.agent_id,
                tools_executed=len(updated_state["tool_calls"])
            )

            return updated_state

        except Exception as e:
            total_execution_time_ms = (time.time() - tool_execution_start_time) * 1000

            # Log tool execution node failure
            backend_logger.error(
                f"Tool execution node failed: {str(e)}",
                LogCategory.AGENT_OPERATIONS,
                "LangGraphAgent",
                data={
                    "agent_id": state["agent_id"],
                    "session_id": state.get("session_id"),
                    "execution_time_ms": total_execution_time_ms,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "iteration": state.get("iteration_count", 0)
                }
            )

            logger.error(
                "Tool execution node failed",
                agent_id=self.agent_id,
                error=str(e)
            )
            updated_state = state.copy()
            updated_state["errors"].append(f"Tool execution failed: {str(e)}")
            return updated_state

    async def _decision_node(self, state: AgentGraphState) -> AgentGraphState:
        """
        LangGraph decision node - determines next action.

        Args:
            state: Current graph state

        Returns:
            Updated graph state
        """
        # This node just passes through the state
        # The actual decision logic is in _should_continue
        return state

    def _should_continue(self, state: AgentGraphState) -> str:
        """
        Determine if the agent should continue or end execution.

        Args:
            state: Current graph state

        Returns:
            "continue" or "end"
        """
        # Check iteration limit (reduce to prevent infinite loops)
        if state["iteration_count"] >= min(state["max_iterations"], 3):
            logger.info(
                "Agent reached max iterations",
                agent_id=self.agent_id,
                iterations=state["iteration_count"]
            )
            return "end"

        # Check if there are errors
        if state["errors"]:
            logger.warning(
                "Agent has errors, ending execution",
                agent_id=self.agent_id,
                errors=state["errors"]
            )
            return "end"

        # CRITICAL: Check if the last message has tool calls - if so, continue to execute them
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and isinstance(last_message, AIMessage):
            # First priority: Check for tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(
                    "Agent wants to use tools, continuing to tool execution",
                    agent_id=self.agent_id,
                    tool_calls=len(last_message.tool_calls)
                )
                return "continue"

            content = last_message.content.lower()

            # Check for explicit completion indicators only if no tool calls
            completion_phrases = [
                "task completed", "finished", "done", "complete",
                "final answer", "result is", "answer is", "calculation shows",
                "therefore", "in conclusion", "the solution is"
            ]

            if any(phrase in content for phrase in completion_phrases):
                logger.info(
                    "Agent indicated task completion",
                    agent_id=self.agent_id
                )
                return "end"

            # For math problems, check if we have a numerical answer
            if "calculate" in state["current_task"].lower():
                import re
                # Look for mathematical expressions or final numbers
                if re.search(r'\d+\s*[×*]\s*\d+\s*=\s*\d+', content) or \
                   re.search(r'=\s*\d+', content) or \
                   re.search(r'\b\d{2,}\b', content):  # Multi-digit numbers
                    logger.info(
                        "Agent provided mathematical answer",
                        agent_id=self.agent_id
                    )
                    return "end"

            # If the response is substantial (>100 chars) and addresses the task, consider it complete
            if len(content) > 100 and any(word in content for word in state["current_task"].lower().split()[:3]):
                logger.info(
                    "Agent provided substantial response addressing the task",
                    agent_id=self.agent_id
                )
                return "end"

        # For simple tasks, end after 2 iterations if we have a response
        if state["iteration_count"] >= 2 and len(state["messages"]) > 1:
            logger.info(
                "Agent completed simple task after 2 iterations",
                agent_id=self.agent_id
            )
            return "end"

        # Continue by default
        return "continue"

    async def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent's toolkit.
        
        Args:
            tool: Tool to add
        """
        self.tools[tool.name] = tool
        if tool.name not in self.config.tools:
            self.config.tools.append(tool.name)
        
        logger.info(
            "Tool added to agent",
            agent_id=self.agent_id,
            tool_name=tool.name,
            total_tools=len(self.tools),
        )
    
    async def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the agent's toolkit.
        
        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            if tool_name in self.config.tools:
                self.config.tools.remove(tool_name)
            
            logger.info(
                "Tool removed from agent",
                agent_id=self.agent_id,
                tool_name=tool_name,
                total_tools=len(self.tools),
            )
    
    async def get_state(self) -> AgentState:
        """
        Get current agent state.
        
        Returns:
            Current agent state
        """
        return self.state.copy(deep=True)
    
    async def set_state(self, state: AgentState) -> None:
        """
        Set agent state (for recovery/restoration).
        
        Args:
            state: State to restore
        """
        self.state = state
        logger.info(
            "Agent state restored",
            agent_id=self.agent_id,
            status=state.status,
            execution_step=state.execution_step,
        )
    
    async def reset(self) -> None:
        """Reset agent to initial state."""
        self.state = AgentState(
            agent_id=self.agent_id,
            session_id=str(uuid.uuid4())
        )
        
        logger.info("Agent reset", agent_id=self.agent_id)
    
    async def cancel(self) -> None:
        """Cancel current execution."""
        if self.state.status == AgentStatus.RUNNING:
            self.state.status = AgentStatus.CANCELLED
            self.state.updated_at = datetime.utcnow()
            
            logger.info("Agent execution cancelled", agent_id=self.agent_id)
    
    def _update_state(self, **kwargs) -> None:
        """
        Update agent state with provided fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        self.state.updated_at = datetime.utcnow()
    
    async def _execute_with_timeout(
        self,
        coro,
        timeout_seconds: Optional[int] = None
    ) -> Any:
        """
        Execute a coroutine with timeout.
        
        Args:
            coro: Coroutine to execute
            timeout_seconds: Timeout in seconds
            
        Returns:
            Coroutine result
            
        Raises:
            AgentTimeoutError: If execution times out
        """
        timeout = timeout_seconds or self.config.timeout_seconds
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            await self.cancel()
            raise AgentTimeoutError(self.agent_id, timeout)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _estimate_tokens(self, messages: List) -> int:
        """Estimate token count from messages."""
        try:
            total_chars = sum(len(str(msg.content)) for msg in messages if hasattr(msg, 'content'))
            # Rough estimation: 1 token ≈ 4 characters
            return total_chars // 4
        except Exception:
            return 0

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id[:8]}, "
            f"name={self.name}, "
            f"status={self.state.status}"
            f")"
        )

    async def _handle_manual_tool_calling(self, response: AIMessage, state: AgentGraphState) -> AIMessage:
        """
        Handle manual tool calling for LLMs that don't support bind_tools.
        Parse the response for explicit tool usage requests and create tool calls.
        """
        try:
            import re
            import time
            content = response.content
            tool_calls = []

            # Look for natural tool usage patterns - more flexible detection
            tool_usage_patterns = [
                # Explicit tool mentions
                r'I will use the (\w+) tool',
                r'I\'ll use the (\w+) tool',
                r'Using the (\w+) tool',
                r'Let me use the (\w+) tool',
                r'I need to use the (\w+) tool',
                # Natural language patterns
                r'I need to search for',
                r'Let me search for',
                r'I should search for',
                r'I\'ll search for',
                r'Let me find information about',
                r'I need to find information about',
                r'I should look up',
                r'Let me look up',
                r'I need to research',
                r'Let me research',
                r'I should research',
                r'I\'ll research',
                # Calculation patterns
                r'I need to calculate',
                r'Let me calculate',
                r'I should calculate',
                r'I\'ll calculate',
                r'I need to compute',
                r'Let me compute'
            ]

            # Check for explicit tool name patterns first
            explicit_patterns = [
                r'I will use the (\w+) tool',
                r'I\'ll use the (\w+) tool',
                r'Using the (\w+) tool',
                r'Let me use the (\w+) tool',
                r'I need to use the (\w+) tool'
            ]

            for pattern in explicit_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    tool_name = match.lower()
                    if tool_name in self.tools:
                        tool_call = await self._create_tool_call(tool_name, state)
                        if tool_call:
                            tool_calls.append(tool_call)
                            logger.info(f"🔧 Detected explicit tool usage: {tool_name}")

            # If no explicit tool usage found, check for natural language patterns
            if not tool_calls:
                # Check for research/search patterns
                search_patterns = [
                    r'I need to search for',
                    r'Let me search for',
                    r'I should search for',
                    r'I\'ll search for',
                    r'Let me find information about',
                    r'I need to find information about',
                    r'I should look up',
                    r'Let me look up',
                    r'I need to research',
                    r'Let me research',
                    r'I should research',
                    r'I\'ll research'
                ]

                for pattern in search_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        if 'web_research' in self.tools:
                            tool_call = await self._create_tool_call('web_research', state)
                            if tool_call:
                                tool_calls.append(tool_call)
                                logger.info(f"🔧 Detected natural research intent, using web_research tool")
                                break

                # Check for calculation patterns
                if not tool_calls:
                    calc_patterns = [
                        r'I need to calculate',
                        r'Let me calculate',
                        r'I should calculate',
                        r'I\'ll calculate',
                        r'I need to compute',
                        r'Let me compute'
                    ]

                    for pattern in calc_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            if 'calculator' in self.tools:
                                tool_call = await self._create_tool_call('calculator', state)
                                if tool_call:
                                    tool_calls.append(tool_call)
                                    logger.info(f"🔧 Detected natural calculation intent, using calculator tool")
                                    break

            # If tool calls were detected, create a new response with tool calls
            if tool_calls:
                response.tool_calls = tool_calls
                logger.info(f"✅ Created {len(tool_calls)} tool calls")

            return response

        except Exception as e:
            logger.error(f"Error in manual tool calling: {e}")
            return response

    async def _create_tool_call(self, tool_name: str, state: AgentGraphState) -> Optional[Dict]:
        """
        Create a tool call with appropriate arguments based on the tool type and context.
        """
        try:
            import re
            import time
            current_task = state.get("current_task", "")

            if tool_name == "calculator":
                # Extract mathematical expression from the task
                math_patterns = [
                    r'(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)*)',
                    r'calculate[:\s]+([^.!?]+)',
                    r'compute[:\s]+([^.!?]+)'
                ]

                for pattern in math_patterns:
                    matches = re.findall(pattern, current_task, re.IGNORECASE)
                    if matches:
                        expression = matches[0].strip()
                        # Clean up the expression
                        expression = re.sub(r'[^\d+\-*/().\s]', '', expression).strip()
                        if expression and any(op in expression for op in ['+', '-', '*', '/']):
                            return {
                                "name": "calculator",
                                "args": {"expression": expression, "precision": 2},
                                "id": f"call_calculator_{int(time.time())}"
                            }

            elif tool_name == "web_search":
                # Extract search query from task
                return {
                    "name": "web_search",
                    "args": {"query": current_task, "num_results": 5},
                    "id": f"call_web_search_{int(time.time())}"
                }

            elif tool_name == "business_intelligence":
                # Create business analysis call
                return {
                    "name": "business_intelligence",
                    "args": {"analysis_type": "general", "data": current_task},
                    "id": f"call_business_intelligence_{int(time.time())}"
                }

            return None

        except Exception as e:
            logger.error(f"Error creating tool call for {tool_name}: {e}")
            return None


class AgentInterface(ABC):
    """
    Interface for agent management and interaction.

    This interface defines the contract for agent managers and orchestrators.
    """

    @abstractmethod
    async def create_agent(
        self,
        agent_type: str,
        config: AgentConfig,
        **kwargs
    ) -> LangGraphAgent:
        """Create a new agent instance."""
        pass

    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[LangGraphAgent]:
        """Get an agent by ID."""
        pass

    @abstractmethod
    async def list_agents(self) -> List[LangGraphAgent]:
        """List all active agents."""
        pass

    @abstractmethod
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        pass
