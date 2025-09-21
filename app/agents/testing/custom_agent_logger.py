"""
Custom Agent Logger - Comprehensive Agent Interaction Logging System.

This logger captures every aspect of agent behavior including:
- Agent metadata and configuration
- System prompts and user queries
- Thinking and reasoning processes
- Tool usage and results
- Memory operations
- RAG operations
- Performance metrics
- Final answers and outcomes
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class LogLevel(str, Enum):
    """Agent log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AgentAction(str, Enum):
    """Types of agent actions."""
    INITIALIZATION = "initialization"
    QUERY_RECEIVED = "query_received"
    THINKING = "thinking"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MEMORY_OPERATION = "memory_operation"
    RAG_OPERATION = "rag_operation"
    DECISION_MAKING = "decision_making"
    RESPONSE_GENERATION = "response_generation"
    FINAL_ANSWER = "final_answer"
    ERROR_HANDLING = "error_handling"


@dataclass
class AgentMetadata:
    """Agent metadata information."""
    agent_id: str
    agent_type: str
    agent_name: str
    capabilities: List[str]
    tools_available: List[str]
    memory_type: Optional[str] = None
    rag_enabled: bool = False
    autonomy_level: Optional[str] = None
    learning_mode: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolUsage:
    """Tool usage information."""
    tool_name: str
    parameters: Dict[str, Any]
    execution_time: float
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryOperation:
    """Memory operation information."""
    operation_type: str  # store, retrieve, update, delete
    memory_type: str     # short_term, long_term, episodic, semantic, etc.
    content: Optional[str] = None
    query: Optional[str] = None
    results_count: Optional[int] = None
    execution_time: float = 0.0
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RAGOperation:
    """RAG operation information."""
    operation_type: str  # query, add_document, update_knowledge
    collection_name: Optional[str] = None
    query: Optional[str] = None
    documents_count: Optional[int] = None
    results_count: Optional[int] = None
    similarity_threshold: Optional[float] = None
    execution_time: float = 0.0
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ThinkingProcess:
    """Agent thinking and reasoning process."""
    step_number: int
    thought: str
    reasoning: Optional[str] = None
    decision: Optional[str] = None
    confidence: Optional[float] = None
    alternatives_considered: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentLogEntry:
    """Complete agent log entry."""
    session_id: str
    agent_metadata: AgentMetadata
    action: AgentAction
    level: LogLevel
    message: str
    
    # Context information
    user_query: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # Process information
    thinking_process: Optional[ThinkingProcess] = None
    tool_usage: Optional[ToolUsage] = None
    memory_operation: Optional[MemoryOperation] = None
    rag_operation: Optional[RAGOperation] = None
    
    # Performance metrics
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    
    # Results
    intermediate_result: Optional[Any] = None
    final_answer: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)


class CustomAgentLogger:
    """
    Comprehensive agent logger that captures all agent interactions.
    
    Features:
    - Detailed logging of all agent activities
    - Performance metrics tracking
    - Tool usage analytics
    - Memory and RAG operation logging
    - Thinking process capture
    - Session-based organization
    - Multiple output formats (JSON, structured logs)
    """
    
    def __init__(self, log_directory: str = "logs/agents"):
        """Initialize the custom agent logger."""
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self.current_sessions: Dict[str, List[AgentLogEntry]] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_sessions": 0,
            "total_log_entries": 0,
            "average_response_time": 0.0,
            "tool_usage_count": {},
            "memory_operations_count": 0,
            "rag_operations_count": 0
        }
        
        logger.info("Custom Agent Logger initialized", log_directory=str(self.log_directory))
    
    def start_session(self, session_id: str, agent_metadata: AgentMetadata) -> None:
        """Start a new logging session for an agent."""
        self.current_sessions[session_id] = []
        self.performance_stats["total_sessions"] += 1
        
        # Log session start
        entry = AgentLogEntry(
            session_id=session_id,
            agent_metadata=agent_metadata,
            action=AgentAction.INITIALIZATION,
            level=LogLevel.INFO,
            message=f"Started session for agent {agent_metadata.agent_name}"
        )
        
        self._add_log_entry(entry)
        logger.info("Started agent session", session_id=session_id, agent_name=agent_metadata.agent_name)
    
    def log_query_received(self, session_id: str, user_query: str, system_prompt: Optional[str] = None) -> None:
        """Log when agent receives a user query."""
        agent_metadata = self._get_agent_metadata(session_id)
        
        entry = AgentLogEntry(
            session_id=session_id,
            agent_metadata=agent_metadata,
            action=AgentAction.QUERY_RECEIVED,
            level=LogLevel.INFO,
            message="Received user query",
            user_query=user_query,
            system_prompt=system_prompt
        )
        
        self._add_log_entry(entry)
    
    def log_thinking_process(self, session_id: str, thinking: ThinkingProcess) -> None:
        """Log agent thinking and reasoning process."""
        agent_metadata = self._get_agent_metadata(session_id)
        
        entry = AgentLogEntry(
            session_id=session_id,
            agent_metadata=agent_metadata,
            action=AgentAction.THINKING,
            level=LogLevel.DEBUG,
            message=f"Thinking step {thinking.step_number}: {thinking.thought}",
            thinking_process=thinking
        )
        
        self._add_log_entry(entry)
    
    def log_tool_usage(self, session_id: str, tool_usage: ToolUsage) -> None:
        """Log tool usage by agent."""
        agent_metadata = self._get_agent_metadata(session_id)
        
        # Update performance stats
        tool_name = tool_usage.tool_name
        if tool_name not in self.performance_stats["tool_usage_count"]:
            self.performance_stats["tool_usage_count"][tool_name] = 0
        self.performance_stats["tool_usage_count"][tool_name] += 1
        
        entry = AgentLogEntry(
            session_id=session_id,
            agent_metadata=agent_metadata,
            action=AgentAction.TOOL_CALL,
            level=LogLevel.INFO,
            message=f"Used tool {tool_usage.tool_name}",
            tool_usage=tool_usage,
            execution_time=tool_usage.execution_time
        )
        
        self._add_log_entry(entry)
    
    def log_memory_operation(self, session_id: str, memory_op: MemoryOperation) -> None:
        """Log memory operations."""
        agent_metadata = self._get_agent_metadata(session_id)
        self.performance_stats["memory_operations_count"] += 1
        
        entry = AgentLogEntry(
            session_id=session_id,
            agent_metadata=agent_metadata,
            action=AgentAction.MEMORY_OPERATION,
            level=LogLevel.DEBUG,
            message=f"Memory {memory_op.operation_type}: {memory_op.memory_type}",
            memory_operation=memory_op,
            execution_time=memory_op.execution_time
        )
        
        self._add_log_entry(entry)
    
    def log_rag_operation(self, session_id: str, rag_op: RAGOperation) -> None:
        """Log RAG operations."""
        agent_metadata = self._get_agent_metadata(session_id)
        self.performance_stats["rag_operations_count"] += 1
        
        entry = AgentLogEntry(
            session_id=session_id,
            agent_metadata=agent_metadata,
            action=AgentAction.RAG_OPERATION,
            level=LogLevel.DEBUG,
            message=f"RAG {rag_op.operation_type}: {rag_op.collection_name}",
            rag_operation=rag_op,
            execution_time=rag_op.execution_time
        )
        
        self._add_log_entry(entry)
    
    def log_final_answer(self, session_id: str, final_answer: str, execution_time: float) -> None:
        """Log final answer from agent."""
        agent_metadata = self._get_agent_metadata(session_id)
        
        entry = AgentLogEntry(
            session_id=session_id,
            agent_metadata=agent_metadata,
            action=AgentAction.FINAL_ANSWER,
            level=LogLevel.INFO,
            message="Generated final answer",
            final_answer=final_answer,
            execution_time=execution_time
        )
        
        self._add_log_entry(entry)
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End logging session and return summary."""
        if session_id not in self.current_sessions:
            logger.warning("Session not found", session_id=session_id)
            return {}
        
        session_logs = self.current_sessions[session_id]
        
        # Generate session summary
        summary = self._generate_session_summary(session_id, session_logs)
        
        # Save session logs to file
        self._save_session_logs(session_id, session_logs, summary)
        
        # Clean up
        del self.current_sessions[session_id]
        
        logger.info("Ended agent session", session_id=session_id, total_entries=len(session_logs))
        return summary
    
    def _add_log_entry(self, entry: AgentLogEntry) -> None:
        """Add log entry to current session."""
        if entry.session_id not in self.current_sessions:
            logger.warning("Session not found for log entry", session_id=entry.session_id)
            return
        
        self.current_sessions[entry.session_id].append(entry)
        self.performance_stats["total_log_entries"] += 1
    
    def _get_agent_metadata(self, session_id: str) -> AgentMetadata:
        """Get agent metadata for session."""
        if session_id not in self.current_sessions or not self.current_sessions[session_id]:
            # Return default metadata if session not found
            return AgentMetadata(
                agent_id="unknown",
                agent_type="unknown",
                agent_name="Unknown Agent",
                capabilities=[],
                tools_available=[]
            )
        
        return self.current_sessions[session_id][0].agent_metadata
    
    def _generate_session_summary(self, session_id: str, logs: List[AgentLogEntry]) -> Dict[str, Any]:
        """Generate comprehensive session summary."""
        if not logs:
            return {}
        
        # Basic stats
        total_entries = len(logs)
        start_time = logs[0].timestamp
        end_time = logs[-1].timestamp
        total_duration = (end_time - start_time).total_seconds()
        
        # Action counts
        action_counts = {}
        for log in logs:
            action = log.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Tool usage
        tools_used = []
        for log in logs:
            if log.tool_usage:
                tools_used.append({
                    "tool": log.tool_usage.tool_name,
                    "success": log.tool_usage.success,
                    "execution_time": log.tool_usage.execution_time
                })
        
        # Memory operations
        memory_ops = [log.memory_operation for log in logs if log.memory_operation]
        
        # RAG operations
        rag_ops = [log.rag_operation for log in logs if log.rag_operation]
        
        # Performance metrics
        execution_times = [log.execution_time for log in logs if log.execution_time]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "session_id": session_id,
            "agent_metadata": asdict(logs[0].agent_metadata),
            "summary": {
                "total_entries": total_entries,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "action_counts": action_counts,
                "tools_used": tools_used,
                "memory_operations": len(memory_ops),
                "rag_operations": len(rag_ops),
                "average_execution_time": avg_execution_time
            },
            "user_query": next((log.user_query for log in logs if log.user_query), None),
            "final_answer": next((log.final_answer for log in logs if log.final_answer), None),
            "thinking_steps": [log.thinking_process for log in logs if log.thinking_process],
            "performance_metrics": {
                "total_tool_calls": len(tools_used),
                "successful_tool_calls": sum(1 for tool in tools_used if tool["success"]),
                "total_execution_time": sum(execution_times),
                "memory_operations": len(memory_ops),
                "rag_operations": len(rag_ops)
            }
        }
    
    def _save_session_logs(self, session_id: str, logs: List[AgentLogEntry], summary: Dict[str, Any]) -> None:
        """Save session logs to file."""
        try:
            # Create session directory
            session_dir = self.log_directory / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save detailed logs
            logs_file = session_dir / "detailed_logs.json"
            with open(logs_file, 'w') as f:
                json.dump([asdict(log) for log in logs], f, indent=2, default=str)
            
            # Save summary
            summary_file = session_dir / "session_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save human-readable report
            report_file = session_dir / "session_report.md"
            self._generate_markdown_report(report_file, summary, logs)
            
        except Exception as e:
            logger.error("Failed to save session logs", session_id=session_id, error=str(e))
    
    def _generate_markdown_report(self, file_path: Path, summary: Dict[str, Any], logs: List[AgentLogEntry]) -> None:
        """Generate human-readable markdown report."""
        with open(file_path, 'w') as f:
            f.write(f"# Agent Session Report\n\n")
            f.write(f"**Session ID:** {summary['session_id']}\n")
            f.write(f"**Agent:** {summary['agent_metadata']['agent_name']}\n")
            f.write(f"**Agent Type:** {summary['agent_metadata']['agent_type']}\n")
            f.write(f"**Duration:** {summary['summary']['total_duration_seconds']:.2f} seconds\n\n")
            
            if summary.get('user_query'):
                f.write(f"## User Query\n{summary['user_query']}\n\n")
            
            if summary.get('final_answer'):
                f.write(f"## Final Answer\n{summary['final_answer']}\n\n")
            
            f.write(f"## Performance Summary\n")
            f.write(f"- Total Log Entries: {summary['summary']['total_entries']}\n")
            f.write(f"- Tools Used: {summary['performance_metrics']['total_tool_calls']}\n")
            f.write(f"- Memory Operations: {summary['performance_metrics']['memory_operations']}\n")
            f.write(f"- RAG Operations: {summary['performance_metrics']['rag_operations']}\n")
            f.write(f"- Average Execution Time: {summary['summary']['average_execution_time']:.3f}s\n\n")
            
            # Detailed timeline
            f.write(f"## Detailed Timeline\n")
            for i, log in enumerate(logs, 1):
                f.write(f"{i}. **{log.action.value.title()}** ({log.timestamp.strftime('%H:%M:%S')})\n")
                f.write(f"   - {log.message}\n")
                if log.thinking_process:
                    f.write(f"   - Thought: {log.thinking_process.thought}\n")
                if log.tool_usage:
                    f.write(f"   - Tool: {log.tool_usage.tool_name} ({'✅' if log.tool_usage.success else '❌'})\n")
                f.write("\n")


# Global logger instance
custom_agent_logger = CustomAgentLogger()
