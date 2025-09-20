"""
Conversational Agent Creation API endpoints.

This module provides conversational agent creation where users can talk to an LLM
to design and create custom agents through natural conversation, and then execute
tasks autonomously while showing reasoning processes.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.core.dependencies import get_orchestrator
# from app.orchestration.enhanced_orchestrator import enhanced_orchestrator

# Import backend logging system
from app.backend_logging.backend_logger import get_logger
from app.backend_logging.models import LogLevel, LogCategory, PerformanceMetrics, AgentMetrics
from app.backend_logging.context import CorrelationContext

logger = structlog.get_logger(__name__)
backend_logger = get_logger()

router = APIRouter(prefix="/conversational", tags=["Conversational Agent Creation"])


class ConversationMessage(BaseModel):
    """Message in the agent creation conversation."""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentCreationRequest(BaseModel):
    """Request to start conversational agent creation."""
    user_message: str = Field(..., description="Initial user message describing what they want")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")


class AgentCreationResponse(BaseModel):
    """Response from conversational agent creation."""
    session_id: str = Field(..., description="Session ID for this conversation")
    assistant_message: str = Field(..., description="Assistant's response")
    suggested_agent_config: Optional[Dict[str, Any]] = Field(default=None, description="Suggested agent configuration")
    conversation_stage: str = Field(..., description="Current stage of conversation")
    next_questions: Optional[List[str]] = Field(default=None, description="Suggested follow-up questions")


class TaskExecutionRequest(BaseModel):
    """Request to execute a task autonomously."""
    agent_id: str = Field(..., description="ID of the agent to use")
    task_description: str = Field(..., description="Description of the task to execute")
    show_reasoning: bool = Field(default=True, description="Whether to show reasoning process")
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")


class ReasoningStep(BaseModel):
    """A step in the agent's reasoning process."""
    step_number: int = Field(..., description="Step number in the reasoning process")
    step_type: str = Field(..., description="Type of reasoning step")
    description: str = Field(..., description="Description of what the agent is doing")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskExecutionResponse(BaseModel):
    """Response from autonomous task execution."""
    execution_id: str = Field(..., description="Unique execution ID")
    status: str = Field(..., description="Current execution status")
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list, description="Reasoning steps taken")
    current_step: Optional[str] = Field(default=None, description="Current step being executed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Final result if completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")


@router.post("/create-agent", response_model=AgentCreationResponse)
async def start_conversational_agent_creation(
    request: AgentCreationRequest,
    orchestrator = Depends(get_orchestrator)
):
    """
    Start a conversational agent creation session.
    
    The LLM will engage in a conversation with the user to understand their needs
    and help design a custom agent through natural dialogue.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Set up correlation context
        CorrelationContext.update_context(
            session_id=session_id,
            user_id=request.user_id,
            component="ConversationalAgentCreation",
            operation="start_conversation"
        )
        
        backend_logger.info(
            "Starting conversational agent creation",
            category=LogCategory.AGENT_OPERATIONS,
            component="ConversationalAgentCreation",
            data={
                "session_id": session_id,
                "user_message": request.user_message[:100] + "..." if len(request.user_message) > 100 else request.user_message
            }
        )
        
        # Create a specialized agent creation assistant prompt
        system_prompt = """You are an expert AI agent designer assistant. Your role is to help users create custom AI agents through natural conversation.

Your goals:
1. Understand what the user wants their agent to do
2. Ask clarifying questions to gather requirements
3. Suggest appropriate agent configurations
4. Guide them through the agent creation process

Key areas to explore:
- Agent purpose and main tasks
- Required capabilities and tools
- Interaction style and personality
- Performance requirements
- Integration needs

Be conversational, helpful, and thorough. Ask one question at a time to avoid overwhelming the user.
"""
        
        # Use the orchestrator to process the conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_message}
        ]
        
        # Get response from LLM
        start_time = time.time()
        response = await orchestrator.process_message(
            messages=messages,
            agent_type="conversational_designer",
            session_id=session_id
        )
        execution_time = (time.time() - start_time) * 1000
        
        # Analyze the conversation to determine stage and suggestions
        conversation_stage = "requirements_gathering"
        next_questions = [
            "What specific tasks should your agent be able to perform?",
            "What tools or capabilities does your agent need?",
            "How should your agent interact with users?"
        ]
        
        # Log performance metrics
        backend_logger.info(
            "Conversational agent creation response generated",
            category=LogCategory.AGENT_OPERATIONS,
            component="ConversationalAgentCreation",
            performance=PerformanceMetrics(
                duration_ms=execution_time,
                memory_usage_mb=0,  # Would need actual measurement
                cpu_usage_percent=0,  # Would need actual measurement
                api_calls_count=1
            ),
            data={
                "session_id": session_id,
                "response_length": len(response),
                "conversation_stage": conversation_stage
            }
        )
        
        return AgentCreationResponse(
            session_id=session_id,
            assistant_message=response,
            conversation_stage=conversation_stage,
            next_questions=next_questions
        )
        
    except Exception as e:
        backend_logger.error(
            f"Error in conversational agent creation: {str(e)}",
            category=LogCategory.ERROR_TRACKING,
            component="ConversationalAgentCreation",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to start agent creation conversation: {str(e)}")


@router.post("/continue-conversation", response_model=AgentCreationResponse)
async def continue_agent_creation_conversation(
    session_id: str,
    user_message: str,
    conversation_history: List[ConversationMessage],
    orchestrator = Depends(get_orchestrator)
):
    """
    Continue the conversational agent creation process.
    
    This endpoint maintains the conversation context and helps refine
    the agent configuration based on user feedback.
    """
    try:
        # Set up correlation context
        CorrelationContext.update_context(
            session_id=session_id,
            component="ConversationalAgentCreation",
            operation="continue_conversation"
        )
        
        backend_logger.info(
            "Continuing conversational agent creation",
            category=LogCategory.AGENT_OPERATIONS,
            component="ConversationalAgentCreation",
            data={
                "session_id": session_id,
                "message_count": len(conversation_history),
                "user_message": user_message[:100] + "..." if len(user_message) > 100 else user_message
            }
        )
        
        # Convert conversation history to messages format
        messages = []
        for msg in conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add the new user message
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        # Get response from LLM
        start_time = time.time()
        response = await orchestrator.process_message(
            messages=messages,
            agent_type="conversational_designer",
            session_id=session_id
        )
        execution_time = (time.time() - start_time) * 1000
        
        # Analyze conversation to determine if we can suggest an agent config
        suggested_config = None
        conversation_stage = "requirements_gathering"
        
        # Simple heuristic: if conversation has enough back-and-forth, suggest config
        if len(conversation_history) >= 4:
            conversation_stage = "configuration_ready"
            suggested_config = {
                "name": "Custom Agent",
                "description": "Agent created through conversation",
                "capabilities": ["general_assistance"],
                "tools": ["web_search", "text_processing"],
                "model": "llama3.2:latest",
                "temperature": 0.7
            }
        
        backend_logger.info(
            "Conversation continued successfully",
            category=LogCategory.AGENT_OPERATIONS,
            component="ConversationalAgentCreation",
            performance=PerformanceMetrics(
                duration_ms=execution_time,
                api_calls_count=1
            ),
            data={
                "session_id": session_id,
                "conversation_stage": conversation_stage,
                "config_suggested": suggested_config is not None
            }
        )
        
        return AgentCreationResponse(
            session_id=session_id,
            assistant_message=response,
            suggested_agent_config=suggested_config,
            conversation_stage=conversation_stage
        )
        
    except Exception as e:
        backend_logger.error(
            f"Error continuing conversation: {str(e)}",
            category=LogCategory.ERROR_TRACKING,
            component="ConversationalAgentCreation",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to continue conversation: {str(e)}")


@router.post("/execute-task", response_model=TaskExecutionResponse)
async def execute_task_autonomously(
    request: TaskExecutionRequest,
    orchestrator = Depends(get_orchestrator)
):
    """
    Execute a task autonomously with an agent, showing reasoning process.
    
    This endpoint demonstrates true agentic AI by having the agent work
    independently while providing visibility into its reasoning process.
    """
    try:
        execution_id = str(uuid.uuid4())
        
        # Set up correlation context
        CorrelationContext.update_context(
            session_id=request.session_id,
            agent_id=request.agent_id,
            component="AutonomousTaskExecution",
            operation="execute_task"
        )
        
        backend_logger.info(
            "Starting autonomous task execution",
            category=LogCategory.AGENT_OPERATIONS,
            component="AutonomousTaskExecution",
            data={
                "execution_id": execution_id,
                "agent_id": request.agent_id,
                "task_description": request.task_description,
                "show_reasoning": request.show_reasoning
            }
        )
        
        # Initialize reasoning steps
        reasoning_steps = []
        
        if request.show_reasoning:
            reasoning_steps.append(ReasoningStep(
                step_number=1,
                step_type="task_analysis",
                description=f"Analyzing task: {request.task_description}",
                details={"task": request.task_description}
            ))
            
            reasoning_steps.append(ReasoningStep(
                step_number=2,
                step_type="planning",
                description="Creating execution plan and identifying required tools",
                details={"planning_stage": "tool_identification"}
            ))
        
        # Execute the task using the enhanced orchestrator
        start_time = time.time()
        
        try:
            # Create a task execution context
            task_context = {
                "task_description": request.task_description,
                "agent_id": request.agent_id,
                "execution_id": execution_id,
                "show_reasoning": request.show_reasoning
            }
            
            if request.show_reasoning:
                reasoning_steps.append(ReasoningStep(
                    step_number=3,
                    step_type="execution",
                    description="Executing task with selected tools and approach",
                    details={"execution_context": task_context}
                ))
            
            # Use orchestrator for task execution
            result = await orchestrator.execute_autonomous_task(
                task_description=request.task_description,
                agent_id=request.agent_id,
                context=task_context
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            if request.show_reasoning:
                reasoning_steps.append(ReasoningStep(
                    step_number=4,
                    step_type="completion",
                    description="Task completed successfully",
                    details={"result_summary": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)}
                ))
            
            backend_logger.info(
                "Autonomous task execution completed",
                category=LogCategory.AGENT_OPERATIONS,
                component="AutonomousTaskExecution",
                performance=PerformanceMetrics(
                    duration_ms=execution_time,
                    api_calls_count=1
                ),
                agent_metrics=AgentMetrics(
                    agent_id=request.agent_id,
                    task_type="autonomous_execution",
                    success_rate=1.0,
                    average_response_time=execution_time
                ),
                data={
                    "execution_id": execution_id,
                    "task_completed": True,
                    "reasoning_steps_count": len(reasoning_steps)
                }
            )
            
            return TaskExecutionResponse(
                execution_id=execution_id,
                status="completed",
                reasoning_steps=reasoning_steps,
                result=result
            )
            
        except Exception as task_error:
            if request.show_reasoning:
                reasoning_steps.append(ReasoningStep(
                    step_number=len(reasoning_steps) + 1,
                    step_type="error",
                    description=f"Task execution failed: {str(task_error)}",
                    details={"error": str(task_error)}
                ))
            
            backend_logger.error(
                f"Task execution failed: {str(task_error)}",
                category=LogCategory.ERROR_TRACKING,
                component="AutonomousTaskExecution",
                error=task_error,
                data={"execution_id": execution_id}
            )
            
            return TaskExecutionResponse(
                execution_id=execution_id,
                status="failed",
                reasoning_steps=reasoning_steps,
                error=str(task_error)
            )
        
    except Exception as e:
        backend_logger.error(
            f"Error in autonomous task execution: {str(e)}",
            category=LogCategory.ERROR_TRACKING,
            component="AutonomousTaskExecution",
            error=e
        )
        raise HTTPException(status_code=500, detail=f"Failed to execute task: {str(e)}")


@router.websocket("/reasoning-stream/{execution_id}")
async def stream_reasoning_process(websocket: WebSocket, execution_id: str):
    """
    WebSocket endpoint for real-time reasoning process streaming.
    
    This allows the frontend to show live updates of the agent's
    reasoning process as it works through a task.
    """
    await websocket.accept()
    
    try:
        backend_logger.info(
            "Reasoning stream connection established",
            category=LogCategory.API_LAYER,
            component="ReasoningStream",
            data={"execution_id": execution_id}
        )
        
        # Simulate real-time reasoning updates
        reasoning_steps = [
            "Analyzing task requirements...",
            "Identifying necessary tools and resources...",
            "Creating execution strategy...",
            "Beginning task execution...",
            "Processing intermediate results...",
            "Finalizing output...",
            "Task completed successfully!"
        ]
        
        for i, step in enumerate(reasoning_steps):
            reasoning_update = {
                "step_number": i + 1,
                "step_type": "reasoning",
                "description": step,
                "timestamp": datetime.now().isoformat(),
                "execution_id": execution_id
            }
            
            await websocket.send_json(reasoning_update)
            await asyncio.sleep(2)  # Simulate processing time
        
        # Send completion signal
        await websocket.send_json({
            "step_type": "completion",
            "description": "Reasoning process completed",
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except WebSocketDisconnect:
        backend_logger.info(
            "Reasoning stream connection closed",
            category=LogCategory.API_LAYER,
            component="ReasoningStream",
            data={"execution_id": execution_id}
        )
    except Exception as e:
        backend_logger.error(
            f"Error in reasoning stream: {str(e)}",
            category=LogCategory.ERROR_TRACKING,
            component="ReasoningStream",
            error=e,
            data={"execution_id": execution_id}
        )
        await websocket.close(code=1011, reason="Internal server error")
