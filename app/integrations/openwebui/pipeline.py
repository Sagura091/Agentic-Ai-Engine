"""
OpenWebUI Pipelines integration for the Agentic AI Microservice.

This module implements the OpenWebUI Pipelines framework to expose our
LangChain/LangGraph agents as OpenAI-compatible models in OpenWebUI.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator, Union

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory
from app.config.settings import get_settings
from app.orchestration.subgraphs import HierarchicalWorkflowOrchestrator

_backend_logger = get_backend_logger()


class OpenAIMessage(BaseModel):
    """OpenAI-compatible message format."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Message name")


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(..., description="Model identifier")
    messages: List[OpenAIMessage] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=2048, description="Maximum tokens")
    stream: Optional[bool] = Field(default=False, description="Stream response")
    user: Optional[str] = Field(default=None, description="User identifier")


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Token usage")


class OpenAIChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")


class OpenWebUIPipeline:
    """
    OpenWebUI Pipeline implementation for Agentic AI integration.
    
    This class implements the OpenWebUI Pipelines framework to expose
    our LangGraph agents as OpenAI-compatible models.
    """
    
    def __init__(self):
        """Initialize the OpenWebUI pipeline."""
        self.settings = get_settings()
        self.orchestrator: Optional[HierarchicalWorkflowOrchestrator] = None
        self.available_models: Dict[str, Dict[str, Any]] = {}

        _backend_logger.info(
            "OpenWebUI pipeline initialized",
            LogCategory.API_OPERATIONS,
            "app.integrations.openwebui.pipeline"
        )

    async def initialize(self) -> None:
        """Initialize the pipeline and orchestrator."""
        try:
            # Initialize the orchestrator
            from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
            self.orchestrator = get_enhanced_system_orchestrator()

            if not self.orchestrator.status.is_initialized:
                await self.orchestrator.initialize()

            # Register default agent models
            await self._register_default_models()

            _backend_logger.info(
                "OpenWebUI pipeline ready",
                LogCategory.API_OPERATIONS,
                "app.integrations.openwebui.pipeline",
                data={"models_count": len(self.available_models)}
            )

        except Exception as e:
            _backend_logger.error(
                "Failed to initialize OpenWebUI pipeline",
                LogCategory.API_OPERATIONS,
                "app.integrations.openwebui.pipeline",
                data={"error": str(e)}
            )
            raise
    
    async def _register_default_models(self) -> None:
        """Register default agent models."""
        default_models = [
            {
                "id": "agentic-general",
                "name": "Agentic General Assistant",
                "description": "General-purpose AI agent with reasoning and tool capabilities",
                "agent_type": "general",
                "capabilities": ["reasoning", "tool_use", "memory"]
            },
            {
                "id": "agentic-research",
                "name": "Agentic Research Agent",
                "description": "Specialized research agent with web search and analysis capabilities",
                "agent_type": "research",
                "capabilities": ["reasoning", "tool_use", "memory", "web_search"]
            },
            {
                "id": "agentic-workflow",
                "name": "Agentic Multi-Agent Workflow",
                "description": "Multi-agent workflow orchestrator for complex tasks",
                "agent_type": "workflow",
                "capabilities": ["reasoning", "tool_use", "memory", "collaboration"]
            }
        ]
        
        for model in default_models:
            self.available_models[model["id"]] = model

        _backend_logger.info(
            "Default agent models registered",
            LogCategory.API_OPERATIONS,
            "app.integrations.openwebui.pipeline",
            data={"count": len(default_models)}
        )
    
    async def get_models(self) -> Dict[str, Any]:
        """
        Get available models in OpenAI format.
        
        Returns:
            OpenAI-compatible models response
        """
        models = []
        
        for model_id, model_info in self.available_models.items():
            models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agentic-ai",
                "permission": [],
                "root": model_id,
                "parent": None
            })
        
        return {
            "object": "list",
            "data": models
        }
    
    async def chat_completion(
        self,
        request: OpenAIChatCompletionRequest
    ) -> Union[OpenAIChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        Handle OpenAI-compatible chat completion request.
        
        Args:
            request: Chat completion request
            
        Returns:
            Chat completion response or streaming generator
        """
        try:
            # Validate model
            if request.model not in self.available_models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {request.model} not found"
                )
            
            model_info = self.available_models[request.model]
            
            # Convert messages to our format
            task = self._extract_task_from_messages(request.messages)
            
            # Execute with appropriate agent/workflow
            if model_info["agent_type"] == "workflow":
                result = await self._execute_workflow(task, request)
            else:
                result = await self._execute_agent(task, request, model_info)
            
            # Return response
            if request.stream:
                return self._create_streaming_response(result, request)
            else:
                return self._create_completion_response(result, request)

        except Exception as e:
            _backend_logger.error(
                "Chat completion failed",
                LogCategory.API_OPERATIONS,
                "app.integrations.openwebui.pipeline",
                data={"error": str(e), "model": request.model}
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    def _extract_task_from_messages(self, messages: List[OpenAIMessage]) -> str:
        """Extract the main task from conversation messages."""
        # Get the last user message as the primary task
        user_messages = [msg for msg in messages if msg.role == "user"]
        if user_messages:
            return user_messages[-1].content
        
        # Fallback to any message content
        if messages:
            return messages[-1].content
        
        return "Hello! How can I help you today?"
    
    async def _execute_agent(
        self,
        task: str,
        request: OpenAIChatCompletionRequest,
        model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute task with a single agent."""
        try:
            # Create agent configuration
            agent_config = {
                "name": model_info["name"],
                "description": model_info["description"],
                "capabilities": model_info["capabilities"],
                "temperature": request.temperature or 0.7,
                "max_tokens": request.max_tokens or 2048,
                "system_prompt": f"You are {model_info['name']}. {model_info['description']}"
            }
            
            # Create agent
            agent_id = await self.orchestrator.create_agent(
                agent_type=model_info["agent_type"],
                config=agent_config
            )
            
            # Get the agent and execute
            agent = await self.orchestrator.get_agent(agent_id)
            if not agent:
                raise Exception(f"Failed to create agent {agent_id}")
            
            # Execute the task
            result = await agent.execute(
                task=task,
                context={
                    "user": request.user,
                    "model": request.model,
                    "openwebui_request": True
                }
            )
            
            return result

        except Exception as e:
            _backend_logger.error(
                "Agent execution failed",
                LogCategory.API_OPERATIONS,
                "app.integrations.openwebui.pipeline",
                data={"error": str(e)}
            )
            raise

    async def _execute_workflow(
        self,
        task: str,
        request: OpenAIChatCompletionRequest
    ) -> Dict[str, Any]:
        """Execute task with multi-agent workflow."""
        try:
            # Execute workflow
            result = await self.orchestrator.execute_workflow(
                workflow_id="default_multi_agent",
                inputs={
                    "task": task,
                    "temperature": request.temperature or 0.7,
                    "max_tokens": request.max_tokens or 2048,
                    "user": request.user,
                    "model": request.model
                }
            )

            return result

        except Exception as e:
            _backend_logger.error(
                "Workflow execution failed",
                LogCategory.API_OPERATIONS,
                "app.integrations.openwebui.pipeline",
                data={"error": str(e)}
            )
            raise
    
    def _create_completion_response(
        self,
        result: Dict[str, Any],
        request: OpenAIChatCompletionRequest
    ) -> OpenAIChatCompletionResponse:
        """Create OpenAI-compatible completion response."""
        
        # Extract content from result
        content = self._extract_content_from_result(result)
        
        response = OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(request.messages[-1].content.split()) if request.messages else 0,
                "completion_tokens": len(content.split()),
                "total_tokens": len(request.messages[-1].content.split()) + len(content.split()) if request.messages else len(content.split())
            }
        )
        
        return response
    
    async def _create_streaming_response(
        self,
        result: Dict[str, Any],
        request: OpenAIChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Create OpenAI-compatible streaming response."""
        
        content = self._extract_content_from_result(result)
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        # Split content into chunks for streaming
        words = content.split()
        chunk_size = 3  # Words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunk = OpenAIChatCompletionChunk(
                id=response_id,
                created=created,
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk_content + (" " if i + chunk_size < len(words) else "")
                        },
                        "finish_reason": None
                    }
                ]
            )
            
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0.05)  # Small delay for streaming effect
        
        # Send final chunk
        final_chunk = OpenAIChatCompletionChunk(
            id=response_id,
            created=created,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        )
        
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    
    def _extract_content_from_result(self, result: Dict[str, Any]) -> str:
        """Extract content from agent/workflow result."""
        
        # Try different result formats
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                return last_message["content"]
        
        if "outputs" in result and result["outputs"]:
            if isinstance(result["outputs"], str):
                return result["outputs"]
            elif isinstance(result["outputs"], dict):
                # Try to find text content
                for key in ["result", "content", "response", "output"]:
                    if key in result["outputs"]:
                        return str(result["outputs"][key])
        
        if "agent_outputs" in result:
            # Multi-agent workflow result
            outputs = []
            for agent_id, agent_result in result["agent_outputs"].items():
                if isinstance(agent_result, dict) and "outputs" in agent_result:
                    outputs.append(f"Agent {agent_id}: {agent_result['outputs']}")
            
            if outputs:
                return "\n\n".join(outputs)
        
        # Fallback
        return f"Task completed successfully. Status: {result.get('status', 'unknown')}"


# Global pipeline instance
openwebui_pipeline = OpenWebUIPipeline()
