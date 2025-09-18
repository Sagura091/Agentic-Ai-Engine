"""
Models package for the Agentic AI Microservice.

This package contains database models, schemas, and database configuration.
"""

# Import all models to ensure they are registered with SQLAlchemy
from .agent import Agent, Conversation, Message, TaskExecution
from .workflow import Workflow, WorkflowExecution, WorkflowStepExecution, WorkflowTemplate
from .tool import Tool, AgentTool, ToolExecution, ToolCategory, ToolTemplate
from .user import User, UserCreate, UserUpdate, UserInDB, Token, TokenData
from .autonomous import (
    AutonomousAgentState, AutonomousGoalDB, AutonomousDecisionDB,
    AgentMemoryDB, LearningExperienceDB, PerformanceMetricDB
)
from .document import (
    DocumentDB, DocumentChunkDB, DocumentMetadata, DocumentChunkMetadata,
    DocumentCreateRequest, DocumentUploadResponse, DocumentSearchResult
)
from .database.base import Base

__all__ = [
    # Database base
    "Base",

    # Agent models
    "Agent",
    "Conversation",
    "Message",
    "TaskExecution",

    # Workflow models
    "Workflow",
    "WorkflowExecution",
    "WorkflowStepExecution",
    "WorkflowTemplate",

    # Tool models
    "Tool",
    "AgentTool",
    "ToolExecution",
    "ToolCategory",
    "ToolTemplate",

    # User models
    "User",
    "UserCreate",
    "UserUpdate",
    "UserInDB",
    "Token",
    "TokenData",

    # Autonomous agent models
    "AutonomousAgentState",
    "AutonomousGoalDB",
    "AutonomousDecisionDB",
    "AgentMemoryDB",
    "LearningExperienceDB",
    "PerformanceMetricDB",

    # Document models
    "DocumentDB",
    "DocumentChunkDB",
    "DocumentMetadata",
    "DocumentChunkMetadata",
    "DocumentCreateRequest",
    "DocumentUploadResponse",
    "DocumentSearchResult"
]
