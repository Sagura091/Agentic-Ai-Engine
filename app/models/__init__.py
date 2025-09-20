"""
Models package for the Agentic AI Microservice.

This package contains database models, schemas, and database configuration.
"""

# Import all models to ensure they are registered with SQLAlchemy
from .agent import Agent, Conversation, Message, TaskExecution
from .workflow import (
    Workflow, WorkflowExecution, WorkflowStepExecution, WorkflowTemplate,
    NodeDefinition, WorkflowNode, WorkflowConnection, NodeExecutionState
)
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
# New enhanced models
from .model_management import (
    ModelRegistry, ModelUsageLog, ModelDownloadHistory, ModelPerformanceMetrics
)
from .knowledge_base import (
    KnowledgeBase, KnowledgeBaseAccess, KnowledgeBaseUsageLog, KnowledgeBaseTemplate
)
from .enhanced_user import (
    UserDB, UserSession, Role, UserRoleAssignment, UserAuditLog
)
from .api_management import (
    APIKey, APIKeyUsageLog, RateLimitLog, APIQuotaUsage, APIEndpointMetrics
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

    # Advanced node system models
    "NodeDefinition",
    "WorkflowNode",
    "WorkflowConnection",
    "NodeExecutionState",

    # Tool models
    "Tool",
    "AgentTool",
    "ToolExecution",
    "ToolCategory",
    "ToolTemplate",

    # User models (legacy)
    "User",
    "UserCreate",
    "UserUpdate",
    "UserInDB",
    "Token",
    "TokenData",

    # Enhanced user models
    "UserDB",
    "UserSession",
    "Role",
    "UserRoleAssignment",
    "UserAuditLog",

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
    "DocumentSearchResult",

    # Model management models
    "ModelRegistry",
    "ModelUsageLog",
    "ModelDownloadHistory",
    "ModelPerformanceMetrics",

    # Knowledge base models
    "KnowledgeBase",
    "KnowledgeBaseAccess",
    "KnowledgeBaseUsageLog",
    "KnowledgeBaseTemplate",

    # API management models
    "APIKey",
    "APIKeyUsageLog",
    "RateLimitLog",
    "APIQuotaUsage",
    "APIEndpointMetrics"
]
