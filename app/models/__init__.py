"""
Models package for the Agentic AI Microservice.

This package contains database models, schemas, and database configuration.
"""

# Import all models to ensure they are registered with SQLAlchemy
from .agent import Agent, TaskExecution
from .auth import ConversationDB as Conversation, MessageDB as Message
from .workflow import (
    Workflow, WorkflowExecution, WorkflowStepExecution, WorkflowTemplate,
    NodeDefinition, WorkflowNode, WorkflowConnection, NodeExecutionState
)
from .tool import Tool, AgentTool, ToolExecution, ToolCategory, ToolTemplate
from .user import User, UserCreate, UserUpdate, UserInDB, Token, TokenData
from .autonomous import (
    AutonomousAgentState, AutonomousGoalDB, AutonomousDecisionDB,
    AgentMemoryDB, LearningExperienceDB
    # REMOVED: PerformanceMetricDB (system metrics, not learning-related)
)
from .document import (
    DocumentDB, DocumentChunkDB, DocumentMetadata, DocumentChunkMetadata,
    DocumentCreateRequest, DocumentUploadResponse, DocumentSearchResult
)
# OPTIMIZED: Only essential models imported
from .knowledge_base import (
    KnowledgeBase, KnowledgeBaseAccess
    # REMOVED: KnowledgeBaseUsageLog, KnowledgeBaseTemplate (unnecessary complexity)
)
from .auth import UserDB
from .enhanced_user import (
    UserSession
    # REMOVED: Role, UserRoleAssignment, UserAuditLog (roles in users.user_group, audit not needed)
)
# REMOVED: All model_management imports (handled by external APIs)
# API management models removed - replaced with user-owned API key system
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

    # Enhanced user models (OPTIMIZED)
    "UserDB",
    "UserSession",
    # REMOVED: "Role", "UserRoleAssignment", "UserAuditLog" (roles in users.user_group)

    # Autonomous agent models (PRESERVED - Essential for Learning)
    "AutonomousAgentState",
    "AutonomousGoalDB",
    "AutonomousDecisionDB",
    "AgentMemoryDB",
    "LearningExperienceDB",
    # REMOVED: "PerformanceMetricDB" (system metrics, not learning)

    # Document models
    "DocumentDB",
    "DocumentChunkDB",
    "DocumentMetadata",
    "DocumentChunkMetadata",
    "DocumentCreateRequest",
    "DocumentUploadResponse",
    "DocumentSearchResult",

    # Knowledge base models (OPTIMIZED)
    "KnowledgeBase",
    "KnowledgeBaseAccess",
    # REMOVED: "KnowledgeBaseUsageLog", "KnowledgeBaseTemplate" (unnecessary complexity)

    # REMOVED: All model management models (handled by external APIs)
    # REMOVED: "ModelRegistry", "ModelUsageLog", "ModelDownloadHistory", "ModelPerformanceMetrics"

    # API management models removed - replaced with user-owned API key system in auth.py
]
