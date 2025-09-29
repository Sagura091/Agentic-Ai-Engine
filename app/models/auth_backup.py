"""
Authentication and User Management Models.

This module provides SQLAlchemy models for user authentication,
session management, and user profiles with proper database integration.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4

from sqlalchemy import Column, String, Boolean, DateTime, JSON, Integer, Float, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, EmailStr
from pydantic.types import UUID4

from app.models.database.base import Base


class UserDB(Base):
    """OPTIMIZED user model matching simplified database schema."""

    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Basic information (ESSENTIAL ONLY)
    username = Column(String(255), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)

    # Authentication (ESSENTIAL ONLY)
    hashed_password = Column(String(255), nullable=False)

    # Account status (ESSENTIAL ONLY)
    is_active = Column(Boolean, default=True, nullable=False)

    # User Groups (integrated roles - simplified 3-tier system)
    user_group = Column(String(50), default='user', nullable=False)  # user, moderator, admin

    # Timestamps (matching optimized database schema)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # REMOVED: All extra columns not in optimized schema
    # (deleted_at, user_metadata, profile fields, MFA, SSO, etc.)
    
    # Relationships (OPTIMIZED - only essential tables)
    conversations = relationship("ConversationDB", back_populates="user", cascade="all, delete-orphan")
    # REMOVED: sessions, api_keys, owned_agents, owned_workflows, projects, notifications,
    # project_memberships, role_assignments, audit_logs (tables not in optimized schema)

    def __repr__(self):
        return f"<UserDB(id={self.id}, username='{self.username}', email='{self.email}', group='{self.user_group}')>"

    @property
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.user_group == 'admin'

    @property
    def is_moderator(self) -> bool:
        """Check if user is moderator or admin."""
        return self.user_group in ['moderator', 'admin']

    @property
    def is_superuser(self) -> bool:
        """Compatibility property for existing code."""
        return self.is_admin

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission based on group."""
        permissions = {
            'user': ['read_own', 'write_own', 'create_agents', 'create_workflows', 'create_projects'],
            'moderator': ['read_own', 'write_own', 'create_agents', 'create_workflows', 'create_projects',
                         'moderate_content', 'manage_users', 'view_all_projects'],
            'admin': ['*']  # All permissions
        }

        user_permissions = permissions.get(self.user_group, [])
        return '*' in user_permissions or permission in user_permissions


# REMOVED: ProjectDB, ProjectMemberDB, NotificationDB, UserAPIKeyDB, UserAgentDB,
# UserWorkflowDB, KeycloakConfigDB (not in optimized schema)


class ProjectMemberDB(Base):
    """Project membership model for collaboration."""

    __tablename__ = "project_members"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Membership information
    role = Column(String(50), default='member')  # owner, admin, member, viewer
    permissions = Column(JSON, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    project = relationship("ProjectDB", back_populates="members")
    user = relationship("UserDB", foreign_keys=[user_id])
    inviter = relationship("UserDB", foreign_keys=[invited_by])

    def __repr__(self):
        return f"<ProjectMember(project_id={self.project_id}, user_id={self.user_id}, role={self.role})>"


class ConversationDB(Base):
    """Conversation model for chat history management."""

    __tablename__ = "conversations"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    
    # Conversation information
    title = Column(String(255), nullable=False)
    description = Column(Text)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True, index=True)  # Associated agent
    
    # Conversation settings
    is_archived = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    tags = Column(JSON, default=list)
    conversation_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_message_at = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))  # Soft delete
    
    # Relationships
    user = relationship("UserDB", back_populates="conversations")
    project = relationship("ProjectDB", back_populates="conversations")
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship("MessageDB", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title}, user_id={self.user_id})>"


class MessageDB(Base):
    """Message model for conversation history."""

    __tablename__ = "messages"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    
    # Message information
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default='text')  # text, markdown, html, json
    
    # Message metadata
    agent_id = Column(String(255))  # Which agent sent this message
    tool_calls = Column(JSON, default=list)
    attachments = Column(JSON, default=list)
    message_metadata = Column(JSON, default=dict)
    
    # Message status
    is_edited = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    parent_message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    conversation = relationship("ConversationDB", back_populates="messages")
    parent_message = relationship("MessageDB", remote_side=[id])

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, conversation_id={self.conversation_id})>"


class NotificationDB(Base):
    """Notification model for user alerts."""

    __tablename__ = "notifications"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Notification information
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(50), nullable=False)  # info, success, warning, error
    category = Column(String(50), default='general')  # agent, system, project, etc.
    
    # Notification data
    data = Column(JSON, default=dict)
    action_url = Column(String(500))
    
    # Status
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("UserDB", back_populates="notifications")

    def __repr__(self):
        return f"<Notification(id={self.id}, title={self.title}, user_id={self.user_id})>"


# Pydantic models for API requests/responses
class UserCreate(BaseModel):
    """OPTIMIZED user creation request model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    # REMOVED: full_name (not in optimized schema)


class UserLogin(BaseModel):
    """User login request model."""
    username_or_email: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(default=False, description="Remember login")


class UserResponse(BaseModel):
    """OPTIMIZED user response model."""
    id: str
    username: str
    email: str
    is_active: bool
    user_group: str
    created_at: datetime
    updated_at: datetime
    # REMOVED: full_name, is_verified, avatar_url, bio, timezone, language,
    # subscription_tier, last_login (not in optimized schema)

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ProjectCreate(BaseModel):
    """Project creation request model."""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    color: Optional[str] = Field("#3B82F6", description="Project color (hex)")
    icon: Optional[str] = Field("folder", description="Project icon")
    is_public: bool = Field(default=False, description="Is project public")


class ProjectResponse(BaseModel):
    """Project response model."""
    id: str
    name: str
    description: Optional[str]
    color: str
    icon: str
    is_public: bool
    is_archived: bool
    owner_id: str
    created_at: datetime
    updated_at: datetime
    member_count: Optional[int] = 0

    class Config:
        from_attributes = True


# ============================================================================
# NEW ENHANCED MODELS FOR USER API KEYS, AGENT/WORKFLOW OWNERSHIP, AND SSO
# ============================================================================

class UserAPIKeyDB(Base):
    """User API keys for external providers (OpenAI, Anthropic, Google, Microsoft)."""

    __tablename__ = "user_api_keys"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign key
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # API key information
    provider = Column(String(50), nullable=False, index=True)  # openai, anthropic, google, microsoft
    key_name = Column(String(100), nullable=False)  # User-friendly name
    encrypted_api_key = Column(Text, nullable=False)  # Encrypted API key
    key_hash = Column(String(64), nullable=False, index=True)  # Hash for validation

    # Key configuration
    is_active = Column(Boolean, default=True, nullable=False)
    is_default = Column(Boolean, default=False)  # Default key for this provider

    # Usage tracking (not for billing, just monitoring)
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)

    # Key metadata
    key_metadata = Column(JSON, default=dict)  # Provider-specific settings

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True))  # Optional expiration

    # Relationships
    user = relationship("UserDB", back_populates="api_keys")

    def __repr__(self):
        return f"<UserAPIKeyDB(id={self.id}, user_id={self.user_id}, provider='{self.provider}', name='{self.key_name}')>"


class UserAgentDB(Base):
    """User-owned agents tracking."""

    __tablename__ = "user_agents"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign key
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Agent information
    agent_id = Column(String(255), nullable=False, index=True)  # Reference to actual agent
    agent_name = Column(String(255), nullable=False)
    agent_type = Column(String(50), nullable=False)  # basic, react, rag, autonomous, etc.

    # Agent configuration
    agent_config = Column(JSON, default=dict)
    is_public = Column(Boolean, default=False)  # Can other users see/use this agent
    is_template = Column(Boolean, default=False)  # Can be used as template

    # Status
    is_active = Column(Boolean, default=True)
    is_favorite = Column(Boolean, default=False)

    # Usage statistics
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Metadata
    agent_metadata = Column(JSON, default=dict)

    # Relationships
    user = relationship("UserDB", back_populates="owned_agents")

    def __repr__(self):
        return f"<UserAgentDB(id={self.id}, user_id={self.user_id}, agent_name='{self.agent_name}', type='{self.agent_type}')>"


class UserWorkflowDB(Base):
    """User-owned workflows tracking."""

    __tablename__ = "user_workflows"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign key
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Workflow information
    workflow_id = Column(String(255), nullable=False, index=True)  # Reference to actual workflow
    workflow_name = Column(String(255), nullable=False)
    workflow_type = Column(String(50), nullable=False)  # hierarchical, multi_agent, visual, etc.

    # Workflow configuration
    workflow_config = Column(JSON, default=dict)
    is_public = Column(Boolean, default=False)  # Can other users see/use this workflow
    is_template = Column(Boolean, default=False)  # Can be used as template

    # Status
    is_active = Column(Boolean, default=True)
    is_favorite = Column(Boolean, default=False)

    # Usage statistics
    execution_count = Column(Integer, default=0)
    last_executed = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Metadata
    workflow_metadata = Column(JSON, default=dict)

    # Relationships
    user = relationship("UserDB", back_populates="owned_workflows")

    def __repr__(self):
        return f"<UserWorkflowDB(id={self.id}, user_id={self.user_id}, workflow_name='{self.workflow_name}', type='{self.workflow_type}')>"


class KeycloakConfigDB(Base):
    """Keycloak SSO configuration."""

    __tablename__ = "keycloak_config"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Keycloak configuration
    realm = Column(String(255), nullable=False)
    server_url = Column(String(500), nullable=False)
    client_id = Column(String(255), nullable=False)
    client_secret = Column(String(500))  # Encrypted

    # Configuration settings
    is_active = Column(Boolean, default=True)
    auto_create_users = Column(Boolean, default=True)
    default_user_group = Column(String(20), default='user')

    # Role mapping
    role_mappings = Column(JSON, default=dict)  # Map Keycloak roles to local groups

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<KeycloakConfigDB(id={self.id}, realm='{self.realm}', client_id='{self.client_id}')>"


# ============================================================================
# PYDANTIC MODELS FOR NEW FEATURES
# ============================================================================

class UserAPIKeyCreate(BaseModel):
    """User API key creation model."""
    provider: str = Field(..., description="Provider name (openai, anthropic, google, microsoft)")
    key_name: str = Field(..., description="User-friendly name for the key")
    api_key: str = Field(..., description="The actual API key")
    is_default: bool = Field(default=False, description="Set as default key for this provider")
    key_metadata: Optional[dict] = Field(default=None, description="Provider-specific settings")


class UserAPIKeyResponse(BaseModel):
    """User API key response model."""
    id: str
    provider: str
    key_name: str
    is_active: bool
    is_default: bool
    last_used: Optional[datetime]
    usage_count: int
    created_at: datetime
    expires_at: Optional[datetime]
    key_metadata: dict

    class Config:
        from_attributes = True


class UserAPIKeyUpdate(BaseModel):
    """User API key update model."""
    key_name: Optional[str] = Field(default=None, description="Updated key name")
    is_active: Optional[bool] = Field(default=None, description="Active status")
    is_default: Optional[bool] = Field(default=None, description="Set as default for provider")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration date")
    key_metadata: Optional[dict] = Field(default=None, description="Provider-specific settings")


class UserAgentCreate(BaseModel):
    """User agent creation model."""
    agent_id: str = Field(..., description="Reference to actual agent")
    agent_name: str = Field(..., description="Agent name")
    agent_type: str = Field(..., description="Agent type")
    agent_config: Optional[dict] = Field(default=None, description="Agent configuration")
    is_public: bool = Field(default=False, description="Make agent public")
    is_template: bool = Field(default=False, description="Allow as template")


class UserAgentResponse(BaseModel):
    """User agent response model."""
    id: str
    agent_id: str
    agent_name: str
    agent_type: str
    is_public: bool
    is_template: bool
    is_active: bool
    is_favorite: bool
    usage_count: int
    last_used: Optional[datetime]
    created_at: datetime
    agent_metadata: dict

    class Config:
        from_attributes = True


class UserWorkflowCreate(BaseModel):
    """User workflow creation model."""
    workflow_id: str = Field(..., description="Reference to actual workflow")
    workflow_name: str = Field(..., description="Workflow name")
    workflow_type: str = Field(..., description="Workflow type")
    workflow_config: Optional[dict] = Field(default=None, description="Workflow configuration")
    is_public: bool = Field(default=False, description="Make workflow public")
    is_template: bool = Field(default=False, description="Allow as template")


class UserWorkflowResponse(BaseModel):
    """User workflow response model."""
    id: str
    workflow_id: str
    workflow_name: str
    workflow_type: str
    is_public: bool
    is_template: bool
    is_active: bool
    is_favorite: bool
    execution_count: int
    last_executed: Optional[datetime]
    created_at: datetime
    workflow_metadata: dict

    class Config:
        from_attributes = True


class KeycloakConfigCreate(BaseModel):
    """Keycloak configuration creation model."""
    realm: str = Field(..., description="Keycloak realm")
    server_url: str = Field(..., description="Keycloak server URL")
    client_id: str = Field(..., description="Client ID")
    client_secret: str = Field(..., description="Client secret")
    auto_create_users: bool = Field(default=True, description="Auto-create users from Keycloak")
    default_user_group: str = Field(default='user', description="Default group for new users")
    role_mappings: Optional[dict] = Field(default=None, description="Role mappings")


class KeycloakConfigResponse(BaseModel):
    """Keycloak configuration response model."""
    id: str
    realm: str
    server_url: str
    client_id: str
    is_active: bool
    auto_create_users: bool
    default_user_group: str
    role_mappings: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class KeycloakConfigCreate(BaseModel):
    """Keycloak configuration creation model."""
    realm: str = Field(..., description="Keycloak realm name")
    server_url: str = Field(..., description="Keycloak server URL")
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: str = Field(..., description="OAuth2 client secret")
    auto_create_users: bool = Field(default=True, description="Auto-create users from SSO")
    default_user_group: str = Field(default="user", description="Default group for SSO users")
    role_mappings: Optional[dict] = Field(default=None, description="Role mappings")


class KeycloakConfigUpdate(BaseModel):
    """Keycloak configuration update model."""
    realm: Optional[str] = Field(default=None, description="Keycloak realm name")
    server_url: Optional[str] = Field(default=None, description="Keycloak server URL")
    client_id: Optional[str] = Field(default=None, description="OAuth2 client ID")
    client_secret: Optional[str] = Field(default=None, description="OAuth2 client secret")
    is_active: Optional[bool] = Field(default=None, description="Configuration active status")
    auto_create_users: Optional[bool] = Field(default=None, description="Auto-create users from SSO")
    default_user_group: Optional[str] = Field(default=None, description="Default group for SSO users")
    role_mappings: Optional[dict] = Field(default=None, description="Role mappings")


class KeycloakConfigResponse(BaseModel):
    """Keycloak configuration response model."""
    id: str
    realm: str
    server_url: str
    client_id: str
    is_active: bool
    auto_create_users: bool
    default_user_group: str
    role_mappings: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Conversation Pydantic models
class ConversationCreate(BaseModel):
    """Model for creating a new conversation."""

    title: Optional[str] = None
    conversation_metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Response model for conversation data."""

    id: UUID4
    user_id: UUID4
    title: Optional[str]
    conversation_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    """Model for creating a new message."""

    role: str
    content: str
    conversation_metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    """Response model for message data."""

    id: UUID4
    conversation_id: UUID4
    role: str
    content: str
    conversation_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Notification Pydantic models
class NotificationResponse(BaseModel):
    """Response model for notification data."""

    id: UUID4
    user_id: UUID4
    title: str
    message: str
    notification_type: str
    is_read: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
