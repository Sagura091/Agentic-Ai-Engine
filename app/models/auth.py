"""
OPTIMIZED Authentication and User Management Models

This module contains the essential database models for the optimized Agentic AI platform.
Only includes models that match the simplified database schema.

PRESERVED:
- UserDB: Essential user authentication
- ConversationDB: Chat history management  
- MessageDB: Message storage
- Pydantic models for API requests/responses

REMOVED:
- ProjectDB, ProjectMemberDB: Project management not implemented
- NotificationDB: Notifications not implemented
- UserAPIKeyDB, UserAgentDB, UserWorkflowDB: Complex ownership tracking removed
- KeycloakConfigDB: SSO not implemented in optimized schema
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from sqlalchemy import Column, String, Boolean, DateTime, Text, ForeignKey, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
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
    name = Column(String(255), nullable=True)  # Display name/full name
    
    # Authentication (ESSENTIAL ONLY)
    hashed_password = Column(String(255), nullable=False)
    password_salt = Column(String(255), nullable=True)  # For password hashing

    # Account status (ESSENTIAL ONLY)
    is_active = Column(Boolean, default=True, nullable=False)

    # User Groups (integrated roles - simplified 3-tier system)
    user_group = Column(String(50), default='user', nullable=False)  # user, moderator, admin

    # Login tracking and security
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    login_count = Column(Integer, default=0, nullable=False)

    # API Keys Storage (ESSENTIAL for external providers)
    api_keys = Column(JSON, default=dict)  # {"openai": "sk-...", "anthropic": "sk-...", "google": "...", "microsoft": "..."}

    # Timestamps (matching optimized database schema)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships (OPTIMIZED - only essential tables)
    conversations = relationship("ConversationDB", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

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

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission based on user group."""
        permissions = {
            'user': ['read', 'create_conversation', 'create_message'],
            'moderator': ['read', 'write', 'moderate', 'create_conversation', 'create_message', 'manage_users'],
            'admin': ['*']  # All permissions
        }
        user_permissions = permissions.get(self.user_group, [])
        return '*' in user_permissions or permission in user_permissions

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        if not self.api_keys:
            return None
        return self.api_keys.get(provider)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a specific provider."""
        if not self.api_keys:
            self.api_keys = {}
        self.api_keys[provider] = api_key

    def remove_api_key(self, provider: str) -> bool:
        """Remove API key for a specific provider. Returns True if key existed."""
        if not self.api_keys or provider not in self.api_keys:
            return False
        del self.api_keys[provider]
        return True

    def get_available_providers(self) -> list:
        """Get list of providers with stored API keys."""
        if not self.api_keys:
            return []
        return list(self.api_keys.keys())


class ConversationDB(Base):
    """Conversation model for chat history management."""

    __tablename__ = "conversations"
    __table_args__ = {'extend_existing': True}
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'), nullable=True, index=True)  # Optional agent

    # Conversation information
    title = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Metadata
    conversation_metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("UserDB", back_populates="conversations")
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
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False, index=True)
    
    # Message information
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    message_metadata = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("ConversationDB", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, conversation_id={self.conversation_id})>"


# ============================================================================
# PYDANTIC MODELS FOR API REQUESTS/RESPONSES
# ============================================================================

class UserCreate(BaseModel):
    """OPTIMIZED user creation request model."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    name: str = Field(..., min_length=1, max_length=255, description="Full name or display name")
    password: str = Field(..., min_length=8, description="Password")


class UserLogin(BaseModel):
    """User login request model."""
    username_or_email: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(default=False, description="Remember login")


class UserResponse(BaseModel):
    """OPTIMIZED user response model."""
    id: UUID4  # Changed from str to UUID4 to match database UUID type
    username: str
    email: str
    name: Optional[str] = None  # Display name/full name
    is_active: bool
    user_group: str
    api_keys: Optional[Dict[str, str]] = None  # Provider -> API key mapping
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ConversationCreate(BaseModel):
    """Model for creating a new conversation."""
    title: Optional[str] = None
    conversation_metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """Response model for conversation data."""
    id: UUID4
    user_id: UUID4
    title: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    """Model for creating a new message."""
    role: str
    content: str
    message_metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    """Response model for message data."""
    id: UUID4
    conversation_id: UUID4
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# API KEY MANAGEMENT MODELS
# ============================================================================

class APIKeyUpdate(BaseModel):
    """Model for updating user API keys."""
    provider: str = Field(..., description="Provider name (openai, anthropic, google, microsoft, etc.)")
    api_key: str = Field(..., description="The API key")


class APIKeyDelete(BaseModel):
    """Model for deleting user API keys."""
    provider: str = Field(..., description="Provider name to remove")


class APIKeysResponse(BaseModel):
    """Response model for user API keys (without exposing actual keys)."""
    providers: list = Field(..., description="List of providers with stored keys")

    class Config:
        from_attributes = True
