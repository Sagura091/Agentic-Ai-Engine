"""
Comprehensive input validation schemas for all API endpoints.

This module provides strict validation for all API inputs to prevent
security vulnerabilities and ensure data integrity.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from enum import Enum


class AgentType(str, Enum):
    """Valid agent types."""
    RAG = "rag"
    REACT = "react"
    AUTONOMOUS = "autonomous"
    CONVERSATIONAL = "conversational"
    WORKFLOW = "workflow"


class AgentStatus(str, Enum):
    """Valid agent statuses."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"


class ToolCategory(str, Enum):
    """Valid tool categories."""
    RAG_ENABLED = "rag_enabled"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    BUSINESS = "business"
    UTILITY = "utility"
    DATA = "data"
    ANALYSIS = "analysis"
    SECURITY = "security"
    AUTOMATION = "automation"


class ScrapingMode(str, Enum):
    """Valid scraping modes."""
    BASIC = "basic"
    ADVANCED = "advanced"
    STEALTH = "stealth"
    REVOLUTIONARY = "revolutionary"


# ============================================================================
# AGENT VALIDATION SCHEMAS
# ============================================================================

class AgentCreateRequest(BaseModel):
    """Strict validation for agent creation."""
    name: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        regex=r'^[a-zA-Z0-9_-]+$',
        description="Agent name (alphanumeric, underscore, hyphen only)"
    )
    description: Optional[str] = Field(
        None, 
        max_length=500,
        description="Agent description"
    )
    agent_type: AgentType = Field(
        default=AgentType.REACT,
        description="Type of agent to create"
    )
    model: str = Field(
        default="llama3.1:8b",
        min_length=1,
        max_length=100,
        regex=r'^[a-zA-Z0-9._:-]+$',
        description="LLM model to use"
    )
    model_provider: str = Field(
        default="ollama",
        regex=r'^(ollama|openai|anthropic|google)$',
        description="LLM provider"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature"
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32000,
        description="Maximum tokens"
    )
    system_prompt: Optional[str] = Field(
        None,
        max_length=10000,
        description="System prompt"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        max_items=20,
        description="Agent capabilities"
    )
    tools: List[str] = Field(
        default_factory=list,
        max_items=50,
        description="Available tools"
    )
    autonomy_level: str = Field(
        default="basic",
        regex=r'^(basic|intermediate|advanced|expert)$',
        description="Autonomy level"
    )
    learning_mode: str = Field(
        default="passive",
        regex=r'^(passive|active|adaptive)$',
        description="Learning mode"
    )
    decision_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Decision threshold"
    )

    @validator('capabilities')
    def validate_capabilities(cls, v):
        """Validate capabilities list."""
        valid_capabilities = {
            'reasoning', 'tool_use', 'memory', 'learning', 'communication',
            'monitoring', 'analysis', 'optimization', 'security', 'collaboration'
        }
        for capability in v:
            if capability not in valid_capabilities:
                raise ValueError(f"Invalid capability: {capability}")
        return v

    @validator('tools')
    def validate_tools(cls, v):
        """Validate tools list."""
        if len(v) > 50:
            raise ValueError("Too many tools (max 50)")
        return v


class AgentUpdateRequest(BaseModel):
    """Strict validation for agent updates."""
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        regex=r'^[a-zA-Z0-9_-]+$'
    )
    description: Optional[str] = Field(None, max_length=500)
    status: Optional[AgentStatus] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)
    system_prompt: Optional[str] = Field(None, max_length=10000)
    capabilities: Optional[List[str]] = Field(None, max_items=20)
    tools: Optional[List[str]] = Field(None, max_items=50)
    autonomy_level: Optional[str] = Field(
        None,
        regex=r'^(basic|intermediate|advanced|expert)$'
    )
    learning_mode: Optional[str] = Field(
        None,
        regex=r'^(passive|active|adaptive)$'
    )
    decision_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class AgentQueryRequest(BaseModel):
    """Strict validation for agent queries."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Query message"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context"
    )
    stream: bool = Field(default=False, description="Stream response")
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )

    @validator('context')
    def validate_context(cls, v):
        """Validate context dictionary."""
        if v and len(str(v)) > 50000:  # 50KB limit
            raise ValueError("Context too large (max 50KB)")
        return v


# ============================================================================
# TOOL VALIDATION SCHEMAS
# ============================================================================

class ToolCreateRequest(BaseModel):
    """Strict validation for tool creation."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        regex=r'^[a-zA-Z0-9_-]+$'
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )
    category: ToolCategory
    access_level: str = Field(
        ...,
        regex=r'^(public|private|conditional)$'
    )
    requires_rag: bool = Field(default=False)
    use_cases: List[str] = Field(
        default_factory=list,
        max_items=20
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Tool parameters schema"
    )

    @validator('use_cases')
    def validate_use_cases(cls, v):
        """Validate use cases."""
        valid_use_cases = {
            'calculation', 'research', 'communication', 'analysis',
            'automation', 'data_processing', 'security', 'monitoring'
        }
        for use_case in v:
            if use_case not in valid_use_cases:
                raise ValueError(f"Invalid use case: {use_case}")
        return v


# ============================================================================
# WEB SCRAPING VALIDATION SCHEMAS
# ============================================================================

class WebScrapingRequest(BaseModel):
    """Strict validation for web scraping requests."""
    url: Optional[HttpUrl] = Field(None, description="Target URL")
    query: Optional[str] = Field(
        None,
        min_length=1,
        max_length=500,
        description="Search query"
    )
    search_engines: List[str] = Field(
        default=["google", "bing", "duckduckgo"],
        max_items=5
    )
    num_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of results"
    )
    scraping_mode: ScrapingMode = Field(
        default=ScrapingMode.REVOLUTIONARY,
        description="Scraping mode"
    )
    use_javascript: bool = Field(default=True)
    extract_links: bool = Field(default=True)
    extract_images: bool = Field(default=True)
    extract_videos: bool = Field(default=True)
    extract_documents: bool = Field(default=True)
    extract_structured_data: bool = Field(default=True)
    crawl_depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Crawling depth"
    )
    use_proxy: bool = Field(default=False)
    bypass_cloudflare: bool = Field(default=True)
    human_behavior: bool = Field(default=True)
    stealth_mode: bool = Field(default=True)
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries"
    )

    @validator('search_engines')
    def validate_search_engines(cls, v):
        """Validate search engines."""
        valid_engines = {"google", "bing", "duckduckgo", "yahoo"}
        for engine in v:
            if engine not in valid_engines:
                raise ValueError(f"Invalid search engine: {engine}")
        return v

    @validator('url')
    def validate_url_or_query(cls, v, values):
        """Ensure either URL or query is provided."""
        if not v and not values.get('query'):
            raise ValueError("Either URL or query must be provided")
        return v


# ============================================================================
# WORKFLOW VALIDATION SCHEMAS
# ============================================================================

class WorkflowCreateRequest(BaseModel):
    """Strict validation for workflow creation."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        regex=r'^[a-zA-Z0-9_-]+$'
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )
    steps: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="Workflow steps"
    )
    execution_mode: str = Field(
        default="sequential",
        regex=r'^(sequential|parallel|autonomous)$'
    )
    timeout: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Workflow timeout"
    )

    @validator('steps')
    def validate_steps(cls, v):
        """Validate workflow steps."""
        for i, step in enumerate(v):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} must be a dictionary")
            if 'type' not in step:
                raise ValueError(f"Step {i} must have a 'type' field")
            if 'name' not in step:
                raise ValueError(f"Step {i} must have a 'name' field")
        return v


# ============================================================================
# USER VALIDATION SCHEMAS
# ============================================================================

class UserCreateRequest(BaseModel):
    """Strict validation for user creation."""
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        regex=r'^[a-zA-Z0-9_-]+$'
    )
    email: EmailStr = Field(..., description="User email")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password"
    )
    full_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Full name"
    )

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        return v


class UserUpdateRequest(BaseModel):
    """Strict validation for user updates."""
    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        regex=r'^[a-zA-Z0-9_-]+$'
    )
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="User preferences"
    )


# ============================================================================
# CONVERSATION VALIDATION SCHEMAS
# ============================================================================

class ConversationCreateRequest(BaseModel):
    """Strict validation for conversation creation."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Conversation title"
    )
    agent_id: Optional[str] = Field(
        None,
        regex=r'^[a-f0-9-]{36}$',
        description="Agent UUID"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Conversation context"
    )


class MessageCreateRequest(BaseModel):
    """Strict validation for message creation."""
    content: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Message content"
    )
    role: str = Field(
        ...,
        regex=r'^(user|assistant|system)$',
        description="Message role"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Message metadata"
    )

    @validator('content')
    def validate_content(cls, v):
        """Validate message content."""
        if len(v.strip()) == 0:
            raise ValueError("Message content cannot be empty")
        return v.strip()


# ============================================================================
# RAG VALIDATION SCHEMAS
# ============================================================================

class DocumentUploadRequest(BaseModel):
    """Strict validation for document upload."""
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        regex=r'^[a-zA-Z0-9._-]+$'
    )
    content_type: str = Field(
        ...,
        regex=r'^(text/|application/pdf|application/msword|application/vnd\.openxmlformats-officedocument\.)'
    )
    size: int = Field(
        ...,
        ge=1,
        le=50 * 1024 * 1024,  # 50MB limit
        description="File size in bytes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Document metadata"
    )

    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate content type."""
        allowed_types = {
            'text/plain', 'text/markdown', 'text/html',
            'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        if v not in allowed_types:
            raise ValueError(f"Unsupported content type: {v}")
        return v


class RAGQueryRequest(BaseModel):
    """Strict validation for RAG queries."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query"
    )
    collection_id: Optional[str] = Field(
        None,
        regex=r'^[a-f0-9-]{36}$',
        description="Collection UUID"
    )
    num_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold"
    )


# ============================================================================
# PAGINATION AND FILTERING
# ============================================================================

class PaginationParams(BaseModel):
    """Strict validation for pagination."""
    page: int = Field(
        default=1,
        ge=1,
        le=10000,
        description="Page number"
    )
    size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Page size"
    )
    sort_by: Optional[str] = Field(
        None,
        max_length=50,
        description="Sort field"
    )
    sort_order: str = Field(
        default="asc",
        regex=r'^(asc|desc)$',
        description="Sort order"
    )


class FilterParams(BaseModel):
    """Strict validation for filtering."""
    status: Optional[str] = Field(
        None,
        regex=r'^(active|inactive|paused|error)$'
    )
    agent_type: Optional[str] = Field(
        None,
        regex=r'^(rag|react|autonomous|conversational|workflow)$'
    )
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = Field(
        None,
        max_items=20,
        description="Filter by tags"
    )

    @validator('created_after', 'created_before')
    def validate_dates(cls, v):
        """Validate date filters."""
        if v and v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class StandardResponse(BaseModel):
    """Standard API response format."""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response format."""
    success: bool = False
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# ============================================================================
# SECURITY VALIDATION
# ============================================================================

class SecurityHeaders(BaseModel):
    """Security headers validation."""
    user_agent: Optional[str] = Field(None, max_length=500)
    x_forwarded_for: Optional[str] = Field(None, max_length=100)
    x_real_ip: Optional[str] = Field(None, max_length=45)  # IPv6 max length
    authorization: Optional[str] = Field(None, max_length=1000)
    
    @validator('x_forwarded_for', 'x_real_ip')
    def validate_ip_addresses(cls, v):
        """Validate IP addresses."""
        if v:
            import ipaddress
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError(f"Invalid IP address: {v}")
        return v


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Requests per minute"
    )
    requests_per_hour: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Requests per hour"
    )
    burst_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Burst limit"
    )


# Export all schemas
__all__ = [
    "AgentCreateRequest", "AgentUpdateRequest", "AgentQueryRequest",
    "ToolCreateRequest", "WebScrapingRequest", "WorkflowCreateRequest",
    "UserCreateRequest", "UserUpdateRequest", "ConversationCreateRequest",
    "MessageCreateRequest", "DocumentUploadRequest", "RAGQueryRequest",
    "PaginationParams", "FilterParams", "StandardResponse", "ErrorResponse",
    "SecurityHeaders", "RateLimitConfig"
]


