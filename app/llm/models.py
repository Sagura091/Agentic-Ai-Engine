"""
LLM Provider Models and Data Structures.

This module defines the data models, enums, and structures used across
the LLM provider integration system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    CONVERSATION = "conversation"
    FUNCTION_CALLING = "function_calling"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CODE_GENERATION = "code_generation"
    MULTIMODAL = "multimodal"


class ProviderCredentials(BaseModel):
    """Provider authentication credentials."""
    provider: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    additional_headers: Optional[Dict[str, str]] = None
    
    class Config:
        extra = "allow"


class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    provider: ProviderType = Field(..., description="Provider type")
    description: Optional[str] = None
    capabilities: List[ModelCapability] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    context_length: Optional[int] = None
    cost_per_token: Optional[float] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="available")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class LLMConfig(BaseModel):
    """Configuration for LLM usage."""
    provider: ProviderType = Field(..., description="LLM provider")
    model_id: str = Field(..., description="Model identifier")
    model_name: Optional[str] = None
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, gt=0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    
    # Provider-specific settings
    credentials: Optional[ProviderCredentials] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    # Manual selection enhancements
    manual_selection: bool = Field(default=False, description="Whether this model was manually selected by user")
    selection_reason: Optional[str] = Field(default=None, description="Reason for model selection")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific preferences")
    auto_optimize: bool = Field(default=True, description="Allow automatic optimization of parameters")
    recommended_for: List[str] = Field(default_factory=list, description="Tasks this model is recommended for")

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v


class ProviderStatus(BaseModel):
    """Status information for a provider."""
    provider: ProviderType
    is_available: bool = False
    is_authenticated: bool = False
    error_message: Optional[str] = None
    available_models: List[str] = Field(default_factory=list)
    last_checked: datetime = Field(default_factory=datetime.now)
    response_time_ms: Optional[float] = None


class LLMResponse(BaseModel):
    """Response from LLM generation."""
    content: str
    model: str
    provider: ProviderType
    usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class StreamingLLMResponse(BaseModel):
    """Streaming response chunk from LLM."""
    content: str
    is_complete: bool = False
    model: str
    provider: ProviderType
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Default model configurations for each provider
DEFAULT_MODELS = {
    ProviderType.OLLAMA: [
        "llama3.2:latest",
        "llama3.1:latest", 
        "qwen2.5:latest",
        "mistral:latest",
        "codellama:latest",
        "llama3.2:3b",
        "phi3:latest"
    ],
    ProviderType.OPENAI: [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini"
    ],
    ProviderType.ANTHROPIC: [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ],
    ProviderType.GOOGLE: [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",
        "gemini-pro-vision"
    ]
}


# Provider-specific default parameters
PROVIDER_DEFAULTS = {
    ProviderType.OLLAMA: {
        "base_url": "http://localhost:11434",
        "timeout": 120,
        "retry_attempts": 3
    },
    ProviderType.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "timeout": 60,
        "max_retries": 3
    },
    ProviderType.ANTHROPIC: {
        "base_url": "https://api.anthropic.com",
        "timeout": 60,
        "max_retries": 3
    },
    ProviderType.GOOGLE: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "timeout": 60,
        "max_retries": 3
    }
}
