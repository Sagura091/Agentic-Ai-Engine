"""
User models for the agentic AI system.

This module defines the user data models and related functionality.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


class User(BaseModel):
    """User model."""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: Optional[str] = Field(default=None, description="Full name")
    is_active: bool = Field(default=True, description="Whether user is active")
    is_superuser: bool = Field(default=False, description="Whether user is superuser")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    preferences: Optional[dict] = Field(default_factory=dict, description="User preferences")


class UserCreate(BaseModel):
    """User creation model."""
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    full_name: Optional[str] = Field(default=None, description="Full name")


class UserUpdate(BaseModel):
    """User update model."""
    username: Optional[str] = Field(default=None, description="Username")
    email: Optional[str] = Field(default=None, description="Email address")
    full_name: Optional[str] = Field(default=None, description="Full name")
    is_active: Optional[bool] = Field(default=None, description="Whether user is active")
    preferences: Optional[dict] = Field(default=None, description="User preferences")


class UserInDB(User):
    """User model with password hash."""
    hashed_password: str = Field(..., description="Hashed password")


class Token(BaseModel):
    """Token model."""
    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="bearer", description="Token type")


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = Field(default=None, description="Username")
    scopes: List[str] = Field(default_factory=list, description="Token scopes")
