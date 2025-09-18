"""
Security utilities for the Agentic AI Microservice.

This module provides authentication, authorization, and security utilities
including JWT token handling and password management.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog
from jose import JWTError, jwt
from passlib.context import CryptContext

logger = structlog.get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    data: Dict[str, Any],
    secret_key: str,
    algorithm: str = "HS256",
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        secret_key: Secret key for signing
        algorithm: JWT algorithm to use
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
        logger.info("Access token created", user_id=data.get("sub"))
        return encoded_jwt
    except Exception as e:
        logger.error("Failed to create access token", error=str(e))
        raise


def decode_access_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256"
) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: JWT token to decode
        secret_key: Secret key for verification
        algorithm: JWT algorithm used
        
    Returns:
        Decoded token payload
        
    Raises:
        JWTError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except JWTError as e:
        logger.warning("Invalid token", error=str(e))
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error("Password verification error", error=str(e))
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error("Password hashing error", error=str(e))
        raise


def generate_api_key() -> str:
    """
    Generate a secure API key.
    
    Returns:
        Generated API key
    """
    import secrets
    return secrets.token_urlsafe(32)


def validate_api_key(api_key: str, valid_keys: list) -> bool:
    """
    Validate an API key.
    
    Args:
        api_key: API key to validate
        valid_keys: List of valid API keys
        
    Returns:
        True if valid, False otherwise
    """
    return api_key in valid_keys
