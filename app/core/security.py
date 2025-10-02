"""
Security system for the Agentic AI platform.
"""

import hashlib
import secrets
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import structlog
from functools import wraps
import re
import html
from passlib.context import CryptContext

logger = structlog.get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: str
    severity: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]


class InputSanitizer:
    """Sanitizes and validates user inputs."""
    
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
        r'<form[^>]*>',
        r'<input[^>]*>',
        r'<textarea[^>]*>',
        r'<select[^>]*>',
        r'<option[^>]*>',
        r'<button[^>]*>',
        r'<link[^>]*>',
        r'<meta[^>]*>',
        r'<style[^>]*>',
        r'<base[^>]*>',
        r'<applet[^>]*>',
        r'<param[^>]*>',
    ]
    
    def __init__(self):
        self.dangerous_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
    
    def sanitize_string(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize a string input."""
        if not isinstance(input_str, str):
            return str(input_str)
        
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        sanitized = html.escape(input_str, quote=True)
        
        for pattern in self.dangerous_patterns:
            sanitized = pattern.sub('', sanitized)
        
        return sanitized.strip()
    
    def validate_input(self, input_str: str) -> bool:
        """Validate input against security rules."""
        if not isinstance(input_str, str):
            return False
        
        for pattern in self.dangerous_patterns:
            if pattern.search(input_str):
                return False
        
        return True


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
    
    def is_rate_limited(self, identifier: str, limit: int, window_seconds: int) -> bool:
        """Check if identifier is rate limited."""
        now = time.time()
        window_start = now - window_seconds
        
        requests = self.requests[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        if len(requests) >= limit:
            return True
        
        requests.append(now)
        return False


class SecurityManager:
    """Main security manager."""
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter()
        self.security_events: List[SecurityEvent] = []
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
        
        self.max_failed_attempts = 5
        self.lockout_duration = 3600
        self.rate_limit_requests = 100
        self.rate_limit_window = 3600
        self.max_input_length = 10000
    
    def record_security_event(self, event_type: str, severity: str, user_id: Optional[str], 
                           ip_address: str, user_agent: str, details: Dict[str, Any]):
        """Record a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            details=details
        )
        self.security_events.append(event)
        
        logger.warning(
            f"Security event: {event_type}",
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
    
    def check_security(self, user_id: Optional[str], ip_address: str, user_agent: str) -> bool:
        """Check if request is allowed."""
        if ip_address in self.blocked_ips:
            return False
        
        if user_id and user_id in self.blocked_users:
            return False
        
        if self.rate_limiter.is_rate_limited(f"ip:{ip_address}", self.rate_limit_requests, self.rate_limit_window):
            return False
        
        if user_id and self.rate_limiter.is_rate_limited(f"user:{user_id}", self.rate_limit_requests, self.rate_limit_window):
            return False
        
        return True
    
    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input data."""
        if isinstance(input_data, str):
            return self.input_sanitizer.sanitize_string(input_data, self.max_input_length)
        elif isinstance(input_data, dict):
            return {key: self.sanitize_input(value) for key, value in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        else:
            return input_data
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, str):
            return self.input_sanitizer.validate_input(input_data)
        elif isinstance(input_data, dict):
            return all(self.validate_input(value) for value in input_data.values())
        elif isinstance(input_data, list):
            return all(self.validate_input(item) for item in input_data)
        else:
            return True


# Global security manager instance
security_manager = SecurityManager()


# Security decorators
def require_security_check(func):
    """Decorator for security checks."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request')
        if not request:
            raise ValueError("Request object required for security check")
        
        user_id = getattr(request.state, 'user_id', None)
        ip_address = request.client.host
        user_agent = request.headers.get('user-agent', '')
        
        if not security_manager.check_security(user_id, ip_address, user_agent):
            raise PermissionError("Security check failed")
        
        return await func(*args, **kwargs)
    return wrapper


def sanitize_inputs(func):
    """Decorator for input sanitization."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, (str, dict, list)):
                kwargs[key] = security_manager.sanitize_input(value)
        
        return await func(*args, **kwargs)
    return wrapper


def rate_limit(requests_per_minute: int = 60):
    """Decorator for rate limiting."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                raise ValueError("Request object required for rate limiting")
            
            user_id = getattr(request.state, 'user_id', None)
            ip_address = request.client.host
            
            if user_id:
                identifier = f"user:{user_id}"
            else:
                identifier = f"ip:{ip_address}"
            
            if security_manager.rate_limiter.is_rate_limited(identifier, requests_per_minute, 60):
                raise PermissionError("Rate limit exceeded")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "SecurityEvent", "InputSanitizer", "RateLimiter", "SecurityManager",
    "security_manager", "require_security_check", "sanitize_inputs", "rate_limit",
    "get_password_hash", "verify_password"
]