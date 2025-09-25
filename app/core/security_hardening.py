"""
Security hardening system for the Agentic AI platform.

This module provides advanced security features including authentication,
authorization, encryption, and security monitoring.
"""

import hashlib
import secrets
import time
import jwt
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import structlog
from functools import wraps
import bcrypt
import cryptography.fernet
from cryptography.fernet import Fernet
import base64
import os
import json

logger = structlog.get_logger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    MANAGE_USERS = "manage_users"
    MANAGE_AGENTS = "manage_agents"
    MANAGE_TOOLS = "manage_tools"
    MANAGE_SYSTEM = "manage_system"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_login_attempts: int = 5
    lockout_duration: int = 3600  # 1 hour
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    session_timeout: int = 3600  # 1 hour
    token_expiry: int = 86400  # 24 hours
    enable_2fa: bool = True
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000


@dataclass
class UserSession:
    """User session information."""
    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    permissions: Set[Permission]
    is_active: bool = True
    expires_at: Optional[datetime] = None


class PasswordManager:
    """Secure password management."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        errors = []
        warnings = []
        
        if len(password) < self.policy.password_min_length:
            errors.append(f"Password must be at least {self.policy.password_min_length} characters")
        
        if self.policy.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain uppercase letters")
        
        if self.policy.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain lowercase letters")
        
        if self.policy.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain numbers")
        
        if self.policy.password_require_symbols and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain symbols")
        
        # Check for common patterns
        if password.lower() in ['password', '123456', 'admin', 'user']:
            errors.append("Password is too common")
        
        # Check for repeated characters
        if len(set(password)) < len(password) * 0.6:
            warnings.append("Password has too many repeated characters")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "strength_score": self._calculate_strength_score(password)
        }
    
    def _calculate_strength_score(self, password: str) -> int:
        """Calculate password strength score (0-100)."""
        score = 0
        
        # Length bonus
        if len(password) >= 8:
            score += 10
        if len(password) >= 12:
            score += 10
        if len(password) >= 16:
            score += 10
        
        # Character variety bonus
        if any(c.isupper() for c in password):
            score += 10
        if any(c.islower() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 10
        
        # Complexity bonus
        if len(set(password)) >= len(password) * 0.8:
            score += 10
        
        # Uniqueness bonus
        if not any(common in password.lower() for common in ['password', '123456', 'admin', 'user']):
            score += 10
        
        return min(score, 100)


class EncryptionManager:
    """Encryption and decryption management."""
    
    def __init__(self, secret_key: Optional[str] = None):
        if secret_key:
            self.key = secret_key.encode()
        else:
            # Generate or load key
            key_file = "app/security/encryption.key"
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.key = f.read()
            else:
                self.key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_file), exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(self.key)
        
        self.fernet = Fernet(self.key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.fernet.encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary."""
        json_data = json.dumps(data, default=str)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt a dictionary."""
        decrypted_json = self.decrypt(encrypted_data)
        return json.loads(decrypted_json)


class TokenManager:
    """JWT token management."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, user_id: str, permissions: Set[Permission], 
                    expires_in: int = 3600) -> str:
        """Create a JWT token."""
        payload = {
            "user_id": user_id,
            "permissions": [p.value for p in permissions],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(seconds=expires_in)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def refresh_token(self, token: str, expires_in: int = 3600) -> Optional[str]:
        """Refresh a token."""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        return self.create_token(
            payload["user_id"],
            set(Permission(p) for p in payload["permissions"]),
            expires_in
        )


class SessionManager:
    """User session management."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str,
                      permissions: Set[Permission]) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions,
            expires_at=datetime.utcnow() + timedelta(seconds=self.policy.session_timeout)
        )
        
        self.sessions[session_id] = session
        self.user_sessions[user_id].add(session_id)
        
        logger.info(f"Created session for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get a user session."""
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        # Check if session is expired
        if session.expires_at and datetime.utcnow() > session.expires_at:
            self.destroy_session(session_id)
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        
        return session
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a user session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Remove from user sessions
        if session.user_id in self.user_sessions:
            self.user_sessions[session.user_id].discard(session_id)
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info(f"Destroyed session {session_id}")
        return True
    
    def destroy_user_sessions(self, user_id: str) -> int:
        """Destroy all sessions for a user."""
        if user_id not in self.user_sessions:
            return 0
        
        session_ids = list(self.user_sessions[user_id])
        for session_id in session_ids:
            self.destroy_session(session_id)
        
        return len(session_ids)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.expires_at and now > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.destroy_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class AccessControl:
    """Access control and authorization."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.permission_matrix: Dict[Permission, Set[Permission]] = {
            Permission.ADMIN: {p for p in Permission},
            Permission.MANAGE_SYSTEM: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.EXECUTE, Permission.MANAGE_USERS, Permission.MANAGE_AGENTS, Permission.MANAGE_TOOLS},
            Permission.MANAGE_USERS: {Permission.READ, Permission.WRITE, Permission.DELETE},
            Permission.MANAGE_AGENTS: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.EXECUTE},
            Permission.MANAGE_TOOLS: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.EXECUTE},
            Permission.EXECUTE: {Permission.READ},
            Permission.WRITE: {Permission.READ},
            Permission.DELETE: {Permission.READ},
            Permission.READ: set()
        }
    
    def check_permission(self, user_permissions: Set[Permission], required_permission: Permission) -> bool:
        """Check if user has required permission."""
        for user_permission in user_permissions:
            if user_permission == required_permission:
                return True
            
            # Check if user permission includes required permission
            if required_permission in self.permission_matrix.get(user_permission, set()):
                return True
        
        return False
    
    def check_multiple_permissions(self, user_permissions: Set[Permission], 
                                 required_permissions: List[Permission]) -> bool:
        """Check if user has all required permissions."""
        return all(self.check_permission(user_permissions, perm) for perm in required_permissions)
    
    def get_accessible_resources(self, user_permissions: Set[Permission]) -> Set[str]:
        """Get list of resources user can access."""
        accessible = set()
        
        for permission in user_permissions:
            if permission == Permission.ADMIN:
                accessible.update(["users", "agents", "tools", "system", "logs", "config"])
            elif permission == Permission.MANAGE_SYSTEM:
                accessible.update(["users", "agents", "tools", "system", "logs"])
            elif permission == Permission.MANAGE_USERS:
                accessible.update(["users"])
            elif permission == Permission.MANAGE_AGENTS:
                accessible.update(["agents"])
            elif permission == Permission.MANAGE_TOOLS:
                accessible.update(["tools"])
            elif permission == Permission.READ:
                accessible.update(["public"])
        
        return accessible


class SecurityAuditLogger:
    """Security audit logging."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.audit_logs: List[Dict[str, Any]] = []
    
    def log_security_event(self, event_type: str, user_id: Optional[str], 
                          ip_address: str, details: Dict[str, Any]):
        """Log a security event."""
        if not self.policy.enable_audit_logging:
            return
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details
        }
        
        self.audit_logs.append(log_entry)
        
        # Log to structured logger
        logger.info(
            f"Security event: {event_type}",
            user_id=user_id,
            ip_address=ip_address,
            details=details
        )
    
    def log_authentication(self, user_id: str, ip_address: str, success: bool, details: Dict[str, Any]):
        """Log authentication event."""
        self.log_security_event(
            "authentication",
            user_id,
            ip_address,
            {"success": success, **details}
        )
    
    def log_authorization(self, user_id: str, ip_address: str, resource: str, 
                         action: str, success: bool):
        """Log authorization event."""
        self.log_security_event(
            "authorization",
            user_id,
            ip_address,
            {"resource": resource, "action": action, "success": success}
        )
    
    def log_permission_change(self, user_id: str, target_user: str, 
                             old_permissions: Set[Permission], 
                             new_permissions: Set[Permission]):
        """Log permission change event."""
        self.log_security_event(
            "permission_change",
            user_id,
            None,
            {
                "target_user": target_user,
                "old_permissions": [p.value for p in old_permissions],
                "new_permissions": [p.value for p in new_permissions]
            }
        )
    
    def get_audit_logs(self, user_id: Optional[str] = None, 
                      event_type: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs with optional filtering."""
        logs = self.audit_logs
        
        if user_id:
            logs = [log for log in logs if log.get("user_id") == user_id]
        
        if event_type:
            logs = [log for log in logs if log.get("event_type") == event_type]
        
        return logs[-limit:]


class SecurityHardening:
    """Main security hardening system."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        
        # Initialize components
        self.password_manager = PasswordManager(self.policy)
        self.encryption_manager = EncryptionManager()
        self.token_manager = TokenManager(secrets.token_urlsafe(32))
        self.session_manager = SessionManager(self.policy)
        self.access_control = AccessControl(self.policy)
        self.audit_logger = SecurityAuditLogger(self.policy)
        
        # Security state
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[str]:
        """Authenticate a user."""
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            self.audit_logger.log_authentication(
                username, ip_address, False, {"reason": "IP blocked"}
            )
            return None
        
        # Check if user is blocked
        if username in self.blocked_users:
            self.audit_logger.log_authentication(
                username, ip_address, False, {"reason": "User blocked"}
            )
            return None
        
        # Check rate limiting
        if self._is_rate_limited(ip_address):
            self.audit_logger.log_authentication(
                username, ip_address, False, {"reason": "Rate limited"}
            )
            return None
        
        # TODO: Implement actual user authentication
        # For now, return a mock user ID
        user_id = f"user_{username}"
        
        # Log successful authentication
        self.audit_logger.log_authentication(
            username, ip_address, True, {"user_id": user_id}
        )
        
        # Reset failed attempts
        self.failed_attempts[username] = 0
        
        return user_id
    
    def create_user_session(self, user_id: str, ip_address: str, user_agent: str,
                           permissions: Set[Permission]) -> str:
        """Create a user session."""
        session_id = self.session_manager.create_session(
            user_id, ip_address, user_agent, permissions
        )
        
        return session_id
    
    def check_permission(self, session_id: str, required_permission: Permission) -> bool:
        """Check if session has required permission."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return False
        
        return self.access_control.check_permission(session.permissions, required_permission)
    
    def _is_rate_limited(self, ip_address: str) -> bool:
        """Check if IP is rate limited."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        # Clean old entries
        self.rate_limits[ip_address] = [
            timestamp for timestamp in self.rate_limits[ip_address]
            if timestamp > window_start
        ]
        
        # Check rate limit
        if len(self.rate_limits[ip_address]) >= self.policy.max_requests_per_minute:
            return True
        
        # Record this request
        self.rate_limits[ip_address].append(now)
        return False
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "policy": {
                "max_login_attempts": self.policy.max_login_attempts,
                "lockout_duration": self.policy.lockout_duration,
                "session_timeout": self.policy.session_timeout,
                "enable_2fa": self.policy.enable_2fa,
                "enable_audit_logging": self.policy.enable_audit_logging,
                "enable_encryption": self.policy.enable_encryption,
                "enable_rate_limiting": self.policy.enable_rate_limiting
            },
            "sessions": {
                "active_sessions": len(self.session_manager.sessions),
                "total_users": len(self.session_manager.user_sessions)
            },
            "security": {
                "blocked_ips": len(self.blocked_ips),
                "blocked_users": len(self.blocked_users),
                "failed_attempts": dict(self.failed_attempts)
            },
            "audit_logs": {
                "total_events": len(self.audit_logger.audit_logs),
                "recent_events": self.audit_logger.get_audit_logs(limit=10)
            }
        }


# Security decorators
def require_authentication(func):
    """Decorator for requiring authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract session from request
        request = kwargs.get('request')
        if not request:
            raise ValueError("Request object required for authentication")
        
        session_id = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not session_id:
            raise PermissionError("Authentication required")
        
        # TODO: Implement session validation
        return await func(*args, **kwargs)
    return wrapper


def require_permission(permission: Permission):
    """Decorator for requiring specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract session from request
            request = kwargs.get('request')
            if not request:
                raise ValueError("Request object required for permission check")
            
            session_id = request.headers.get('Authorization', '').replace('Bearer ', '')
            if not session_id:
                raise PermissionError("Authentication required")
            
            # TODO: Implement permission check
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_security_level(level: SecurityLevel):
    """Decorator for requiring specific security level."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # TODO: Implement security level check
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Export all components
__all__ = [
    "SecurityLevel", "Permission", "SecurityPolicy", "UserSession",
    "PasswordManager", "EncryptionManager", "TokenManager", "SessionManager",
    "AccessControl", "SecurityAuditLogger", "SecurityHardening",
    "require_authentication", "require_permission", "require_security_level"
]


