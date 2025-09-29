"""
Authentication Service.

This module provides authentication and user management services
with proper password hashing, JWT token management, and session handling.
"""

import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
from uuid import uuid4

import bcrypt
import jwt
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from app.config.settings import get_settings
from app.models.auth import UserDB, UserCreate, UserLogin, TokenResponse, UserResponse
from app.models.enhanced_user import UserSession
from app.models.database.base import get_database_session

logger = structlog.get_logger(__name__)


class AuthService:
    """Authentication service for user management and security."""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 30
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15

    async def is_first_time_setup(self) -> bool:
        """
        Check if this is the first-time setup (no users exist in the system).

        Returns:
            True if no users exist, False otherwise
        """
        async for session in get_database_session():
            try:
                # Count total users in the system
                result = await session.execute(
                    select(func.count(UserDB.id))
                )
                user_count = result.scalar()
                logger.info("First-time setup check", user_count=user_count)
                return user_count == 0
            except Exception as e:
                logger.error("Error checking first-time setup status", error=str(e))
                # If users table doesn't exist or there's a DB error, it's likely first-time setup
                logger.info("Assuming first-time setup due to database error")
                return True

    async def register_user(self, user_data: UserCreate) -> Tuple[UserResponse, TokenResponse]:
        """
        Register a new user with proper validation and security.

        If this is the first user in the system, they automatically become an admin.

        Args:
            user_data: User registration data

        Returns:
            Tuple of user response and authentication tokens

        Raises:
            ValueError: If user already exists or validation fails
        """
        async for session in get_database_session():
            try:
                # Check if this is first-time setup
                is_first_user = await self.is_first_time_setup()

                # Check if user already exists
                existing_user = await self._get_user_by_username_or_email(
                    session, user_data.username, user_data.email
                )
                if existing_user:
                    if existing_user.username == user_data.username:
                        raise ValueError("Username already exists")
                    else:
                        raise ValueError("Email already exists")

                # Hash password
                password_salt = secrets.token_hex(16)
                hashed_password = self._hash_password(user_data.password, password_salt)

                # Create user - first user becomes admin
                user = UserDB(
                    username=user_data.username,
                    email=user_data.email,
                    name=user_data.name,  # Add the name field
                    hashed_password=hashed_password,
                    password_salt=password_salt,  # Add the password salt
                    is_active=True,
                    user_group='admin' if is_first_user else 'user'  # First user becomes admin
                )
                
                session.add(user)
                await session.commit()
                await session.refresh(user)

                if is_first_user:
                    logger.info(
                        "First admin user created successfully",
                        user_id=str(user.id),
                        username=user.username,
                        is_admin=True,
                        first_time_setup=True
                    )
                else:
                    logger.info("User registered successfully", user_id=str(user.id), username=user.username)
                
                # Refresh user object to ensure all attributes are loaded
                await session.refresh(user)

                # Create authentication tokens
                tokens = await self._create_user_session(session, user)
                # Create UserResponse manually to avoid async attribute issues
                user_response = UserResponse(
                    id=user.id,
                    username=user.username,
                    email=user.email,
                    name=user.name,
                    is_active=user.is_active,
                    user_group=user.user_group,
                    api_keys=user.api_keys,
                    created_at=user.created_at,
                    updated_at=user.updated_at
                )

                return user_response, tokens
                
            except Exception as e:
                await session.rollback()
                logger.error("User registration failed", error=str(e))
                raise

    async def authenticate_user(self, login_data: UserLogin, ip_address: str = None, user_agent: str = None) -> TokenResponse:
        """
        Authenticate user with credentials and create session.
        
        Args:
            login_data: Login credentials
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Authentication tokens
            
        Raises:
            ValueError: If authentication fails
        """
        async for session in get_database_session():
            try:
                # Get user by username or email
                user = await self._get_user_by_username_or_email(
                    session, login_data.username_or_email, login_data.username_or_email
                )
                
                if not user:
                    raise ValueError("Invalid credentials")
                
                # Check if account is locked
                if user.locked_until and user.locked_until > datetime.now(timezone.utc):
                    raise ValueError("Account is temporarily locked due to too many failed attempts")
                
                # Check if account is active
                if not user.is_active:
                    raise ValueError("Account is disabled")
                
                # Verify password
                if not self._verify_password(login_data.password, user.hashed_password, user.password_salt):
                    # Increment failed attempts
                    await self._handle_failed_login(session, user)
                    raise ValueError("Invalid credentials")
                
                # Reset failed attempts on successful login
                if user.failed_login_attempts > 0:
                    user.failed_login_attempts = 0
                    user.locked_until = None
                
                # Update login tracking
                user.last_login = datetime.now(timezone.utc)
                user.login_count += 1
                
                await session.commit()
                
                # Create session
                tokens = await self._create_user_session(session, user, ip_address, user_agent)
                
                logger.info("User authenticated successfully", user_id=str(user.id), username=user.username)
                
                return tokens
                
            except Exception as e:
                await session.rollback()
                logger.error("User authentication failed", error=str(e))
                raise

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New authentication tokens
            
        Raises:
            ValueError: If refresh token is invalid
        """
        async for session in get_database_session():
            try:
                # Find session by refresh token
                stmt = select(UserSession).options(selectinload(UserSession.user)).where(
                    and_(
                        UserSession.refresh_token == refresh_token,
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.now(timezone.utc)
                    )
                )
                result = await session.execute(stmt)
                user_session = result.scalar_one_or_none()
                
                if not user_session:
                    raise ValueError("Invalid or expired refresh token")
                
                # Check if user is still active
                if not user_session.user.is_active:
                    raise ValueError("User account is disabled")
                
                # Create new tokens
                tokens = await self._create_user_session(session, user_session.user)
                
                # Deactivate old session
                user_session.is_active = False
                await session.commit()
                
                logger.info("Token refreshed successfully", user_id=str(user_session.user.id))
                
                return tokens
                
            except Exception as e:
                await session.rollback()
                logger.error("Token refresh failed", error=str(e))
                raise

    async def logout_user(self, access_token: str) -> bool:
        """
        Logout user by deactivating session.
        
        Args:
            access_token: User's access token
            
        Returns:
            True if logout successful
        """
        async for session in get_database_session():
            try:
                # Decode token to get session info
                payload = jwt.decode(access_token, self.secret_key, algorithms=[self.algorithm])
                session_id = payload.get("session_id")
                
                if session_id:
                    # Deactivate session
                    stmt = update(UserSession).where(
                        UserSession.id == session_id
                    ).values(is_active=False)
                    
                    await session.execute(stmt)
                    await session.commit()
                    
                    logger.info("User logged out successfully", session_id=session_id)
                
                return True
                
            except Exception as e:
                logger.error("Logout failed", error=str(e))
                return False

    async def get_current_user(self, access_token: str) -> Optional[UserResponse]:
        """
        Get current user from access token.
        
        Args:
            access_token: User's access token
            
        Returns:
            User information if token is valid
        """
        try:
            # Decode token
            payload = jwt.decode(access_token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            
            if not user_id or not session_id:
                return None
            
            async for session in get_database_session():
                # Verify session is still active
                stmt = select(UserSession).options(selectinload(UserSession.user)).where(
                    and_(
                        UserSession.id == session_id,
                        UserSession.user_id == user_id,
                        UserSession.is_active == True,
                        UserSession.expires_at > datetime.now(timezone.utc)
                    )
                )
                result = await session.execute(stmt)
                user_session = result.scalar_one_or_none()
                
                if not user_session or not user_session.user.is_active:
                    return None
                
                # Update last activity
                user_session.last_activity = datetime.now(timezone.utc)
                await session.commit()
                
                # Create UserResponse manually to avoid async attribute issues
                user = user_session.user
                return UserResponse(
                    id=user.id,
                    username=user.username,
                    email=user.email,
                    name=user.name,
                    is_active=user.is_active,
                    user_group=user.user_group,
                    api_keys=user.api_keys,
                    created_at=user.created_at,
                    updated_at=user.updated_at
                )
                
        except jwt.ExpiredSignatureError:
            logger.warning("Access token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid access token")
            return None
        except Exception as e:
            logger.error("Error getting current user", error=str(e))
            return None

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using bcrypt."""
        # Combine password and salt
        password_with_salt = f"{password}{salt}"
        # Hash with bcrypt
        hashed = bcrypt.hashpw(password_with_salt.encode('utf-8'), bcrypt.gensalt())
        return hashed.decode('utf-8')

    def _verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against hash."""
        password_with_salt = f"{password}{salt}"
        return bcrypt.checkpw(password_with_salt.encode('utf-8'), hashed_password.encode('utf-8'))

    async def _get_user_by_username_or_email(self, session: AsyncSession, username: str, email: str) -> Optional[UserDB]:
        """Get user by username or email."""
        stmt = select(UserDB).where(
            or_(UserDB.username == username, UserDB.email == email)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _handle_failed_login(self, session: AsyncSession, user: UserDB):
        """Handle failed login attempt."""
        user.failed_login_attempts += 1
        
        if user.failed_login_attempts >= self.max_failed_attempts:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=self.lockout_duration_minutes)
            logger.warning("User account locked due to failed attempts", user_id=str(user.id))
        
        await session.commit()

    async def _create_user_session(self, session: AsyncSession, user: UserDB, ip_address: str = None, user_agent: str = None) -> TokenResponse:
        """Create user session and tokens."""
        # Create session
        session_id = uuid4()
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        user_session = UserSession(
            id=session_id,
            user_id=user.id,
            session_token=secrets.token_urlsafe(32),
            refresh_token=secrets.token_urlsafe(32),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            device_info={}
        )
        
        session.add(user_session)
        await session.commit()

        # Refresh user object to ensure all attributes are loaded
        await session.refresh(user)

        # Create JWT tokens
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self._create_access_token(
            data={"user_id": str(user.id), "session_id": str(session_id)},
            expires_delta=access_token_expires
        )

        # Create UserResponse manually to avoid async attribute issues
        user_response = UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            user_group=user.user_group,
            api_keys=user.api_keys,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=user_session.refresh_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            user=user_response
        )

    def _create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt


# Global auth service instance
auth_service = AuthService()
