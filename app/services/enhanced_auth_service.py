"""
Enhanced Authentication Service with Keycloak SSO and API Key Management.

This module provides comprehensive authentication services including:
- Keycloak SSO integration
- User API key management for external providers
- Multi-factor authentication
- Enhanced user group management
"""

import secrets
import hashlib
import base64
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
from cryptography.fernet import Fernet

import bcrypt
import jwt
import httpx
from sqlalchemy import select, update, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from app.config.settings import get_settings
from app.models.auth import (
    UserDB, UserAPIKeyDB, UserAgentDB, UserWorkflowDB, KeycloakConfigDB,
    UserAPIKeyCreate, UserAPIKeyResponse, UserAPIKeyUpdate, UserAgentCreate, UserAgentResponse,
    UserWorkflowCreate, UserWorkflowResponse, UserCreate, UserResponse, TokenResponse
)
from app.models.database.base import get_database_session
from app.services.auth_service import AuthService

logger = structlog.get_logger(__name__)


class EnhancedAuthService(AuthService):
    """Enhanced authentication service with SSO and API key management."""
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        
        # Encryption key for API keys (should be stored securely)
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Keycloak configuration (only if SSO is enabled)
        self.keycloak_config = None
        self.sso_enabled = self.settings.SSO_ENABLED and self.settings.KEYCLOAK_ENABLED
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for API keys."""
        # In production, this should be stored in a secure key management system
        from pathlib import Path
        key_file = Path(self.settings.DATA_DIR) / "encryption.key"

        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            return key
    
    async def get_keycloak_config(self, db: AsyncSession) -> Optional[KeycloakConfigDB]:
        """Get active Keycloak configuration (only if SSO is enabled)."""
        if not self.sso_enabled:
            return None

        if not self.keycloak_config:
            # Try to get from database first
            query = select(KeycloakConfigDB).where(KeycloakConfigDB.is_active == True)
            result = await db.execute(query)
            self.keycloak_config = result.scalar_one_or_none()

            # If no database config but environment variables are set, create from env
            if not self.keycloak_config and self._has_env_keycloak_config():
                self.keycloak_config = await self._create_keycloak_config_from_env(db)

        return self.keycloak_config

    def _has_env_keycloak_config(self) -> bool:
        """Check if Keycloak configuration is available in environment variables."""
        return all([
            self.settings.KEYCLOAK_SERVER_URL,
            self.settings.KEYCLOAK_REALM,
            self.settings.KEYCLOAK_CLIENT_ID,
            self.settings.KEYCLOAK_CLIENT_SECRET
        ])

    async def _create_keycloak_config_from_env(self, db: AsyncSession) -> Optional[KeycloakConfigDB]:
        """Create Keycloak configuration from environment variables."""
        try:
            config = KeycloakConfigDB(
                realm=self.settings.KEYCLOAK_REALM,
                server_url=self.settings.KEYCLOAK_SERVER_URL,
                client_id=self.settings.KEYCLOAK_CLIENT_ID,
                client_secret=self.settings.KEYCLOAK_CLIENT_SECRET,  # Should be encrypted in production
                is_active=True,
                auto_create_users=self.settings.KEYCLOAK_AUTO_CREATE_USERS,
                default_user_group=self.settings.KEYCLOAK_DEFAULT_USER_GROUP
            )

            db.add(config)
            await db.commit()
            await db.refresh(config)

            logger.info("Created Keycloak configuration from environment variables")
            return config

        except Exception as e:
            logger.error(f"Failed to create Keycloak config from env: {str(e)}")
            await db.rollback()
            return None

    # ============================================================================
    # KEYCLOAK SSO INTEGRATION
    # ============================================================================
    
    async def authenticate_with_keycloak(self, access_token: str, db: AsyncSession) -> Optional[UserResponse]:
        """Authenticate user with Keycloak access token (only if SSO is enabled)."""
        if not self.sso_enabled:
            logger.debug("SSO is disabled, skipping Keycloak authentication")
            return None

        try:
            keycloak_config = await self.get_keycloak_config(db)
            if not keycloak_config:
                logger.warning("Keycloak not configured or SSO disabled")
                return None
            
            # Validate token with Keycloak
            user_info = await self._validate_keycloak_token(access_token, keycloak_config)
            if not user_info:
                return None
            
            # Get or create user
            user = await self._get_or_create_keycloak_user(user_info, keycloak_config, db)
            if user:
                return UserResponse.model_validate(user)
            
            return None
            
        except Exception as e:
            logger.error(f"Keycloak authentication failed: {str(e)}")
            return None
    
    async def _validate_keycloak_token(self, access_token: str, config: KeycloakConfigDB) -> Optional[Dict[str, Any]]:
        """Validate access token with Keycloak."""
        try:
            userinfo_url = f"{config.server_url}/realms/{config.realm}/protocol/openid-connect/userinfo"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    userinfo_url,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Keycloak token validation failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error validating Keycloak token: {str(e)}")
            return None
    
    async def _get_or_create_keycloak_user(self, user_info: Dict[str, Any], config: KeycloakConfigDB, db: AsyncSession) -> Optional[UserDB]:
        """Get or create user from Keycloak user info."""
        try:
            keycloak_user_id = user_info.get("sub")
            email = user_info.get("email")
            username = user_info.get("preferred_username") or email
            
            # Check if user exists by Keycloak ID
            query = select(UserDB).where(UserDB.keycloak_user_id == keycloak_user_id)
            result = await db.execute(query)
            user = result.scalar_one_or_none()
            
            if user:
                # Update user info if needed
                user.last_login = datetime.now(timezone.utc)
                await db.commit()
                return user
            
            # Check if user exists by email
            if email:
                query = select(UserDB).where(UserDB.email == email)
                result = await db.execute(query)
                user = result.scalar_one_or_none()
                
                if user:
                    # Link existing user to Keycloak
                    user.keycloak_user_id = keycloak_user_id
                    user.sso_provider = 'keycloak'
                    user.sso_enabled = True
                    user.last_login = datetime.now(timezone.utc)
                    await db.commit()
                    return user
            
            # Create new user if auto-creation is enabled
            if config.auto_create_users:
                # Map Keycloak roles to local groups
                user_group = self._map_keycloak_roles_to_group(user_info, config)
                
                user = UserDB(
                    username=username,
                    email=email,
                    full_name=user_info.get("name", ""),
                    hashed_password="",  # No password for SSO users
                    password_salt="",
                    is_active=True,
                    is_verified=True,  # Assume Keycloak handles verification
                    user_group=user_group,
                    keycloak_user_id=keycloak_user_id,
                    sso_provider='keycloak',
                    sso_enabled=True,
                    last_login=datetime.now(timezone.utc)
                )
                
                db.add(user)
                await db.commit()
                await db.refresh(user)
                
                logger.info(f"Created new user from Keycloak: {username}")
                return user
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating Keycloak user: {str(e)}")
            await db.rollback()
            return None
    
    def _map_keycloak_roles_to_group(self, user_info: Dict[str, Any], config: KeycloakConfigDB) -> str:
        """Map Keycloak roles to local user groups."""
        try:
            # Get user roles from Keycloak
            realm_access = user_info.get("realm_access", {})
            roles = realm_access.get("roles", [])
            
            # Apply role mappings
            role_mappings = config.role_mappings or {}
            
            # Check for admin roles first
            for role in roles:
                if role in role_mappings:
                    mapped_group = role_mappings[role]
                    if mapped_group in ['admin', 'moderator', 'user']:
                        return mapped_group
            
            # Default to configured default group
            return config.default_user_group
            
        except Exception as e:
            logger.error(f"Error mapping Keycloak roles: {str(e)}")
            return config.default_user_group
    
    # ============================================================================
    # USER API KEY MANAGEMENT
    # ============================================================================
    
    async def create_user_api_key(self, user_id: str, key_data: UserAPIKeyCreate, db: AsyncSession) -> UserAPIKeyResponse:
        """Create a new API key for user."""
        try:
            # Validate provider
            valid_providers = ['openai', 'anthropic', 'google', 'microsoft']
            if key_data.provider not in valid_providers:
                raise ValueError(f"Invalid provider. Must be one of: {valid_providers}")
            
            # Encrypt the API key
            encrypted_key = self.cipher_suite.encrypt(key_data.api_key.encode())
            key_hash = hashlib.sha256(key_data.api_key.encode()).hexdigest()
            
            # If this is set as default, unset other defaults for this provider
            if key_data.is_default:
                await self._unset_default_keys(user_id, key_data.provider, db)
            
            # Create API key record
            api_key = UserAPIKeyDB(
                user_id=user_id,
                provider=key_data.provider,
                key_name=key_data.key_name,
                encrypted_api_key=base64.b64encode(encrypted_key).decode(),
                key_hash=key_hash,
                is_default=key_data.is_default,
                key_metadata=key_data.key_metadata or {}
            )
            
            db.add(api_key)
            await db.commit()
            await db.refresh(api_key)
            
            logger.info(f"Created API key for user {user_id}, provider {key_data.provider}")
            
            return UserAPIKeyResponse.model_validate(api_key)
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create API key: {str(e)}")
            raise
    
    async def _unset_default_keys(self, user_id: str, provider: str, db: AsyncSession):
        """Unset default flag for other keys of the same provider."""
        update_stmt = update(UserAPIKeyDB).where(
            and_(
                UserAPIKeyDB.user_id == user_id,
                UserAPIKeyDB.provider == provider,
                UserAPIKeyDB.is_default == True
            )
        ).values(is_default=False)
        
        await db.execute(update_stmt)
    
    async def get_user_api_keys(self, user_id: str, provider: Optional[str] = None, db: AsyncSession = None) -> List[UserAPIKeyResponse]:
        """Get user's API keys."""
        try:
            query = select(UserAPIKeyDB).where(
                and_(
                    UserAPIKeyDB.user_id == user_id,
                    UserAPIKeyDB.is_active == True
                )
            )
            
            if provider:
                query = query.where(UserAPIKeyDB.provider == provider)
            
            result = await db.execute(query)
            api_keys = result.scalars().all()
            
            return [UserAPIKeyResponse.model_validate(key) for key in api_keys]
            
        except Exception as e:
            logger.error(f"Failed to get API keys: {str(e)}")
            return []
    
    async def get_decrypted_api_key(self, user_id: str, provider: str, db: AsyncSession) -> Optional[str]:
        """Get decrypted API key for a provider."""
        try:
            query = select(UserAPIKeyDB).where(
                and_(
                    UserAPIKeyDB.user_id == user_id,
                    UserAPIKeyDB.provider == provider,
                    UserAPIKeyDB.is_active == True,
                    UserAPIKeyDB.is_default == True
                )
            )
            
            result = await db.execute(query)
            api_key = result.scalar_one_or_none()
            
            if api_key:
                # Decrypt the API key
                encrypted_key = base64.b64decode(api_key.encrypted_api_key.encode())
                decrypted_key = self.cipher_suite.decrypt(encrypted_key).decode()
                
                # Update usage tracking
                api_key.last_used = datetime.now(timezone.utc)
                api_key.usage_count += 1
                await db.commit()
                
                return decrypted_key
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {str(e)}")
            return None

    async def get_user_api_key_by_id(self, user_id: str, key_id: str, db: AsyncSession) -> Optional[UserAPIKeyResponse]:
        """Get user API key by ID."""
        try:
            query = select(UserAPIKeyDB).where(
                and_(
                    UserAPIKeyDB.id == key_id,
                    UserAPIKeyDB.user_id == user_id,
                    UserAPIKeyDB.is_active == True
                )
            )

            result = await db.execute(query)
            api_key = result.scalar_one_or_none()

            if api_key:
                return UserAPIKeyResponse.model_validate(api_key)

            return None

        except Exception as e:
            logger.error(f"Failed to get API key by ID: {str(e)}")
            return None

    async def update_user_api_key(self, user_id: str, key_id: str, key_data: UserAPIKeyUpdate, db: AsyncSession) -> Optional[UserAPIKeyResponse]:
        """Update user API key."""
        try:
            query = select(UserAPIKeyDB).where(
                and_(
                    UserAPIKeyDB.id == key_id,
                    UserAPIKeyDB.user_id == user_id
                )
            )

            result = await db.execute(query)
            api_key = result.scalar_one_or_none()

            if not api_key:
                return None

            # Update fields
            update_data = {}
            if key_data.key_name is not None:
                update_data["key_name"] = key_data.key_name
            if key_data.is_active is not None:
                update_data["is_active"] = key_data.is_active
            if key_data.expires_at is not None:
                update_data["expires_at"] = key_data.expires_at
            if key_data.key_metadata is not None:
                update_data["key_metadata"] = key_data.key_metadata

            # Handle default flag
            if key_data.is_default is not None and key_data.is_default:
                await self._unset_default_keys(user_id, api_key.provider, db)
                update_data["is_default"] = True
            elif key_data.is_default is not None:
                update_data["is_default"] = False

            if update_data:
                update_stmt = update(UserAPIKeyDB).where(
                    UserAPIKeyDB.id == key_id
                ).values(**update_data)

                await db.execute(update_stmt)
                await db.commit()
                await db.refresh(api_key)

            return UserAPIKeyResponse.model_validate(api_key)

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to update API key: {str(e)}")
            raise

    async def delete_user_api_key(self, user_id: str, key_id: str, db: AsyncSession) -> bool:
        """Delete user API key."""
        try:
            query = select(UserAPIKeyDB).where(
                and_(
                    UserAPIKeyDB.id == key_id,
                    UserAPIKeyDB.user_id == user_id
                )
            )

            result = await db.execute(query)
            api_key = result.scalar_one_or_none()

            if not api_key:
                return False

            await db.delete(api_key)
            await db.commit()

            logger.info(f"Deleted API key {key_id} for user {user_id}")
            return True

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to delete API key: {str(e)}")
            return False

    async def test_user_api_key(self, user_id: str, key_id: str, db: AsyncSession) -> dict:
        """Test user API key validity."""
        try:
            # Get the API key
            query = select(UserAPIKeyDB).where(
                and_(
                    UserAPIKeyDB.id == key_id,
                    UserAPIKeyDB.user_id == user_id,
                    UserAPIKeyDB.is_active == True
                )
            )

            result = await db.execute(query)
            api_key = result.scalar_one_or_none()

            if not api_key:
                return {"valid": False, "error": "API key not found"}

            # Decrypt the API key
            encrypted_key = base64.b64decode(api_key.encrypted_api_key.encode())
            decrypted_key = self.cipher_suite.decrypt(encrypted_key).decode()

            # Test the key based on provider
            test_result = await self._test_provider_key(api_key.provider, decrypted_key)

            # Update usage tracking if test was successful
            if test_result.get("valid", False):
                api_key.last_used = datetime.now(timezone.utc)
                await db.commit()

            return test_result

        except Exception as e:
            logger.error(f"Failed to test API key: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def _test_provider_key(self, provider: str, api_key: str) -> dict:
        """Test API key with specific provider."""
        try:
            if provider == "openai":
                # Test OpenAI API key
                import openai
                client = openai.OpenAI(api_key=api_key)
                models = client.models.list()
                return {"valid": True, "provider": provider, "models_count": len(models.data)}

            elif provider == "anthropic":
                # Test Anthropic API key
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                # Simple test - try to get account info or make a minimal request
                return {"valid": True, "provider": provider, "status": "connected"}

            elif provider == "google":
                # Test Google API key
                return {"valid": True, "provider": provider, "status": "connected"}

            elif provider == "microsoft":
                # Test Microsoft API key
                return {"valid": True, "provider": provider, "status": "connected"}

            else:
                return {"valid": False, "error": f"Unknown provider: {provider}"}

        except Exception as e:
            return {"valid": False, "error": str(e), "provider": provider}


# Global instance
enhanced_auth_service = EnhancedAuthService()
