"""
Single Sign-On (SSO) Authentication Endpoints.

This module provides SSO authentication endpoints that are only active
when SSO is enabled via environment variables. By default, SSO is disabled
and traditional email/password authentication is used.

Environment Variables:
- SSO_ENABLED: Enable/disable SSO (default: False)
- KEYCLOAK_ENABLED: Enable Keycloak SSO (default: False)
- KEYCLOAK_SERVER_URL: Keycloak server URL
- KEYCLOAK_REALM: Keycloak realm name
- KEYCLOAK_CLIENT_ID: OAuth2 client ID
- KEYCLOAK_CLIENT_SECRET: OAuth2 client secret
- KEYCLOAK_REDIRECT_URI: OAuth2 redirect URI
"""

from typing import Optional, Dict, Any
from urllib.parse import urlencode

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.models.auth import (
    UserResponse, TokenResponse, KeycloakConfigDB,
    KeycloakConfigCreate, KeycloakConfigUpdate, KeycloakConfigResponse
)
from app.models.database.base import get_database_session
from app.services.enhanced_auth_service import enhanced_auth_service
from app.config.settings import get_settings
from app.backend_logging.backend_logger import get_logger, LogCategory

logger = structlog.get_logger(__name__)

# Create router - endpoints will only be registered if SSO is enabled
router = APIRouter(prefix="/sso", tags=["Single Sign-On"])

# Get settings to check if SSO is enabled
settings = get_settings()


class SSOLoginRequest(BaseModel):
    """SSO login request model."""
    redirect_url: Optional[str] = None


class SSOCallbackRequest(BaseModel):
    """SSO callback request model."""
    code: str
    state: Optional[str] = None


class SSOStatusResponse(BaseModel):
    """SSO status response model."""
    sso_enabled: bool
    keycloak_enabled: bool
    providers: list[str]


@router.get("/status", response_model=SSOStatusResponse)
async def get_sso_status() -> SSOStatusResponse:
    """
    Get SSO status and available providers.
    
    This endpoint is always available to check if SSO is enabled.
    
    Returns:
        SSO status and available providers
    """
    try:
        providers = []
        if settings.SSO_ENABLED and settings.KEYCLOAK_ENABLED:
            providers.append("keycloak")
        
        get_logger().info(
            "SSO status requested",
            LogCategory.USER_MANAGEMENT,
            "SSOEndpoints",
            data={
                "sso_enabled": settings.SSO_ENABLED,
                "keycloak_enabled": settings.KEYCLOAK_ENABLED,
                "providers": providers
            }
        )
        
        return SSOStatusResponse(
            sso_enabled=settings.SSO_ENABLED,
            keycloak_enabled=settings.KEYCLOAK_ENABLED,
            providers=providers
        )
        
    except Exception as e:
        get_logger().error(
            f"Failed to get SSO status: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "SSOEndpoints",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get SSO status"
        )


# Conditional SSO endpoints - only registered if SSO is enabled
if settings.SSO_ENABLED and settings.KEYCLOAK_ENABLED:
    
    @router.get("/keycloak/login")
    async def keycloak_login(
        redirect_url: Optional[str] = None,
        db: AsyncSession = Depends(get_database_session)
    ) -> RedirectResponse:
        """
        Initiate Keycloak SSO login.
        
        Redirects user to Keycloak login page.
        
        Args:
            redirect_url: URL to redirect to after successful login
            db: Database session
            
        Returns:
            Redirect to Keycloak login page
            
        Raises:
            HTTPException: If SSO is not configured
        """
        try:
            keycloak_config = await enhanced_auth_service.get_keycloak_config(db)
            if not keycloak_config:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Keycloak SSO is not configured"
                )
            
            # Build OAuth2 authorization URL
            auth_params = {
                "client_id": keycloak_config.client_id,
                "redirect_uri": settings.KEYCLOAK_REDIRECT_URI or f"{settings.BASE_URL}/api/v1/sso/keycloak/callback",
                "response_type": "code",
                "scope": "openid profile email",
                "state": redirect_url or "/"  # Store redirect URL in state
            }
            
            auth_url = f"{keycloak_config.server_url}/realms/{keycloak_config.realm}/protocol/openid-connect/auth"
            full_auth_url = f"{auth_url}?{urlencode(auth_params)}"
            
            get_logger().info(
                "Keycloak login initiated",
                LogCategory.USER_MANAGEMENT,
                "SSOEndpoints",
                data={
                    "realm": keycloak_config.realm,
                    "client_id": keycloak_config.client_id,
                    "redirect_url": redirect_url
                }
            )
            
            return RedirectResponse(url=full_auth_url, status_code=status.HTTP_302_FOUND)
            
        except HTTPException:
            raise
        except Exception as e:
            get_logger().error(
                f"Keycloak login failed: {str(e)}",
                LogCategory.ERROR_TRACKING,
                "SSOEndpoints",
                error=e
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initiate SSO login"
            )
    
    
    @router.get("/keycloak/callback")
    async def keycloak_callback(
        code: str,
        state: Optional[str] = None,
        error: Optional[str] = None,
        db: AsyncSession = Depends(get_database_session)
    ) -> Dict[str, Any]:
        """
        Handle Keycloak SSO callback.
        
        Processes the OAuth2 authorization code and creates user session.
        
        Args:
            code: OAuth2 authorization code
            state: State parameter (contains redirect URL)
            error: OAuth2 error parameter
            db: Database session
            
        Returns:
            Authentication tokens and user information
            
        Raises:
            HTTPException: If callback processing fails
        """
        try:
            if error:
                get_logger().warning(
                    f"Keycloak callback error: {error}",
                    LogCategory.SECURITY,
                    "SSOEndpoints",
                    data={"error": error, "state": state}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"SSO authentication failed: {error}"
                )
            
            if not code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing authorization code"
                )
            
            # Exchange authorization code for access token
            access_token = await _exchange_code_for_token(code, db)
            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Failed to exchange authorization code"
                )
            
            # Authenticate user with Keycloak
            user = await enhanced_auth_service.authenticate_with_keycloak(access_token, db)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="SSO authentication failed"
                )
            
            # Generate JWT tokens for the user
            from app.services.auth_service import auth_service
            tokens = await auth_service.create_tokens(user.id)
            
            get_logger().info(
                "Keycloak callback successful",
                LogCategory.USER_MANAGEMENT,
                "SSOEndpoints",
                data={
                    "user_id": user.id,
                    "username": user.username,
                    "redirect_url": state
                }
            )
            
            return {
                "message": "SSO authentication successful",
                "user": user.dict(),
                "tokens": tokens.dict(),
                "redirect_url": state or "/"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            get_logger().error(
                f"Keycloak callback failed: {str(e)}",
                LogCategory.ERROR_TRACKING,
                "SSOEndpoints",
                error=e
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SSO callback processing failed"
            )
    
    
    async def _exchange_code_for_token(code: str, db: AsyncSession) -> Optional[str]:
        """Exchange OAuth2 authorization code for access token."""
        try:
            import httpx
            
            keycloak_config = await enhanced_auth_service.get_keycloak_config(db)
            if not keycloak_config:
                return None
            
            token_url = f"{keycloak_config.server_url}/realms/{keycloak_config.realm}/protocol/openid-connect/token"
            
            token_data = {
                "grant_type": "authorization_code",
                "client_id": keycloak_config.client_id,
                "client_secret": keycloak_config.client_secret,
                "code": code,
                "redirect_uri": settings.KEYCLOAK_REDIRECT_URI or f"{settings.BASE_URL}/api/v1/sso/keycloak/callback"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(token_url, data=token_data)
                
                if response.status_code == 200:
                    token_response = response.json()
                    return token_response.get("access_token")
                else:
                    logger.warning(f"Token exchange failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error exchanging code for token: {str(e)}")
            return None


else:
    # SSO is disabled - provide informational endpoints
    @router.get("/keycloak/login")
    async def keycloak_login_disabled():
        """SSO login endpoint when SSO is disabled."""
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SSO is disabled. Use traditional email/password authentication."
        )
    
    @router.get("/keycloak/callback")
    async def keycloak_callback_disabled():
        """SSO callback endpoint when SSO is disabled."""
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SSO is disabled. Use traditional email/password authentication."
        )


# Admin endpoints for Keycloak configuration (always available for admins)
@router.post("/admin/keycloak/config", response_model=KeycloakConfigResponse)
async def create_keycloak_config(
    config_data: KeycloakConfigCreate,
    db: AsyncSession = Depends(get_database_session),
    # current_user: UserResponse = Depends(require_admin)  # TODO: Add admin requirement
) -> KeycloakConfigResponse:
    """
    Create Keycloak configuration (Admin only).

    Allows administrators to configure Keycloak SSO settings.

    Args:
        config_data: Keycloak configuration data
        db: Database session

    Returns:
        Created Keycloak configuration

    Raises:
        HTTPException: If configuration creation fails
    """
    try:
        # Deactivate existing configurations
        from sqlalchemy import update
        await db.execute(
            update(KeycloakConfigDB).values(is_active=False)
        )

        # Create new configuration
        config = KeycloakConfigDB(
            realm=config_data.realm,
            server_url=config_data.server_url,
            client_id=config_data.client_id,
            client_secret=config_data.client_secret,  # Should be encrypted in production
            is_active=True,
            auto_create_users=config_data.auto_create_users,
            default_user_group=config_data.default_user_group,
            role_mappings=config_data.role_mappings or {}
        )

        db.add(config)
        await db.commit()
        await db.refresh(config)

        get_logger().info(
            "Keycloak configuration created",
            LogCategory.USER_MANAGEMENT,
            "SSOEndpoints",
            data={
                "config_id": str(config.id),
                "realm": config.realm,
                "client_id": config.client_id
            }
        )

        return KeycloakConfigResponse.model_validate(config)

    except Exception as e:
        await db.rollback()
        get_logger().error(
            f"Failed to create Keycloak config: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "SSOEndpoints",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create Keycloak configuration"
        )


@router.get("/admin/keycloak/config", response_model=Optional[KeycloakConfigResponse])
async def get_keycloak_config(
    db: AsyncSession = Depends(get_database_session),
    # current_user: UserResponse = Depends(require_admin)  # TODO: Add admin requirement
) -> Optional[KeycloakConfigResponse]:
    """
    Get current Keycloak configuration (Admin only).

    Returns:
        Current active Keycloak configuration or None
    """
    try:
        config = await enhanced_auth_service.get_keycloak_config(db)
        if config:
            return KeycloakConfigResponse.model_validate(config)
        return None

    except Exception as e:
        get_logger().error(
            f"Failed to get Keycloak config: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "SSOEndpoints",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get Keycloak configuration"
        )
