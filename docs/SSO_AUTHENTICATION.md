# Single Sign-On (SSO) Authentication

## Overview

The Agentic AI Microservice supports **optional** Single Sign-On (SSO) authentication via Keycloak. By default, SSO is **disabled** and the system uses traditional email/password authentication with JWT tokens.

## Key Features

- 🔐 **Disabled by Default** - Traditional email/password authentication is the primary method
- ⚙️ **Environment-Controlled** - SSO is enabled only via environment variables
- 🔄 **Dual Authentication** - When SSO is enabled, both traditional and SSO authentication work simultaneously
- 🛡️ **Secure Integration** - OAuth2/OpenID Connect with Keycloak
- 👤 **Auto User Provisioning** - Automatic user creation from SSO providers
- 🎯 **Flexible Configuration** - Database or environment variable configuration

## Authentication Modes

### Default Mode: Traditional Authentication
- Email/password login via `/api/v1/auth/login`
- JWT token-based sessions
- User registration via `/api/v1/auth/register`
- Password reset functionality
- Full user management capabilities

### SSO Mode: Keycloak Integration
- OAuth2/OpenID Connect authentication
- Automatic user provisioning
- Role mapping from Keycloak to local groups
- Traditional authentication remains available
- Seamless user experience

## Environment Configuration

### Enabling SSO

To enable SSO, set these environment variables:

```bash
# Enable SSO authentication
AGENTIC_SSO_ENABLED=true

# Enable Keycloak SSO integration
AGENTIC_KEYCLOAK_ENABLED=true

# Keycloak server configuration
AGENTIC_KEYCLOAK_SERVER_URL="https://your-keycloak-server.com"
AGENTIC_KEYCLOAK_REALM="your-realm"
AGENTIC_KEYCLOAK_CLIENT_ID="agentic-ai-client"
AGENTIC_KEYCLOAK_CLIENT_SECRET="your-client-secret"
AGENTIC_KEYCLOAK_REDIRECT_URI="https://your-domain.com/api/v1/sso/keycloak/callback"

# User management settings
AGENTIC_KEYCLOAK_AUTO_CREATE_USERS=true
AGENTIC_KEYCLOAK_DEFAULT_USER_GROUP="user"
```

### Disabling SSO (Default)

```bash
# SSO is disabled by default
AGENTIC_SSO_ENABLED=false
AGENTIC_KEYCLOAK_ENABLED=false
```

## API Endpoints

### SSO Status (Always Available)

```http
GET /api/v1/sso/status
```

Returns SSO availability and configuration status.

**Response:**
```json
{
  "sso_enabled": false,
  "keycloak_enabled": false,
  "providers": []
}
```

### SSO Authentication (When Enabled)

#### Initiate SSO Login
```http
GET /api/v1/sso/keycloak/login?redirect_url=/dashboard
```

Redirects to Keycloak login page.

#### SSO Callback
```http
GET /api/v1/sso/keycloak/callback?code=AUTH_CODE&state=REDIRECT_URL
```

Handles OAuth2 callback and returns JWT tokens.

**Response:**
```json
{
  "message": "SSO authentication successful",
  "user": {
    "id": "user-uuid",
    "username": "john.doe",
    "email": "john@example.com",
    "full_name": "John Doe"
  },
  "tokens": {
    "access_token": "jwt-access-token",
    "refresh_token": "jwt-refresh-token",
    "token_type": "bearer"
  },
  "redirect_url": "/dashboard"
}
```

### Admin Configuration (Always Available)

#### Create Keycloak Configuration
```http
POST /api/v1/sso/admin/keycloak/config
```

**Request Body:**
```json
{
  "realm": "agentic-ai",
  "server_url": "https://keycloak.example.com",
  "client_id": "agentic-ai-client",
  "client_secret": "client-secret",
  "auto_create_users": true,
  "default_user_group": "user",
  "role_mappings": {
    "admin": "admin",
    "moderator": "moderator",
    "user": "user"
  }
}
```

#### Get Keycloak Configuration
```http
GET /api/v1/sso/admin/keycloak/config
```

## Keycloak Setup

### 1. Create Realm
1. Login to Keycloak Admin Console
2. Create a new realm (e.g., "agentic-ai")
3. Configure realm settings

### 2. Create Client
1. Go to Clients → Create
2. Set Client ID: `agentic-ai-client`
3. Set Client Protocol: `openid-connect`
4. Set Access Type: `confidential`
5. Set Valid Redirect URIs: `https://your-domain.com/api/v1/sso/keycloak/callback`
6. Save and note the Client Secret

### 3. Configure Roles (Optional)
1. Create roles: `admin`, `moderator`, `user`
2. Assign roles to users
3. Configure role mappings in the application

### 4. Test Configuration
1. Create a test user
2. Assign appropriate roles
3. Test SSO login flow

## User Experience

### With SSO Disabled (Default)
- Users see traditional login form
- Email/password authentication
- Standard registration process
- Password reset functionality

### With SSO Enabled
- Users see both login options:
  - Traditional email/password
  - "Login with SSO" button
- SSO users are automatically created
- Existing users can link SSO accounts
- Seamless authentication experience

## Security Considerations

### Token Management
- JWT tokens are issued for both traditional and SSO users
- Same token validation and refresh mechanisms
- Consistent session management

### User Provisioning
- SSO users are created with verified email status
- Default user group assignment
- Role mapping from Keycloak roles
- No password stored for SSO-only users

### Configuration Security
- Client secrets should be encrypted in production
- Use environment variables for sensitive data
- Secure Keycloak server communication (HTTPS)
- Validate redirect URIs

## Troubleshooting

### SSO Not Working
1. Check environment variables are set correctly
2. Verify Keycloak server is accessible
3. Confirm client configuration in Keycloak
4. Check redirect URI matches exactly
5. Review application logs for errors

### User Creation Issues
1. Verify `KEYCLOAK_AUTO_CREATE_USERS=true`
2. Check default user group exists
3. Review role mapping configuration
4. Ensure email is provided by Keycloak

### Token Exchange Failures
1. Verify client secret is correct
2. Check authorization code is valid
3. Confirm redirect URI matches
4. Review Keycloak logs

## Migration Guide

### Enabling SSO on Existing System
1. Set environment variables
2. Restart application
3. Configure Keycloak client
4. Test SSO login flow
5. Existing users continue using traditional auth

### Disabling SSO
1. Set `AGENTIC_SSO_ENABLED=false`
2. Restart application
3. SSO endpoints return 503 Service Unavailable
4. Traditional authentication remains functional

## Development

### Local Development with Keycloak
```bash
# Run Keycloak in Docker
docker run -p 8080:8080 -e KEYCLOAK_ADMIN=admin -e KEYCLOAK_ADMIN_PASSWORD=admin quay.io/keycloak/keycloak:latest start-dev

# Configure environment
AGENTIC_SSO_ENABLED=true
AGENTIC_KEYCLOAK_ENABLED=true
AGENTIC_KEYCLOAK_SERVER_URL="http://localhost:8080"
AGENTIC_KEYCLOAK_REALM="master"
AGENTIC_KEYCLOAK_CLIENT_ID="agentic-ai-client"
AGENTIC_KEYCLOAK_CLIENT_SECRET="your-client-secret"
AGENTIC_BASE_URL="http://localhost:8888"
```

### Testing SSO Integration
1. Start Keycloak server
2. Configure client and test user
3. Set environment variables
4. Start application
5. Test SSO login flow
6. Verify user creation and token generation
