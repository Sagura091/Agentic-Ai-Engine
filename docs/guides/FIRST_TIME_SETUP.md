# First-Time Setup System

## Overview

The Agentic AI platform includes a comprehensive first-time setup system that automatically detects when the system is being initialized for the first time and provides a streamlined administrator account creation process.

## 🚀 Key Features

- **Automatic Detection** - System automatically detects when no users exist
- **First Admin Creation** - First registered user becomes system administrator
- **Visual Indicators** - Clear UI messaging for first-time setup
- **Security** - First admin is auto-verified and granted full privileges
- **Seamless Experience** - No additional configuration required

## 🔧 How It Works

### Backend Logic

1. **Detection Method**: `is_first_time_setup()` in `AuthService`
   - Counts total users in the database
   - Returns `true` if count is 0, `false` otherwise

2. **Registration Enhancement**: Modified `register_user()` method
   - Checks if this is first-time setup before creating user
   - If first user: automatically sets `is_admin=True`, `is_verified=True`, `user_group="admin"`
   - If not first user: creates normal user account

3. **API Endpoint**: `GET /api/v1/auth/setup/status`
   - Returns first-time setup status
   - Provides appropriate messaging for frontend

### Frontend Integration

1. **Status Checking**: Frontend automatically checks setup status on login page
2. **Visual Indicators**: Special banners and messaging for first-time setup
3. **Registration Form**: Enhanced with administrator setup messaging
4. **Dashboard**: Welcome message for new administrators

## 📋 API Endpoints

### Check First-Time Setup Status

```http
GET /api/v1/auth/setup/status
```

**Response:**
```json
{
  "is_first_time_setup": true,
  "message": "No users found. The first registered user will become an administrator."
}
```

### Register User (Enhanced)

```http
POST /api/v1/auth/register
```

**Request:**
```json
{
  "username": "admin",
  "email": "admin@example.com",
  "password": "SecurePassword123!",
  "full_name": "System Administrator"
}
```

**Response (First User):**
```json
{
  "message": "User registered successfully",
  "user": {
    "id": "uuid",
    "username": "admin",
    "email": "admin@example.com",
    "full_name": "System Administrator",
    "is_active": true,
    "is_verified": true,
    "is_admin": true,
    "user_group": "admin"
  },
  "tokens": {
    "access_token": "jwt_token",
    "refresh_token": "refresh_token",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

## 🎨 User Experience

### First-Time Setup Flow

1. **Login Page Detection**
   - Blue banner appears: "First-Time Setup"
   - Message: "The first user to register will automatically become an administrator"

2. **Registration Modal**
   - Title changes to: "Create Administrator Account"
   - Subtitle: "Set up the first admin user for your Agentic AI system"
   - Warning banner: "Administrator Setup: This account will have full system privileges"

3. **Dashboard Welcome**
   - Title: "Administrator Dashboard"
   - Green success banner: "System Administrator"
   - Message: "Congratulations! You are now the system administrator..."

### Normal Operation (After Setup)

1. **Login Page**
   - Standard login form
   - No first-time setup banners

2. **Registration Modal**
   - Standard "Create Your Account" title
   - Normal user registration flow

3. **Dashboard**
   - Standard "Welcome to Your Dashboard" for regular users
   - "Administrator Dashboard" for admin users

## 🔒 Security Considerations

### First Admin Privileges

The first user automatically receives:
- **`is_admin: true`** - Full administrative privileges
- **`is_verified: true`** - Email verification bypassed
- **`user_group: "admin"`** - Administrator group membership
- **Full System Access** - Can manage users, settings, and all features

### Subsequent Users

All users after the first are created as:
- **`is_admin: false`** - Standard user privileges
- **`is_verified: false`** - Email verification required
- **`user_group: "user"`** - Standard user group
- **Limited Access** - Standard user permissions

## 🛠️ Implementation Details

### Backend Changes

**File: `app/services/auth_service.py`**
- Added `is_first_time_setup()` method
- Modified `register_user()` to handle first admin creation
- Enhanced logging for first admin creation

**File: `app/api/v1/endpoints/auth.py`**
- Added `FirstTimeSetupResponse` model
- Added `GET /auth/setup/status` endpoint
- Updated registration documentation

### Frontend Changes

**File: `frontend/src/lib/types/auth.ts`**
- Added `FirstTimeSetupStatus` interface
- Updated `AuthState` with `isFirstTimeSetup` property

**File: `frontend/src/lib/api/auth.ts`**
- Added `getFirstTimeSetupStatus()` method

**File: `frontend/src/lib/stores/auth.ts`**
- Added `checkFirstTimeSetup()` method
- Added `isFirstTimeSetup` derived store

**File: `frontend/src/routes/login/+page.svelte`**
- Added first-time setup banner
- Automatic setup status checking

**File: `frontend/src/lib/components/auth/RegisterForm.svelte`**
- Enhanced header for admin setup
- Administrator privilege warning

**File: `frontend/src/routes/dashboard/+page.svelte`**
- Administrator welcome message
- System admin success banner

## 🧪 Testing the Feature

### Test First-Time Setup

1. **Ensure Clean Database**
   ```bash
   # Reset database or ensure no users exist
   ```

2. **Start Backend**
   ```bash
   cd app
   python -m uvicorn main:app --host 0.0.0.0 --port 8888 --reload
   ```

3. **Start Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Test Flow**
   - Visit `http://localhost:3000`
   - Should see first-time setup banner
   - Click "Sign up here"
   - Should see "Create Administrator Account" modal
   - Register first user
   - Should redirect to "Administrator Dashboard"
   - Check user has admin privileges

### Test Normal Operation

1. **After First User Exists**
   - Visit login page - no first-time setup banner
   - Register additional users - normal user privileges
   - Login as admin - see administrator dashboard
   - Login as regular user - see standard dashboard

## 🔍 Troubleshooting

### First-Time Setup Not Detected

**Problem**: Banner doesn't appear even with empty database
**Solution**: 
- Check database connection
- Verify `is_first_time_setup()` method is working
- Check browser console for API errors

### First User Not Admin

**Problem**: First registered user doesn't have admin privileges
**Solution**:
- Verify `is_first_time_setup()` returns `true` before registration
- Check database user count query
- Verify user creation logic in `register_user()`

### Setup Status API Error

**Problem**: `/auth/setup/status` endpoint returns error
**Solution**:
- Check database connectivity
- Verify endpoint is properly registered
- Check server logs for detailed error information

## 📝 Configuration

No additional configuration is required. The first-time setup system works automatically based on the presence or absence of users in the database.

### Environment Variables

The system uses existing authentication configuration:
- `AGENTIC_DATABASE_URL` - Database connection
- `AGENTIC_SECRET_KEY` - JWT token signing
- All other standard authentication settings

## 🚀 Future Enhancements

Potential improvements for the first-time setup system:

1. **Setup Wizard** - Multi-step setup process for system configuration
2. **Organization Setup** - Company/organization information collection
3. **Initial Configuration** - System settings, themes, preferences
4. **Welcome Tour** - Guided tour of administrator features
5. **Backup Admin** - Option to create multiple admin users during setup

## 📄 Related Documentation

- [Authentication System](./SSO_AUTHENTICATION.md)
- [User Management](./USER_MANAGEMENT.md)
- [API Documentation](./API_REFERENCE.md)
- [Frontend Guide](../frontend/README.md)
