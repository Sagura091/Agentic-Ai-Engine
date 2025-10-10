"""
🚀 Revolutionary Configuration Broadcasting System

Real-time notification system for configuration changes with:
- Customizable broadcasting levels (PUBLIC, ADMIN_ONLY, SYSTEM_ONLY)
- User preference-based filtering
- WebSocket real-time notifications
- Fail-safe error handling
- Comprehensive audit trails

BROADCASTING LEVELS:
✅ PUBLIC: Broadcast to all users via WebSocket
✅ ADMIN_ONLY: Notify only admins, no user notifications
✅ SYSTEM_ONLY: Internal changes, no notifications
✅ ENCRYPTED: Secure settings with limited access
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory
from ..api.websocket.manager import websocket_manager
from ..models.auth import UserDB
from ..models.database.base import get_database_session

_backend_logger = get_backend_logger()


class BroadcastLevel(str, Enum):
    """Broadcasting levels for configuration changes."""
    PUBLIC = "public"           # Broadcast to all users
    ADMIN_ONLY = "admin_only"   # Admin only, no user notifications  
    SYSTEM_ONLY = "system_only" # Internal only, no notifications
    ENCRYPTED = "encrypted"     # Secure settings, limited access


class NotificationType(str, Enum):
    """Types of notifications for user preferences."""
    MODEL_UPDATES = "model_updates"
    SYSTEM_UPDATES = "system_updates"
    AGENT_SUGGESTIONS = "agent_suggestions"
    PERFORMANCE_ALERTS = "performance_alerts"
    SECURITY_UPDATES = "security_updates"
    RAG_UPDATES = "rag_updates"
    LLM_UPDATES = "llm_updates"
    MEMORY_UPDATES = "memory_updates"
    DATABASE_UPDATES = "database_updates"
    STORAGE_UPDATES = "storage_updates"


class ConfigurationBroadcaster:
    """🚀 Revolutionary Configuration Broadcasting System"""
    
    def __init__(self):
        self._active_users: Dict[str, Dict[str, Any]] = {}
        self._admin_users: Set[str] = set()
        self._user_preferences: Dict[str, Dict[str, bool]] = {}
        
    async def initialize(self) -> None:
        """Initialize the broadcaster with current user data."""
        try:
            async for session in get_database_session():
                # Load active users and their preferences
                from sqlalchemy import select
                result = await session.execute(
                    select(UserDB).where(UserDB.is_active == True)
                )
                users = result.scalars().all()
                
                for user in users:
                    self._active_users[str(user.id)] = {
                        "username": user.username,
                        "email": user.email,
                        "user_group": user.user_group,
                        "is_admin": user.user_group in ["admin", "moderator"]
                    }
                    
                    if user.user_group in ["admin", "moderator"]:
                        self._admin_users.add(str(user.id))
                    
                    # Load user notification preferences (default to all enabled)
                    self._user_preferences[str(user.id)] = {
                        NotificationType.MODEL_UPDATES: True,
                        NotificationType.SYSTEM_UPDATES: True,
                        NotificationType.AGENT_SUGGESTIONS: True,
                        NotificationType.PERFORMANCE_ALERTS: True,
                        NotificationType.SECURITY_UPDATES: True,
                        NotificationType.RAG_UPDATES: True,
                        NotificationType.LLM_UPDATES: True,
                        NotificationType.MEMORY_UPDATES: True,
                        NotificationType.DATABASE_UPDATES: True,
                        NotificationType.STORAGE_UPDATES: True,
                    }
                
                _backend_logger.info(
                    f"✅ Configuration broadcaster initialized with {len(self._active_users)} users",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.configuration_broadcaster"
                )

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to initialize configuration broadcaster: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )
            raise
    
    async def broadcast_configuration_change(
        self,
        section: str,
        setting_key: str,
        changes: Dict[str, Any],
        broadcast_level: BroadcastLevel,
        admin_user_id: str,
        notification_type: NotificationType
    ) -> Dict[str, Any]:
        """
        Broadcast configuration changes based on level and user preferences.
        
        Args:
            section: Configuration section (e.g., 'rag_configuration', 'llm_providers')
            setting_key: Specific setting that changed
            changes: Dictionary of changes made
            broadcast_level: Level of broadcasting (PUBLIC, ADMIN_ONLY, etc.)
            admin_user_id: ID of admin who made the change
            notification_type: Type of notification for user preference filtering
            
        Returns:
            Dictionary with broadcast results and statistics
        """
        try:
            broadcast_stats = {
                "total_users": len(self._active_users),
                "notifications_sent": 0,
                "admin_notifications": 0,
                "user_notifications": 0,
                "filtered_out": 0,
                "errors": []
            }
            
            # Create base notification message
            base_message = {
                "type": "configuration_update",
                "section": section,
                "setting_key": setting_key,
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat(),
                "admin_user": admin_user_id,
                "notification_type": notification_type.value
            }
            
            if broadcast_level == BroadcastLevel.SYSTEM_ONLY:
                _backend_logger.info(
                    f"🔒 System-only change: {section}.{setting_key} - No notifications sent",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.configuration_broadcaster"
                )
                return broadcast_stats

            elif broadcast_level == BroadcastLevel.ADMIN_ONLY:
                # Notify only admins
                admin_message = {
                    **base_message,
                    "admin_only": True,
                    "message": f"Admin setting updated: {setting_key}"
                }

                for admin_id in self._admin_users:
                    try:
                        await websocket_manager.send_personal_message(admin_id, admin_message)
                        broadcast_stats["admin_notifications"] += 1
                        broadcast_stats["notifications_sent"] += 1
                    except Exception as e:
                        broadcast_stats["errors"].append(f"Failed to notify admin {admin_id}: {str(e)}")

                _backend_logger.info(
                    f"📢 Admin-only broadcast: {section}.{setting_key} - {broadcast_stats['admin_notifications']} admins notified",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.configuration_broadcaster"
                )

            elif broadcast_level == BroadcastLevel.PUBLIC:
                # Broadcast to all users based on their preferences
                user_message = {
                    **base_message,
                    "public": True,
                    "message": self._generate_user_friendly_message(section, setting_key, changes)
                }

                for user_id, user_data in self._active_users.items():
                    try:
                        # Check user notification preferences
                        user_prefs = self._user_preferences.get(user_id, {})
                        if not user_prefs.get(notification_type, True):
                            broadcast_stats["filtered_out"] += 1
                            continue

                        # Send notification
                        await websocket_manager.send_personal_message(user_id, user_message)

                        if user_data["is_admin"]:
                            broadcast_stats["admin_notifications"] += 1
                        else:
                            broadcast_stats["user_notifications"] += 1

                        broadcast_stats["notifications_sent"] += 1

                    except Exception as e:
                        broadcast_stats["errors"].append(f"Failed to notify user {user_id}: {str(e)}")

                _backend_logger.info(
                    f"📢 Public broadcast: {section}.{setting_key} - {broadcast_stats['notifications_sent']} users notified",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.configuration_broadcaster"
                )
            
            elif broadcast_level == BroadcastLevel.ENCRYPTED:
                # Special handling for encrypted/secure settings
                secure_message = {
                    **base_message,
                    "encrypted": True,
                    "message": "Secure system configuration updated"
                }
                
                # Only notify super admins
                super_admins = [uid for uid, data in self._active_users.items() 
                              if data["user_group"] == "admin"]
                
                for admin_id in super_admins:
                    try:
                        await websocket_manager.send_personal_message(admin_id, secure_message)
                        broadcast_stats["admin_notifications"] += 1
                        broadcast_stats["notifications_sent"] += 1
                    except Exception as e:
                        broadcast_stats["errors"].append(f"Failed to notify super admin {admin_id}: {str(e)}")
                
                _backend_logger.info(
                    f"🔐 Encrypted broadcast: {section}.{setting_key} - {broadcast_stats['admin_notifications']} super admins notified",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.configuration_broadcaster"
                )

            return broadcast_stats

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to broadcast configuration change: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )
            return {
                "total_users": 0,
                "notifications_sent": 0,
                "admin_notifications": 0,
                "user_notifications": 0,
                "filtered_out": 0,
                "errors": [str(e)]
            }
    
    def _generate_user_friendly_message(self, section: str, setting_key: str, changes: Dict[str, Any]) -> str:
        """Generate user-friendly messages for configuration changes."""
        if section == "llm_providers":
            if "available_models_update" in changes:
                action = changes["available_models_update"].get("action", "updated")
                model_name = changes["available_models_update"].get("model_name", "Unknown")
                if action == "added":
                    return f"🚀 New AI model available: {model_name}"
                elif action == "removed":
                    return f"📋 AI model removed: {model_name}"
                else:
                    return f"🔄 AI model updated: {model_name}"
            else:
                return f"🧠 LLM provider settings updated: {setting_key}"
        
        elif section == "rag_configuration":
            return f"🔍 RAG system updated: {setting_key}"
        
        elif section == "memory_system":
            return f"🧠 Memory system updated: {setting_key}"
        
        elif section == "system_configuration":
            return f"⚙️ System configuration updated: {setting_key}"
        
        else:
            return f"🔄 System updated: {section}.{setting_key}"
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, bool]) -> bool:
        """Update user notification preferences."""
        try:
            self._user_preferences[user_id] = preferences
            _backend_logger.info(
                f"✅ Updated notification preferences for user {user_id}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )
            return True
        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to update user preferences: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )
            return False

    async def add_user(self, user_id: str, user_data: Dict[str, Any]) -> None:
        """Add a new user to the broadcaster."""
        self._active_users[user_id] = user_data
        if user_data.get("is_admin", False):
            self._admin_users.add(user_id)
        
        # Set default preferences
        self._user_preferences[user_id] = {nt: True for nt in NotificationType}

        _backend_logger.info(
            f"✅ Added user {user_id} to configuration broadcaster",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.configuration_broadcaster"
        )

    async def remove_user(self, user_id: str) -> None:
        """Remove a user from the broadcaster."""
        self._active_users.pop(user_id, None)
        self._admin_users.discard(user_id)
        self._user_preferences.pop(user_id, None)

        _backend_logger.info(
            f"✅ Removed user {user_id} from configuration broadcaster",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.configuration_broadcaster"
        )

    async def broadcast_model_availability(
        self,
        model_id: str,
        model_type: str,
        is_available: bool,
        admin_user_id: str,
        is_public: bool = True
    ) -> None:
        """Broadcast model availability changes to users."""
        try:
            _backend_logger.info(
                f"📡 Broadcasting model availability: {model_id} ({'available' if is_available else 'removed'})",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )

            # Create notification message
            action = "added" if is_available else "removed"
            message = {
                "type": "model_availability_update",
                "model_id": model_id,
                "model_type": model_type,
                "is_available": is_available,
                "is_public": is_public,
                "action": action,
                "admin_user_id": admin_user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Model {action}: {model_id} ({model_type})"
            }

            # Determine who should receive the notification
            if is_public:
                # Broadcast to all users
                await self._broadcast_to_all_users(message)
                _backend_logger.info(
                    f"✅ Model availability broadcasted to all users: {model_id}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.configuration_broadcaster"
                )
            else:
                # Broadcast only to admins
                await self._broadcast_to_admins(message)
                _backend_logger.info(
                    f"✅ Model availability broadcasted to admins only: {model_id}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.configuration_broadcaster"
                )

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to broadcast model availability: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )

    async def broadcast_embedding_model_update(
        self,
        old_model: str,
        new_model: str,
        admin_user_id: str
    ) -> None:
        """Broadcast embedding model updates to RAG system users."""
        try:
            _backend_logger.info(
                f"📡 Broadcasting embedding model update: {old_model} → {new_model}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )

            message = {
                "type": "embedding_model_update",
                "old_model": old_model,
                "new_model": new_model,
                "admin_user_id": admin_user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Embedding model updated: {old_model} → {new_model}",
                "requires_rag_restart": True
            }

            # Broadcast to all users since this affects RAG functionality
            await self._broadcast_to_all_users(message)
            _backend_logger.info(
                f"✅ Embedding model update broadcasted: {new_model}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to broadcast embedding model update: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )

    async def broadcast_vision_model_update(
        self,
        old_model: str,
        new_model: str,
        admin_user_id: str
    ) -> None:
        """Broadcast vision model updates to users."""
        try:
            _backend_logger.info(
                f"📡 Broadcasting vision model update: {old_model} → {new_model}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )

            message = {
                "type": "vision_model_update",
                "old_model": old_model,
                "new_model": new_model,
                "admin_user_id": admin_user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Vision model updated: {old_model} → {new_model}",
                "affects_multimodal": True
            }

            # Broadcast to all users
            await self._broadcast_to_all_users(message)
            _backend_logger.info(
                f"✅ Vision model update broadcasted: {new_model}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to broadcast vision model update: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.configuration_broadcaster"
            )


# Global instance
configuration_broadcaster = ConfigurationBroadcaster()
