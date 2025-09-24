"""
🚀 Revolutionary WebSocket Notification Handlers

Real-time notification system for configuration changes and system updates.
Handles user-specific notifications with preference filtering and fail-safe delivery.

NOTIFICATION TYPES:
✅ Model availability updates
✅ System configuration changes
✅ Agent upgrade suggestions
✅ Performance alerts
✅ Security updates
✅ RAG system updates
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
import structlog

from ...core.configuration_broadcaster import configuration_broadcaster, NotificationType, BroadcastLevel
from ...models.auth import UserDB
from ...models.database.base import get_database_session
from .manager import websocket_manager

logger = structlog.get_logger(__name__)


class NotificationHandler:
    """🚀 Revolutionary WebSocket Notification Handler"""
    
    def __init__(self):
        self._notification_queue: Dict[str, List[Dict[str, Any]]] = {}
        self._user_connections: Dict[str, str] = {}  # user_id -> connection_id
        
    async def initialize(self) -> None:
        """Initialize the notification handler."""
        try:
            # Initialize configuration broadcaster
            await configuration_broadcaster.initialize()
            logger.info("✅ Notification handler initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize notification handler: {str(e)}")
            raise
    
    async def register_user_connection(self, user_id: str, connection_id: str) -> None:
        """Register a user's WebSocket connection for notifications."""
        try:
            self._user_connections[user_id] = connection_id
            
            # Add user to configuration broadcaster
            async for session in get_database_session():
                from sqlalchemy import select
                result = await session.execute(
                    select(UserDB).where(UserDB.id == UUID(user_id))
                )
                user = result.scalar_one_or_none()
                
                if user:
                    await configuration_broadcaster.add_user(user_id, {
                        "username": user.username,
                        "email": user.email,
                        "user_group": user.user_group,
                        "is_admin": user.user_group in ["admin", "moderator"]
                    })
            
            # Send any queued notifications
            await self._send_queued_notifications(user_id)
            
            logger.info(f"✅ Registered user {user_id} for notifications")
            
        except Exception as e:
            logger.error(f"❌ Failed to register user connection: {str(e)}")
    
    async def unregister_user_connection(self, user_id: str) -> None:
        """Unregister a user's WebSocket connection."""
        try:
            self._user_connections.pop(user_id, None)
            await configuration_broadcaster.remove_user(user_id)
            
            logger.info(f"✅ Unregistered user {user_id} from notifications")
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister user connection: {str(e)}")
    
    async def send_model_availability_notification(
        self,
        model_name: str,
        action: str,  # "added", "removed", "updated"
        admin_user_id: str,
        model_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send model availability notifications to users."""
        try:
            logger.info(f"📢 Broadcasting model {action}: {model_name}")
            
            changes = {
                "available_models_update": {
                    "action": action,
                    "model_name": model_name,
                    "details": model_details or {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Broadcast via configuration broadcaster
            result = await configuration_broadcaster.broadcast_configuration_change(
                section="llm_providers",
                setting_key="available_models",
                changes=changes,
                broadcast_level=BroadcastLevel.PUBLIC,
                admin_user_id=admin_user_id,
                notification_type=NotificationType.MODEL_UPDATES
            )
            
            logger.info(f"✅ Model availability notification sent: {result['notifications_sent']} users notified")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to send model availability notification: {str(e)}")
            return {"notifications_sent": 0, "errors": [str(e)]}
    
    async def send_agent_upgrade_suggestion(
        self,
        user_id: str,
        agent_id: str,
        agent_name: str,
        current_model: str,
        suggested_model: str,
        improvement_details: Dict[str, Any]
    ) -> bool:
        """Send agent upgrade suggestion to specific user."""
        try:
            connection_id = self._user_connections.get(user_id)
            if not connection_id:
                # Queue notification for when user connects
                await self._queue_notification(user_id, {
                    "type": "agent_upgrade_suggestion",
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "current_model": current_model,
                    "suggested_model": suggested_model,
                    "improvement_details": improvement_details,
                    "timestamp": datetime.utcnow().isoformat()
                })
                return True
            
            notification = {
                "type": "agent_upgrade_suggestion",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "current_model": current_model,
                "suggested_model": suggested_model,
                "improvement_details": improvement_details,
                "message": f"💡 Upgrade suggestion for {agent_name}: Switch to {suggested_model} for better performance",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket_manager.send_personal_message(connection_id, notification)
            
            logger.info(f"✅ Agent upgrade suggestion sent to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send agent upgrade suggestion: {str(e)}")
            return False
    
    async def send_system_update_notification(
        self,
        update_type: str,
        message: str,
        details: Dict[str, Any],
        broadcast_level: BroadcastLevel = BroadcastLevel.PUBLIC
    ) -> Dict[str, Any]:
        """Send system update notifications."""
        try:
            logger.info(f"📢 Broadcasting system update: {update_type}")
            
            changes = {
                "system_update": {
                    "type": update_type,
                    "message": message,
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            result = await configuration_broadcaster.broadcast_configuration_change(
                section="system_configuration",
                setting_key="system_update",
                changes=changes,
                broadcast_level=broadcast_level,
                admin_user_id="system",
                notification_type=NotificationType.SYSTEM_UPDATES
            )
            
            logger.info(f"✅ System update notification sent: {result['notifications_sent']} users notified")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to send system update notification: {str(e)}")
            return {"notifications_sent": 0, "errors": [str(e)]}
    
    async def send_performance_alert(
        self,
        user_id: str,
        alert_type: str,
        message: str,
        severity: str,  # "info", "warning", "error"
        details: Dict[str, Any]
    ) -> bool:
        """Send performance alert to specific user."""
        try:
            connection_id = self._user_connections.get(user_id)
            if not connection_id:
                await self._queue_notification(user_id, {
                    "type": "performance_alert",
                    "alert_type": alert_type,
                    "message": message,
                    "severity": severity,
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat()
                })
                return True
            
            notification = {
                "type": "performance_alert",
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket_manager.send_personal_message(connection_id, notification)
            
            logger.info(f"✅ Performance alert sent to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send performance alert: {str(e)}")
            return False
    
    async def broadcast_bulk_agent_update_progress(
        self,
        user_id: str,
        job_id: str,
        progress: float,
        status: str,
        details: Dict[str, Any]
    ) -> bool:
        """Send bulk agent update progress to user."""
        try:
            connection_id = self._user_connections.get(user_id)
            if not connection_id:
                return False
            
            notification = {
                "type": "bulk_update_progress",
                "job_id": job_id,
                "progress": progress,
                "status": status,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket_manager.send_personal_message(connection_id, notification)
            
            logger.debug(f"📊 Bulk update progress sent to user {user_id}: {progress}%")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send bulk update progress: {str(e)}")
            return False
    
    async def _queue_notification(self, user_id: str, notification: Dict[str, Any]) -> None:
        """Queue notification for offline user."""
        if user_id not in self._notification_queue:
            self._notification_queue[user_id] = []
        
        self._notification_queue[user_id].append(notification)
        
        # Limit queue size to prevent memory issues
        if len(self._notification_queue[user_id]) > 50:
            self._notification_queue[user_id] = self._notification_queue[user_id][-50:]
        
        logger.info(f"📥 Queued notification for offline user {user_id}")
    
    async def _send_queued_notifications(self, user_id: str) -> None:
        """Send all queued notifications to user."""
        try:
            queued = self._notification_queue.get(user_id, [])
            if not queued:
                return
            
            connection_id = self._user_connections.get(user_id)
            if not connection_id:
                return
            
            for notification in queued:
                try:
                    await websocket_manager.send_personal_message(connection_id, notification)
                except Exception as e:
                    logger.error(f"❌ Failed to send queued notification: {str(e)}")
            
            # Clear queue
            self._notification_queue[user_id] = []
            
            logger.info(f"✅ Sent {len(queued)} queued notifications to user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send queued notifications: {str(e)}")


# Global instance
notification_handler = NotificationHandler()
