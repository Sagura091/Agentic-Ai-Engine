"""
Notification System API Endpoints.

This module provides REST API endpoints for managing user notifications,
including creation, retrieval, marking as read, and notification preferences.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy import select, update, delete, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.models.auth import NotificationDB, NotificationResponse, UserResponse
from app.models.database.base import get_database_session
from app.api.v1.endpoints.auth import get_current_user
from app.backend_logging.backend_logger import get_logger, LogCategory

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/notifications", tags=["Notification System"])


class NotificationCreate(BaseModel):
    """Notification creation model."""
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    type: str = Field(default="info", description="Notification type")
    priority: str = Field(default="normal", description="Notification priority")
    action_url: Optional[str] = Field(default=None, description="Action URL")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class NotificationUpdate(BaseModel):
    """Notification update model."""
    is_read: Optional[bool] = Field(default=None, description="Mark as read/unread")


class NotificationPreferences(BaseModel):
    """User notification preferences."""
    email_notifications: bool = Field(default=True, description="Enable email notifications")
    push_notifications: bool = Field(default=True, description="Enable push notifications")
    in_app_notifications: bool = Field(default=True, description="Enable in-app notifications")
    notification_types: dict = Field(
        default={
            "agent_completion": True,
            "workflow_status": True,
            "system_updates": True,
            "collaboration": True,
            "security": True
        },
        description="Notification type preferences"
    )


@router.post("/", response_model=NotificationResponse, status_code=status.HTTP_201_CREATED)
async def create_notification(
    notification_data: NotificationCreate,
    target_user_id: str = Query(..., description="Target user ID"),
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> NotificationResponse:
    """
    Create a new notification.
    
    Creates a notification for a specific user. This is typically used
    by system processes or admin users.
    
    Args:
        notification_data: Notification creation data
        target_user_id: ID of user to receive notification
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created notification information
        
    Raises:
        HTTPException: If notification creation fails
    """
    try:
        # Create notification
        notification = NotificationDB(
            user_id=UUID(target_user_id),
            title=notification_data.title,
            message=notification_data.message,
            type=notification_data.type,
            priority=notification_data.priority,
            action_url=notification_data.action_url,
            metadata=notification_data.metadata or {},
            created_by=UUID(current_user.id)
        )
        
        db.add(notification)
        await db.commit()
        await db.refresh(notification)
        
        get_logger().info(
            f"Notification created: {notification.title}",
            LogCategory.NOTIFICATION_SYSTEM,
            "NotificationAPI",
            data={
                "notification_id": str(notification.id),
                "target_user_id": target_user_id,
                "created_by": current_user.id,
                "type": notification.type,
                "priority": notification.priority
            }
        )
        
        return NotificationResponse(
            id=str(notification.id),
            user_id=str(notification.user_id),
            title=notification.title,
            message=notification.message,
            type=notification.type,
            priority=notification.priority,
            is_read=notification.is_read,
            action_url=notification.action_url,
            created_at=notification.created_at,
            read_at=notification.read_at,
            metadata=notification.metadata
        )
        
    except Exception as e:
        await db.rollback()
        get_logger().error(
            f"Notification creation failed: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "NotificationAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create notification"
        )


@router.get("/", response_model=List[NotificationResponse])
async def list_notifications(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
    unread_only: bool = Query(default=False, description="Show only unread notifications"),
    notification_type: Optional[str] = Query(default=None, description="Filter by notification type"),
    priority: Optional[str] = Query(default=None, description="Filter by priority"),
    limit: int = Query(default=50, le=100, description="Maximum number of notifications"),
    offset: int = Query(default=0, description="Number of notifications to skip")
) -> List[NotificationResponse]:
    """
    List user's notifications.
    
    Returns notifications for the current user with optional filtering.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        unread_only: Show only unread notifications
        notification_type: Filter by notification type
        priority: Filter by priority level
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List of notifications
    """
    try:
        user_id = UUID(current_user.id)
        
        # Build query
        query = select(NotificationDB).where(
            and_(
                NotificationDB.user_id == user_id,
                NotificationDB.deleted_at.is_(None)
            )
        )
        
        # Add filters
        if unread_only:
            query = query.where(NotificationDB.is_read == False)
        
        if notification_type:
            query = query.where(NotificationDB.type == notification_type)
        
        if priority:
            query = query.where(NotificationDB.priority == priority)
        
        # Order by priority and creation time
        priority_order = {
            "urgent": 1,
            "high": 2,
            "normal": 3,
            "low": 4
        }
        
        query = query.order_by(
            desc(NotificationDB.is_read == False),  # Unread first
            NotificationDB.priority,  # Then by priority
            desc(NotificationDB.created_at)  # Then by creation time
        ).limit(limit).offset(offset)
        
        result = await db.execute(query)
        notifications = result.scalars().all()
        
        notifications_list = []
        for notification in notifications:
            notifications_list.append(NotificationResponse(
                id=str(notification.id),
                user_id=str(notification.user_id),
                title=notification.title,
                message=notification.message,
                type=notification.type,
                priority=notification.priority,
                is_read=notification.is_read,
                action_url=notification.action_url,
                created_at=notification.created_at,
                read_at=notification.read_at,
                metadata=notification.metadata
            ))
        
        get_logger().info(
            f"Listed {len(notifications_list)} notifications for user",
            LogCategory.NOTIFICATION_SYSTEM,
            "NotificationAPI",
            data={
                "user_id": current_user.id,
                "notification_count": len(notifications_list),
                "unread_only": unread_only,
                "notification_type": notification_type,
                "priority": priority
            }
        )
        
        return notifications_list
        
    except Exception as e:
        get_logger().error(
            f"Failed to list notifications: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "NotificationAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list notifications"
        )


@router.get("/unread-count")
async def get_unread_count(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> dict:
    """
    Get count of unread notifications.
    
    Returns the number of unread notifications for the current user.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Unread notification count
    """
    try:
        user_id = UUID(current_user.id)
        
        query = select(NotificationDB).where(
            and_(
                NotificationDB.user_id == user_id,
                NotificationDB.is_read == False,
                NotificationDB.deleted_at.is_(None)
            )
        )
        
        result = await db.execute(query)
        notifications = result.scalars().all()
        
        # Count by priority
        count_by_priority = {
            "urgent": 0,
            "high": 0,
            "normal": 0,
            "low": 0
        }
        
        for notification in notifications:
            if notification.priority in count_by_priority:
                count_by_priority[notification.priority] += 1
        
        total_count = len(notifications)
        
        return {
            "total_unread": total_count,
            "by_priority": count_by_priority,
            "has_urgent": count_by_priority["urgent"] > 0
        }
        
    except Exception as e:
        get_logger().error(
            f"Failed to get unread count: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "NotificationAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get unread count"
        )


@router.put("/{notification_id}", response_model=NotificationResponse)
async def update_notification(
    notification_id: str,
    notification_data: NotificationUpdate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> NotificationResponse:
    """
    Update notification (mark as read/unread).
    
    Updates notification status if it belongs to the current user.
    
    Args:
        notification_id: Notification ID
        notification_data: Update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated notification information
        
    Raises:
        HTTPException: If notification not found or access denied
    """
    try:
        notification_uuid = UUID(notification_id)
        user_id = UUID(current_user.id)
        
        query = select(NotificationDB).where(
            and_(
                NotificationDB.id == notification_uuid,
                NotificationDB.user_id == user_id,
                NotificationDB.deleted_at.is_(None)
            )
        )
        
        result = await db.execute(query)
        notification = result.scalar_one_or_none()
        
        if not notification:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found or access denied"
            )
        
        # Update notification
        update_data = {}
        if notification_data.is_read is not None:
            update_data["is_read"] = notification_data.is_read
            if notification_data.is_read:
                update_data["read_at"] = datetime.utcnow()
            else:
                update_data["read_at"] = None
        
        if update_data:
            update_stmt = update(NotificationDB).where(
                NotificationDB.id == notification_uuid
            ).values(**update_data)
            
            await db.execute(update_stmt)
            await db.commit()
            await db.refresh(notification)
        
        get_logger().info(
            f"Notification updated: {notification.title}",
            LogCategory.NOTIFICATION_SYSTEM,
            "NotificationAPI",
            data={
                "notification_id": notification_id,
                "user_id": current_user.id,
                "is_read": notification.is_read
            }
        )
        
        return NotificationResponse(
            id=str(notification.id),
            user_id=str(notification.user_id),
            title=notification.title,
            message=notification.message,
            type=notification.type,
            priority=notification.priority,
            is_read=notification.is_read,
            action_url=notification.action_url,
            created_at=notification.created_at,
            read_at=notification.read_at,
            metadata=notification.metadata
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid notification ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        get_logger().error(
            f"Failed to update notification {notification_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "NotificationAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update notification"
        )
