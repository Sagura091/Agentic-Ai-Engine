"""
Conversation Management API Endpoints.

This module provides REST API endpoints for managing chat conversations,
message history, and conversation metadata.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy import select, update, delete, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.auth import ConversationDB, MessageDB, ConversationCreate, ConversationResponse, MessageResponse, UserResponse
from app.models.database.base import get_database_session
from app.api.v1.endpoints.auth import get_current_user
from app.backend_logging.backend_logger import get_logger, LogCategory

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversation Management"])


@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ConversationResponse:
    """
    Create a new conversation.
    
    Creates a new conversation thread for the user.
    
    Args:
        conversation_data: Conversation creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created conversation information
        
    Raises:
        HTTPException: If conversation creation fails
    """
    try:
        # Create conversation
        conversation = ConversationDB(
            title=conversation_data.title,
            user_id=UUID(current_user.id),
            project_id=UUID(conversation_data.project_id) if conversation_data.project_id else None,
            agent_id=UUID(conversation_data.agent_id) if conversation_data.agent_id else None,
            metadata={
                "model": conversation_data.model or "llama3.2:latest",
                "temperature": 0.7,
                "max_tokens": 2048,
                "system_prompt": conversation_data.system_prompt
            }
        )
        
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        
        get_logger().info(
            f"Conversation created: {conversation.title}",
            LogCategory.CONVERSATION_MANAGEMENT,
            "ConversationAPI",
            data={
                "conversation_id": str(conversation.id),
                "user_id": current_user.id,
                "project_id": conversation_data.project_id,
                "agent_id": conversation_data.agent_id
            }
        )
        
        return ConversationResponse(
            id=str(conversation.id),
            title=conversation.title,
            user_id=str(conversation.user_id),
            project_id=str(conversation.project_id) if conversation.project_id else None,
            agent_id=str(conversation.agent_id) if conversation.agent_id else None,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=0,
            last_message_at=None,
            is_pinned=conversation.is_pinned,
            is_archived=conversation.is_archived,
            metadata=conversation.metadata
        )
        
    except Exception as e:
        await db.rollback()
        get_logger().error(
            f"Conversation creation failed: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ConversationAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )


@router.get("/", response_model=List[ConversationResponse])
async def list_conversations(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
    project_id: Optional[str] = Query(default=None, description="Filter by project ID"),
    agent_id: Optional[str] = Query(default=None, description="Filter by agent ID"),
    include_archived: bool = Query(default=False, description="Include archived conversations"),
    limit: int = Query(default=50, le=100, description="Maximum number of conversations"),
    offset: int = Query(default=0, description="Number of conversations to skip")
) -> List[ConversationResponse]:
    """
    List user's conversations.
    
    Returns conversations owned by the user with optional filtering.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        project_id: Optional project filter
        agent_id: Optional agent filter
        include_archived: Whether to include archived conversations
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List of conversations
    """
    try:
        user_id = UUID(current_user.id)
        
        # Build query
        query = select(ConversationDB).options(
            selectinload(ConversationDB.messages)
        ).where(
            and_(
                ConversationDB.user_id == user_id,
                ConversationDB.deleted_at.is_(None)
            )
        )
        
        # Add filters
        if project_id:
            query = query.where(ConversationDB.project_id == UUID(project_id))
        
        if agent_id:
            query = query.where(ConversationDB.agent_id == UUID(agent_id))
        
        if not include_archived:
            query = query.where(ConversationDB.is_archived == False)
        
        # Order by last activity (pinned first)
        query = query.order_by(
            desc(ConversationDB.is_pinned),
            desc(ConversationDB.updated_at)
        ).limit(limit).offset(offset)
        
        result = await db.execute(query)
        conversations = result.scalars().all()
        
        conversations_list = []
        for conv in conversations:
            # Get message count and last message time
            message_count = len(conv.messages)
            last_message_at = None
            if conv.messages:
                last_message_at = max(msg.created_at for msg in conv.messages)
            
            conversations_list.append(ConversationResponse(
                id=str(conv.id),
                title=conv.title,
                user_id=str(conv.user_id),
                project_id=str(conv.project_id) if conv.project_id else None,
                agent_id=str(conv.agent_id) if conv.agent_id else None,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=message_count,
                last_message_at=last_message_at,
                is_pinned=conv.is_pinned,
                is_archived=conv.is_archived,
                metadata=conv.metadata
            ))
        
        get_logger().info(
            f"Listed {len(conversations_list)} conversations for user",
            LogCategory.CONVERSATION_MANAGEMENT,
            "ConversationAPI",
            data={
                "user_id": current_user.id,
                "conversation_count": len(conversations_list),
                "project_id": project_id,
                "agent_id": agent_id,
                "include_archived": include_archived
            }
        )
        
        return conversations_list
        
    except Exception as e:
        get_logger().error(
            f"Failed to list conversations: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ConversationAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list conversations"
        )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ConversationResponse:
    """
    Get specific conversation details.
    
    Returns conversation information if user owns it.
    
    Args:
        conversation_id: Conversation ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Conversation information
        
    Raises:
        HTTPException: If conversation not found or access denied
    """
    try:
        conversation_uuid = UUID(conversation_id)
        user_id = UUID(current_user.id)
        
        query = select(ConversationDB).options(
            selectinload(ConversationDB.messages)
        ).where(
            and_(
                ConversationDB.id == conversation_uuid,
                ConversationDB.user_id == user_id,
                ConversationDB.deleted_at.is_(None)
            )
        )
        
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        # Get message count and last message time
        message_count = len(conversation.messages)
        last_message_at = None
        if conversation.messages:
            last_message_at = max(msg.created_at for msg in conversation.messages)
        
        return ConversationResponse(
            id=str(conversation.id),
            title=conversation.title,
            user_id=str(conversation.user_id),
            project_id=str(conversation.project_id) if conversation.project_id else None,
            agent_id=str(conversation.agent_id) if conversation.agent_id else None,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=message_count,
            last_message_at=last_message_at,
            is_pinned=conversation.is_pinned,
            is_archived=conversation.is_archived,
            metadata=conversation.metadata
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        get_logger().error(
            f"Failed to get conversation {conversation_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ConversationAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation"
        )
