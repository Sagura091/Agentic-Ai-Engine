"""
Project Management API Endpoints.

This module provides REST API endpoints for project/workspace management,
including creation, collaboration, and member management.
"""

from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Depends, status, Query
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.auth import ProjectDB, ProjectMemberDB, ProjectCreate, ProjectResponse, UserResponse
from app.models.database.base import get_database_session
from app.api.v1.endpoints.auth import get_current_user
from app.backend_logging.backend_logger import get_logger, LogCategory

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/projects", tags=["Project Management"])


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ProjectResponse:
    """
    Create a new project/workspace.
    
    Creates a new project with the current user as owner and
    sets up initial project structure.
    
    Args:
        project_data: Project creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created project information
        
    Raises:
        HTTPException: If project creation fails
    """
    try:
        # Create project
        project = ProjectDB(
            name=project_data.name,
            description=project_data.description,
            color=project_data.color,
            icon=project_data.icon,
            is_public=project_data.is_public,
            owner_id=UUID(current_user.id),
            settings={
                "default_agent_model": "llama3.2:latest",
                "auto_save": True,
                "collaboration_enabled": True
            }
        )
        
        db.add(project)
        await db.commit()
        await db.refresh(project)
        
        # Add owner as admin member
        owner_member = ProjectMemberDB(
            project_id=project.id,
            user_id=UUID(current_user.id),
            role="owner",
            permissions={
                "read": True,
                "write": True,
                "admin": True,
                "delete": True,
                "invite": True
            }
        )
        
        db.add(owner_member)
        await db.commit()
        
        get_logger().info(
            f"Project created: {project.name}",
            LogCategory.PROJECT_MANAGEMENT,
            "ProjectAPI",
            data={
                "project_id": str(project.id),
                "project_name": project.name,
                "owner_id": current_user.id,
                "is_public": project.is_public
            }
        )
        
        return ProjectResponse(
            id=str(project.id),
            name=project.name,
            description=project.description,
            color=project.color,
            icon=project.icon,
            is_public=project.is_public,
            is_archived=project.is_archived,
            owner_id=str(project.owner_id),
            created_at=project.created_at,
            updated_at=project.updated_at,
            member_count=1
        )
        
    except Exception as e:
        await db.rollback()
        get_logger().error(
            f"Project creation failed: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ProjectAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create project"
        )


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
    include_archived: bool = Query(default=False, description="Include archived projects"),
    include_public: bool = Query(default=True, description="Include public projects")
) -> List[ProjectResponse]:
    """
    List user's projects and accessible public projects.
    
    Returns projects where the user is a member or owner,
    plus public projects if requested.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        include_archived: Whether to include archived projects
        include_public: Whether to include public projects
        
    Returns:
        List of accessible projects
    """
    try:
        user_id = UUID(current_user.id)
        
        # Build query for user's projects
        query = select(ProjectDB).options(
            selectinload(ProjectDB.members)
        ).join(
            ProjectMemberDB, ProjectDB.id == ProjectMemberDB.project_id
        ).where(
            and_(
                ProjectMemberDB.user_id == user_id,
                ProjectMemberDB.is_active == True,
                ProjectDB.deleted_at.is_(None)
            )
        )
        
        # Add archived filter
        if not include_archived:
            query = query.where(ProjectDB.is_archived == False)
        
        # Execute query for user projects
        result = await db.execute(query)
        user_projects = result.unique().scalars().all()
        
        projects_list = []
        
        # Process user projects
        for project in user_projects:
            member_count = len([m for m in project.members if m.is_active])
            projects_list.append(ProjectResponse(
                id=str(project.id),
                name=project.name,
                description=project.description,
                color=project.color,
                icon=project.icon,
                is_public=project.is_public,
                is_archived=project.is_archived,
                owner_id=str(project.owner_id),
                created_at=project.created_at,
                updated_at=project.updated_at,
                member_count=member_count
            ))
        
        # Add public projects if requested
        if include_public:
            public_query = select(ProjectDB).options(
                selectinload(ProjectDB.members)
            ).where(
                and_(
                    ProjectDB.is_public == True,
                    ProjectDB.deleted_at.is_(None),
                    ProjectDB.owner_id != user_id  # Exclude user's own public projects
                )
            )
            
            if not include_archived:
                public_query = public_query.where(ProjectDB.is_archived == False)
            
            public_result = await db.execute(public_query)
            public_projects = public_result.scalars().all()
            
            for project in public_projects:
                member_count = len([m for m in project.members if m.is_active])
                projects_list.append(ProjectResponse(
                    id=str(project.id),
                    name=project.name,
                    description=project.description,
                    color=project.color,
                    icon=project.icon,
                    is_public=project.is_public,
                    is_archived=project.is_archived,
                    owner_id=str(project.owner_id),
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                    member_count=member_count
                ))
        
        get_logger().info(
            f"Listed {len(projects_list)} projects for user",
            LogCategory.PROJECT_MANAGEMENT,
            "ProjectAPI",
            data={
                "user_id": current_user.id,
                "project_count": len(projects_list),
                "include_archived": include_archived,
                "include_public": include_public
            }
        )
        
        return projects_list
        
    except Exception as e:
        get_logger().error(
            f"Failed to list projects: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ProjectAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list projects"
        )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ProjectResponse:
    """
    Get specific project details.
    
    Returns project information if user has access.
    
    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Project information
        
    Raises:
        HTTPException: If project not found or access denied
    """
    try:
        project_uuid = UUID(project_id)
        user_id = UUID(current_user.id)
        
        # Check if user has access to project
        query = select(ProjectDB).options(
            selectinload(ProjectDB.members)
        ).where(
            and_(
                ProjectDB.id == project_uuid,
                ProjectDB.deleted_at.is_(None),
                or_(
                    ProjectDB.is_public == True,
                    ProjectDB.owner_id == user_id,
                    ProjectDB.members.any(
                        and_(
                            ProjectMemberDB.user_id == user_id,
                            ProjectMemberDB.is_active == True
                        )
                    )
                )
            )
        )
        
        result = await db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or access denied"
            )
        
        member_count = len([m for m in project.members if m.is_active])
        
        return ProjectResponse(
            id=str(project.id),
            name=project.name,
            description=project.description,
            color=project.color,
            icon=project.icon,
            is_public=project.is_public,
            is_archived=project.is_archived,
            owner_id=str(project.owner_id),
            created_at=project.created_at,
            updated_at=project.updated_at,
            member_count=member_count
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        get_logger().error(
            f"Failed to get project {project_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ProjectAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get project"
        )


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    project_data: ProjectCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ProjectResponse:
    """
    Update project information.

    Updates project details if user has admin permissions.

    Args:
        project_id: Project ID
        project_data: Updated project data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Updated project information

    Raises:
        HTTPException: If project not found or insufficient permissions
    """
    try:
        project_uuid = UUID(project_id)
        user_id = UUID(current_user.id)

        # Check if user has admin access
        query = select(ProjectDB).options(
            selectinload(ProjectDB.members)
        ).where(
            and_(
                ProjectDB.id == project_uuid,
                ProjectDB.deleted_at.is_(None),
                or_(
                    ProjectDB.owner_id == user_id,
                    ProjectDB.members.any(
                        and_(
                            ProjectMemberDB.user_id == user_id,
                            ProjectMemberDB.is_active == True,
                            ProjectMemberDB.permissions["admin"].astext.cast(bool) == True
                        )
                    )
                )
            )
        )

        result = await db.execute(query)
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or insufficient permissions"
            )

        # Update project fields
        update_stmt = update(ProjectDB).where(
            ProjectDB.id == project_uuid
        ).values(
            name=project_data.name,
            description=project_data.description,
            color=project_data.color,
            icon=project_data.icon,
            is_public=project_data.is_public
        )

        await db.execute(update_stmt)
        await db.commit()

        # Refresh project data
        await db.refresh(project)

        member_count = len([m for m in project.members if m.is_active])

        get_logger().info(
            f"Project updated: {project.name}",
            LogCategory.PROJECT_MANAGEMENT,
            "ProjectAPI",
            data={
                "project_id": project_id,
                "updated_by": current_user.id
            }
        )

        return ProjectResponse(
            id=str(project.id),
            name=project.name,
            description=project.description,
            color=project.color,
            icon=project.icon,
            is_public=project.is_public,
            is_archived=project.is_archived,
            owner_id=str(project.owner_id),
            created_at=project.created_at,
            updated_at=project.updated_at,
            member_count=member_count
        )

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        get_logger().error(
            f"Failed to update project {project_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ProjectAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update project"
        )


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> dict:
    """
    Delete project (soft delete).

    Marks project as deleted if user is owner.

    Args:
        project_id: Project ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Deletion confirmation

    Raises:
        HTTPException: If project not found or insufficient permissions
    """
    try:
        project_uuid = UUID(project_id)
        user_id = UUID(current_user.id)

        # Check if user is owner
        query = select(ProjectDB).where(
            and_(
                ProjectDB.id == project_uuid,
                ProjectDB.owner_id == user_id,
                ProjectDB.deleted_at.is_(None)
            )
        )

        result = await db.execute(query)
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found or insufficient permissions"
            )

        # Soft delete project
        from datetime import datetime
        update_stmt = update(ProjectDB).where(
            ProjectDB.id == project_uuid
        ).values(
            deleted_at=datetime.utcnow()
        )

        await db.execute(update_stmt)
        await db.commit()

        get_logger().info(
            f"Project deleted: {project.name}",
            LogCategory.PROJECT_MANAGEMENT,
            "ProjectAPI",
            data={
                "project_id": project_id,
                "deleted_by": current_user.id
            }
        )

        return {"message": "Project deleted successfully"}

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        get_logger().error(
            f"Failed to delete project {project_id}: {str(e)}",
            LogCategory.ERROR_TRACKING,
            "ProjectAPI",
            error=e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete project"
        )
