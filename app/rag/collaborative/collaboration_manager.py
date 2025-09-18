"""
Revolutionary Collaborative Intelligence System for RAG 4.0.

This module provides advanced collaborative features including:
- Shared workspaces and team collaboration
- Knowledge contribution and curation system
- Collaborative intelligence protocols
- Real-time collaboration and synchronization
- Community-driven knowledge enhancement
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import structlog
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from ..core.embeddings import EmbeddingManager
from ..core.caching import get_rag_cache, CacheType
from ..core.resilience_manager import get_resilience_manager

logger = structlog.get_logger(__name__)


class WorkspaceType(Enum):
    """Types of collaborative workspaces."""
    PRIVATE = "private"
    TEAM = "team"
    ORGANIZATION = "organization"
    PUBLIC = "public"
    RESEARCH = "research"
    PROJECT = "project"


class ContributionType(Enum):
    """Types of knowledge contributions."""
    DOCUMENT = "document"
    ANNOTATION = "annotation"
    CORRECTION = "correction"
    ENHANCEMENT = "enhancement"
    REVIEW = "review"
    RATING = "rating"
    TAG = "tag"
    SUMMARY = "summary"


class CollaborationRole(Enum):
    """Roles in collaborative workspaces."""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


@dataclass
class Workspace:
    """Collaborative workspace definition."""
    id: str
    name: str
    description: str
    workspace_type: WorkspaceType
    owner_id: str
    members: List[Dict[str, Any]]
    knowledge_bases: List[str]
    settings: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class KnowledgeContribution:
    """Knowledge contribution from users."""
    id: str
    contributor_id: str
    workspace_id: str
    contribution_type: ContributionType
    target_document_id: Optional[str]
    content: str
    metadata: Dict[str, Any]
    status: str = "pending"  # pending, approved, rejected
    votes: Dict[str, int] = None  # user_id -> vote (1, 0, -1)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.votes is None:
            self.votes = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class CollaborativeSession:
    """Real-time collaborative session."""
    id: str
    workspace_id: str
    participants: List[str]
    active_document_id: Optional[str]
    session_data: Dict[str, Any]
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.started_at is None:
            self.started_at = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()


@dataclass
class CollaborativeIntelligence:
    """Collaborative intelligence insights."""
    workspace_id: str
    knowledge_quality_score: float
    collaboration_activity: Dict[str, Any]
    contribution_patterns: Dict[str, Any]
    knowledge_gaps: List[str]
    trending_topics: List[str]
    expert_recommendations: List[Dict[str, Any]]
    generated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.utcnow()


class WorkspaceManager:
    """Manages collaborative workspaces."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.workspaces: Dict[str, Workspace] = {}
        self.workspace_permissions: Dict[str, Dict[str, CollaborationRole]] = defaultdict(dict)
    
    async def create_workspace(
        self, 
        name: str, 
        description: str, 
        workspace_type: WorkspaceType,
        owner_id: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Workspace:
        """Create a new collaborative workspace."""
        try:
            workspace = Workspace(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                workspace_type=workspace_type,
                owner_id=owner_id,
                members=[{
                    "user_id": owner_id,
                    "role": CollaborationRole.OWNER.value,
                    "joined_at": datetime.utcnow().isoformat()
                }],
                knowledge_bases=[],
                settings=settings or {}
            )
            
            self.workspaces[workspace.id] = workspace
            self.workspace_permissions[workspace.id][owner_id] = CollaborationRole.OWNER
            
            logger.info(f"Created workspace: {workspace.id} ({name})")
            return workspace
            
        except Exception as e:
            logger.error(f"Failed to create workspace: {str(e)}")
            raise
    
    async def add_member(
        self, 
        workspace_id: str, 
        user_id: str, 
        role: CollaborationRole,
        inviter_id: str
    ) -> bool:
        """Add a member to workspace."""
        try:
            if workspace_id not in self.workspaces:
                raise ValueError("Workspace not found")
            
            # Check permissions
            if not await self._has_permission(workspace_id, inviter_id, "invite_members"):
                raise PermissionError("Insufficient permissions to invite members")
            
            workspace = self.workspaces[workspace_id]
            
            # Check if user is already a member
            existing_member = next(
                (m for m in workspace.members if m["user_id"] == user_id), 
                None
            )
            
            if existing_member:
                # Update role
                existing_member["role"] = role.value
            else:
                # Add new member
                workspace.members.append({
                    "user_id": user_id,
                    "role": role.value,
                    "joined_at": datetime.utcnow().isoformat(),
                    "invited_by": inviter_id
                })
            
            self.workspace_permissions[workspace_id][user_id] = role
            workspace.updated_at = datetime.utcnow()
            
            logger.info(f"Added member {user_id} to workspace {workspace_id} with role {role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add member: {str(e)}")
            return False
    
    async def get_user_workspaces(self, user_id: str) -> List[Workspace]:
        """Get workspaces accessible to user."""
        user_workspaces = []
        
        for workspace in self.workspaces.values():
            if any(member["user_id"] == user_id for member in workspace.members):
                user_workspaces.append(workspace)
            elif workspace.workspace_type == WorkspaceType.PUBLIC:
                user_workspaces.append(workspace)
        
        return user_workspaces
    
    async def _has_permission(
        self, 
        workspace_id: str, 
        user_id: str, 
        action: str
    ) -> bool:
        """Check if user has permission for action."""
        if workspace_id not in self.workspace_permissions:
            return False
        
        user_role = self.workspace_permissions[workspace_id].get(user_id)
        if not user_role:
            return False
        
        # Define permission matrix
        permissions = {
            CollaborationRole.OWNER: ["*"],  # All permissions
            CollaborationRole.ADMIN: [
                "invite_members", "remove_members", "manage_knowledge_bases",
                "approve_contributions", "moderate_content"
            ],
            CollaborationRole.EDITOR: [
                "create_contributions", "edit_content", "review_contributions"
            ],
            CollaborationRole.CONTRIBUTOR: [
                "create_contributions", "vote_contributions"
            ],
            CollaborationRole.REVIEWER: [
                "review_contributions", "vote_contributions"
            ],
            CollaborationRole.VIEWER: [
                "view_content"
            ]
        }
        
        user_permissions = permissions.get(user_role, [])
        return "*" in user_permissions or action in user_permissions


class ContributionManager:
    """Manages knowledge contributions and curation."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.contributions: Dict[str, KnowledgeContribution] = {}
        self.contribution_queue: Dict[str, List[str]] = defaultdict(list)  # workspace_id -> contribution_ids
    
    async def submit_contribution(
        self, 
        contributor_id: str,
        workspace_id: str,
        contribution_type: ContributionType,
        content: str,
        target_document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeContribution:
        """Submit a knowledge contribution."""
        try:
            contribution = KnowledgeContribution(
                id=str(uuid.uuid4()),
                contributor_id=contributor_id,
                workspace_id=workspace_id,
                contribution_type=contribution_type,
                target_document_id=target_document_id,
                content=content,
                metadata=metadata or {}
            )
            
            self.contributions[contribution.id] = contribution
            self.contribution_queue[workspace_id].append(contribution.id)
            
            # Auto-approve certain types of contributions
            if contribution_type in [ContributionType.RATING, ContributionType.TAG]:
                contribution.status = "approved"
            
            logger.info(f"Submitted contribution: {contribution.id} ({contribution_type.value})")
            return contribution
            
        except Exception as e:
            logger.error(f"Failed to submit contribution: {str(e)}")
            raise
    
    async def vote_contribution(
        self, 
        contribution_id: str, 
        voter_id: str, 
        vote: int
    ) -> bool:
        """Vote on a contribution (1=upvote, 0=neutral, -1=downvote)."""
        try:
            if contribution_id not in self.contributions:
                return False
            
            contribution = self.contributions[contribution_id]
            contribution.votes[voter_id] = vote
            
            # Auto-approve if enough positive votes
            positive_votes = sum(1 for v in contribution.votes.values() if v > 0)
            negative_votes = sum(1 for v in contribution.votes.values() if v < 0)
            
            if positive_votes >= 3 and positive_votes > negative_votes * 2:
                contribution.status = "approved"
            elif negative_votes >= 3 and negative_votes > positive_votes:
                contribution.status = "rejected"
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to vote on contribution: {str(e)}")
            return False
    
    async def get_pending_contributions(self, workspace_id: str) -> List[KnowledgeContribution]:
        """Get pending contributions for workspace."""
        workspace_contributions = self.contribution_queue.get(workspace_id, [])
        return [
            self.contributions[contrib_id] 
            for contrib_id in workspace_contributions
            if self.contributions[contrib_id].status == "pending"
        ]
    
    async def approve_contribution(
        self, 
        contribution_id: str, 
        approver_id: str
    ) -> bool:
        """Approve a contribution."""
        try:
            if contribution_id not in self.contributions:
                return False
            
            contribution = self.contributions[contribution_id]
            contribution.status = "approved"
            contribution.metadata["approved_by"] = approver_id
            contribution.metadata["approved_at"] = datetime.utcnow().isoformat()
            
            # Apply the contribution to knowledge base
            await self._apply_contribution(contribution)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve contribution: {str(e)}")
            return False
    
    async def _apply_contribution(self, contribution: KnowledgeContribution) -> None:
        """Apply approved contribution to knowledge base."""
        # Implementation depends on contribution type
        if contribution.contribution_type == ContributionType.ANNOTATION:
            await self._apply_annotation(contribution)
        elif contribution.contribution_type == ContributionType.CORRECTION:
            await self._apply_correction(contribution)
        elif contribution.contribution_type == ContributionType.ENHANCEMENT:
            await self._apply_enhancement(contribution)
        # Add more contribution types as needed
    
    async def _apply_annotation(self, contribution: KnowledgeContribution) -> None:
        """Apply annotation contribution."""
        # Add annotation to document metadata
        logger.info(f"Applied annotation: {contribution.id}")
    
    async def _apply_correction(self, contribution: KnowledgeContribution) -> None:
        """Apply correction contribution."""
        # Update document content with correction
        logger.info(f"Applied correction: {contribution.id}")
    
    async def _apply_enhancement(self, contribution: KnowledgeContribution) -> None:
        """Apply enhancement contribution."""
        # Add enhancement to document
        logger.info(f"Applied enhancement: {contribution.id}")


class CollaborativeSessionManager:
    """Manages real-time collaborative sessions."""
    
    def __init__(self):
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
    
    async def start_session(
        self, 
        workspace_id: str, 
        initiator_id: str,
        document_id: Optional[str] = None
    ) -> CollaborativeSession:
        """Start a collaborative session."""
        try:
            session = CollaborativeSession(
                id=str(uuid.uuid4()),
                workspace_id=workspace_id,
                participants=[initiator_id],
                active_document_id=document_id,
                session_data={}
            )
            
            self.active_sessions[session.id] = session
            self.user_sessions[initiator_id].add(session.id)
            
            logger.info(f"Started collaborative session: {session.id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to start session: {str(e)}")
            raise
    
    async def join_session(self, session_id: str, user_id: str) -> bool:
        """Join a collaborative session."""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            if user_id not in session.participants:
                session.participants.append(user_id)
                self.user_sessions[user_id].add(session_id)
                session.last_activity = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to join session: {str(e)}")
            return False
    
    async def update_session_data(
        self, 
        session_id: str, 
        user_id: str, 
        data: Dict[str, Any]
    ) -> bool:
        """Update session data."""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            if user_id not in session.participants:
                return False
            
            session.session_data.update(data)
            session.last_activity = datetime.utcnow()
            
            # Broadcast update to other participants
            await self._broadcast_update(session_id, user_id, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session data: {str(e)}")
            return False
    
    async def _broadcast_update(
        self, 
        session_id: str, 
        sender_id: str, 
        data: Dict[str, Any]
    ) -> None:
        """Broadcast update to session participants."""
        # In a real implementation, this would use WebSockets or similar
        logger.info(f"Broadcasting update from {sender_id} in session {session_id}")


class CollaborativeIntelligenceEngine:
    """Generates collaborative intelligence insights."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
    
    async def generate_insights(
        self, 
        workspace_id: str,
        contributions: List[KnowledgeContribution],
        workspace_activity: Dict[str, Any]
    ) -> CollaborativeIntelligence:
        """Generate collaborative intelligence insights."""
        try:
            # Calculate knowledge quality score
            quality_score = await self._calculate_knowledge_quality(contributions)
            
            # Analyze collaboration activity
            activity_analysis = await self._analyze_collaboration_activity(workspace_activity)
            
            # Identify contribution patterns
            contribution_patterns = await self._analyze_contribution_patterns(contributions)
            
            # Identify knowledge gaps
            knowledge_gaps = await self._identify_knowledge_gaps(contributions)
            
            # Find trending topics
            trending_topics = await self._find_trending_topics(contributions)
            
            # Generate expert recommendations
            expert_recommendations = await self._generate_expert_recommendations(
                contributions, workspace_activity
            )
            
            insights = CollaborativeIntelligence(
                workspace_id=workspace_id,
                knowledge_quality_score=quality_score,
                collaboration_activity=activity_analysis,
                contribution_patterns=contribution_patterns,
                knowledge_gaps=knowledge_gaps,
                trending_topics=trending_topics,
                expert_recommendations=expert_recommendations
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            raise
    
    async def _calculate_knowledge_quality(
        self, 
        contributions: List[KnowledgeContribution]
    ) -> float:
        """Calculate overall knowledge quality score."""
        if not contributions:
            return 0.5
        
        # Simple quality calculation based on votes and approval rate
        total_score = 0
        total_contributions = len(contributions)
        
        for contribution in contributions:
            if contribution.votes:
                positive_votes = sum(1 for v in contribution.votes.values() if v > 0)
                total_votes = len(contribution.votes)
                vote_ratio = positive_votes / total_votes if total_votes > 0 else 0.5
            else:
                vote_ratio = 0.5
            
            approval_bonus = 0.2 if contribution.status == "approved" else 0
            total_score += vote_ratio + approval_bonus
        
        return min(1.0, total_score / total_contributions)
    
    async def _analyze_collaboration_activity(
        self, 
        workspace_activity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze collaboration activity patterns."""
        return {
            "active_contributors": workspace_activity.get("active_contributors", 0),
            "contributions_per_day": workspace_activity.get("contributions_per_day", 0),
            "collaboration_frequency": workspace_activity.get("collaboration_frequency", "low"),
            "peak_activity_hours": workspace_activity.get("peak_activity_hours", [])
        }
    
    async def _analyze_contribution_patterns(
        self, 
        contributions: List[KnowledgeContribution]
    ) -> Dict[str, Any]:
        """Analyze contribution patterns."""
        if not contributions:
            return {}
        
        # Analyze contribution types
        type_counts = defaultdict(int)
        for contribution in contributions:
            type_counts[contribution.contribution_type.value] += 1
        
        # Analyze contributor activity
        contributor_counts = defaultdict(int)
        for contribution in contributions:
            contributor_counts[contribution.contributor_id] += 1
        
        return {
            "contribution_types": dict(type_counts),
            "top_contributors": dict(sorted(
                contributor_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]),
            "total_contributions": len(contributions)
        }
    
    async def _identify_knowledge_gaps(
        self, 
        contributions: List[KnowledgeContribution]
    ) -> List[str]:
        """Identify knowledge gaps based on contributions."""
        # Simple gap identification based on rejected contributions
        gaps = []
        
        for contribution in contributions:
            if contribution.status == "rejected":
                # Analyze why it was rejected and identify gaps
                gaps.append(f"Gap in {contribution.contribution_type.value} content")
        
        return list(set(gaps))
    
    async def _find_trending_topics(
        self, 
        contributions: List[KnowledgeContribution]
    ) -> List[str]:
        """Find trending topics in contributions."""
        # Simple topic extraction from recent contributions
        recent_contributions = [
            c for c in contributions 
            if c.created_at and (datetime.utcnow() - c.created_at).days <= 7
        ]
        
        # Extract keywords from content
        topics = []
        for contribution in recent_contributions:
            words = contribution.content.lower().split()
            # Simple keyword extraction
            keywords = [word for word in words if len(word) > 5]
            topics.extend(keywords[:3])  # Top 3 keywords per contribution
        
        # Count frequency and return top topics
        topic_counts = defaultdict(int)
        for topic in topics:
            topic_counts[topic] += 1
        
        return [topic for topic, _ in sorted(
            topic_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]]
    
    async def _generate_expert_recommendations(
        self, 
        contributions: List[KnowledgeContribution],
        workspace_activity: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate expert recommendations."""
        recommendations = []
        
        # Recommend active contributors as experts
        contributor_scores = defaultdict(float)
        for contribution in contributions:
            if contribution.status == "approved":
                contributor_scores[contribution.contributor_id] += 1.0
            
            # Add vote-based scoring
            if contribution.votes:
                positive_votes = sum(1 for v in contribution.votes.values() if v > 0)
                contributor_scores[contribution.contributor_id] += positive_votes * 0.1
        
        # Generate recommendations
        for contributor_id, score in sorted(
            contributor_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]:
            recommendations.append({
                "user_id": contributor_id,
                "expertise_score": score,
                "recommendation_type": "active_contributor",
                "reason": f"High-quality contributions with score {score:.1f}"
            })
        
        return recommendations


class CollaborationManager:
    """
    Revolutionary collaboration manager for RAG 4.0.
    
    Features:
    - Shared workspaces and team collaboration
    - Knowledge contribution and curation system
    - Real-time collaborative sessions
    - Collaborative intelligence insights
    - Community-driven knowledge enhancement
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        
        # Component managers
        self.workspace_manager = WorkspaceManager(embedding_manager)
        self.contribution_manager = ContributionManager(embedding_manager)
        self.session_manager = CollaborativeSessionManager()
        self.intelligence_engine = CollaborativeIntelligenceEngine(embedding_manager)
        
        # Cache and resilience
        self.cache = None
        self.resilience_manager = None
    
    async def initialize(self) -> None:
        """Initialize the collaboration manager."""
        try:
            self.cache = await get_rag_cache()
            self.resilience_manager = await get_resilience_manager()
            
            await self.resilience_manager.register_component(
                "collaboration_manager",
                recovery_strategies=["retry", "graceful_degradation"]
            )
            
            logger.info("Collaboration manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize collaboration manager: {str(e)}")
            raise
    
    async def get_collaboration_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get collaboration statistics for workspace."""
        try:
            workspace = self.workspace_manager.workspaces.get(workspace_id)
            if not workspace:
                return {}
            
            contributions = [
                c for c in self.contribution_manager.contributions.values()
                if c.workspace_id == workspace_id
            ]
            
            active_sessions = [
                s for s in self.session_manager.active_sessions.values()
                if s.workspace_id == workspace_id
            ]
            
            stats = {
                "workspace_info": asdict(workspace),
                "total_members": len(workspace.members),
                "total_contributions": len(contributions),
                "pending_contributions": len([c for c in contributions if c.status == "pending"]),
                "approved_contributions": len([c for c in contributions if c.status == "approved"]),
                "active_sessions": len(active_sessions),
                "contribution_types": {},
                "recent_activity": []
            }
            
            # Contribution type breakdown
            for contribution in contributions:
                contrib_type = contribution.contribution_type.value
                stats["contribution_types"][contrib_type] = stats["contribution_types"].get(contrib_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collaboration stats: {str(e)}")
            return {}


# Global collaboration manager instance
collaboration_manager = None


async def get_collaboration_manager(embedding_manager: EmbeddingManager) -> CollaborationManager:
    """Get the global collaboration manager instance."""
    global collaboration_manager
    
    if collaboration_manager is None:
        collaboration_manager = CollaborationManager(embedding_manager)
        await collaboration_manager.initialize()
    
    return collaboration_manager
