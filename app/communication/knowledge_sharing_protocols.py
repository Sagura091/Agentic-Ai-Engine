"""
Knowledge Sharing Protocols for Multi-Agent Systems.

This module provides protocols and mechanisms for secure and efficient
knowledge sharing between agents, including permission management,
knowledge validation, and sharing analytics.

Features:
- Secure knowledge sharing protocols
- Permission-based access control
- Knowledge validation and verification
- Sharing request/response workflows
- Knowledge sharing analytics
- Automated knowledge discovery
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.rag.core.collection_based_kb_manager import CollectionBasedKBManager, AccessLevel
from app.rag.core.agent_isolation_manager import AgentIsolationManager, ResourceType
from .agent_communication_system import AgentCommunicationSystem, MessageType

logger = structlog.get_logger(__name__)


class KnowledgeShareType(str, Enum):
    """Types of knowledge sharing."""
    DOCUMENT = "document"             # Share specific documents
    COLLECTION = "collection"         # Share entire knowledge base
    MEMORY = "memory"                 # Share memory entries
    INSIGHT = "insight"               # Share derived insights
    SKILL = "skill"                   # Share procedural knowledge


class SharingPermission(str, Enum):
    """Permission levels for knowledge sharing."""
    READ_ONLY = "read_only"           # Can only read shared knowledge
    READ_WRITE = "read_write"         # Can read and modify shared knowledge
    FULL_ACCESS = "full_access"       # Full access including sharing rights


class RequestStatus(str, Enum):
    """Status of sharing requests."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class SharingRequest:
    """Knowledge sharing request structure."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester_id: str = ""
    owner_id: str = ""
    
    # What is being requested
    knowledge_type: KnowledgeShareType = KnowledgeShareType.DOCUMENT
    resource_id: str = ""  # Document ID, collection ID, etc.
    resource_name: str = ""
    
    # Access details
    requested_permission: SharingPermission = SharingPermission.READ_ONLY
    justification: str = ""
    intended_use: str = ""
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    
    # Status
    status: RequestStatus = RequestStatus.PENDING
    response_message: str = ""
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharingResponse:
    """Response to a knowledge sharing request."""
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    responder_id: str = ""
    
    # Response details
    approved: bool = False
    granted_permission: Optional[SharingPermission] = None
    conditions: List[str] = field(default_factory=list)
    expiration_date: Optional[datetime] = None
    
    # Response content
    message: str = ""
    shared_resources: List[str] = field(default_factory=list)  # Resource IDs
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeSharingProtocol:
    """
    Knowledge Sharing Protocol Manager.
    
    Manages secure and efficient knowledge sharing between agents
    with proper access controls and validation.
    """
    
    def __init__(
        self,
        unified_rag: UnifiedRAGSystem,
        kb_manager: CollectionBasedKBManager,
        isolation_manager: AgentIsolationManager,
        communication_system: AgentCommunicationSystem
    ):
        """Initialize the knowledge sharing protocol."""
        self.unified_rag = unified_rag
        self.kb_manager = kb_manager
        self.isolation_manager = isolation_manager
        self.communication_system = communication_system
        
        # Request management
        self.sharing_requests: Dict[str, SharingRequest] = {}
        self.sharing_responses: Dict[str, SharingResponse] = {}
        
        # Active shares
        self.active_shares: Dict[str, Dict[str, Any]] = {}  # share_id -> share_info
        
        # Permission tracking
        self.agent_permissions: Dict[str, Dict[str, SharingPermission]] = {}  # owner_id -> {requester_id -> permission}
        
        # Analytics
        self.stats = {
            "total_requests": 0,
            "approved_requests": 0,
            "denied_requests": 0,
            "active_shares": 0,
            "knowledge_transfers": 0
        }
        
        logger.info("Knowledge sharing protocol initialized")
    
    async def request_knowledge_access(
        self,
        requester_id: str,
        owner_id: str,
        knowledge_type: KnowledgeShareType,
        resource_id: str,
        resource_name: str,
        requested_permission: SharingPermission = SharingPermission.READ_ONLY,
        justification: str = "",
        intended_use: str = "",
        expires_in_hours: int = 24
    ) -> str:
        """
        Request access to another agent's knowledge.
        
        Args:
            requester_id: Agent requesting access
            owner_id: Agent who owns the knowledge
            knowledge_type: Type of knowledge being requested
            resource_id: ID of the specific resource
            resource_name: Name of the resource
            requested_permission: Level of access requested
            justification: Reason for the request
            intended_use: How the knowledge will be used
            expires_in_hours: Request expiration time
            
        Returns:
            Request ID
        """
        try:
            # Validate agents exist
            if not await self._validate_agent_exists(requester_id):
                raise ValueError(f"Requester agent {requester_id} not found")
            
            if not await self._validate_agent_exists(owner_id):
                raise ValueError(f"Owner agent {owner_id} not found")
            
            # Check if requester can make sharing requests
            if not await self.isolation_manager.validate_access(
                requester_id,
                owner_id,
                ResourceType.KNOWLEDGE,
                "read"
            ):
                raise PermissionError(f"Agent {requester_id} cannot request knowledge from {owner_id}")
            
            # Create sharing request
            request = SharingRequest(
                requester_id=requester_id,
                owner_id=owner_id,
                knowledge_type=knowledge_type,
                resource_id=resource_id,
                resource_name=resource_name,
                requested_permission=requested_permission,
                justification=justification,
                intended_use=intended_use,
                expires_at=datetime.utcnow() + timedelta(hours=expires_in_hours)
            )
            
            # Store request
            self.sharing_requests[request.request_id] = request
            
            # Send notification to owner
            await self.communication_system.send_message(
                sender_id="system",
                recipient_id=owner_id,
                content=f"Knowledge sharing request from {requester_id} for '{resource_name}'",
                message_type=MessageType.KNOWLEDGE_SHARE,
                subject="Knowledge Sharing Request",
                metadata={
                    "request_id": request.request_id,
                    "requester_id": requester_id,
                    "resource_name": resource_name,
                    "requested_permission": requested_permission.value
                }
            )
            
            self.stats["total_requests"] += 1
            
            logger.info(f"Knowledge sharing request created: {request.request_id}")
            return request.request_id
            
        except Exception as e:
            logger.error(f"Failed to create knowledge sharing request: {str(e)}")
            raise
    
    async def respond_to_request(
        self,
        request_id: str,
        responder_id: str,
        approved: bool,
        granted_permission: Optional[SharingPermission] = None,
        conditions: Optional[List[str]] = None,
        message: str = "",
        expiration_hours: Optional[int] = None
    ) -> str:
        """
        Respond to a knowledge sharing request.
        
        Args:
            request_id: ID of the request to respond to
            responder_id: Agent responding to the request
            approved: Whether the request is approved
            granted_permission: Permission level granted (if approved)
            conditions: Any conditions for the sharing
            message: Response message
            expiration_hours: Hours until the share expires
            
        Returns:
            Response ID
        """
        try:
            # Get the request
            if request_id not in self.sharing_requests:
                raise ValueError(f"Sharing request {request_id} not found")
            
            request = self.sharing_requests[request_id]
            
            # Validate responder is the owner
            if responder_id != request.owner_id:
                raise PermissionError(f"Only the owner ({request.owner_id}) can respond to this request")
            
            # Check if request is still valid
            if request.status != RequestStatus.PENDING:
                raise ValueError(f"Request {request_id} is no longer pending")
            
            if request.expires_at and request.expires_at < datetime.utcnow():
                request.status = RequestStatus.EXPIRED
                raise ValueError(f"Request {request_id} has expired")
            
            # Create response
            response = SharingResponse(
                request_id=request_id,
                responder_id=responder_id,
                approved=approved,
                granted_permission=granted_permission if approved else None,
                conditions=conditions or [],
                message=message
            )
            
            if approved and expiration_hours:
                response.expiration_date = datetime.utcnow() + timedelta(hours=expiration_hours)
            
            # Update request status
            if approved:
                request.status = RequestStatus.APPROVED
                request.approved_at = datetime.utcnow()
                self.stats["approved_requests"] += 1
                
                # Create active share
                await self._create_active_share(request, response)
                
            else:
                request.status = RequestStatus.DENIED
                self.stats["denied_requests"] += 1
            
            request.response_message = message
            
            # Store response
            self.sharing_responses[response.response_id] = response
            
            # Notify requester
            await self.communication_system.send_message(
                sender_id="system",
                recipient_id=request.requester_id,
                content=f"Knowledge sharing request {'approved' if approved else 'denied'}: {request.resource_name}",
                message_type=MessageType.KNOWLEDGE_SHARE,
                subject="Knowledge Sharing Response",
                metadata={
                    "request_id": request_id,
                    "response_id": response.response_id,
                    "approved": approved,
                    "granted_permission": granted_permission.value if granted_permission else None
                }
            )
            
            logger.info(f"Knowledge sharing request {request_id} {'approved' if approved else 'denied'}")
            return response.response_id
            
        except Exception as e:
            logger.error(f"Failed to respond to sharing request: {str(e)}")
            raise
    
    async def access_shared_knowledge(
        self,
        requester_id: str,
        share_id: str,
        operation: str = "read"
    ) -> Any:
        """
        Access shared knowledge.
        
        Args:
            requester_id: Agent accessing the knowledge
            share_id: ID of the active share
            operation: Type of operation (read, write)
            
        Returns:
            Knowledge content or operation result
        """
        try:
            # Get active share
            if share_id not in self.active_shares:
                raise ValueError(f"Active share {share_id} not found")
            
            share_info = self.active_shares[share_id]
            
            # Validate access
            if requester_id != share_info["requester_id"]:
                raise PermissionError(f"Agent {requester_id} does not have access to share {share_id}")
            
            # Check expiration
            if share_info.get("expires_at") and share_info["expires_at"] < datetime.utcnow():
                await self._revoke_share(share_id)
                raise PermissionError(f"Share {share_id} has expired")
            
            # Check permission level
            granted_permission = SharingPermission(share_info["granted_permission"])
            if operation == "write" and granted_permission == SharingPermission.READ_ONLY:
                raise PermissionError(f"Write access not granted for share {share_id}")
            
            # Access the knowledge based on type
            knowledge_type = KnowledgeShareType(share_info["knowledge_type"])
            resource_id = share_info["resource_id"]
            
            if knowledge_type == KnowledgeShareType.DOCUMENT:
                # Access specific document
                result = await self._access_shared_document(requester_id, resource_id, operation)
            
            elif knowledge_type == KnowledgeShareType.COLLECTION:
                # Access knowledge base
                result = await self._access_shared_collection(requester_id, resource_id, operation)
            
            elif knowledge_type == KnowledgeShareType.MEMORY:
                # Access memory entries
                result = await self._access_shared_memory(requester_id, resource_id, operation)
            
            else:
                raise ValueError(f"Unsupported knowledge type: {knowledge_type}")
            
            # Update access statistics
            share_info["access_count"] = share_info.get("access_count", 0) + 1
            share_info["last_accessed"] = datetime.utcnow()
            
            self.stats["knowledge_transfers"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to access shared knowledge: {str(e)}")
            raise
    
    async def _create_active_share(self, request: SharingRequest, response: SharingResponse) -> str:
        """Create an active knowledge share."""
        try:
            share_id = f"share_{request.request_id}"
            
            share_info = {
                "share_id": share_id,
                "request_id": request.request_id,
                "response_id": response.response_id,
                "requester_id": request.requester_id,
                "owner_id": request.owner_id,
                "knowledge_type": request.knowledge_type.value,
                "resource_id": request.resource_id,
                "resource_name": request.resource_name,
                "granted_permission": response.granted_permission.value,
                "conditions": response.conditions,
                "created_at": datetime.utcnow(),
                "expires_at": response.expiration_date,
                "access_count": 0,
                "last_accessed": None
            }
            
            self.active_shares[share_id] = share_info
            self.stats["active_shares"] += 1
            
            # Grant access permissions in isolation manager
            await self.isolation_manager.grant_access(
                request.owner_id,
                request.requester_id,
                [ResourceType.KNOWLEDGE]
            )
            
            logger.info(f"Created active share: {share_id}")
            return share_id
            
        except Exception as e:
            logger.error(f"Failed to create active share: {str(e)}")
            raise
    
    async def _access_shared_document(self, requester_id: str, resource_id: str, operation: str) -> Any:
        """Access a shared document."""
        try:
            # This would integrate with the knowledge base manager
            # For now, return placeholder
            return {
                "type": "document",
                "resource_id": resource_id,
                "operation": operation,
                "content": "Shared document content"
            }
            
        except Exception as e:
            logger.error(f"Failed to access shared document: {str(e)}")
            raise
    
    async def _access_shared_collection(self, requester_id: str, resource_id: str, operation: str) -> Any:
        """Access a shared knowledge collection."""
        try:
            # This would integrate with the collection-based KB manager
            # For now, return placeholder
            return {
                "type": "collection",
                "resource_id": resource_id,
                "operation": operation,
                "content": "Shared collection content"
            }
            
        except Exception as e:
            logger.error(f"Failed to access shared collection: {str(e)}")
            raise
    
    async def _access_shared_memory(self, requester_id: str, resource_id: str, operation: str) -> Any:
        """Access shared memory entries."""
        try:
            # This would integrate with the unified memory system
            # For now, return placeholder
            return {
                "type": "memory",
                "resource_id": resource_id,
                "operation": operation,
                "content": "Shared memory content"
            }
            
        except Exception as e:
            logger.error(f"Failed to access shared memory: {str(e)}")
            raise
    
    async def _revoke_share(self, share_id: str) -> None:
        """Revoke an active share."""
        try:
            if share_id in self.active_shares:
                del self.active_shares[share_id]
                self.stats["active_shares"] -= 1
                logger.info(f"Revoked share: {share_id}")
                
        except Exception as e:
            logger.error(f"Failed to revoke share: {str(e)}")
    
    async def _validate_agent_exists(self, agent_id: str) -> bool:
        """Validate that an agent exists in the system."""
        try:
            # Check if agent has isolation profile
            profile = self.isolation_manager.get_agent_profile(agent_id)
            return profile is not None
            
        except Exception as e:
            logger.error(f"Failed to validate agent existence: {str(e)}")
            return False
    
    def get_agent_requests(self, agent_id: str, as_requester: bool = True) -> List[SharingRequest]:
        """Get sharing requests for an agent."""
        requests = []
        
        for request in self.sharing_requests.values():
            if as_requester and request.requester_id == agent_id:
                requests.append(request)
            elif not as_requester and request.owner_id == agent_id:
                requests.append(request)
        
        return requests
    
    def get_active_shares_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get active shares for an agent."""
        shares = []
        
        for share_info in self.active_shares.values():
            if (share_info["requester_id"] == agent_id or 
                share_info["owner_id"] == agent_id):
                shares.append(share_info)
        
        return shares
    
    def get_protocol_stats(self) -> Dict[str, Any]:
        """Get knowledge sharing protocol statistics."""
        return {
            **self.stats,
            "pending_requests": sum(1 for r in self.sharing_requests.values() if r.status == RequestStatus.PENDING),
            "total_responses": len(self.sharing_responses)
        }
