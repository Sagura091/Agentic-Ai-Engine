"""
Knowledge Sharing Protocols - Streamlined Foundation.

This module provides simple knowledge sharing mechanisms for multi-agent systems.

Features:
- Basic knowledge sharing protocols
- Simple permission management
- Knowledge sharing requests
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import structlog

from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.rag.core.collection_based_kb_manager import CollectionBasedKBManager, AccessLevel
from .agent_communication_system import AgentCommunicationSystem, MessageType

logger = structlog.get_logger(__name__)


class KnowledgeShareType(str, Enum):
    """Types of knowledge sharing."""
    DOCUMENT = "document"             # Share specific documents
    COLLECTION = "collection"         # Share entire knowledge base


class SharingPermission(str, Enum):
    """Permission levels for knowledge sharing."""
    READ_ONLY = "read_only"           # Can only read shared knowledge
    READ_WRITE = "read_write"         # Can read and modify shared knowledge


class RequestStatus(str, Enum):
    """Status of sharing requests."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


@dataclass
class SharingRequest:
    """Simple knowledge sharing request structure."""
    request_id: str
    requester_id: str
    owner_id: str
    knowledge_type: KnowledgeShareType
    resource_id: str
    requested_permission: SharingPermission
    created_at: datetime
    status: RequestStatus = RequestStatus.PENDING

    @classmethod
    def create(
        cls,
        requester_id: str,
        owner_id: str,
        knowledge_type: KnowledgeShareType,
        resource_id: str,
        requested_permission: SharingPermission = SharingPermission.READ_ONLY
    ) -> "SharingRequest":
        """Create a new sharing request."""
        return cls(
            request_id=str(uuid.uuid4()),
            requester_id=requester_id,
            owner_id=owner_id,
            knowledge_type=knowledge_type,
            resource_id=resource_id,
            requested_permission=requested_permission,
            created_at=datetime.now()
        )


@dataclass
class SharingResponse:
    """Simple response to a knowledge sharing request."""
    response_id: str
    request_id: str
    responder_id: str
    approved: bool
    granted_permission: Optional[SharingPermission]
    message: str
    created_at: datetime

    @classmethod
    def create(
        cls,
        request_id: str,
        responder_id: str,
        approved: bool,
        message: str = "",
        granted_permission: Optional[SharingPermission] = None
    ) -> "SharingResponse":
        """Create a new sharing response."""
        return cls(
            response_id=str(uuid.uuid4()),
            request_id=request_id,
            responder_id=responder_id,
            approved=approved,
            granted_permission=granted_permission,
            message=message,
            created_at=datetime.now()
        )


class KnowledgeSharingProtocol:
    """
    Knowledge Sharing Protocol Manager - Streamlined Foundation.

    Manages simple knowledge sharing between agents.
    """

    def __init__(
        self,
        unified_rag: UnifiedRAGSystem,
        kb_manager: CollectionBasedKBManager,
        communication_system: AgentCommunicationSystem
    ):
        """Initialize the knowledge sharing protocol."""
        self.unified_rag = unified_rag
        self.kb_manager = kb_manager
        self.communication_system = communication_system
        self.is_initialized = False

        # Request management
        self.sharing_requests: Dict[str, SharingRequest] = {}
        self.sharing_responses: Dict[str, SharingResponse] = {}

        # Permission tracking
        self.agent_permissions: Dict[str, Dict[str, SharingPermission]] = {}

        # Simple stats
        self.stats = {
            "total_requests": 0,
            "approved_requests": 0,
            "denied_requests": 0
        }

        logger.info("Knowledge sharing protocol initialized")

    async def initialize(self) -> None:
        """Initialize the knowledge sharing protocol."""
        try:
            if self.is_initialized:
                return

            self.is_initialized = True
            logger.info("Knowledge sharing protocol initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize knowledge sharing protocol: {str(e)}")
            raise
    async def request_knowledge_access(
        self,
        requester_id: str,
        owner_id: str,
        knowledge_type: KnowledgeShareType,
        resource_id: str,
        requested_permission: SharingPermission = SharingPermission.READ_ONLY
    ) -> str:
        """
        Request access to another agent's knowledge.

        Args:
            requester_id: Agent requesting access
            owner_id: Agent who owns the knowledge
            knowledge_type: Type of knowledge being requested
            resource_id: ID of the specific resource
            requested_permission: Level of access requested

        Returns:
            Request ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Create sharing request
            request = SharingRequest.create(
                requester_id=requester_id,
                owner_id=owner_id,
                knowledge_type=knowledge_type,
                resource_id=resource_id,
                requested_permission=requested_permission
            )

            # Store request
            self.sharing_requests[request.request_id] = request

            # Send notification to owner
            await self.communication_system.send_message(
                sender_id="system",
                recipient_id=owner_id,
                content=f"Knowledge sharing request from {requester_id} for resource {resource_id}",
                message_type=MessageType.SYSTEM
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
        message: str = "",
        granted_permission: Optional[SharingPermission] = None
    ) -> str:
        """
        Respond to a knowledge sharing request.

        Args:
            request_id: ID of the request to respond to
            responder_id: Agent responding to the request
            approved: Whether the request is approved
            message: Response message
            granted_permission: Permission level granted (if approved)

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
                raise PermissionError(f"Only the owner can respond to this request")

            # Create response
            response = SharingResponse.create(
                request_id=request_id,
                responder_id=responder_id,
                approved=approved,
                message=message,
                granted_permission=granted_permission if approved else None
            )

            # Update request status
            if approved:
                request.status = RequestStatus.APPROVED
                self.stats["approved_requests"] += 1
            else:
                request.status = RequestStatus.DENIED
                self.stats["denied_requests"] += 1

            # Store response
            self.sharing_responses[response.response_id] = response

            # Notify requester
            await self.communication_system.send_message(
                sender_id="system",
                recipient_id=request.requester_id,
                content=f"Knowledge sharing request {'approved' if approved else 'denied'}",
                message_type=MessageType.SYSTEM
            )

            logger.info(f"Knowledge sharing request {request_id} {'approved' if approved else 'denied'}")
            return response.response_id

        except Exception as e:
            logger.error(f"Failed to respond to sharing request: {str(e)}")
            raise
    def get_request(self, request_id: str) -> Optional[SharingRequest]:
        """Get a sharing request by ID."""
        return self.sharing_requests.get(request_id)

    def get_response(self, response_id: str) -> Optional[SharingResponse]:
        """Get a sharing response by ID."""
        return self.sharing_responses.get(response_id)

    def get_agent_requests(self, agent_id: str, as_requester: bool = True) -> List[SharingRequest]:
        """Get sharing requests for an agent."""
        requests = []

        for request in self.sharing_requests.values():
            if as_requester and request.requester_id == agent_id:
                requests.append(request)
            elif not as_requester and request.owner_id == agent_id:
                requests.append(request)

        return requests

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge sharing protocol statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "pending_requests": sum(1 for r in self.sharing_requests.values() if r.status == RequestStatus.PENDING),
            "total_responses": len(self.sharing_responses)
        }
