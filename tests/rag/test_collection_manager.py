"""
Test Suite for Hierarchical Collection Manager.

Tests the revolutionary collection management system including:
- Collection lifecycle management
- Permission-based access control
- Hierarchical relationships
- Automatic cleanup and archiving
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone

from app.rag.core.collection_manager import (
    CollectionManager,
    CollectionType,
    CollectionLifecycleState,
    KnowledgeCollectionMetadata
)
from app.rag.core.agent_knowledge_manager import KnowledgeScope, KnowledgePermission


class TestCollectionManager:
    """Test suite for the hierarchical collection manager."""
    
    @pytest.mark.asyncio
    async def test_collection_creation(self, collection_manager):
        """Test creating collections with different types."""
        manager = collection_manager
        
        # Test creating agent private collection
        private_metadata = await manager.create_collection(
            collection_name="agent_test_001_private",
            collection_type=CollectionType.AGENT_PRIVATE,
            owner_agent_id="test_001",
            description="Test agent's private knowledge",
            tags=["test", "private"]
        )
        
        assert private_metadata.collection_name == "agent_test_001_private"
        assert private_metadata.scope == KnowledgeScope.PRIVATE
        assert private_metadata.owner_agent_id == "test_001"
        assert "test_001" in private_metadata.allowed_agents
        assert private_metadata.auto_cleanup is True
        
        # Test creating global collection
        global_metadata = await manager.create_collection(
            collection_name="global_test_knowledge",
            collection_type=CollectionType.GLOBAL,
            description="Global test knowledge base"
        )
        
        assert global_metadata.scope == KnowledgeScope.GLOBAL
        assert global_metadata.auto_cleanup is False
        assert global_metadata.retention_days is None
        
        # Verify collections are tracked
        assert "agent_test_001_private" in manager.collections
        assert "global_test_knowledge" in manager.collections
        assert len(manager.collections) >= 8  # 6 default + 2 created
    
    @pytest.mark.asyncio
    async def test_collection_permissions(self, collection_manager):
        """Test permission-based access control."""
        manager = collection_manager
        
        # Create collection with specific owner
        await manager.create_collection(
            collection_name="restricted_collection",
            collection_type=CollectionType.AGENT_PRIVATE,
            owner_agent_id="owner_agent"
        )
        
        # Test owner access
        owner_has_read = await manager._has_collection_access(
            "owner_agent", "restricted_collection", KnowledgePermission.READ
        )
        owner_has_write = await manager._has_collection_access(
            "owner_agent", "restricted_collection", KnowledgePermission.WRITE
        )
        owner_has_delete = await manager._has_collection_access(
            "owner_agent", "restricted_collection", KnowledgePermission.DELETE
        )
        
        assert owner_has_read is True
        assert owner_has_write is True
        assert owner_has_delete is True
        
        # Test non-owner access
        other_has_read = await manager._has_collection_access(
            "other_agent", "restricted_collection", KnowledgePermission.READ
        )
        other_has_write = await manager._has_collection_access(
            "other_agent", "restricted_collection", KnowledgePermission.WRITE
        )
        
        assert other_has_read is False
        assert other_has_write is False
        
        # Test global collection access
        global_read = await manager._has_collection_access(
            "any_agent", "global_knowledge", KnowledgePermission.READ
        )
        global_write = await manager._has_collection_access(
            "any_agent", "global_knowledge", KnowledgePermission.WRITE
        )
        
        assert global_read is True
        assert global_write is False
    
    @pytest.mark.asyncio
    async def test_agent_collections_retrieval(self, collection_manager):
        """Test retrieving collections accessible to an agent."""
        manager = collection_manager
        
        # Create collections for different agents
        await manager.create_collection(
            "agent_alpha_private",
            CollectionType.AGENT_PRIVATE,
            owner_agent_id="agent_alpha"
        )
        
        await manager.create_collection(
            "agent_beta_private", 
            CollectionType.AGENT_PRIVATE,
            owner_agent_id="agent_beta"
        )
        
        await manager.create_collection(
            "shared_research",
            CollectionType.SHARED,
            description="Shared research collection"
        )
        
        # Get collections for agent_alpha
        alpha_collections = await manager.get_agent_collections("agent_alpha")
        
        # Should include: own private + global collections
        alpha_collection_names = [c.collection_name for c in alpha_collections]
        
        assert "agent_alpha_private" in alpha_collection_names
        assert "agent_beta_private" not in alpha_collection_names
        assert "global_knowledge" in alpha_collection_names
        
        # Verify agent has appropriate access
        assert len(alpha_collections) >= 7  # 6 global + 1 private
    
    @pytest.mark.asyncio
    async def test_collection_lifecycle_management(self, collection_manager):
        """Test automatic collection lifecycle management."""
        manager = collection_manager
        
        # Create a session collection with short retention
        session_metadata = await manager.create_collection(
            "test_session_collection",
            CollectionType.SESSION,
            custom_config={"retention_days": 1}
        )
        
        # Simulate collection being old
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=2)
        manager.collections["test_session_collection"].last_updated = old_timestamp
        
        # Run cleanup
        cleanup_stats = await manager.cleanup_expired_collections()
        
        # Verify collection was archived
        assert cleanup_stats["collections_archived"] >= 1
        assert manager.lifecycle_states["test_session_collection"] == CollectionLifecycleState.ARCHIVED
        
        # Simulate grace period expiry
        grace_expiry = old_timestamp - timedelta(days=8)
        manager.collections["test_session_collection"].last_updated = grace_expiry
        manager.lifecycle_states["test_session_collection"] = CollectionLifecycleState.ARCHIVED
        
        # Run cleanup again
        cleanup_stats_2 = await manager.cleanup_expired_collections()
        
        # Verify collection was deleted
        assert cleanup_stats_2["collections_deleted"] >= 1
        assert "test_session_collection" not in manager.collections
    
    @pytest.mark.asyncio
    async def test_collection_templates(self, collection_manager):
        """Test collection creation with different templates."""
        manager = collection_manager
        
        # Test each collection type template
        test_cases = [
            (CollectionType.GLOBAL, KnowledgeScope.GLOBAL, False, None),
            (CollectionType.DOMAIN, KnowledgeScope.DOMAIN, True, 365),
            (CollectionType.AGENT_PRIVATE, KnowledgeScope.PRIVATE, True, 30),
            (CollectionType.AGENT_MEMORY, KnowledgeScope.PRIVATE, True, 30),
            (CollectionType.SESSION, KnowledgeScope.SESSION, True, 1)
        ]
        
        for i, (ctype, expected_scope, expected_cleanup, expected_retention) in enumerate(test_cases):
            collection_name = f"test_template_{i}"
            
            metadata = await manager.create_collection(
                collection_name,
                ctype,
                owner_agent_id="test_agent" if ctype in [CollectionType.AGENT_PRIVATE, CollectionType.AGENT_MEMORY] else None
            )
            
            assert metadata.scope == expected_scope
            assert metadata.auto_cleanup == expected_cleanup
            assert metadata.retention_days == expected_retention
            
            # Verify template-specific settings
            if ctype == CollectionType.GLOBAL:
                assert metadata.max_documents is None
            elif ctype == CollectionType.SESSION:
                assert metadata.max_documents == 1000
    
    @pytest.mark.asyncio
    async def test_default_collections_creation(self, mock_vector_store):
        """Test that default collections are created during initialization."""
        manager = CollectionManager(mock_vector_store)
        
        # Initialize should create default collections
        await manager.initialize()
        
        expected_defaults = [
            "global_knowledge",
            "shared_procedures", 
            "public_documents",
            "domain_research",
            "domain_creative",
            "domain_technical"
        ]
        
        for collection_name in expected_defaults:
            assert collection_name in manager.collections
            metadata = manager.collections[collection_name]
            
            if collection_name.startswith("global_") or collection_name.startswith("shared_") or collection_name.startswith("public_"):
                assert metadata.scope == KnowledgeScope.GLOBAL
            elif collection_name.startswith("domain_"):
                assert metadata.scope == KnowledgeScope.DOMAIN
    
    @pytest.mark.asyncio
    async def test_collection_statistics_tracking(self, collection_manager):
        """Test collection statistics and metadata tracking."""
        manager = collection_manager
        
        # Create a test collection
        metadata = await manager.create_collection(
            "stats_test_collection",
            CollectionType.AGENT_PRIVATE,
            owner_agent_id="stats_agent"
        )
        
        # Verify initial statistics
        assert metadata.document_count == 0
        assert metadata.total_chunks == 0
        assert "stats_test_collection" in manager.collection_stats
        
        stats = manager.collection_stats["stats_test_collection"]
        assert stats["documents"] == 0
        assert stats["queries"] == 0
        assert "created_at" in stats
        assert "last_accessed" in stats
        
        # Simulate document addition (would be done by knowledge manager)
        manager.collections["stats_test_collection"].document_count += 1
        manager.collections["stats_test_collection"].last_updated = datetime.now(timezone.utc)
        
        assert manager.collections["stats_test_collection"].document_count == 1
    
    @pytest.mark.asyncio
    async def test_collection_hierarchy_setup(self, collection_manager):
        """Test hierarchical relationships between collections."""
        manager = collection_manager
        
        # Verify hierarchy was setup during initialization
        assert "global_knowledge" in manager.hierarchy
        
        global_hierarchy = manager.hierarchy["global_knowledge"]
        assert global_hierarchy.collection_name == "global_knowledge"
        assert len(global_hierarchy.child_collections) > 0
        assert "domain_research" in global_hierarchy.child_collections
        
        # Test domain collection hierarchy
        if "domain_research" in manager.hierarchy:
            domain_hierarchy = manager.hierarchy["domain_research"]
            assert "global_knowledge" in domain_hierarchy.parent_collections
    
    @pytest.mark.asyncio
    async def test_collection_error_handling(self, collection_manager):
        """Test error handling in collection management."""
        manager = collection_manager
        
        # Test creating duplicate collection
        await manager.create_collection(
            "duplicate_test",
            CollectionType.GLOBAL
        )
        
        with pytest.raises(ValueError, match="already exists"):
            await manager.create_collection(
                "duplicate_test",
                CollectionType.GLOBAL
            )
        
        # Test accessing non-existent collection
        has_access = await manager._has_collection_access(
            "test_agent",
            "non_existent_collection",
            KnowledgePermission.READ
        )
        
        assert has_access is False
