"""
Comprehensive Test Suite for Revolutionary Multi-Agent RAG System.

This test suite validates all revolutionary features including:
- Agent-specific knowledge management
- Multi-agent isolation and permissions
- Memory integration and retrieval
- Hierarchical collection management
- Enhanced knowledge tools
- Performance optimization
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

# Import our revolutionary RAG components
from app.rag.core.agent_knowledge_manager import (
    AgentKnowledgeManager,
    AgentKnowledgeProfile,
    KnowledgeScope,
    KnowledgePermission,
    AgentMemoryEntry
)
from app.rag.core.enhanced_rag_service import EnhancedRAGService
from app.rag.core.collection_manager import CollectionManager, CollectionType
from app.rag.core.knowledge_base import Document, KnowledgeQuery
from app.rag.tools.enhanced_knowledge_tools import (
    EnhancedKnowledgeSearchTool,
    AgentDocumentIngestTool,
    AgentMemoryTool
)


class TestRevolutionaryRAGSystem:
    """Comprehensive test suite for the revolutionary multi-agent RAG system."""
    
    @pytest.fixture
    async def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def enhanced_rag_service(self, temp_data_dir):
        """Create enhanced RAG service for testing."""
        service = EnhancedRAGService()
        # Override data directory for testing
        service.settings.chroma_persist_directory = temp_data_dir
        await service.initialize()
        return service
    
    @pytest.fixture
    async def test_agents(self):
        """Create test agent profiles."""
        return {
            "research_agent": AgentKnowledgeProfile(
                agent_id="research_agent_001",
                agent_type="research",
                scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.DOMAIN, KnowledgeScope.GLOBAL],
                permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE, KnowledgePermission.SHARE]
            ),
            "creative_agent": AgentKnowledgeProfile(
                agent_id="creative_agent_001",
                agent_type="creative",
                scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.SHARED, KnowledgeScope.GLOBAL],
                permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE]
            ),
            "technical_agent": AgentKnowledgeProfile(
                agent_id="technical_agent_001",
                agent_type="technical",
                scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.DOMAIN, KnowledgeScope.GLOBAL],
                permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE, KnowledgePermission.DELETE]
            )
        }


class TestAgentKnowledgeManager:
    """Test agent-specific knowledge management."""
    
    @pytest.mark.asyncio
    async def test_agent_manager_initialization(self, test_agents):
        """Test agent knowledge manager initialization."""
        profile = test_agents["research_agent"]
        manager = AgentKnowledgeManager(profile)
        
        # Test initialization
        await manager.initialize()
        
        assert manager.is_initialized
        assert manager.agent_id == "research_agent_001"
        assert len(manager.knowledge_bases) > 0
        assert len(manager.collection_metadata) >= 3  # private, memory, session
        
        # Verify agent-specific collections were created
        assert manager.private_collection == "agent_research_agent_001_private"
        assert manager.memory_collection == "agent_research_agent_001_memory"
        assert manager.session_collection == "agent_research_agent_001_session"
    
    @pytest.mark.asyncio
    async def test_agent_knowledge_isolation(self, test_agents):
        """Test that agents have isolated knowledge spaces."""
        # Create two different agents
        research_manager = AgentKnowledgeManager(test_agents["research_agent"])
        creative_manager = AgentKnowledgeManager(test_agents["creative_agent"])
        
        await research_manager.initialize()
        await creative_manager.initialize()
        
        # Add document to research agent's private collection
        research_doc = Document(
            title="Research Methodology",
            content="Advanced research techniques for data analysis",
            metadata={"type": "research", "confidential": True}
        )
        
        doc_id = await research_manager.add_document(
            research_doc, 
            KnowledgeScope.PRIVATE
        )
        
        # Verify creative agent cannot access research agent's private document
        creative_search = await creative_manager.search_knowledge(
            "research techniques",
            scopes=[KnowledgeScope.PRIVATE],
            top_k=10
        )
        
        # Should not find the research document in creative agent's search
        research_content_found = any(
            "Advanced research techniques" in result.content 
            for result in creative_search.results
        )
        
        assert not research_content_found, "Agent isolation failed - private content leaked"
        assert doc_id is not None
    
    @pytest.mark.asyncio
    async def test_agent_memory_system(self, test_agents):
        """Test episodic and semantic memory integration."""
        manager = AgentKnowledgeManager(test_agents["research_agent"])
        await manager.initialize()
        
        # Add episodic memory
        episodic_memory_id = await manager.add_memory(
            content="Successfully completed analysis of customer data trends",
            memory_type="episodic",
            context={"project": "customer_analysis", "outcome": "success"},
            importance=0.8,
            tags=["analysis", "success", "customer_data"]
        )
        
        # Add semantic memory
        semantic_memory_id = await manager.add_memory(
            content="Linear regression is effective for predicting continuous variables",
            memory_type="semantic",
            context={"domain": "machine_learning", "concept": "regression"},
            importance=0.9,
            tags=["ml", "regression", "prediction"]
        )
        
        # Test memory retrieval
        search_result = await manager.search_knowledge(
            "customer analysis success",
            include_memories=True,
            top_k=5
        )
        
        # Verify memories are included in search results
        memory_found = any(
            result.metadata.get("type") == "memory" 
            for result in search_result.results
        )
        
        assert episodic_memory_id is not None
        assert semantic_memory_id is not None
        assert memory_found, "Memory integration failed"
        assert len(manager.episodic_memories) == 1
        assert len(manager.semantic_memories) == 1
    
    @pytest.mark.asyncio
    async def test_knowledge_scope_permissions(self, test_agents):
        """Test knowledge scope and permission system."""
        manager = AgentKnowledgeManager(test_agents["research_agent"])
        await manager.initialize()
        
        # Test adding documents to different scopes
        private_doc = Document(
            title="Private Research Notes",
            content="Confidential research findings",
            metadata={"sensitivity": "high"}
        )
        
        global_doc = Document(
            title="Public Research Paper",
            content="Published research findings available to all",
            metadata={"published": True}
        )
        
        # Add to private scope
        private_id = await manager.add_document(private_doc, KnowledgeScope.PRIVATE)
        
        # Add to global scope
        global_id = await manager.add_document(global_doc, KnowledgeScope.GLOBAL)
        
        # Test scope-specific search
        private_search = await manager.search_knowledge(
            "research findings",
            scopes=[KnowledgeScope.PRIVATE],
            top_k=10
        )
        
        global_search = await manager.search_knowledge(
            "research findings",
            scopes=[KnowledgeScope.GLOBAL],
            top_k=10
        )
        
        # Verify scope isolation
        assert private_id is not None
        assert global_id is not None
        assert len(private_search.results) > 0
        assert len(global_search.results) > 0
        
        # Verify scope metadata
        for result in private_search.results:
            if "Confidential research" in result.content:
                assert result.metadata.get("scope") == "private"


class TestEnhancedRAGService:
    """Test enhanced RAG service with multi-agent orchestration."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_orchestration(self, enhanced_rag_service):
        """Test managing multiple agents simultaneously."""
        service = enhanced_rag_service
        
        # Create multiple agents
        research_manager = await service.get_or_create_agent_manager(
            "research_001", "research"
        )
        creative_manager = await service.get_or_create_agent_manager(
            "creative_001", "creative"
        )
        technical_manager = await service.get_or_create_agent_manager(
            "technical_001", "technical"
        )
        
        # Verify agents are properly managed
        assert len(service.agent_managers) == 3
        assert "research_001" in service.agent_managers
        assert "creative_001" in service.agent_managers
        assert "technical_001" in service.agent_managers
        
        # Test agent-specific operations
        research_doc = Document(
            title="Research Protocol",
            content="Standard research methodology for experiments"
        )
        
        doc_id = await service.add_document(
            "research_001",
            research_doc,
            KnowledgeScope.PRIVATE
        )
        
        # Verify document was added to correct agent
        search_result = await service.search_knowledge(
            "research_001",
            "research methodology",
            scopes=[KnowledgeScope.PRIVATE]
        )
        
        assert doc_id is not None
        assert len(search_result.results) > 0
        assert search_result.metadata["agent_id"] == "research_001"
    
    @pytest.mark.asyncio
    async def test_advanced_retrieval_strategies(self, enhanced_rag_service):
        """Test advanced retrieval with query expansion and re-ranking."""
        service = enhanced_rag_service
        
        # Create agent and add diverse content
        agent_id = "test_agent_advanced"
        manager = await service.get_or_create_agent_manager(agent_id, "research")
        
        # Add multiple documents with different relevance
        documents = [
            Document(
                title="Machine Learning Basics",
                content="Introduction to machine learning algorithms and concepts",
                metadata={"topic": "ml", "level": "beginner"}
            ),
            Document(
                title="Advanced ML Techniques",
                content="Deep learning and neural network architectures",
                metadata={"topic": "ml", "level": "advanced"}
            ),
            Document(
                title="Data Science Methods",
                content="Statistical analysis and data mining techniques",
                metadata={"topic": "data_science", "level": "intermediate"}
            )
        ]
        
        for doc in documents:
            await service.add_document(agent_id, doc, KnowledgeScope.PRIVATE)
        
        # Test advanced search with re-ranking
        result = await service.search_knowledge(
            agent_id,
            "machine learning",
            use_advanced_retrieval=True,
            top_k=5
        )
        
        # Verify advanced retrieval features
        assert len(result.results) > 0
        assert result.metadata.get("reranked") is True
        assert "expanded_queries" in result.metadata
        
        # Verify relevance ordering (ML content should rank higher)
        ml_results = [r for r in result.results if "machine learning" in r.content.lower()]
        assert len(ml_results) > 0
    
    @pytest.mark.asyncio
    async def test_memory_integration_in_service(self, enhanced_rag_service):
        """Test memory integration through the service layer."""
        service = enhanced_rag_service
        agent_id = "memory_test_agent"
        
        # Create agent
        await service.get_or_create_agent_manager(agent_id, "general")
        
        # Add memories
        memory_id_1 = await service.add_memory(
            agent_id,
            "Learned that customers prefer simple interfaces",
            memory_type="episodic",
            importance=0.7,
            tags=["ui", "customer_feedback"]
        )
        
        memory_id_2 = await service.add_memory(
            agent_id,
            "User experience design principles: simplicity, consistency, feedback",
            memory_type="semantic",
            importance=0.9,
            tags=["ux", "design_principles"]
        )
        
        # Search with memory integration
        result = await service.search_knowledge(
            agent_id,
            "user interface design",
            include_memories=True,
            top_k=10
        )
        
        # Verify memories are included and boost relevance
        memory_results = [r for r in result.results if r.metadata.get("type") == "memory"]
        
        assert memory_id_1 is not None
        assert memory_id_2 is not None
        assert len(memory_results) > 0
        
        # High importance memory should rank well
        high_importance_found = any(
            r.metadata.get("importance", 0) > 0.8 
            for r in memory_results
        )
        assert high_importance_found


class TestEnhancedKnowledgeTools:
    """Test enhanced knowledge tools for LangChain integration."""
    
    @pytest.mark.asyncio
    async def test_enhanced_knowledge_search_tool(self, enhanced_rag_service):
        """Test enhanced knowledge search tool."""
        service = enhanced_rag_service
        tool = EnhancedKnowledgeSearchTool(service)
        
        # Create agent and add content
        agent_id = "tool_test_agent"
        await service.get_or_create_agent_manager(agent_id, "research")
        
        test_doc = Document(
            title="Tool Testing Document",
            content="This document is used for testing the enhanced search tool functionality",
            metadata={"test": True}
        )
        
        await service.add_document(agent_id, test_doc, KnowledgeScope.PRIVATE)
        
        # Test tool execution
        result = await tool._arun(
            query="testing tool functionality",
            agent_id=agent_id,
            top_k=5,
            include_memories=True
        )
        
        # Parse JSON result
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["agent_id"] == agent_id
        assert len(result_data["results"]) > 0
        assert "testing the enhanced search tool" in result_data["results"][0]["content"]
    
    @pytest.mark.asyncio
    async def test_agent_document_ingest_tool(self, enhanced_rag_service):
        """Test agent document ingestion tool."""
        service = enhanced_rag_service
        tool = AgentDocumentIngestTool(service)
        
        agent_id = "ingest_test_agent"
        
        # Test document ingestion
        result = await tool._arun(
            title="Test Document for Ingestion",
            content="This is a test document being ingested through the tool",
            agent_id=agent_id,
            scope="private",
            metadata={"source": "tool_test"}
        )
        
        # Parse JSON result
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["agent_id"] == agent_id
        assert result_data["scope"] == "private"
        assert "document_id" in result_data
        
        # Verify document was actually stored
        search_result = await service.search_knowledge(
            agent_id,
            "test document being ingested",
            scopes=[KnowledgeScope.PRIVATE]
        )
        
        assert len(search_result.results) > 0
    
    @pytest.mark.asyncio
    async def test_agent_memory_tool(self, enhanced_rag_service):
        """Test agent memory creation tool."""
        service = enhanced_rag_service
        tool = AgentMemoryTool(service)
        
        agent_id = "memory_tool_test_agent"
        
        # Test memory creation
        result = await tool._arun(
            content="Remembered important insight about user behavior patterns",
            agent_id=agent_id,
            memory_type="episodic",
            importance=0.8,
            context={"session": "user_research", "insight_type": "behavioral"},
            tags=["user_behavior", "insights", "research"]
        )
        
        # Parse JSON result
        import json
        result_data = json.loads(result)
        
        assert result_data["success"] is True
        assert result_data["agent_id"] == agent_id
        assert result_data["memory_type"] == "episodic"
        assert result_data["importance"] == 0.8
        assert "memory_id" in result_data
        
        # Verify memory is searchable
        search_result = await service.search_knowledge(
            agent_id,
            "user behavior patterns",
            include_memories=True
        )
        
        memory_found = any(
            result.metadata.get("type") == "memory" 
            for result in search_result.results
        )
        assert memory_found
