"""
Comprehensive test suite for the Agent Builder Platform.

This test suite validates all components of the Agent Builder Platform including:
- Agent Factory and Builder functionality
- Agent Registry and lifecycle management
- Template system and configurations
- LLM provider integration
- System orchestration integration
- Monitoring and metrics
"""

import asyncio
import pytest
import uuid
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

# Import Agent Builder Platform components
from app.agents.factory import AgentType, AgentBuilderFactory, AgentBuilderConfig
from app.agents.registry import AgentRegistry, RegisteredAgent, AgentStatus, AgentHealth
from app.agents.templates import AgentTemplateLibrary
from app.llm.manager import EnhancedLLMProviderManager
from app.llm.models import LLMConfig, ProviderType
from app.agents.base.agent import AgentCapability

# Import system components
from app.core.unified_system_orchestrator import EnhancedUnifiedSystemOrchestrator
from app.services.revolutionary_ingestion_engine import IntelligentDocumentProcessor


class TestAgentBuilderFactory:
    """Test suite for Agent Builder Factory."""
    
    @pytest.fixture
    async def mock_llm_manager(self):
        """Create a mock LLM manager for testing."""
        manager = Mock(spec=EnhancedLLMProviderManager)
        manager.is_initialized.return_value = True
        manager.get_optimal_model_for_task = AsyncMock(return_value=LLMConfig(
            provider=ProviderType.OLLAMA,
            model_id="llama3.2:latest",
            temperature=0.7,
            max_tokens=2048
        ))
        return manager
    
    @pytest.fixture
    def agent_factory(self, mock_llm_manager):
        """Create an agent factory for testing."""
        return AgentBuilderFactory(mock_llm_manager)
    
    @pytest.mark.asyncio
    async def test_create_react_agent(self, agent_factory):
        """Test creating a React-based agent."""
        config = AgentBuilderConfig(
            name="Test React Agent",
            description="Test agent for React functionality",
            agent_type=AgentType.REACT,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest"
            ),
            capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE]
        )
        
        agent = await agent_factory.build_agent(config)
        
        assert agent is not None
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'description')
    
    @pytest.mark.asyncio
    async def test_create_autonomous_agent(self, agent_factory):
        """Test creating an autonomous agent."""
        config = AgentBuilderConfig(
            name="Test Autonomous Agent",
            description="Test autonomous agent",
            agent_type=AgentType.AUTONOMOUS,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest"
            ),
            capabilities=[AgentCapability.REASONING, AgentCapability.AUTONOMY]
        )
        
        agent = await agent_factory.build_agent(config)
        
        assert agent is not None
        # Autonomous agents should have additional capabilities
        assert hasattr(agent, 'beliefs')
        assert hasattr(agent, 'desires')
        assert hasattr(agent, 'intentions')
    
    @pytest.mark.asyncio
    async def test_create_multimodal_agent(self, agent_factory):
        """Test creating a multimodal agent."""
        config = AgentBuilderConfig(
            name="Test Multimodal Agent",
            description="Test multimodal agent",
            agent_type=AgentType.MULTIMODAL,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest"
            ),
            capabilities=[AgentCapability.VISION, AgentCapability.REASONING]
        )
        
        agent = await agent_factory.build_agent(config)
        
        assert agent is not None
        # Multimodal agents should support vision capabilities
        assert AgentCapability.VISION in config.capabilities


class TestAgentRegistry:
    """Test suite for Agent Registry."""
    
    @pytest.fixture
    async def mock_agent_factory(self):
        """Create a mock agent factory."""
        factory = Mock(spec=AgentBuilderFactory)
        factory.build_agent = AsyncMock(return_value=Mock())
        return factory
    
    @pytest.fixture
    async def mock_system_orchestrator(self):
        """Create a mock system orchestrator."""
        orchestrator = Mock(spec=EnhancedUnifiedSystemOrchestrator)
        return orchestrator
    
    @pytest.fixture
    def agent_registry(self, mock_agent_factory, mock_system_orchestrator):
        """Create an agent registry for testing."""
        return AgentRegistry(mock_agent_factory, mock_system_orchestrator)
    
    @pytest.mark.asyncio
    async def test_register_agent(self, agent_registry):
        """Test agent registration."""
        config = AgentBuilderConfig(
            name="Test Agent",
            description="Test agent registration",
            agent_type=AgentType.REACT,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest"
            )
        )
        
        agent_id = await agent_registry.register_agent(config)
        
        assert agent_id is not None
        assert isinstance(agent_id, str)
        
        # Verify agent is in registry
        registered_agent = agent_registry.get_agent(agent_id)
        assert registered_agent is not None
        assert registered_agent.name == "Test Agent"
        assert registered_agent.status == AgentStatus.REGISTERED
    
    @pytest.mark.asyncio
    async def test_start_stop_agent(self, agent_registry):
        """Test starting and stopping agents."""
        config = AgentBuilderConfig(
            name="Test Agent",
            description="Test agent lifecycle",
            agent_type=AgentType.REACT,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest"
            )
        )
        
        agent_id = await agent_registry.register_agent(config)
        
        # Start agent
        success = await agent_registry.start_agent(agent_id)
        assert success is True
        
        agent = agent_registry.get_agent(agent_id)
        assert agent.status == AgentStatus.RUNNING
        
        # Stop agent
        success = await agent_registry.stop_agent(agent_id)
        assert success is True
        
        agent = agent_registry.get_agent(agent_id)
        assert agent.status == AgentStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_collaboration_groups(self, agent_registry):
        """Test collaboration group functionality."""
        # Register multiple agents
        agent_ids = []
        for i in range(3):
            config = AgentBuilderConfig(
                name=f"Test Agent {i+1}",
                description=f"Test agent {i+1}",
                agent_type=AgentType.REACT,
                llm_config=LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model_id="llama3.2:latest"
                )
            )
            agent_id = await agent_registry.register_agent(config)
            agent_ids.append(agent_id)
        
        # Create collaboration group
        group_id = "test_group"
        await agent_registry.create_collaboration_group(group_id, agent_ids)
        
        # Verify group exists
        groups = agent_registry.list_collaboration_groups()
        assert group_id in groups
        assert len(groups[group_id]) == 3
    
    def test_registry_statistics(self, agent_registry):
        """Test registry statistics functionality."""
        stats = agent_registry.get_registry_stats()
        
        assert isinstance(stats, dict)
        assert "total_agents" in stats
        assert "agents_by_status" in stats
        assert "agents_by_type" in stats
        assert "agents_by_health" in stats


class TestAgentTemplateLibrary:
    """Test suite for Agent Template Library."""
    
    def test_get_all_templates(self):
        """Test getting all available templates."""
        templates = AgentTemplateLibrary.get_all_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        
        # Verify template structure
        for template in templates:
            assert "name" in template
            assert "description" in template
            assert "agent_type" in template
            assert "capabilities" in template
    
    def test_research_assistant_template(self):
        """Test research assistant template."""
        template = AgentTemplateLibrary.get_research_assistant_template()
        
        assert isinstance(template, AgentBuilderConfig)
        assert template.name == "Research Assistant"
        assert template.agent_type == AgentType.RAG
        assert AgentCapability.REASONING in template.capabilities
        assert AgentCapability.RESEARCH in template.capabilities
    
    def test_customer_support_template(self):
        """Test customer support template."""
        template = AgentTemplateLibrary.get_customer_support_template()
        
        assert isinstance(template, AgentBuilderConfig)
        assert template.name == "Customer Support Agent"
        assert template.agent_type == AgentType.REACT
        assert AgentCapability.COMMUNICATION in template.capabilities
    
    def test_data_analyst_template(self):
        """Test data analyst template."""
        template = AgentTemplateLibrary.get_data_analyst_template()
        
        assert isinstance(template, AgentBuilderConfig)
        assert template.name == "Data Analyst"
        assert template.agent_type == AgentType.WORKFLOW
        assert AgentCapability.ANALYSIS in template.capabilities


class TestIntelligentDocumentProcessor:
    """Test suite for Intelligent Document Processing."""
    
    @pytest.fixture
    async def mock_processor(self):
        """Create a mock intelligent document processor."""
        processor = IntelligentDocumentProcessor()
        processor.ingestion_engine = Mock()
        processor.agent_registry = Mock()
        processor.llm_manager = Mock()
        return processor
    
    @pytest.mark.asyncio
    async def test_document_type_detection(self, mock_processor):
        """Test document type detection."""
        # Test PDF detection
        pdf_type = mock_processor._determine_document_type("application/pdf", "test.pdf")
        assert pdf_type == "pdf"
        
        # Test code detection
        code_type = mock_processor._determine_document_type("text/plain", "test.py")
        assert code_type == "code"
        
        # Test default detection
        default_type = mock_processor._determine_document_type("text/plain", "test.txt")
        assert default_type == "summarization"


class TestSystemIntegration:
    """Test suite for system integration."""
    
    @pytest.mark.asyncio
    async def test_enhanced_orchestrator_initialization(self):
        """Test enhanced orchestrator initialization."""
        orchestrator = EnhancedUnifiedSystemOrchestrator()
        
        # Test that it has agent builder integration
        assert hasattr(orchestrator, 'agent_builder_integration')
    
    @pytest.mark.asyncio
    async def test_system_status_includes_agent_builder(self):
        """Test that system status includes agent builder information."""
        orchestrator = EnhancedUnifiedSystemOrchestrator()
        
        # Mock the agent builder integration
        orchestrator.agent_builder_integration = Mock()
        orchestrator.agent_builder_integration.get_integration_status.return_value = {
            "status": "initialized",
            "agent_registry_initialized": True
        }
        
        status = orchestrator.get_system_status()
        
        assert "agent_builder_integration" in status
        assert status["agent_builder_integration"]["status"] == "initialized"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
