"""
Comprehensive Backend Testing Configuration and Fixtures.

This module provides shared fixtures, test configuration, and utilities for
comprehensive backend testing of the agentic AI system.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import uuid
from pathlib import Path
from typing import Generator, AsyncGenerator, Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Import test dependencies
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import application components
from app.main import app
from app.config.settings import get_settings
from app.models.database.base import Base, get_session
from app.models.agent import Agent, Conversation, Message, TaskExecution
from app.models.tool import Tool, AgentTool, ToolExecution
from app.models.user import User

# Import core components for testing
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType
from app.orchestration.orchestrator import orchestrator
from app.agents.base.agent import LangGraphAgent, AgentConfig
from app.agents.autonomous import AutonomousLangGraphAgent, AutonomousAgentConfig, AutonomyLevel, LearningMode
from app.llm.manager import LLMProviderManager
from app.llm.models import LLMConfig, ProviderType
from app.tools.dynamic_tool_factory import DynamicToolFactory, ToolCategory, ToolComplexity
from app.rag.core.enhanced_rag_service import EnhancedRAGService
from app.rag.core.knowledge_base import KnowledgeBase, KnowledgeConfig
from app.rag.core.agent_knowledge_manager import AgentKnowledgeManager, AgentKnowledgeProfile

# Configure structured logging for tests
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# Test Configuration
@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    settings = get_settings()
    settings.database_url = "sqlite+aiosqlite:///./test.db"
    settings.chroma_persist_directory = "./test_data/chroma"
    settings.enable_debug = True
    settings.log_level = "DEBUG"
    return settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Database Fixtures
@pytest.fixture
async def test_db_engine(test_settings):
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.database_url,
        echo=False,
        future=True
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_db_session(test_db_engine):
    """Create test database session."""
    async_session = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def override_get_session(test_db_session):
    """Override the get_session dependency."""
    def _override_get_session():
        return test_db_session
    
    app.dependency_overrides[get_session] = _override_get_session
    yield
    app.dependency_overrides.clear()


# Temporary Directory Fixtures
@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="backend_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_chroma_directory(temp_directory) -> str:
    """Create temporary ChromaDB directory."""
    chroma_dir = os.path.join(temp_directory, "chroma")
    os.makedirs(chroma_dir, exist_ok=True)
    return chroma_dir


# HTTP Client Fixtures
@pytest.fixture
def test_client(override_get_session):
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_test_client(override_get_session):
    """Create async test client for FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Mock LLM for Testing
class MockLLM:
    """Mock LLM for testing without external dependencies."""
    
    def __init__(self, model_name: str = "mock-llm", responses: Optional[List[str]] = None):
        self.model_name = model_name
        self.responses = responses or [
            "I understand the task and will proceed with the analysis.",
            "Based on the information provided, I recommend the following approach.",
            "Task completed successfully. Here are the results.",
            "I need to use tools to gather more information.",
            "Let me think about this step by step."
        ]
        self.call_count = 0
    
    async def ainvoke(self, messages, **kwargs):
        """Mock async invoke method."""
        self.call_count += 1
        response_index = (self.call_count - 1) % len(self.responses)
        return Mock(content=self.responses[response_index])
    
    def invoke(self, messages, **kwargs):
        """Mock sync invoke method."""
        self.call_count += 1
        response_index = (self.call_count - 1) % len(self.responses)
        return Mock(content=self.responses[response_index])


@pytest.fixture
def mock_llm():
    """Create mock LLM for testing."""
    return MockLLM()


@pytest.fixture
def mock_llm_with_responses():
    """Create mock LLM with specific responses."""
    def _create_mock_llm(responses: List[str]):
        return MockLLM(responses=responses)
    return _create_mock_llm


# LLM Provider Fixtures
@pytest.fixture
async def mock_llm_provider_manager():
    """Create mock LLM provider manager."""
    manager = Mock(spec=LLMProviderManager)
    manager.create_llm_instance = AsyncMock()
    manager.get_available_models = AsyncMock(return_value=[
        {"id": "mock-model-1", "name": "Mock Model 1"},
        {"id": "mock-model-2", "name": "Mock Model 2"}
    ])
    return manager


@pytest.fixture
def test_llm_config():
    """Create test LLM configuration."""
    return LLMConfig(
        provider=ProviderType.OLLAMA,
        model_id="llama3.2:latest",
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9
    )


# Agent Fixtures
@pytest.fixture
def test_agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="Test Agent",
        description="A test agent for validation",
        agent_type="test",
        model_name="mock-model",
        temperature=0.7,
        max_tokens=1024,
        capabilities=["reasoning", "tool_use"],
        system_prompt="You are a helpful test agent."
    )


@pytest.fixture
def test_autonomous_agent_config():
    """Create test autonomous agent configuration."""
    return AutonomousAgentConfig(
        name="Test Autonomous Agent",
        description="A test autonomous agent",
        autonomy_level=AutonomyLevel.ADAPTIVE,
        learning_mode=LearningMode.ACTIVE,
        decision_threshold=0.7,
        capabilities=["reasoning", "tool_use", "memory", "planning"],
        enable_proactive_behavior=True,
        enable_goal_setting=True
    )


@pytest.fixture
async def test_basic_agent(test_agent_config, mock_llm):
    """Create test basic agent."""
    agent = LangGraphAgent(
        config=test_agent_config,
        llm=mock_llm,
        tools=[]
    )
    await agent.initialize()
    return agent


@pytest.fixture
async def test_autonomous_agent(test_autonomous_agent_config, mock_llm):
    """Create test autonomous agent."""
    agent = AutonomousLangGraphAgent(
        config=test_autonomous_agent_config,
        llm=mock_llm,
        tools=[]
    )
    await agent.initialize()
    return agent


# Tool Fixtures
@pytest.fixture
def mock_tool_factory():
    """Create mock dynamic tool factory."""
    factory = Mock(spec=DynamicToolFactory)
    factory.create_tool_from_schema = AsyncMock()
    factory.create_tool_from_function = AsyncMock()
    factory.create_tool_from_description = AsyncMock()
    factory.register_tool = AsyncMock()
    factory.get_tool = Mock()
    factory.list_tools = Mock(return_value=[])
    return factory


@pytest.fixture
def test_tool_schemas():
    """Create test tool schemas."""
    return [
        {
            "name": "calculator",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
            },
            "category": ToolCategory.UTILITY,
            "complexity": ToolComplexity.SIMPLE
        },
        {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results", "default": 10}
            },
            "category": ToolCategory.RESEARCH,
            "complexity": ToolComplexity.MEDIUM
        }
    ]


# RAG System Fixtures
@pytest.fixture
async def test_rag_service(temp_chroma_directory):
    """Create test RAG service."""
    service = EnhancedRAGService()
    # Override settings for testing
    service.settings.chroma_persist_directory = temp_chroma_directory
    await service.initialize()
    return service


@pytest.fixture
async def test_knowledge_base(temp_chroma_directory):
    """Create test knowledge base."""
    config = KnowledgeConfig()
    config.vector_store.persist_directory = temp_chroma_directory
    
    kb = KnowledgeBase(config)
    await kb.initialize()
    return kb


@pytest.fixture
def test_agent_knowledge_profiles():
    """Create test agent knowledge profiles."""
    return {
        "research_agent": AgentKnowledgeProfile(
            agent_id="research_agent_001",
            agent_type="research",
            preferred_collections=["research_docs", "scientific_papers"]
        ),
        "creative_agent": AgentKnowledgeProfile(
            agent_id="creative_agent_001",
            agent_type="creative",
            preferred_collections=["creative_content", "inspiration"]
        ),
        "technical_agent": AgentKnowledgeProfile(
            agent_id="technical_agent_001",
            agent_type="technical",
            preferred_collections=["technical_docs", "code_examples"]
        )
    }


# Test Data Fixtures
@pytest.fixture
def test_documents():
    """Create test documents for ingestion."""
    return [
        {
            "title": "Introduction to AI",
            "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. It involves developing algorithms and systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, problem-solving, and decision-making.",
            "metadata": {"category": "education", "difficulty": "beginner"}
        },
        {
            "title": "Machine Learning Fundamentals",
            "content": "Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            "metadata": {"category": "education", "difficulty": "intermediate"}
        },
        {
            "title": "Neural Networks Explained",
            "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections. Deep learning uses multi-layer neural networks to solve complex problems.",
            "metadata": {"category": "education", "difficulty": "advanced"}
        }
    ]


# Performance Testing Fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_agents": 50,
        "max_concurrent_operations": 20,
        "max_response_time": 5.0,
        "max_memory_usage_mb": 1000,
        "test_duration_seconds": 60,
        "stress_test_iterations": 100
    }


# Test Utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def generate_test_agent_id() -> str:
        """Generate unique test agent ID."""
        return f"test_agent_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def generate_test_session_id() -> str:
        """Generate unique test session ID."""
        return f"test_session_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout=10.0, interval=0.1):
        """Wait for a condition to become true."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def run_concurrent_operations(operations, max_concurrent=10):
        """Run operations concurrently with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation()
        
        return await asyncio.gather(*[
            run_with_semaphore(op) for op in operations
        ])


@pytest.fixture
def test_utils():
    """Provide test utility functions."""
    return TestUtils


# Pytest Configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "agent: mark test as agent-specific test")
    config.addinivalue_line("markers", "llm: mark test as LLM-related test")
    config.addinivalue_line("markers", "tool: mark test as tool-related test")
    config.addinivalue_line("markers", "rag: mark test as RAG-related test")
    config.addinivalue_line("markers", "knowledge: mark test as knowledge base test")
    config.addinivalue_line("markers", "autonomous: mark test as autonomous agent test")
    config.addinivalue_line("markers", "orchestration: mark test as orchestration test")
    config.addinivalue_line("markers", "stress: mark test as stress test")
    config.addinivalue_line("markers", "behavior: mark test as behavior validation test")


# Cleanup Fixtures
@pytest.fixture(autouse=True)
async def cleanup_test_environment():
    """Automatically cleanup test environment after each test."""
    yield
    
    # Cleanup any test data, connections, etc.
    try:
        # Clear any cached data
        if hasattr(enhanced_orchestrator, 'agent_registry'):
            enhanced_orchestrator.agent_registry.clear()
        
        if hasattr(orchestrator, 'agents'):
            orchestrator.agents.clear()
        
        # Reset any global state
        logger.info("Test environment cleaned up")
    except Exception as e:
        logger.warning(f"Error during test cleanup: {e}")
