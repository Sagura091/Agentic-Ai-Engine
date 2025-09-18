"""
Pytest configuration and fixtures for Revolutionary RAG System tests.

Provides shared fixtures, test configuration, and utilities for testing
the multi-agent RAG system components.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock

# Import test dependencies
import structlog

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


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_vector_store():
    """Create mock vector store for testing."""
    from app.rag.core.vector_store import ChromaVectorStore
    mock_store = Mock(spec=ChromaVectorStore)
    mock_store.get_or_create_collection = AsyncMock()
    mock_store.delete_collection = AsyncMock()
    mock_store.list_collections = AsyncMock(return_value=[])
    mock_store.get_collection = AsyncMock()
    return mock_store


@pytest.fixture
async def collection_manager(mock_vector_store):
    """Create collection manager for testing."""
    from app.rag.core.collection_manager import CollectionManager
    manager = CollectionManager(mock_vector_store)
    # Don't initialize to avoid dependency issues in tests
    return manager


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.chroma_persist_directory = "./test_data/chroma"
    settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    settings.chunk_size = 1000
    settings.chunk_overlap = 200
    settings.top_k = 10
    settings.score_threshold = 0.7
    return settings


@pytest.fixture
async def mock_chroma_client():
    """Create mock ChromaDB client for testing."""
    mock_client = Mock()
    mock_client.get_or_create_collection = AsyncMock()
    mock_client.add = AsyncMock()
    mock_client.query = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.update = AsyncMock()
    mock_client.peek = AsyncMock()
    mock_client.count = AsyncMock(return_value=0)
    
    # Mock collection
    mock_collection = Mock()
    mock_collection.add = AsyncMock()
    mock_collection.query = AsyncMock(return_value={
        'ids': [['doc1', 'doc2']],
        'distances': [[0.1, 0.2]],
        'documents': [['Test document 1', 'Test document 2']],
        'metadatas': [[{'source': 'test'}, {'source': 'test'}]]
    })
    mock_collection.count = AsyncMock(return_value=0)
    mock_collection.peek = AsyncMock(return_value={'ids': [], 'documents': [], 'metadatas': []})
    
    mock_client.get_or_create_collection.return_value = mock_collection
    
    return mock_client


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    from app.rag.core.knowledge_base import Document
    
    return [
        Document(
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            metadata={"category": "education", "difficulty": "beginner", "topic": "ml"},
            source="test_source_1"
        ),
        Document(
            title="Advanced Neural Networks",
            content="Deep neural networks with multiple hidden layers can learn complex patterns in data through backpropagation.",
            metadata={"category": "education", "difficulty": "advanced", "topic": "ml"},
            source="test_source_2"
        ),
        Document(
            title="Data Science Best Practices",
            content="Effective data science requires proper data cleaning, feature engineering, and model validation techniques.",
            metadata={"category": "methodology", "difficulty": "intermediate", "topic": "data_science"},
            source="test_source_3"
        ),
        Document(
            title="Python Programming Guide",
            content="Python is a versatile programming language widely used in data science, web development, and automation.",
            metadata={"category": "programming", "difficulty": "beginner", "topic": "python"},
            source="test_source_4"
        ),
        Document(
            title="Research Methodology",
            content="Systematic research approaches include hypothesis formation, data collection, analysis, and conclusion drawing.",
            metadata={"category": "methodology", "difficulty": "intermediate", "topic": "research"},
            source="test_source_5"
        )
    ]


@pytest.fixture
def sample_agent_profiles():
    """Create sample agent profiles for testing."""
    from app.rag.core.agent_knowledge_manager import (
        AgentKnowledgeProfile, 
        KnowledgeScope, 
        KnowledgePermission
    )
    
    return {
        "researcher": AgentKnowledgeProfile(
            agent_id="test_researcher_001",
            agent_type="research",
            scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.DOMAIN, KnowledgeScope.GLOBAL],
            permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE, KnowledgePermission.SHARE],
            preferred_collections=["domain_research", "global_knowledge"],
            memory_retention_days=60,
            max_memory_items=5000
        ),
        "creative": AgentKnowledgeProfile(
            agent_id="test_creative_001",
            agent_type="creative",
            scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.SHARED, KnowledgeScope.GLOBAL],
            permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE],
            preferred_collections=["domain_creative", "global_knowledge"],
            memory_retention_days=30,
            max_memory_items=3000
        ),
        "technical": AgentKnowledgeProfile(
            agent_id="test_technical_001",
            agent_type="technical",
            scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.DOMAIN, KnowledgeScope.GLOBAL],
            permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE, KnowledgePermission.DELETE],
            preferred_collections=["domain_technical", "global_knowledge"],
            memory_retention_days=90,
            max_memory_items=10000
        ),
        "general": AgentKnowledgeProfile(
            agent_id="test_general_001",
            agent_type="general",
            scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.GLOBAL],
            permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE],
            preferred_collections=["global_knowledge"],
            memory_retention_days=30,
            max_memory_items=2000
        )
    }


@pytest.fixture
def sample_memories():
    """Create sample memory entries for testing."""
    from app.rag.core.agent_knowledge_manager import AgentMemoryEntry
    from datetime import datetime
    
    return [
        AgentMemoryEntry(
            agent_id="test_agent_001",
            content="Successfully completed data analysis project with 95% accuracy",
            memory_type="episodic",
            context={"project": "data_analysis", "outcome": "success", "accuracy": 0.95},
            importance=0.8,
            tags=["success", "data_analysis", "project"]
        ),
        AgentMemoryEntry(
            agent_id="test_agent_001",
            content="Linear regression works well for continuous target variables",
            memory_type="semantic",
            context={"domain": "machine_learning", "algorithm": "linear_regression"},
            importance=0.7,
            tags=["ml", "regression", "knowledge"]
        ),
        AgentMemoryEntry(
            agent_id="test_agent_001",
            content="User prefers visual charts over text-based reports",
            memory_type="episodic",
            context={"user_interaction": True, "preference": "visual"},
            importance=0.6,
            tags=["user_preference", "visualization", "interaction"]
        ),
        AgentMemoryEntry(
            agent_id="test_agent_002",
            content="Python pandas library is efficient for data manipulation",
            memory_type="semantic",
            context={"programming": "python", "library": "pandas"},
            importance=0.9,
            tags=["python", "pandas", "data_manipulation"]
        )
    ]


@pytest.fixture
async def isolated_test_environment(temp_directory):
    """Create isolated test environment with temporary data directory."""
    # Set environment variables for testing
    os.environ["RAG_TEST_MODE"] = "true"
    os.environ["CHROMA_PERSIST_DIRECTORY"] = temp_directory
    
    yield temp_directory
    
    # Cleanup
    if "RAG_TEST_MODE" in os.environ:
        del os.environ["RAG_TEST_MODE"]
    if "CHROMA_PERSIST_DIRECTORY" in os.environ:
        del os.environ["CHROMA_PERSIST_DIRECTORY"]


# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "agent: mark test as agent-specific test"
    )
    config.addinivalue_line(
        "markers", "memory: mark test as memory-related test"
    )
    config.addinivalue_line(
        "markers", "collection: mark test as collection management test"
    )
    config.addinivalue_line(
        "markers", "tool: mark test as tool-related test"
    )


# Custom test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_document_in_results(document_title: str, search_results: list) -> bool:
        """Check if a document with given title is in search results."""
        return any(
            document_title.lower() in result.content.lower() or 
            document_title.lower() in result.metadata.get("title", "").lower()
            for result in search_results
        )
    
    @staticmethod
    def assert_memory_in_results(memory_content: str, search_results: list) -> bool:
        """Check if a memory with given content is in search results."""
        return any(
            memory_content.lower() in result.content.lower() and
            result.metadata.get("type") == "memory"
            for result in search_results
        )
    
    @staticmethod
    def get_results_by_type(search_results: list, result_type: str) -> list:
        """Filter search results by type (document, memory, etc.)."""
        return [
            result for result in search_results
            if result.metadata.get("type") == result_type
        ]
    
    @staticmethod
    def get_results_by_scope(search_results: list, scope: str) -> list:
        """Filter search results by knowledge scope."""
        return [
            result for result in search_results
            if result.metadata.get("scope") == scope
        ]


@pytest.fixture
def test_utils():
    """Provide test utility functions."""
    return TestUtils


# Performance test configuration
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_agents": 20,
        "max_documents_per_agent": 100,
        "max_memories_per_agent": 50,
        "max_search_time": 5.0,
        "max_ingestion_time": 30.0,
        "min_cache_hit_rate": 0.8,
        "max_concurrent_operations": 50
    }


# Async test helpers
@pytest.fixture
async def async_test_helper():
    """Helper for async test operations."""
    
    class AsyncTestHelper:
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
    
    return AsyncTestHelper()
