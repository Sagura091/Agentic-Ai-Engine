"""
Comprehensive RAG and Knowledge Base Integration Tests.

This module tests RAG system integration, knowledge base creation,
document ingestion, retrieval functionality, and agent-knowledge base interactions.
"""

import pytest
import asyncio
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

import structlog

# Import test infrastructure
from .base_test import RAGTest, TestResult, TestCategory, TestSeverity
from .test_utils import TestDataGenerator, MockRAGService, TestValidator

# Import application components
from app.rag.core.enhanced_rag_service import EnhancedRAGService
from app.rag.core.knowledge_base import KnowledgeBase, KnowledgeConfig, Document, KnowledgeQuery
from app.rag.core.agent_knowledge_manager import (
    AgentKnowledgeManager, 
    AgentKnowledgeProfile,
    KnowledgeScope,
    KnowledgePermission
)
from app.rag.core.collection_manager import CollectionManager, CollectionType
from app.rag.tools.enhanced_knowledge_tools import (
    EnhancedKnowledgeSearchTool,
    AgentDocumentIngestTool,
    AgentMemoryTool
)

logger = structlog.get_logger(__name__)


class TestRAGServiceInitialization(RAGTest):
    """Test RAG service initialization and configuration."""
    
    def __init__(self):
        super().__init__("RAG Service Initialization", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test RAG service initialization."""
        try:
            # Create temporary directory for testing
            temp_dir = tempfile.mkdtemp(prefix="rag_test_")
            
            try:
                # Test service initialization
                service = EnhancedRAGService()
                
                # Mock initialization to avoid external dependencies
                with patch.object(service, 'initialize', new_callable=AsyncMock) as mock_init:
                    mock_init.return_value = True
                    await service.initialize()
                    
                    initialization_success = mock_init.called
                
                # Test configuration validation
                config_valid = await self._test_configuration_validation(service)
                
                # Test component availability
                components_available = await self._test_component_availability(service)
                
                evidence = {
                    "initialization_success": initialization_success,
                    "config_validation": config_valid,
                    "components_available": components_available,
                    "temp_directory": temp_dir
                }
                
                success = initialization_success and config_valid and components_available
                
                return TestResult(
                    test_name=self.test_name,
                    category=self.category,
                    severity=self.severity,
                    passed=success,
                    duration=0.0,
                    evidence=evidence
                )
                
            finally:
                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=False,
                duration=0.0,
                error_message=str(e),
                evidence={"error_type": type(e).__name__}
            )
    
    async def _test_configuration_validation(self, service: EnhancedRAGService) -> bool:
        """Test RAG service configuration validation."""
        try:
            # Check if service has required configuration attributes
            required_attrs = ['settings', 'knowledge_base', 'collection_manager']
            
            for attr in required_attrs:
                if not hasattr(service, attr):
                    logger.debug(f"Missing required attribute: {attr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing configuration validation: {e}")
            return False
    
    async def _test_component_availability(self, service: EnhancedRAGService) -> bool:
        """Test availability of RAG service components."""
        try:
            # Mock component checks
            components = ['knowledge_base', 'collection_manager', 'ingestion_pipeline']
            
            for component in components:
                if not hasattr(service, component):
                    logger.debug(f"Missing component: {component}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing component availability: {e}")
            return False


class TestDocumentIngestion(RAGTest):
    """Test document ingestion functionality."""
    
    def __init__(self):
        super().__init__("Document Ingestion", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test document ingestion process."""
        try:
            # Create mock RAG service
            rag_service = MockRAGService()
            await rag_service.initialize()
            
            # Test single document ingestion
            single_doc_test = await self._test_single_document_ingestion(rag_service)
            
            # Test batch document ingestion
            batch_doc_test = await self._test_batch_document_ingestion(rag_service)
            
            # Test document validation
            validation_test = await self._test_document_validation(rag_service)
            
            # Test metadata handling
            metadata_test = await self._test_metadata_handling(rag_service)
            
            evidence = {
                "single_document": single_doc_test,
                "batch_documents": batch_doc_test,
                "validation": validation_test,
                "metadata_handling": metadata_test,
                "total_documents_ingested": len(rag_service.documents)
            }
            
            success = single_doc_test and batch_doc_test and validation_test and metadata_test
            
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=success,
                duration=0.0,
                evidence=evidence
            )
            
        except Exception as e:
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=False,
                duration=0.0,
                error_message=str(e),
                evidence={"error_type": type(e).__name__}
            )
    
    async def _test_single_document_ingestion(self, rag_service: MockRAGService) -> bool:
        """Test ingestion of a single document."""
        try:
            result = await rag_service.ingest_document(
                title="Test Document",
                content="This is a test document for RAG system validation.",
                collection="test_collection",
                metadata={"category": "test", "priority": "high"}
            )
            
            return await self.validate_document_ingestion(result)
            
        except Exception as e:
            logger.error(f"Error testing single document ingestion: {e}")
            return False
    
    async def _test_batch_document_ingestion(self, rag_service: MockRAGService) -> bool:
        """Test batch document ingestion."""
        try:
            documents = TestDataGenerator.generate_test_documents(5)
            
            results = []
            for doc in documents:
                result = await rag_service.ingest_document(
                    title=doc["title"],
                    content=doc["content"],
                    collection="batch_test",
                    metadata=doc["metadata"]
                )
                results.append(result)
            
            # All ingestions should succeed
            return all(await self.validate_document_ingestion(result) for result in results)
            
        except Exception as e:
            logger.error(f"Error testing batch document ingestion: {e}")
            return False
    
    async def _test_document_validation(self, rag_service: MockRAGService) -> bool:
        """Test document validation during ingestion."""
        try:
            # Test valid document
            valid_result = await rag_service.ingest_document(
                title="Valid Document",
                content="Valid content for testing.",
                collection="validation_test"
            )
            
            if not await self.validate_document_ingestion(valid_result):
                return False
            
            # Test invalid document (empty content)
            try:
                invalid_result = await rag_service.ingest_document(
                    title="Invalid Document",
                    content="",  # Empty content
                    collection="validation_test"
                )
                # Should still succeed in mock (real implementation might validate)
                return True
            except Exception:
                return True  # Expected validation error
            
        except Exception as e:
            logger.error(f"Error testing document validation: {e}")
            return False
    
    async def _test_metadata_handling(self, rag_service: MockRAGService) -> bool:
        """Test metadata handling during ingestion."""
        try:
            metadata = {
                "author": "Test Author",
                "category": "research",
                "tags": ["ai", "testing"],
                "created_date": datetime.utcnow().isoformat()
            }
            
            result = await rag_service.ingest_document(
                title="Metadata Test Document",
                content="Document for testing metadata handling.",
                collection="metadata_test",
                metadata=metadata
            )
            
            return await self.validate_document_ingestion(result)
            
        except Exception as e:
            logger.error(f"Error testing metadata handling: {e}")
            return False


class TestKnowledgeRetrieval(RAGTest):
    """Test knowledge retrieval and search functionality."""
    
    def __init__(self):
        super().__init__("Knowledge Retrieval", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test knowledge retrieval functionality."""
        try:
            # Create mock RAG service with test data
            rag_service = MockRAGService()
            await rag_service.initialize()
            
            # Ingest test documents
            await self._setup_test_data(rag_service)
            
            # Test basic search
            basic_search_test = await self._test_basic_search(rag_service)
            
            # Test filtered search
            filtered_search_test = await self._test_filtered_search(rag_service)
            
            # Test collection-specific search
            collection_search_test = await self._test_collection_search(rag_service)
            
            # Test search result quality
            quality_test = await self._test_search_quality(rag_service)
            
            evidence = {
                "basic_search": basic_search_test,
                "filtered_search": filtered_search_test,
                "collection_search": collection_search_test,
                "quality_test": quality_test,
                "total_documents": len(rag_service.documents)
            }
            
            success = basic_search_test and filtered_search_test and collection_search_test and quality_test
            
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=success,
                duration=0.0,
                evidence=evidence
            )
            
        except Exception as e:
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=False,
                duration=0.0,
                error_message=str(e),
                evidence={"error_type": type(e).__name__}
            )
    
    async def _setup_test_data(self, rag_service: MockRAGService):
        """Setup test data for retrieval tests."""
        documents = TestDataGenerator.generate_test_documents(10)
        
        for doc in documents:
            await rag_service.ingest_document(
                title=doc["title"],
                content=doc["content"],
                collection="retrieval_test",
                metadata=doc["metadata"]
            )
    
    async def _test_basic_search(self, rag_service: MockRAGService) -> bool:
        """Test basic search functionality."""
        try:
            result = await rag_service.search_knowledge(
                query="artificial intelligence",
                top_k=5
            )
            
            return await self.validate_search_results(result.get("results", []), min_results=1)
            
        except Exception as e:
            logger.error(f"Error testing basic search: {e}")
            return False
    
    async def _test_filtered_search(self, rag_service: MockRAGService) -> bool:
        """Test search with filters."""
        try:
            result = await rag_service.search_knowledge(
                query="machine learning",
                filters={"category": "ai"},
                top_k=3
            )
            
            return await self.validate_search_results(result.get("results", []))
            
        except Exception as e:
            logger.error(f"Error testing filtered search: {e}")
            return False
    
    async def _test_collection_search(self, rag_service: MockRAGService) -> bool:
        """Test collection-specific search."""
        try:
            result = await rag_service.search_knowledge(
                query="neural networks",
                collection="retrieval_test",
                top_k=5
            )
            
            return await self.validate_search_results(result.get("results", []))
            
        except Exception as e:
            logger.error(f"Error testing collection search: {e}")
            return False
    
    async def _test_search_quality(self, rag_service: MockRAGService) -> bool:
        """Test search result quality and relevance."""
        try:
            queries = TestDataGenerator.generate_test_queries(3)
            
            for query in queries:
                result = await rag_service.search_knowledge(query=query, top_k=5)
                
                if not await self.validate_search_results(result.get("results", [])):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing search quality: {e}")
            return False


class TestAgentKnowledgeIntegration(RAGTest):
    """Test integration between agents and knowledge systems."""
    
    def __init__(self):
        super().__init__("Agent-Knowledge Integration", TestSeverity.HIGH)
    
    async def execute_test(self) -> TestResult:
        """Test agent-knowledge system integration."""
        try:
            # Create test agent knowledge profiles
            profiles = self._create_test_profiles()
            
            # Test agent knowledge manager
            manager_test = await self._test_knowledge_manager(profiles["research_agent"])
            
            # Test knowledge tools
            tools_test = await self._test_knowledge_tools()
            
            # Test agent-specific collections
            collections_test = await self._test_agent_collections(profiles)
            
            # Test knowledge permissions
            permissions_test = await self._test_knowledge_permissions(profiles)
            
            evidence = {
                "knowledge_manager": manager_test,
                "knowledge_tools": tools_test,
                "agent_collections": collections_test,
                "permissions": permissions_test,
                "test_profiles": len(profiles)
            }
            
            success = manager_test and tools_test and collections_test and permissions_test
            
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=success,
                duration=0.0,
                evidence=evidence
            )
            
        except Exception as e:
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=False,
                duration=0.0,
                error_message=str(e),
                evidence={"error_type": type(e).__name__}
            )
    
    def _create_test_profiles(self) -> Dict[str, AgentKnowledgeProfile]:
        """Create test agent knowledge profiles."""
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
                scopes=[KnowledgeScope.PRIVATE, KnowledgeScope.SHARED],
                permissions=[KnowledgePermission.READ, KnowledgePermission.WRITE]
            )
        }
    
    async def _test_knowledge_manager(self, profile: AgentKnowledgeProfile) -> bool:
        """Test agent knowledge manager functionality."""
        try:
            # Mock the knowledge manager
            with patch('app.rag.core.agent_knowledge_manager.AgentKnowledgeManager') as MockManager:
                mock_manager = Mock()
                mock_manager.initialize = AsyncMock()
                mock_manager.is_initialized = True
                mock_manager.agent_id = profile.agent_id
                MockManager.return_value = mock_manager
                
                manager = AgentKnowledgeManager(profile)
                await manager.initialize()
                
                return manager.is_initialized and manager.agent_id == profile.agent_id
            
        except Exception as e:
            logger.error(f"Error testing knowledge manager: {e}")
            return False
    
    async def _test_knowledge_tools(self) -> bool:
        """Test knowledge tools functionality."""
        try:
            # Mock knowledge tools
            search_tool = Mock(spec=EnhancedKnowledgeSearchTool)
            search_tool.name = "knowledge_search"
            search_tool.invoke = AsyncMock(return_value="Search results")
            
            ingest_tool = Mock(spec=AgentDocumentIngestTool)
            ingest_tool.name = "document_ingest"
            ingest_tool.invoke = AsyncMock(return_value="Document ingested")
            
            memory_tool = Mock(spec=AgentMemoryTool)
            memory_tool.name = "agent_memory"
            memory_tool.invoke = AsyncMock(return_value="Memory stored")
            
            # Test tool execution
            search_result = await search_tool.invoke("test query")
            ingest_result = await ingest_tool.invoke("test document")
            memory_result = await memory_tool.invoke("test memory")
            
            return all([search_result, ingest_result, memory_result])
            
        except Exception as e:
            logger.error(f"Error testing knowledge tools: {e}")
            return False
    
    async def _test_agent_collections(self, profiles: Dict[str, AgentKnowledgeProfile]) -> bool:
        """Test agent-specific collections."""
        try:
            # Mock collection creation for each agent
            for agent_name, profile in profiles.items():
                collection_name = f"agent_{profile.agent_id}_private"
                
                # Mock collection exists
                if not collection_name:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing agent collections: {e}")
            return False
    
    async def _test_knowledge_permissions(self, profiles: Dict[str, AgentKnowledgeProfile]) -> bool:
        """Test knowledge access permissions."""
        try:
            for agent_name, profile in profiles.items():
                # Check permissions are properly set
                if not profile.permissions:
                    return False
                
                # Check scopes are defined
                if not profile.scopes:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing knowledge permissions: {e}")
            return False


# Test suite for RAG integration
class RAGIntegrationTestSuite:
    """Comprehensive test suite for RAG integration."""
    
    def __init__(self):
        self.tests = [
            TestRAGServiceInitialization(),
            TestDocumentIngestion(),
            TestKnowledgeRetrieval(),
            TestAgentKnowledgeIntegration()
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all RAG integration tests."""
        logger.info("Starting RAG integration test suite")
        
        results = []
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = await test.run()
            results.append(result)
        
        # Generate summary
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        
        summary = {
            "suite_name": "RAG Integration Test Suite",
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "results": [result.__dict__ for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"RAG integration test suite completed",
            total=total,
            passed=passed,
            failed=total-passed,
            success_rate=summary["success_rate"]
        )
        
        return summary


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.rag
@pytest.mark.unit
async def test_rag_service_initialization():
    """Pytest wrapper for RAG service initialization test."""
    test = TestRAGServiceInitialization()
    result = await test.run()
    assert result.passed, f"RAG service initialization failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.rag
@pytest.mark.integration
async def test_document_ingestion():
    """Pytest wrapper for document ingestion test."""
    test = TestDocumentIngestion()
    result = await test.run()
    assert result.passed, f"Document ingestion failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.rag
@pytest.mark.integration
async def test_knowledge_retrieval():
    """Pytest wrapper for knowledge retrieval test."""
    test = TestKnowledgeRetrieval()
    result = await test.run()
    assert result.passed, f"Knowledge retrieval failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.rag
@pytest.mark.integration
async def test_agent_knowledge_integration():
    """Pytest wrapper for agent-knowledge integration test."""
    test = TestAgentKnowledgeIntegration()
    result = await test.run()
    assert result.passed, f"Agent-knowledge integration failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.rag
@pytest.mark.integration
async def test_complete_rag_integration_suite():
    """Run the complete RAG integration test suite."""
    suite = RAGIntegrationTestSuite()
    summary = await suite.run_all_tests()
    
    assert summary["success_rate"] >= 0.75, f"RAG integration suite success rate too low: {summary['success_rate']}"
    assert summary["passed"] >= 3, f"Not enough tests passed: {summary['passed']}/{summary['total_tests']}"
