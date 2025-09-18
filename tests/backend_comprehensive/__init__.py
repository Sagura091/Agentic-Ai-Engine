"""
Comprehensive Backend Testing Suite for Agentic AI System.

This package provides a complete testing framework to validate that the backend
agentic AI system is truly functional and not just appearing to work.

Key Features:
- Agent creation and configuration validation
- LLM integration testing
- Tool integration and usage validation
- RAG system and knowledge base testing
- Agent behavior and autonomy validation
- Performance and stress testing
- Integration and end-to-end workflow testing

Usage:
    # Run all critical tests
    python -m tests.backend_comprehensive.run_comprehensive_tests --critical-only
    
    # Run complete test suite
    python -m tests.backend_comprehensive.run_comprehensive_tests --include-slow --include-stress
    
    # Run with pytest
    pytest tests/backend_comprehensive/ -m "critical"
"""

from .run_comprehensive_tests import ComprehensiveTestRunner

__version__ = "1.0.0"
__author__ = "Agentic AI Development Team"

# Test suite imports
from .test_agent_creation import AgentCreationTestSuite
from .test_llm_integration import LLMIntegrationTestSuite
from .test_tool_integration import ToolIntegrationTestSuite
from .test_rag_integration import RAGIntegrationTestSuite
from .test_agent_behavior import AgentBehaviorTestSuite
from .test_performance_stress import PerformanceStressTestSuite
from .test_integration_e2e import IntegrationE2ETestSuite

# Base test classes
from .base_test import (
    BaseTest, AgentTest, LLMTest, ToolTest, RAGTest, 
    BehaviorTest, PerformanceTest, StressTest, 
    IntegrationTest, EndToEndTest,
    TestResult, TestCategory, TestSeverity
)

# Test utilities
from .test_utils import (
    TestDataGenerator, MockLLMProvider, MockRAGService,
    MockToolFactory, TestValidator, AsyncTestHelper
)

__all__ = [
    # Main runner
    "ComprehensiveTestRunner",
    
    # Test suites
    "AgentCreationTestSuite",
    "LLMIntegrationTestSuite", 
    "ToolIntegrationTestSuite",
    "RAGIntegrationTestSuite",
    "AgentBehaviorTestSuite",
    "PerformanceStressTestSuite",
    "IntegrationE2ETestSuite",
    
    # Base classes
    "BaseTest", "AgentTest", "LLMTest", "ToolTest", "RAGTest",
    "BehaviorTest", "PerformanceTest", "StressTest",
    "IntegrationTest", "EndToEndTest",
    "TestResult", "TestCategory", "TestSeverity",
    
    # Utilities
    "TestDataGenerator", "MockLLMProvider", "MockRAGService",
    "MockToolFactory", "TestValidator", "AsyncTestHelper"
]
