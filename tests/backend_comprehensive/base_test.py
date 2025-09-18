"""
Base Test Classes for Comprehensive Backend Testing.

This module provides base test classes with common functionality for
testing different aspects of the agentic AI system.
"""

import pytest
import asyncio
import time
import psutil
import tracemalloc
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class TestCategory(Enum):
    """Categories of tests for organization and reporting."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    BEHAVIOR = "behavior"
    STRESS = "stress"
    END_TO_END = "end_to_end"


class TestSeverity(Enum):
    """Test severity levels for prioritization."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestResult:
    """Standardized test result structure."""
    test_name: str
    category: TestCategory
    severity: TestSeverity
    passed: bool
    duration: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    evidence: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metrics is None:
            self.metrics = {}
        if self.evidence is None:
            self.evidence = {}


@dataclass
class PerformanceMetrics:
    """Performance metrics for test validation."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    operations_per_second: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[float] = None


class BaseTest(ABC):
    """
    Base test class with common functionality for all test types.
    
    Provides:
    - Standardized test execution
    - Performance monitoring
    - Result collection
    - Error handling
    - Logging integration
    """
    
    def __init__(self, test_name: str, category: TestCategory, severity: TestSeverity):
        self.test_name = test_name
        self.category = category
        self.severity = severity
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.performance_metrics: Optional[PerformanceMetrics] = None
        
        logger.info(f"Initialized test: {test_name}", category=category.value, severity=severity.value)
    
    async def setup(self) -> None:
        """Setup test environment. Override in subclasses."""
        logger.debug(f"Setting up test: {self.test_name}")
    
    async def teardown(self) -> None:
        """Cleanup test environment. Override in subclasses."""
        logger.debug(f"Tearing down test: {self.test_name}")
    
    @abstractmethod
    async def execute_test(self) -> TestResult:
        """Execute the actual test. Must be implemented by subclasses."""
        pass
    
    async def run(self) -> TestResult:
        """
        Run the complete test with setup, execution, and teardown.
        
        Returns:
            TestResult with execution details
        """
        try:
            # Setup
            await self.setup()
            
            # Start performance monitoring
            self._start_performance_monitoring()
            
            # Execute test
            self.start_time = time.time()
            result = await self.execute_test()
            self.end_time = time.time()
            
            # Stop performance monitoring
            self._stop_performance_monitoring()
            
            # Add performance metrics to result
            if self.performance_metrics:
                result.metrics.update({
                    "performance": {
                        "execution_time": self.performance_metrics.execution_time,
                        "memory_usage_mb": self.performance_metrics.memory_usage_mb,
                        "cpu_usage_percent": self.performance_metrics.cpu_usage_percent,
                        "peak_memory_mb": self.performance_metrics.peak_memory_mb
                    }
                })
            
            self.results.append(result)
            
            logger.info(
                f"Test completed: {self.test_name}",
                passed=result.passed,
                duration=result.duration,
                category=self.category.value
            )
            
            return result
            
        except Exception as e:
            error_result = TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=False,
                duration=time.time() - (self.start_time or time.time()),
                error_message=str(e),
                evidence={"traceback": traceback.format_exc()}
            )
            
            self.results.append(error_result)
            
            logger.error(
                f"Test failed: {self.test_name}",
                error=str(e),
                category=self.category.value
            )
            
            return error_result
            
        finally:
            # Always cleanup
            try:
                await self.teardown()
            except Exception as e:
                logger.warning(f"Error during teardown: {e}")
    
    def _start_performance_monitoring(self):
        """Start performance monitoring."""
        tracemalloc.start()
        self._initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self._initial_cpu_time = psutil.Process().cpu_times()
    
    def _stop_performance_monitoring(self):
        """Stop performance monitoring and calculate metrics."""
        if self.start_time and self.end_time:
            execution_time = self.end_time - self.start_time
            
            # Memory metrics
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = current_memory - self._initial_memory
            
            # Peak memory
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / 1024 / 1024
            tracemalloc.stop()
            
            # CPU metrics (simplified)
            final_cpu_time = psutil.Process().cpu_times()
            cpu_usage = ((final_cpu_time.user - self._initial_cpu_time.user) + 
                        (final_cpu_time.system - self._initial_cpu_time.system)) / execution_time * 100
            
            self.performance_metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                peak_memory_mb=peak_memory_mb
            )


class AgentTest(BaseTest):
    """Base class for agent-specific tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(test_name, TestCategory.UNIT, severity)
        self.agent = None
        self.agent_id = None
    
    async def create_test_agent(self, agent_config: Dict[str, Any], llm=None, tools=None) -> Any:
        """Create a test agent with given configuration."""
        # This will be implemented by specific test classes
        raise NotImplementedError("Subclasses must implement create_test_agent")
    
    async def validate_agent_creation(self, agent) -> bool:
        """Validate that agent was created correctly."""
        if not agent:
            return False
        
        # Basic validation
        if not hasattr(agent, 'agent_id'):
            return False
        
        if not hasattr(agent, 'config'):
            return False
        
        return True
    
    async def validate_agent_response(self, response: Any) -> bool:
        """Validate agent response quality."""
        if not response:
            return False
        
        # Check if response has content
        if hasattr(response, 'content') and response.content:
            return len(response.content.strip()) > 0
        
        return False


class LLMTest(BaseTest):
    """Base class for LLM integration tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.CRITICAL):
        super().__init__(test_name, TestCategory.INTEGRATION, severity)
        self.llm = None
        self.llm_config = None
    
    async def validate_llm_response(self, response: Any, expected_keywords: List[str] = None) -> bool:
        """Validate LLM response quality."""
        if not response:
            return False
        
        content = ""
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        
        if not content or len(content.strip()) < 10:
            return False
        
        # Check for expected keywords if provided
        if expected_keywords:
            content_lower = content.lower()
            found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in content_lower)
            return found_keywords >= len(expected_keywords) * 0.5  # At least 50% of keywords
        
        return True


class ToolTest(BaseTest):
    """Base class for tool integration tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(test_name, TestCategory.INTEGRATION, severity)
        self.tool = None
        self.tool_factory = None
    
    async def validate_tool_creation(self, tool) -> bool:
        """Validate that tool was created correctly."""
        if not tool:
            return False
        
        # Check required attributes
        required_attrs = ['name', 'description']
        for attr in required_attrs:
            if not hasattr(tool, attr) or not getattr(tool, attr):
                return False
        
        return True
    
    async def validate_tool_execution(self, result: Any) -> bool:
        """Validate tool execution result."""
        # Basic validation - tool should return something
        return result is not None


class RAGTest(BaseTest):
    """Base class for RAG system tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(test_name, TestCategory.INTEGRATION, severity)
        self.rag_service = None
        self.knowledge_base = None
    
    async def validate_document_ingestion(self, result: Dict[str, Any]) -> bool:
        """Validate document ingestion result."""
        if not result:
            return False
        
        return result.get('success', False) and 'document_id' in result
    
    async def validate_search_results(self, results: List[Any], min_results: int = 1) -> bool:
        """Validate search results quality."""
        if not results or len(results) < min_results:
            return False
        
        # Check that results have required fields
        for result in results:
            if not hasattr(result, 'content') or not result.content:
                return False
            if not hasattr(result, 'score') or result.score <= 0:
                return False
        
        return True


class PerformanceTest(BaseTest):
    """Base class for performance tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.MEDIUM):
        super().__init__(test_name, TestCategory.PERFORMANCE, severity)
        self.performance_thresholds = {
            'max_response_time': 5.0,
            'max_memory_usage_mb': 500,
            'max_cpu_usage_percent': 80,
            'min_throughput': 1.0
        }
    
    def set_performance_thresholds(self, thresholds: Dict[str, float]):
        """Set custom performance thresholds."""
        self.performance_thresholds.update(thresholds)
    
    async def validate_performance(self, metrics: PerformanceMetrics) -> bool:
        """Validate performance against thresholds."""
        if metrics.execution_time > self.performance_thresholds['max_response_time']:
            return False
        
        if metrics.memory_usage_mb > self.performance_thresholds['max_memory_usage_mb']:
            return False
        
        if metrics.cpu_usage_percent > self.performance_thresholds['max_cpu_usage_percent']:
            return False
        
        if (metrics.throughput is not None and 
            metrics.throughput < self.performance_thresholds['min_throughput']):
            return False
        
        return True


class BehaviorTest(BaseTest):
    """Base class for agent behavior validation tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.CRITICAL):
        super().__init__(test_name, TestCategory.BEHAVIOR, severity)
        self.behavior_metrics = {}
    
    async def validate_autonomous_behavior(self, agent_responses: List[Any]) -> bool:
        """Validate that agent exhibits autonomous behavior."""
        if not agent_responses or len(agent_responses) < 2:
            return False
        
        # Check for variation in responses (not scripted)
        unique_responses = set(str(response) for response in agent_responses)
        variation_ratio = len(unique_responses) / len(agent_responses)
        
        # Should have at least 70% unique responses for autonomy
        return variation_ratio >= 0.7
    
    async def validate_decision_making(self, decisions: List[Dict[str, Any]]) -> bool:
        """Validate agent decision-making capabilities."""
        if not decisions:
            return False
        
        # Check that decisions have reasoning
        for decision in decisions:
            if 'reasoning' not in decision or not decision['reasoning']:
                return False
            if 'action' not in decision or not decision['action']:
                return False
        
        return True


class IntegrationTest(BaseTest):
    """Base class for integration tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.HIGH):
        super().__init__(test_name, TestCategory.INTEGRATION, severity)
        self.components = []
    
    async def validate_component_integration(self, components: List[Any]) -> bool:
        """Validate that components integrate correctly."""
        if not components:
            return False
        
        # Basic validation that all components are initialized
        for component in components:
            if not component:
                return False
        
        return True


class StressTest(BaseTest):
    """Base class for stress tests."""
    
    def __init__(self, test_name: str, severity: TestSeverity = TestSeverity.MEDIUM):
        super().__init__(test_name, TestCategory.STRESS, severity)
        self.stress_config = {
            'max_concurrent_operations': 50,
            'test_duration_seconds': 60,
            'max_error_rate': 0.05  # 5% error rate
        }
    
    def set_stress_config(self, config: Dict[str, Any]):
        """Set stress test configuration."""
        self.stress_config.update(config)
    
    async def validate_stress_results(self, results: List[Any]) -> bool:
        """Validate stress test results."""
        if not results:
            return False
        
        # Calculate error rate
        errors = sum(1 for result in results if not result.get('success', True))
        error_rate = errors / len(results)
        
        return error_rate <= self.stress_config['max_error_rate']


# Test Suite Manager
class TestSuiteManager:
    """Manages execution of multiple tests and result aggregation."""
    
    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.tests: List[BaseTest] = []
        self.results: List[TestResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def add_test(self, test: BaseTest):
        """Add a test to the suite."""
        self.tests.append(test)
    
    async def run_all_tests(self, parallel: bool = False) -> Dict[str, Any]:
        """Run all tests in the suite."""
        self.start_time = datetime.utcnow()
        
        logger.info(f"Starting test suite: {self.suite_name}", test_count=len(self.tests))
        
        if parallel:
            # Run tests in parallel
            tasks = [test.run() for test in self.tests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run tests sequentially
            results = []
            for test in self.tests:
                result = await test.run()
                results.append(result)
        
        self.end_time = datetime.utcnow()
        self.results = [r for r in results if isinstance(r, TestResult)]
        
        # Generate summary
        summary = self._generate_summary()
        
        logger.info(
            f"Test suite completed: {self.suite_name}",
            total_tests=len(self.results),
            passed=summary['passed'],
            failed=summary['failed'],
            duration=summary['total_duration']
        )
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test suite summary."""
        passed = sum(1 for result in self.results if result.passed)
        failed = len(self.results) - passed
        total_duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        return {
            'suite_name': self.suite_name,
            'total_tests': len(self.results),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(self.results) if self.results else 0,
            'total_duration': total_duration,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'results': self.results
        }
