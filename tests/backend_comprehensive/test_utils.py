"""
Test Utilities and Helpers for Comprehensive Backend Testing.

This module provides utility functions, mock objects, and helper classes
for comprehensive testing of the agentic AI system.
"""

import asyncio
import json
import random
import string
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class MockResponseType(Enum):
    """Types of mock responses for different scenarios."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARTIAL = "partial"
    RANDOM = "random"


@dataclass
class MockScenario:
    """Configuration for mock behavior scenarios."""
    response_type: MockResponseType
    delay_seconds: float = 0.0
    error_message: Optional[str] = None
    success_probability: float = 1.0
    response_data: Optional[Dict[str, Any]] = None


class TestDataGenerator:
    """Generates test data for various testing scenarios."""
    
    @staticmethod
    def generate_agent_id() -> str:
        """Generate a unique agent ID for testing."""
        return f"test_agent_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID for testing."""
        return f"test_session_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def generate_task_id() -> str:
        """Generate a unique task ID for testing."""
        return f"test_task_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def generate_random_string(length: int = 10) -> str:
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def generate_agent_config(agent_type: str = "test") -> Dict[str, Any]:
        """Generate a test agent configuration."""
        return {
            "name": f"Test Agent {TestDataGenerator.generate_random_string(5)}",
            "description": f"A test agent for {agent_type} validation",
            "agent_type": agent_type,
            "model_name": "mock-model",
            "temperature": random.uniform(0.1, 1.0),
            "max_tokens": random.randint(512, 4096),
            "capabilities": random.sample(
                ["reasoning", "tool_use", "memory", "planning", "learning"], 
                k=random.randint(2, 4)
            ),
            "system_prompt": f"You are a helpful {agent_type} agent for testing purposes."
        }
    
    @staticmethod
    def generate_autonomous_agent_config() -> Dict[str, Any]:
        """Generate a test autonomous agent configuration."""
        base_config = TestDataGenerator.generate_agent_config("autonomous")
        base_config.update({
            "autonomy_level": random.choice(["basic", "adaptive", "autonomous"]),
            "learning_mode": random.choice(["passive", "active", "continuous"]),
            "decision_threshold": random.uniform(0.3, 0.9),
            "enable_proactive_behavior": random.choice([True, False]),
            "enable_goal_setting": random.choice([True, False]),
            "safety_constraints": [
                "respect_user_privacy",
                "avoid_harmful_actions",
                "maintain_ethical_behavior"
            ]
        })
        return base_config
    
    @staticmethod
    def generate_tool_schema(tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a test tool schema."""
        if not tool_name:
            tool_name = f"test_tool_{TestDataGenerator.generate_random_string(5)}"
        
        return {
            "name": tool_name,
            "description": f"A test tool for {tool_name} operations",
            "parameters": {
                "input": {
                    "type": "string",
                    "description": "Input parameter for the tool"
                },
                "options": {
                    "type": "object",
                    "description": "Optional configuration",
                    "default": {}
                }
            },
            "category": random.choice(["utility", "research", "analysis", "communication"]),
            "complexity": random.choice(["simple", "medium", "complex"])
        }
    
    @staticmethod
    def generate_test_documents(count: int = 5) -> List[Dict[str, Any]]:
        """Generate test documents for RAG testing."""
        documents = []
        topics = ["AI", "Machine Learning", "Neural Networks", "Data Science", "Robotics"]
        
        for i in range(count):
            topic = random.choice(topics)
            documents.append({
                "title": f"{topic} - Document {i+1}",
                "content": f"This is a comprehensive document about {topic}. " * 10,
                "metadata": {
                    "category": topic.lower().replace(" ", "_"),
                    "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                    "created_at": datetime.utcnow().isoformat(),
                    "author": f"Test Author {i+1}"
                }
            })
        
        return documents
    
    @staticmethod
    def generate_test_queries(count: int = 10) -> List[str]:
        """Generate test queries for search testing."""
        query_templates = [
            "What is {}?",
            "How does {} work?",
            "Explain {} in detail",
            "What are the benefits of {}?",
            "Compare {} with other approaches",
            "What are the challenges in {}?",
            "How to implement {}?",
            "Best practices for {}",
            "Future of {}",
            "Applications of {}"
        ]
        
        topics = ["artificial intelligence", "machine learning", "neural networks", 
                 "deep learning", "natural language processing", "computer vision",
                 "robotics", "data science", "algorithms", "automation"]
        
        queries = []
        for _ in range(count):
            template = random.choice(query_templates)
            topic = random.choice(topics)
            queries.append(template.format(topic))
        
        return queries


class MockLLMProvider:
    """Mock LLM provider for testing without external dependencies."""
    
    def __init__(self, scenario: MockScenario = None):
        self.scenario = scenario or MockScenario(MockResponseType.SUCCESS)
        self.call_count = 0
        self.call_history = []
    
    async def ainvoke(self, messages, **kwargs):
        """Mock async invoke method."""
        self.call_count += 1
        self.call_history.append({
            "messages": messages,
            "kwargs": kwargs,
            "timestamp": datetime.utcnow()
        })
        
        # Apply delay if specified
        if self.scenario.delay_seconds > 0:
            await asyncio.sleep(self.scenario.delay_seconds)
        
        # Handle different response types
        if self.scenario.response_type == MockResponseType.ERROR:
            raise Exception(self.scenario.error_message or "Mock LLM error")
        
        elif self.scenario.response_type == MockResponseType.TIMEOUT:
            await asyncio.sleep(30)  # Simulate timeout
        
        elif self.scenario.response_type == MockResponseType.RANDOM:
            if random.random() > self.scenario.success_probability:
                raise Exception("Random mock failure")
        
        # Return success response
        response_content = self._generate_response_content(messages)
        return Mock(content=response_content)
    
    def invoke(self, messages, **kwargs):
        """Mock sync invoke method."""
        return asyncio.run(self.ainvoke(messages, **kwargs))
    
    def _generate_response_content(self, messages) -> str:
        """Generate appropriate response content based on input."""
        if self.scenario.response_data:
            return json.dumps(self.scenario.response_data)
        
        # Generate contextual response
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content.lower()
                
                if "calculate" in content or "math" in content:
                    return "I'll perform the calculation: The result is 42."
                elif "search" in content or "find" in content:
                    return "I found relevant information about your query."
                elif "analyze" in content:
                    return "Based on my analysis, here are the key findings..."
                elif "create" in content or "generate" in content:
                    return "I've created the requested content according to your specifications."
                elif "plan" in content or "strategy" in content:
                    return "Here's a comprehensive plan to address your requirements..."
        
        return "I understand your request and will help you with that task."


class MockToolFactory:
    """Mock tool factory for testing tool creation and management."""
    
    def __init__(self):
        self.tools = {}
        self.creation_history = []
    
    async def create_tool_from_schema(self, schema: Dict[str, Any]) -> Mock:
        """Mock tool creation from schema."""
        tool_name = schema.get("name", f"mock_tool_{len(self.tools)}")
        
        tool = Mock()
        tool.name = tool_name
        tool.description = schema.get("description", "Mock tool")
        tool.parameters = schema.get("parameters", {})
        tool.invoke = AsyncMock(return_value=f"Mock result from {tool_name}")
        tool.ainvoke = AsyncMock(return_value=f"Mock async result from {tool_name}")
        
        self.tools[tool_name] = tool
        self.creation_history.append({
            "schema": schema,
            "tool_name": tool_name,
            "timestamp": datetime.utcnow()
        })
        
        return tool
    
    async def create_tool_from_function(self, func: Callable, **kwargs) -> Mock:
        """Mock tool creation from function."""
        tool_name = kwargs.get("name", func.__name__)
        
        tool = Mock()
        tool.name = tool_name
        tool.description = kwargs.get("description", f"Mock tool for {func.__name__}")
        tool.invoke = AsyncMock(return_value=f"Mock result from function {func.__name__}")
        tool.ainvoke = AsyncMock(return_value=f"Mock async result from function {func.__name__}")
        
        self.tools[tool_name] = tool
        return tool
    
    def get_tool(self, tool_name: str) -> Optional[Mock]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())


class MockRAGService:
    """Mock RAG service for testing knowledge operations."""
    
    def __init__(self):
        self.documents = {}
        self.collections = {}
        self.search_history = []
        self.ingestion_history = []
    
    async def initialize(self):
        """Mock initialization."""
        pass
    
    async def ingest_document(self, title: str, content: str, collection: str = "default", 
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock document ingestion."""
        doc_id = str(uuid.uuid4())
        
        document = {
            "id": doc_id,
            "title": title,
            "content": content,
            "collection": collection,
            "metadata": metadata or {},
            "ingested_at": datetime.utcnow()
        }
        
        self.documents[doc_id] = document
        
        if collection not in self.collections:
            self.collections[collection] = []
        self.collections[collection].append(doc_id)
        
        self.ingestion_history.append(document)
        
        return {
            "success": True,
            "document_id": doc_id,
            "collection": collection,
            "message": "Document ingested successfully"
        }
    
    async def search_knowledge(self, query: str, collection: str = None, 
                             top_k: int = 10, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock knowledge search."""
        self.search_history.append({
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "filters": filters,
            "timestamp": datetime.utcnow()
        })
        
        # Generate mock search results
        results = []
        available_docs = list(self.documents.values())
        
        if collection and collection in self.collections:
            available_docs = [self.documents[doc_id] for doc_id in self.collections[collection]]
        
        # Simple mock scoring based on query keywords
        for doc in available_docs[:top_k]:
            score = random.uniform(0.5, 1.0)
            if any(word.lower() in doc["content"].lower() for word in query.split()):
                score = random.uniform(0.8, 1.0)
            
            results.append({
                "content": doc["content"][:200] + "...",
                "metadata": doc["metadata"],
                "score": score,
                "document_id": doc["id"]
            })
        
        return {
            "success": True,
            "results": sorted(results, key=lambda x: x["score"], reverse=True),
            "total_results": len(results),
            "query": query,
            "collection": collection
        }


class TestValidator:
    """Utility class for validating test results and behaviors."""
    
    @staticmethod
    def validate_agent_response(response: Any, min_length: int = 10) -> bool:
        """Validate that an agent response is meaningful."""
        if not response:
            return False
        
        content = ""
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        
        return len(content.strip()) >= min_length
    
    @staticmethod
    def validate_tool_result(result: Any) -> bool:
        """Validate that a tool execution result is valid."""
        return result is not None
    
    @staticmethod
    def validate_search_results(results: List[Dict[str, Any]], min_score: float = 0.5) -> bool:
        """Validate search results quality."""
        if not results:
            return False
        
        for result in results:
            if not isinstance(result, dict):
                return False
            if 'content' not in result or not result['content']:
                return False
            if 'score' not in result or result['score'] < min_score:
                return False
        
        return True
    
    @staticmethod
    def validate_autonomous_behavior(responses: List[Any], min_variation: float = 0.7) -> bool:
        """Validate that responses show autonomous behavior (not scripted)."""
        if len(responses) < 2:
            return False
        
        # Check for variation in responses
        unique_responses = set(str(response) for response in responses)
        variation_ratio = len(unique_responses) / len(responses)
        
        return variation_ratio >= min_variation
    
    @staticmethod
    def validate_performance_metrics(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
        """Validate performance metrics against thresholds."""
        for metric, threshold in thresholds.items():
            if metric not in metrics:
                return False
            if metrics[metric] > threshold:
                return False
        
        return True


class AsyncTestHelper:
    """Helper class for async test operations."""
    
    @staticmethod
    async def wait_for_condition(condition_func: Callable, timeout: float = 10.0, 
                                interval: float = 0.1) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if await condition_func():
                    return True
            except Exception as e:
                logger.debug(f"Condition check failed: {e}")
            
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def run_concurrent_operations(operations: List[Callable], 
                                      max_concurrent: int = 10) -> List[Any]:
        """Run operations concurrently with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation()
        
        return await asyncio.gather(*[
            run_with_semaphore(op) for op in operations
        ], return_exceptions=True)
    
    @staticmethod
    async def measure_execution_time(operation: Callable) -> Tuple[Any, float]:
        """Measure execution time of an async operation."""
        start_time = time.time()
        result = await operation()
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    async def retry_operation(operation: Callable, max_retries: int = 3, 
                            delay: float = 1.0) -> Any:
        """Retry an operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = delay * (2 ** attempt)
                logger.debug(f"Operation failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)


class TestReporter:
    """Utility class for generating test reports."""
    
    @staticmethod
    def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary report from test results."""
        total_tests = len(results)
        passed_tests = sum(1 for result in results if result.get('passed', False))
        failed_tests = total_tests - passed_tests
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate average execution time
        execution_times = [result.get('duration', 0) for result in results]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Group by category
        categories = {}
        for result in results:
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = {'total': 0, 'passed': 0, 'failed': 0}
            
            categories[category]['total'] += 1
            if result.get('passed', False):
                categories[category]['passed'] += 1
            else:
                categories[category]['failed'] += 1
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'average_execution_time': avg_execution_time
            },
            'categories': categories,
            'detailed_results': results,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def export_results_to_json(results: Dict[str, Any], filename: str):
        """Export test results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results exported to {filename}")


# Convenience functions for common test patterns
async def create_mock_agent(agent_type: str = "test", **kwargs) -> Mock:
    """Create a mock agent for testing."""
    agent = Mock()
    agent.agent_id = TestDataGenerator.generate_agent_id()
    agent.config = TestDataGenerator.generate_agent_config(agent_type)
    agent.config.update(kwargs)
    agent.invoke = AsyncMock(return_value=Mock(content="Mock agent response"))
    agent.ainvoke = AsyncMock(return_value=Mock(content="Mock async agent response"))
    agent.initialize = AsyncMock()
    return agent


async def create_mock_tool(tool_name: str = None, **kwargs) -> Mock:
    """Create a mock tool for testing."""
    if not tool_name:
        tool_name = f"mock_tool_{TestDataGenerator.generate_random_string(5)}"
    
    tool = Mock()
    tool.name = tool_name
    tool.description = kwargs.get("description", f"Mock tool: {tool_name}")
    tool.invoke = AsyncMock(return_value=f"Mock result from {tool_name}")
    tool.ainvoke = AsyncMock(return_value=f"Mock async result from {tool_name}")
    return tool


def create_test_scenario(scenario_type: str, **kwargs) -> Dict[str, Any]:
    """Create a test scenario configuration."""
    scenarios = {
        "basic_agent_creation": {
            "description": "Test basic agent creation and initialization",
            "agent_config": TestDataGenerator.generate_agent_config(),
            "expected_capabilities": ["reasoning", "tool_use"]
        },
        "autonomous_agent_behavior": {
            "description": "Test autonomous agent behavior and decision making",
            "agent_config": TestDataGenerator.generate_autonomous_agent_config(),
            "test_tasks": ["analyze data", "make decision", "plan strategy"]
        },
        "tool_integration": {
            "description": "Test tool creation and integration with agents",
            "tools": [TestDataGenerator.generate_tool_schema() for _ in range(3)],
            "test_operations": ["create", "assign", "execute"]
        },
        "rag_knowledge_management": {
            "description": "Test RAG system and knowledge management",
            "documents": TestDataGenerator.generate_test_documents(),
            "queries": TestDataGenerator.generate_test_queries()
        }
    }
    
    scenario = scenarios.get(scenario_type, {})
    scenario.update(kwargs)
    return scenario
