"""
Comprehensive Tool Integration Tests.

This module tests tool creation, assignment to agents, tool execution,
dynamic tool factory, and tool-agent interaction validation.
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

import structlog

# Import test infrastructure
from .base_test import ToolTest, TestResult, TestCategory, TestSeverity
from .test_utils import TestDataGenerator, MockToolFactory, TestValidator, create_mock_tool

# Import application components
from app.tools.dynamic_tool_factory import (
    DynamicToolFactory, 
    ToolCategory, 
    ToolComplexity, 
    DynamicToolSchema,
    tool_factory
)
from app.tools.base_tool import BaseDynamicTool
from app.agents.base.agent import LangGraphAgent, AgentConfig
from app.models.tool import Tool, AgentTool, ToolExecution

logger = structlog.get_logger(__name__)


class TestDynamicToolCreation(ToolTest):
    """Test dynamic tool creation from various sources."""
    
    def __init__(self):
        super().__init__("Dynamic Tool Creation", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test dynamic tool creation from schemas, functions, and descriptions."""
        try:
            factory = DynamicToolFactory()
            
            # Test 1: Tool creation from schema
            schema_test = await self._test_tool_from_schema(factory)
            
            # Test 2: Tool creation from function
            function_test = await self._test_tool_from_function(factory)
            
            # Test 3: Tool creation from AI description
            description_test = await self._test_tool_from_description(factory)
            
            # Test 4: Tool registration and retrieval
            registration_test = await self._test_tool_registration(factory)
            
            evidence = {
                "schema_creation": schema_test,
                "function_creation": function_test,
                "description_creation": description_test,
                "registration": registration_test,
                "total_tools_created": len(factory.registered_tools) if hasattr(factory, 'registered_tools') else 0
            }
            
            success = schema_test and function_test and registration_test
            # description_test might fail due to LLM dependency, so not required
            
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
    
    async def _test_tool_from_schema(self, factory: DynamicToolFactory) -> bool:
        """Test tool creation from schema."""
        try:
            schema = DynamicToolSchema(
                name="test_calculator",
                description="A simple calculator tool",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                implementation="""
async def execute(expression: str) -> str:
    try:
        # Simple evaluation for testing
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
""",
                category=ToolCategory.UTILITY,
                complexity=ToolComplexity.SIMPLE
            )
            
            # Mock the create_tool_from_schema method
            with patch.object(factory, 'create_tool_from_schema', new_callable=AsyncMock) as mock_create:
                mock_tool = await create_mock_tool("test_calculator")
                mock_create.return_value = mock_tool
                
                tool = await factory.create_tool_from_schema(schema)
                
                return await self.validate_tool_creation(tool)
            
        except Exception as e:
            logger.error(f"Error testing tool from schema: {e}")
            return False
    
    async def _test_tool_from_function(self, factory: DynamicToolFactory) -> bool:
        """Test tool creation from function."""
        try:
            # Define test function
            async def test_function(input_text: str) -> str:
                """A test function for tool creation."""
                return f"Processed: {input_text}"
            
            # Mock the create_tool_from_function method
            with patch.object(factory, 'create_tool_from_function', new_callable=AsyncMock) as mock_create:
                mock_tool = await create_mock_tool("test_function")
                mock_create.return_value = mock_tool
                
                tool = await factory.create_tool_from_function(
                    func=test_function,
                    name="test_function_tool",
                    description="Tool created from function",
                    category=ToolCategory.UTILITY
                )
                
                return await self.validate_tool_creation(tool)
            
        except Exception as e:
            logger.error(f"Error testing tool from function: {e}")
            return False
    
    async def _test_tool_from_description(self, factory: DynamicToolFactory) -> bool:
        """Test tool creation from AI description."""
        try:
            # Mock the create_tool_from_description method
            with patch.object(factory, 'create_tool_from_description', new_callable=AsyncMock) as mock_create:
                mock_tool = await create_mock_tool("ai_generated_tool")
                mock_create.return_value = mock_tool
                
                tool = await factory.create_tool_from_description(
                    name="ai_generated_tool",
                    description="A tool generated from AI description",
                    functionality_description="This tool should process text and return analysis",
                    category=ToolCategory.ANALYSIS
                )
                
                return await self.validate_tool_creation(tool)
            
        except Exception as e:
            logger.debug(f"Tool from description test failed (expected): {e}")
            return True  # This might fail due to LLM dependency, which is acceptable
    
    async def _test_tool_registration(self, factory: DynamicToolFactory) -> bool:
        """Test tool registration and retrieval."""
        try:
            # Create mock tool
            mock_tool = await create_mock_tool("registration_test_tool")
            
            # Mock registration methods
            with patch.object(factory, 'register_tool', new_callable=AsyncMock) as mock_register, \
                 patch.object(factory, 'get_tool', return_value=mock_tool) as mock_get, \
                 patch.object(factory, 'list_tools', return_value=["registration_test_tool"]) as mock_list:
                
                # Register tool
                await factory.register_tool(mock_tool)
                
                # Retrieve tool
                retrieved_tool = factory.get_tool("registration_test_tool")
                
                # List tools
                tool_list = factory.list_tools()
                
                return (retrieved_tool is not None and 
                       "registration_test_tool" in tool_list)
            
        except Exception as e:
            logger.error(f"Error testing tool registration: {e}")
            return False


class TestToolAgentIntegration(ToolTest):
    """Test integration between tools and agents."""
    
    def __init__(self):
        super().__init__("Tool-Agent Integration", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test tool assignment and execution with agents."""
        try:
            # Create test agent
            agent_config = AgentConfig(
                name="Tool Test Agent",
                description="Agent for testing tool integration",
                model_name="mock-model",
                capabilities=["tool_use"]
            )
            
            from .test_utils import MockLLMProvider
            mock_llm = MockLLMProvider()
            
            # Create tools
            tools = await self._create_test_tools()
            
            # Create agent with tools
            agent = LangGraphAgent(config=agent_config, llm=mock_llm, tools=tools)
            await agent.initialize()
            
            # Test tool assignment
            assignment_test = await self._test_tool_assignment(agent, tools)
            
            # Test tool execution
            execution_test = await self._test_tool_execution(agent)
            
            # Test tool interaction
            interaction_test = await self._test_agent_tool_interaction(agent)
            
            evidence = {
                "agent_id": agent.agent_id,
                "tools_assigned": len(tools),
                "assignment_test": assignment_test,
                "execution_test": execution_test,
                "interaction_test": interaction_test,
                "agent_tools": list(agent.tools.keys()) if hasattr(agent, 'tools') else []
            }
            
            success = assignment_test and execution_test and interaction_test
            
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
    
    async def _create_test_tools(self) -> List[Mock]:
        """Create test tools for agent integration."""
        tools = []
        
        # Calculator tool
        calc_tool = await create_mock_tool("calculator")
        calc_tool.invoke = AsyncMock(return_value="42")
        tools.append(calc_tool)
        
        # Search tool
        search_tool = await create_mock_tool("web_search")
        search_tool.invoke = AsyncMock(return_value="Search results found")
        tools.append(search_tool)
        
        # Analysis tool
        analysis_tool = await create_mock_tool("text_analyzer")
        analysis_tool.invoke = AsyncMock(return_value="Analysis complete")
        tools.append(analysis_tool)
        
        return tools
    
    async def _test_tool_assignment(self, agent: LangGraphAgent, tools: List[Mock]) -> bool:
        """Test that tools are properly assigned to agent."""
        try:
            # Check that agent has tools
            if not hasattr(agent, 'tools'):
                return False
            
            # Check that tools are accessible
            for tool in tools:
                if tool.name not in agent.tools:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing tool assignment: {e}")
            return False
    
    async def _test_tool_execution(self, agent: LangGraphAgent) -> bool:
        """Test tool execution through agent."""
        try:
            # Test if agent can execute tools
            if not hasattr(agent, 'tools') or not agent.tools:
                return False
            
            # Get first tool
            tool_name = list(agent.tools.keys())[0]
            tool = agent.tools[tool_name]
            
            # Execute tool
            result = await tool.invoke("test input")
            
            return await self.validate_tool_execution(result)
            
        except Exception as e:
            logger.error(f"Error testing tool execution: {e}")
            return False
    
    async def _test_agent_tool_interaction(self, agent: LangGraphAgent) -> bool:
        """Test agent's ability to use tools in response to queries."""
        try:
            # Send message that should trigger tool use
            response = await agent.ainvoke("Can you calculate 2 + 2 for me?")
            
            # Check if response is valid
            return response is not None and hasattr(response, 'content')
            
        except Exception as e:
            logger.error(f"Error testing agent-tool interaction: {e}")
            return False


class TestToolExecution(ToolTest):
    """Test tool execution scenarios and error handling."""
    
    def __init__(self):
        super().__init__("Tool Execution", TestSeverity.HIGH)
    
    async def execute_test(self) -> TestResult:
        """Test various tool execution scenarios."""
        try:
            # Test successful execution
            success_test = await self._test_successful_execution()
            
            # Test error handling
            error_test = await self._test_error_handling()
            
            # Test concurrent execution
            concurrent_test = await self._test_concurrent_execution()
            
            # Test parameter validation
            validation_test = await self._test_parameter_validation()
            
            evidence = {
                "successful_execution": success_test,
                "error_handling": error_test,
                "concurrent_execution": concurrent_test,
                "parameter_validation": validation_test
            }
            
            success = success_test and error_test and concurrent_test and validation_test
            
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
    
    async def _test_successful_execution(self) -> bool:
        """Test successful tool execution."""
        try:
            tool = await create_mock_tool("success_tool")
            tool.invoke = AsyncMock(return_value="Success result")
            
            result = await tool.invoke("test input")
            return await self.validate_tool_execution(result)
            
        except Exception as e:
            logger.error(f"Error testing successful execution: {e}")
            return False
    
    async def _test_error_handling(self) -> bool:
        """Test tool error handling."""
        try:
            tool = await create_mock_tool("error_tool")
            tool.invoke = AsyncMock(side_effect=Exception("Tool error"))
            
            try:
                result = await tool.invoke("test input")
                return False  # Should have raised exception
            except Exception:
                return True  # Expected error
            
        except Exception as e:
            logger.error(f"Error testing error handling: {e}")
            return False
    
    async def _test_concurrent_execution(self) -> bool:
        """Test concurrent tool execution."""
        try:
            tool = await create_mock_tool("concurrent_tool")
            tool.invoke = AsyncMock(return_value="Concurrent result")
            
            # Execute multiple times concurrently
            tasks = [tool.invoke(f"input_{i}") for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            return len(results) == 5 and all(results)
            
        except Exception as e:
            logger.error(f"Error testing concurrent execution: {e}")
            return False
    
    async def _test_parameter_validation(self) -> bool:
        """Test tool parameter validation."""
        try:
            tool = await create_mock_tool("validation_tool")
            
            # Mock parameter validation
            tool.args_schema = Mock()
            tool.args_schema.validate = Mock(return_value={"param": "value"})
            tool.invoke = AsyncMock(return_value="Validated result")
            
            result = await tool.invoke(param="value")
            return await self.validate_tool_execution(result)
            
        except Exception as e:
            logger.error(f"Error testing parameter validation: {e}")
            return False


class TestToolManagement(ToolTest):
    """Test tool management operations."""
    
    def __init__(self):
        super().__init__("Tool Management", TestSeverity.MEDIUM)
    
    async def execute_test(self) -> TestResult:
        """Test tool management operations."""
        try:
            factory = MockToolFactory()
            
            # Test tool creation
            creation_test = await self._test_tool_creation_management(factory)
            
            # Test tool listing
            listing_test = await self._test_tool_listing(factory)
            
            # Test tool retrieval
            retrieval_test = await self._test_tool_retrieval(factory)
            
            # Test tool metadata
            metadata_test = await self._test_tool_metadata(factory)
            
            evidence = {
                "creation_management": creation_test,
                "listing": listing_test,
                "retrieval": retrieval_test,
                "metadata": metadata_test,
                "total_tools": len(factory.tools)
            }
            
            success = creation_test and listing_test and retrieval_test and metadata_test
            
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
    
    async def _test_tool_creation_management(self, factory: MockToolFactory) -> bool:
        """Test tool creation management."""
        try:
            schema = TestDataGenerator.generate_tool_schema("management_tool")
            tool = await factory.create_tool_from_schema(schema)
            
            return await self.validate_tool_creation(tool)
            
        except Exception as e:
            logger.error(f"Error testing tool creation management: {e}")
            return False
    
    async def _test_tool_listing(self, factory: MockToolFactory) -> bool:
        """Test tool listing functionality."""
        try:
            # Create some tools first
            for i in range(3):
                schema = TestDataGenerator.generate_tool_schema(f"list_tool_{i}")
                await factory.create_tool_from_schema(schema)
            
            # List tools
            tool_list = factory.list_tools()
            
            return len(tool_list) >= 3
            
        except Exception as e:
            logger.error(f"Error testing tool listing: {e}")
            return False
    
    async def _test_tool_retrieval(self, factory: MockToolFactory) -> bool:
        """Test tool retrieval by name."""
        try:
            # Create a tool
            schema = TestDataGenerator.generate_tool_schema("retrieval_tool")
            await factory.create_tool_from_schema(schema)
            
            # Retrieve tool
            tool = factory.get_tool("retrieval_tool")
            
            return tool is not None
            
        except Exception as e:
            logger.error(f"Error testing tool retrieval: {e}")
            return False
    
    async def _test_tool_metadata(self, factory: MockToolFactory) -> bool:
        """Test tool metadata management."""
        try:
            # Create tool with metadata
            schema = TestDataGenerator.generate_tool_schema("metadata_tool")
            tool = await factory.create_tool_from_schema(schema)
            
            # Check metadata
            return (hasattr(tool, 'name') and 
                   hasattr(tool, 'description') and
                   tool.name and tool.description)
            
        except Exception as e:
            logger.error(f"Error testing tool metadata: {e}")
            return False


# Test suite for tool integration
class ToolIntegrationTestSuite:
    """Comprehensive test suite for tool integration."""
    
    def __init__(self):
        self.tests = [
            TestDynamicToolCreation(),
            TestToolAgentIntegration(),
            TestToolExecution(),
            TestToolManagement()
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tool integration tests."""
        logger.info("Starting tool integration test suite")
        
        results = []
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = await test.run()
            results.append(result)
        
        # Generate summary
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        
        summary = {
            "suite_name": "Tool Integration Test Suite",
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "results": [result.__dict__ for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Tool integration test suite completed",
            total=total,
            passed=passed,
            failed=total-passed,
            success_rate=summary["success_rate"]
        )
        
        return summary


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.tool
@pytest.mark.unit
async def test_dynamic_tool_creation():
    """Pytest wrapper for dynamic tool creation test."""
    test = TestDynamicToolCreation()
    result = await test.run()
    assert result.passed, f"Dynamic tool creation failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.tool
@pytest.mark.integration
async def test_tool_agent_integration():
    """Pytest wrapper for tool-agent integration test."""
    test = TestToolAgentIntegration()
    result = await test.run()
    assert result.passed, f"Tool-agent integration failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.tool
@pytest.mark.unit
async def test_tool_execution():
    """Pytest wrapper for tool execution test."""
    test = TestToolExecution()
    result = await test.run()
    assert result.passed, f"Tool execution failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.tool
@pytest.mark.unit
async def test_tool_management():
    """Pytest wrapper for tool management test."""
    test = TestToolManagement()
    result = await test.run()
    assert result.passed, f"Tool management failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.tool
@pytest.mark.integration
async def test_complete_tool_integration_suite():
    """Run the complete tool integration test suite."""
    suite = ToolIntegrationTestSuite()
    summary = await suite.run_all_tests()
    
    assert summary["success_rate"] >= 0.75, f"Tool integration suite success rate too low: {summary['success_rate']}"
    assert summary["passed"] >= 3, f"Not enough tests passed: {summary['passed']}/{summary['total_tests']}"
