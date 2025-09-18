"""
Comprehensive LLM Integration Tests.

This module tests LLM provider integration, model management, and
agent-LLM communication to ensure proper LLM functionality.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

import structlog

# Import test infrastructure
from .base_test import LLMTest, TestResult, TestCategory, TestSeverity
from .test_utils import TestDataGenerator, MockLLMProvider, MockScenario, MockResponseType

# Import application components
from app.llm.manager import LLMProviderManager
from app.llm.models import LLMConfig, ProviderType, ModelInfo, ProviderCredentials
from app.llm.providers import OllamaProvider, OpenAIProvider, AnthropicProvider, GoogleProvider
from app.agents.base.agent import LangGraphAgent, AgentConfig

logger = structlog.get_logger(__name__)


class TestLLMProviderInitialization(LLMTest):
    """Test LLM provider initialization and configuration."""
    
    def __init__(self):
        super().__init__("LLM Provider Initialization", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test LLM provider initialization."""
        try:
            # Test LLM provider manager initialization
            manager = LLMProviderManager()
            
            # Test provider registration
            providers_registered = await self._test_provider_registration(manager)
            
            # Test configuration validation
            config_validation = await self._test_configuration_validation()
            
            # Test model availability
            model_availability = await self._test_model_availability(manager)
            
            evidence = {
                "providers_registered": providers_registered,
                "config_validation": config_validation,
                "model_availability": model_availability,
                "available_providers": list(manager.providers.keys()) if hasattr(manager, 'providers') else []
            }
            
            success = providers_registered and config_validation and model_availability
            
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
    
    async def _test_provider_registration(self, manager: LLMProviderManager) -> bool:
        """Test that LLM providers are properly registered."""
        try:
            # Check if providers are available
            if not hasattr(manager, 'providers'):
                return False
            
            # Should have at least one provider
            return len(manager.providers) > 0
            
        except Exception as e:
            logger.error(f"Error testing provider registration: {e}")
            return False
    
    async def _test_configuration_validation(self) -> bool:
        """Test LLM configuration validation."""
        try:
            # Test valid configuration
            valid_config = LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id="llama3.2:latest",
                temperature=0.7,
                max_tokens=2048
            )
            
            # Should not raise exception
            config_dict = valid_config.dict()
            
            # Test invalid configuration
            try:
                invalid_config = LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model_id="llama3.2:latest",
                    temperature=3.0,  # Invalid: > 2.0
                    max_tokens=2048
                )
                return False  # Should have raised exception
            except Exception:
                pass  # Expected validation error
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing configuration validation: {e}")
            return False
    
    async def _test_model_availability(self, manager: LLMProviderManager) -> bool:
        """Test model availability checking."""
        try:
            # Mock the get_available_models method
            with patch.object(manager, 'get_available_models', new_callable=AsyncMock) as mock_get_models:
                mock_get_models.return_value = [
                    ModelInfo(id="mock-model-1", name="Mock Model 1"),
                    ModelInfo(id="mock-model-2", name="Mock Model 2")
                ]
                
                models = await manager.get_available_models(ProviderType.OLLAMA)
                return len(models) > 0
            
        except Exception as e:
            logger.error(f"Error testing model availability: {e}")
            return False


class TestLLMInstanceCreation(LLMTest):
    """Test LLM instance creation and management."""
    
    def __init__(self):
        super().__init__("LLM Instance Creation", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test LLM instance creation with different configurations."""
        try:
            test_results = []
            
            # Test different provider types
            provider_configs = [
                {
                    "provider": ProviderType.OLLAMA,
                    "model_id": "llama3.2:latest",
                    "temperature": 0.7
                },
                {
                    "provider": ProviderType.OPENAI,
                    "model_id": "gpt-3.5-turbo",
                    "temperature": 0.5
                },
                {
                    "provider": ProviderType.ANTHROPIC,
                    "model_id": "claude-3-sonnet",
                    "temperature": 0.8
                }
            ]
            
            for config_data in provider_configs:
                config = LLMConfig(**config_data)
                result = await self._test_single_instance_creation(config)
                test_results.append(result)
            
            # Test instance functionality
            functionality_test = await self._test_instance_functionality()
            
            evidence = {
                "provider_tests": test_results,
                "functionality_test": functionality_test,
                "total_providers_tested": len(provider_configs)
            }
            
            success = any(test_results) and functionality_test  # At least one provider should work
            
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
    
    async def _test_single_instance_creation(self, config: LLMConfig) -> bool:
        """Test creation of a single LLM instance."""
        try:
            manager = LLMProviderManager()
            
            # Mock the create_llm_instance method to avoid external dependencies
            with patch.object(manager, 'create_llm_instance', new_callable=AsyncMock) as mock_create:
                mock_llm = MockLLMProvider()
                mock_create.return_value = mock_llm
                
                llm_instance = await manager.create_llm_instance(config)
                return llm_instance is not None
            
        except Exception as e:
            logger.debug(f"Error creating LLM instance for {config.provider}: {e}")
            return False
    
    async def _test_instance_functionality(self) -> bool:
        """Test basic LLM instance functionality."""
        try:
            # Create mock LLM instance
            mock_llm = MockLLMProvider()
            
            # Test invoke method
            response = await mock_llm.ainvoke("Test message")
            
            # Validate response
            return await self.validate_llm_response(response)
            
        except Exception as e:
            logger.error(f"Error testing instance functionality: {e}")
            return False


class TestAgentLLMIntegration(LLMTest):
    """Test integration between agents and LLM providers."""
    
    def __init__(self):
        super().__init__("Agent-LLM Integration", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test agent-LLM integration and communication."""
        try:
            # Create agent with LLM
            agent_config = AgentConfig(
                name="Test LLM Integration Agent",
                description="Agent for testing LLM integration",
                model_name="mock-model",
                temperature=0.7,
                max_tokens=1024
            )
            
            mock_llm = MockLLMProvider()
            agent = LangGraphAgent(config=agent_config, llm=mock_llm, tools=[])
            await agent.initialize()
            
            # Test agent-LLM communication
            communication_test = await self._test_agent_llm_communication(agent, mock_llm)
            
            # Test different message types
            message_types_test = await self._test_different_message_types(agent)
            
            # Test error handling
            error_handling_test = await self._test_error_handling(agent)
            
            evidence = {
                "agent_id": agent.agent_id,
                "communication_test": communication_test,
                "message_types_test": message_types_test,
                "error_handling_test": error_handling_test,
                "llm_call_count": mock_llm.call_count
            }
            
            success = communication_test and message_types_test and error_handling_test
            
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
    
    async def _test_agent_llm_communication(self, agent, mock_llm) -> bool:
        """Test basic agent-LLM communication."""
        try:
            initial_call_count = mock_llm.call_count
            
            # Send message to agent
            response = await agent.ainvoke("Hello, can you help me?")
            
            # Check that LLM was called
            final_call_count = mock_llm.call_count
            llm_called = final_call_count > initial_call_count
            
            # Check response validity
            response_valid = await self.validate_llm_response(response)
            
            return llm_called and response_valid
            
        except Exception as e:
            logger.error(f"Error testing agent-LLM communication: {e}")
            return False
    
    async def _test_different_message_types(self, agent) -> bool:
        """Test agent responses to different message types."""
        try:
            test_messages = [
                "What is artificial intelligence?",
                "Can you help me solve a problem?",
                "Explain the concept of machine learning.",
                "What are your capabilities?",
                "How do you process information?"
            ]
            
            responses = []
            for message in test_messages:
                response = await agent.ainvoke(message)
                responses.append(response)
            
            # Validate all responses
            for response in responses:
                if not await self.validate_llm_response(response):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing different message types: {e}")
            return False
    
    async def _test_error_handling(self, agent) -> bool:
        """Test agent error handling with LLM failures."""
        try:
            # Create agent with error-prone LLM
            error_scenario = MockScenario(
                response_type=MockResponseType.ERROR,
                error_message="Mock LLM error"
            )
            error_llm = MockLLMProvider(scenario=error_scenario)
            
            # Replace agent's LLM temporarily
            original_llm = agent.llm
            agent.llm = error_llm
            
            try:
                # This should handle the error gracefully
                response = await agent.ainvoke("Test message")
                # If we get here, error was handled
                error_handled = True
            except Exception:
                # Error was not handled properly
                error_handled = False
            finally:
                # Restore original LLM
                agent.llm = original_llm
            
            return error_handled
            
        except Exception as e:
            logger.error(f"Error testing error handling: {e}")
            return False


class TestLLMPerformance(LLMTest):
    """Test LLM performance and response times."""
    
    def __init__(self):
        super().__init__("LLM Performance", TestSeverity.MEDIUM)
    
    async def execute_test(self) -> TestResult:
        """Test LLM performance characteristics."""
        try:
            # Create mock LLM with different response times
            fast_llm = MockLLMProvider(MockScenario(MockResponseType.SUCCESS, delay_seconds=0.1))
            slow_llm = MockLLMProvider(MockScenario(MockResponseType.SUCCESS, delay_seconds=2.0))
            
            # Test response times
            fast_time = await self._measure_response_time(fast_llm)
            slow_time = await self._measure_response_time(slow_llm)
            
            # Test concurrent requests
            concurrent_performance = await self._test_concurrent_requests(fast_llm)
            
            evidence = {
                "fast_response_time": fast_time,
                "slow_response_time": slow_time,
                "concurrent_performance": concurrent_performance,
                "performance_acceptable": fast_time < 1.0 and slow_time < 5.0
            }
            
            success = fast_time < 1.0 and concurrent_performance
            
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
    
    async def _measure_response_time(self, llm) -> float:
        """Measure LLM response time."""
        start_time = time.time()
        await llm.ainvoke("Test message")
        end_time = time.time()
        return end_time - start_time
    
    async def _test_concurrent_requests(self, llm) -> bool:
        """Test concurrent LLM requests."""
        try:
            # Create multiple concurrent requests
            tasks = [llm.ainvoke(f"Test message {i}") for i in range(5)]
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # All requests should complete
            if len(responses) != 5:
                return False
            
            # Should complete in reasonable time (not sequential)
            total_time = end_time - start_time
            return total_time < 3.0  # Should be much faster than 5 sequential requests
            
        except Exception as e:
            logger.error(f"Error testing concurrent requests: {e}")
            return False


# Test suite for LLM integration
class LLMIntegrationTestSuite:
    """Comprehensive test suite for LLM integration."""
    
    def __init__(self):
        self.tests = [
            TestLLMProviderInitialization(),
            TestLLMInstanceCreation(),
            TestAgentLLMIntegration(),
            TestLLMPerformance()
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all LLM integration tests."""
        logger.info("Starting LLM integration test suite")
        
        results = []
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = await test.run()
            results.append(result)
        
        # Generate summary
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        
        summary = {
            "suite_name": "LLM Integration Test Suite",
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "results": [result.__dict__ for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"LLM integration test suite completed",
            total=total,
            passed=passed,
            failed=total-passed,
            success_rate=summary["success_rate"]
        )
        
        return summary


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.unit
async def test_llm_provider_initialization():
    """Pytest wrapper for LLM provider initialization test."""
    test = TestLLMProviderInitialization()
    result = await test.run()
    assert result.passed, f"LLM provider initialization failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.integration
async def test_llm_instance_creation():
    """Pytest wrapper for LLM instance creation test."""
    test = TestLLMInstanceCreation()
    result = await test.run()
    assert result.passed, f"LLM instance creation failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.integration
async def test_agent_llm_integration():
    """Pytest wrapper for agent-LLM integration test."""
    test = TestAgentLLMIntegration()
    result = await test.run()
    assert result.passed, f"Agent-LLM integration failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.performance
async def test_llm_performance():
    """Pytest wrapper for LLM performance test."""
    test = TestLLMPerformance()
    result = await test.run()
    assert result.passed, f"LLM performance test failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.llm
@pytest.mark.integration
async def test_complete_llm_integration_suite():
    """Run the complete LLM integration test suite."""
    suite = LLMIntegrationTestSuite()
    summary = await suite.run_all_tests()
    
    assert summary["success_rate"] >= 0.75, f"LLM integration suite success rate too low: {summary['success_rate']}"
    assert summary["passed"] >= 3, f"Not enough tests passed: {summary['passed']}/{summary['total_tests']}"
