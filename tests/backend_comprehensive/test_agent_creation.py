"""
Comprehensive Agent Creation and Configuration Tests.

This module tests all aspects of agent creation, configuration, and lifecycle
management to ensure agents are properly created and configured with LLMs.
"""

import pytest
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

import structlog
from unittest.mock import Mock, AsyncMock, patch

# Import test infrastructure
from .base_test import AgentTest, TestResult, TestCategory, TestSeverity
from .test_utils import TestDataGenerator, MockLLMProvider, TestValidator, create_mock_agent

# Import application components
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType
from app.orchestration.orchestrator import orchestrator
from app.agents.base.agent import LangGraphAgent, AgentConfig, AgentCapability
from app.agents.autonomous import (
    AutonomousLangGraphAgent, 
    AutonomousAgentConfig, 
    AutonomyLevel, 
    LearningMode,
    create_autonomous_agent,
    create_research_agent,
    create_creative_agent,
    create_optimization_agent
)
from app.llm.manager import LLMProviderManager
from app.llm.models import LLMConfig, ProviderType
from app.models.agent import Agent

logger = structlog.get_logger(__name__)


class TestBasicAgentCreation(AgentTest):
    """Test basic agent creation functionality."""
    
    def __init__(self):
        super().__init__("Basic Agent Creation", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test basic agent creation with LLM integration."""
        try:
            # Create test configuration
            config = AgentConfig(
                name="Test Basic Agent",
                description="A basic test agent",
                agent_type="basic",
                model_name="mock-model",
                temperature=0.7,
                max_tokens=1024,
                capabilities=["reasoning", "tool_use"],
                system_prompt="You are a helpful test agent."
            )
            
            # Create mock LLM
            mock_llm = MockLLMProvider()
            
            # Create agent
            agent = LangGraphAgent(
                config=config,
                llm=mock_llm,
                tools=[]
            )
            
            # Initialize agent
            await agent.initialize()
            
            # Validate agent creation
            validation_passed = await self.validate_agent_creation(agent)
            
            # Test agent response
            response = await agent.ainvoke("Hello, can you help me?")
            response_valid = await self.validate_agent_response(response)
            
            evidence = {
                "agent_id": agent.agent_id,
                "config": agent.config.dict(),
                "initialization_successful": agent.graph is not None,
                "response_content": response.content if hasattr(response, 'content') else str(response),
                "llm_call_count": mock_llm.call_count
            }
            
            success = validation_passed and response_valid and agent.graph is not None
            
            return TestResult(
                test_name=self.test_name,
                category=self.category,
                severity=self.severity,
                passed=success,
                duration=0.0,  # Will be set by base class
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


class TestAutonomousAgentCreation(AgentTest):
    """Test autonomous agent creation with advanced capabilities."""
    
    def __init__(self):
        super().__init__("Autonomous Agent Creation", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test autonomous agent creation with full configuration."""
        try:
            # Create autonomous agent configuration
            config = AutonomousAgentConfig(
                name="Test Autonomous Agent",
                description="An autonomous test agent",
                autonomy_level=AutonomyLevel.ADAPTIVE,
                learning_mode=LearningMode.ACTIVE,
                decision_threshold=0.7,
                capabilities=["reasoning", "tool_use", "memory", "planning"],
                enable_proactive_behavior=True,
                enable_goal_setting=True,
                safety_constraints=[
                    "respect_user_privacy",
                    "avoid_harmful_actions"
                ]
            )
            
            # Create mock LLM
            mock_llm = MockLLMProvider()
            
            # Create autonomous agent
            agent = AutonomousLangGraphAgent(
                config=config,
                llm=mock_llm,
                tools=[]
            )
            
            # Initialize agent
            await agent.initialize()
            
            # Validate agent creation
            validation_passed = await self.validate_agent_creation(agent)
            
            # Test autonomous capabilities
            autonomous_features_valid = await self._validate_autonomous_features(agent)
            
            # Test agent decision making
            decision_result = await self._test_decision_making(agent)
            
            evidence = {
                "agent_id": agent.agent_id,
                "config": agent.config.dict(),
                "autonomy_level": config.autonomy_level.value,
                "learning_mode": config.learning_mode.value,
                "autonomous_features": autonomous_features_valid,
                "decision_making": decision_result,
                "initialization_successful": agent.graph is not None
            }
            
            success = (validation_passed and autonomous_features_valid and 
                      decision_result and agent.graph is not None)
            
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
    
    async def _validate_autonomous_features(self, agent: AutonomousLangGraphAgent) -> bool:
        """Validate autonomous agent specific features."""
        try:
            # Check autonomous configuration
            if not hasattr(agent.config, 'autonomy_level'):
                return False
            
            if not hasattr(agent.config, 'learning_mode'):
                return False
            
            if not hasattr(agent.config, 'decision_threshold'):
                return False
            
            # Check autonomous capabilities
            if not agent.config.capabilities:
                return False
            
            required_capabilities = ["reasoning", "memory", "planning"]
            for capability in required_capabilities:
                if capability not in agent.config.capabilities:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating autonomous features: {e}")
            return False
    
    async def _test_decision_making(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test agent decision making capabilities."""
        try:
            # Test with a decision-requiring scenario
            scenario = "You need to choose between three options: A, B, or C. Consider the pros and cons."
            response = await agent.ainvoke(scenario)
            
            # Validate response contains decision-making elements
            if not response or not hasattr(response, 'content'):
                return False
            
            content = response.content.lower()
            decision_indicators = ["choose", "decision", "option", "consider", "pros", "cons"]
            
            return any(indicator in content for indicator in decision_indicators)
            
        except Exception as e:
            logger.error(f"Error testing decision making: {e}")
            return False


class TestSpecializedAgentCreation(AgentTest):
    """Test creation of specialized agent types."""
    
    def __init__(self):
        super().__init__("Specialized Agent Creation", TestSeverity.HIGH)
    
    async def execute_test(self) -> TestResult:
        """Test creation of research, creative, and optimization agents."""
        try:
            # Create mock LLM
            mock_llm = MockLLMProvider()
            
            # Test research agent creation
            research_agent = create_research_agent(mock_llm, [])
            await research_agent.initialize()
            
            # Test creative agent creation
            creative_agent = create_creative_agent(mock_llm, [])
            await creative_agent.initialize()
            
            # Test optimization agent creation
            optimization_agent = create_optimization_agent(mock_llm, [])
            await optimization_agent.initialize()
            
            # Validate all agents
            agents = [research_agent, creative_agent, optimization_agent]
            agent_types = ["research", "creative", "optimization"]
            
            validation_results = []
            for agent, agent_type in zip(agents, agent_types):
                valid = await self.validate_agent_creation(agent)
                specialized_valid = await self._validate_specialized_features(agent, agent_type)
                validation_results.append(valid and specialized_valid)
            
            evidence = {
                "research_agent": {
                    "id": research_agent.agent_id,
                    "valid": validation_results[0],
                    "capabilities": research_agent.config.capabilities
                },
                "creative_agent": {
                    "id": creative_agent.agent_id,
                    "valid": validation_results[1],
                    "capabilities": creative_agent.config.capabilities
                },
                "optimization_agent": {
                    "id": optimization_agent.agent_id,
                    "valid": validation_results[2],
                    "capabilities": optimization_agent.config.capabilities
                }
            }
            
            success = all(validation_results)
            
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
    
    async def _validate_specialized_features(self, agent, agent_type: str) -> bool:
        """Validate specialized agent features."""
        try:
            # Check agent name contains type
            if agent_type.lower() not in agent.config.name.lower():
                return False
            
            # Check specialized capabilities
            expected_capabilities = {
                "research": ["reasoning", "tool_use", "memory"],
                "creative": ["reasoning", "tool_use", "memory"],
                "optimization": ["reasoning", "tool_use", "memory", "planning"]
            }
            
            required_caps = expected_capabilities.get(agent_type, [])
            for cap in required_caps:
                if cap not in agent.config.capabilities:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating specialized features: {e}")
            return False


class TestAgentConfigurationValidation(AgentTest):
    """Test agent configuration validation and error handling."""
    
    def __init__(self):
        super().__init__("Agent Configuration Validation", TestSeverity.HIGH)
    
    async def execute_test(self) -> TestResult:
        """Test various configuration scenarios and validation."""
        try:
            test_results = []
            
            # Test 1: Valid configuration
            valid_config = AgentConfig(
                name="Valid Agent",
                description="A valid test agent",
                model_name="mock-model",
                temperature=0.7,
                max_tokens=1024
            )
            test_results.append(await self._test_config_validation(valid_config, should_pass=True))
            
            # Test 2: Invalid temperature
            invalid_temp_config = AgentConfig(
                name="Invalid Temp Agent",
                description="Agent with invalid temperature",
                model_name="mock-model",
                temperature=2.5,  # Invalid: > 2.0
                max_tokens=1024
            )
            test_results.append(await self._test_config_validation(invalid_temp_config, should_pass=False))
            
            # Test 3: Invalid max_tokens
            invalid_tokens_config = AgentConfig(
                name="Invalid Tokens Agent",
                description="Agent with invalid max_tokens",
                model_name="mock-model",
                temperature=0.7,
                max_tokens=0  # Invalid: <= 0
            )
            test_results.append(await self._test_config_validation(invalid_tokens_config, should_pass=False))
            
            # Test 4: Empty name
            try:
                empty_name_config = AgentConfig(
                    name="",  # Invalid: empty name
                    description="Agent with empty name",
                    model_name="mock-model"
                )
                test_results.append(False)  # Should not reach here
            except Exception:
                test_results.append(True)  # Expected validation error
            
            evidence = {
                "valid_config_test": test_results[0],
                "invalid_temperature_test": test_results[1],
                "invalid_tokens_test": test_results[2],
                "empty_name_test": test_results[3] if len(test_results) > 3 else False,
                "total_tests": len(test_results)
            }
            
            success = all(test_results)
            
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
    
    async def _test_config_validation(self, config: AgentConfig, should_pass: bool) -> bool:
        """Test configuration validation."""
        try:
            # Try to create agent with configuration
            mock_llm = MockLLMProvider()
            agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
            await agent.initialize()
            
            # If we reach here, validation passed
            return should_pass
            
        except Exception as e:
            # If exception occurred, validation failed
            return not should_pass


class TestAgentLifecycleManagement(AgentTest):
    """Test agent lifecycle management operations."""
    
    def __init__(self):
        super().__init__("Agent Lifecycle Management", TestSeverity.HIGH)
    
    async def execute_test(self) -> TestResult:
        """Test agent creation, initialization, execution, and cleanup."""
        try:
            # Create agent
            config = TestDataGenerator.generate_agent_config()
            mock_llm = MockLLMProvider()
            
            agent = LangGraphAgent(
                config=AgentConfig(**config),
                llm=mock_llm,
                tools=[]
            )
            
            # Test lifecycle phases
            lifecycle_results = {}
            
            # Phase 1: Initialization
            await agent.initialize()
            lifecycle_results["initialization"] = agent.graph is not None
            
            # Phase 2: Execution
            response = await agent.ainvoke("Test message")
            lifecycle_results["execution"] = await self.validate_agent_response(response)
            
            # Phase 3: State management
            state_valid = await self._test_state_management(agent)
            lifecycle_results["state_management"] = state_valid
            
            # Phase 4: Cleanup (if applicable)
            cleanup_successful = await self._test_cleanup(agent)
            lifecycle_results["cleanup"] = cleanup_successful
            
            evidence = {
                "agent_id": agent.agent_id,
                "lifecycle_phases": lifecycle_results,
                "total_phases_passed": sum(lifecycle_results.values())
            }
            
            success = all(lifecycle_results.values())
            
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
    
    async def _test_state_management(self, agent: LangGraphAgent) -> bool:
        """Test agent state management."""
        try:
            # Test that agent maintains state
            initial_call_count = agent.llm.call_count if hasattr(agent.llm, 'call_count') else 0
            
            # Make multiple calls
            await agent.ainvoke("First message")
            await agent.ainvoke("Second message")
            
            final_call_count = agent.llm.call_count if hasattr(agent.llm, 'call_count') else 0
            
            # State should be maintained (call count increased)
            return final_call_count > initial_call_count
            
        except Exception as e:
            logger.error(f"Error testing state management: {e}")
            return False
    
    async def _test_cleanup(self, agent: LangGraphAgent) -> bool:
        """Test agent cleanup operations."""
        try:
            # For now, just verify agent still exists and is functional
            # In a real implementation, this might test resource cleanup
            return agent.agent_id is not None
            
        except Exception as e:
            logger.error(f"Error testing cleanup: {e}")
            return False


# Test suite for agent creation
class AgentCreationTestSuite:
    """Comprehensive test suite for agent creation and configuration."""
    
    def __init__(self):
        self.tests = [
            TestBasicAgentCreation(),
            TestAutonomousAgentCreation(),
            TestSpecializedAgentCreation(),
            TestAgentConfigurationValidation(),
            TestAgentLifecycleManagement()
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all agent creation tests."""
        logger.info("Starting agent creation test suite")
        
        results = []
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = await test.run()
            results.append(result)
        
        # Generate summary
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        
        summary = {
            "suite_name": "Agent Creation Test Suite",
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "results": [result.__dict__ for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Agent creation test suite completed",
            total=total,
            passed=passed,
            failed=total-passed,
            success_rate=summary["success_rate"]
        )
        
        return summary


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.agent
@pytest.mark.unit
async def test_basic_agent_creation():
    """Pytest wrapper for basic agent creation test."""
    test = TestBasicAgentCreation()
    result = await test.run()
    assert result.passed, f"Basic agent creation failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.agent
@pytest.mark.autonomous
async def test_autonomous_agent_creation():
    """Pytest wrapper for autonomous agent creation test."""
    test = TestAutonomousAgentCreation()
    result = await test.run()
    assert result.passed, f"Autonomous agent creation failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.agent
@pytest.mark.integration
async def test_specialized_agent_creation():
    """Pytest wrapper for specialized agent creation test."""
    test = TestSpecializedAgentCreation()
    result = await test.run()
    assert result.passed, f"Specialized agent creation failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.agent
@pytest.mark.unit
async def test_agent_configuration_validation():
    """Pytest wrapper for agent configuration validation test."""
    test = TestAgentConfigurationValidation()
    result = await test.run()
    assert result.passed, f"Agent configuration validation failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.agent
@pytest.mark.integration
async def test_agent_lifecycle_management():
    """Pytest wrapper for agent lifecycle management test."""
    test = TestAgentLifecycleManagement()
    result = await test.run()
    assert result.passed, f"Agent lifecycle management failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.agent
@pytest.mark.integration
async def test_complete_agent_creation_suite():
    """Run the complete agent creation test suite."""
    suite = AgentCreationTestSuite()
    summary = await suite.run_all_tests()
    
    assert summary["success_rate"] >= 0.8, f"Agent creation suite success rate too low: {summary['success_rate']}"
    assert summary["passed"] >= 4, f"Not enough tests passed: {summary['passed']}/{summary['total_tests']}"
