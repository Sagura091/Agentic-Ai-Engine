"""
Comprehensive Agent Behavior Validation Tests.

This module tests agent behavior, autonomous decision making, task execution,
and validates that agents are truly agentic rather than scripted.
"""

import pytest
import asyncio
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

import structlog

# Import test infrastructure
from .base_test import BehaviorTest, TestResult, TestCategory, TestSeverity
from .test_utils import TestDataGenerator, MockLLMProvider, MockScenario, MockResponseType, TestValidator

# Import application components
from app.agents.base.agent import LangGraphAgent, AgentConfig
from app.agents.autonomous import (
    AutonomousLangGraphAgent, 
    AutonomousAgentConfig, 
    AutonomyLevel, 
    LearningMode,
    create_autonomous_agent,
    create_research_agent
)

logger = structlog.get_logger(__name__)


class TestAutonomousDecisionMaking(BehaviorTest):
    """Test agent autonomous decision making capabilities."""
    
    def __init__(self):
        super().__init__("Autonomous Decision Making", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test agent autonomous decision making."""
        try:
            # Create autonomous agent
            agent = await self._create_autonomous_agent()
            
            # Test decision scenarios
            decision_scenarios = await self._create_decision_scenarios()
            
            # Test each scenario
            decision_results = []
            for scenario in decision_scenarios:
                result = await self._test_decision_scenario(agent, scenario)
                decision_results.append(result)
            
            # Validate decision quality
            decision_quality = await self._validate_decision_quality(decision_results)
            
            # Test decision consistency
            consistency_test = await self._test_decision_consistency(agent)
            
            # Test decision adaptation
            adaptation_test = await self._test_decision_adaptation(agent)
            
            evidence = {
                "agent_id": agent.agent_id,
                "decision_scenarios_tested": len(decision_scenarios),
                "successful_decisions": sum(1 for r in decision_results if r["success"]),
                "decision_quality": decision_quality,
                "consistency_test": consistency_test,
                "adaptation_test": adaptation_test,
                "decision_details": decision_results
            }
            
            success = (decision_quality and consistency_test and adaptation_test and
                      sum(1 for r in decision_results if r["success"]) >= len(decision_scenarios) * 0.7)
            
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
    
    async def _create_autonomous_agent(self) -> AutonomousLangGraphAgent:
        """Create autonomous agent for testing."""
        config = AutonomousAgentConfig(
            name="Decision Test Agent",
            description="Agent for testing autonomous decision making",
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            learning_mode=LearningMode.ACTIVE,
            decision_threshold=0.6,
            capabilities=["reasoning", "decision_making", "planning"],
            enable_proactive_behavior=True,
            enable_goal_setting=True
        )
        
        # Create LLM with varied responses for decision making
        responses = [
            "After careful analysis, I choose option A because it provides the best long-term benefits.",
            "Based on the available information, option B seems most appropriate given the constraints.",
            "I need to consider multiple factors here. Option C offers the optimal balance.",
            "This is a complex decision. Let me weigh the pros and cons systematically.",
            "Given the uncertainty, I'll take a calculated risk with option A."
        ]
        mock_llm = MockLLMProvider(MockScenario(MockResponseType.SUCCESS, responses=responses))
        
        agent = AutonomousLangGraphAgent(config=config, llm=mock_llm, tools=[])
        await agent.initialize()
        return agent
    
    async def _create_decision_scenarios(self) -> List[Dict[str, Any]]:
        """Create decision-making scenarios for testing."""
        return [
            {
                "name": "Resource Allocation",
                "description": "You have limited resources. How do you allocate them between three projects?",
                "options": ["Focus on Project A", "Split equally", "Prioritize Project C"],
                "context": "Project A is high-risk/high-reward, Project B is stable, Project C is innovative"
            },
            {
                "name": "Problem Solving",
                "description": "A system is failing. What's your approach to diagnose and fix it?",
                "options": ["Immediate restart", "Systematic diagnosis", "Rollback to previous version"],
                "context": "Users are affected, but data integrity is critical"
            },
            {
                "name": "Strategic Planning",
                "description": "Plan the next quarter's priorities for your team.",
                "options": ["Focus on new features", "Improve existing systems", "Research new technologies"],
                "context": "Market is competitive, team has mixed skills, budget is limited"
            },
            {
                "name": "Risk Assessment",
                "description": "Evaluate the risk of implementing a new technology.",
                "options": ["Implement immediately", "Pilot test first", "Wait for more maturity"],
                "context": "Technology is promising but unproven, competitors are moving fast"
            }
        ]
    
    async def _test_decision_scenario(self, agent: AutonomousLangGraphAgent, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test agent decision making in a specific scenario."""
        try:
            prompt = f"""
Scenario: {scenario['description']}
Context: {scenario['context']}
Options: {', '.join(scenario['options'])}

Please make a decision and explain your reasoning. Consider the trade-offs and justify your choice.
"""
            
            response = await agent.ainvoke(prompt)
            
            # Analyze response for decision-making indicators
            decision_made = await self._analyze_decision_response(response, scenario)
            
            return {
                "scenario": scenario["name"],
                "success": decision_made["has_decision"],
                "reasoning_quality": decision_made["reasoning_score"],
                "response": response.content if hasattr(response, 'content') else str(response),
                "decision_indicators": decision_made["indicators"]
            }
            
        except Exception as e:
            logger.error(f"Error testing decision scenario {scenario['name']}: {e}")
            return {
                "scenario": scenario["name"],
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_decision_response(self, response: Any, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze response for decision-making quality."""
        if not response or not hasattr(response, 'content'):
            return {"has_decision": False, "reasoning_score": 0.0, "indicators": []}
        
        content = response.content.lower()
        
        # Check for decision indicators
        decision_words = ["choose", "select", "decide", "option", "recommend", "prefer"]
        reasoning_words = ["because", "since", "due to", "considering", "analysis", "weigh"]
        option_mentions = sum(1 for option in scenario["options"] if option.lower() in content)
        
        decision_indicators = sum(1 for word in decision_words if word in content)
        reasoning_indicators = sum(1 for word in reasoning_words if word in content)
        
        has_decision = decision_indicators > 0 and option_mentions > 0
        reasoning_score = min(1.0, (reasoning_indicators + option_mentions) / 5.0)
        
        return {
            "has_decision": has_decision,
            "reasoning_score": reasoning_score,
            "indicators": {
                "decision_words": decision_indicators,
                "reasoning_words": reasoning_indicators,
                "option_mentions": option_mentions
            }
        }
    
    async def _validate_decision_quality(self, decision_results: List[Dict[str, Any]]) -> bool:
        """Validate overall decision quality."""
        if not decision_results:
            return False
        
        # Check that most decisions were successful
        success_rate = sum(1 for r in decision_results if r.get("success", False)) / len(decision_results)
        
        # Check reasoning quality
        avg_reasoning = sum(r.get("reasoning_quality", 0) for r in decision_results) / len(decision_results)
        
        return success_rate >= 0.7 and avg_reasoning >= 0.5
    
    async def _test_decision_consistency(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test decision consistency across similar scenarios."""
        try:
            # Present similar scenarios multiple times
            scenario = "You need to prioritize tasks. How do you decide which task to do first?"
            
            responses = []
            for i in range(3):
                response = await agent.ainvoke(f"{scenario} (Iteration {i+1})")
                responses.append(response)
            
            # Check for consistent decision-making approach
            return await self.validate_autonomous_behavior(responses, min_variation=0.3)
            
        except Exception as e:
            logger.error(f"Error testing decision consistency: {e}")
            return False
    
    async def _test_decision_adaptation(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test agent's ability to adapt decisions based on new information."""
        try:
            # Initial scenario
            initial_prompt = "Choose between Option A and Option B for a project."
            initial_response = await agent.ainvoke(initial_prompt)
            
            # Add new information
            updated_prompt = """Choose between Option A and Option B for a project.
            NEW INFORMATION: Option A has just been found to have significant risks."""
            updated_response = await agent.ainvoke(updated_prompt)
            
            # Check if decision changed appropriately
            initial_content = initial_response.content if hasattr(initial_response, 'content') else ""
            updated_content = updated_response.content if hasattr(updated_response, 'content') else ""
            
            # Should show adaptation to new information
            return initial_content != updated_content and len(updated_content) > 10
            
        except Exception as e:
            logger.error(f"Error testing decision adaptation: {e}")
            return False


class TestAgenticBehaviorPatterns(BehaviorTest):
    """Test patterns that indicate truly agentic behavior."""
    
    def __init__(self):
        super().__init__("Agentic Behavior Patterns", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test for patterns indicating agentic vs scripted behavior."""
        try:
            # Create test agent
            agent = await self._create_test_agent()
            
            # Test response variability
            variability_test = await self._test_response_variability(agent)
            
            # Test contextual awareness
            context_test = await self._test_contextual_awareness(agent)
            
            # Test emergent behavior
            emergent_test = await self._test_emergent_behavior(agent)
            
            # Test goal-oriented behavior
            goal_test = await self._test_goal_oriented_behavior(agent)
            
            # Test learning indicators
            learning_test = await self._test_learning_indicators(agent)
            
            evidence = {
                "agent_id": agent.agent_id,
                "variability_test": variability_test,
                "contextual_awareness": context_test,
                "emergent_behavior": emergent_test,
                "goal_oriented": goal_test,
                "learning_indicators": learning_test
            }
            
            success = (variability_test and context_test and emergent_test and 
                      goal_test and learning_test)
            
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
    
    async def _create_test_agent(self) -> AutonomousLangGraphAgent:
        """Create agent for behavior testing."""
        config = AutonomousAgentConfig(
            name="Behavior Test Agent",
            description="Agent for testing agentic behavior patterns",
            autonomy_level=AutonomyLevel.ADAPTIVE,
            learning_mode=LearningMode.ACTIVE,
            capabilities=["reasoning", "learning", "adaptation", "goal_setting"]
        )
        
        # Create LLM with diverse responses
        responses = [
            "I need to analyze this situation carefully before proceeding.",
            "Based on my understanding, I should approach this differently.",
            "Let me consider the implications of each possible action.",
            "This reminds me of a similar situation where I learned that...",
            "I'll adapt my strategy based on what I've observed.",
            "My goal is to find the most effective solution here.",
            "I notice a pattern that suggests I should...",
            "Given the context, I believe the best approach is..."
        ]
        mock_llm = MockLLMProvider(MockScenario(MockResponseType.RANDOM, responses=responses))
        
        agent = AutonomousLangGraphAgent(config=config, llm=mock_llm, tools=[])
        await agent.initialize()
        return agent
    
    async def _test_response_variability(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test that agent shows variability in responses (not scripted)."""
        try:
            # Ask the same question multiple times
            question = "How would you approach solving a complex problem?"
            
            responses = []
            for i in range(5):
                response = await agent.ainvoke(question)
                responses.append(response)
            
            # Check for variability
            return await self.validate_autonomous_behavior(responses, min_variation=0.6)
            
        except Exception as e:
            logger.error(f"Error testing response variability: {e}")
            return False
    
    async def _test_contextual_awareness(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test agent's contextual awareness."""
        try:
            # Test with different contexts
            contexts = [
                "In a business meeting, how would you present an idea?",
                "In an emergency situation, how would you respond?",
                "When teaching a child, how would you explain something complex?"
            ]
            
            responses = []
            for context in contexts:
                response = await agent.ainvoke(context)
                responses.append(response)
            
            # Responses should be different for different contexts
            unique_responses = set(str(r) for r in responses)
            return len(unique_responses) >= len(contexts) * 0.8
            
        except Exception as e:
            logger.error(f"Error testing contextual awareness: {e}")
            return False
    
    async def _test_emergent_behavior(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test for emergent behavior patterns."""
        try:
            # Present novel scenarios that require creative thinking
            novel_scenarios = [
                "You discover a new type of problem you've never seen before. How do you approach it?",
                "Combine concepts from different domains to solve this challenge.",
                "What would you do if all your usual strategies failed?"
            ]
            
            responses = []
            for scenario in novel_scenarios:
                response = await agent.ainvoke(scenario)
                responses.append(response)
            
            # Check for creative/emergent responses
            for response in responses:
                if not response or not hasattr(response, 'content'):
                    return False
                
                content = response.content.lower()
                creative_indicators = ["creative", "innovative", "novel", "combine", "adapt", "experiment"]
                
                if not any(indicator in content for indicator in creative_indicators):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing emergent behavior: {e}")
            return False
    
    async def _test_goal_oriented_behavior(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test goal-oriented behavior."""
        try:
            # Present goal-oriented scenarios
            goal_prompt = """Your goal is to improve team productivity. 
            You have one month and limited resources. What's your plan?"""
            
            response = await agent.ainvoke(goal_prompt)
            
            if not response or not hasattr(response, 'content'):
                return False
            
            content = response.content.lower()
            goal_indicators = ["plan", "strategy", "steps", "achieve", "improve", "goal", "objective"]
            
            return sum(1 for indicator in goal_indicators if indicator in content) >= 3
            
        except Exception as e:
            logger.error(f"Error testing goal-oriented behavior: {e}")
            return False
    
    async def _test_learning_indicators(self, agent: AutonomousLangGraphAgent) -> bool:
        """Test for learning behavior indicators."""
        try:
            # Present learning scenarios
            learning_prompts = [
                "You made a mistake in your previous approach. What did you learn?",
                "How would you improve your performance based on feedback?",
                "What patterns have you noticed in similar situations?"
            ]
            
            responses = []
            for prompt in learning_prompts:
                response = await agent.ainvoke(prompt)
                responses.append(response)
            
            # Check for learning indicators
            learning_words = ["learn", "improve", "adapt", "pattern", "feedback", "experience", "adjust"]
            
            for response in responses:
                if not response or not hasattr(response, 'content'):
                    return False
                
                content = response.content.lower()
                if not any(word in content for word in learning_words):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing learning indicators: {e}")
            return False


class TestTaskExecution(BehaviorTest):
    """Test agent task execution capabilities."""
    
    def __init__(self):
        super().__init__("Task Execution", TestSeverity.HIGH)
    
    async def execute_test(self) -> TestResult:
        """Test agent task execution capabilities."""
        try:
            # Create agent for task execution
            agent = await self._create_task_agent()
            
            # Test simple task execution
            simple_task_test = await self._test_simple_task_execution(agent)
            
            # Test complex task execution
            complex_task_test = await self._test_complex_task_execution(agent)
            
            # Test multi-step task execution
            multistep_task_test = await self._test_multistep_task_execution(agent)
            
            # Test task adaptation
            adaptation_test = await self._test_task_adaptation(agent)
            
            evidence = {
                "agent_id": agent.agent_id,
                "simple_tasks": simple_task_test,
                "complex_tasks": complex_task_test,
                "multistep_tasks": multistep_task_test,
                "task_adaptation": adaptation_test
            }
            
            success = simple_task_test and complex_task_test and multistep_task_test and adaptation_test
            
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
    
    async def _create_task_agent(self) -> LangGraphAgent:
        """Create agent for task execution testing."""
        config = AgentConfig(
            name="Task Execution Agent",
            description="Agent for testing task execution",
            capabilities=["task_execution", "planning", "reasoning"]
        )
        
        mock_llm = MockLLMProvider()
        agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
        await agent.initialize()
        return agent
    
    async def _test_simple_task_execution(self, agent: LangGraphAgent) -> bool:
        """Test simple task execution."""
        try:
            tasks = [
                "Summarize the key points of artificial intelligence.",
                "Explain the difference between machine learning and deep learning.",
                "List three benefits of automation."
            ]
            
            for task in tasks:
                response = await agent.ainvoke(task)
                if not await self.validate_agent_response(response):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing simple task execution: {e}")
            return False
    
    async def _test_complex_task_execution(self, agent: LangGraphAgent) -> bool:
        """Test complex task execution."""
        try:
            complex_task = """Analyze the following scenario and provide recommendations:
            A company wants to implement AI but has limited budget and technical expertise.
            Consider risks, benefits, and implementation strategies."""
            
            response = await agent.ainvoke(complex_task)
            
            if not response or not hasattr(response, 'content'):
                return False
            
            content = response.content.lower()
            analysis_indicators = ["analyze", "recommend", "consider", "strategy", "risk", "benefit"]
            
            return sum(1 for indicator in analysis_indicators if indicator in content) >= 3
            
        except Exception as e:
            logger.error(f"Error testing complex task execution: {e}")
            return False
    
    async def _test_multistep_task_execution(self, agent: LangGraphAgent) -> bool:
        """Test multi-step task execution."""
        try:
            multistep_task = """Plan a project to develop a new software feature:
            1. Define requirements
            2. Design the solution
            3. Implement the feature
            4. Test and validate
            5. Deploy and monitor
            
            Provide details for each step."""
            
            response = await agent.ainvoke(multistep_task)
            
            if not response or not hasattr(response, 'content'):
                return False
            
            content = response.content.lower()
            step_indicators = ["step", "first", "second", "third", "next", "then", "finally"]
            
            return sum(1 for indicator in step_indicators if indicator in content) >= 2
            
        except Exception as e:
            logger.error(f"Error testing multistep task execution: {e}")
            return False
    
    async def _test_task_adaptation(self, agent: LangGraphAgent) -> bool:
        """Test task adaptation capabilities."""
        try:
            # Initial task
            initial_task = "Create a marketing plan for a new product."
            initial_response = await agent.ainvoke(initial_task)
            
            # Modified task with constraints
            modified_task = """Create a marketing plan for a new product.
            CONSTRAINTS: Budget is very limited, target audience is seniors, product is a health app."""
            modified_response = await agent.ainvoke(modified_task)
            
            # Responses should be different due to constraints
            initial_content = initial_response.content if hasattr(initial_response, 'content') else ""
            modified_content = modified_response.content if hasattr(modified_response, 'content') else ""
            
            return initial_content != modified_content and len(modified_content) > len(initial_content) * 0.5
            
        except Exception as e:
            logger.error(f"Error testing task adaptation: {e}")
            return False


# Test suite for agent behavior
class AgentBehaviorTestSuite:
    """Comprehensive test suite for agent behavior validation."""
    
    def __init__(self):
        self.tests = [
            TestAutonomousDecisionMaking(),
            TestAgenticBehaviorPatterns(),
            TestTaskExecution()
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all agent behavior tests."""
        logger.info("Starting agent behavior test suite")
        
        results = []
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = await test.run()
            results.append(result)
        
        # Generate summary
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        
        summary = {
            "suite_name": "Agent Behavior Test Suite",
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "results": [result.__dict__ for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Agent behavior test suite completed",
            total=total,
            passed=passed,
            failed=total-passed,
            success_rate=summary["success_rate"]
        )
        
        return summary


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.behavior
@pytest.mark.autonomous
async def test_autonomous_decision_making():
    """Pytest wrapper for autonomous decision making test."""
    test = TestAutonomousDecisionMaking()
    result = await test.run()
    assert result.passed, f"Autonomous decision making failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.behavior
@pytest.mark.autonomous
async def test_agentic_behavior_patterns():
    """Pytest wrapper for agentic behavior patterns test."""
    test = TestAgenticBehaviorPatterns()
    result = await test.run()
    assert result.passed, f"Agentic behavior patterns failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.behavior
@pytest.mark.integration
async def test_task_execution():
    """Pytest wrapper for task execution test."""
    test = TestTaskExecution()
    result = await test.run()
    assert result.passed, f"Task execution failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.behavior
@pytest.mark.integration
async def test_complete_agent_behavior_suite():
    """Run the complete agent behavior test suite."""
    suite = AgentBehaviorTestSuite()
    summary = await suite.run_all_tests()
    
    assert summary["success_rate"] >= 0.8, f"Agent behavior suite success rate too low: {summary['success_rate']}"
    assert summary["passed"] >= 2, f"Not enough tests passed: {summary['passed']}/{summary['total_tests']}"
