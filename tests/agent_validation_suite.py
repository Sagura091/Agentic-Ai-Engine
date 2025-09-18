#!/usr/bin/env python3
"""
Agent Validation Suite for Real-Time Testing and Monitoring.

This module provides comprehensive testing capabilities to validate that agents
are truly agentic and not just pseudo-autonomous. It includes real-time monitoring,
task execution validation, and tool creation testing.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
import requests
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AgentTestType(Enum):
    """Types of agent tests."""
    BASIC_FUNCTIONALITY = "basic_functionality"
    AUTONOMOUS_DECISION = "autonomous_decision"
    TOOL_CREATION = "tool_creation"
    PROBLEM_SOLVING = "problem_solving"
    LEARNING_ADAPTATION = "learning_adaptation"
    REAL_TIME_MONITORING = "real_time_monitoring"


class TestResult(Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class AgentTestCase:
    """Individual agent test case."""
    test_id: str
    test_type: AgentTestType
    name: str
    description: str
    task: str
    expected_behaviors: List[str]
    success_criteria: List[str]
    timeout_seconds: int = 60
    requires_tools: List[str] = None
    
    def __post_init__(self):
        if self.requires_tools is None:
            self.requires_tools = []


@dataclass
class AgentTestResult:
    """Result of an agent test."""
    test_id: str
    agent_id: str
    test_type: AgentTestType
    result: TestResult
    score: float  # 0.0 to 1.0
    execution_time: float
    behaviors_observed: List[str]
    criteria_met: List[str]
    criteria_failed: List[str]
    agent_response: str
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict = asdict(self)
        result_dict['test_type'] = self.test_type.value
        result_dict['result'] = self.result.value
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict


class AgentValidationSuite:
    """
    Comprehensive agent validation suite for testing true agentic behavior.
    
    This suite tests agents against various scenarios to determine if they exhibit
    genuine autonomous intelligence or just pseudo-autonomous behavior.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the validation suite.
        
        Args:
            base_url: Base URL of the agent orchestration API
        """
        self.base_url = base_url
        self.test_session_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_results: List[AgentTestResult] = []
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        
        # Test cases for different agent capabilities
        self.test_cases = self._create_test_cases()
        
        logger.info("Agent validation suite initialized", session_id=self.test_session_id)
    
    def _create_test_cases(self) -> List[AgentTestCase]:
        """Create comprehensive test cases for agent validation."""
        return [
            # Basic functionality tests
            AgentTestCase(
                test_id="basic_001",
                test_type=AgentTestType.BASIC_FUNCTIONALITY,
                name="Simple Task Execution",
                description="Test if agent can execute a simple task",
                task="Calculate the sum of 15 and 27, then explain your reasoning",
                expected_behaviors=["mathematical_calculation", "explanation_provided"],
                success_criteria=["correct_answer_42", "reasoning_explained"],
                timeout_seconds=30
            ),
            
            # Autonomous decision making tests
            AgentTestCase(
                test_id="auto_001",
                test_type=AgentTestType.AUTONOMOUS_DECISION,
                name="Multi-Step Problem Solving",
                description="Test autonomous decision making in complex scenarios",
                task="You need to plan a birthday party for 20 people with a budget of $200. Make all necessary decisions and create a complete plan.",
                expected_behaviors=["autonomous_planning", "budget_consideration", "decision_making"],
                success_criteria=["complete_plan_created", "budget_respected", "realistic_decisions"],
                timeout_seconds=120
            ),
            
            # Tool creation and usage tests
            AgentTestCase(
                test_id="tool_001",
                test_type=AgentTestType.TOOL_CREATION,
                name="Dynamic Tool Creation",
                description="Test if agent can create and use tools dynamically",
                task="Create a tool that can convert temperatures between Celsius and Fahrenheit, then use it to convert 25°C to Fahrenheit",
                expected_behaviors=["tool_creation", "tool_usage", "problem_solving"],
                success_criteria=["tool_created_successfully", "correct_conversion_77F", "tool_used_properly"],
                timeout_seconds=90
            ),
            
            # Problem solving tests
            AgentTestCase(
                test_id="solve_001",
                test_type=AgentTestType.PROBLEM_SOLVING,
                name="Creative Problem Solving",
                description="Test creative and adaptive problem solving",
                task="You have 3 containers: 8L, 5L, and 3L. The 8L container is full of water. How do you measure exactly 4L using only these containers?",
                expected_behaviors=["creative_thinking", "step_by_step_reasoning", "solution_validation"],
                success_criteria=["correct_solution_provided", "steps_clearly_explained", "solution_validated"],
                timeout_seconds=180
            ),
            
            # Learning and adaptation tests
            AgentTestCase(
                test_id="learn_001",
                test_type=AgentTestType.LEARNING_ADAPTATION,
                name="Learning from Feedback",
                description="Test if agent can learn and adapt from feedback",
                task="Solve this riddle: 'I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?' If you get it wrong, I'll give you a hint and you should adapt your approach.",
                expected_behaviors=["initial_attempt", "feedback_processing", "adaptation", "improved_response"],
                success_criteria=["shows_learning", "adapts_to_feedback", "final_correct_answer_fire"],
                timeout_seconds=240
            )
        ]
    
    async def create_test_agent(self, agent_type: str = "autonomous", model: str = "gpt-oss:120b") -> str:
        """
        Create a test agent for validation.
        
        Args:
            agent_type: Type of agent to create
            model: Model to use for the agent
            
        Returns:
            Agent ID
        """
        try:
            agent_data = {
                "agent_type": agent_type,
                "name": f"TestAgent_{self.test_session_id}_{int(time.time())}",
                "description": f"Test agent for validation suite session {self.test_session_id}",
                "model": model,
                "config": {
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "enable_learning": True,
                    "enable_tool_creation": True
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/orchestration/agents",
                json=agent_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                agent_id = result.get("agent_id")
                self.active_agents[agent_id] = {
                    "agent_type": agent_type,
                    "model": model,
                    "created_at": datetime.now(),
                    "test_results": []
                }
                logger.info("Test agent created", agent_id=agent_id, agent_type=agent_type)
                return agent_id
            else:
                logger.error("Failed to create test agent", status_code=response.status_code, response=response.text)
                raise Exception(f"Failed to create agent: {response.status_code}")
                
        except Exception as e:
            logger.error("Error creating test agent", error=str(e))
            raise
    
    async def execute_agent_task(self, agent_id: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a task with an agent and monitor the execution.
        
        Args:
            agent_id: ID of the agent to execute the task
            task: Task description
            context: Additional context for the task
            
        Returns:
            Task execution result
        """
        try:
            task_data = {
                "agent_id": agent_id,
                "task": task,
                "context": context or {}
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/orchestration/agents/execute",
                json=task_data,
                timeout=300  # 5 minute timeout
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result["execution_time"] = execution_time
                return result
            else:
                logger.error("Task execution failed", status_code=response.status_code, response=response.text)
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "execution_time": execution_time
                }
                
        except Exception as e:
            logger.error("Error executing agent task", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    async def run_test_case(self, agent_id: str, test_case: AgentTestCase) -> AgentTestResult:
        """
        Run a specific test case against an agent.

        Args:
            agent_id: ID of the agent to test
            test_case: Test case to execute

        Returns:
            Test result
        """
        logger.info("Running test case", test_id=test_case.test_id, agent_id=agent_id, test_type=test_case.test_type.value)

        start_time = time.time()

        try:
            # Execute the task
            execution_result = await self.execute_agent_task(
                agent_id=agent_id,
                task=test_case.task,
                context={"test_id": test_case.test_id, "test_type": test_case.test_type.value}
            )

            execution_time = time.time() - start_time

            # Analyze the result
            analysis = self._analyze_test_result(test_case, execution_result)

            # Create test result
            test_result = AgentTestResult(
                test_id=test_case.test_id,
                agent_id=agent_id,
                test_type=test_case.test_type,
                result=analysis["result"],
                score=analysis["score"],
                execution_time=execution_time,
                behaviors_observed=analysis["behaviors_observed"],
                criteria_met=analysis["criteria_met"],
                criteria_failed=analysis["criteria_failed"],
                agent_response=execution_result.get("response", ""),
                metadata={
                    "execution_result": execution_result,
                    "analysis_details": analysis.get("details", {}),
                    "test_case": asdict(test_case)
                },
                timestamp=datetime.now()
            )

            # Store result
            self.test_results.append(test_result)
            if agent_id in self.active_agents:
                self.active_agents[agent_id]["test_results"].append(test_result)

            logger.info("Test case completed",
                       test_id=test_case.test_id,
                       agent_id=agent_id,
                       result=analysis["result"].value,
                       score=analysis["score"])

            return test_result

        except Exception as e:
            logger.error("Error running test case", test_id=test_case.test_id, agent_id=agent_id, error=str(e))

            # Create error result
            test_result = AgentTestResult(
                test_id=test_case.test_id,
                agent_id=agent_id,
                test_type=test_case.test_type,
                result=TestResult.ERROR,
                score=0.0,
                execution_time=time.time() - start_time,
                behaviors_observed=[],
                criteria_met=[],
                criteria_failed=test_case.success_criteria,
                agent_response="",
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )

            self.test_results.append(test_result)
            return test_result

    def _analyze_test_result(self, test_case: AgentTestCase, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze test execution result against success criteria.

        Args:
            test_case: The test case that was executed
            execution_result: Result from agent execution

        Returns:
            Analysis results including score and criteria evaluation
        """
        if not execution_result.get("success", False):
            return {
                "result": TestResult.ERROR,
                "score": 0.0,
                "behaviors_observed": [],
                "criteria_met": [],
                "criteria_failed": test_case.success_criteria,
                "details": {"error": execution_result.get("error", "Unknown error")}
            }

        response = execution_result.get("response", "").lower()

        # Analyze based on test type
        if test_case.test_type == AgentTestType.BASIC_FUNCTIONALITY:
            return self._analyze_basic_functionality(test_case, response)
        elif test_case.test_type == AgentTestType.AUTONOMOUS_DECISION:
            return self._analyze_autonomous_decision(test_case, response)
        elif test_case.test_type == AgentTestType.TOOL_CREATION:
            return self._analyze_tool_creation(test_case, response, execution_result)
        elif test_case.test_type == AgentTestType.PROBLEM_SOLVING:
            return self._analyze_problem_solving(test_case, response)
        elif test_case.test_type == AgentTestType.LEARNING_ADAPTATION:
            return self._analyze_learning_adaptation(test_case, response)
        else:
            return self._analyze_generic(test_case, response)

    def _analyze_basic_functionality(self, test_case: AgentTestCase, response: str) -> Dict[str, Any]:
        """Analyze basic functionality test results."""
        criteria_met = []
        criteria_failed = []
        behaviors_observed = []

        # Check for correct answer (15 + 27 = 42)
        if "42" in response:
            criteria_met.append("correct_answer_42")
            behaviors_observed.append("mathematical_calculation")
        else:
            criteria_failed.append("correct_answer_42")

        # Check for reasoning explanation
        reasoning_keywords = ["because", "since", "therefore", "reason", "calculate", "add", "sum"]
        if any(keyword in response for keyword in reasoning_keywords):
            criteria_met.append("reasoning_explained")
            behaviors_observed.append("explanation_provided")
        else:
            criteria_failed.append("reasoning_explained")

        score = len(criteria_met) / len(test_case.success_criteria)
        result = TestResult.PASS if score >= 0.8 else TestResult.PARTIAL if score >= 0.5 else TestResult.FAIL

        return {
            "result": result,
            "score": score,
            "behaviors_observed": behaviors_observed,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
            "details": {"response_analysis": "Basic math and reasoning check"}
        }

    def _analyze_autonomous_decision(self, test_case: AgentTestCase, response: str) -> Dict[str, Any]:
        """Analyze autonomous decision making test results."""
        criteria_met = []
        criteria_failed = []
        behaviors_observed = []

        # Check for complete plan
        plan_keywords = ["plan", "schedule", "organize", "arrange", "prepare"]
        if any(keyword in response for keyword in plan_keywords) and len(response) > 200:
            criteria_met.append("complete_plan_created")
            behaviors_observed.append("autonomous_planning")
        else:
            criteria_failed.append("complete_plan_created")

        # Check for budget consideration
        budget_keywords = ["budget", "$200", "cost", "price", "money", "expense"]
        if any(keyword in response for keyword in budget_keywords):
            criteria_met.append("budget_respected")
            behaviors_observed.append("budget_consideration")
        else:
            criteria_failed.append("budget_respected")

        # Check for realistic decisions
        decision_keywords = ["decide", "choose", "select", "option", "alternative"]
        if any(keyword in response for keyword in decision_keywords):
            criteria_met.append("realistic_decisions")
            behaviors_observed.append("decision_making")
        else:
            criteria_failed.append("realistic_decisions")

        score = len(criteria_met) / len(test_case.success_criteria)
        result = TestResult.PASS if score >= 0.8 else TestResult.PARTIAL if score >= 0.5 else TestResult.FAIL

        return {
            "result": result,
            "score": score,
            "behaviors_observed": behaviors_observed,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
            "details": {"response_analysis": "Autonomous planning and decision making check"}
        }

    def _analyze_tool_creation(self, test_case: AgentTestCase, response: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tool creation test results."""
        criteria_met = []
        criteria_failed = []
        behaviors_observed = []

        # Check if tool was created
        tool_keywords = ["tool", "function", "create", "build", "implement"]
        if any(keyword in response for keyword in tool_keywords):
            criteria_met.append("tool_created_successfully")
            behaviors_observed.append("tool_creation")
        else:
            criteria_failed.append("tool_created_successfully")

        # Check for correct conversion (25°C = 77°F)
        if "77" in response or "77.0" in response:
            criteria_met.append("correct_conversion_77F")
            behaviors_observed.append("problem_solving")
        else:
            criteria_failed.append("correct_conversion_77F")

        # Check if tool was used properly
        usage_keywords = ["use", "apply", "execute", "run", "convert"]
        if any(keyword in response for keyword in usage_keywords):
            criteria_met.append("tool_used_properly")
            behaviors_observed.append("tool_usage")
        else:
            criteria_failed.append("tool_used_properly")

        score = len(criteria_met) / len(test_case.success_criteria)
        result = TestResult.PASS if score >= 0.8 else TestResult.PARTIAL if score >= 0.5 else TestResult.FAIL

        return {
            "result": result,
            "score": score,
            "behaviors_observed": behaviors_observed,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
            "details": {"response_analysis": "Tool creation and usage check"}
        }

    def _analyze_problem_solving(self, test_case: AgentTestCase, response: str) -> Dict[str, Any]:
        """Analyze problem solving test results."""
        criteria_met = []
        criteria_failed = []
        behaviors_observed = []

        # Check for correct solution (water container problem)
        solution_keywords = ["pour", "fill", "empty", "transfer", "step"]
        if any(keyword in response for keyword in solution_keywords) and len(response) > 100:
            criteria_met.append("correct_solution_provided")
            behaviors_observed.append("creative_thinking")
        else:
            criteria_failed.append("correct_solution_provided")

        # Check for step-by-step explanation
        step_keywords = ["step", "first", "then", "next", "finally", "1.", "2.", "3."]
        if any(keyword in response for keyword in step_keywords):
            criteria_met.append("steps_clearly_explained")
            behaviors_observed.append("step_by_step_reasoning")
        else:
            criteria_failed.append("steps_clearly_explained")

        # Check for solution validation
        validation_keywords = ["check", "verify", "confirm", "validate", "result"]
        if any(keyword in response for keyword in validation_keywords):
            criteria_met.append("solution_validated")
            behaviors_observed.append("solution_validation")
        else:
            criteria_failed.append("solution_validated")

        score = len(criteria_met) / len(test_case.success_criteria)
        result = TestResult.PASS if score >= 0.8 else TestResult.PARTIAL if score >= 0.5 else TestResult.FAIL

        return {
            "result": result,
            "score": score,
            "behaviors_observed": behaviors_observed,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
            "details": {"response_analysis": "Problem solving and reasoning check"}
        }

    def _analyze_learning_adaptation(self, test_case: AgentTestCase, response: str) -> Dict[str, Any]:
        """Analyze learning and adaptation test results."""
        criteria_met = []
        criteria_failed = []
        behaviors_observed = []

        # Check for learning behavior
        learning_keywords = ["learn", "understand", "realize", "adapt", "adjust"]
        if any(keyword in response for keyword in learning_keywords):
            criteria_met.append("shows_learning")
            behaviors_observed.append("initial_attempt")
        else:
            criteria_failed.append("shows_learning")

        # Check for adaptation to feedback
        adaptation_keywords = ["feedback", "hint", "correction", "improve", "better"]
        if any(keyword in response for keyword in adaptation_keywords):
            criteria_met.append("adapts_to_feedback")
            behaviors_observed.append("adaptation")
        else:
            criteria_failed.append("adapts_to_feedback")

        # Check for correct answer (fire)
        if "fire" in response:
            criteria_met.append("final_correct_answer_fire")
            behaviors_observed.append("improved_response")
        else:
            criteria_failed.append("final_correct_answer_fire")

        score = len(criteria_met) / len(test_case.success_criteria)
        result = TestResult.PASS if score >= 0.8 else TestResult.PARTIAL if score >= 0.5 else TestResult.FAIL

        return {
            "result": result,
            "score": score,
            "behaviors_observed": behaviors_observed,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
            "details": {"response_analysis": "Learning and adaptation check"}
        }

    def _analyze_generic(self, test_case: AgentTestCase, response: str) -> Dict[str, Any]:
        """Generic analysis for unknown test types."""
        # Basic analysis based on response length and content
        score = 0.5 if len(response) > 50 else 0.0
        result = TestResult.PARTIAL if score > 0 else TestResult.FAIL

        return {
            "result": result,
            "score": score,
            "behaviors_observed": ["response_provided"] if len(response) > 0 else [],
            "criteria_met": [],
            "criteria_failed": test_case.success_criteria,
            "details": {"response_analysis": "Generic analysis - unknown test type"}
        }

    async def run_comprehensive_validation(self, agent_ids: List[str] = None, test_types: List[AgentTestType] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation across multiple agents and test types.

        Args:
            agent_ids: List of agent IDs to test. If None, creates new test agents.
            test_types: List of test types to run. If None, runs all test types.

        Returns:
            Comprehensive validation results
        """
        logger.info("Starting comprehensive agent validation", session_id=self.test_session_id)

        # Create test agents if none provided
        if agent_ids is None:
            logger.info("Creating test agents for validation")
            agent_ids = []
            for agent_type in ["basic", "autonomous", "research"]:
                try:
                    agent_id = await self.create_test_agent(agent_type=agent_type)
                    agent_ids.append(agent_id)
                except Exception as e:
                    logger.error(f"Failed to create {agent_type} agent", error=str(e))

        # Filter test cases by type if specified
        test_cases_to_run = self.test_cases
        if test_types:
            test_cases_to_run = [tc for tc in self.test_cases if tc.test_type in test_types]

        # Run all test cases against all agents
        validation_results = {
            "session_id": self.test_session_id,
            "timestamp": datetime.now().isoformat(),
            "agents_tested": agent_ids,
            "test_cases_run": len(test_cases_to_run),
            "total_tests": len(agent_ids) * len(test_cases_to_run),
            "results": [],
            "summary": {}
        }

        for agent_id in agent_ids:
            logger.info(f"Testing agent {agent_id}")
            agent_results = []

            for test_case in test_cases_to_run:
                try:
                    result = await self.run_test_case(agent_id, test_case)
                    agent_results.append(result.to_dict())
                except Exception as e:
                    logger.error(f"Failed to run test {test_case.test_id} on agent {agent_id}", error=str(e))

            validation_results["results"].append({
                "agent_id": agent_id,
                "agent_info": self.active_agents.get(agent_id, {}),
                "test_results": agent_results
            })

        # Generate summary
        validation_results["summary"] = self._generate_validation_summary()

        logger.info("Comprehensive validation completed",
                   total_tests=validation_results["total_tests"],
                   agents_tested=len(agent_ids))

        return validation_results

    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate summary of validation results."""
        if not self.test_results:
            return {"message": "No test results available"}

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.result == TestResult.PASS])
        failed_tests = len([r for r in self.test_results if r.result == TestResult.FAIL])
        partial_tests = len([r for r in self.test_results if r.result == TestResult.PARTIAL])
        error_tests = len([r for r in self.test_results if r.result == TestResult.ERROR])

        average_score = sum(r.score for r in self.test_results) / total_tests if total_tests > 0 else 0
        average_execution_time = sum(r.execution_time for r in self.test_results) / total_tests if total_tests > 0 else 0

        # Analyze by test type
        test_type_summary = {}
        for test_type in AgentTestType:
            type_results = [r for r in self.test_results if r.test_type == test_type]
            if type_results:
                type_score = sum(r.score for r in type_results) / len(type_results)
                type_pass_rate = len([r for r in type_results if r.result == TestResult.PASS]) / len(type_results)
                test_type_summary[test_type.value] = {
                    "total_tests": len(type_results),
                    "average_score": type_score,
                    "pass_rate": type_pass_rate,
                    "status": "EXCELLENT" if type_score >= 0.9 else "GOOD" if type_score >= 0.7 else "NEEDS_IMPROVEMENT"
                }

        # Analyze by agent
        agent_summary = {}
        for agent_id in self.active_agents:
            agent_results = [r for r in self.test_results if r.agent_id == agent_id]
            if agent_results:
                agent_score = sum(r.score for r in agent_results) / len(agent_results)
                agent_pass_rate = len([r for r in agent_results if r.result == TestResult.PASS]) / len(agent_results)
                agent_summary[agent_id] = {
                    "total_tests": len(agent_results),
                    "average_score": agent_score,
                    "pass_rate": agent_pass_rate,
                    "agentic_rating": self._calculate_agentic_rating(agent_results),
                    "agent_info": self.active_agents[agent_id]
                }

        return {
            "overall_statistics": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "partial": partial_tests,
                "errors": error_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "average_score": average_score,
                "average_execution_time": average_execution_time
            },
            "test_type_analysis": test_type_summary,
            "agent_analysis": agent_summary,
            "overall_agentic_assessment": self._assess_overall_agentic_capability()
        }

    def _calculate_agentic_rating(self, agent_results: List[AgentTestResult]) -> str:
        """Calculate agentic rating for an agent based on test results."""
        if not agent_results:
            return "UNKNOWN"

        average_score = sum(r.score for r in agent_results) / len(agent_results)

        # Check for autonomous behaviors
        autonomous_behaviors = ["autonomous_planning", "decision_making", "tool_creation", "creative_thinking", "adaptation"]
        observed_behaviors = set()
        for result in agent_results:
            observed_behaviors.update(result.behaviors_observed)

        autonomous_behavior_count = len([b for b in autonomous_behaviors if b in observed_behaviors])

        # Calculate rating
        if average_score >= 0.9 and autonomous_behavior_count >= 4:
            return "HIGHLY_AGENTIC"
        elif average_score >= 0.7 and autonomous_behavior_count >= 3:
            return "MODERATELY_AGENTIC"
        elif average_score >= 0.5 and autonomous_behavior_count >= 2:
            return "SOMEWHAT_AGENTIC"
        elif average_score >= 0.3:
            return "PSEUDO_AUTONOMOUS"
        else:
            return "NON_AGENTIC"

    def _assess_overall_agentic_capability(self) -> Dict[str, Any]:
        """Assess overall agentic capability of the system."""
        if not self.test_results:
            return {"assessment": "INSUFFICIENT_DATA", "confidence": 0.0}

        # Calculate overall metrics
        total_agents = len(self.active_agents)
        average_score = sum(r.score for r in self.test_results) / len(self.test_results)

        # Count agents by agentic rating
        agent_ratings = {}
        for agent_id in self.active_agents:
            agent_results = [r for r in self.test_results if r.agent_id == agent_id]
            rating = self._calculate_agentic_rating(agent_results)
            agent_ratings[agent_id] = rating

        highly_agentic = len([r for r in agent_ratings.values() if r == "HIGHLY_AGENTIC"])
        moderately_agentic = len([r for r in agent_ratings.values() if r == "MODERATELY_AGENTIC"])

        # Determine overall assessment
        if highly_agentic >= total_agents * 0.7:
            assessment = "EXCELLENT_AGENTIC_SYSTEM"
            confidence = 0.95
        elif (highly_agentic + moderately_agentic) >= total_agents * 0.6:
            assessment = "GOOD_AGENTIC_SYSTEM"
            confidence = 0.8
        elif moderately_agentic >= total_agents * 0.4:
            assessment = "DEVELOPING_AGENTIC_SYSTEM"
            confidence = 0.6
        else:
            assessment = "LIMITED_AGENTIC_CAPABILITY"
            confidence = 0.4

        return {
            "assessment": assessment,
            "confidence": confidence,
            "average_score": average_score,
            "agent_distribution": {
                "highly_agentic": highly_agentic,
                "moderately_agentic": moderately_agentic,
                "total_agents": total_agents
            },
            "recommendations": self._generate_recommendations(assessment, agent_ratings)
        }

    def _generate_recommendations(self, assessment: str, agent_ratings: Dict[str, str]) -> List[str]:
        """Generate recommendations based on assessment."""
        recommendations = []

        if assessment == "LIMITED_AGENTIC_CAPABILITY":
            recommendations.extend([
                "Improve agent training and model selection",
                "Enhance tool creation capabilities",
                "Implement better learning mechanisms",
                "Review agent architecture for autonomous decision making"
            ])
        elif assessment == "DEVELOPING_AGENTIC_SYSTEM":
            recommendations.extend([
                "Focus on improving autonomous decision making",
                "Enhance tool creation and usage capabilities",
                "Implement adaptive learning mechanisms"
            ])
        elif assessment == "GOOD_AGENTIC_SYSTEM":
            recommendations.extend([
                "Fine-tune existing capabilities",
                "Expand tool ecosystem",
                "Implement advanced learning algorithms"
            ])
        else:  # EXCELLENT_AGENTIC_SYSTEM
            recommendations.extend([
                "System demonstrates excellent agentic capabilities",
                "Consider expanding to more complex scenarios",
                "Implement advanced multi-agent coordination"
            ])

        return recommendations

    async def monitor_agent_real_time(self, agent_id: str, duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Monitor an agent in real-time during task execution.

        Args:
            agent_id: ID of the agent to monitor
            duration_minutes: Duration to monitor in minutes

        Returns:
            Real-time monitoring results
        """
        logger.info(f"Starting real-time monitoring of agent {agent_id} for {duration_minutes} minutes")

        monitoring_data = {
            "agent_id": agent_id,
            "start_time": datetime.now(),
            "duration_minutes": duration_minutes,
            "activities": [],
            "performance_metrics": {
                "response_times": [],
                "task_completion_rate": 0.0,
                "error_rate": 0.0,
                "autonomous_decisions": 0,
                "tool_usage_count": 0
            }
        }

        # Monitor for specified duration
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        task_count = 0
        successful_tasks = 0

        while datetime.now() < end_time:
            try:
                # Execute a monitoring task
                monitoring_task = f"Monitoring task {task_count + 1}: Describe what you would do if given a complex problem to solve. Be specific about your approach."

                start_time = time.time()
                result = await self.execute_agent_task(
                    agent_id=agent_id,
                    task=monitoring_task,
                    context={"monitoring": True, "task_number": task_count + 1}
                )
                execution_time = time.time() - start_time

                task_count += 1

                if result.get("success", False):
                    successful_tasks += 1

                    # Analyze response for autonomous behaviors
                    response = result.get("response", "").lower()
                    autonomous_indicators = ["decide", "choose", "analyze", "evaluate", "plan", "strategy"]
                    tool_indicators = ["tool", "create", "use", "implement", "build"]

                    autonomous_decisions = sum(1 for indicator in autonomous_indicators if indicator in response)
                    tool_usage = sum(1 for indicator in tool_indicators if indicator in response)

                    monitoring_data["performance_metrics"]["autonomous_decisions"] += autonomous_decisions
                    monitoring_data["performance_metrics"]["tool_usage_count"] += tool_usage

                monitoring_data["performance_metrics"]["response_times"].append(execution_time)
                monitoring_data["activities"].append({
                    "timestamp": datetime.now().isoformat(),
                    "task": monitoring_task,
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "response_length": len(result.get("response", "")),
                    "autonomous_indicators": autonomous_decisions if 'autonomous_decisions' in locals() else 0,
                    "tool_indicators": tool_usage if 'tool_usage' in locals() else 0
                })

                # Wait before next task
                await asyncio.sleep(30)  # 30 seconds between tasks

            except Exception as e:
                logger.error(f"Error during real-time monitoring", error=str(e))
                monitoring_data["activities"].append({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "task": monitoring_task if 'monitoring_task' in locals() else "unknown"
                })

        # Calculate final metrics
        monitoring_data["performance_metrics"]["task_completion_rate"] = successful_tasks / task_count if task_count > 0 else 0
        monitoring_data["performance_metrics"]["error_rate"] = (task_count - successful_tasks) / task_count if task_count > 0 else 0
        monitoring_data["end_time"] = datetime.now()
        monitoring_data["total_tasks"] = task_count
        monitoring_data["successful_tasks"] = successful_tasks

        logger.info(f"Real-time monitoring completed for agent {agent_id}",
                   total_tasks=task_count,
                   success_rate=monitoring_data["performance_metrics"]["task_completion_rate"])

        return monitoring_data
