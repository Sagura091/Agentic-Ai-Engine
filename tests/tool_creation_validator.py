#!/usr/bin/env python3
"""
Tool Creation Validator for Agent Testing.

This module tests the dynamic tool creation capabilities of agents to ensure
they can create, register, and use tools autonomously.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import structlog
import requests

logger = structlog.get_logger(__name__)


@dataclass
class ToolCreationTest:
    """Test case for tool creation validation."""
    test_id: str
    name: str
    description: str
    tool_requirement: str
    expected_functionality: str
    validation_task: str
    success_criteria: List[str]
    timeout_seconds: int = 120


class ToolCreationValidator:
    """
    Validator for dynamic tool creation capabilities.
    
    This validator tests whether agents can:
    1. Create tools dynamically based on requirements
    2. Register tools properly in the system
    3. Use created tools to solve problems
    4. Demonstrate understanding of tool functionality
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the tool creation validator.
        
        Args:
            base_url: Base URL of the agent orchestration API
        """
        self.base_url = base_url
        self.test_session_id = f"tool_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.created_tools: List[Dict[str, Any]] = []
        self.test_results: List[Dict[str, Any]] = []
        
        # Define tool creation test cases
        self.test_cases = self._create_tool_test_cases()
        
        logger.info("Tool creation validator initialized", session_id=self.test_session_id)
    
    def _create_tool_test_cases(self) -> List[ToolCreationTest]:
        """Create test cases for tool creation validation."""
        return [
            ToolCreationTest(
                test_id="tool_001",
                name="Temperature Converter Tool",
                description="Create a tool that converts temperatures between Celsius and Fahrenheit",
                tool_requirement="Create a temperature conversion tool that can convert between Celsius and Fahrenheit",
                expected_functionality="Convert temperatures bidirectionally with accurate formulas",
                validation_task="Use your created tool to convert 25°C to Fahrenheit and 77°F to Celsius",
                success_criteria=["tool_created", "correct_c_to_f_conversion", "correct_f_to_c_conversion", "tool_used_properly"],
                timeout_seconds=90
            ),
            
            ToolCreationTest(
                test_id="tool_002",
                name="Text Analysis Tool",
                description="Create a tool that analyzes text for word count, character count, and sentiment",
                tool_requirement="Create a text analysis tool that counts words, characters, and determines basic sentiment",
                expected_functionality="Analyze text and return word count, character count, and sentiment (positive/negative/neutral)",
                validation_task="Use your tool to analyze this text: 'I love working with AI agents! They are incredibly helpful and efficient.'",
                success_criteria=["tool_created", "word_count_accurate", "character_count_accurate", "sentiment_detected", "tool_used_properly"],
                timeout_seconds=120
            ),
            
            ToolCreationTest(
                test_id="tool_003",
                name="Math Calculator Tool",
                description="Create a tool that performs advanced mathematical operations",
                tool_requirement="Create a calculator tool that can perform basic arithmetic, square roots, and exponents",
                expected_functionality="Calculate arithmetic operations, square roots, and power operations",
                validation_task="Use your tool to calculate: (15 + 27) * 2, square root of 144, and 3 to the power of 4",
                success_criteria=["tool_created", "arithmetic_correct", "sqrt_correct", "power_correct", "tool_used_properly"],
                timeout_seconds=90
            ),
            
            ToolCreationTest(
                test_id="tool_004",
                name="Data Formatter Tool",
                description="Create a tool that formats data into different structures",
                tool_requirement="Create a data formatting tool that can convert data between JSON, CSV, and plain text formats",
                expected_functionality="Convert data between different formats while preserving structure",
                validation_task="Use your tool to convert this JSON to CSV format: {'name': 'John', 'age': 30, 'city': 'New York'}",
                success_criteria=["tool_created", "json_parsed_correctly", "csv_format_correct", "data_preserved", "tool_used_properly"],
                timeout_seconds=120
            ),
            
            ToolCreationTest(
                test_id="tool_005",
                name="Creative Problem Solver Tool",
                description="Create a tool that generates creative solutions to problems",
                tool_requirement="Create a creative problem-solving tool that generates multiple solution approaches",
                expected_functionality="Generate creative and diverse solutions to given problems",
                validation_task="Use your tool to generate solutions for: 'How to organize a team meeting when everyone is in different time zones'",
                success_criteria=["tool_created", "multiple_solutions_generated", "solutions_creative", "solutions_practical", "tool_used_properly"],
                timeout_seconds=150
            )
        ]
    
    async def create_test_agent_with_tools(self, agent_type: str = "autonomous") -> str:
        """
        Create a test agent with tool creation capabilities.
        
        Args:
            agent_type: Type of agent to create
            
        Returns:
            Agent ID
        """
        try:
            agent_data = {
                "agent_type": agent_type,
                "name": f"ToolCreatorAgent_{self.test_session_id}",
                "description": "Agent for testing dynamic tool creation capabilities",
                "model": "gpt-oss:120b",
                "config": {
                    "temperature": 0.7,
                    "max_tokens": 3000,
                    "enable_tool_creation": True,
                    "enable_learning": True,
                    "tool_creation_autonomy": True
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
                logger.info("Tool creation test agent created", agent_id=agent_id)
                return agent_id
            else:
                logger.error("Failed to create tool creation agent", status_code=response.status_code)
                raise Exception(f"Failed to create agent: {response.status_code}")
                
        except Exception as e:
            logger.error("Error creating tool creation agent", error=str(e))
            raise
    
    async def test_tool_creation(self, agent_id: str, test_case: ToolCreationTest) -> Dict[str, Any]:
        """
        Test tool creation with a specific test case.
        
        Args:
            agent_id: ID of the agent to test
            test_case: Tool creation test case
            
        Returns:
            Test result
        """
        logger.info("Testing tool creation", test_id=test_case.test_id, agent_id=agent_id)
        
        start_time = time.time()
        
        try:
            # Step 1: Request tool creation
            creation_task = f"""
            {test_case.tool_requirement}
            
            Requirements:
            - {test_case.expected_functionality}
            - The tool should be properly named and documented
            - The tool should be registered for use
            
            After creating the tool, please confirm it was created successfully.
            """
            
            creation_result = await self._execute_agent_task(
                agent_id=agent_id,
                task=creation_task,
                context={"test_id": test_case.test_id, "phase": "creation"}
            )
            
            # Step 2: Test tool usage
            if creation_result.get("success", False):
                usage_result = await self._execute_agent_task(
                    agent_id=agent_id,
                    task=test_case.validation_task,
                    context={"test_id": test_case.test_id, "phase": "validation"}
                )
            else:
                usage_result = {"success": False, "error": "Tool creation failed"}
            
            execution_time = time.time() - start_time
            
            # Analyze results
            analysis = self._analyze_tool_creation_result(test_case, creation_result, usage_result)
            
            test_result = {
                "test_id": test_case.test_id,
                "agent_id": agent_id,
                "test_name": test_case.name,
                "execution_time": execution_time,
                "creation_result": creation_result,
                "usage_result": usage_result,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(test_result)
            
            logger.info("Tool creation test completed", 
                       test_id=test_case.test_id,
                       success=analysis["overall_success"],
                       score=analysis["score"])
            
            return test_result
            
        except Exception as e:
            logger.error("Error in tool creation test", test_id=test_case.test_id, error=str(e))
            
            test_result = {
                "test_id": test_case.test_id,
                "agent_id": agent_id,
                "test_name": test_case.name,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "analysis": {"overall_success": False, "score": 0.0, "criteria_met": [], "criteria_failed": test_case.success_criteria},
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(test_result)
            return test_result
    
    async def _execute_agent_task(self, agent_id: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an agent."""
        try:
            task_data = {
                "agent_id": agent_id,
                "task": task,
                "context": context
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/orchestration/agents/execute",
                json=task_data,
                timeout=180  # 3 minute timeout for tool creation
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_tool_creation_result(self, test_case: ToolCreationTest, creation_result: Dict[str, Any], usage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tool creation test results."""
        criteria_met = []
        criteria_failed = []
        
        creation_response = creation_result.get("response", "").lower()
        usage_response = usage_result.get("response", "").lower()
        
        # Analyze based on test case
        if test_case.test_id == "tool_001":  # Temperature converter
            criteria_met, criteria_failed = self._analyze_temperature_tool(creation_response, usage_response, test_case.success_criteria)
        elif test_case.test_id == "tool_002":  # Text analysis
            criteria_met, criteria_failed = self._analyze_text_analysis_tool(creation_response, usage_response, test_case.success_criteria)
        elif test_case.test_id == "tool_003":  # Math calculator
            criteria_met, criteria_failed = self._analyze_math_tool(creation_response, usage_response, test_case.success_criteria)
        elif test_case.test_id == "tool_004":  # Data formatter
            criteria_met, criteria_failed = self._analyze_data_formatter_tool(creation_response, usage_response, test_case.success_criteria)
        elif test_case.test_id == "tool_005":  # Creative problem solver
            criteria_met, criteria_failed = self._analyze_creative_tool(creation_response, usage_response, test_case.success_criteria)
        else:
            # Generic analysis
            if "tool" in creation_response and "create" in creation_response:
                criteria_met.append("tool_created")
            else:
                criteria_failed.append("tool_created")
            
            remaining_criteria = [c for c in test_case.success_criteria if c not in criteria_met]
            criteria_failed.extend(remaining_criteria)
        
        score = len(criteria_met) / len(test_case.success_criteria) if test_case.success_criteria else 0.0
        overall_success = score >= 0.8
        
        return {
            "overall_success": overall_success,
            "score": score,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
            "creation_success": creation_result.get("success", False),
            "usage_success": usage_result.get("success", False),
            "details": {
                "creation_analysis": "Tool creation attempted" if "tool" in creation_response else "No tool creation detected",
                "usage_analysis": "Tool usage attempted" if usage_result.get("success", False) else "Tool usage failed"
            }
        }
    
    def _analyze_temperature_tool(self, creation_response: str, usage_response: str, criteria: List[str]) -> tuple:
        """Analyze temperature conversion tool results."""
        criteria_met = []
        criteria_failed = []
        
        # Check tool creation
        if "tool" in creation_response and ("temperature" in creation_response or "convert" in creation_response):
            criteria_met.append("tool_created")
        else:
            criteria_failed.append("tool_created")
        
        # Check conversions (25°C = 77°F, 77°F = 25°C)
        if "77" in usage_response:
            criteria_met.append("correct_c_to_f_conversion")
        else:
            criteria_failed.append("correct_c_to_f_conversion")
        
        if "25" in usage_response:
            criteria_met.append("correct_f_to_c_conversion")
        else:
            criteria_failed.append("correct_f_to_c_conversion")
        
        # Check tool usage
        if "use" in usage_response or "convert" in usage_response:
            criteria_met.append("tool_used_properly")
        else:
            criteria_failed.append("tool_used_properly")
        
        return criteria_met, criteria_failed
