"""
Self-Modification Capabilities for Truly Agentic AI.

This module implements the ability for agents to modify their own code, capabilities,
and behavior patterns, including:
- Dynamic code generation and integration
- Capability extension and enhancement
- Behavioral pattern modification
- Self-optimization and improvement
- Safe self-modification with constraints
"""

import asyncio
import json
import uuid
import ast
import inspect
import importlib
import types
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sys
import os
from pathlib import Path

import structlog
from langchain_core.language_models import BaseLanguageModel

logger = structlog.get_logger(__name__)


class ModificationType(str, Enum):
    """Types of self-modifications."""
    CODE_GENERATION = "code_generation"     # Generate new code/functions
    CAPABILITY_EXTENSION = "capability_extension"  # Add new capabilities
    BEHAVIOR_MODIFICATION = "behavior_modification"  # Modify behavior patterns
    OPTIMIZATION = "optimization"          # Optimize existing code
    PARAMETER_TUNING = "parameter_tuning"  # Adjust parameters
    ARCHITECTURE_CHANGE = "architecture_change"  # Modify agent architecture
    TOOL_CREATION = "tool_creation"        # Create new tools
    SKILL_LEARNING = "skill_learning"      # Learn new skills


class SafetyLevel(str, Enum):
    """Safety levels for self-modifications."""
    SAFE = "safe"                          # Safe modifications only
    CAUTIOUS = "cautious"                  # Moderate risk modifications
    EXPERIMENTAL = "experimental"          # High risk modifications
    UNRESTRICTED = "unrestricted"          # No safety constraints


class ModificationStatus(str, Enum):
    """Status of a modification."""
    PLANNED = "planned"                    # Modification planned
    GENERATING = "generating"              # Code being generated
    TESTING = "testing"                    # Testing modification
    INTEGRATING = "integrating"            # Integrating into agent
    ACTIVE = "active"                      # Modification active
    FAILED = "failed"                      # Modification failed
    REVERTED = "reverted"                  # Modification reverted


@dataclass
class SelfModification:
    """Represents a self-modification made by the agent."""
    modification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    modification_type: ModificationType = ModificationType.CODE_GENERATION
    description: str = ""
    
    # Modification details
    target_component: str = ""             # What component to modify
    modification_code: str = ""            # Generated code
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Safety and validation
    safety_level: SafetyLevel = SafetyLevel.SAFE
    safety_checks: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    status: ModificationStatus = ModificationStatus.PLANNED
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Performance tracking
    performance_before: Dict[str, float] = field(default_factory=dict)
    performance_after: Dict[str, float] = field(default_factory=dict)
    improvement_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Rollback information
    backup_code: str = ""                  # Original code for rollback
    rollback_available: bool = True
    
    # Metadata
    reasoning: str = ""                    # Why this modification was made
    expected_benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedCapability:
    """Represents a dynamically generated capability."""
    capability_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Implementation
    function_code: str = ""
    function_name: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = "Any"
    
    # Integration
    module_name: str = ""
    is_integrated: bool = False
    integration_path: str = ""
    
    # Performance
    usage_count: int = 0
    success_rate: float = 1.0
    average_execution_time: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfModificationEngine:
    """
    Advanced self-modification engine for autonomous agents.
    
    Enables agents to modify their own code, capabilities, and behavior
    patterns while maintaining safety and performance constraints.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseLanguageModel,
        safety_level: SafetyLevel = SafetyLevel.SAFE,
        max_modifications: int = 100
    ):
        """Initialize the self-modification engine."""
        self.agent_id = agent_id
        self.llm = llm
        self.safety_level = safety_level
        self.max_modifications = max_modifications
        
        # Modification tracking
        self.modifications: Dict[str, SelfModification] = {}
        self.generated_capabilities: Dict[str, GeneratedCapability] = {}
        self.modification_history: List[str] = []
        
        # Safety constraints
        self.safety_constraints = {
            "forbidden_imports": ["os", "sys", "subprocess", "eval", "exec"],
            "forbidden_functions": ["eval", "exec", "compile", "__import__"],
            "max_code_length": 10000,
            "max_execution_time": 30.0,
            "require_testing": True
        }
        
        # Performance tracking
        self.performance_baseline: Dict[str, float] = {}
        self.modification_stats = {
            "total_modifications": 0,
            "successful_modifications": 0,
            "failed_modifications": 0,
            "reverted_modifications": 0,
            "generated_capabilities": 0,
            "performance_improvements": 0
        }
        
        # Code generation templates
        self.code_templates = {
            "function": """
async def {function_name}({parameters}):
    \"\"\"
    {description}
    \"\"\"
    try:
        {implementation}
        return result
    except Exception as e:
        logger.error("Function execution failed", function="{function_name}", error=str(e))
        raise
""",
            "class": """
class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self, {init_parameters}):
        {init_implementation}
    
    {methods}
""",
            "tool": """
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class {tool_name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self):
        self.name = "{tool_name}"
        self.description = "{description}"
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Execute the tool with given parameters.\"\"\"
        try:
            {implementation}
            return {{"success": True, "result": result}}
        except Exception as e:
            logger.error("Tool execution failed", tool=self.name, error=str(e))
            return {{"success": False, "error": str(e)}}
"""
        }
        
        logger.info(
            "Self-modification engine initialized",
            agent_id=agent_id,
            safety_level=safety_level.value,
            max_modifications=max_modifications
        )
    
    async def analyze_improvement_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze current agent performance to identify improvement opportunities."""
        try:
            opportunities = []
            
            # Analyze performance metrics
            current_performance = await self._get_current_performance_metrics()
            
            # Identify bottlenecks
            bottlenecks = await self._identify_performance_bottlenecks(current_performance)
            for bottleneck in bottlenecks:
                opportunities.append({
                    "type": "performance_optimization",
                    "target": bottleneck["component"],
                    "issue": bottleneck["issue"],
                    "potential_improvement": bottleneck["potential_gain"],
                    "priority": "high" if bottleneck["potential_gain"] > 0.2 else "medium"
                })
            
            # Identify missing capabilities
            missing_capabilities = await self._identify_missing_capabilities()
            for capability in missing_capabilities:
                opportunities.append({
                    "type": "capability_extension",
                    "target": "new_capability",
                    "capability": capability["name"],
                    "description": capability["description"],
                    "priority": capability["priority"]
                })
            
            # Identify behavior improvements
            behavior_improvements = await self._identify_behavior_improvements()
            for improvement in behavior_improvements:
                opportunities.append({
                    "type": "behavior_modification",
                    "target": improvement["component"],
                    "current_behavior": improvement["current"],
                    "suggested_behavior": improvement["suggested"],
                    "priority": improvement["priority"]
                })
            
            logger.info(
                "Improvement opportunities analyzed",
                total_opportunities=len(opportunities),
                performance_optimizations=len([o for o in opportunities if o["type"] == "performance_optimization"]),
                capability_extensions=len([o for o in opportunities if o["type"] == "capability_extension"])
            )
            
            return opportunities
            
        except Exception as e:
            logger.error("Failed to analyze improvement opportunities", error=str(e))
            return []
    
    async def plan_self_modification(
        self,
        opportunity: Dict[str, Any],
        modification_type: ModificationType = None
    ) -> Optional[SelfModification]:
        """Plan a self-modification based on an improvement opportunity."""
        try:
            if modification_type is None:
                modification_type = ModificationType(opportunity["type"])
            
            # Create modification plan
            modification = SelfModification(
                modification_type=modification_type,
                description=f"Improve {opportunity['target']}: {opportunity.get('issue', 'enhancement')}",
                target_component=opportunity["target"],
                safety_level=self.safety_level,
                reasoning=f"Identified improvement opportunity: {opportunity.get('description', 'performance enhancement')}"
            )
            
            # Set expected benefits
            if "potential_improvement" in opportunity:
                modification.expected_benefits.append(f"Performance improvement: {opportunity['potential_improvement']}")
            
            if "capability" in opportunity:
                modification.expected_benefits.append(f"New capability: {opportunity['capability']}")
            
            # Assess risks based on modification type
            risks = await self._assess_modification_risks(modification)
            modification.risks = risks
            
            # Generate modification code
            code_generated = await self._generate_modification_code(modification, opportunity)
            
            if not code_generated:
                logger.warning("Failed to generate modification code")
                return None
            
            # Perform safety checks
            safety_passed = await self._perform_safety_checks(modification)
            
            if not safety_passed:
                logger.warning("Modification failed safety checks", modification_id=modification.modification_id)
                return None
            
            # Store modification
            self.modifications[modification.modification_id] = modification
            
            logger.info(
                "Self-modification planned",
                modification_id=modification.modification_id,
                type=modification_type.value,
                target=opportunity["target"]
            )
            
            return modification
            
        except Exception as e:
            logger.error("Failed to plan self-modification", error=str(e))
            return None

    async def execute_self_modification(self, modification_id: str) -> bool:
        """Execute a planned self-modification."""
        try:
            modification = self.modifications.get(modification_id)
            if not modification:
                raise ValueError(f"Modification {modification_id} not found")

            if modification.status != ModificationStatus.PLANNED:
                logger.warning("Modification not in planned state", modification_id=modification_id, status=modification.status.value)
                return False

            # Update status
            modification.status = ModificationStatus.TESTING
            modification.executed_at = datetime.utcnow()

            # Capture performance baseline
            modification.performance_before = await self._capture_performance_metrics()

            # Test modification in safe environment
            test_passed = await self._test_modification(modification)

            if not test_passed:
                modification.status = ModificationStatus.FAILED
                logger.error("Modification failed testing", modification_id=modification_id)
                return False

            # Integrate modification
            modification.status = ModificationStatus.INTEGRATING
            integration_success = await self._integrate_modification(modification)

            if not integration_success:
                modification.status = ModificationStatus.FAILED
                logger.error("Modification integration failed", modification_id=modification_id)
                return False

            # Activate modification
            modification.status = ModificationStatus.ACTIVE
            modification.completed_at = datetime.utcnow()

            # Capture post-modification performance
            await asyncio.sleep(1.0)  # Allow time for effects
            modification.performance_after = await self._capture_performance_metrics()

            # Calculate improvement metrics
            modification.improvement_metrics = await self._calculate_improvement_metrics(
                modification.performance_before,
                modification.performance_after
            )

            # Update statistics
            self.modification_stats["total_modifications"] += 1
            self.modification_stats["successful_modifications"] += 1

            if any(metric > 0 for metric in modification.improvement_metrics.values()):
                self.modification_stats["performance_improvements"] += 1

            # Add to history
            self.modification_history.append(modification_id)

            logger.info(
                "Self-modification executed successfully",
                modification_id=modification_id,
                type=modification.modification_type.value,
                improvements=modification.improvement_metrics
            )

            return True

        except Exception as e:
            logger.error("Failed to execute self-modification", modification_id=modification_id, error=str(e))

            # Update failure statistics
            modification = self.modifications.get(modification_id)
            if modification:
                modification.status = ModificationStatus.FAILED
                self.modification_stats["failed_modifications"] += 1

            return False

    async def generate_new_capability(
        self,
        capability_name: str,
        description: str,
        requirements: Dict[str, Any] = None
    ) -> Optional[GeneratedCapability]:
        """Generate a new capability for the agent."""
        try:
            requirements = requirements or {}

            # Create capability specification
            capability = GeneratedCapability(
                name=capability_name,
                description=description,
                function_name=capability_name.lower().replace(" ", "_"),
                tags=requirements.get("tags", [])
            )

            # Generate function code
            function_code = await self._generate_capability_code(capability, requirements)

            if not function_code:
                logger.warning("Failed to generate capability code", capability=capability_name)
                return None

            capability.function_code = function_code

            # Validate generated code
            validation_passed = await self._validate_generated_code(capability.function_code)

            if not validation_passed:
                logger.warning("Generated capability failed validation", capability=capability_name)
                return None

            # Test capability
            test_passed = await self._test_generated_capability(capability)

            if not test_passed:
                logger.warning("Generated capability failed testing", capability=capability_name)
                return None

            # Integrate capability
            integration_success = await self._integrate_capability(capability)

            if integration_success:
                capability.is_integrated = True
                self.generated_capabilities[capability.capability_id] = capability
                self.modification_stats["generated_capabilities"] += 1

                logger.info(
                    "New capability generated and integrated",
                    capability_id=capability.capability_id,
                    name=capability_name
                )

                return capability
            else:
                logger.warning("Failed to integrate generated capability", capability=capability_name)
                return None

        except Exception as e:
            logger.error("Failed to generate new capability", capability=capability_name, error=str(e))
            return None

    async def revert_modification(self, modification_id: str) -> bool:
        """Revert a previously applied modification."""
        try:
            modification = self.modifications.get(modification_id)
            if not modification:
                raise ValueError(f"Modification {modification_id} not found")

            if modification.status != ModificationStatus.ACTIVE:
                logger.warning("Modification not active, cannot revert", modification_id=modification_id)
                return False

            if not modification.rollback_available:
                logger.warning("Rollback not available for modification", modification_id=modification_id)
                return False

            # Perform rollback
            rollback_success = await self._perform_rollback(modification)

            if rollback_success:
                modification.status = ModificationStatus.REVERTED
                self.modification_stats["reverted_modifications"] += 1

                logger.info("Modification reverted successfully", modification_id=modification_id)
                return True
            else:
                logger.error("Failed to revert modification", modification_id=modification_id)
                return False

        except Exception as e:
            logger.error("Failed to revert modification", modification_id=modification_id, error=str(e))
            return False

    async def _get_current_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics for the agent."""
        try:
            # Placeholder metrics - in real implementation would measure actual performance
            metrics = {
                "response_time": 1.0,
                "accuracy": 0.85,
                "efficiency": 0.75,
                "memory_usage": 0.6,
                "cpu_usage": 0.4,
                "task_completion_rate": 0.9,
                "error_rate": 0.05
            }

            return metrics

        except Exception as e:
            logger.error("Failed to get performance metrics", error=str(e))
            return {}

    async def _identify_performance_bottlenecks(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from metrics."""
        try:
            bottlenecks = []

            # Define thresholds for bottleneck detection
            thresholds = {
                "response_time": 2.0,      # > 2 seconds is slow
                "accuracy": 0.8,           # < 80% accuracy is poor
                "efficiency": 0.7,         # < 70% efficiency is low
                "memory_usage": 0.8,       # > 80% memory usage is high
                "cpu_usage": 0.8,          # > 80% CPU usage is high
                "error_rate": 0.1          # > 10% error rate is high
            }

            for metric, value in metrics.items():
                if metric in ["response_time", "memory_usage", "cpu_usage", "error_rate"]:
                    # Higher is worse for these metrics
                    if value > thresholds[metric]:
                        bottlenecks.append({
                            "component": metric,
                            "issue": f"High {metric}: {value:.2f}",
                            "current_value": value,
                            "threshold": thresholds[metric],
                            "potential_gain": min(0.5, (value - thresholds[metric]) / value)
                        })
                else:
                    # Lower is worse for these metrics
                    if value < thresholds[metric]:
                        bottlenecks.append({
                            "component": metric,
                            "issue": f"Low {metric}: {value:.2f}",
                            "current_value": value,
                            "threshold": thresholds[metric],
                            "potential_gain": min(0.5, (thresholds[metric] - value) / thresholds[metric])
                        })

            return bottlenecks

        except Exception as e:
            logger.error("Failed to identify performance bottlenecks", error=str(e))
            return []

    async def _identify_missing_capabilities(self) -> List[Dict[str, Any]]:
        """Identify capabilities that the agent is missing."""
        try:
            # Analyze recent tasks and identify gaps
            missing_capabilities = [
                {
                    "name": "advanced_data_analysis",
                    "description": "Advanced statistical analysis and data visualization",
                    "priority": "high",
                    "reasoning": "Frequently requested but not available"
                },
                {
                    "name": "multi_language_translation",
                    "description": "Real-time translation between multiple languages",
                    "priority": "medium",
                    "reasoning": "Would expand communication capabilities"
                },
                {
                    "name": "image_generation",
                    "description": "Generate images based on text descriptions",
                    "priority": "medium",
                    "reasoning": "Creative tasks often require visual output"
                },
                {
                    "name": "code_optimization",
                    "description": "Automatically optimize code for performance",
                    "priority": "high",
                    "reasoning": "Performance improvements are always valuable"
                }
            ]

            # Filter based on current capabilities
            current_capabilities = set(cap.name for cap in self.generated_capabilities.values())

            missing = [
                cap for cap in missing_capabilities
                if cap["name"] not in current_capabilities
            ]

            return missing

        except Exception as e:
            logger.error("Failed to identify missing capabilities", error=str(e))
            return []

    async def _identify_behavior_improvements(self) -> List[Dict[str, Any]]:
        """Identify potential behavior improvements."""
        try:
            improvements = [
                {
                    "component": "decision_making",
                    "current": "reactive_decision_making",
                    "suggested": "proactive_decision_making",
                    "priority": "high",
                    "reasoning": "Proactive decisions lead to better outcomes"
                },
                {
                    "component": "error_handling",
                    "current": "basic_error_handling",
                    "suggested": "adaptive_error_recovery",
                    "priority": "medium",
                    "reasoning": "Better error recovery improves reliability"
                },
                {
                    "component": "learning_strategy",
                    "current": "passive_learning",
                    "suggested": "active_learning",
                    "priority": "high",
                    "reasoning": "Active learning accelerates improvement"
                }
            ]

            return improvements

        except Exception as e:
            logger.error("Failed to identify behavior improvements", error=str(e))
            return []

    async def _assess_modification_risks(self, modification: SelfModification) -> List[str]:
        """Assess risks associated with a modification."""
        try:
            risks = []

            # Risk based on modification type
            if modification.modification_type == ModificationType.ARCHITECTURE_CHANGE:
                risks.append("Potential system instability")
                risks.append("Possible performance degradation")

            elif modification.modification_type == ModificationType.CODE_GENERATION:
                risks.append("Potential security vulnerabilities")
                risks.append("Possible runtime errors")

            elif modification.modification_type == ModificationType.BEHAVIOR_MODIFICATION:
                risks.append("Unexpected behavior changes")
                risks.append("Possible goal misalignment")

            # Risk based on safety level
            if self.safety_level == SafetyLevel.EXPERIMENTAL:
                risks.append("Experimental modification - higher failure risk")

            elif self.safety_level == SafetyLevel.UNRESTRICTED:
                risks.append("Unrestricted modification - maximum risk")

            # Risk based on target component
            critical_components = ["decision_engine", "memory_system", "safety_system"]
            if modification.target_component in critical_components:
                risks.append(f"Modifying critical component: {modification.target_component}")

            return risks

        except Exception as e:
            logger.error("Failed to assess modification risks", error=str(e))
            return ["Unknown risks due to assessment failure"]

    async def _generate_modification_code(
        self,
        modification: SelfModification,
        opportunity: Dict[str, Any]
    ) -> bool:
        """Generate code for the modification."""
        try:
            modification.status = ModificationStatus.GENERATING

            if modification.modification_type == ModificationType.CODE_GENERATION:
                # Generate new function code
                code = await self._generate_function_code(opportunity)

            elif modification.modification_type == ModificationType.CAPABILITY_EXTENSION:
                # Generate capability extension code
                code = await self._generate_capability_extension_code(opportunity)

            elif modification.modification_type == ModificationType.OPTIMIZATION:
                # Generate optimization code
                code = await self._generate_optimization_code(opportunity)

            elif modification.modification_type == ModificationType.TOOL_CREATION:
                # Generate new tool code
                code = await self._generate_tool_code(opportunity)

            else:
                # Generic code generation
                code = await self._generate_generic_code(opportunity)

            if code:
                modification.modification_code = code
                return True
            else:
                logger.warning("No code generated for modification")
                return False

        except Exception as e:
            logger.error("Failed to generate modification code", error=str(e))
            return False

    async def _perform_safety_checks(self, modification: SelfModification) -> bool:
        """Perform safety checks on a modification."""
        try:
            safety_checks = []

            # Check code length
            if len(modification.modification_code) > self.safety_constraints["max_code_length"]:
                safety_checks.append("Code length exceeds maximum allowed")
                return False

            # Check for forbidden imports
            for forbidden in self.safety_constraints["forbidden_imports"]:
                if f"import {forbidden}" in modification.modification_code:
                    safety_checks.append(f"Forbidden import detected: {forbidden}")
                    return False

            # Check for forbidden functions
            for forbidden in self.safety_constraints["forbidden_functions"]:
                if forbidden in modification.modification_code:
                    safety_checks.append(f"Forbidden function detected: {forbidden}")
                    return False

            # Syntax check
            try:
                ast.parse(modification.modification_code)
                safety_checks.append("Syntax check passed")
            except SyntaxError as e:
                safety_checks.append(f"Syntax error: {str(e)}")
                return False

            # Security analysis (simplified)
            security_issues = await self._analyze_code_security(modification.modification_code)
            if security_issues:
                safety_checks.extend(security_issues)
                return False

            modification.safety_checks = safety_checks
            return True

        except Exception as e:
            logger.error("Failed to perform safety checks", error=str(e))
            return False

    async def _test_modification(self, modification: SelfModification) -> bool:
        """Test a modification in a safe environment."""
        try:
            # Create test environment
            test_env = await self._create_test_environment()

            # Execute modification code in test environment
            test_result = await self._execute_in_test_environment(
                modification.modification_code,
                test_env
            )

            # Validate test results
            if test_result["success"]:
                modification.validation_results = test_result
                return True
            else:
                modification.validation_results = test_result
                logger.warning("Modification failed testing", error=test_result.get("error"))
                return False

        except Exception as e:
            logger.error("Failed to test modification", error=str(e))
            return False

    async def _integrate_modification(self, modification: SelfModification) -> bool:
        """Integrate a modification into the agent."""
        try:
            # Backup current code if rollback is needed
            modification.backup_code = await self._backup_current_code(modification.target_component)

            # Apply modification
            if modification.modification_type == ModificationType.CODE_GENERATION:
                success = await self._integrate_generated_code(modification)

            elif modification.modification_type == ModificationType.CAPABILITY_EXTENSION:
                success = await self._integrate_capability_extension(modification)

            elif modification.modification_type == ModificationType.OPTIMIZATION:
                success = await self._integrate_optimization(modification)

            else:
                success = await self._integrate_generic_modification(modification)

            return success

        except Exception as e:
            logger.error("Failed to integrate modification", error=str(e))
            return False

    async def _capture_performance_metrics(self) -> Dict[str, float]:
        """Capture current performance metrics."""
        try:
            # This would interface with actual performance monitoring
            metrics = await self._get_current_performance_metrics()
            return metrics

        except Exception as e:
            logger.error("Failed to capture performance metrics", error=str(e))
            return {}

    async def _calculate_improvement_metrics(
        self,
        before: Dict[str, float],
        after: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement metrics."""
        try:
            improvements = {}

            for metric in before.keys():
                if metric in after:
                    if metric in ["response_time", "memory_usage", "cpu_usage", "error_rate"]:
                        # Lower is better for these metrics
                        improvement = (before[metric] - after[metric]) / before[metric]
                    else:
                        # Higher is better for these metrics
                        improvement = (after[metric] - before[metric]) / before[metric]

                    improvements[metric] = improvement

            return improvements

        except Exception as e:
            logger.error("Failed to calculate improvement metrics", error=str(e))
            return {}

    async def _generate_function_code(self, opportunity: Dict[str, Any]) -> str:
        """Generate function code for an opportunity."""
        try:
            # Use LLM to generate code based on opportunity
            prompt = f"""
            Generate a Python function to address this improvement opportunity:

            Target: {opportunity['target']}
            Issue: {opportunity.get('issue', 'enhancement needed')}
            Description: {opportunity.get('description', 'No description provided')}

            Requirements:
            - Function should be async
            - Include proper error handling
            - Add logging
            - Follow Python best practices
            - Include docstring

            Generate only the function code, no additional text.
            """

            # In a real implementation, would use the LLM to generate code
            # For now, return a template
            function_name = opportunity['target'].lower().replace(' ', '_') + '_improvement'

            code = f"""
async def {function_name}(parameters: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    Improvement function for {opportunity['target']}.

    Addresses: {opportunity.get('issue', 'enhancement')}
    \"\"\"
    try:
        # Implementation would be generated by LLM
        result = {{"success": True, "improvement": "implemented"}}
        logger.info("Improvement function executed", function="{function_name}")
        return result
    except Exception as e:
        logger.error("Improvement function failed", function="{function_name}", error=str(e))
        return {{"success": False, "error": str(e)}}
"""

            return code.strip()

        except Exception as e:
            logger.error("Failed to generate function code", error=str(e))
            return ""

    async def _analyze_code_security(self, code: str) -> List[str]:
        """Analyze code for security issues."""
        try:
            security_issues = []

            # Check for dangerous patterns
            dangerous_patterns = [
                "eval(",
                "exec(",
                "compile(",
                "__import__",
                "subprocess",
                "os.system",
                "open(",  # File operations might be risky
            ]

            for pattern in dangerous_patterns:
                if pattern in code:
                    security_issues.append(f"Potentially dangerous pattern: {pattern}")

            return security_issues

        except Exception as e:
            logger.error("Failed to analyze code security", error=str(e))
            return ["Security analysis failed"]

    def get_modification_summary(self) -> Dict[str, Any]:
        """Get summary of all modifications."""
        try:
            active_modifications = [
                mod for mod in self.modifications.values()
                if mod.status == ModificationStatus.ACTIVE
            ]

            return {
                "total_modifications": len(self.modifications),
                "active_modifications": len(active_modifications),
                "generated_capabilities": len(self.generated_capabilities),
                "modification_stats": self.modification_stats.copy(),
                "safety_level": self.safety_level.value,
                "recent_modifications": [
                    {
                        "id": mod.modification_id,
                        "type": mod.modification_type.value,
                        "target": mod.target_component,
                        "status": mod.status.value,
                        "created": mod.created_at.isoformat()
                    }
                    for mod in list(self.modifications.values())[-5:]  # Last 5
                ]
            }

        except Exception as e:
            logger.error("Failed to get modification summary", error=str(e))
            return {"error": str(e)}
