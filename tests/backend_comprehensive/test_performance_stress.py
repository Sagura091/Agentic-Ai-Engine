"""
Comprehensive Performance and Stress Testing Suite.

This module tests performance characteristics, concurrent operations,
stress testing with multiple agents, and scalability validation.
"""

import pytest
import asyncio
import time
import psutil
import gc
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

import structlog

# Import test infrastructure
from .base_test import PerformanceTest, StressTest, TestResult, TestCategory, TestSeverity
from .test_utils import TestDataGenerator, MockLLMProvider, AsyncTestHelper

# Import application components
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType
from app.agents.base.agent import LangGraphAgent, AgentConfig
from app.agents.autonomous import AutonomousLangGraphAgent, AutonomousAgentConfig

logger = structlog.get_logger(__name__)


class TestConcurrentAgentOperations(PerformanceTest):
    """Test concurrent agent operations and performance."""
    
    def __init__(self):
        super().__init__("Concurrent Agent Operations", TestSeverity.HIGH)
        self.set_performance_thresholds({
            'max_response_time': 10.0,  # 10 seconds for concurrent operations
            'max_memory_usage_mb': 1000,
            'max_cpu_usage_percent': 90,
            'min_throughput': 5.0  # operations per second
        })
    
    async def execute_test(self) -> TestResult:
        """Test concurrent agent operations."""
        try:
            # Test concurrent agent creation
            creation_test = await self._test_concurrent_agent_creation()
            
            # Test concurrent agent execution
            execution_test = await self._test_concurrent_agent_execution()
            
            # Test concurrent tool usage
            tool_test = await self._test_concurrent_tool_usage()
            
            # Test resource management
            resource_test = await self._test_resource_management()
            
            evidence = {
                "concurrent_creation": creation_test,
                "concurrent_execution": execution_test,
                "concurrent_tools": tool_test,
                "resource_management": resource_test
            }
            
            success = (creation_test["success"] and execution_test["success"] and 
                      tool_test["success"] and resource_test["success"])
            
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
    
    async def _test_concurrent_agent_creation(self) -> Dict[str, Any]:
        """Test concurrent agent creation."""
        try:
            start_time = time.time()
            
            # Create multiple agents concurrently
            async def create_agent(agent_id: int):
                config = AgentConfig(
                    name=f"Concurrent Agent {agent_id}",
                    description=f"Agent {agent_id} for concurrent testing",
                    model_name="mock-model"
                )
                mock_llm = MockLLMProvider()
                agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                await agent.initialize()
                return agent
            
            # Create 10 agents concurrently
            tasks = [create_agent(i) for i in range(10)]
            agents = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Count successful creations
            successful_agents = [a for a in agents if not isinstance(a, Exception)]
            success_rate = len(successful_agents) / len(tasks)
            
            return {
                "success": success_rate >= 0.8,
                "agents_created": len(successful_agents),
                "total_attempted": len(tasks),
                "success_rate": success_rate,
                "duration": duration,
                "throughput": len(successful_agents) / duration if duration > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error testing concurrent agent creation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_concurrent_agent_execution(self) -> Dict[str, Any]:
        """Test concurrent agent execution."""
        try:
            # Create agents for testing
            agents = []
            for i in range(5):
                config = AgentConfig(
                    name=f"Execution Agent {i}",
                    description=f"Agent {i} for execution testing",
                    model_name="mock-model"
                )
                mock_llm = MockLLMProvider()
                agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                await agent.initialize()
                agents.append(agent)
            
            # Execute tasks concurrently
            start_time = time.time()
            
            async def execute_task(agent, task_id):
                return await agent.ainvoke(f"Execute task {task_id}")
            
            tasks = [execute_task(agent, i) for i, agent in enumerate(agents)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Count successful executions
            successful_results = [r for r in results if not isinstance(r, Exception)]
            success_rate = len(successful_results) / len(tasks)
            
            return {
                "success": success_rate >= 0.8,
                "executions_completed": len(successful_results),
                "total_attempted": len(tasks),
                "success_rate": success_rate,
                "duration": duration,
                "throughput": len(successful_results) / duration if duration > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error testing concurrent agent execution: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_concurrent_tool_usage(self) -> Dict[str, Any]:
        """Test concurrent tool usage."""
        try:
            # Create mock tools
            from .test_utils import create_mock_tool
            tools = [await create_mock_tool(f"tool_{i}") for i in range(3)]
            
            # Create agent with tools
            config = AgentConfig(
                name="Tool Test Agent",
                description="Agent for testing concurrent tool usage",
                model_name="mock-model",
                capabilities=["tool_use"]
            )
            mock_llm = MockLLMProvider()
            agent = LangGraphAgent(config=config, llm=mock_llm, tools=tools)
            await agent.initialize()
            
            # Execute tool operations concurrently
            start_time = time.time()
            
            async def use_tool(tool_name, input_data):
                if tool_name in agent.tools:
                    return await agent.tools[tool_name].invoke(input_data)
                return None
            
            tasks = [use_tool(f"tool_{i}", f"input_{i}") for i in range(3) for _ in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            successful_results = [r for r in results if not isinstance(r, Exception) and r is not None]
            success_rate = len(successful_results) / len(tasks)
            
            return {
                "success": success_rate >= 0.8,
                "tool_operations": len(successful_results),
                "total_attempted": len(tasks),
                "success_rate": success_rate,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error testing concurrent tool usage: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_resource_management(self) -> Dict[str, Any]:
        """Test resource management during concurrent operations."""
        try:
            # Monitor resource usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Perform resource-intensive operations
            agents = []
            for i in range(20):
                config = AgentConfig(
                    name=f"Resource Agent {i}",
                    description=f"Agent {i} for resource testing",
                    model_name="mock-model"
                )
                mock_llm = MockLLMProvider()
                agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                await agent.initialize()
                agents.append(agent)
            
            # Execute operations
            tasks = [agent.ainvoke(f"Resource test {i}") for i, agent in enumerate(agents)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check final memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Cleanup
            del agents
            gc.collect()
            
            return {
                "success": memory_increase < 500,  # Less than 500MB increase
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase
            }
            
        except Exception as e:
            logger.error(f"Error testing resource management: {e}")
            return {"success": False, "error": str(e)}


class TestAgentStressTesting(StressTest):
    """Test agent system under stress conditions."""
    
    def __init__(self):
        super().__init__("Agent Stress Testing", TestSeverity.MEDIUM)
        self.set_stress_config({
            'max_concurrent_operations': 100,
            'test_duration_seconds': 120,
            'max_error_rate': 0.1  # 10% error rate
        })
    
    async def execute_test(self) -> TestResult:
        """Execute stress testing scenarios."""
        try:
            # Test high-load agent creation
            creation_stress = await self._test_agent_creation_stress()
            
            # Test sustained operation stress
            operation_stress = await self._test_sustained_operations()
            
            # Test memory stress
            memory_stress = await self._test_memory_stress()
            
            # Test recovery from failures
            recovery_test = await self._test_failure_recovery()
            
            evidence = {
                "creation_stress": creation_stress,
                "operation_stress": operation_stress,
                "memory_stress": memory_stress,
                "failure_recovery": recovery_test
            }
            
            success = (creation_stress["success"] and operation_stress["success"] and 
                      memory_stress["success"] and recovery_test["success"])
            
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
    
    async def _test_agent_creation_stress(self) -> Dict[str, Any]:
        """Test stress conditions for agent creation."""
        try:
            start_time = time.time()
            
            # Create many agents rapidly
            async def create_stress_agent(agent_id: int):
                try:
                    config = AgentConfig(
                        name=f"Stress Agent {agent_id}",
                        description=f"Stress test agent {agent_id}",
                        model_name="mock-model"
                    )
                    mock_llm = MockLLMProvider()
                    agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                    await agent.initialize()
                    return {"success": True, "agent_id": agent_id}
                except Exception as e:
                    return {"success": False, "agent_id": agent_id, "error": str(e)}
            
            # Create 50 agents with limited concurrency
            semaphore = asyncio.Semaphore(10)
            
            async def create_with_semaphore(agent_id):
                async with semaphore:
                    return await create_stress_agent(agent_id)
            
            tasks = [create_with_semaphore(i) for i in range(50)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            successful_creations = sum(1 for r in results if r.get("success", False))
            error_rate = (len(results) - successful_creations) / len(results)
            
            return {
                "success": error_rate <= self.stress_config['max_error_rate'],
                "agents_created": successful_creations,
                "total_attempted": len(results),
                "error_rate": error_rate,
                "duration": duration,
                "creation_rate": successful_creations / duration if duration > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in agent creation stress test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_sustained_operations(self) -> Dict[str, Any]:
        """Test sustained operations over time."""
        try:
            # Create a few agents for sustained testing
            agents = []
            for i in range(5):
                config = AgentConfig(
                    name=f"Sustained Agent {i}",
                    description=f"Agent {i} for sustained testing",
                    model_name="mock-model"
                )
                mock_llm = MockLLMProvider()
                agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                await agent.initialize()
                agents.append(agent)
            
            # Run operations for specified duration
            start_time = time.time()
            end_time = start_time + 30  # 30 seconds for testing
            
            operation_count = 0
            errors = 0
            
            while time.time() < end_time:
                try:
                    # Execute operations on random agents
                    agent = agents[operation_count % len(agents)]
                    await agent.ainvoke(f"Sustained operation {operation_count}")
                    operation_count += 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    errors += 1
                    logger.debug(f"Operation error: {e}")
            
            duration = time.time() - start_time
            error_rate = errors / operation_count if operation_count > 0 else 1.0
            
            return {
                "success": error_rate <= self.stress_config['max_error_rate'],
                "operations_completed": operation_count,
                "errors": errors,
                "error_rate": error_rate,
                "duration": duration,
                "operations_per_second": operation_count / duration if duration > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in sustained operations test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_memory_stress(self) -> Dict[str, Any]:
        """Test memory usage under stress."""
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create many agents to stress memory
            agents = []
            for i in range(30):
                config = AgentConfig(
                    name=f"Memory Stress Agent {i}",
                    description=f"Agent {i} for memory stress testing",
                    model_name="mock-model"
                )
                mock_llm = MockLLMProvider()
                agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                await agent.initialize()
                agents.append(agent)
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Execute operations to increase memory usage
            tasks = [agent.ainvoke(f"Memory test {i}") for i, agent in enumerate(agents)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Cleanup
            del agents
            gc.collect()
            
            cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            
            return {
                "success": memory_increase < 1000,  # Less than 1GB increase
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": peak_memory,
                "final_memory_mb": final_memory,
                "cleanup_memory_mb": cleanup_memory,
                "memory_increase_mb": memory_increase
            }
            
        except Exception as e:
            logger.error(f"Error in memory stress test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_failure_recovery(self) -> Dict[str, Any]:
        """Test system recovery from failures."""
        try:
            # Create agents that will experience failures
            agents = []
            for i in range(5):
                config = AgentConfig(
                    name=f"Recovery Agent {i}",
                    description=f"Agent {i} for recovery testing",
                    model_name="mock-model"
                )
                
                # Some agents will have failing LLMs
                if i % 2 == 0:
                    from .test_utils import MockScenario, MockResponseType
                    error_scenario = MockScenario(MockResponseType.ERROR, error_message="Simulated failure")
                    mock_llm = MockLLMProvider(scenario=error_scenario)
                else:
                    mock_llm = MockLLMProvider()
                
                agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                await agent.initialize()
                agents.append(agent)
            
            # Execute operations expecting some failures
            tasks = [agent.ainvoke(f"Recovery test {i}") for i, agent in enumerate(agents)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes and failures
            successes = sum(1 for r in results if not isinstance(r, Exception))
            failures = len(results) - successes
            
            # Test that system continues to function despite failures
            recovery_success = successes > 0 and failures > 0
            
            return {
                "success": recovery_success,
                "successful_operations": successes,
                "failed_operations": failures,
                "total_operations": len(results),
                "failure_rate": failures / len(results)
            }
            
        except Exception as e:
            logger.error(f"Error in failure recovery test: {e}")
            return {"success": False, "error": str(e)}


class TestScalabilityValidation(PerformanceTest):
    """Test system scalability characteristics."""
    
    def __init__(self):
        super().__init__("Scalability Validation", TestSeverity.MEDIUM)
    
    async def execute_test(self) -> TestResult:
        """Test system scalability."""
        try:
            # Test scaling agent count
            scaling_test = await self._test_agent_scaling()
            
            # Test throughput scaling
            throughput_test = await self._test_throughput_scaling()
            
            # Test resource efficiency
            efficiency_test = await self._test_resource_efficiency()
            
            evidence = {
                "agent_scaling": scaling_test,
                "throughput_scaling": throughput_test,
                "resource_efficiency": efficiency_test
            }
            
            success = (scaling_test["success"] and throughput_test["success"] and 
                      efficiency_test["success"])
            
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
    
    async def _test_agent_scaling(self) -> Dict[str, Any]:
        """Test scaling with increasing number of agents."""
        try:
            scaling_results = []
            
            # Test with different agent counts
            for agent_count in [1, 5, 10, 20]:
                start_time = time.time()
                
                # Create agents
                agents = []
                for i in range(agent_count):
                    config = AgentConfig(
                        name=f"Scale Agent {i}",
                        description=f"Agent {i} for scaling test",
                        model_name="mock-model"
                    )
                    mock_llm = MockLLMProvider()
                    agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                    await agent.initialize()
                    agents.append(agent)
                
                # Execute operations
                tasks = [agent.ainvoke(f"Scale test {i}") for i, agent in enumerate(agents)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.time()
                duration = end_time - start_time
                
                successful_ops = sum(1 for r in results if not isinstance(r, Exception))
                
                scaling_results.append({
                    "agent_count": agent_count,
                    "duration": duration,
                    "successful_operations": successful_ops,
                    "throughput": successful_ops / duration if duration > 0 else 0
                })
            
            # Check if throughput scales reasonably
            throughputs = [r["throughput"] for r in scaling_results]
            scaling_efficiency = throughputs[-1] / throughputs[0] if throughputs[0] > 0 else 0
            
            return {
                "success": scaling_efficiency > 0.5,  # Should maintain at least 50% efficiency
                "scaling_results": scaling_results,
                "scaling_efficiency": scaling_efficiency
            }
            
        except Exception as e:
            logger.error(f"Error in agent scaling test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_throughput_scaling(self) -> Dict[str, Any]:
        """Test throughput scaling characteristics."""
        try:
            # Create a fixed number of agents
            agents = []
            for i in range(10):
                config = AgentConfig(
                    name=f"Throughput Agent {i}",
                    description=f"Agent {i} for throughput testing",
                    model_name="mock-model"
                )
                mock_llm = MockLLMProvider()
                agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                await agent.initialize()
                agents.append(agent)
            
            # Test with increasing operation counts
            throughput_results = []
            
            for op_count in [10, 50, 100]:
                start_time = time.time()
                
                # Create operations
                tasks = []
                for i in range(op_count):
                    agent = agents[i % len(agents)]
                    tasks.append(agent.ainvoke(f"Throughput test {i}"))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.time()
                duration = end_time - start_time
                
                successful_ops = sum(1 for r in results if not isinstance(r, Exception))
                
                throughput_results.append({
                    "operation_count": op_count,
                    "duration": duration,
                    "successful_operations": successful_ops,
                    "throughput": successful_ops / duration if duration > 0 else 0
                })
            
            # Check throughput consistency
            throughputs = [r["throughput"] for r in throughput_results]
            throughput_variance = max(throughputs) - min(throughputs)
            
            return {
                "success": throughput_variance < max(throughputs) * 0.5,  # Variance < 50% of max
                "throughput_results": throughput_results,
                "throughput_variance": throughput_variance
            }
            
        except Exception as e:
            logger.error(f"Error in throughput scaling test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_resource_efficiency(self) -> Dict[str, Any]:
        """Test resource efficiency under different loads."""
        try:
            efficiency_results = []
            
            for load_level in ["light", "medium", "heavy"]:
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Adjust load based on level
                if load_level == "light":
                    agent_count, op_count = 5, 10
                elif load_level == "medium":
                    agent_count, op_count = 10, 50
                else:  # heavy
                    agent_count, op_count = 15, 100
                
                # Create agents and execute operations
                agents = []
                for i in range(agent_count):
                    config = AgentConfig(
                        name=f"Efficiency Agent {i}",
                        description=f"Agent {i} for efficiency testing",
                        model_name="mock-model"
                    )
                    mock_llm = MockLLMProvider()
                    agent = LangGraphAgent(config=config, llm=mock_llm, tools=[])
                    await agent.initialize()
                    agents.append(agent)
                
                start_time = time.time()
                
                tasks = []
                for i in range(op_count):
                    agent = agents[i % len(agents)]
                    tasks.append(agent.ainvoke(f"Efficiency test {i}"))
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.time()
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                efficiency_results.append({
                    "load_level": load_level,
                    "agent_count": agent_count,
                    "operation_count": op_count,
                    "duration": end_time - start_time,
                    "memory_usage": final_memory - initial_memory
                })
                
                # Cleanup
                del agents
                gc.collect()
            
            # Check resource efficiency
            memory_per_op = [r["memory_usage"] / r["operation_count"] for r in efficiency_results]
            efficiency_consistent = max(memory_per_op) / min(memory_per_op) < 3.0  # Less than 3x difference
            
            return {
                "success": efficiency_consistent,
                "efficiency_results": efficiency_results,
                "memory_per_operation": memory_per_op
            }
            
        except Exception as e:
            logger.error(f"Error in resource efficiency test: {e}")
            return {"success": False, "error": str(e)}


# Test suite for performance and stress testing
class PerformanceStressTestSuite:
    """Comprehensive test suite for performance and stress testing."""
    
    def __init__(self):
        self.tests = [
            TestConcurrentAgentOperations(),
            TestAgentStressTesting(),
            TestScalabilityValidation()
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance and stress tests."""
        logger.info("Starting performance and stress test suite")
        
        results = []
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = await test.run()
            results.append(result)
        
        # Generate summary
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        
        summary = {
            "suite_name": "Performance and Stress Test Suite",
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "results": [result.__dict__ for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Performance and stress test suite completed",
            total=total,
            passed=passed,
            failed=total-passed,
            success_rate=summary["success_rate"]
        )
        
        return summary


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
async def test_concurrent_agent_operations():
    """Pytest wrapper for concurrent agent operations test."""
    test = TestConcurrentAgentOperations()
    result = await test.run()
    assert result.passed, f"Concurrent agent operations failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.stress
@pytest.mark.slow
async def test_agent_stress_testing():
    """Pytest wrapper for agent stress testing."""
    test = TestAgentStressTesting()
    result = await test.run()
    assert result.passed, f"Agent stress testing failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
async def test_scalability_validation():
    """Pytest wrapper for scalability validation test."""
    test = TestScalabilityValidation()
    result = await test.run()
    assert result.passed, f"Scalability validation failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.stress
@pytest.mark.slow
async def test_complete_performance_stress_suite():
    """Run the complete performance and stress test suite."""
    suite = PerformanceStressTestSuite()
    summary = await suite.run_all_tests()
    
    assert summary["success_rate"] >= 0.7, f"Performance stress suite success rate too low: {summary['success_rate']}"
    assert summary["passed"] >= 2, f"Not enough tests passed: {summary['passed']}/{summary['total_tests']}"
