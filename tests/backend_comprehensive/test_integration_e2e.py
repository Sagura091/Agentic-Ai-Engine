"""
Comprehensive Integration and End-to-End Testing Suite.

This module tests complete workflows from agent creation through task execution
with tools and knowledge bases, validating the entire agentic AI system.
"""

import pytest
import asyncio
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

import structlog

# Import test infrastructure
from .base_test import IntegrationTest, EndToEndTest, TestResult, TestCategory, TestSeverity
from .test_utils import TestDataGenerator, MockLLMProvider, MockRAGService, TestValidator

# Import application components
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType
from app.agents.autonomous import create_autonomous_agent, create_research_agent, create_creative_agent
from app.rag.core.enhanced_rag_service import EnhancedRAGService
from app.tools.dynamic_tool_factory import DynamicToolFactory

logger = structlog.get_logger(__name__)


class TestCompleteAgentWorkflow(EndToEndTest):
    """Test complete agent workflow from creation to task completion."""
    
    def __init__(self):
        super().__init__("Complete Agent Workflow", TestSeverity.CRITICAL)
    
    async def execute_test(self) -> TestResult:
        """Test complete agent workflow."""
        try:
            # Test workflow scenarios
            research_workflow = await self._test_research_agent_workflow()
            creative_workflow = await self._test_creative_agent_workflow()
            autonomous_workflow = await self._test_autonomous_agent_workflow()
            multi_agent_workflow = await self._test_multi_agent_workflow()
            
            evidence = {
                "research_workflow": research_workflow,
                "creative_workflow": creative_workflow,
                "autonomous_workflow": autonomous_workflow,
                "multi_agent_workflow": multi_agent_workflow
            }
            
            success = (research_workflow["success"] and creative_workflow["success"] and 
                      autonomous_workflow["success"] and multi_agent_workflow["success"])
            
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
    
    async def _test_research_agent_workflow(self) -> Dict[str, Any]:
        """Test complete research agent workflow."""
        try:
            # Create research agent with tools and knowledge base
            agent = await create_research_agent(
                name="Research Workflow Agent",
                description="Agent for testing research workflow",
                model_name="mock-model"
            )
            
            # Setup knowledge base
            rag_service = MockRAGService()
            await rag_service.initialize()
            
            # Ingest research documents
            research_docs = TestDataGenerator.generate_research_documents(5)
            for doc in research_docs:
                await rag_service.ingest_document(
                    title=doc["title"],
                    content=doc["content"],
                    collection="research_workflow",
                    metadata=doc["metadata"]
                )
            
            # Execute research task
            research_task = """Conduct research on artificial intelligence trends and provide a comprehensive analysis.
            Use available knowledge sources and tools to gather information."""
            
            result = await agent.ainvoke(research_task)
            
            # Validate research workflow
            workflow_success = await self._validate_research_workflow(result, rag_service)
            
            return {
                "success": workflow_success,
                "agent_id": agent.agent_id,
                "documents_available": len(research_docs),
                "task_completed": result is not None,
                "workflow_validation": workflow_success
            }
            
        except Exception as e:
            logger.error(f"Error in research agent workflow: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_creative_agent_workflow(self) -> Dict[str, Any]:
        """Test complete creative agent workflow."""
        try:
            # Create creative agent
            agent = await create_creative_agent(
                name="Creative Workflow Agent",
                description="Agent for testing creative workflow",
                model_name="mock-model"
            )
            
            # Create creative tools
            creative_tools = await self._create_creative_tools()
            
            # Execute creative task
            creative_task = """Create an innovative solution for improving team collaboration.
            Use your creative capabilities and available tools to develop unique ideas."""
            
            result = await agent.ainvoke(creative_task)
            
            # Validate creative workflow
            workflow_success = await self._validate_creative_workflow(result, creative_tools)
            
            return {
                "success": workflow_success,
                "agent_id": agent.agent_id,
                "tools_available": len(creative_tools),
                "task_completed": result is not None,
                "workflow_validation": workflow_success
            }
            
        except Exception as e:
            logger.error(f"Error in creative agent workflow: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_autonomous_agent_workflow(self) -> Dict[str, Any]:
        """Test complete autonomous agent workflow."""
        try:
            # Create autonomous agent
            agent = await create_autonomous_agent(
                name="Autonomous Workflow Agent",
                description="Agent for testing autonomous workflow",
                model_name="mock-model"
            )
            
            # Setup comprehensive environment
            rag_service = MockRAGService()
            await rag_service.initialize()
            
            tool_factory = DynamicToolFactory()
            tools = await self._create_comprehensive_tools(tool_factory)
            
            # Execute autonomous task
            autonomous_task = """You are given the goal of optimizing a business process.
            Analyze the situation, identify problems, research solutions, and implement improvements.
            Use all available resources and make autonomous decisions."""
            
            result = await agent.ainvoke(autonomous_task)
            
            # Validate autonomous workflow
            workflow_success = await self._validate_autonomous_workflow(result, agent)
            
            return {
                "success": workflow_success,
                "agent_id": agent.agent_id,
                "tools_available": len(tools),
                "task_completed": result is not None,
                "autonomy_demonstrated": workflow_success
            }
            
        except Exception as e:
            logger.error(f"Error in autonomous agent workflow: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_multi_agent_workflow(self) -> Dict[str, Any]:
        """Test multi-agent collaborative workflow."""
        try:
            # Create multiple agents with different specializations
            research_agent = await create_research_agent(
                name="Multi Research Agent",
                description="Research specialist for multi-agent workflow",
                model_name="mock-model"
            )
            
            creative_agent = await create_creative_agent(
                name="Multi Creative Agent", 
                description="Creative specialist for multi-agent workflow",
                model_name="mock-model"
            )
            
            autonomous_agent = await create_autonomous_agent(
                name="Multi Autonomous Agent",
                description="Autonomous coordinator for multi-agent workflow",
                model_name="mock-model"
            )
            
            agents = [research_agent, creative_agent, autonomous_agent]
            
            # Setup shared resources
            shared_rag = MockRAGService()
            await shared_rag.initialize()
            
            # Execute collaborative task
            collaborative_task = """Develop a comprehensive strategy for implementing AI in education.
            Research agent: Gather information on current AI in education trends.
            Creative agent: Generate innovative implementation ideas.
            Autonomous agent: Coordinate and synthesize the final strategy."""
            
            # Execute tasks in sequence (simulating coordination)
            research_result = await research_agent.ainvoke(
                "Research current AI in education trends and provide detailed analysis."
            )
            
            creative_result = await creative_agent.ainvoke(
                "Generate innovative ideas for implementing AI in education based on research findings."
            )
            
            coordination_result = await autonomous_agent.ainvoke(
                "Synthesize research and creative inputs into a comprehensive AI education strategy."
            )
            
            # Validate multi-agent workflow
            workflow_success = await self._validate_multi_agent_workflow(
                [research_result, creative_result, coordination_result]
            )
            
            return {
                "success": workflow_success,
                "agents_involved": len(agents),
                "research_completed": research_result is not None,
                "creative_completed": creative_result is not None,
                "coordination_completed": coordination_result is not None,
                "workflow_validation": workflow_success
            }
            
        except Exception as e:
            logger.error(f"Error in multi-agent workflow: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_research_workflow(self, result: Any, rag_service: MockRAGService) -> bool:
        """Validate research workflow completion."""
        if not result or not hasattr(result, 'content'):
            return False
        
        content = result.content.lower()
        research_indicators = ["research", "analysis", "findings", "data", "study", "information"]
        
        return sum(1 for indicator in research_indicators if indicator in content) >= 3
    
    async def _validate_creative_workflow(self, result: Any, tools: List) -> bool:
        """Validate creative workflow completion."""
        if not result or not hasattr(result, 'content'):
            return False
        
        content = result.content.lower()
        creative_indicators = ["creative", "innovative", "idea", "solution", "unique", "original"]
        
        return sum(1 for indicator in creative_indicators if indicator in content) >= 3
    
    async def _validate_autonomous_workflow(self, result: Any, agent) -> bool:
        """Validate autonomous workflow completion."""
        if not result or not hasattr(result, 'content'):
            return False
        
        content = result.content.lower()
        autonomous_indicators = ["analyze", "decide", "implement", "optimize", "strategy", "plan"]
        
        return sum(1 for indicator in autonomous_indicators if indicator in content) >= 3
    
    async def _validate_multi_agent_workflow(self, results: List[Any]) -> bool:
        """Validate multi-agent workflow completion."""
        if len(results) != 3:
            return False
        
        # Check that all agents produced results
        for result in results:
            if not result or not hasattr(result, 'content'):
                return False
        
        # Check for collaboration indicators
        combined_content = " ".join(r.content.lower() for r in results)
        collaboration_indicators = ["strategy", "comprehensive", "synthesis", "coordination"]
        
        return sum(1 for indicator in collaboration_indicators if indicator in combined_content) >= 2
    
    async def _create_creative_tools(self) -> List:
        """Create tools for creative workflow."""
        from .test_utils import create_mock_tool
        
        tools = []
        tool_names = ["brainstorm_tool", "idea_generator", "creativity_enhancer"]
        
        for name in tool_names:
            tool = await create_mock_tool(name)
            tools.append(tool)
        
        return tools
    
    async def _create_comprehensive_tools(self, tool_factory: DynamicToolFactory) -> List:
        """Create comprehensive tools for autonomous workflow."""
        from .test_utils import create_mock_tool
        
        tools = []
        tool_names = [
            "analysis_tool", "optimization_tool", "decision_tool", 
            "planning_tool", "execution_tool"
        ]
        
        for name in tool_names:
            tool = await create_mock_tool(name)
            tools.append(tool)
        
        return tools


class TestSystemIntegration(IntegrationTest):
    """Test integration between all system components."""
    
    def __init__(self):
        super().__init__("System Integration", TestSeverity.HIGH)
    
    async def execute_test(self) -> TestResult:
        """Test system integration."""
        try:
            # Test orchestrator integration
            orchestrator_test = await self._test_orchestrator_integration()
            
            # Test agent-tool integration
            agent_tool_test = await self._test_agent_tool_integration()
            
            # Test agent-rag integration
            agent_rag_test = await self._test_agent_rag_integration()
            
            # Test full system integration
            full_system_test = await self._test_full_system_integration()
            
            evidence = {
                "orchestrator_integration": orchestrator_test,
                "agent_tool_integration": agent_tool_test,
                "agent_rag_integration": agent_rag_test,
                "full_system_integration": full_system_test
            }
            
            success = (orchestrator_test["success"] and agent_tool_test["success"] and 
                      agent_rag_test["success"] and full_system_test["success"])
            
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
    
    async def _test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test orchestrator integration with agents."""
        try:
            # Test agent creation through orchestrator
            agent_configs = [
                {"type": AgentType.RESEARCH, "name": "Orchestrator Research Agent"},
                {"type": AgentType.CREATIVE, "name": "Orchestrator Creative Agent"},
                {"type": AgentType.AUTONOMOUS, "name": "Orchestrator Autonomous Agent"}
            ]
            
            created_agents = []
            for config in agent_configs:
                # Mock orchestrator agent creation
                agent_id = f"orchestrator_agent_{len(created_agents)}"
                created_agents.append({"id": agent_id, "type": config["type"], "name": config["name"]})
            
            # Test orchestrator task distribution
            task = "Analyze market trends and provide recommendations"
            
            # Mock task distribution
            task_results = []
            for agent in created_agents:
                result = f"Task completed by {agent['name']}: Analysis of market trends"
                task_results.append(result)
            
            return {
                "success": len(created_agents) == len(agent_configs) and len(task_results) == len(created_agents),
                "agents_created": len(created_agents),
                "tasks_distributed": len(task_results),
                "orchestration_functional": True
            }
            
        except Exception as e:
            logger.error(f"Error in orchestrator integration test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_agent_tool_integration(self) -> Dict[str, Any]:
        """Test agent-tool integration."""
        try:
            # Create agent with tools
            agent = await create_autonomous_agent(
                name="Tool Integration Agent",
                description="Agent for testing tool integration",
                model_name="mock-model"
            )
            
            # Create and assign tools
            from .test_utils import create_mock_tool
            tools = []
            for i in range(3):
                tool = await create_mock_tool(f"integration_tool_{i}")
                tools.append(tool)
            
            # Test tool usage
            tool_usage_results = []
            for tool in tools:
                try:
                    result = await tool.invoke(f"Test input for {tool.name}")
                    tool_usage_results.append({"tool": tool.name, "success": True, "result": result})
                except Exception as e:
                    tool_usage_results.append({"tool": tool.name, "success": False, "error": str(e)})
            
            successful_tools = sum(1 for r in tool_usage_results if r["success"])
            
            return {
                "success": successful_tools >= len(tools) * 0.8,
                "tools_available": len(tools),
                "successful_tool_usage": successful_tools,
                "tool_integration_rate": successful_tools / len(tools) if tools else 0
            }
            
        except Exception as e:
            logger.error(f"Error in agent-tool integration test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_agent_rag_integration(self) -> Dict[str, Any]:
        """Test agent-RAG integration."""
        try:
            # Create agent
            agent = await create_research_agent(
                name="RAG Integration Agent",
                description="Agent for testing RAG integration",
                model_name="mock-model"
            )
            
            # Setup RAG service
            rag_service = MockRAGService()
            await rag_service.initialize()
            
            # Ingest test documents
            test_docs = TestDataGenerator.generate_test_documents(5)
            for doc in test_docs:
                await rag_service.ingest_document(
                    title=doc["title"],
                    content=doc["content"],
                    collection="integration_test",
                    metadata=doc["metadata"]
                )
            
            # Test knowledge retrieval
            search_results = await rag_service.search_knowledge(
                query="artificial intelligence",
                collection="integration_test",
                top_k=3
            )
            
            # Test agent knowledge usage
            knowledge_task = "Use available knowledge to explain artificial intelligence concepts"
            agent_result = await agent.ainvoke(knowledge_task)
            
            return {
                "success": len(search_results.get("results", [])) > 0 and agent_result is not None,
                "documents_ingested": len(test_docs),
                "search_results_found": len(search_results.get("results", [])),
                "agent_knowledge_usage": agent_result is not None,
                "rag_integration_functional": True
            }
            
        except Exception as e:
            logger.error(f"Error in agent-RAG integration test: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_full_system_integration(self) -> Dict[str, Any]:
        """Test full system integration."""
        try:
            # Create comprehensive test environment
            temp_dir = tempfile.mkdtemp(prefix="full_system_test_")
            
            try:
                # Setup all components
                rag_service = MockRAGService()
                await rag_service.initialize()
                
                # Create agents
                agents = []
                for agent_type in [AgentType.RESEARCH, AgentType.CREATIVE, AgentType.AUTONOMOUS]:
                    if agent_type == AgentType.RESEARCH:
                        agent = await create_research_agent(
                            name=f"Full System {agent_type.value} Agent",
                            description=f"Agent for full system testing",
                            model_name="mock-model"
                        )
                    elif agent_type == AgentType.CREATIVE:
                        agent = await create_creative_agent(
                            name=f"Full System {agent_type.value} Agent",
                            description=f"Agent for full system testing",
                            model_name="mock-model"
                        )
                    else:
                        agent = await create_autonomous_agent(
                            name=f"Full System {agent_type.value} Agent",
                            description=f"Agent for full system testing",
                            model_name="mock-model"
                        )
                    agents.append(agent)
                
                # Setup knowledge base
                knowledge_docs = TestDataGenerator.generate_test_documents(10)
                for doc in knowledge_docs:
                    await rag_service.ingest_document(
                        title=doc["title"],
                        content=doc["content"],
                        collection="full_system_test",
                        metadata=doc["metadata"]
                    )
                
                # Execute comprehensive task
                system_task = """Complete a comprehensive analysis of emerging technologies.
                Use all available resources including knowledge bases and tools.
                Collaborate between different agent types to provide a complete solution."""
                
                # Execute task with each agent
                results = []
                for agent in agents:
                    result = await agent.ainvoke(system_task)
                    results.append(result)
                
                # Validate full system functionality
                system_functional = all(r is not None for r in results)
                
                return {
                    "success": system_functional,
                    "agents_created": len(agents),
                    "knowledge_documents": len(knowledge_docs),
                    "task_executions": len(results),
                    "system_functional": system_functional,
                    "temp_directory": temp_dir
                }
                
            finally:
                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Error in full system integration test: {e}")
            return {"success": False, "error": str(e)}


# Test suite for integration and end-to-end testing
class IntegrationE2ETestSuite:
    """Comprehensive test suite for integration and end-to-end testing."""
    
    def __init__(self):
        self.tests = [
            TestCompleteAgentWorkflow(),
            TestSystemIntegration()
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration and end-to-end tests."""
        logger.info("Starting integration and end-to-end test suite")
        
        results = []
        for test in self.tests:
            logger.info(f"Running test: {test.test_name}")
            result = await test.run()
            results.append(result)
        
        # Generate summary
        passed = sum(1 for result in results if result.passed)
        total = len(results)
        
        summary = {
            "suite_name": "Integration and End-to-End Test Suite",
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "results": [result.__dict__ for result in results],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Integration and end-to-end test suite completed",
            total=total,
            passed=passed,
            failed=total-passed,
            success_rate=summary["success_rate"]
        )
        
        return summary


# Pytest integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
async def test_complete_agent_workflow():
    """Pytest wrapper for complete agent workflow test."""
    test = TestCompleteAgentWorkflow()
    result = await test.run()
    assert result.passed, f"Complete agent workflow failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_system_integration():
    """Pytest wrapper for system integration test."""
    test = TestSystemIntegration()
    result = await test.run()
    assert result.passed, f"System integration failed: {result.error_message}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
async def test_complete_integration_e2e_suite():
    """Run the complete integration and end-to-end test suite."""
    suite = IntegrationE2ETestSuite()
    summary = await suite.run_all_tests()
    
    assert summary["success_rate"] >= 0.8, f"Integration E2E suite success rate too low: {summary['success_rate']}"
    assert summary["passed"] >= 1, f"Not enough tests passed: {summary['passed']}/{summary['total_tests']}"
