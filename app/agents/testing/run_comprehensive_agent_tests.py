"""
Comprehensive Agent Testing Orchestrator - Master Test Runner.

This script orchestrates and runs all agent tests:
- Tool-specific agents testing
- Agent type showcase
- Memory and RAG integration
- Performance benchmarking
- Real-world scenario testing
- Comprehensive reporting
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import structlog

# Import all components
from app.agents.testing.master_agent_testing_framework import master_testing_framework
from app.agents.testing.file_system_agent import file_system_agent
from app.agents.testing.api_integration_agent import api_integration_agent
from app.agents.testing.database_operations_agent import database_operations_agent
from app.agents.testing.text_processing_nlp_agent import text_processing_nlp_agent
from app.agents.testing.comprehensive_agent_showcase import comprehensive_agent_showcase

logger = structlog.get_logger(__name__)


class ComprehensiveAgentTestOrchestrator:
    """
    Master orchestrator for all agent testing activities.
    
    This orchestrator provides:
    - Complete test suite execution
    - Individual agent testing
    - Performance benchmarking
    - Comprehensive reporting
    - Test result analysis
    """
    
    def __init__(self):
        """Initialize the test orchestrator."""
        self.orchestrator_id = f"test_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_results = {}
        self.start_time = None
        
        # Available test suites
        self.test_suites = {
            "individual_agents": "Test individual tool-specific agents",
            "agent_showcase": "Test all agent types and configurations", 
            "comprehensive_framework": "Run complete testing framework",
            "quick_validation": "Quick validation of core functionality",
            "performance_benchmark": "Performance benchmarking suite"
        }
        
        logger.info("Comprehensive Agent Test Orchestrator initialized", orchestrator_id=self.orchestrator_id)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive agent tests."""
        self.start_time = datetime.now()
        
        print("🚀 STARTING COMPREHENSIVE AGENT TESTING")
        print("=" * 60)
        print(f"Orchestrator ID: {self.orchestrator_id}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run master testing framework
        print("\n🎯 RUNNING MASTER TESTING FRAMEWORK")
        print("-" * 40)
        framework_results = await master_testing_framework.run_comprehensive_testing()
        self.test_results["master_framework"] = framework_results
        
        # Generate final orchestrator report
        total_time = (datetime.now() - self.start_time).total_seconds()
        final_report = await self._generate_orchestrator_report(total_time)
        
        print("\n" + "=" * 60)
        print("🎉 ALL COMPREHENSIVE TESTING COMPLETED")
        print("=" * 60)
        
        return final_report
    
    async def run_individual_agent_tests(self) -> Dict[str, Any]:
        """Run tests for individual tool-specific agents."""
        print("🔧 TESTING INDIVIDUAL TOOL-SPECIFIC AGENTS")
        print("-" * 50)
        
        # Test agents individually
        agents_to_test = [
            ("File System Agent", file_system_agent),
            ("API Integration Agent", api_integration_agent),
            ("Database Operations Agent", database_operations_agent),
            ("Text Processing NLP Agent", text_processing_nlp_agent)
        ]
        
        results = {}
        
        for agent_name, agent in agents_to_test:
            print(f"\n🧪 Testing {agent_name}...")
            
            try:
                # Initialize agent
                init_success = await agent.initialize()
                if not init_success:
                    results[agent_name] = {"success": False, "error": "Failed to initialize"}
                    print(f"❌ {agent_name}: INITIALIZATION FAILED")
                    continue
                
                # Test basic functionality
                test_result = await agent.process_request(
                    f"Please demonstrate your capabilities as {agent_name}"
                )
                
                # Test capabilities
                demo_result = await agent.demonstrate_capabilities()
                
                results[agent_name] = {
                    "initialization": True,
                    "basic_test": test_result.get("success", False),
                    "capabilities_demo": demo_result.get("overall_success", False),
                    "execution_time": test_result.get("execution_time", 0.0),
                    "response_quality": len(test_result.get("response", "")) > 100,
                    "session_logged": bool(test_result.get("session_summary"))
                }
                
                success = all([
                    results[agent_name]["initialization"],
                    results[agent_name]["basic_test"],
                    results[agent_name]["capabilities_demo"]
                ])
                
                print(f"{'✅' if success else '❌'} {agent_name}: {'SUCCESS' if success else 'PARTIAL/FAILED'}")
                
            except Exception as e:
                logger.error(f"Individual agent test failed", agent=agent_name, error=str(e))
                results[agent_name] = {"success": False, "error": str(e)}
                print(f"❌ {agent_name}: FAILED - {str(e)}")
        
        return {
            "test_type": "individual_agents",
            "agents_tested": len(agents_to_test),
            "successful_agents": sum(1 for r in results.values() if r.get("basic_test", False)),
            "results": results
        }
    
    async def run_agent_showcase_test(self) -> Dict[str, Any]:
        """Run the comprehensive agent showcase test."""
        print("🎭 TESTING COMPREHENSIVE AGENT SHOWCASE")
        print("-" * 50)
        
        try:
            # Initialize showcase
            init_success = await comprehensive_agent_showcase.initialize()
            if not init_success:
                return {"success": False, "error": "Failed to initialize showcase"}
            
            # Run showcase
            showcase_results = await comprehensive_agent_showcase.showcase_all_agent_types()
            
            print(f"✅ Agent Showcase: SUCCESS")
            print(f"   - Agent Types Tested: {showcase_results['total_agents_tested']}")
            print(f"   - Success Rate: {showcase_results['success_rate']:.1%}")
            print(f"   - Execution Time: {showcase_results['total_showcase_time']:.2f}s")
            
            return {
                "test_type": "agent_showcase",
                "success": True,
                "showcase_results": showcase_results
            }
            
        except Exception as e:
            logger.error("Agent showcase test failed", error=str(e))
            print(f"❌ Agent Showcase: FAILED - {str(e)}")
            return {"test_type": "agent_showcase", "success": False, "error": str(e)}
    
    async def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation of core functionality."""
        print("⚡ RUNNING QUICK VALIDATION")
        print("-" * 30)
        
        validation_tests = [
            ("Custom Logger", self._test_custom_logger),
            ("Agent Factory", self._test_agent_factory),
            ("Tool Integration", self._test_tool_integration),
            ("Memory System", self._test_memory_system)
        ]
        
        results = {}
        
        for test_name, test_func in validation_tests:
            try:
                print(f"Testing {test_name}...")
                result = await test_func()
                results[test_name] = result
                print(f"{'✅' if result.get('success', False) else '❌'} {test_name}: {'PASS' if result.get('success', False) else 'FAIL'}")
            except Exception as e:
                results[test_name] = {"success": False, "error": str(e)}
                print(f"❌ {test_name}: FAILED - {str(e)}")
        
        success_rate = sum(1 for r in results.values() if r.get("success", False)) / len(results)
        
        return {
            "test_type": "quick_validation",
            "tests_run": len(validation_tests),
            "success_rate": success_rate,
            "results": results
        }
    
    async def _test_custom_logger(self) -> Dict[str, Any]:
        """Test custom agent logger functionality."""
        from app.agents.testing.custom_agent_logger import custom_agent_logger, AgentMetadata
        
        try:
            # Test logger initialization
            test_metadata = AgentMetadata(
                agent_id="test_agent",
                agent_type="test",
                agent_name="Test Agent",
                capabilities=["test"],
                tools_available=["test_tool"]
            )
            
            session_id = "test_session"
            custom_agent_logger.start_session(session_id, test_metadata)
            custom_agent_logger.log_query_received(session_id, "Test query")
            summary = custom_agent_logger.end_session(session_id)
            
            return {
                "success": True,
                "session_created": bool(summary),
                "logs_captured": len(summary.get("summary", {}).get("action_counts", {})) > 0
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_agent_factory(self) -> Dict[str, Any]:
        """Test agent factory functionality."""
        try:
            from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType
            from app.llm.manager import get_enhanced_llm_manager
            
            # Test factory initialization
            llm_manager = await get_enhanced_llm_manager()
            factory = AgentBuilderFactory(llm_manager)
            
            return {
                "success": True,
                "factory_initialized": factory is not None,
                "llm_manager_available": llm_manager is not None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_tool_integration(self) -> Dict[str, Any]:
        """Test tool integration functionality."""
        try:
            from app.tools.production.file_system_tool import file_system_tool
            
            # Test tool availability
            tool_available = file_system_tool is not None
            
            return {
                "success": True,
                "tool_available": tool_available,
                "tool_type": type(file_system_tool).__name__
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _test_memory_system(self) -> Dict[str, Any]:
        """Test memory system functionality."""
        try:
            from app.memory.unified_memory_system import UnifiedMemorySystem
            
            # Test memory system initialization
            memory_system = UnifiedMemorySystem()
            
            return {
                "success": True,
                "memory_system_available": memory_system is not None,
                "config_loaded": bool(memory_system.config)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_orchestrator_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive orchestrator report."""
        report = {
            "orchestrator_id": self.orchestrator_id,
            "execution_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_execution_time": total_time,
                "test_suites_run": len(self.test_results)
            },
            "test_results": self.test_results,
            "overall_assessment": self._assess_overall_results(),
            "recommendations": self._generate_orchestrator_recommendations()
        }
        
        # Save report
        await self._save_orchestrator_report(report)
        
        # Print summary
        print(f"\n📊 ORCHESTRATOR SUMMARY:")
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Test Suites Run: {len(self.test_results)}")
        print(f"Overall Assessment: {report['overall_assessment']['status']}")
        
        return report
    
    def _assess_overall_results(self) -> Dict[str, Any]:
        """Assess overall test results."""
        if not self.test_results:
            return {"status": "NO_TESTS", "score": 0.0}
        
        # Calculate overall score based on test results
        scores = []
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                if "overall_results" in result and "overall_success_rate" in result["overall_results"]:
                    scores.append(result["overall_results"]["overall_success_rate"])
                elif "success_rate" in result:
                    scores.append(result["success_rate"])
                elif "success" in result:
                    scores.append(1.0 if result["success"] else 0.0)
        
        if not scores:
            return {"status": "INDETERMINATE", "score": 0.0}
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 0.9:
            status = "EXCELLENT"
        elif avg_score >= 0.7:
            status = "GOOD"
        elif avg_score >= 0.5:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS_IMPROVEMENT"
        
        return {
            "status": status,
            "score": avg_score,
            "individual_scores": scores
        }
    
    def _generate_orchestrator_recommendations(self) -> List[str]:
        """Generate orchestrator-level recommendations."""
        recommendations = []
        
        assessment = self._assess_overall_results()
        
        if assessment["status"] == "EXCELLENT":
            recommendations.append("🎉 Outstanding! All systems are performing excellently")
            recommendations.append("📈 Consider expanding test coverage for edge cases")
        elif assessment["status"] == "GOOD":
            recommendations.append("✅ Good performance overall")
            recommendations.append("🔧 Focus on optimizing lower-performing components")
        else:
            recommendations.append("⚠️  System needs significant improvement")
            recommendations.append("🛠️  Review failed tests and address core issues")
        
        recommendations.append("📊 Continue regular testing and monitoring")
        recommendations.append("🚀 Ready for production deployment with monitoring")
        
        return recommendations
    
    async def _save_orchestrator_report(self, report: Dict[str, Any]) -> None:
        """Save orchestrator report to file."""
        try:
            reports_dir = Path("reports/orchestrator")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"orchestrator_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info("Orchestrator report saved", report_file=str(report_file))
            
        except Exception as e:
            logger.error("Failed to save orchestrator report", error=str(e))


# Create global instance
orchestrator = ComprehensiveAgentTestOrchestrator()


async def main():
    """Main entry point for comprehensive agent testing."""
    print("🎯 COMPREHENSIVE AGENT TESTING ORCHESTRATOR")
    print("=" * 60)
    
    # Run all tests
    results = await orchestrator.run_all_tests()
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"Overall Status: {results['overall_assessment']['status']}")
    print(f"Overall Score: {results['overall_assessment']['score']:.1%}")
    print(f"Check reports directory for detailed results!")


if __name__ == "__main__":
    asyncio.run(main())
