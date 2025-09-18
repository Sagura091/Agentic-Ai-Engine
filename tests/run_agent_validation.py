#!/usr/bin/env python3
"""
Agent Validation Test Runner.

This script runs comprehensive agent validation tests to verify that the
agent orchestration system is working correctly and agents exhibit true
agentic behavior.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from agent_validation_suite import AgentValidationSuite, AgentTestType
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class AgentValidationRunner:
    """Main runner for agent validation tests."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the validation runner.
        
        Args:
            base_url: Base URL of the agent orchestration API
        """
        self.base_url = base_url
        self.validation_suite = AgentValidationSuite(base_url)
        self.results_dir = "validation_results"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def run_basic_validation(self) -> Dict[str, Any]:
        """
        Run basic validation tests to verify system functionality.
        
        Returns:
            Basic validation results
        """
        print("🔍 BASIC AGENT VALIDATION")
        print("=" * 50)
        print("Testing basic agent functionality and system health...")
        print()
        
        try:
            # Test basic functionality only
            results = await self.validation_suite.run_comprehensive_validation(
                agent_ids=None,  # Create new agents
                test_types=[AgentTestType.BASIC_FUNCTIONALITY]
            )
            
            self._print_basic_results(results)
            self._save_results(results, "basic_validation")
            
            return results
            
        except Exception as e:
            logger.error("Basic validation failed", error=str(e))
            print(f"❌ Basic validation failed: {e}")
            return {"error": str(e), "success": False}
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation tests across all agent capabilities.
        
        Returns:
            Comprehensive validation results
        """
        print("🧠 COMPREHENSIVE AGENT VALIDATION")
        print("=" * 50)
        print("Testing all agent capabilities including autonomous behavior...")
        print()
        
        try:
            # Run all test types
            results = await self.validation_suite.run_comprehensive_validation(
                agent_ids=None,  # Create new agents
                test_types=None   # Run all test types
            )
            
            self._print_comprehensive_results(results)
            self._save_results(results, "comprehensive_validation")
            
            return results
            
        except Exception as e:
            logger.error("Comprehensive validation failed", error=str(e))
            print(f"❌ Comprehensive validation failed: {e}")
            return {"error": str(e), "success": False}
    
    async def run_real_time_monitoring(self, agent_id: str = None, duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Run real-time monitoring of agent behavior.
        
        Args:
            agent_id: Specific agent to monitor. If None, creates a new agent.
            duration_minutes: Duration to monitor in minutes
            
        Returns:
            Real-time monitoring results
        """
        print(f"📊 REAL-TIME AGENT MONITORING ({duration_minutes} minutes)")
        print("=" * 50)
        print("Monitoring agent behavior in real-time...")
        print()
        
        try:
            # Create agent if none specified
            if agent_id is None:
                print("Creating test agent for monitoring...")
                agent_id = await self.validation_suite.create_test_agent(agent_type="autonomous")
                print(f"✅ Created agent: {agent_id}")
            
            # Start monitoring
            print(f"🔍 Monitoring agent {agent_id} for {duration_minutes} minutes...")
            results = await self.validation_suite.monitor_agent_real_time(agent_id, duration_minutes)
            
            self._print_monitoring_results(results)
            self._save_results(results, f"monitoring_{agent_id}")
            
            return results
            
        except Exception as e:
            logger.error("Real-time monitoring failed", error=str(e))
            print(f"❌ Real-time monitoring failed: {e}")
            return {"error": str(e), "success": False}
    
    async def test_tool_creation(self) -> Dict[str, Any]:
        """
        Test dynamic tool creation capabilities.
        
        Returns:
            Tool creation test results
        """
        print("🔧 DYNAMIC TOOL CREATION TEST")
        print("=" * 50)
        print("Testing agent ability to create and use tools dynamically...")
        print()
        
        try:
            # Run tool creation tests
            results = await self.validation_suite.run_comprehensive_validation(
                agent_ids=None,
                test_types=[AgentTestType.TOOL_CREATION]
            )
            
            self._print_tool_creation_results(results)
            self._save_results(results, "tool_creation_test")
            
            return results
            
        except Exception as e:
            logger.error("Tool creation test failed", error=str(e))
            print(f"❌ Tool creation test failed: {e}")
            return {"error": str(e), "success": False}
    
    def _print_basic_results(self, results: Dict[str, Any]) -> None:
        """Print basic validation results."""
        summary = results.get("summary", {})
        overall_stats = summary.get("overall_statistics", {})
        
        print(f"📊 BASIC VALIDATION RESULTS")
        print(f"   Total Tests: {overall_stats.get('total_tests', 0)}")
        print(f"   Passed: {overall_stats.get('passed', 0)}")
        print(f"   Failed: {overall_stats.get('failed', 0)}")
        print(f"   Pass Rate: {overall_stats.get('pass_rate', 0):.1%}")
        print(f"   Average Score: {overall_stats.get('average_score', 0):.2f}")
        print()
        
        if overall_stats.get('pass_rate', 0) >= 0.8:
            print("✅ Basic functionality: EXCELLENT")
        elif overall_stats.get('pass_rate', 0) >= 0.6:
            print("⚠️  Basic functionality: GOOD")
        else:
            print("❌ Basic functionality: NEEDS IMPROVEMENT")
        print()
    
    def _print_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """Print comprehensive validation results."""
        summary = results.get("summary", {})
        overall_stats = summary.get("overall_statistics", {})
        agentic_assessment = summary.get("overall_agentic_assessment", {})
        
        print(f"📊 COMPREHENSIVE VALIDATION RESULTS")
        print(f"   Total Tests: {overall_stats.get('total_tests', 0)}")
        print(f"   Pass Rate: {overall_stats.get('pass_rate', 0):.1%}")
        print(f"   Average Score: {overall_stats.get('average_score', 0):.2f}")
        print()
        
        print(f"🤖 AGENTIC CAPABILITY ASSESSMENT")
        print(f"   Assessment: {agentic_assessment.get('assessment', 'UNKNOWN')}")
        print(f"   Confidence: {agentic_assessment.get('confidence', 0):.1%}")
        print()
        
        # Print test type breakdown
        test_type_analysis = summary.get("test_type_analysis", {})
        if test_type_analysis:
            print("📋 TEST TYPE BREAKDOWN:")
            for test_type, analysis in test_type_analysis.items():
                status = analysis.get("status", "UNKNOWN")
                score = analysis.get("average_score", 0)
                print(f"   {test_type.replace('_', ' ').title()}: {status} (Score: {score:.2f})")
        print()
        
        # Print recommendations
        recommendations = agentic_assessment.get("recommendations", [])
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        print()
    
    def _print_monitoring_results(self, results: Dict[str, Any]) -> None:
        """Print real-time monitoring results."""
        metrics = results.get("performance_metrics", {})
        
        print(f"📊 MONITORING RESULTS")
        print(f"   Total Tasks: {results.get('total_tasks', 0)}")
        print(f"   Success Rate: {metrics.get('task_completion_rate', 0):.1%}")
        print(f"   Error Rate: {metrics.get('error_rate', 0):.1%}")
        print(f"   Autonomous Decisions: {metrics.get('autonomous_decisions', 0)}")
        print(f"   Tool Usage Count: {metrics.get('tool_usage_count', 0)}")
        
        response_times = metrics.get("response_times", [])
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            print(f"   Average Response Time: {avg_response_time:.2f}s")
        print()
    
    def _print_tool_creation_results(self, results: Dict[str, Any]) -> None:
        """Print tool creation test results."""
        summary = results.get("summary", {})
        test_type_analysis = summary.get("test_type_analysis", {})
        tool_analysis = test_type_analysis.get("tool_creation", {})
        
        print(f"🔧 TOOL CREATION RESULTS")
        if tool_analysis:
            print(f"   Tests Run: {tool_analysis.get('total_tests', 0)}")
            print(f"   Pass Rate: {tool_analysis.get('pass_rate', 0):.1%}")
            print(f"   Average Score: {tool_analysis.get('average_score', 0):.2f}")
            print(f"   Status: {tool_analysis.get('status', 'UNKNOWN')}")
        print()
    
    def _save_results(self, results: Dict[str, Any], test_name: str) -> None:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/{test_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {filename}")
        except Exception as e:
            logger.error("Failed to save results", error=str(e))
            print(f"⚠️  Failed to save results: {e}")


async def main():
    """Main entry point for agent validation."""
    print("🚀 AGENT ORCHESTRATION VALIDATION SYSTEM")
    print("=" * 60)
    print("This system will test your agents to verify they exhibit")
    print("true agentic behavior and not just pseudo-autonomous responses.")
    print("=" * 60)
    print()
    
    runner = AgentValidationRunner()
    
    # Check if backend is running
    try:
        import requests
        response = requests.get(f"{runner.base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not responding correctly")
            print("   Please ensure the backend is running on http://localhost:8001")
            return
    except Exception as e:
        print("❌ Cannot connect to backend")
        print("   Please ensure the backend is running on http://localhost:8001")
        print(f"   Error: {e}")
        return
    
    print("✅ Backend connection verified")
    print()
    
    # Run validation tests
    try:
        # 1. Basic validation
        print("Step 1: Basic Validation")
        basic_results = await runner.run_basic_validation()
        
        if basic_results.get("success", True):  # Continue if no explicit failure
            print("\nStep 2: Tool Creation Test")
            tool_results = await runner.test_tool_creation()
            
            print("\nStep 3: Real-Time Monitoring")
            monitoring_results = await runner.run_real_time_monitoring(duration_minutes=2)
            
            print("\nStep 4: Comprehensive Validation")
            comprehensive_results = await runner.run_comprehensive_validation()
            
            print("\n🎉 VALIDATION COMPLETE!")
            print("Check the validation_results directory for detailed reports.")
        else:
            print("\n❌ Basic validation failed. Please check your system configuration.")
    
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        logger.error("Validation failed", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
