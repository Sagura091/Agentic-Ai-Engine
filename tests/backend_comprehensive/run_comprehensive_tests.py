"""
Comprehensive Backend Testing Suite Runner.

This is the main entry point for running all comprehensive backend tests
to validate the agentic AI system functionality.
"""

import asyncio
import argparse
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

import structlog

# Import all test suites
from .test_agent_creation import AgentCreationTestSuite
from .test_llm_integration import LLMIntegrationTestSuite
from .test_tool_integration import ToolIntegrationTestSuite
from .test_rag_integration import RAGIntegrationTestSuite
from .test_agent_behavior import AgentBehaviorTestSuite
from .test_performance_stress import PerformanceStressTestSuite
from .test_integration_e2e import IntegrationE2ETestSuite

logger = structlog.get_logger(__name__)


class ComprehensiveTestRunner:
    """Main test runner for comprehensive backend testing."""
    
    def __init__(self):
        self.test_suites = {
            "agent_creation": AgentCreationTestSuite(),
            "llm_integration": LLMIntegrationTestSuite(),
            "tool_integration": ToolIntegrationTestSuite(),
            "rag_integration": RAGIntegrationTestSuite(),
            "agent_behavior": AgentBehaviorTestSuite(),
            "performance_stress": PerformanceStressTestSuite(),
            "integration_e2e": IntegrationE2ETestSuite()
        }
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    async def run_all_tests(self, include_slow: bool = False, include_stress: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        logger.info("Starting comprehensive backend testing suite")
        self.start_time = time.time()
        
        # Determine which suites to run
        suites_to_run = self._get_suites_to_run(include_slow, include_stress)
        
        # Run test suites
        for suite_name in suites_to_run:
            logger.info(f"Running test suite: {suite_name}")
            
            try:
                suite = self.test_suites[suite_name]
                result = await suite.run_all_tests()
                self.results[suite_name] = result
                
                logger.info(
                    f"Test suite {suite_name} completed",
                    passed=result["passed"],
                    failed=result["failed"],
                    success_rate=result["success_rate"]
                )
                
            except Exception as e:
                logger.error(f"Error running test suite {suite_name}: {e}")
                self.results[suite_name] = {
                    "suite_name": suite_name,
                    "error": str(e),
                    "passed": 0,
                    "failed": 1,
                    "success_rate": 0.0
                }
        
        self.end_time = time.time()
        
        # Generate comprehensive summary
        summary = self._generate_comprehensive_summary()
        
        logger.info(
            "Comprehensive backend testing completed",
            total_duration=summary["total_duration"],
            overall_success_rate=summary["overall_success_rate"],
            critical_tests_passed=summary["critical_tests_passed"]
        )
        
        return summary
    
    async def run_specific_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        logger.info(f"Running specific test suite: {suite_name}")
        
        suite = self.test_suites[suite_name]
        result = await suite.run_all_tests()
        
        return result
    
    async def run_critical_tests_only(self) -> Dict[str, Any]:
        """Run only critical tests for quick validation."""
        logger.info("Running critical tests only")
        
        critical_suites = ["agent_creation", "llm_integration", "agent_behavior"]
        
        for suite_name in critical_suites:
            logger.info(f"Running critical test suite: {suite_name}")
            
            try:
                suite = self.test_suites[suite_name]
                result = await suite.run_all_tests()
                self.results[suite_name] = result
                
            except Exception as e:
                logger.error(f"Error running critical test suite {suite_name}: {e}")
                self.results[suite_name] = {
                    "suite_name": suite_name,
                    "error": str(e),
                    "passed": 0,
                    "failed": 1,
                    "success_rate": 0.0
                }
        
        return self._generate_comprehensive_summary()
    
    def _get_suites_to_run(self, include_slow: bool, include_stress: bool) -> List[str]:
        """Determine which test suites to run based on options."""
        base_suites = ["agent_creation", "llm_integration", "tool_integration", "rag_integration", "agent_behavior"]
        
        if include_slow:
            base_suites.append("integration_e2e")
        
        if include_stress:
            base_suites.append("performance_stress")
        
        return base_suites
    
    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_tests = sum(result.get("total_tests", 0) for result in self.results.values())
        total_passed = sum(result.get("passed", 0) for result in self.results.values())
        total_failed = sum(result.get("failed", 0) for result in self.results.values())
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Categorize results by severity
        critical_tests = self._count_tests_by_severity("CRITICAL")
        high_tests = self._count_tests_by_severity("HIGH")
        medium_tests = self._count_tests_by_severity("MEDIUM")
        
        # Calculate duration
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # Determine overall status
        overall_status = self._determine_overall_status(overall_success_rate, critical_tests)
        
        summary = {
            "test_run_id": f"comprehensive_test_{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "total_duration": duration,
            "overall_status": overall_status,
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "suites_run": len(self.results),
            "suite_results": self.results,
            "test_breakdown": {
                "critical": critical_tests,
                "high": high_tests,
                "medium": medium_tests
            },
            "critical_tests_passed": critical_tests["passed"] / critical_tests["total"] if critical_tests["total"] > 0 else 0,
            "recommendations": self._generate_recommendations(overall_success_rate, critical_tests)
        }
        
        return summary
    
    def _count_tests_by_severity(self, severity: str) -> Dict[str, int]:
        """Count tests by severity level."""
        # This is a simplified implementation
        # In a real implementation, you would parse individual test results
        total = 0
        passed = 0
        
        for suite_name, result in self.results.items():
            if severity == "CRITICAL":
                # Agent creation, LLM integration, and behavior are critical
                if suite_name in ["agent_creation", "llm_integration", "agent_behavior"]:
                    total += result.get("total_tests", 0)
                    passed += result.get("passed", 0)
            elif severity == "HIGH":
                # Tool integration, RAG integration, and integration tests are high
                if suite_name in ["tool_integration", "rag_integration", "integration_e2e"]:
                    total += result.get("total_tests", 0)
                    passed += result.get("passed", 0)
            elif severity == "MEDIUM":
                # Performance and stress tests are medium
                if suite_name in ["performance_stress"]:
                    total += result.get("total_tests", 0)
                    passed += result.get("passed", 0)
        
        return {"total": total, "passed": passed, "failed": total - passed}
    
    def _determine_overall_status(self, success_rate: float, critical_tests: Dict[str, int]) -> str:
        """Determine overall test status."""
        critical_success_rate = critical_tests["passed"] / critical_tests["total"] if critical_tests["total"] > 0 else 0
        
        if critical_success_rate >= 0.9 and success_rate >= 0.8:
            return "EXCELLENT"
        elif critical_success_rate >= 0.8 and success_rate >= 0.7:
            return "GOOD"
        elif critical_success_rate >= 0.7 and success_rate >= 0.6:
            return "ACCEPTABLE"
        elif critical_success_rate >= 0.5:
            return "NEEDS_IMPROVEMENT"
        else:
            return "CRITICAL_ISSUES"
    
    def _generate_recommendations(self, success_rate: float, critical_tests: Dict[str, int]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        critical_success_rate = critical_tests["passed"] / critical_tests["total"] if critical_tests["total"] > 0 else 0
        
        if critical_success_rate < 0.8:
            recommendations.append("URGENT: Critical agent functionality tests are failing. Review agent creation, LLM integration, and behavior validation.")
        
        if success_rate < 0.7:
            recommendations.append("Overall test success rate is below acceptable threshold. Comprehensive system review needed.")
        
        # Check specific suite failures
        for suite_name, result in self.results.items():
            suite_success_rate = result.get("success_rate", 0)
            if suite_success_rate < 0.6:
                recommendations.append(f"Test suite '{suite_name}' has low success rate ({suite_success_rate:.2%}). Investigate specific failures.")
        
        if not recommendations:
            recommendations.append("All tests are performing well. System appears to be functioning correctly.")
        
        return recommendations
    
    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to file."""
        if not output_file:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"comprehensive_test_results_{timestamp}.json"
        
        summary = self._generate_comprehensive_summary()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {output_path}")
        return str(output_path)


async def main():
    """Main entry point for running comprehensive tests."""
    parser = argparse.ArgumentParser(description="Run comprehensive backend tests for agentic AI system")
    parser.add_argument("--suite", type=str, help="Run specific test suite")
    parser.add_argument("--critical-only", action="store_true", help="Run only critical tests")
    parser.add_argument("--include-slow", action="store_true", help="Include slow tests (integration/e2e)")
    parser.add_argument("--include-stress", action="store_true", help="Include stress tests")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        )
    
    runner = ComprehensiveTestRunner()
    
    try:
        if args.suite:
            # Run specific suite
            result = await runner.run_specific_suite(args.suite)
            print(f"\nTest Suite '{args.suite}' Results:")
            print(f"Passed: {result['passed']}/{result['total_tests']}")
            print(f"Success Rate: {result['success_rate']:.2%}")
            
        elif args.critical_only:
            # Run critical tests only
            summary = await runner.run_critical_tests_only()
            print(f"\nCritical Tests Results:")
            print(f"Overall Status: {summary['overall_status']}")
            print(f"Success Rate: {summary['overall_success_rate']:.2%}")
            
        else:
            # Run all tests
            summary = await runner.run_all_tests(
                include_slow=args.include_slow,
                include_stress=args.include_stress
            )
            
            print(f"\nComprehensive Test Results:")
            print(f"Overall Status: {summary['overall_status']}")
            print(f"Total Tests: {summary['total_passed']}/{summary['total_tests']}")
            print(f"Success Rate: {summary['overall_success_rate']:.2%}")
            print(f"Duration: {summary['total_duration']:.2f} seconds")
            
            print(f"\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"- {rec}")
        
        # Save results if requested
        if args.output:
            output_file = runner.save_results(args.output)
            print(f"\nResults saved to: {output_file}")
    
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
